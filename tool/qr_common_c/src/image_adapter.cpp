#include "qr_common.h"
#include "reference_data.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <cmath>
#include <cstring>
#include <fstream>
#include <vector>
#include <locale>
#include <codecvt>

/* UTF-8 file names are decoded explicitly on Windows; never inspect label names. */
static cv::Mat load_image(const char *path){
#ifdef _WIN32
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    std::ifstream stream(converter.from_bytes(path),std::ios::binary);
#else
    std::ifstream stream(path,std::ios::binary);
#endif
    if(!stream)return {};
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(stream)),std::istreambuf_iterator<char>());
    return cv::imdecode(bytes,cv::IMREAD_GRAYSCALE);
}
extern "C" int qc_predict_image(const char *path,const char *calibration,int fold,qc_result *out){
    if(!path||!calibration||!out)return QC_BAD_ARGUMENT;
    std::memset(out,0,sizeof(*out));
    try{
        cv::setNumThreads(1);cv::Mat image=load_image(path);
        if(image.empty()||image.cols!=3840||image.rows!=2160||image.type()!=CV_8UC1)return QC_IMAGE_ERROR;
        qc_gray gray={image.data,image.cols,image.rows,(int)image.step};qc_point fitted[QC_POINTS];
        double warps[3][6],basis[4];int counts[3]={0,0,0};
        for(int q=0;q<3;q++){
            int x=qc_patch_bounds[q][0],y=qc_patch_bounds[q][1],w=qc_patch_bounds[q][2],h=qc_patch_bounds[q][3];
            cv::Mat templ=load_image((std::string(calibration)+"/template_"+char('0'+q)+".png").c_str());
            if(templ.empty()||templ.cols!=w||templ.rows!=h)return QC_IMAGE_ERROR;
            int left=std::max(0,x-300),top=std::max(0,y-120);
            cv::Mat search=image(cv::Rect(left,top,std::min(image.cols,x+w+300)-left,std::min(image.rows,y+h+120)-top));
            cv::Mat scores;cv::matchTemplate(search,templ,scores,cv::TM_CCOEFF_NORMED);double score;cv::Point loc;
            cv::minMaxLoc(scores,nullptr,&score,nullptr,&loc);if(score<.5)return QC_QR_ERROR;
            int cx=left+loc.x,cy=top+loc.y;cv::Mat patch=image(cv::Rect(cx,cy,w,h));
            cv::Mat warp=cv::Mat::eye(2,3,CV_32F);
            double corr=cv::findTransformECC(templ,patch,warp,cv::MOTION_AFFINE,cv::TermCriteria(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,100,1e-7),cv::noArray(),1);
            if(corr<.5)return QC_QR_ERROR;
            for(int i=0;i<6;i++)warps[q][i]=warp.at<float>(i/3,i%3);
            warps[q][2]+=cx-warps[q][0]*x-warps[q][1]*y;warps[q][5]+=cy-warps[q][3]*x-warps[q][4]*y;
        }
        for(int i=0;i<QC_POINTS;i++){
            const qc_corner_spec &s=qc_corner_specs[i];int q=s.qr;double *a=warps[q];
            double sx=a[0]*s.seed[0]+a[1]*s.seed[1]+a[2],sy=a[3]*s.seed[0]+a[4]*s.seed[1]+a[5],rmse;
            basis[0]=a[0]*s.basis[0]+a[1]*s.basis[2];basis[1]=a[0]*s.basis[1]+a[1]*s.basis[3];
            basis[2]=a[3]*s.basis[0]+a[4]*s.basis[2];basis[3]=a[3]*s.basis[1]+a[4]*s.basis[3];
            for(int j=0;j<2;j++){double norm=std::hypot(basis[j],basis[j+2]);basis[j]/=norm;basis[j+2]/=norm;}
            for(int pass=0;pass<3;pass++){
                double nx,ny;int status=qc_fit_corner(&gray,sx,sy,s.bits,basis,&nx,&ny,&rmse);
                if(status!=QC_OK)return status;sx=nx;sy=ny;
            }
            fitted[i]={sx,sy,q};counts[q]++;out->features[q*2]+=sx-s.reference[0];out->features[q*2+1]+=sy-s.reference[1];
        }
        for(int q=0;q<3;q++){out->features[q*2]/=counts[q];out->features[q*2+1]/=counts[q];}
        out->qr_points=QC_POINTS;qc_edge_sample samples[120];
        int status=qc_extract_edges(&gray,fitted,QC_POINTS,out->features+6,samples,out);
        if(status!=QC_OK)return status;
        return qc_predict_features(out->features,fold,&out->pose);
    }catch(const cv::Exception &){return QC_QR_ERROR;}catch(const std::exception &){return QC_IMAGE_ERROR;}
}
