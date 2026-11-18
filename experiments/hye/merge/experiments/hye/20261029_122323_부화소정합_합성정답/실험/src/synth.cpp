#include "bench.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace bench {
const cv::Rect kCrop(1340,800,1130,530);
const cv::Rect kQr[3]={cv::Rect(1552,969,184,198),cv::Rect(1809,971,184,198),cv::Rect(2067,973,185,198)};
const cv::Point2d kBoardCenter(1906,1070);

FixedPoints load_points(const fs::path& file){
    cv::FileStorage fsr(file.u8string(),cv::FileStorage::READ);if(!fsr.isOpened())throw std::runtime_error("Cannot read "+file.u8string());
    auto rows=fsr["points"];if(!rows.isSeq()||rows.size()!=630)throw std::runtime_error("Expected630 fixed points");
    FixedPoints out;
    for(auto it=rows.begin();it!=rows.end();++it){auto r=*it;out.p.emplace_back(double(r[0]),double(r[1]));out.region.push_back(int(r[2]));out.selected.push_back(static_cast<unsigned char>(int(r[3])));}
    return out;
}
cv::Point2d warp_point(const Condition& c,cv::Point2d p){
    double a=c.angle_deg*CV_PI/180,ca=std::cos(a),sa=std::sin(a);cv::Point2d q=p-kBoardCenter;
    return cv::Point2d(c.scale*(ca*q.x-sa*q.y),c.scale*(sa*q.x+ca*q.y))+kBoardCenter+cv::Point2d(c.tx,c.ty);
}
static cv::Mat fft_translate(const cv::Mat& src,double tx,double ty){
    // 홀수 크기라 나이퀴스트 성분 모호성이 없다. J(x)=I(x−t) ⇔ Ĵ(k)=Î(k)·exp(−2πi k·t/N)
    CV_Assert(src.cols%2==1&&src.rows%2==1);
    cv::Mat spectrum;cv::dft(src,spectrum,cv::DFT_COMPLEX_OUTPUT);
    for(int v=0;v<src.rows;++v){double fv=double(v<=src.rows/2?v:v-src.rows)/src.rows;auto row=spectrum.ptr<cv::Vec2d>(v);
        for(int u=0;u<src.cols;++u){double fu=double(u<=src.cols/2?u:u-src.cols)/src.cols,ph=-2*CV_PI*(fu*tx+fv*ty);
            double c=std::cos(ph),s=std::sin(ph),re=row[u][0],im=row[u][1];row[u]={re*c-im*s,re*s+im*c};}}
    cv::Mat out;cv::idft(spectrum,out,cv::DFT_SCALE|cv::DFT_REAL_OUTPUT);return out;
}
cv::Mat synthesize(const cv::Mat& center,const Condition& c){
    CV_Assert(center.type()==CV_8UC1&&center.size()==cv::Size(3840,2160));
    // 주기 경계·리샘플 가장자리 영향을 피하려고 64px 여유를 두고 만든 뒤 kCrop만 잘라 넣는다.
    cv::Rect padded(kCrop.x-64,kCrop.y-64,kCrop.width+128+1,kCrop.height+128+1);
    cv::Mat src;center(padded).convertTo(src,CV_64F);cv::Mat warped;
    if(c.angle_deg==0&&c.scale==1)warped=fft_translate(src,c.tx,c.ty);
    else{
        double a=c.angle_deg*CV_PI/180,ca=std::cos(a)*c.scale,sa=std::sin(a)*c.scale;cv::Point2d o=kBoardCenter-cv::Point2d(padded.tl());
        cv::Mat m=(cv::Mat_<double>(2,3)<<ca,-sa,o.x-ca*o.x+sa*o.y+c.tx,sa,ca,o.y-sa*o.x-ca*o.y+c.ty);
        cv::warpAffine(src,warped,m,src.size(),cv::INTER_LANCZOS4,cv::BORDER_REPLICATE);
    }
    warped=warped*c.gain+c.offset;
    if(c.blur>0)cv::GaussianBlur(warped,warped,cv::Size(),c.blur);
    if(c.noise>0){cv::Mat n(warped.size(),CV_64F);cv::RNG rng(0x5eed0000u+c.seed);rng.fill(n,cv::RNG::NORMAL,0,c.noise);warped+=n;}
    cv::Mat out=center.clone(),local;warped.convertTo(local,CV_8U);
    local(cv::Rect(64,64,kCrop.width,kCrop.height)).copyTo(out(kCrop));
    if(c.occlusions>0){cv::RNG rng(0x0cc10000u+c.seed);
        for(int k=0;k<3;++k)for(int i=0;i<c.occlusions;++i){int x=rng.uniform(kQr[k].x+10,kQr[k].x+kQr[k].width-31),y=rng.uniform(kQr[k].y+10,kQr[k].y+kQr[k].height-31);out(cv::Rect(x,y,21,21)).setTo(128);}}
    return out;
}
double estimate_noise(const cv::Mat& a,const cv::Mat& b){
    cv::Mat fa,fb;a(kCrop).convertTo(fa,CV_64F);b(kCrop).convertTo(fb,CV_64F);
    auto flat=[](const cv::Mat& f){cv::Mat s,gx,gy,g;cv::GaussianBlur(f,s,cv::Size(),1.5);cv::Sobel(s,gx,CV_64F,1,0,3,.125);cv::Sobel(s,gy,CV_64F,0,1,3,.125);cv::magnitude(gx,gy,g);return g;};
    cv::Mat ga=flat(fa),gb=flat(fb);std::vector<double> diff;
    for(int y=0;y<fa.rows;++y)for(int x=0;x<fa.cols;++x){double va=fa.at<double>(y,x),vb=fb.at<double>(y,x);
        if(ga.at<double>(y,x)<1&&gb.at<double>(y,x)<1&&va>5&&va<250&&vb>5&&vb<250)diff.push_back(va-vb);}
    if(diff.size()<10000)throw std::runtime_error("Too few flat pixels for noise estimate");
    auto median=[](std::vector<double> v){std::nth_element(v.begin(),v.begin()+v.size()/2,v.end());return v[v.size()/2];};
    double m=median(diff);for(auto& d:diff)d=std::abs(d-m);return 1.4826*median(diff)/std::sqrt(2.);
}
}
