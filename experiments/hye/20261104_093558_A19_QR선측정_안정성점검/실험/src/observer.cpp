#include "observer_internal.hpp"
#include "ecc_compat.hpp"
#ifdef _WIN32
#include <windows.h>
#include <bcrypt.h>
#else
#include <CommonCrypto/CommonDigest.h>
#endif
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <limits>
namespace alignment_detail {
static const cv::Rect qr_boxes[]={cv::Rect(1552,969,184,198),cv::Rect(1809,971,184,198),cv::Rect(2067,973,185,198)};
static cv::Mat translation(double x,double y){return (cv::Mat_<double>(3,3)<<1,0,x,0,1,y,0,0,1);}
static cv::Point2d project(const cv::Mat& h,const cv::Point2d& p){double z=h.at<double>(2,0)*p.x+h.at<double>(2,1)*p.y+h.at<double>(2,2);return {(h.at<double>(0,0)*p.x+h.at<double>(0,1)*p.y+h.at<double>(0,2))/z,(h.at<double>(1,0)*p.x+h.at<double>(1,1)*p.y+h.at<double>(1,2))/z};}
static std::vector<unsigned char> bytes(const fs::path& p){std::ifstream f(p,std::ios::binary);if(!f)throw std::runtime_error("Cannot read "+p.u8string());return {std::istreambuf_iterator<char>(f),std::istreambuf_iterator<char>()};}
cv::Mat read_gray(const fs::path& p){auto m=cv::imdecode(bytes(p),cv::IMREAD_GRAYSCALE);if(m.empty())throw std::runtime_error("Cannot decode "+p.u8string());return m;}
std::string sha256(const fs::path& p){
#ifdef _WIN32

    auto b=bytes(p);BCRYPT_ALG_HANDLE a=nullptr;BCRYPT_HASH_HANDLE h=nullptr;DWORD length=0,written=0;
    if(BCryptOpenAlgorithmProvider(&a,BCRYPT_SHA256_ALGORITHM,nullptr,0)<0)throw std::runtime_error("SHA256 provider failure");
    BCryptGetProperty(a,BCRYPT_OBJECT_LENGTH,reinterpret_cast<PUCHAR>(&length),sizeof(length),&written,0);std::vector<unsigned char> object(length),sum(32);
    NTSTATUS status=BCryptCreateHash(a,&h,object.data(),length,nullptr,0,0);
    if(status>=0)status=BCryptHashData(h,b.data(),ULONG(b.size()),0);
    if(status>=0)status=BCryptFinishHash(h,sum.data(),32,0);
    if(h)BCryptDestroyHash(h);BCryptCloseAlgorithmProvider(a,0);
    if(status<0)throw std::runtime_error("SHA256 computation failure");std::ostringstream s;s<<std::hex<<std::setfill('0');for(auto v:sum)s<<std::setw(2)<<int(v);return s.str();

#else
    auto b=bytes(p);unsigned char sum[CC_SHA256_DIGEST_LENGTH];CC_SHA256(b.data(),CC_LONG(b.size()),sum);
    std::ostringstream s;s<<std::hex<<std::setfill('0');for(auto v:sum)s<<std::setw(2)<<int(v);return s.str();
#endif
}
cv::Mat node_matrix(const cv::FileNode& n){
    if(!n.isSeq()||n.size()==0)throw std::runtime_error("Numeric array required");bool nested=n[0].isSeq();int rows=nested?int(n.size()):1,cols=nested?int(n[0].size()):int(n.size());cv::Mat m(rows,cols,CV_64F);
    if(cols==0)throw std::runtime_error("Nonempty matrix rows required");
    for(int i=0;i<rows;++i){
        if(nested&&(!n[i].isSeq()||int(n[i].size())!=cols))throw std::runtime_error("Rectangular numeric matrix required");
        for(int j=0;j<cols;++j){auto v=nested?n[i][j]:n[j];if(!v.isInt()&&!v.isReal())throw std::runtime_error("Numeric matrix element required");m.at<double>(i,j)=double(v);}
    }return m;
}
cv::Mat feature_map(const cv::Mat& x){
    if(x.rows!=1||x.cols!=102||x.type()!=CV_64F||!cv::checkRange(x))throw std::runtime_error("Expected finite1x102 observation");
    cv::Mat z(1,204,CV_64F);x.copyTo(z.colRange(0,102));double h=0;for(int i=0;i<54;i+=2)h+=x.at<double>(i);h/=2700.;
    for(int i=0;i<102;++i)z.at<double>(i+102)=x.at<double>(i)*h;return z;
}
Observer::Observer(const fs::path& assets){
    auto center=assets/"center.bmp";if(sha256(center)!="01a69df6771b840d2fcf9cf727760592b5fb199c12277bfc173af514e26fd685")throw std::runtime_error("Fixed center hash changed");
    center_=read_gray(center);if(center_.size()!=cv::Size(3840,2160))throw std::runtime_error("Reference dimensions changed");
    cv::FileStorage geometry((assets/"geometry.json").u8string(),cv::FileStorage::READ);
    if(!geometry.isOpened())throw std::runtime_error("Missing geometry.json");
    for(auto it=geometry["corners"].begin();it!=geometry["corners"].end();++it){auto n=*it;int k=int(n["qr"])-1;if(k<0||k>2)throw std::runtime_error("Invalid reference QR region");reference_points_[k].push_back({double(n["cx"]),double(n["cy"])});}
    size_t count=0;for(int k=0;k<3;++k){count+=reference_points_[k].size();templates_[k]=center_(qr_boxes[k]).clone();masks_[k]=cv::Mat::zeros(qr_boxes[k].size(),CV_8U);masks_[k](cv::Rect(6,6,qr_boxes[k].width-12,qr_boxes[k].height-12)).setTo(255);}if(count!=630)throw std::runtime_error("Expected630 reference check coordinates");
    cv::FileStorage protocol((assets/"line_protocol.json").u8string(),cv::FileStorage::READ);if(!protocol.isOpened())throw std::runtime_error("Missing line protocol");
    for(auto it=protocol["blocks"].begin();it!=protocol["blocks"].end();++it){auto b=(*it)["read_box"];cv::Rect r(int(b[0]),int(b[1]),int(b[2])-int(b[0]),int(b[3])-int(b[1]));if(r.height!=49||r.width!=64||(r&cv::Rect(0,0,3840,2160))!=r)throw std::runtime_error("Invalid permitted line block");line_boxes_.push_back(r);reference_profiles_.push_back(line_profile(center_,r));}if(line_boxes_.size()!=48)throw std::runtime_error("Expected48 line blocks");
}
Observation Observer::measure(const cv::Mat& input) const {
    if(input.empty()||input.depth()!=CV_8U||input.size()!=center_.size())throw std::invalid_argument("Input must be3840x2160 uint8 image");
    cv::Mat image;if(input.channels()==1)image=input;else if(input.channels()==3)cv::cvtColor(input,image,cv::COLOR_BGR2GRAY);else if(input.channels()==4)cv::cvtColor(input,image,cv::COLOR_BGRA2GRAY);else throw std::invalid_argument("Expected1/3/4 channels");
    cv::Point shifts[3];double scores[3];
    for(int pad:{130,260,512}){for(int k=0;k<3;++k){auto b=qr_boxes[k];int x=std::max(0,b.x-pad),y=std::max(0,b.y-pad);cv::Rect window(x,y,std::min(image.cols,b.x+b.width+pad)-x,std::min(image.rows,b.y+b.height+pad)-y);cv::Mat c;cv::matchTemplate(image(window),templates_[k],c,cv::TM_CCOEFF_NORMED);cv::Point p;cv::minMaxLoc(c,nullptr,&scores[k],nullptr,&p);shifts[k]=p+window.tl()-b.tl();}if(*std::min_element(scores,scores+3)>=.6)break;}
    Observation out{cv::Mat(1,102,CV_64F),1.,1.};int feature=0;
    for(int k=0;k<3;++k){
        if(scores[k]<.6)throw std::runtime_error("QR NCC below.6");auto b=qr_boxes[k];cv::Rect read(b.x+shifts[k].x-12,b.y+shifts[k].y-12,b.width+24,b.height+24);
        if((read&cv::Rect(0,0,image.cols,image.rows))!=read)throw std::runtime_error("QR crop outside image");cv::Mat local;translation(12,12).convertTo(local,CV_32F);
        double cc=semes::ecc5::findTransformECCWithMask(templates_[k],image(read),masks_[k],cv::Mat(read.size(),CV_8U,cv::Scalar(255)),local,cv::MOTION_HOMOGRAPHY,cv::TermCriteria(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,200,1e-8),5);
        cv::Mat d;local.convertTo(d,CV_64F);cv::Mat h=translation(read.x,read.y)*d*translation(-b.x,-b.y);h/=h.at<double>(2,2);
        if(!cv::checkRange(h)||cc<.8)throw std::runtime_error("Invalid QR ECC transform");out.qr_correlation=std::min(out.qr_correlation,cc);
        for(const auto& p:reference_points_[k]){auto v=project(h,p)-p-cv::Point2d(shifts[k]);if(!std::isfinite(v.x)||!std::isfinite(v.y)||std::max(std::abs(v.x),std::abs(v.y))>10)throw std::runtime_error("QR refinement exceeds10px");}
        for(double y:{.2,.5,.8})for(double x:{.2,.5,.8}){cv::Point2d p(b.x+x*b.width,b.y+y*b.height);auto q=project(h,p)-p;out.values.at<double>(feature++)=q.x;out.values.at<double>(feature++)=q.y;}
    }
    for(size_t k=0;k<line_boxes_.size();++k){auto fit=match_line(reference_profiles_[k],line_profile(image,line_boxes_[k]));if(!fit.valid)throw std::runtime_error("Line normal-profile match failed at block"+std::to_string(k));out.line_correlation=std::min(out.line_correlation,fit.correlation);out.values.at<double>(feature++)=fit.shift;}
    if(feature!=102||!cv::checkRange(out.values))throw std::runtime_error("Invalid observation vector");return out;
}
}

namespace alignment_detail {
DetailedObservation Observer::measure_detailed(const cv::Mat& image) const {
    DetailedObservation out; out.values=cv::Mat(1,102,CV_64F,cv::Scalar(std::numeric_limits<double>::quiet_NaN()));
    cv::Point shifts[3];double scores[3];
    for(int pad:{130,260,512}){out.pad=pad;for(int k=0;k<3;++k){auto b=qr_boxes[k];int x=std::max(0,b.x-pad),y=std::max(0,b.y-pad);cv::Rect window(x,y,std::min(image.cols,b.x+b.width+pad)-x,std::min(image.rows,b.y+b.height+pad)-y);cv::Mat c;cv::matchTemplate(image(window),templates_[k],c,cv::TM_CCOEFF_NORMED);cv::Point p;cv::minMaxLoc(c,nullptr,&scores[k],nullptr,&p);shifts[k]=p+window.tl()-b.tl();}if(*std::min_element(scores,scores+3)>=.6)break;}
    int feature=0;out.qr_ok=true;
    for(int k=0;k<3;++k){
        out.ncc[k]=scores[k];out.int_shift[k]=shifts[k];
        auto b=qr_boxes[k];cv::Rect read(b.x+shifts[k].x-12,b.y+shifts[k].y-12,b.width+24,b.height+24);
        if(scores[k]<.6||(read&cv::Rect(0,0,image.cols,image.rows))!=read){out.qr_ok=false;feature+=18;continue;}
        cv::Mat local;translation(12,12).convertTo(local,CV_32F);double cc=-1;
        try{cc=semes::ecc5::findTransformECCWithMask(templates_[k],image(read),masks_[k],cv::Mat(read.size(),CV_8U,cv::Scalar(255)),local,cv::MOTION_HOMOGRAPHY,cv::TermCriteria(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,200,1e-8),5);}catch(...){cc=-1;}
        out.ecc[k]=cc;
        cv::Mat d;local.convertTo(d,CV_64F);cv::Mat h=translation(read.x,read.y)*d*translation(-b.x,-b.y);h/=h.at<double>(2,2);
        if(!cv::checkRange(h)||cc<.8)out.qr_ok=false;
        double worst=0;for(const auto& p:reference_points_[k]){auto v=project(h,p)-p-cv::Point2d(shifts[k]);worst=std::max(worst,std::max(std::abs(v.x),std::abs(v.y)));}
        out.refine_px[k]=worst;if(!(worst<=10))out.qr_ok=false;
        for(double y:{.2,.5,.8})for(double x:{.2,.5,.8}){cv::Point2d p(b.x+x*b.width,b.y+y*b.height);auto q=project(h,p)-p;out.values.at<double>(feature++)=q.x;out.values.at<double>(feature++)=q.y;}
    }
    out.lines_ok=0;
    for(size_t k=0;k<line_boxes_.size();++k){auto fit=match_line(reference_profiles_[k],line_profile(image,line_boxes_[k]));out.line_corr[k]=fit.correlation;out.line_valid[k]=fit.valid;out.lines_ok+=fit.valid;out.values.at<double>(feature++)=fit.shift;}
    return out;
}
}
