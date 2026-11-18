#include "detectors.hpp"
#include <algorithm>
#include <cmath>
namespace bench {
namespace {
// 이전 검출기 비교 실험(20261029_112722)과 같은 설정
const auto kSubPix=cv::TermCriteria(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,40,.001);
void to_keypoints(const std::vector<cv::Point2f>& p,std::vector<cv::KeyPoint>& k){k.clear();for(auto q:p)k.emplace_back(q,7.f);}
}
std::vector<cv::Point2f> forstner(const cv::Mat& image,const cv::Mat& mask,int max_points,double min_distance){
    cv::Mat f;image.convertTo(f,CV_32F);cv::GaussianBlur(f,f,cv::Size(),.7);
    cv::Mat gx,gy;cv::Sobel(f,gx,CV_32F,1,0,3,1./8);cv::Sobel(f,gy,CV_32F,0,1,3,1./8);
    cv::Mat xx=gx.mul(gx),yy=gy.mul(gy),xy=gx.mul(gy);for(auto* m:{&xx,&yy,&xy})cv::GaussianBlur(*m,*m,cv::Size(),1.5);
    cv::Mat det=xx.mul(yy)-xy.mul(xy),trace=xx+yy,w,q;cv::divide(det,trace+1e-6f,w);cv::divide(4*det,trace.mul(trace)+1e-6f,q);
    double wmax;cv::minMaxLoc(w,nullptr,&wmax,nullptr,nullptr,mask);cv::Mat dilated;cv::dilate(w,dilated,cv::Mat(),cv::Point(-1,-1),2);
    struct C{float w;cv::Point p;};std::vector<C> cand;
    for(int y=4;y<w.rows-4;++y)for(int x=4;x<w.cols-4;++x){float v=w.at<float>(y,x);
        if(mask.at<unsigned char>(y,x)&&v==dilated.at<float>(y,x)&&q.at<float>(y,x)>.5f&&v>.01*wmax)cand.push_back({v,{x,y}});}
    std::sort(cand.begin(),cand.end(),[](const C& a,const C& b){return a.w>b.w;});
    std::vector<cv::Point2f> out;const int r=3;
    for(auto& c:cand){if(int(out.size())>=max_points)break;
        bool near=false;for(auto& o:out)if(std::hypot(o.x-c.p.x,o.y-c.p.y)<min_distance){near=true;break;}if(near)continue;
        double a=0,b=0,d=0,bx=0,by=0;
        for(int y=c.p.y-r;y<=c.p.y+r;++y)for(int x=c.p.x-r;x<=c.p.x+r;++x){double u=gx.at<float>(y,x),v=gy.at<float>(y,x);a+=u*u;b+=u*v;d+=v*v;bx+=u*u*x+u*v*y;by+=u*v*x+v*v*y;}
        double den=a*d-b*b;if(std::abs(den)<1e-9)continue;cv::Point2f s(float((d*bx-b*by)/den),float((a*by-b*bx)/den));
        if(std::hypot(s.x-c.p.x,s.y-c.p.y)>1.5)continue;out.push_back(s);}
    return out;
}
std::vector<Detector> detectors(){
    std::vector<Detector> d;
    auto gftt=[](bool harris,bool subpix){return [=](const cv::Mat& im,const cv::Mat& m,std::vector<cv::KeyPoint>& k,cv::Mat*){std::vector<cv::Point2f> p;
        cv::goodFeaturesToTrack(im,p,800,.01,4,m,5,harris,.04);if(subpix&&!p.empty())cv::cornerSubPix(im,p,cv::Size(5,5),cv::Size(-1,-1),kSubPix);to_keypoints(p,k);};};
    d.push_back({"Harris",false,true,0,gftt(true,false)});
    d.push_back({"Harris+cornerSubPix",false,true,0,gftt(true,true)});
    d.push_back({"Shi-Tomasi",false,true,0,gftt(false,false)});
    d.push_back({"Shi-Tomasi+cornerSubPix",false,true,0,gftt(false,true)});
    d.push_back({"Förstner",false,true,0,[](const cv::Mat& im,const cv::Mat& m,std::vector<cv::KeyPoint>& k,cv::Mat*){to_keypoints(forstner(im,m,800,4),k);}});
    d.push_back({"FAST",false,true,0,[](const cv::Mat& im,const cv::Mat& m,std::vector<cv::KeyPoint>& k,cv::Mat*){cv::FastFeatureDetector::create(25)->detect(im,k,m);cv::KeyPointsFilter::retainBest(k,800);}});
    auto feature=[](cv::Ptr<cv::Feature2D> f){return [f](const cv::Mat& im,const cv::Mat& m,std::vector<cv::KeyPoint>& k,cv::Mat* desc){if(desc)f->detectAndCompute(im,m,k,*desc);else f->detect(im,k,m);};};
    d.push_back({"ORB",true,false,cv::NORM_HAMMING,feature(cv::ORB::create(3000))});
    d.push_back({"BRISK",true,false,cv::NORM_HAMMING,feature(cv::BRISK::create(20))});
    d.push_back({"AKAZE",true,false,cv::NORM_HAMMING,feature(cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB,0,3,.0001f))});
    d.push_back({"KAZE",true,false,cv::NORM_L2,feature(cv::KAZE::create(false,false,.0001f))});
    return d;
}
}
