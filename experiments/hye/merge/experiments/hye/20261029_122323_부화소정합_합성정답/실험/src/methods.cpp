#include "bench.hpp"
#include "control_internal.hpp"
#include <cmath>
#include <limits>
namespace bench {
namespace {
const double kNan=std::numeric_limits<double>::quiet_NaN();
enum {NCC,PARABOLA,PHASE,CORNER,LK,PYR_LK,GN,ECC,PHASE64,COUNT_};
const char* kNames[]={"NCC(정수)","NCC+포물선","NCC+위상상관","NCC+cornerSubPix","NCC+LK(1단)","피라미드LK(3단·NCC없음)","NCC+LK15+GN스플라인(채택측정기)","NCC+ECC","QR단위NCC+위상상관64(점NCC없음)"};
const int kSearch=3;   // QR 단위 NCC 뒤 점 단위 NCC 탐색 반경 (px)
const int kHalf=8;     // 점 단위 NCC 템플릿 17×17
bool finite(cv::Point2d p){return std::isfinite(p.x)&&std::isfinite(p.y);}
}
struct Methods::Impl {
    cv::Mat center,ref_crop,ref_blur8,ref_float;
    FixedPoints points;std::vector<cv::Point> rounded;std::vector<cv::Point2f> crop_points;std::vector<cv::Point2d> ref_corner;
    legacy_control::Tracker tracker;
    Impl(const cv::Mat& c,const FixedPoints& p,const fs::path& assets):center(c),points(p),tracker(assets){
        ref_crop=center(kCrop);cv::GaussianBlur(ref_crop,ref_blur8,cv::Size(),.8);ref_crop.convertTo(ref_float,CV_32F);cv::GaussianBlur(ref_float,ref_float,cv::Size(),.8);
        std::vector<cv::Point2f> corners;
        for(auto q:points.p){rounded.emplace_back(int(std::lround(q.x)),int(std::lround(q.y)));crop_points.emplace_back(float(q.x-kCrop.x),float(q.y-kCrop.y));corners.emplace_back(float(q.x),float(q.y));}
        cv::cornerSubPix(center,corners,cv::Size(5,5),cv::Size(-1,-1),cv::TermCriteria(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,40,.001));
        for(auto q:corners)ref_corner.emplace_back(q.x,q.y);
    }
    // 채택 측정기와 같은 QR 단위 NCC (가로 ±110, 세로 ±22)
    std::array<cv::Point,3> panel_shift(const cv::Mat& gray)const{
        std::array<cv::Point,3> out;
        for(int k=0;k<3;++k){cv::Rect search(kQr[k].x-110,kQr[k].y-22,kQr[k].width+220,kQr[k].height+44);cv::Mat r;cv::matchTemplate(gray(search),center(kQr[k]),r,cv::TM_CCOEFF_NORMED);
            cv::Point loc;cv::minMaxLoc(r,nullptr,nullptr,nullptr,&loc);out[k]=loc-cv::Point(110,22);}
        return out;
    }
    // 점마다 17×17 템플릿의 정수 NCC. map은 (2r+1)² 점수, peak는 그 안의 위치.
    void point_ncc(const cv::Mat& gray,const std::array<cv::Point,3>& shift,std::vector<cv::Point>& integer,std::vector<cv::Mat>& maps)const{
        size_t n=rounded.size();integer.resize(n);maps.resize(n);
        for(size_t i=0;i<n;++i){auto c=rounded[i];auto s=shift[points.region[i]];
            cv::Rect t(c.x-kHalf,c.y-kHalf,2*kHalf+1,2*kHalf+1),w(c.x+s.x-kHalf-kSearch,c.y+s.y-kHalf-kSearch,2*(kHalf+kSearch)+1,2*(kHalf+kSearch)+1);
            cv::matchTemplate(gray(w),center(t),maps[i],cv::TM_CCOEFF_NORMED);cv::Point loc;cv::minMaxLoc(maps[i],nullptr,nullptr,nullptr,&loc);
            integer[i]=s+loc-cv::Point(kSearch,kSearch);}
    }
    Estimate run(int m,const cv::Mat& gray)const{
        size_t n=rounded.size();Estimate e{std::vector<cv::Point2d>(n,{kNan,kNan}),std::vector<unsigned char>(n,0),std::vector<unsigned char>(n,0),std::vector<double>(n,kNan)};
        auto finish=[&](){for(size_t i=0;i<n;++i){if(!finite(e.d[i]))e.ok[i]=0;if(!e.ok[i])e.own_valid[i]=0;}return e;};
        if(m==GN){auto t=tracker.measure(gray);
            for(size_t i=0;i<n;++i){e.d[i]={t.displacement.at<float>(int(i),0),t.displacement.at<float>(int(i),1)};e.ok[i]=1;e.own_valid[i]=t.valid[i];e.fb[i]=t.quality.at<float>(int(i),4);}
            return finish();}
        if(m==PYR_LK){cv::Mat cur;cv::GaussianBlur(gray(kCrop),cur,cv::Size(),.8);std::vector<cv::Point2f> next=crop_points;std::vector<unsigned char> st;std::vector<float> err;
            cv::calcOpticalFlowPyrLK(ref_blur8,cur,crop_points,next,st,err,cv::Size(21,21),3,cv::TermCriteria(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,60,1e-4),0,1e-5);
            for(size_t i=0;i<n;++i){e.d[i]={double(next[i].x-crop_points[i].x),double(next[i].y-crop_points[i].y)};e.ok[i]=e.own_valid[i]=st[i];}
            return finish();}
        auto shift=panel_shift(gray);
        if(m==PHASE64){cv::Mat hann;cv::createHanningWindow(hann,cv::Size(64,64),CV_64F);
            // 점 단위 정수 NCC 없이 QR 단위 이동만 초기값으로 두고 위상 상관이 정수·부화소를 함께 찾게 한다.
            for(size_t i=0;i<n;++i){auto c=rounded[i];auto s=shift[points.region[i]];cv::Mat a,b;center(cv::Rect(c.x-32,c.y-32,64,64)).convertTo(a,CV_64F);gray(cv::Rect(c.x+s.x-32,c.y+s.y-32,64,64)).convertTo(b,CV_64F);
                cv::Point2d d=cv::phaseCorrelate(a,b,hann);e.d[i]=cv::Point2d(s)+d;e.ok[i]=e.own_valid[i]=std::abs(d.x)<=8&&std::abs(d.y)<=8;}
            return finish();}
        std::vector<cv::Point> integer;std::vector<cv::Mat> maps;point_ncc(gray,shift,integer,maps);
        switch(m){
        case NCC:for(size_t i=0;i<n;++i){e.d[i]=cv::Point2d(integer[i]);e.ok[i]=e.own_valid[i]=1;}break;
        case PARABOLA:for(size_t i=0;i<n;++i){cv::Point loc=integer[i]-shift[points.region[i]]+cv::Point(kSearch,kSearch);const cv::Mat& r=maps[i];
            if(loc.x<1||loc.y<1||loc.x>r.cols-2||loc.y>r.rows-2)continue;
            auto v=[&](int dx,int dy){return double(r.at<float>(loc.y+dy,loc.x+dx));};
            double ax=v(-1,0)-2*v(0,0)+v(1,0),ay=v(0,-1)-2*v(0,0)+v(0,1);if(ax>=0||ay>=0)continue;
            e.d[i]=cv::Point2d(integer[i])+cv::Point2d((v(-1,0)-v(1,0))/(2*ax),(v(0,-1)-v(0,1))/(2*ay));e.ok[i]=e.own_valid[i]=1;}break;
        case PHASE:{cv::Mat hann;cv::createHanningWindow(hann,cv::Size(32,32),CV_64F);
            for(size_t i=0;i<n;++i){auto c=rounded[i];cv::Mat a,b;center(cv::Rect(c.x-16,c.y-16,32,32)).convertTo(a,CV_64F);gray(cv::Rect(c.x+integer[i].x-16,c.y+integer[i].y-16,32,32)).convertTo(b,CV_64F);
                cv::Point2d s=cv::phaseCorrelate(a,b,hann);e.d[i]=cv::Point2d(integer[i])+s;e.ok[i]=e.own_valid[i]=std::abs(s.x)<=2&&std::abs(s.y)<=2;}}break;
        case CORNER:{std::vector<cv::Point2f> q;for(size_t i=0;i<n;++i)q.emplace_back(float(points.p[i].x+integer[i].x),float(points.p[i].y+integer[i].y));auto start=q;
            cv::cornerSubPix(gray,q,cv::Size(5,5),cv::Size(-1,-1),cv::TermCriteria(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,40,.001));
            for(size_t i=0;i<n;++i){e.d[i]=cv::Point2d(q[i].x,q[i].y)-ref_corner[i];e.ok[i]=e.own_valid[i]=cv::norm(q[i]-start[i])<3;}}break;
        case LK:{cv::Mat cur;cv::GaussianBlur(gray(kCrop),cur,cv::Size(),.8);std::vector<cv::Point2f> next(n),back=crop_points;std::vector<unsigned char> f,r;std::vector<float> e1,e2;
            for(size_t i=0;i<n;++i)next[i]=crop_points[i]+cv::Point2f(float(integer[i].x),float(integer[i].y));
            auto crit=cv::TermCriteria(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,60,1e-4);
            cv::calcOpticalFlowPyrLK(ref_blur8,cur,crop_points,next,f,e1,cv::Size(15,15),0,crit,cv::OPTFLOW_USE_INITIAL_FLOW,1e-5);
            cv::calcOpticalFlowPyrLK(cur,ref_blur8,next,back,r,e2,cv::Size(15,15),0,crit,cv::OPTFLOW_USE_INITIAL_FLOW,1e-5);
            for(size_t i=0;i<n;++i){e.d[i]={double(next[i].x-crop_points[i].x),double(next[i].y-crop_points[i].y)};e.ok[i]=f[i];
                e.fb[i]=r[i]?cv::norm(back[i]-crop_points[i]):kNan;e.own_valid[i]=f[i]&&r[i]&&e.fb[i]<.5;}}break;
        case ECC:{cv::Mat cur;gray(kCrop).convertTo(cur,CV_32F);cv::GaussianBlur(cur,cur,cv::Size(),.8);const int h=15,margin=8;
            auto crit=cv::TermCriteria(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,50,1e-5);
            for(size_t i=0;i<n;++i){cv::Point c=rounded[i]-kCrop.tl();cv::Mat templ=ref_float(cv::Rect(c.x-h,c.y-h,2*h+1,2*h+1));
                cv::Mat input=cur(cv::Rect(c.x+integer[i].x-h-margin,c.y+integer[i].y-h-margin,2*(h+margin)+1,2*(h+margin)+1));
                cv::Mat warp=(cv::Mat_<float>(2,3)<<1,0,margin,0,1,margin);
                try{// 작은 창마다 따로 흐리면 창 경계 처리가 달라 점별 편향이 생기므로, 영역 전체를 한 번 흐리고 ECC 내부 흐림(gaussFiltSize)은 끈다.
                    cv::findTransformECC(templ,input,warp,cv::MOTION_TRANSLATION,crit,cv::Mat(),1);
                    e.d[i]=cv::Point2d(integer[i])+cv::Point2d(warp.at<float>(0,2)-margin,warp.at<float>(1,2)-margin);e.ok[i]=e.own_valid[i]=1;}
                catch(const cv::Exception&){}}}break;
        default:throw std::invalid_argument("Unknown method");
        }
        return finish();
    }
};
Methods::Methods(const cv::Mat& center,const FixedPoints& points,const fs::path& assets):impl_(new Impl(center,points,assets)){}
Methods::~Methods(){delete impl_;}
int Methods::count(){return COUNT_;}
const char* Methods::name(int m){return kNames[m];}
Estimate Methods::run(int m,const cv::Mat& image)const{return impl_->run(m,image);}
}
