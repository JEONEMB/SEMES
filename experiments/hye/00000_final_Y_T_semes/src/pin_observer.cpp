#include "pin_observer.hpp"
#include "site.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include <array>
#include <exception>
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace final04_pin {
namespace {
const int radius=45, search=260, side=91;
void image_check(const cv::Mat& image) {
    if(image.type()!=CV_8UC1 || image.size()!=cv::Size(3840,2160))
        throw std::invalid_argument("Pin observer requires 3840x2160 grayscale uint8");
}
double cubic(double x){x=std::abs(x);const double a=-.75;if(x<=1)return (a+2)*x*x*x-(a+3)*x*x+1;if(x<2)return a*x*x*x-5*a*x*x+8*a*x-4*a;return 0;}
cv::Mat sample(const cv::Mat& image,double x0,double y0){
    int ix=int(std::floor(x0)),iy=int(std::floor(y0));
    if(ix<1||iy<1||ix+side+2>=image.cols||iy+side+2>=image.rows)throw std::runtime_error("Pin sample outside image");
    double wx[4],wy[4];for(int k=0;k<4;++k){wx[k]=cubic(x0-ix-(k-1));wy[k]=cubic(y0-iy-(k-1));}
    cv::Mat out(side,side,CV_64F);
    for(int y=0;y<side;++y)for(int x=0;x<side;++x){double v=0;for(int j=0;j<4;++j){const auto* p=image.ptr<unsigned char>(iy+y+j-1)+ix+x-1;double h=0;for(int i=0;i<4;++i)h+=wx[i]*p[i];v+=wy[j]*h;}out.at<double>(y,x)=v;}
    return out;
}
}
Observer::Observer(const cv::Mat& center) {
    image_check(center);
    for(int k=0;k<4;++k)
        center(cv::Rect(site::get().screws[k].x-radius,site::get().screws[k].y-radius,side,side)).convertTo(templates_[k],CV_64F);
}
bool Observer::find_peak(const cv::Mat& image,int k,cv::Point center_shift,int r,cv::Point& shift,double& ncc) const {
    const auto c=site::get().screws[k]+center_shift;
    cv::Rect window(c.x-radius-r,c.y-radius-r,side+2*r,side+2*r);
    if((window&cv::Rect(0,0,image.cols,image.rows))!=window)return false;
    cv::Mat t8,scores;templates_[k].convertTo(t8,CV_8U);
    cv::matchTemplate(image(window),t8,scores,cv::TM_CCOEFF_NORMED);
    cv::Point peak;cv::minMaxLoc(scores,nullptr,&ncc,nullptr,&peak);
    shift=peak-cv::Point(r,r)+center_shift;
    return peak.x>0&&peak.y>0&&peak.x<scores.cols-1&&peak.y<scores.rows-1;   // 창 끝 봉우리는 믿지 않는다
}
Measurement Observer::measure(const cv::Mat& image) const { return measure(image,cv::Point(0,0)); }
Measurement Observer::measure(const cv::Mat& image,cv::Point hint) const {
    image_check(image);
    Measurement m{cv::Mat(1,8,CV_64F),cv::Mat(1,8,CV_64F)};
    // 나사 4개는 서로 독립이라 동시에 계산한다(결과는 한 개씩 할 때와 같다). 예외는 모았다가 나사 순서대로 다시 던진다.
    std::array<std::exception_ptr,4> errors;
    cv::parallel_for_(cv::Range(0,4),[&](const cv::Range& range){ for(int k=range.start;k<range.end;++k) try {
        const auto c=site::get().screws[k];
        cv::Point shift;double ncc=0;
        const bool near=(hint!=cv::Point(0,0))&&find_peak(image,k,hint,60,shift,ncc)&&ncc>=.6;
        if(!near){
            cv::Rect window(c.x-radius-search,c.y-radius-search,side+2*search,side+2*search);
            cv::Mat t8,scores;templates_[k].convertTo(t8,CV_8U);
            cv::matchTemplate(image(window),t8,scores,cv::TM_CCOEFF_NORMED);
            cv::Point peak;cv::minMaxLoc(scores,nullptr,&ncc,nullptr,&peak);shift=peak-cv::Point(search,search);
        }
        if(ncc<.6)throw std::runtime_error("Pin NCC below 0.6 at patch "+std::to_string(k));
        const cv::Point peak=shift+cv::Point(search,search);
        double dx=shift.x,dy=shift.y,rms=1e9;
        bool converged=false;
        // Forward-additive translation GN, gain/offset fitted at each step.
        // Match the template with current(c + displacement); positive dx means right.
        for(int it=0;it<50;++it) {
            cv::Mat cur=sample(image,c.x-radius+dx,c.y-radius+dy),ix,iy;
            cv::Sobel(cur,ix,CV_64F,1,0,3,1./8);cv::Sobel(cur,iy,CV_64F,0,1,3,1./8);
            cv::Scalar cm,cs,tm,ts;cv::meanStdDev(cur,cm,cs);cv::meanStdDev(templates_[k],tm,ts);
            cv::Mat z=cur-cm[0],t=templates_[k]-tm[0];
            double gain=z.dot(t)/std::max(z.dot(z),1e-12),offset=tm[0]-gain*cm[0];
            cv::Mat residual=gain*cur+offset-templates_[k];ix*=gain;iy*=gain;
            double hxx=ix.dot(ix),hxy=ix.dot(iy),hyy=iy.dot(iy),det=hxx*hyy-hxy*hxy;
            if(det<1e-12)throw std::runtime_error("Pin GN singular Hessian");
            double bx=-ix.dot(residual),by=-iy.dot(residual);
            double sx=(hyy*bx-hxy*by)/det,sy=(hxx*by-hxy*bx)/det;
            rms=std::sqrt(residual.dot(residual)/residual.total());
            if(!std::isfinite(sx+sy)||std::abs(sx)>3||std::abs(sy)>3)
                throw std::runtime_error("Pin GN diverged");
            dx+=sx;dy+=sy;
            if(std::abs(sx)+std::abs(sy)<1e-4){converged=true;break;}
        }
        if(!converged||std::abs(dx-peak.x+search)>8||std::abs(dy-peak.y+search)>8)
            throw std::runtime_error("Pin refinement did not converge");
        m.displacement.at<double>(2*k)=dx;m.displacement.at<double>(2*k+1)=dy;
        m.quality.at<double>(2*k)=ncc;m.quality.at<double>(2*k+1)=rms;
    } catch(...) { errors[k]=std::current_exception(); } });
    for(auto& e:errors) if(e) std::rethrow_exception(e);
    return m;
}
}
