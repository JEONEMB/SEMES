#include "final_pin.hpp"
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace final_pin {
namespace {
const cv::Point centers[]={{974,1352},{2864,1358},{1010,1677},{2817,1681}};
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
        center(cv::Rect(centers[k].x-radius,centers[k].y-radius,side,side)).convertTo(templates_[k],CV_64F);
}
Measurement Observer::measure(const cv::Mat& image) const {
    image_check(image);
    Measurement m{cv::Mat(1,8,CV_64F),cv::Mat(1,8,CV_64F)};
    for(int k=0;k<4;++k) {
        const auto c=centers[k];
        cv::Rect window(c.x-radius-search,c.y-radius-search,side+2*search,side+2*search);
        cv::Mat t8,scores;templates_[k].convertTo(t8,CV_8U);
        cv::matchTemplate(image(window),t8,scores,cv::TM_CCOEFF_NORMED);
        cv::Point peak;double ncc;cv::minMaxLoc(scores,nullptr,&ncc,nullptr,&peak);
        if(ncc<.6)throw std::runtime_error("Pin NCC below 0.6 at patch "+std::to_string(k));
        double dx=peak.x-search,dy=peak.y-search,rms=1e9;
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
    }
    return m;
}
cv::Mat relative_features(const cv::Mat& qr,const cv::Mat& pins) {
    if(qr.type()!=CV_64F||qr.rows!=1||qr.cols!=102||pins.type()!=CV_64F||pins.rows!=1||pins.cols!=8||!cv::checkRange(qr)||!cv::checkRange(pins))
        throw std::invalid_argument("Expected finite QR102 and pin8");
    cv::Mat f(1,14,CV_64F);double wx=0,wy=0;
    for(int k=0;k<4;++k){wx+=pins.at<double>(2*k)/4;wy+=pins.at<double>(2*k+1)/4;}
    for(int k=0;k<3;++k){double x=0,y=0;for(int p=0;p<9;++p){x+=qr.at<double>(18*k+2*p)/9;y+=qr.at<double>(18*k+2*p+1)/9;}f.at<double>(2*k)=x-wx;f.at<double>(2*k+1)=y-wy;}
    pins.copyTo(f.colRange(6,14));return f;
}
cv::Mat with_bars_features(const cv::Mat& qr,const cv::Mat& pins,const std::vector<double>& block_x) {
    auto base=relative_features(qr,pins);
    if(block_x.size()!=48)throw std::invalid_argument("Expected 48 bar block centers");
    cv::Mat f(1,18,CV_64F);base.copyTo(f.colRange(0,14));double wall_y=0;
    for(int k=0;k<4;++k)wall_y+=pins.at<double>(2*k+1)/4;
    for(int bar=0;bar<2;++bar){double mx=0,my=0,sxx=0,sxy=0;
        for(int k=bar*24;k<(bar+1)*24;++k){if(!std::isfinite(block_x[k]))throw std::invalid_argument("Nonfinite bar center");mx+=block_x[k]/1000./24;my+=qr.at<double>(54+k)/24;}
        for(int k=bar*24;k<(bar+1)*24;++k){double x=block_x[k]/1000.-mx;sxx+=x*x;sxy+=x*(qr.at<double>(54+k)-my);}
        if(sxx<1e-12)throw std::invalid_argument("Degenerate bar geometry");
        f.at<double>(14+2*bar)=my-wall_y;f.at<double>(15+2*bar)=sxy/sxx;
    }return f;
}
int Regressor::feature_count()const{return mean_.cols;}
cv::Mat Regressor::design(const cv::Mat& features)const {
    if(features.type()!=CV_64F||features.cols!=mean_.cols||!cv::checkRange(features))throw std::invalid_argument("Invalid regression features");
    int n=features.cols,cols=1+n+(degree_==2?n*(n+1)/2:0);cv::Mat a=cv::Mat::ones(features.rows,cols,CV_64F);
    for(int i=0;i<features.rows;++i){int c=1;std::vector<double> z(n);for(int j=0;j<n;++j){z[j]=(features.at<double>(i,j)-mean_.at<double>(j))/scale_.at<double>(j);a.at<double>(i,c++)=z[j];}if(degree_==2)for(int j=0;j<n;++j)for(int k=j;k<n;++k)a.at<double>(i,c++)=z[j]*z[k];}return a;
}
void Regressor::fit(const cv::Mat& x,const cv::Mat& y,int degree,double ridge){
    if(x.rows<10||x.type()!=CV_64F||y.type()!=CV_64F||y.rows!=x.rows||y.cols!=2||!cv::checkRange(x)||!cv::checkRange(y)||(degree!=1&&degree!=2)||ridge<0)throw std::invalid_argument("Invalid model training input");
    degree_=degree;ridge_=ridge;mean_=cv::Mat(1,x.cols,CV_64F);scale_=mean_.clone();
    for(int j=0;j<x.cols;++j){cv::Scalar m,s;cv::meanStdDev(x.col(j),m,s);mean_.at<double>(j)=m[0];scale_.at<double>(j)=std::max(s[0],1e-6);}
    cv::Mat a=design(x),penalty=cv::Mat::eye(a.cols,a.cols,CV_64F)*ridge;penalty.at<double>(0,0)=0;
    if(!cv::solve(a.t()*a+penalty,a.t()*y,weights_,cv::DECOMP_SVD)||!cv::checkRange(weights_))throw std::runtime_error("Regression fit failed");
}
cv::Vec2d Regressor::predict(const cv::Mat& x)const{if(weights_.empty()||x.rows!=1)throw std::runtime_error("Model not trained or invalid row");cv::Mat p=design(x)*weights_;return {p.at<double>(0),p.at<double>(1)};}
void Regressor::save(const std::string& path)const{cv::FileStorage f(path,cv::FileStorage::WRITE);if(!f.isOpened())throw std::runtime_error("Cannot save pin model");f<<"degree"<<degree_<<"ridge"<<ridge_<<"mean"<<mean_<<"scale"<<scale_<<"weights"<<weights_;}
void Regressor::load(const std::string& path){cv::FileStorage f(path,cv::FileStorage::READ);if(!f.isOpened())throw std::runtime_error("Cannot load pin model");f["degree"]>>degree_;f["ridge"]>>ridge_;f["mean"]>>mean_;f["scale"]>>scale_;f["weights"]>>weights_;if((degree_!=1&&degree_!=2)||weights_.cols!=2||mean_.empty()||scale_.size()!=mean_.size()||!cv::checkRange(mean_)||!cv::checkRange(scale_)||!cv::checkRange(weights_))throw std::runtime_error("Invalid saved pin model");for(int j=0;j<scale_.cols;++j)if(scale_.at<double>(j)<=0)throw std::runtime_error("Invalid scale");}
}
