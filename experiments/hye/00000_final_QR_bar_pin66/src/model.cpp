#include "final66.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace final66 {
cv::Mat features(const cv::Mat& qr, const cv::Mat& pins,
                 const std::vector<double>& block_x) {
    if(qr.type()!=CV_64F || qr.rows!=1 || qr.cols!=102 ||
       pins.type()!=CV_64F || pins.rows!=1 || pins.cols!=8 ||
       !cv::checkRange(qr) || !cv::checkRange(pins) || block_x.size()!=48)
        throw std::invalid_argument("Expected finite QR102, pins8 and 48 bar centers");
    cv::Mat out(1,66,CV_64F);
    qr.colRange(0,54).copyTo(out.colRange(0,54));
    for(int bar=0;bar<2;++bar) {
        double mx=0,my=0,sxx=0,sxy=0;
        for(int k=24*bar;k<24*(bar+1);++k) {
            if(!std::isfinite(block_x[k])) throw std::invalid_argument("Nonfinite bar center");
            mx+=block_x[k]/1000./24; my+=qr.at<double>(54+k)/24;
        }
        for(int k=24*bar;k<24*(bar+1);++k) {
            const double x=block_x[k]/1000.-mx;
            sxx+=x*x; sxy+=x*(qr.at<double>(54+k)-my);
        }
        if(sxx<1e-12) throw std::invalid_argument("Degenerate bar geometry");
        out.at<double>(54+2*bar)=my;
        out.at<double>(55+2*bar)=sxy/sxx;
    }
    pins.copyTo(out.colRange(58,66));
    return out;
}
cv::Mat LinearModel::design(const cv::Mat& x) const {
    if(mean_.empty() || x.type()!=CV_64F || x.cols!=66 || x.rows<1 || !cv::checkRange(x))
        throw std::invalid_argument("Expected finite Nx66 model features");
    cv::Mat a=cv::Mat::ones(x.rows,67,CV_64F);
    for(int i=0;i<x.rows;++i) for(int j=0;j<66;++j)
        a.at<double>(i,j+1)=(x.at<double>(i,j)-mean_.at<double>(j))/scale_.at<double>(j);
    return a;
}
cv::Mat LinearModel::predict(const cv::Mat& x) const {
    if(weights_.empty()) throw std::logic_error("Model not loaded");
    cv::Mat p=design(x)*weights_;
    if(!cv::checkRange(p)) throw std::runtime_error("Nonfinite estimate");
    return p;
}
void LinearModel::load(const std::string& path) {
    cv::FileStorage f(path,cv::FileStorage::READ);
    if(!f.isOpened()) throw std::runtime_error("Cannot load weighted_linear66.yml");
    int degree=0; cv::Mat mean,scale,w;
    f["degree"]>>degree; f["mean"]>>mean; f["scale"]>>scale; f["weights"]>>w;
    if(degree!=1 || mean.type()!=CV_64F || scale.type()!=CV_64F || w.type()!=CV_64F ||
       mean.rows!=1 || mean.cols!=66 || scale.size()!=mean.size() || w.rows!=67 || w.cols!=2 ||
       !cv::checkRange(mean) || !cv::checkRange(scale) || !cv::checkRange(w))
        throw std::runtime_error("Invalid linear66 model schema");
    for(int j=0;j<66;++j) if(scale.at<double>(j)<=0) throw std::runtime_error("Invalid scale");
    mean_=mean; scale_=scale; weights_=w;
}
void LinearModel::save(const std::string& path) const {
    if(weights_.empty()) throw std::logic_error("No trained model");
    cv::FileStorage f(path,cv::FileStorage::WRITE);
    if(!f.isOpened()) throw std::runtime_error("Cannot save model");
    f<<"degree"<<1<<"mean"<<mean_<<"scale"<<scale_<<"weights"<<weights_;
}
void LinearModel::fit(const cv::Mat& x,const cv::Mat& y) {
    if(x.type()!=CV_64F || x.cols!=66 || x.rows<67 || y.type()!=CV_64F ||
       y.rows!=x.rows || y.cols!=2 || !cv::checkRange(x) || !cv::checkRange(y))
        throw std::invalid_argument("Expected >=67 finite training rows,66 features,2 targets");
    mean_=cv::Mat(1,66,CV_64F); scale_=mean_.clone();
    for(int j=0;j<66;++j) {
        cv::Scalar mean,sd; cv::meanStdDev(x.col(j),mean,sd);
        mean_.at<double>(j)=mean[0]; scale_.at<double>(j)=std::max(sd[0],1e-6);
    }
    const auto a=design(x);
    cv::Mat penalty=cv::Mat::eye(67,67,CV_64F)*(1e-7*x.rows);
    penalty.at<double>(0,0)=0;
    std::vector<double> weights(x.rows,1);
    for(int iteration=0;iteration<6;++iteration) {
        cv::Mat aw=a.clone(),yw=y.clone();
        for(int i=0;i<a.rows;++i) {aw.row(i)*=std::sqrt(weights[i]); yw.row(i)*=std::sqrt(weights[i]);}
        if(!cv::solve(aw.t()*aw+penalty,aw.t()*yw,weights_,cv::DECOMP_SVD) || !cv::checkRange(weights_))
            throw std::runtime_error("Weighted ridge fit failed");
        cv::Mat p=a*weights_; double sum=0;
        for(int i=0;i<a.rows;++i) {
            const double r=std::max(std::abs(p.at<double>(i,0)-y.at<double>(i,0))/.045,
                                    std::abs(p.at<double>(i,1)-y.at<double>(i,1))/.01);
            weights[i]=.5*weights[i]+.5*std::min(25.,std::max(.25,r*r)); sum+=weights[i];
        }
        for(auto& w:weights) w*=weights.size()/sum;
    }
}
}
