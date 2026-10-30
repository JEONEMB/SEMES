#pragma once
#include "alignment.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
namespace semes::numeric {
inline cv::Mat least_squares(const cv::Mat& a,const cv::Mat& y,double rcond=-1){
    if(a.type()!=CV_64F||y.type()!=CV_64F||a.rows!=y.rows||a.rows==0)throw std::runtime_error("Invalid least-squares dimensions");
    cv::Mat w,u,vt;cv::SVD::compute(a,w,u,vt);
    cv::Mat z=u.t()*y;
    double cutoff=(rcond<0?std::numeric_limits<double>::epsilon()*std::max(a.rows,a.cols):rcond)*w.at<double>(0);
    for(int i=0;i<w.rows;++i){if(w.at<double>(i)>cutoff)z.row(i)/=w.at<double>(i);else z.row(i).setTo(0);}
    return vt.t()*z;
}
inline cv::Mat rows(const cv::Mat& a,const std::vector<int>& ids){cv::Mat r(int(ids.size()),a.cols,CV_64F);for(int i=0;i<int(ids.size());++i)a.row(ids[i]).copyTo(r.row(i));return r;}
inline double median(std::vector<double> a){if(a.empty())throw std::runtime_error("Median of empty input");std::sort(a.begin(),a.end());auto n=a.size();return n%2?a[n/2]:(a[n/2-1]+a[n/2])*.5;}
inline double rms(const cv::Mat& a){return cv::norm(a)/std::sqrt(double(a.total()));}
inline std::pair<double,double> line(const std::vector<double>& x,const std::vector<double>& y){
    cv::Mat a(int(x.size()),2,CV_64F),b(int(x.size()),1,CV_64F);
    for(int i=0;i<a.rows;++i){a.at<double>(i,0)=x[i];a.at<double>(i,1)=1;b.at<double>(i)=y[i];}
    auto p=least_squares(a,b);return {p.at<double>(0),p.at<double>(1)};
}
inline std::vector<int> indices(int n,int parity=-1){std::vector<int> a;for(int i=0;i<n;++i)if(parity<0||i%2==parity)a.push_back(i);return a;}
inline double dot(const std::vector<double>& a,const std::vector<double>& b){return std::inner_product(a.begin(),a.end(),b.begin(),0.);}
struct BoundedMinimum{double x,value;int evaluations;bool success;};
// Brent's bounded scalar minimization, same golden/parabolic decisions and
// termination constants as the historical SciPy bounded solver.
template<class Function> BoundedMinimum bounded_minimize(Function objective,double a,double b,double tolerance=1e-7,int max_evaluations=60){
    if(!std::isfinite(a)||!std::isfinite(b)||a>b)throw std::runtime_error("Invalid scalar bounds");
    const double epsilon=std::sqrt(2.2e-16),golden=.5*(3-std::sqrt(5.));
    double fulc=a+golden*(b-a),nfc=fulc,xf=fulc,rat=0,e=0,fx=objective(xf),ffulc=fx,fnfc=fx,fu=std::numeric_limits<double>::infinity();int count=1;
    double middle=.5*(a+b),tol1=epsilon*std::abs(xf)+tolerance/3,tol2=2*tol1;bool converged=true;
    while(std::abs(xf-middle)>tol2-.5*(b-a)){
        bool use_golden=true;
        if(std::abs(e)>tol1){
            double r=(xf-nfc)*(fx-ffulc),q=(xf-fulc)*(fx-fnfc),p=(xf-fulc)*q-(xf-nfc)*r;q=2*(q-r);if(q>0)p=-p;q=std::abs(q);r=e;e=rat;
            if(std::abs(p)<std::abs(.5*q*r)&&p>q*(a-xf)&&p<q*(b-xf)){rat=p/q;double x=xf+rat;use_golden=false;if(x-a<tol2||b-x<tol2)rat=tol1*(middle-xf>=0?1:-1);}
        }
        if(use_golden){e=xf>=middle?a-xf:b-xf;rat=golden*e;}
        double x=xf+(rat>=0?1:-1)*std::max(std::abs(rat),tol1);fu=objective(x);++count;
        if(fu<=fx){if(x>=xf)a=xf;else b=xf;fulc=nfc;ffulc=fnfc;nfc=xf;fnfc=fx;xf=x;fx=fu;}
        else{if(x<xf)a=x;else b=x;if(fu<=fnfc||nfc==xf){fulc=nfc;ffulc=fnfc;nfc=x;fnfc=fu;}else if(fu<=ffulc||fulc==xf||fulc==nfc){fulc=x;ffulc=fu;}}
        middle=.5*(a+b);tol1=epsilon*std::abs(xf)+tolerance/3;tol2=2*tol1;if(count>=max_evaluations){converged=false;break;}
    }
    return {xf,fx,count,converged&&!std::isnan(xf)&&!std::isnan(fx)&&!std::isnan(fu)};
}
class CubicSplineUniform {
    std::vector<double> y_,second_;double origin_;
public:
    CubicSplineUniform(const std::vector<double>& values,double origin):y_(values),origin_(origin){
        int n=int(y_.size());if(n<4)throw std::runtime_error("Not-a-knot spline needs four samples");cv::Mat a=cv::Mat::zeros(n,n,CV_64F),rhs=cv::Mat::zeros(n,1,CV_64F);
        a.at<double>(0,0)=1;a.at<double>(0,1)=-2;a.at<double>(0,2)=1;a.at<double>(n-1,n-3)=1;a.at<double>(n-1,n-2)=-2;a.at<double>(n-1,n-1)=1;
        for(int i=1;i<n-1;++i){a.at<double>(i,i-1)=1;a.at<double>(i,i)=4;a.at<double>(i,i+1)=1;rhs.at<double>(i)=6*(y_[i+1]-2*y_[i]+y_[i-1]);}
        cv::Mat answer;if(!cv::solve(a,rhs,answer,cv::DECOMP_LU))throw std::runtime_error("Spline solve failed");second_.assign(answer.ptr<double>(),answer.ptr<double>()+n);
    }
    double operator()(double x)const{double index=x-origin_;int k=std::clamp(int(std::floor(index)),0,int(y_.size())-2);double t=index-k,u=1-t;return u*y_[k]+t*y_[k+1]+((u*u*u-u)*second_[k]+(t*t*t-t)*second_[k+1])/6;}
};
inline std::vector<double> gaussian_reflect(const std::vector<double>& values,double sigma=1,int radius=4){
    if(values.empty()||sigma<=0)throw std::runtime_error("Invalid Gaussian profile");std::vector<double> kernel(2*radius+1),result(values.size());double total=0;for(int k=-radius;k<=radius;++k)total+=kernel[k+radius]=std::exp(-.5*k*k/(sigma*sigma));for(auto& v:kernel)v/=total;
    int n=int(values.size());for(int i=0;i<n;++i)for(int k=-radius;k<=radius;++k){int j=i+k;while(j<0||j>=n)j=j<0?-j-1:2*n-j-1;result[i]+=values[j]*kernel[k+radius];}return result;
}
}
