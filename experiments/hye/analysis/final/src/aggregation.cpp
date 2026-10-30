#include "aggregation.hpp"
#include <algorithm>
#include <numeric>
namespace semes {
    namespace {
        double median(std::vector<double> x) {
            if (x.empty()) throw std::runtime_error("Empty median");
            std::sort(x.begin(), x.end());
            return x.size() % 2 ? x[x.size()/2] : .5*(x[x.size()/2-1]+x[x.size()/2]);
        }
        cv::Mat polynomial(const cv::Mat& p,int degree,double local_center=0) {
            cv::Mat b(p.rows,(degree+1)*(degree+2)/2,CV_64F);
            for(int k=0; k<p.rows; ++k) {
                double u=(p.at<double>(k,0)-(local_center?local_center:1906))/(local_center?100:486);
                double v=(p.at<double>(k,1)-1070)/(local_center?100:122);
                int col=0;
                for(int i=0; i<=degree; ++i) for(int j=0; j<=i; ++j) b.at<double>(k,col++)=std::pow(u,i-j)*std::pow(v,j);
            }
            return b;
        }
        double reweight(double z,const std::string& robust) {
            if(robust=="huber")return std::min(1.,1/std::max(z,1e-10));
            if(robust=="soft_l1")return 1/std::sqrt(1+z*z);
            return 1/(1+z*z);
        }
        struct Aggregated {
            cv::Mat coefficient;
            std::vector<double> weights;
        };
        Aggregated aggregate(const cv::Mat& b,const cv::Mat& d,const std::vector<double>& base,const std::string& robust) {
            auto w=base;
            cv::Mat coefficient;
            for(int iteration=0; iteration<(robust.empty()?1:8); ++iteration) {
                cv::Mat bw=b.clone(),dw=d.clone();
                for(int i=0; i<b.rows; ++i) {
                    double sw=std::sqrt(std::max(w[i],.001));
                    bw.row(i)*=sw;
                    dw.row(i)*=sw;
                }
                cv::solve(bw,dw,coefficient,cv::DECOMP_SVD);
                if(!robust.empty()) {
                    cv::Mat error=d-b*coefficient;
                    std::vector<double> r;
                    for(int i=0; i<error.rows; ++i)r.push_back(cv::norm(error.row(i)));
                    double med=median(r);
                    std::vector<double> deviations;
                    for(double x:r)deviations.push_back(std::abs(x-med));
                    double scale=std::max(1.4826*median(deviations),.015);
                    for(size_t i=0; i<w.size(); ++i)w[i]=std::max(base[i]*reweight(r[i]/(2.5*scale),robust),.001);
                }
            }
            return {
                coefficient.reshape(1,1).clone(),w
            };
        }
        cv::Mat select_rows(const cv::Mat& a,const std::vector<int>& index) {
            cv::Mat b(int(index.size()),a.cols,CV_64F);
            for(int i=0; i<int(index.size()); ++i)a.row(index[i]).copyTo(b.row(i));
            return b;
        }
    }
    cv::Mat aggregate_features(const cv::Mat& p,const cv::Mat& d,const cv::Mat& q,const std::string& measurement,const std::string& feature) {
        if(p.type()!=CV_64F||d.type()!=CV_64F||q.type()!=CV_64F||p.cols!=2||d.cols!=2||q.cols<3||p.rows!=d.rows||q.rows!=d.rows||!cv::checkRange(p)||!cv::checkRange(d)||!cv::checkRange(q)) throw std::runtime_error("Finite aligned points, displacement and quality required");
        std::vector<double> w(p.rows,1.);
        std::string robust;
        if(measurement=="huber"||measurement=="soft_l1"||measurement=="cauchy")robust=measurement;
        else if(measurement!="uniform"&&measurement!="photo"&&measurement!="stability"&&measurement!="contrast")throw std::runtime_error("Unknown measurement weighting");
        for(int i=0; i<p.rows; ++i) {
            double photo=q.at<double>(i,0)/10,stability=q.at<double>(i,1)/.2;
            if(measurement=="photo")w[i]=1/(1+photo*photo);
            else if(measurement=="stability")w[i]=1/(1+stability*stability);
            else if(measurement=="contrast")w[i]=std::pow(std::min(q.at<double>(i,2)/100,2.),2)/(1+photo*photo+stability*stability);
            w[i]=std::clamp(w[i],.001,4.);
        }
        if(feature.rfind("global",0)==0) {
            int degree=std::stoi(feature.substr(6));
            if(degree<1||degree>3)throw std::runtime_error("Unsupported polynomial degree");
            return aggregate(polynomial(p,degree),d,w,robust).coefficient;
        }
        if(feature.rfind("local",0)==0||feature=="quadrants") {
            const double centers[]={
                1644,1901,2159
            };
            std::array<std::vector<int>,3> indices;
            for(int i=0; i<p.rows; ++i) {
                int r=0;
                for(int k=1; k<3; ++k)if(std::abs(p.at<double>(i,0)-centers[k])<std::abs(p.at<double>(i,0)-centers[r]))r=k;
                indices[r].push_back(i);
            }
            std::vector<double> result;
            for(int r=0; r<3; ++r) {
                auto pp=select_rows(p,indices[r]),dd=select_rows(d,indices[r]);
                std::vector<double> weights;
                for(int i:indices[r])weights.push_back(w[i]);
                int degree=feature=="quadrants"?1:std::stoi(feature.substr(5));
                if(degree<1||degree>2)throw std::runtime_error("Unsupported local degree");
                auto a=aggregate(polynomial(pp,degree,centers[r]),dd,weights,robust);
                if(feature!="quadrants") {
                    for(int i=0; i<a.coefficient.cols; ++i)result.push_back(a.coefficient.at<double>(i));
                }
                else {
                    std::vector<double> x,y;
                    for(int i=0; i<pp.rows; ++i){
                        x.push_back(pp.at<double>(i,0));
                        y.push_back(pp.at<double>(i,1));
                    }
                    double mx=median(x),my=median(y);
                    for(bool lr:{
                        false,true
                    })for(bool tb:{
                        false,true
                    }) {
                        double weight=0,dx=0,dy=0;
                        for(int i=0; i<pp.rows; ++i)if((x[i]>mx)==lr&&(y[i]>my)==tb) {
                            weight+=a.weights[i];
                            dx+=a.weights[i]*dd.at<double>(i,0);
                            dy+=a.weights[i]*dd.at<double>(i,1);
                        }
                        if(weight<=0)throw std::runtime_error("Empty quadrant");
                        result.push_back(dx/weight);
                        result.push_back(dy/weight);
                    }
                }
            }
            return cv::Mat(1,int(result.size()),CV_64F,result.data()).clone();
        }
        if(feature=="homography") {
            cv::Mat a=cv::Mat::zeros(2*p.rows,8,CV_64F),destination(2*p.rows,1,CV_64F),co;
            for(int i=0; i<p.rows; ++i) {
                double u=(p.at<double>(i,0)-1906)/486,v=(p.at<double>(i,1)-1070)/122;
                double x=(p.at<double>(i,0)+d.at<double>(i,0)-1906)/486,y=(p.at<double>(i,1)+d.at<double>(i,1)-1070)/122;
                double xx[]={
                    u,v,1,0,0,0,-x*u,-x*v
                },yy[]={
                    0,0,0,u,v,1,-y*u,-y*v
                };
                for(int j=0; j<8; ++j){
                    a.at<double>(2*i,j)=xx[j];
                    a.at<double>(2*i+1,j)=yy[j];
                }
                destination.at<double>(2*i)=x;
                destination.at<double>(2*i+1)=y;
            }
            auto ww=w;
            for(int iteration=0; iteration<(robust.empty()?1:8); ++iteration) {
                auto aw=a.clone(),dw=destination.clone();
                for(int i=0; i<p.rows; ++i)for(int k=0; k<2; ++k){
                    double sw=std::sqrt(ww[i]);
                    aw.row(2*i+k)*=sw;
                    dw.row(2*i+k)*=sw;
                }
                cv::solve(aw,dw,co,cv::DECOMP_SVD);
                if(!robust.empty()) {
                    cv::Mat residual=a*co-destination;
                    std::vector<double> error;
                    for(int i=0; i<p.rows; ++i)error.push_back(std::hypot(residual.at<double>(2*i),residual.at<double>(2*i+1)));
                    double scale=std::max(median(error)*1.4826,1e-5);
                    for(int i=0; i<p.rows; ++i)ww[i]=std::max(w[i]*reweight(error[i]/(2.5*scale),robust),.001);
                }
            }
            co.at<double>(0)-=1;
            co.at<double>(4)-=1;
            return co.reshape(1,1).clone();
        }
        throw std::runtime_error("Unknown feature representation");
    }
}
