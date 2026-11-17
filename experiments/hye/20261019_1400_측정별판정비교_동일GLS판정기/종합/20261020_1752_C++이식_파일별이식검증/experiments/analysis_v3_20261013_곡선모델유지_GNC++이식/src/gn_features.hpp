#pragma once
#include "../../analysis_v1_20261013_고정점분할유지_선형계산C++이식/src/common.hpp"
namespace historical_v1 {
    inline cv::Mat quadratic_basis(const cv::Mat& labels){
        cv::Mat out(labels.rows,6,CV_64F);
        for(int i=0; i<labels.rows; ++i){
            double t=labels.at<double>(i,0),y=labels.at<double>(i,1);
            double values[]={
                t,y,t*t,t*y,y*y,1
            };
            for(int j=0; j<6; ++j)out.at<double>(i,j)=values[j];
        }
        return out;
    }
    inline cv::Mat gn_predict(const cv::Mat& features,const cv::Mat& labels,const std::vector<int>& train,const std::vector<int>& test){
        if(features.cols<2)return cv::Mat();
        auto ftr=selected(features,train),basis=quadratic_basis(selected(labels,train)),weights=least_squares(basis,ftr);
        cv::Mat residual=ftr-basis*weights,mean;
        cv::reduce(residual,mean,0,cv::REDUCE_AVG);
        cv::Mat centered=residual-cv::repeat(mean,residual.rows,1),covariance=centered.t()*centered/(residual.rows-1);
        double jitter=.01*cv::trace(covariance)[0]/covariance.cols;
        for(int j=0; j<covariance.cols; ++j)covariance.at<double>(j,j)+=jitter;
        cv::Mat inverse;
        if(!cv::invert(covariance,inverse,cv::DECOMP_LU))throw std::runtime_error("Historical GN covariance singular");
        auto initial=predict(features,labels,train,test,5);
        cv::Mat output(initial.size(),CV_64F);
        for(int i=0; i<initial.rows; ++i){
            auto observation=features.row(test[i]);
            cv::Mat p=initial.row(i).clone();
            auto cost=[&](const cv::Mat& candidate){
                cv::Mat r=observation-quadratic_basis(candidate)*weights,c=r*inverse*r.t();
                return c.at<double>(0,0);
            };
            double lambda=1e-6,current=cost(p);
            for(int iteration=0; iteration<100; ++iteration){
                double t=p.at<double>(0,0),y=p.at<double>(0,1);
                cv::Mat derivative=(cv::Mat_<double>(6,2)<<1,0,0,1,2*t,0,y,t,0,2*y,0,0),j=weights.t()*derivative,r=observation-quadratic_basis(p)*weights,a=j.t()*inverse*j,g=j.t()*inverse*r.t(),damped=a.clone(),step;
                for(int k=0; k<2; ++k)damped.at<double>(k,k)+=lambda*a.at<double>(k,k);
                if(!cv::solve(damped,g,step,cv::DECOMP_LU))throw std::runtime_error("Historical GN step singular");
                cv::Mat candidate=p+step.t();
                double next=cost(candidate);
                if(next<current){
                    p=candidate;
                    current=next;
                    lambda=std::max(lambda/10,1e-9);
                    if(cv::norm(step,cv::NORM_INF)<1e-9)break;
                }
                else{
                    lambda*=10;
                    if(lambda>1e8)break;
                }
            }
            p.copyTo(output.row(i));
        }
        return output;
    }
}
