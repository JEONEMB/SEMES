#pragma once
#include "alignment.hpp"
namespace semes {
inline Json tolerance_metrics(const cv::Mat& truth,const cv::Mat& prediction){
    if(truth.type()!=CV_64F||prediction.type()!=CV_64F||truth.size()!=prediction.size()||truth.cols!=2)throw std::runtime_error("Expected matching Nx2 arrays");int failures=0,passed=0,finite=0;double sum[2]={0,0},maximum[2]={0,0};
    for(int i=0;i<truth.rows;++i){bool good=true;double e[2];for(int k=0;k<2;++k){double y=truth.at<double>(i,k),p=prediction.at<double>(i,k);good=good&&std::isfinite(y)&&std::isfinite(p);e[k]=std::abs(y-p);}if(!good){++failures;continue;}++finite;passed+=e[0]<=.09+1e-12&&e[1]<=.09+1e-12;for(int k=0;k<2;++k){sum[k]+=e[k];maximum[k]=std::max(maximum[k],e[k]);}}
    Json r=Json::dict();r["n"]=truth.rows;r["failures"]=failures;r["passed"]=passed;for(int k=0;k<2;++k){std::string axis=k==0?"Y":"T";r[axis+"_MAE"]=finite?Json(sum[k]/finite):Json();r[axis+"_max"]=finite?Json(maximum[k]):Json();}return r;
}
inline cv::Mat nullable_pairs(const Json& j){if(j.type!=Json::Array)throw std::runtime_error("Array of coordinate pairs required");cv::Mat m(int(j.size()),2,CV_64F);for(int i=0;i<m.rows;++i){if(j.at(i).type!=Json::Array||j.at(i).size()!=2)throw std::runtime_error("Coordinate pair required");for(int k=0;k<2;++k){auto& v=j.at(i).at(k);m.at<double>(i,k)=v.type==Json::Null?std::numeric_limits<double>::quiet_NaN():v.num();}}return m;}
}
