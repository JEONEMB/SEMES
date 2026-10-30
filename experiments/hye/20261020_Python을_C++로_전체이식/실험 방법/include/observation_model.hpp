#pragma once
#include "forward_gls.hpp"
namespace semes {
inline cv::Mat observation_feature_map(const cv::Mat& X){
    if(X.cols<54||X.type()!=CV_64F)throw std::runtime_error("At least 54 QR feature coordinates required");cv::Mat result(X.rows,X.cols*2,CV_64F);X.copyTo(result.colRange(0,X.cols));for(int i=0;i<X.rows;++i){double sum=0;for(int k=0;k<54;k+=2)sum+=X.at<double>(i,k);double shift=sum/27/100;for(int k=0;k<X.cols;++k)result.at<double>(i,X.cols+k)=X.at<double>(i,k)*shift;}return result;
}
class ObservationModel {
public:
    ForwardGLS model;std::string scope="all";std::vector<int> training_indices;
    ObservationModel(int degree=2,double shrink=.1,std::string selected_scope="all"):model(degree,shrink),scope(std::move(selected_scope)){if(scope!="all"&&scope!="mixed")throw std::runtime_error("Unknown observation training scope");}
    ObservationModel& fit(const cv::Mat& X,const cv::Mat& commands,const std::vector<std::string>& dates,const std::vector<int>& indices={}){
        if(dates.size()!=size_t(X.rows)||X.rows!=commands.rows||(!indices.empty()&&indices.size()!=dates.size()))throw std::runtime_error("Aligned training metadata required");std::vector<int> selected;std::map<std::string,int> counts;for(int i=0;i<X.rows;++i)if(scope=="all"||dates[i]=="20261020"){selected.push_back(i);++counts[dates[i]];}if(selected.empty())throw std::runtime_error("No rows in selected training scope");auto raw=observation_feature_map(X);cv::Mat training(int(selected.size()),raw.cols,CV_64F),y(int(selected.size()),2,CV_64F),weights(int(selected.size()),1,CV_64F);training_indices.clear();for(int i=0;i<training.rows;++i){int r=selected[i];raw.row(r).copyTo(training.row(i));commands.row(r).copyTo(y.row(i));weights.at<double>(i)=1./counts.at(dates[r]);training_indices.push_back(indices.empty()?r:indices[r]);}model.fit(training,y,weights);return *this;
    }
    cv::Mat predict(const cv::Mat& X)const{return model.predict(observation_feature_map(X));}
    Json json()const{Json j=Json::dict();j["scope"]=scope;j["model"]=model.json();j["training_indices"]=Json::list();for(int i:training_indices)j["training_indices"].push(i);return j;}
    static ObservationModel from_json(const Json& j){auto f=ForwardGLS::from_json(j.at("model"));ObservationModel r(f.degree,f.shrink,j.at("scope").str());r.model=f;for(const auto& v:j.at("training_indices").array)r.training_indices.push_back(int(v.num()));return r;}
};
}
