#pragma once
#include "native_lapack.hpp"
#include <map>
#include <tuple>
namespace semes {
inline cv::Mat pair_graph(const cv::Mat& commands,const std::vector<std::string>& dates,const std::string& kind){
    if(commands.cols!=2||commands.rows!=int(dates.size()))throw std::runtime_error("Aligned training graph inputs required");std::map<std::tuple<std::string,double,double>,std::vector<int>> groups;int n=commands.rows;for(int i=0;i<n;++i){double y=commands.at<double>(i,0),t=commands.at<double>(i,1);groups[{dates[i],kind=="repeat"?y:kind=="sameT"?t:y,kind=="repeat"?t:0.}].push_back(i);}cv::Mat result=cv::Mat::zeros(n,n,CV_64F);int count=0;for(const auto& pair:groups){const auto& rows=pair.second;if(rows.size()<2)continue;for(int i:rows)for(int j:rows)result.at<double>(i,j)+=(i==j?1.:0.)-1./rows.size();count+=int(rows.size())-1;}result*=double(n)/std::max(count,1);return result;
}
class PairedRidge {
public:
    double alpha=1,pair_weight=1,repeat_weight=0;cv::Mat mean,stddev,beta,origins;
    PairedRidge(double a=1,double p=1,double r=0):alpha(a),pair_weight(p),repeat_weight(r){if(a<=0||p<0||r<0)throw std::runtime_error("Invalid pair regularization");}
    PairedRidge& fit(const cv::Mat& X,const cv::Mat& commands,const std::vector<std::string>& dates){
        if(X.rows!=commands.rows||commands.cols!=2||!cv::checkRange(X)||!cv::checkRange(commands))throw std::runtime_error("Finite aligned paired training rows required");int n=X.rows;mean=cv::Mat::zeros(1,X.cols,CV_64F);stddev=cv::Mat(1,X.cols,CV_64F);cv::Mat Z=X.clone();for(int k=0;k<X.cols;++k){double sum=0;for(int i=0;i<n;++i)sum+=X.at<double>(i,k);double m=sum/n;mean.at<double>(k)=m;double variance=0;for(int i=0;i<n;++i){double delta=X.at<double>(i,k)-m;variance+=delta*delta;}double scale=std::max(std::sqrt(variance/n),1e-6);stddev.at<double>(k)=scale;for(int i=0;i<n;++i)Z.at<double>(i,k)=(X.at<double>(i,k)-m)/scale;}
        cv::Mat repeat=pair_graph(commands,dates,"repeat"),K=lapack::multiply(Z,Z.t());beta=cv::Mat(X.cols,2,CV_64F);origins=cv::Mat(1,2,CV_64F);
        for(int output=0;output<2;++output){auto axis=pair_graph(commands,dates,output==0?"sameT":"sameY");cv::Mat W(n,n,CV_64F),supervision(n,n,CV_64F);for(int i=0;i<n;++i)for(int j=0;j<n;++j){double value=(i==j?1.:0.)+pair_weight*axis.at<double>(i,j);supervision.at<double>(i,j)=value;W.at<double>(i,j)=value+repeat_weight*repeat.at<double>(i,j);}cv::Mat Wy=lapack::multiply(supervision,commands.col(output));std::vector<double> target_values;for(int i=0;i<n;++i)target_values.push_back(commands.at<double>(i,output));double origin=lapack::pairwise_sum(target_values.data(),target_values.size())/n;origins.at<double>(output)=origin;cv::Mat target=Wy-origin,system=lapack::multiply(W,K);for(int i=0;i<n;++i)system.at<double>(i,i)+=alpha;auto dual=lapack::solve(system,target);auto coefficient=lapack::multiply(Z.t(),dual);coefficient.copyTo(beta.col(output));}return *this;
    }
    cv::Mat predict(const cv::Mat& X)const{if(X.cols!=mean.cols)throw std::runtime_error("Paired feature width mismatch");auto Z=X.clone();for(int i=0;i<X.rows;++i)for(int k=0;k<X.cols;++k)Z.at<double>(i,k)=(X.at<double>(i,k)-mean.at<double>(k))/stddev.at<double>(k);auto p=lapack::multiply(Z,beta);for(int i=0;i<p.rows;++i)p.row(i)+=origins;return p;}
    Json json()const{Json j=Json::dict();j["alpha"]=alpha;j["pair_weight"]=pair_weight;j["repeat_weight"]=repeat_weight;j["mean"]=jsonmatrix(mean);j["std"]=jsonmatrix(stddev);j["beta"]=jsonmatrix(beta);j["origins"]=jsonmatrix(origins);return j;}
};
}
