#pragma once
#include "numeric.hpp"
#include "native_lapack.hpp"
#include <map>
#include <tuple>
namespace semes {
struct Calibration {
    cv::Mat projection,jacobian;double noise_rms;Json diagnostics;std::vector<int> training_indices;
    cv::Mat project(const cv::Mat& x)const{return x*projection;}
    static Calibration fit(const cv::Mat& X,const cv::Mat& commands,const std::vector<std::string>& dates,Json* trace=nullptr,const std::vector<int>& training_ids={}){
        using namespace numeric;
        if(X.type()!=CV_64F||commands.type()!=CV_64F||X.rows!=commands.rows||commands.cols!=2||dates.size()!=X.rows||!cv::checkRange(X)||!cv::checkRange(commands))throw std::runtime_error("Finite, aligned training observations and commands required");
        std::map<std::tuple<std::string,double,double>,std::vector<int>> groups;
        for(int i=0;i<X.rows;++i)groups[{dates[i],commands.at<double>(i,0),commands.at<double>(i,1)}].push_back(i);
        cv::Mat means(int(groups.size()),X.cols,CV_64F),q(int(groups.size()),2,CV_64F),residual;int n=0;
        for(const auto& group:groups){auto mean=means.row(n);mean.setTo(0);for(int i:group.second)mean+=X.row(i);for(int k=0;k<mean.cols;++k)mean.at<double>(k)=mean.at<double>(k)/double(group.second.size());q.at<double>(n,0)=std::get<1>(group.first);q.at<double>(n,1)=std::get<2>(group.first);if(group.second.size()>1)for(int i:group.second)residual.push_back(X.row(i)-mean);++n;}
        if(residual.rows<2)throw std::runtime_error("Repeated training conditions required for noise estimation");
        cv::Mat design(q.rows,6,CV_64F),linear(q.rows,3,CV_64F);
        for(int i=0;i<q.rows;++i){double a=q.at<double>(i,0),b=q.at<double>(i,1);double v[]={1,a,b,a*a,a*b,b*b};for(int k=0;k<6;++k)design.at<double>(i,k)=v[k];for(int k=0;k<3;++k)linear.at<double>(i,k)=v[k];}
        auto coefficients=lapack::least_squares(design,means);cv::Mat signal=coefficients.rowRange(1,6).t();for(int k=0;k<signal.cols;++k){std::vector<double> squared(signal.rows);for(int i=0;i<signal.rows;++i)squared[i]=signal.at<double>(i,k)*signal.at<double>(i,k);double scale=std::max(std::sqrt(lapack::pairwise_sum(squared.data(),squared.size())),1e-12);for(int i=0;i<signal.rows;++i)signal.at<double>(i,k)=signal.at<double>(i,k)/scale;}
        auto noise_svd=lapack::svd(residual);cv::Mat stacked;cv::hconcat(signal,noise_svd.vt.rowRange(0,std::min(7,noise_svd.vt.rows)).t(),stacked);
        auto basis=lapack::svd(stacked);int rank=0;double threshold=std::max(basis.w.at<double>(0)*1e-10,1e-12);while(rank<basis.w.rows&&rank<12&&basis.w.at<double>(rank)>threshold)++rank;if(!rank)throw std::runtime_error("No identifiable training basis");
        cv::Mat Q=basis.u.colRange(0,rank),noise=residual*Q,C=noise.t()*noise/double(std::max(residual.rows-1,1));double floor=std::max(cv::trace(C)[0]/std::max(rank,1)*1e-4,1e-8);cv::Mat shrink=.9*C;for(int i=0;i<rank;++i)shrink.at<double>(i,i)+=.1*C.at<double>(i,i)+floor;
        cv::Mat eigen,vectors;if(!cv::eigen(shrink,eigen,vectors))throw std::runtime_error("Noise covariance eigensystem failed");cv::Mat diagonal=cv::Mat::zeros(rank,rank,CV_64F);for(int i=0;i<rank;++i)diagonal.at<double>(i,i)=1/std::sqrt(std::max(eigen.at<double>(i),floor));
        Calibration result;if(!training_ids.empty()&&training_ids.size()!=size_t(X.rows))throw std::runtime_error("Training index count mismatch");result.training_indices=training_ids.empty()?indices(X.rows):training_ids;result.projection=Q*vectors.t()*diagonal*vectors;auto fitted=lapack::least_squares(linear,means*result.projection);result.jacobian=fitted.rowRange(1,3).t();result.noise_rms=rms(residual*result.projection);
        if(trace){*trace=Json::dict();(*trace)["means"]=jsonmatrix(means);(*trace)["residual"]=jsonmatrix(residual);(*trace)["coefficients"]=jsonmatrix(coefficients);(*trace)["signal"]=jsonmatrix(signal);(*trace)["noise_vt"]=jsonmatrix(noise_svd.vt);(*trace)["basis_singular_values"]=jsonmatrix(basis.w);(*trace)["Q"]=jsonmatrix(Q);(*trace)["C"]=jsonmatrix(C);}
        auto& d=result.diagnostics;d=Json::dict();d["raw_features"]=X.cols;d["projected_features"]=rank;d["train_images"]=X.rows;d["train_conditions"]=means.rows;d["noise_residual_rows"]=residual.rows;d["whitened_repeat_noise_rms"]=result.noise_rms;
        auto singular=lapack::svd(result.jacobian).w;d["jacobian_singular_values"]=Json::list();for(int i=0;i<singular.rows;++i)d["jacobian_singular_values"].push(singular.at<double>(i));d["jacobian_condition"]=singular.at<double>(0)/std::max(singular.at<double>(singular.rows-1),1e-20);for(auto key:{"probe_feature_norms","probe_total_signal_to_difference_noise","probe_axis_distinguishing_signal_to_difference_noise"})d[key]=Json::list();
        double denominator=std::max(std::sqrt(2.)*result.noise_rms*std::sqrt(double(result.jacobian.rows)),1e-12);for(int axis=0;axis<2;++axis){double probe=axis==0?.05:.01;auto current=result.jacobian.col(axis),other=result.jacobian.col(1-axis);double signal=cv::norm(current)*probe;cv::Mat orthogonal=current-other*(other.dot(current)/std::max(other.dot(other),1e-20));d["probe_feature_norms"].push(signal);d["probe_total_signal_to_difference_noise"].push(signal/denominator);d["probe_axis_distinguishing_signal_to_difference_noise"].push(cv::norm(orthogonal)*probe/denominator);}
        d["projection_source"]="TRAIN only: five forward response directions plus seven repeat-noise singular vectors";d["scope"]="Training calibration only; all fitting uses the supplied training rows";return result;
    }
    Json json()const{Json j=Json::dict();j["projection"]=jsonmatrix(projection);j["jacobian"]=jsonmatrix(jacobian);j["noise_rms"]=noise_rms;j["diagnostics"]=diagnostics;j["training_indices"]=Json::list();for(int i:training_indices)j["training_indices"].push(i);return j;}
};
}
