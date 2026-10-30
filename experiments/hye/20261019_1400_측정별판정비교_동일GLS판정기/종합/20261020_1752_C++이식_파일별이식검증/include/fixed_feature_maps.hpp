#pragma once
#include "native_lapack.hpp"
namespace semes {
class FixedFeatureMaps {
    cv::Mat p12_;std::array<cv::Mat,3> affine_,quadratic_;std::array<std::vector<int>,3> groups_;int count_;
public:
    FixedFeatureMaps(const cv::Mat& points,const std::vector<int>& regions):count_(points.rows){
        if(points.cols!=2||points.rows!=int(regions.size())||!cv::checkRange(points))throw std::runtime_error("Finite aligned point geometry required");cv::Mat basis(count_,6,CV_64F);for(int i=0;i<count_;++i){double u=(points.at<double>(i,0)-1906)/486,v=(points.at<double>(i,1)-1070)/122;double values[]={1,u,v,u*u,u*v,v*v};for(int k=0;k<6;++k)basis.at<double>(i,k)=values[k];if(regions[i]>=0&&regions[i]<3)groups_[regions[i]].push_back(i);}p12_=lapack::pinv(basis);
        for(int qr=0;qr<3;++qr){const auto& group=groups_[qr];if(group.empty())throw std::runtime_error("Each QR region requires reference points");std::vector<double> x,y;for(int i:group){x.push_back(points.at<double>(i,0));y.push_back(points.at<double>(i,1));}double mx=lapack::pairwise_sum(x.data(),x.size())/x.size(),my=lapack::pairwise_sum(y.data(),y.size())/y.size();cv::Mat b(int(group.size()),6,CV_64F);for(int i=0;i<b.rows;++i){double u=(x[i]-mx)/100,v=(y[i]-my)/100;double values[]={1,u,v,u*u,u*v,v*v};for(int k=0;k<6;++k)b.at<double>(i,k)=values[k];}affine_[qr]=lapack::pinv(b.colRange(0,3));quadratic_[qr]=lapack::pinv(b);}
    }
    cv::Mat transform(const cv::Mat& displacement,const std::string& name)const{
        if(displacement.rows!=count_||displacement.cols!=2||displacement.type()!=CV_64F)throw std::runtime_error("Displacement geometry mismatch");if(name=="raw1260")return displacement.reshape(1,1).clone();if(name=="P12"){cv::Mat c=p12_*displacement;return c.reshape(1,1).clone();}cv::Mat result(1,name=="QR18"?18:36,CV_64F);int offset=0;for(int qr=0;qr<3;++qr){const auto& m=name=="QR18"?affine_[qr]:quadratic_[qr];cv::Mat c=m*numeric::rows(displacement,groups_[qr]);c=c.reshape(1,1);c.copyTo(result.colRange(offset,offset+c.cols));offset+=c.cols;}return result;
    }
};
inline std::map<std::string,cv::Mat> generalization_feature_bank(const Npy& d,const Npy& p,const Npy& r){
    if(d.shape.size()!=3||d.shape[2]!=2||p.shape.size()!=2||p.shape[1]!=2||d.shape[1]!=p.shape[0]||r.count()!=p.shape[0])throw std::runtime_error("Expected aligned NxPointsx2 displacements");cv::Mat points(int(p.shape[0]),2,CV_64F);std::vector<int> regions;for(int i=0;i<points.rows;++i){points.at<double>(i,0)=p.number(i*2);points.at<double>(i,1)=p.number(i*2+1);regions.push_back(int(r.number(i)));}FixedFeatureMaps maps(points,regions);std::map<std::string,cv::Mat> result;
    for(size_t row=0;row<d.shape[0];++row){cv::Mat delta(points.rows,2,CV_64F);std::vector<double> shifts;for(size_t i=0;i<p.shape[0]*2;++i)delta.ptr<double>()[i]=d.number(row*p.shape[0]*2+i);for(int i=0;i<delta.rows;++i)shifts.push_back(delta.at<double>(i,0));double h=lapack::pairwise_sum(shifts.data(),shifts.size())/shifts.size()/100;
        for(auto name:{"P12","QR18","QR36"})result[name].push_back(maps.transform(delta,name));cv::Mat raw=delta.reshape(1,1),scaled=raw*h,scaled2=scaled*h,shifted,shifted2;cv::hconcat(raw,scaled,shifted);cv::hconcat(shifted,scaled2,shifted2);result["raw1260"].push_back(raw);result["raw_shift"].push_back(shifted);result["raw_shift2"].push_back(shifted2);
    }return result;
}
}
