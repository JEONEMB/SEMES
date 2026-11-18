#include "alignment.hpp"
namespace semes {
cv::Mat matrix(const Json& j){
    if(j.type!=Json::Array||!j.size())throw std::runtime_error("Nonempty numeric array required");bool nested=j.at(0).type==Json::Array;int rows=nested?int(j.size()):1,cols=nested?int(j.at(0).size()):int(j.size());cv::Mat m(rows,cols,CV_64F);
    for(int i=0;i<rows;++i){if(nested&&j.at(i).size()!=cols)throw std::runtime_error("Rectangular numeric array required");for(int k=0;k<cols;++k)m.at<double>(i,k)=nested?j.at(i).at(k).num():j.at(k).num();}return m;
}
Json jsonmatrix(const cv::Mat& m){Json j=Json::list();for(int i=0;i<m.rows;++i){Json row=Json::list();for(int k=0;k<m.cols;++k)row.push(m.at<double>(i,k));j.push(row);}return j;}
cv::Mat loadimage(const fs::path& path){auto image=cv::imdecode(bytes(path),cv::IMREAD_GRAYSCALE);if(image.empty())throw std::runtime_error("Cannot decode "+path.u8string());return image;}
}
