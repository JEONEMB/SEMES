#include "control_internal.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <set>
namespace legacy_control {
cv::Mat read_numbers(const cv::FileNode& node,int rows,int columns){
    bool vector=rows==1;if(!node.isSeq()||int(node.size())!=(vector?columns:rows))throw std::runtime_error("Invalid numeric model dimensions");
    cv::Mat result(rows,columns,CV_64F);
    for(int i=0;i<rows;++i){if(!vector&&(!node[i].isSeq()||int(node[i].size())!=columns))throw std::runtime_error("Ragged numeric model");
        for(int j=0;j<columns;++j){auto v=vector?node[j]:node[i][j];if(!v.isInt()&&!v.isReal())throw std::runtime_error("Non-numeric model coefficient");result.at<double>(i,j)=double(v);}}
    if(!cv::checkRange(result))throw std::runtime_error("Nonfinite model coefficient");return result;
}
Model::Model(const fs::path& file){
    cv::FileStorage f(file.u8string(),cv::FileStorage::READ);if(!f.isOpened())throw std::runtime_error("Cannot open legacy control model");
    whitening_=read_numbers(f["W"],12,12);zero_=f["zero_feature"].empty()?cv::Mat::zeros(1,12,CV_64F):read_numbers(f["zero_feature"],1,12);
    auto branches=f["branches"];std::set<std::string> expected{"Y1+","Y1-","Y2+","Y2-","T+","T-"};
    for(auto it=branches.begin();it!=branches.end();++it){auto n=*it;std::string name=n.name();
        if(!expected.erase(name)||!n.isSeq()||n.size()!=4)throw std::runtime_error("Expected original six-origin control branches");
        Branch b;b.name=name;b.axis=std::string(n[0]);if(b.axis!=name.substr(0,1))throw std::runtime_error("Control branch axis mismatch");
        b.origin=read_numbers(n[1],1,12);b.slope=read_numbers(n[2],1,12);b.curve=read_numbers(n[3],1,12);
        if(cv::norm(whitening_*b.slope.t())==0)throw std::runtime_error("Zero control branch response");branches_.push_back(b);
    }if(!expected.empty())throw std::runtime_error("Missing legacy control branch");
}
Choice Model::predict(const cv::Mat& feature,std::vector<Choice>* alternatives)const{
    if(feature.type()!=CV_64F||feature.rows!=1||feature.cols!=12||!cv::checkRange(feature))throw std::runtime_error("Finite P12 feature required");
    cv::Mat x=feature-zero_;Choice best;best.cost=std::numeric_limits<double>::infinity();if(alternatives)alternatives->clear();
    for(const auto& b:branches_){cv::Mat z=whitening_*(x-b.origin).t(),s=whitening_*b.slope.t(),c=whitening_*b.curve.t();double value=z.dot(s)/s.dot(s);
        for(int k=0;k<8;++k){cv::Mat j=s+2*value*c,r=z-value*s-value*value*c;double denom=j.dot(j);if(denom<=0||!std::isfinite(denom))throw std::runtime_error("Degenerate legacy curve solve");value+=j.dot(r)/denom;}
        cv::Mat residual=z-value*s-value*value*c;double cost=residual.dot(residual);if(value*(b.name.back()=='+'?1:-1)<0)cost=std::numeric_limits<double>::infinity();
        Choice q{b.axis,b.name,value,cost};if(alternatives)alternatives->push_back(q);if(cost<best.cost)best=q;
    }if(!std::isfinite(best.cost)||!std::isfinite(best.value))throw std::runtime_error("No finite compatible legacy control branch");return best;
}
}
