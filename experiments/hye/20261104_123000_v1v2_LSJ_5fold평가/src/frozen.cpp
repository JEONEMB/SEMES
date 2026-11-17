#include "observer_internal.hpp"
#include "control_internal.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <limits>
namespace fs=std::filesystem;
static std::vector<std::string> split(const std::string&s,char delimiter){std::stringstream in(s);std::string x;std::vector<std::string> v;while(std::getline(in,x,delimiter))v.push_back(x);return v;}
static cv::Mat matrix(const std::string&s){auto v=split(s,';');cv::Mat x(1,int(v.size()),CV_64F);for(int k=0;k<x.cols;++k)x.at<double>(k)=std::stod(v[k]);return x;}
static double originalV1(const cv::FileStorage& f,const cv::Mat& feature){auto W=alignment_detail::node_matrix(f["W"]);cv::Mat x=feature;double best=1e300,py=std::numeric_limits<double>::quiet_NaN();auto branches=f["branches"];for(auto it=branches.begin();it!=branches.end();++it){auto n=*it;std::string name=n.name(),axis=n[0];auto b=alignment_detail::node_matrix(n[1]),s=alignment_detail::node_matrix(n[2]),c=alignment_detail::node_matrix(n[3]);cv::Mat z=W*(x-b).t(),sw=W*s.t(),cw=W*c.t();double v=z.dot(sw)/sw.dot(sw);for(int k=0;k<8;++k){cv::Mat j=sw+2*v*cw,r=z-v*sw-v*v*cw;v+=j.dot(r)/j.dot(j);}if(v*(name.back()=='+'?1:-1)<0)continue;cv::Mat r=z-v*sw-v*v*cw;double cost=r.dot(r);if(cost<best){best=cost;py=axis=="Y"?v:0;}}return py;}
int main(int argc,char**argv){try{if(argc!=5)return 2;fs::path repo=fs::u8path(argv[1]),res=fs::u8path(argv[3]);std::ifstream input(argv[2]),feat(res/"features.tsv");std::ofstream out(argv[4]);out<<std::setprecision(17)<<"model\tdate\tfold\tgroup\tindex\tY\tT\tpred_Y\tpred_T\tpredict_ms\terror\n";std::string l;std::getline(input,l);std::vector<std::vector<std::string>> rows;while(std::getline(input,l))rows.push_back(split(l,'\t'));
 auto ja=repo/"experiments/hye/v2_Y_T_joint/assets";alignment_detail::JointModel j;j.load(ja/"joint_v2_model.json");auto ca=repo/"experiments/hye/v2_Y_T control/assets";legacy_control::ModelV2 c(ca/"control_v2_model.json");cv::FileStorage v1((repo/"experiments/hye/00000_final_Y_T control/assets/model.json").u8string(),cv::FileStorage::READ);std::getline(feat,l);
 while(std::getline(feat,l)){auto v=split(l,'\t');int i=std::stoi(v[0]);auto&r=rows.at(i);cv::Mat x=matrix(v[1]),p=matrix(v[2]);double s=std::stod(v[3]),d=std::stod(v[4]);auto emit=[&](const char*m,double y,double t){out<<m<<'\t'<<r[0]<<'\t'<<r[4]<<'\t'<<r[5]<<'\t'<<i<<'\t'<<r[2]<<'\t'<<r[3]<<'\t'<<y<<'\t'<<t<<"\tnan\t\n";};
 if(x.cols==204){cv::Mat obs=x.colRange(0,102);auto q=j.predict(obs);emit("joint_v2_frozen_development",q.y,q.t);}if(p.cols==12)emit("control_v1_frozen_development",originalV1(v1,p),std::numeric_limits<double>::quiet_NaN());if(std::isfinite(s)&&std::isfinite(d)){auto q=c.predict(s,d);emit("control_v2_frozen_development",q.y,q.t);}}
 return 0;}catch(const std::exception&e){std::cerr<<e.what();return 1;}}
