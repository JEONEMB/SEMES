#include "experiment.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <stdexcept>
namespace study {
void require(bool condition,const std::string& message){if(!condition)throw std::runtime_error(message);}
static cv::Mat whitening(const cv::Mat& residual,int parameters,double shrink){
 const int d=residual.cols;
 cv::Mat covariance=residual.t()*residual/double(std::max(residual.rows-parameters,1));
 std::vector<double> sd(d);for(int j=0;j<d;++j)sd[j]=std::sqrt(std::max(covariance.at<double>(j,j),1e-14));
 cv::Mat correlation(d,d,CV_64F);
 for(int i=0;i<d;++i)for(int j=0;j<d;++j)correlation.at<double>(i,j)=(1-shrink)*covariance.at<double>(i,j)/sd[i]/sd[j]+(i==j?shrink:0);
 cv::Mat values,vectors;require(cv::eigen(correlation,values,vectors),"Covariance eigensolve failed");
 cv::Mat W(d,d,CV_64F);
 for(int i=0;i<d;++i){require(values.at<double>(i)>0,"Nonpositive regularized covariance");for(int j=0;j<d;++j)W.at<double>(i,j)=vectors.at<double>(i,j)/std::sqrt(values.at<double>(i))/sd[j];}
 return W;
}
Prediction Model::predict(const cv::Mat& feature)const{
 require(feature.rows==1&&feature.cols==dimension&&feature.type()==CV_64F&&cv::checkRange(feature),"Invalid inference feature");
 if(conditional){cv::Mat value=(feature-origin)*projector;double y=value.at<double>(0);require(std::isfinite(y),"Nonfinite Y");return {y,0,0,"Y","conditional_Y"};}
 Prediction best;best.cost=std::numeric_limits<double>::infinity();
 for(const auto& b:branches){cv::Mat z=W*(feature-b.origin).t(),s=W*b.slope.t(),c=W*b.curve.t();double value=z.dot(s)/s.dot(s);
  for(int k=0;k<8;++k){cv::Mat j=s+2*value*c,r=z-value*s-value*value*c;double den=j.dot(j);require(den>0&&std::isfinite(den),"Degenerate candidate curve");value+=j.dot(r)/den;}
  cv::Mat r=z-value*s-value*value*c;double cost=r.dot(r);if(value*(b.name.back()=='+'?1:-1)<0)cost=std::numeric_limits<double>::infinity();
  if(cost<best.cost)best={b.axis=="Y"?value:0,b.axis=="T"?value:0,cost,b.axis,b.name};
 }
 require(std::isfinite(best.cost),"No finite candidate");return best;
}
Model train_y(const std::vector<Sample>& data,const std::vector<int>& ids,int feature){
 require(ids.size()>2,"Not enough Y training rows");Model model;model.conditional=true;model.shrink=.01;model.dimension=data[ids.front()].feature[feature].cols;
 cv::Mat A(int(ids.size()),2,CV_64F),X(int(ids.size()),model.dimension,CV_64F),coefficient;
 for(int i=0;i<int(ids.size());++i){const auto& s=data[ids[i]];require(s.valid[feature]&&s.t==0,"Invalid T=0 training sample");A.at<double>(i,0)=1;A.at<double>(i,1)=s.y;s.feature[feature].copyTo(X.row(i));}
 require(cv::solve(A,X,coefficient,cv::DECOMP_SVD),"Y forward fit failed");
 model.origin=coefficient.row(0).clone();cv::Mat slope=coefficient.row(1).t(),residual=X-A*coefficient;
 cv::Mat W=whitening(residual,2,model.shrink),response=W*slope;double den=response.dot(response);require(den>0,"Zero Y response");model.projector=W.t()*response/den;return model;
}
Model train_yort(const std::vector<Sample>& data,const std::vector<int>& ids,int feature){
 require(ids.size()>16,"Not enough Y-or-T training rows");Model model;model.dimension=data[ids.front()].feature[feature].cols;model.shrink=.05;
 for(int i:ids)require(data[i].valid[feature],"Training measurement failure; samples are not dropped");
 const std::vector<std::string> names{"Y1+","Y1-","Y2+","Y2-","T+","T-"};std::map<std::string,Branch> fitted;
 for(const auto& axis:{std::string("Y"),std::string("T")}){
  std::vector<int> selected;std::vector<std::string> sessions;for(const auto& name:names)if(name.substr(0,1)==axis)sessions.push_back(name);for(int i:ids)if(data[i].axis==axis)selected.push_back(i);
  cv::Mat A=cv::Mat::zeros(int(selected.size()),int(sessions.size())+2,CV_64F),X(int(selected.size()),model.dimension,CV_64F),coefficient;
  for(int r=0;r<int(selected.size());++r){const auto& s=data[selected[r]];A.at<double>(r,0)=s.value;A.at<double>(r,1)=s.value*s.value;auto pos=std::find(sessions.begin(),sessions.end(),s.session);require(pos!=sessions.end(),"Unknown training session");A.at<double>(r,2+int(pos-sessions.begin()))=1;s.feature[feature].copyTo(X.row(r));}
  require(cv::solve(A,X,coefficient,cv::DECOMP_SVD),"Candidate fit failed");
  for(int j=0;j<int(sessions.size());++j)fitted[sessions[j]]={axis,sessions[j],coefficient.row(j+2).clone(),coefficient.row(0).clone(),coefficient.row(1).clone()};
 }
 for(const auto& name:names)model.branches.push_back(fitted.at(name));
 cv::Mat residual(int(ids.size()),model.dimension,CV_64F);
 for(int r=0;r<int(ids.size());++r){const auto& s=data[ids[r]];const auto& b=fitted.at(s.session);cv::Mat row=s.feature[feature]-b.origin-s.value*b.slope-s.value*s.value*b.curve;row.copyTo(residual.row(r));}
 model.W=whitening(residual,16,model.shrink);return model;
}
void Model::save(const std::filesystem::path& path)const{
 cv::FileStorage f(path.u8string(),cv::FileStorage::WRITE);require(f.isOpened(),"Cannot write model");
 f<<"conditional"<<int(conditional)<<"dimension"<<dimension<<"shrink"<<shrink;
 if(conditional)f<<"origin"<<origin<<"projector"<<projector;
 else {f<<"W"<<W<<"branches"<<"[";for(const auto& b:branches)f<<"{"<<"axis"<<b.axis<<"name"<<b.name<<"origin"<<b.origin<<"slope"<<b.slope<<"curve"<<b.curve<<"}";f<<"]";}
}
Model Model::load(const std::filesystem::path& path){
 cv::FileStorage f(path.u8string(),cv::FileStorage::READ);require(f.isOpened(),"Cannot load model");Model m;m.conditional=int(f["conditional"])!=0;m.dimension=int(f["dimension"]);m.shrink=double(f["shrink"]);
 if(m.conditional){f["origin"]>>m.origin;f["projector"]>>m.projector;require(m.origin.rows==1&&m.origin.cols==m.dimension&&m.projector.rows==m.dimension&&m.projector.cols==1,"Invalid Y model shape");}
 else {f["W"]>>m.W;for(auto it=f["branches"].begin();it!=f["branches"].end();++it){auto n=*it;Branch b;b.axis=std::string(n["axis"]);b.name=std::string(n["name"]);n["origin"]>>b.origin;n["slope"]>>b.slope;n["curve"]>>b.curve;m.branches.push_back(b);}require(m.branches.size()==6&&m.W.rows==m.dimension&&m.W.cols==m.dimension,"Invalid candidate model shape");}return m;
}
}
