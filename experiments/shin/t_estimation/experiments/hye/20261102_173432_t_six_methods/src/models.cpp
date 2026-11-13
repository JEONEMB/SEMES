#include "six.hpp"
#include "t_model.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <stdexcept>
namespace six {
static cv::Mat selected(const cv::Mat& x,const Config& c){
 if(c.method==0){t_only::Features f{};for(int j=0;j<22;j++)f[j]=x.at<double>(j);auto v=t_only::representation(f,c.representation);return cv::Mat(1,(int)v.size(),CV_64F,v.data()).clone();}
 if(c.method==2){if(c.representation==0)return x.colRange(64,96).clone();cv::Mat result;cv::hconcat(x.colRange(64,72),x.colRange(96,104),result);return result;}
 if(c.method==1||c.method==4||(c.representation==1&&(c.method==3||c.method==6)))return x.colRange(0,72).clone();
 return x.colRange(0,10).clone();
}
static cv::Mat normalized(const Model& m,const cv::Mat& x){cv::Mat z=selected(x,m.config)-m.mean;cv::divide(z,m.scale,z);return z;}
static cv::Mat basis(const Config& c,cv::Vec2d v){double y=v[0],t=v[1],a=t,b=t*t;if(c.method==2){double rad=t*CV_PI/180.;a=sin(rad)/sin(CV_PI/180.);b=(1-cos(rad))/(1-cos(CV_PI/180.));}cv::Mat p(1,c.order==1?3:6,CV_64F);p.at<double>(0)=1;p.at<double>(1)=y;p.at<double>(2)=a;if(c.order!=1){p.at<double>(3)=b;p.at<double>(4)=y*a;p.at<double>(5)=y*y;}return p;}
static cv::Mat regress(const cv::Mat& x,const cv::Mat& y,double alpha,const cv::Mat& weight){
 int n=x.rows,p=x.cols;cv::Mat a(n+p,p,CV_64F,cv::Scalar(0)),b(n+p,y.cols,CV_64F,cv::Scalar(0));
 for(int i=0;i<n;i++){double w=weight.empty()?1.:sqrt(weight.at<double>(i));x.row(i).copyTo(a.row(i));a.row(i)*=w;y.row(i).copyTo(b.row(i));b.row(i)*=w;}
 for(int j=1;j<p;j++)a.at<double>(n+j,j)=sqrt(alpha);cv::Mat coef;bool ok=false;if(alpha>0){cv::Mat gram=a.t()*a,rhs=a.t()*b;ok=cv::solve(gram,rhs,coef,cv::DECOMP_CHOLESKY);}if(!ok)ok=cv::solve(a,b,coef,cv::DECOMP_SVD);if(!ok||!cv::checkRange(coef))throw std::runtime_error("Regression failed");return coef;
}
static cv::Mat with_intercept(const cv::Mat& z){cv::Mat x(z.rows,z.cols+1,CV_64F,cv::Scalar(1));z.copyTo(x.colRange(1,x.cols));return x;}
static cv::Mat rbf(const Model& m,const cv::Mat& z){cv::Mat x(1,1+z.cols+m.centers.rows,CV_64F,cv::Scalar(1));z.copyTo(x.colRange(1,1+z.cols));for(int k=0;k<m.centers.rows;k++){double d=cv::norm(z,m.centers.row(k),cv::NORM_L2SQR)/z.cols;x.at<double>(1+z.cols+k)=exp(-m.config.gamma*d);}return x;}
std::vector<Config> candidates(int method){std::vector<Config> out;Config c;c.method=method;
 if(method==0){for(int k=0;k<6;k++)for(double a:{0.,.001,.01,.1,1.,10.,100.}){c.representation=k;c.alpha=a;out.push_back(c);}}
 if(method==1||method==2){for(int order:{1,2})for(double s:{.01,.1,.5}){c.order=order;c.shrink=s;c.alpha=.001;out.push_back(c);} if(method==1){for(double a:{.01,1.,100.})for(double s:{.01,.1,.5}){c.representation=1;c.order=1;c.alpha=a;c.shrink=s;out.push_back(c);}}else{for(int order:{1,2})for(double s:{.01,.1,.5}){c.representation=1;c.order=order;c.alpha=.001;c.shrink=s;out.push_back(c);}}}
 if(method==3){for(int rep:{0,1})for(double g:{.2,1.,5.})for(double a:{.01,1.}){c.representation=rep;c.gamma=g;c.alpha=a;out.push_back(c);}}
 if(method==4){for(int order:{1,2})for(double s:{.01,.1,.5}){c.order=order;c.alpha=.001;c.shrink=s;out.push_back(c);}}
 if(method==5){for(double a:{.001,.1,1.})for(double s:{10.,30.,100.}){c.alpha=a;c.shrink=s;out.push_back(c);}}
 if(method==6){for(int rep:{0,1})for(double a:{.01,.1,1.})for(double eta:{.1,.5}){c.representation=rep;c.alpha=a;c.gamma=eta;out.push_back(c);}}
 return out;
}
Model fit(const std::vector<Row>& rows,const std::vector<int>& ids,Config config){
 if(ids.size()<8)throw std::runtime_error("Insufficient training rows");Model m;m.config=config;int n=(int)ids.size(),p=selected(rows[ids[0]].x,config).cols;cv::Mat raw(n,p,CV_64F),target(n,1,CV_64F);m.min_y=m.max_y=rows[ids[0]].y;m.min_t=m.max_t=rows[ids[0]].t;
 for(int i=0;i<n;i++){auto& r=rows[ids[i]];selected(r.x,config).copyTo(raw.row(i));target.at<double>(i)=r.t/tick_deg;m.min_y=std::min(m.min_y,r.y);m.max_y=std::max(m.max_y,r.y);m.min_t=std::min(m.min_t,r.t);m.max_t=std::max(m.max_t,r.t);}
 cv::reduce(raw,m.mean,0,cv::REDUCE_AVG);m.scale=cv::Mat(1,p,CV_64F);for(int j=0;j<p;j++){double s=0;for(int i=0;i<n;i++)s+=pow(raw.at<double>(i,j)-m.mean.at<double>(j),2);m.scale.at<double>(j)=std::max(1e-8,sqrt(s/n));for(int i=0;i<n;i++)raw.at<double>(i,j)=(raw.at<double>(i,j)-m.mean.at<double>(j))/m.scale.at<double>(j);}
 if(config.method==1||config.method==2||config.method==4){
  cv::Mat phi(n,config.order==1?3:6,CV_64F);for(int i=0;i<n;i++)basis(config,{rows[ids[i]].y,rows[ids[i]].t}).copyTo(phi.row(i));m.coef=regress(phi,raw,config.alpha,{});
  std::map<std::pair<int,int>,std::vector<int>> groups;for(int i=0;i<n;i++)groups[{(int)std::lround(rows[ids[i]].y*100),(int)std::lround(rows[ids[i]].t*100)}].push_back(i);
  cv::Mat cov=cv::Mat::zeros(p,p,CV_64F);int degrees=0;
  for(auto& pair:groups){auto& g=pair.second;if(g.size()<2)continue;cv::Mat mean=cv::Mat::zeros(1,p,CV_64F);for(int i:g)mean+=raw.row(i)/(double)g.size();for(int i:g){cv::Mat e=raw.row(i)-mean;cov+=e.t()*e;}degrees+=(int)g.size()-1;}
  if(degrees>5)cov/=degrees;else{cv::Mat residual=raw-phi*m.coef;cov=residual.t()*residual/std::max(1,n-phi.cols);}
  double trace=cv::trace(cov)[0]/p;for(int j=0;j<p;j++)for(int k=0;k<p;k++){if(j!=k)cov.at<double>(j,k)*=1-config.shrink;else cov.at<double>(j,k)+=std::max(1e-8,trace*1e-4);}
  cv::Mat values,vectors;cv::eigen(cov,values,vectors);m.precision=vectors.t();for(int j=0;j<p;j++)m.precision.col(j)*=1/sqrt(std::max(std::max(1e-8,trace*1e-4),values.at<double>(j)));m.weights=m.coef*m.precision;if(config.method==1&&config.representation==1){cv::Mat whitened=raw*m.precision;m.coef=regress(with_intercept(whitened),target,config.alpha,{});}return m;
 }
 if(config.method==3){
  int count=std::min(32,n);m.centers=cv::Mat(count,p,CV_64F);raw.row(0).copyTo(m.centers.row(0));std::vector<double> distance(n,1e300);
  for(int k=1;k<count;k++){int next=0;for(int i=0;i<n;i++){distance[i]=std::min(distance[i],cv::norm(raw.row(i),m.centers.row(k-1),cv::NORM_L2SQR));if(distance[i]>distance[next])next=i;}raw.row(next).copyTo(m.centers.row(k));}
  cv::Mat x(n,1+p+count,CV_64F);for(int i=0;i<n;i++)rbf(m,raw.row(i)).copyTo(x.row(i));m.coef=regress(x,target,config.alpha,{});return m;
 }
 cv::Mat design=with_intercept(raw);m.coef=regress(design,target,config.alpha,{});
 if(config.method==6){
  std::map<std::pair<int,int>,int> lookup;std::vector<int> group(n),counts;
  for(int i=0;i<n;i++){auto& r=rows[ids[i]];std::pair<int,int> key{(int)floor((r.y+2)/.5),(int)floor((r.t+1)/.25)};auto it=lookup.find(key);int g;if(it==lookup.end()){g=(int)lookup.size();lookup[key]=g;counts.push_back(0);}else g=it->second;group[i]=g;counts[g]++;}
  int ng=(int)counts.size();std::vector<double> q(ng,1./ng);
  for(int iteration=0;iteration<15;iteration++){
   cv::Mat w(n,1,CV_64F);for(int i=0;i<n;i++)w.at<double>(i)=n*q[group[i]]/counts[group[i]];m.coef=regress(design,target,config.alpha,w);cv::Mat residual=design*m.coef-target;std::vector<double> loss(ng,0);for(int i=0;i<n;i++)loss[group[i]]+=pow(residual.at<double>(i),2)/counts[group[i]];double mean=std::accumulate(loss.begin(),loss.end(),0.)/ng,sum=0;
   for(int g=0;g<ng;g++){q[g]*=exp(std::min(4.,config.gamma*loss[g]/std::max(.01,mean)));sum+=q[g];}for(int g=0;g<ng;g++)q[g]=.85*q[g]/sum+.15/ng;
  }
 }
 if(config.method==5){
  struct Stats{double lo=1e30,hi=-1e30,residual=0;int n=0;};std::map<int,Stats> stats;cv::Mat residual=target-design*m.coef;
  std::vector<int> timeOrder(n);std::iota(timeOrder.begin(),timeOrder.end(),0);std::sort(timeOrder.begin(),timeOrder.end(),[&](int a,int b){return rows[ids[a]].time<rows[ids[b]].time;});int group=0;double previous=0;for(int i:timeOrder){auto& r=rows[ids[i]];if(previous!=0&&r.time-previous>120)group++;previous=r.time;auto& a=stats[group];a.lo=std::min(a.lo,r.time);a.hi=std::max(a.hi,r.time);a.residual+=residual.at<double>(i);a.n++;}
  for(auto& s:stats){m.session_start.push_back(s.second.lo);m.session_end.push_back(s.second.hi);m.session_offset.push_back(s.second.residual/(s.second.n+config.shrink));}
 }return m;
}
cv::Mat forward(const Model& m,cv::Vec2d state){cv::Mat z=basis(m.config,state)*m.coef;cv::multiply(z,m.scale,z);z+=m.mean;return z;}
Prediction predict(const Model& m,const cv::Mat& x,double time){
 Prediction result;if(!m.members.empty()){for(int k=0;k<(int)m.members.size();k++){double w=m.weights.at<double>(k);if(w==0)continue;auto p=predict(*m.members[k],x,time);result.t+=w*p.t;result.known_session=result.known_session||p.known_session;}return result;} cv::Mat z=normalized(m,x);
 if(m.config.method==1&&m.config.representation==1){cv::Mat y=with_intercept(z*m.precision)*m.coef;result.t=y.at<double>(0)*tick_deg;}
 else if(m.config.method==1||m.config.method==2||m.config.method==4){
  cv::Mat target=z*m.precision;cv::Mat a=m.weights.rowRange(1,3).t(),rhs=target.t()-m.weights.row(0).t(),initial;cv::solve(a,rhs,initial,cv::DECOMP_SVD);cv::Vec2d state(initial.at<double>(0),initial.at<double>(1));
  auto cost=[&](cv::Vec2d v){cv::Mat e=basis(m.config,v)*m.weights-target;return cv::norm(e,cv::NORM_L2SQR);};double current=cost(state);double damping=1e-5;
  for(int iter=0;iter<35;iter++){
   cv::Mat pred=basis(m.config,state)*m.weights,residual=pred-target,j(2,pred.cols,CV_64F);
   for(int k=0;k<2;k++){auto plus=state,minus=state;plus[k]+=1e-4;minus[k]-=1e-4;cv::Mat d=(basis(m.config,plus)-basis(m.config,minus))*m.weights/(2e-4);d.copyTo(j.row(k));}
   cv::Mat h=j*j.t(),g=j*residual.t(),step;for(int k=0;k<2;k++)h.at<double>(k,k)+=damping*std::max(1.,h.at<double>(k,k));if(!cv::solve(h,-g,step,cv::DECOMP_SVD))break;cv::Vec2d delta(step.at<double>(0),step.at<double>(1));double factor=std::max(1.,std::max(std::abs(delta[0])/.5,std::abs(delta[1])/.25));delta/=factor;auto trial=state+delta;double next=cost(trial);if(next<current){state=trial;current=next;damping=std::max(1e-8,damping/3);if(cv::norm(delta)<1e-7)break;}else{damping*=10;if(damping>1e9)break;}
  }result.latent=state;result.t=state[1];result.cost=current/z.cols;
 }else{cv::Mat y=(m.config.method==3?rbf(m,z):with_intercept(z))*m.coef;result.t=y.at<double>(0)*tick_deg;if(m.config.method==5&&time!=0){double best=1e300;int chosen=-1;for(int g=0;g<(int)m.session_start.size();g++){double distance=std::max(0.,std::max(m.session_start[g]-time,time-m.session_end[g]));if(distance<best){best=distance;chosen=g;}}if(chosen>=0&&best<=120){result.t+=m.session_offset[chosen]*tick_deg;result.known_session=true;}}}
 if(!std::isfinite(result.t))throw std::runtime_error("Nonfinite T inference");return result;
}
void save_model(const Model& m,const std::filesystem::path& path){
 if(!m.members.empty()){
  auto directory=path.parent_path()/(path.stem().u8string()+"_members");std::filesystem::create_directories(directory);std::vector<std::string> files;
  for(int k=0;k<(int)m.members.size();k++){auto member=directory/("model"+std::to_string(k)+".json");save_model(*m.members[k],member);files.push_back(std::filesystem::relative(member,path.parent_path()).generic_u8string());}
  cv::FileStorage out(path.u8string(),cv::FileStorage::WRITE|cv::FileStorage::FORMAT_JSON);out<<"schema"<<"t_six_ensemble.v1"<<"method"<<m.config.method<<"tick_deg"<<tick_deg<<"weights"<<m.weights<<"models"<<files<<"min_t"<<m.min_t<<"max_t"<<m.max_t;return;
 }
 cv::FileStorage f(path.u8string(),cv::FileStorage::WRITE|cv::FileStorage::FORMAT_JSON);if(!f.isOpened())throw std::runtime_error("Cannot save model");f<<"schema"<<"t_six.v1"<<"T_unit"<<"degree"<<"tick_deg"<<tick_deg<<"method"<<m.config.method<<"representation"<<m.config.representation<<"order"<<m.config.order<<"alpha"<<m.config.alpha<<"shrink"<<m.config.shrink<<"gamma"<<m.config.gamma<<"mean"<<m.mean<<"scale"<<m.scale<<"coef"<<m.coef<<"whitener"<<m.precision<<"centers"<<m.centers<<"weighted_forward"<<m.weights<<"min_y"<<m.min_y<<"max_y"<<m.max_y<<"min_t"<<m.min_t<<"max_t"<<m.max_t<<"session_start"<<m.session_start<<"session_end"<<m.session_end<<"session_offset"<<m.session_offset;}
Model load_model(const std::filesystem::path& path){cv::FileStorage f(path.u8string(),cv::FileStorage::READ);if(!f.isOpened())throw std::runtime_error("Missing model");
 if((std::string)f["schema"]=="t_six_ensemble.v1"){
  Model m;m.config.method=(int)f["method"];f["weights"]>>m.weights;std::vector<std::string> files;f["models"]>>files;
  if(m.weights.type()!=CV_64F||m.weights.total()!=files.size()||!cv::checkRange(m.weights)||std::abs(cv::sum(m.weights)[0]-1)>1e-9)throw std::runtime_error("Invalid ensemble weights");
  for(int k=0;k<(int)files.size();k++){if(m.weights.at<double>(k)<0)throw std::runtime_error("Negative ensemble weight");m.members.push_back(std::make_shared<Model>(load_model(path.parent_path()/std::filesystem::u8path(files[k]))));}m.min_t=(double)f["min_t"];m.max_t=(double)f["max_t"];return m;
 }
 if((std::string)f["schema"]!="t_six.v1")throw std::runtime_error("Incompatible six-method model");Model m;m.config={(int)f["method"],(int)f["representation"],(int)f["order"],(double)f["alpha"],(double)f["shrink"],(double)f["gamma"]};f["mean"]>>m.mean;f["scale"]>>m.scale;f["coef"]>>m.coef;f["whitener"]>>m.precision;f["centers"]>>m.centers;f["weighted_forward"]>>m.weights;m.min_y=(double)f["min_y"];m.max_y=(double)f["max_y"];m.min_t=(double)f["min_t"];m.max_t=(double)f["max_t"];f["session_start"]>>m.session_start;f["session_end"]>>m.session_end;f["session_offset"]>>m.session_offset;if(m.mean.type()!=CV_64F||m.scale.type()!=CV_64F||m.coef.type()!=CV_64F||!cv::checkRange(m.mean)||!cv::checkRange(m.scale)||!cv::checkRange(m.coef))throw std::runtime_error("Invalid model matrices");
 if(m.config.method<0||m.config.method>6||m.config.order<1||m.config.order>2)throw std::runtime_error("Invalid model method/order");cv::Mat dummy=cv::Mat::zeros(1,dimensions,CV_64F);int p=selected(dummy,m.config).cols;
 if(m.mean.rows!=1||m.mean.cols!=p||m.scale.rows!=1||m.scale.cols!=p)throw std::runtime_error("Invalid model normalization dimensions");for(int j=0;j<p;j++)if(m.scale.at<double>(j)<=0)throw std::runtime_error("Nonpositive feature scale");
 bool forwardModel=(m.config.method==1&&m.config.representation==0)||m.config.method==2||m.config.method==4;int coefficientRows=forwardModel?(m.config.order==1?3:6):1+p;
 if(m.config.method==3){if(m.centers.type()!=CV_64F||m.centers.cols!=p||!cv::checkRange(m.centers))throw std::runtime_error("Invalid RBF centers");coefficientRows+=m.centers.rows;}
 if(m.coef.rows!=coefficientRows||m.coef.cols!=(forwardModel?p:1))throw std::runtime_error("Invalid coefficient dimensions");
 if(forwardModel||(m.config.method==1&&m.config.representation==1)){if(m.precision.type()!=CV_64F||m.precision.rows!=p||m.precision.cols!=p||!cv::checkRange(m.precision))throw std::runtime_error("Invalid covariance whitener");}
 if(forwardModel&&(m.weights.type()!=CV_64F||m.weights.rows!=coefficientRows||m.weights.cols!=p||!cv::checkRange(m.weights)))throw std::runtime_error("Invalid whitened forward map");
 if(m.session_start.size()!=m.session_end.size()||m.session_start.size()!=m.session_offset.size())throw std::runtime_error("Invalid session table");
 for(int g=0;g<(int)m.session_start.size();g++)if(!std::isfinite(m.session_start[g])||!std::isfinite(m.session_end[g])||!std::isfinite(m.session_offset[g])||m.session_start[g]>m.session_end[g])throw std::runtime_error("Invalid session interval");
 if(!std::isfinite(m.min_t)||!std::isfinite(m.max_t)||m.min_t>m.max_t)throw std::runtime_error("Invalid T calibration range");return m;}
}
