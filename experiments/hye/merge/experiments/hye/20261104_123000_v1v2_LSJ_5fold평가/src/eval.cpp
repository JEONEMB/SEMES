#include <limits>
#include <cmath>
#include "observer_internal.hpp"
#include "control_internal.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <map>
#include <algorithm>
namespace fs=std::filesystem;
using Clock=std::chrono::steady_clock;
static double ms(Clock::time_point a){return std::chrono::duration<double,std::milli>(Clock::now()-a).count();}
static std::vector<std::string> split(std::string s){if(!s.empty() && s.back()=='\r')s.pop_back();std::vector<std::string> v;std::stringstream ss(s);std::string x;while(std::getline(ss,x,'\t'))v.push_back(x);return v;}
struct Row{std::string date,path,group,errorJ,errorC;int fold;double y,t,s=std::numeric_limits<double>::quiet_NaN(),d=std::numeric_limits<double>::quiet_NaN(),jms=std::numeric_limits<double>::quiet_NaN(),cms=std::numeric_limits<double>::quiet_NaN(),decode=std::numeric_limits<double>::quiet_NaN();cv::Mat j,c;};
static cv::Mat fmap(const cv::Mat& x){cv::Mat z(1,204,CV_64F);x.copyTo(z.colRange(0,102));double h=0;for(int i=0;i<54;i+=2)h+=x.at<double>(i);h/=2700.;for(int i=0;i<102;++i)z.at<double>(i+102)=x.at<double>(i)*h;return z;}
struct Linear{cv::Mat origin,P;cv::Mat predict(const cv::Mat& x)const{return (x-origin)*P;}};
// Degree-1 forward GLS with residual correlation shrink=.01.
static Linear fitGLS(const cv::Mat& X,const cv::Mat& Y){
 cv::Mat A=cv::Mat::ones(X.rows,3,CV_64F);Y.copyTo(A.colRange(1,3));cv::Mat co;cv::solve(A,X,co,cv::DECOMP_SVD);
 cv::Mat R=X-A*co,sd(1,X.cols,CV_64F),Z=R.clone();for(int k=0;k<X.cols;++k){sd.at<double>(k)=std::max(std::sqrt(R.col(k).dot(R.col(k))/R.rows),1e-6);Z.col(k)/=sd.at<double>(k);}
 cv::Mat cov=.99*(Z.t()*Z/double(Z.rows))+.01*cv::Mat::eye(X.cols,X.cols,CV_64F),B=co.rowRange(1,3).clone();
 for(int k=0;k<X.cols;++k)B.col(k)/=sd.at<double>(k);
 cv::Mat Q;cv::solve(cov,B.t(),Q,cv::DECOMP_SVD);cv::Mat gram=B*Q,inv;cv::invert(gram,inv,cv::DECOMP_SVD);cv::Mat P=Q*inv;
 for(int k=0;k<X.cols;++k)P.row(k)/=sd.at<double>(k);
 return {co.row(0).clone(),P};
}
static cv::Mat inverseFit(const cv::Mat& X,const cv::Mat& Y){cv::Mat A=cv::Mat::ones(X.rows,X.cols+1,CV_64F);X.copyTo(A.colRange(1,A.cols));cv::Mat W;cv::solve(A,Y,W,cv::DECOMP_SVD);return W;}
static cv::Mat pred(const cv::Mat& W,const cv::Mat& x){cv::Mat a=cv::Mat::ones(1,x.cols+1,CV_64F);x.copyTo(a.colRange(1,a.cols));return a*W;}
struct Curve{cv::Mat co,W;std::vector<std::string> sessions;};
static std::string session(double y){return std::string(std::llabs(std::llround(y/.05))%2?"Y2":"Y1")+(y>0?"+":"-");}
// T=0-only adaptation of v1: four Y branches, shared quadratic slope/curve.
static Curve fitCurve(const std::vector<Row*>& tr){
 Curve m; m.sessions={"Y1+","Y1-","Y2+","Y2-"};cv::Mat A=cv::Mat::zeros(int(tr.size()),6,CV_64F),X(int(tr.size()),12,CV_64F);
 for(int i=0;i<int(tr.size());++i){double y=tr[i]->y;A.at<double>(i,0)=y;A.at<double>(i,1)=y*y;auto it=std::find(m.sessions.begin(),m.sessions.end(),session(y));A.at<double>(i,2+int(it-m.sessions.begin()))=1;tr[i]->c.copyTo(X.row(i));}
 cv::solve(A,X,m.co,cv::DECOMP_SVD);cv::Mat R=X-A*m.co,S=R.t()*R/double(std::max(1,R.rows-6)),D=cv::Mat::eye(12,12,CV_64F);
 for(int k=0;k<12;++k)D.at<double>(k,k)=1/std::sqrt(std::max(S.at<double>(k,k),1e-14));
 cv::Mat corr=.95*(D*S*D)+.05*cv::Mat::eye(12,12,CV_64F),ci;cv::invert(corr,ci,cv::DECOMP_SVD);m.W=D*ci*D;return m;
}
static double curvePred(const Curve& m,const cv::Mat& x){double best=1e300,ans=std::numeric_limits<double>::quiet_NaN();for(int b=0;b<4;++b){cv::Mat z=x-m.co.row(b+2),s=m.co.row(0),c=m.co.row(1);cv::Mat sw=s*m.W;double v=z.dot(sw)/s.dot(sw);for(int k=0;k<8;++k){cv::Mat j=s+2*v*c,r=z-v*s-v*v*c,jw=j*m.W;v+=j.dot(r*m.W)/std::max(j.dot(jw),1e-20);}if(v*(m.sessions[b].back()=='+'?1:-1)<0)continue;cv::Mat r=z-v*s-v*v*c;double cost=r.dot(r*m.W);if(cost<best){best=cost;ans=v;}}return ans;}
static cv::Mat arr(const cv::FileNode& n){return alignment_detail::node_matrix(n);}
struct FrozenJ{cv::Mat o,p,i;double sy,st;cv::Mat predict(const cv::Mat& x){cv::Mat q=(x-o)*p*i;q.at<double>(0)*=sy;q.at<double>(1)*=st;return q;}};
int main(int argc,char**argv){try{
 if(argc!=4)throw std::runtime_error("eval REPO INPUT_TSV OUTPUT");fs::path root=fs::u8path(argv[1]),input=fs::u8path(argv[2]),out=fs::u8path(argv[3]);fs::create_directories(out);cv::setNumThreads(1);
 auto jassets=root/std::filesystem::u8path(u8"experiments/hye/20261104_1435_v2_joint_강등/assets"),cassets=root/std::filesystem::u8path(u8"experiments/hye/20261104_1435_v2_control_강등/assets");
 auto init=Clock::now();alignment_detail::Observer obs(jassets);legacy_control::Tracker tracker(cassets);auto bx=obs.block_centers();double initMs=ms(init);
 cv::FileStorage frozen((root/"experiments/hye/20261028_1258_final_Y_T_joint/assets/both_model.json").u8string(),cv::FileStorage::READ);auto n=frozen["model"];FrozenJ fj{arr(n["origin"]),arr(n["projector"]),arr(n["initial"]),double(n["scale"][0]),double(n["scale"][1])};
 std::ifstream in(input);std::string line;std::getline(in,line);std::vector<Row> rows;while(std::getline(in,line)){auto v=split(line);if(v.size()!=6)throw std::runtime_error("Bad TSV");rows.push_back({v[0],v[1],v[5],"","",std::stoi(v[4]),std::stod(v[2]),std::stod(v[3])});}
 std::ofstream timings(out/"measurements.tsv"),features(out/"features.tsv");timings<<"index\tdate\tfold\tY\tT\tdecode_ms\tjoint_observe_ms\tcontrol_track_ms\tjoint_error\tcontrol_error\n";features<<std::setprecision(17);features<<"index\tjoint204\tcontrol12\ts\td\n";
 for(int i=0;i<int(rows.size());++i){auto&r=rows[i];cv::Mat image;try{auto t=Clock::now();image=alignment_detail::read_gray(fs::u8path(r.path));r.decode=ms(t);}catch(const std::exception&e){r.errorJ=r.errorC=e.what();}
 if(!image.empty()){try{auto t=Clock::now();auto o=obs.measure(image);auto f=alignment_detail::v2_features(o.values,bx);r.jms=ms(t);r.j=fmap(o.values);r.d=f[1];}catch(const std::exception&e){r.errorJ=e.what();}try{auto t=Clock::now();auto tracks=tracker.measure(image);r.c=tracker.features(tracks);r.s=tracker.mean_shift(tracks);r.cms=ms(t);}catch(const std::exception&e){r.errorC=e.what();}}
 timings<<std::setprecision(17)<<i<<'\t'<<r.date<<'\t'<<r.fold<<'\t'<<r.y<<'\t'<<r.t<<'\t'<<r.decode<<'\t'<<r.jms<<'\t'<<r.cms<<'\t'<<r.errorJ<<'\t'<<r.errorC<<'\n';timings.flush();features<<i<<'\t';for(int k=0;k<r.j.cols;++k)features<<r.j.at<double>(k)<<';';features<<'\t';for(int k=0;k<r.c.cols;++k)features<<r.c.at<double>(k)<<';';features<<'\t'<<r.s<<'\t'<<r.d<<'\n';features.flush();if(i%25==0)std::cout<<"measured "<<i+1<<"/"<<rows.size()<<std::endl;}
 std::ofstream pr(out/"predictions.tsv");pr<<std::setprecision(17)<<"model\tdate\tfold\tgroup\tindex\tY\tT\tpred_Y\tpred_T\tpredict_ms\terror\n";
 auto emit=[&](const std::string&m,int idx,double y,double t,double time,std::string error=""){auto&r=rows[idx];pr<<m<<'\t'<<r.date<<'\t'<<r.fold<<'\t'<<r.group<<'\t'<<idx<<'\t'<<r.y<<'\t'<<r.t<<'\t'<<y<<'\t'<<t<<'\t'<<time<<'\t'<<error<<'\n';};
 for(int f=1;f<=5;++f){cv::Mat X,Y,V;std::vector<Row*> ctl;for(auto&r:rows)if(r.fold!=f){if(!r.j.empty()){X.push_back(r.j);Y.push_back(cv::Mat((cv::Mat_<double>(1,2)<<r.y,r.t)));double s=0;for(int k=0;k<54;k+=2)s+=r.j.at<double>(k);V.push_back(cv::Mat((cv::Mat_<double>(1,2)<<s/27,r.d)));}if(r.t==0&&!r.c.empty())ctl.push_back(&r);}
 auto t=Clock::now();auto g=fitGLS(X,Y);double gms=ms(t);t=Clock::now();auto w=inverseFit(V,Y);double wms=ms(t);t=Clock::now();auto c=fitCurve(ctl);double cms=ms(t);cv::Mat CX,CY;for(auto*r:ctl){CX.push_back(cv::Mat((cv::Mat_<double>(1,1)<<r->s)));CY.push_back(cv::Mat((cv::Mat_<double>(1,1)<<r->y)));}auto cy=inverseFit(CX,CY);
 cv::FileStorage save((out/("fold_"+std::to_string(f)+".yml")).u8string(),cv::FileStorage::WRITE);save<<"joint_v1_origin"<<g.origin<<"joint_v1_projector"<<g.P<<"joint_v2_W"<<w<<"control_v1_Y_only_coef"<<c.co<<"control_v1_precision"<<c.W<<"control_v2_T0_y0"<<cy<<"joint_train_n"<<X.rows<<"control_T0_train_n"<<int(ctl.size())<<"joint_v1_train_ms"<<gms<<"joint_v2_train_ms"<<wms<<"control_v1_train_ms"<<cms;
 for(int i=0;i<int(rows.size());++i)if(rows[i].fold==f){auto&r=rows[i];if(!r.j.empty()){t=Clock::now();cv::Mat q=g.predict(r.j);emit("joint_v1_retrained_GLS",i,q.at<double>(0),q.at<double>(1),ms(t));double s=0;for(int k=0;k<54;k+=2)s+=r.j.at<double>(k);cv::Mat v=(cv::Mat_<double>(1,2)<<s/27,r.d);t=Clock::now();q=pred(w,v);emit("joint_v2_retrained",i,q.at<double>(0),q.at<double>(1),ms(t));q=fj.predict(r.j);emit("joint_v1_frozen_development",i,q.at<double>(0),q.at<double>(1),std::numeric_limits<double>::quiet_NaN());}else for(auto m:{"joint_v1_retrained_GLS","joint_v2_retrained","joint_v1_frozen_development"})emit(m,i,std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::quiet_NaN(),r.errorJ);
 if(!r.c.empty()){t=Clock::now();double py=curvePred(c,r.c);emit("control_v1_T0_four_Y_branches",i,py,std::numeric_limits<double>::quiet_NaN(),ms(t));t=Clock::now();cv::Mat q=pred(cy,(cv::Mat_<double>(1,1)<<r.s));emit("control_v2_T0_no_T_correction",i,q.at<double>(0),std::numeric_limits<double>::quiet_NaN(),ms(t));}else for(auto m:{"control_v1_T0_four_Y_branches","control_v2_T0_no_T_correction"})emit(m,i,std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::quiet_NaN(),r.errorC);}
 std::cout<<"trained fold "<<f<<std::endl;}
 std::ofstream info(out/"environment.txt");info<<"OpenCV "<<CV_VERSION<<"\nMSVC "<<_MSC_VER<<"\nOpenCV threads 1\nInitialization joint+tracker ms "<<initMs<<"\nMeasurements on grayscale decoded BMP; decode separate; no physical motion\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
