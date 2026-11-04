#include <map>
#include <limits>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cmath>
namespace fs=std::filesystem;
static std::vector<std::string> split(std::string s,char d){if(!s.empty()&&s.back()=='\r')s.pop_back();std::stringstream ss(s);std::vector<std::string> out;std::string v;while(std::getline(ss,v,d))out.push_back(v);return out;}
static cv::Mat numbers(const std::string&s){auto v=split(s,';');cv::Mat x(1,int(v.size()),CV_64F);for(int k=0;k<x.cols;++k)x.at<double>(k)=std::stod(v[k]);return x;}
struct Row{int id,fold;std::string date,group;double y,t;cv::Mat j,c;};
static cv::Mat basis(double y,double t,int degree){cv::Mat b=(cv::Mat_<double>(1,6)<<1.,y,t,y*y,y*t,t*t);return degree==1?b.colRange(0,3).clone():b;}
static cv::Mat fmap(const cv::Mat& x){cv::Mat z(1,204,CV_64F);x.copyTo(z.colRange(0,102));double h=0;for(int i=0;i<54;i+=2)h+=x.at<double>(i);h/=2700.;for(int i=0;i<102;++i)z.at<double>(i+102)=x.at<double>(i)*h;return z;}
struct Controller{
 cv::Mat jo,jp,co,precision;
 explicit Controller(const fs::path& path){cv::FileStorage f(path.u8string(),cv::FileStorage::READ);if(!f.isOpened())throw std::runtime_error("Cannot read fold model");f["joint_v1_origin"]>>jo;f["joint_v1_projector"]>>jp;f["control_v1_Y_only_coef"]>>co;f["control_v1_precision"]>>precision;}
 double joint(const cv::Mat& raw)const{cv::Mat q=(fmap(raw)-jo)*jp;return q.at<double>(1);}
 double control(const cv::Mat& x)const{double best=1e300,ans=std::numeric_limits<double>::quiet_NaN();for(int k=0;k<4;++k){cv::Mat z=x-co.row(k+2),s=co.row(0),c=co.row(1),sw=s*precision;double v=z.dot(sw)/s.dot(sw);for(int it=0;it<8;++it){cv::Mat jac=s+2*v*c,r=z-v*s-v*v*c;v+=jac.dot(r*precision)/std::max(jac.dot(jac*precision),1e-20);}if(v*(k%2==0?1:-1)<0)continue;cv::Mat r=z-v*s-v*v*c;double cost=r.dot(r*precision);if(cost<best){best=cost;ans=v;}}return ans;}
};
struct World{
 int degree;cv::Mat jc,cc;std::vector<int> train;std::vector<cv::Mat> jr,cr;const std::vector<Row>& rows; std::vector<int> noise_ids; std::vector<cv::Mat> nj,nc;
 World(const std::vector<Row>&r,int fold,int deg):degree(deg),rows(r){cv::Mat A,J,C;for(int i=0;i<int(r.size());++i)if(r[i].fold!=fold){train.push_back(i);A.push_back(basis(r[i].y,r[i].t,deg));J.push_back(r[i].j);C.push_back(r[i].c);}cv::solve(A,J,jc,cv::DECOMP_SVD);cv::solve(A,C,cc,cv::DECOMP_SVD);for(int id:train){jr.push_back(r[id].j-basis(r[id].y,r[id].t,deg)*jc);cr.push_back(r[id].c-basis(r[id].y,r[id].t,deg)*cc);} std::map<std::string,std::vector<int>> groups;for(int id:train)groups[r[id].group].push_back(id);for(auto&g:groups)if(g.second.size()>=2){cv::Mat mj=cv::Mat::zeros(1,102,CV_64F),mc=cv::Mat::zeros(1,12,CV_64F);for(int id:g.second){mj+=r[id].j;mc+=r[id].c;}mj/=double(g.second.size());mc/=double(g.second.size());for(int id:g.second){noise_ids.push_back(id);nj.push_back(r[id].j-mj);nc.push_back(r[id].c-mc);}}}
 void observe(double y,double t,const Row& anchor,bool noise,std::mt19937&rng,cv::Mat&j,cv::Mat&c)const{ const auto& date=anchor.date;
  j=anchor.j+(basis(y,t,degree)-basis(anchor.y,anchor.t,degree))*jc;c=anchor.c+(basis(y,t,degree)-basis(anchor.y,anchor.t,degree))*cc;
  if(noise){std::vector<std::pair<double,int>> near;for(int k=0;k<int(noise_ids.size());++k){const auto&r=rows[noise_ids[k]];if(r.date==date)near.push_back({std::pow((r.y-y)/2.,2)+std::pow((r.t-t)/2.,2),k});}if(near.empty())for(int k=0;k<int(noise_ids.size());++k){const auto&r=rows[noise_ids[k]];near.push_back({std::pow((r.y-y)/2.,2)+std::pow((r.t-t)/2.,2),k});}if(near.empty())throw std::runtime_error("No repeated training groups");std::sort(near.begin(),near.end());unsigned n=unsigned(std::min<std::size_t>(20,near.size()));unsigned threshold=unsigned(-n)%n,x;do{x=rng();}while(x<threshold);int k=near[x%n].second;j+=nj[k];c+=nc[k];}
 }
};
static double median(std::vector<double>x){std::sort(x.begin(),x.end());return x.size()%2?x[x.size()/2]:(x[x.size()/2]+x[x.size()/2-1])/2;}
int main(int argc,char**argv){try{if(argc!=4)throw std::runtime_error("return_sim INPUT_TSV PRIOR_RESULTS OUTPUT");cv::setNumThreads(1);fs::path prior=fs::u8path(argv[2]),out=fs::u8path(argv[3]);fs::create_directories(out);std::ifstream input(argv[1]);std::string l;std::getline(input,l);std::vector<Row> rows;int i=0;while(std::getline(input,l)){auto v=split(l,'\t');rows.push_back({i++,std::stoi(v[4]),v[0],v[5],std::stod(v[2]),std::stod(v[3]),{}, {}});}std::ifstream features(prior/"features.tsv");std::getline(features,l);while(std::getline(features,l)){auto v=split(l,'\t');auto&r=rows.at(std::stoi(v[0]));r.j=numbers(v[1]).colRange(0,102).clone();r.c=numbers(v[2]);}
 std::vector<double> jms[6],cms[6],dms[6];std::ifstream time(prior/"measurements.tsv");std::getline(time,l);while(std::getline(time,l)){auto v=split(l,'\t');int f=std::stoi(v[2]);dms[f].push_back(std::stod(v[5]));jms[f].push_back(std::stod(v[6]));cms[f].push_back(std::stod(v[7]));}
 std::ofstream val(out/"surrogate_validation.tsv");val<<std::setprecision(17)<<"world_degree\tfold\tindex\tdate\tY\tT\tcontrol_estimate_difference\tjoint_estimate_difference\n"; std::ofstream trials(out/"trials.tsv"),trace(out/"trace.tsv"),diag(out/"world_fit.tsv");trials<<std::setprecision(17)<<"world_degree\tscenario\tfold\tindex\tdate\tgroup\trepeat\tstart_Y\tstart_T\tfinal_Y\tfinal_T\tsteps\tcaptures\tcompute_ms\tassumed_total_ms\tdeclared_stop\tactual_tolerance_pass\tstatus\n";trace<<std::setprecision(17)<<"world_degree\tscenario\tfold\tindex\trepeat\tcapture\tsteps_before\ttrue_Y\ttrue_T\test_T\test_Y\tcommand_Y\tcommand_T\taction\n";diag<<"world_degree\tfold\ttrain_n\tjoint_raw_RMSE\tcontrol_raw_RMSE\n";
 for(int degree:{1,2})for(int fold=1;fold<=5;++fold){Controller ctl(prior/("fold_"+std::to_string(fold)+".yml"));World world(rows,fold,degree);double sj=0,sc=0;for(auto&r:world.jr)sj+=r.dot(r);for(auto&r:world.cr)sc+=r.dot(r);diag<<degree<<'\t'<<fold<<'\t'<<world.train.size()<<'\t'<<std::sqrt(sj/(world.train.size()*102))<<'\t'<<std::sqrt(sc/(world.train.size()*12))<<'\n';cv::FileStorage fsout((out/("world_d"+std::to_string(degree)+"_fold"+std::to_string(fold)+".yml")).u8string(),cv::FileStorage::WRITE);fsout<<"joint_forward"<<world.jc<<"control_forward"<<world.cc;
 for(auto&r:rows)if(r.fold==fold){int anchor=-1;double best=1e300;for(int id:world.train)if(rows[id].date==r.date){double dist=std::pow((rows[id].y-r.y)/2.,2)+std::pow((rows[id].t-r.t)/2.,2);if(dist<best){best=dist;anchor=id;}}cv::Mat j,c;std::mt19937 dummy(1);world.observe(r.y,r.t,rows.at(anchor),false,dummy,j,c);val<<degree<<'\t'<<fold<<'\t'<<r.id<<'\t'<<r.date<<'\t'<<r.y<<'\t'<<r.t<<'\t'<<std::abs(ctl.control(c)-ctl.control(r.c))<<'\t'<<std::abs(ctl.joint(j)-ctl.joint(r.j))<<'\n';} for(int scenario=0;scenario<3;++scenario)for(auto&r:rows)if(r.fold==fold)for(int rep=0;rep<5;++rep){std::mt19937 rng(20261104u+degree*100000u+scenario*10000u+r.id*17u+rep);std::normal_distribution<double> normal(0.,1.);double y=r.y,t=r.t,compute=0,total=0;int steps=0,captures=0;bool stopped=false;std::string status="max_steps";
 for(;;){++captures;cv::Mat j,c;if(captures==1){j=r.j;c=r.c;}else world.observe(y,t,r,scenario>0,rng,j,c);double et=ctl.joint(j),ey=std::numeric_limits<double>::quiet_NaN(),dy=0,dt=0;compute+=median(jms[fold]);total+=median(jms[fold])+median(dms[fold])+100.;std::string action;
 if(!std::isfinite(et)){status="nonfinite_joint";action=status;}
 else if(std::abs(et)>.005+1e-12){dt=std::clamp(std::round(-et/.01)*.01,-1.,1.);action="move_T";}
 else {ey=ctl.control(c);compute+=median(cms[fold]);total+=median(cms[fold]);if(!std::isfinite(ey)){status="nonfinite_control";action=status;}else if(std::abs(ey)>.025+1e-12){dy=std::clamp(std::round(-ey/.05)*.05,-.5,.5);action="move_Y";}else{stopped=true;status="estimated_stop";action=status;}}
 trace<<degree<<'\t'<<scenario<<'\t'<<fold<<'\t'<<r.id<<'\t'<<rep+1<<'\t'<<captures<<'\t'<<steps<<'\t'<<y<<'\t'<<t<<'\t'<<et<<'\t'<<ey<<'\t'<<dy<<'\t'<<dt<<'\t'<<action<<'\n';
 if(stopped||action.find("nonfinite")!=std::string::npos)break;if(steps>=20){status="max_steps";break;}if(dy==0&&dt==0){status="zero_command_stall";break;}
 double my=dy,mt=dt;if(scenario==2){my+=.005*normal(rng);mt+=.002*normal(rng);}y+=my;t+=mt;++steps;total+=50.+200.+1000.*std::max(std::abs(dy),std::abs(dt));if(!std::isfinite(y)||!std::isfinite(t)||std::abs(y)>20||std::abs(t)>20){status="diverged";break;}
 }
 bool pass=stopped&&std::abs(y)<=.045+1e-12&&std::abs(t)<=.01+1e-12;trials<<degree<<'\t'<<scenario<<'\t'<<fold<<'\t'<<r.id<<'\t'<<r.date<<'\t'<<r.group<<'\t'<<rep+1<<'\t'<<r.y<<'\t'<<r.t<<'\t'<<y<<'\t'<<t<<'\t'<<steps<<'\t'<<captures<<'\t'<<compute<<'\t'<<total<<'\t'<<stopped<<'\t'<<pass<<'\t'<<status<<'\n';}
 std::cout<<"world degree "<<degree<<" fold "<<fold<<" complete"<<std::endl;}
 return 0;}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
