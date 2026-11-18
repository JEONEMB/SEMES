#include "observation.hpp"
#include "qrwithbar_lanczos.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <map>
#include <set>
#include <cmath>
namespace fs=std::filesystem;
using Features=std::array<double,10>;
struct Row{std::string path,hash,group,error;int fold=-1;bool ok=false,oldtrain=false;double y=0,t=0,repeat=-1;Features x{};cv::Vec2d full{},held{},fixed{};};
struct Model{cv::Mat mean,scale,ym,coef;};
static std::string esc(const std::string&s){std::string r;for(char c:s){if(c=='"'||c=='\\')r+='\\';if(c=='\n'){r+="\\n";continue;}if(c!='\r')r+=c;}return r;}
static std::vector<std::string> split(const std::string&s){std::vector<std::string>v;std::istringstream in(s);std::string x;while(std::getline(in,x,','))v.push_back(x);return v;}
static double median(std::vector<double>v){if(v.empty())return 0;std::sort(v.begin(),v.end());size_t n=v.size();return n%2?v[n/2]:(v[n/2-1]+v[n/2])/2;}
static std::vector<double> weights(const std::vector<Row>&rows,const std::vector<int>&ids){std::vector<double> norms;for(int i:ids)if(rows[i].repeat>=0)norms.push_back(rows[i].repeat);double med=median(norms);std::vector<double>dev;for(double v:norms)dev.push_back(fabs(v-med));double threshold=med+4*std::max(1.4826*median(dev),.02);std::vector<double>w;for(int i:ids)w.push_back(rows[i].repeat>threshold?std::max(.2,pow(threshold/rows[i].repeat,2)):1.);return w;}
static Model fit(const std::vector<Row>&rows,const std::vector<int>&ids){
 if(ids.size()<12)throw std::runtime_error("Too few training images");Model m;int n=(int)ids.size();cv::Mat x(n,10,CV_64F),y(n,2,CV_64F);
 for(int i=0;i<n;i++){for(int j=0;j<10;j++)x.at<double>(i,j)=rows[ids[i]].x[j];y.at<double>(i,0)=rows[ids[i]].y/.05;y.at<double>(i,1)=rows[ids[i]].t/.01;}
 auto w=weights(rows,ids);double sw=std::accumulate(w.begin(),w.end(),0.);m.mean=cv::Mat::zeros(1,10,CV_64F);m.ym=cv::Mat::zeros(1,2,CV_64F);m.scale=cv::Mat(1,10,CV_64F);
 for(int i=0;i<n;i++){for(int j=0;j<10;j++)m.mean.at<double>(0,j)+=w[i]*x.at<double>(i,j)/sw;for(int j=0;j<2;j++)m.ym.at<double>(0,j)+=w[i]*y.at<double>(i,j)/sw;}
 for(int j=0;j<10;j++){double sum=0;for(int i=0;i<n;i++)sum+=w[i]*pow(x.at<double>(i,j)-m.mean.at<double>(0,j),2);m.scale.at<double>(0,j)=std::max(1e-8,sqrt(sum/sw));for(int i=0;i<n;i++)x.at<double>(i,j)=sqrt(w[i])*(x.at<double>(i,j)-m.mean.at<double>(0,j))/m.scale.at<double>(0,j);}
 for(int i=0;i<n;i++)for(int j=0;j<2;j++)y.at<double>(i,j)=sqrt(w[i])*(y.at<double>(i,j)-m.ym.at<double>(0,j));cv::SVD svd(x);cv::Mat z=svd.u.t()*y;
 for(int i=0;i<svd.w.rows;i++){double s=svd.w.at<double>(i);z.row(i)*=s>svd.w.at<double>(0)*1e-10?1/s:0;}m.coef=svd.vt.t()*z;if(!cv::checkRange(m.coef))throw std::runtime_error("Nonfinite fitted model");return m;
}
static cv::Vec2d predict(const Model&m,const Features&x){cv::Vec2d p(m.ym.at<double>(0,0),m.ym.at<double>(0,1));for(int j=0;j<10;j++)for(int k=0;k<2;k++)p[k]+=(x[j]-m.mean.at<double>(0,j))/m.scale.at<double>(0,j)*m.coef.at<double>(j,k);return {p[0]*.05,p[1]*.01};}
static void matrices(cv::FileStorage&f,const Model&m){f<<"mean"<<m.mean<<"scale"<<m.scale<<"ymean_ticks"<<m.ym<<"coef"<<m.coef<<"alpha"<<0.;}
static void data(std::ostream&f,const Row&r,bool held,bool fixedMode=false){auto p=held?r.held:r.full;f<<std::setprecision(17)<<"{\"path\":\""<<esc(r.path)<<"\",\"sha256\":\""<<r.hash<<"\",\"group\":\""<<r.group<<"\",\"Y\":"<<r.y<<",\"T\":"<<r.t<<",\"predY\":"<<p[0]<<",\"predT\":"<<p[1]<<",\"ok\":"<<(r.ok?"true":"false")<<",\"training\":"<<(fixedMode?(r.oldtrain?"true":"false"):(!held?"true":"false"))<<",\"fold\":"<<r.fold<<",\"error\":\""<<esc(r.error)<<"\"}";}
static void plot(const fs::path&out,const std::vector<Row>&rows,bool held){
 cv::Mat canvas(1050,1600,CV_8UC3,cv::Scalar(255,255,255));auto text=[&](std::string s,int x,int y,double scale=.55){cv::putText(canvas,s,{x,y},cv::FONT_HERSHEY_SIMPLEX,scale,{55,55,55},1,cv::LINE_AA);};
 text(held?"All 575 images - held-coordinate predictions (378 folds)":"All 575 images - full-data model, training reapplication",75,40,.8);
 for(int panel=0;panel<3;panel++){int L=panel==0?95:panel==1?95:865,T=panel==0?100:690,W=panel==0?1400:640,H=panel==0?465:240;double minx=0,maxx=0,miny=0,maxy=0;
 for(size_t i=0;i<rows.size();i++){auto&r=rows[i];auto p=held?r.held:r.full;if(panel==0){minx=std::min(minx,std::min(r.y,p[0]));maxx=std::max(maxx,std::max(r.y,p[0]));miny=std::min(miny,std::min(r.t,p[1]));maxy=std::max(maxy,std::max(r.t,p[1]));}else{maxx=(double)rows.size();maxy=std::max(maxy,fabs(p[panel-1]-(panel==1?r.y:r.t)));}}
 if(panel>0)maxy=std::max(maxy,.09);double dx=std::max(1e-6,maxx-minx),dy=std::max(1e-6,maxy-miny);maxx+=dx*.03;if(panel==0){minx-=dx*.03;miny-=dy*.04;}maxy+=dy*.08;
 auto X=[&](double v){return L+cvRound((v-minx)/(maxx-minx)*W);};auto Y=[&](double v){return T+H-cvRound((v-miny)/(maxy-miny)*H);};
 for(int j=0;j<=5;j++){double x=minx+(maxx-minx)*j/5,y=miny+(maxy-miny)*j/5;cv::line(canvas,{X(x),T},{X(x),T+H},{235,235,235});cv::line(canvas,{L,Y(y)},{L+W,Y(y)},{235,235,235});std::ostringstream a,b;a<<std::setprecision(3)<<x;b<<std::setprecision(3)<<y;text(a.str(),X(x)-15,T+H+25,.45);text(b.str(),L-70,Y(y)+4,.45);}cv::rectangle(canvas,{L,T,W,H},{170,170,170});
 text(panel==0?"Yaw (deg): circle = command, cross = prediction":panel==1?"Absolute Y error (mm)":"Absolute yaw error (deg)",L,T-20,.6);text(panel==0?"Y (mm)":"Image index (path order, not time)",L+W/2-80,T+H+52,.5);if(panel>0){for(int a=L;a<L+W;a+=12)cv::line(canvas,{a,Y(.09)},{std::min(a+6,L+W),Y(.09)},{140,140,140});text("0.09",L+W-50,Y(.09)-5,.4);}
 for(size_t i=0;i<rows.size();i++){auto&r=rows[i];auto p=held?r.held:r.full;cv::Scalar blue(227,113,0);if(panel==0){cv::Point a(X(r.y),Y(r.t)),b(X(p[0]),Y(p[1]));cv::line(canvas,a,b,{200,200,200},1,cv::LINE_AA);cv::circle(canvas,a,3,{110,110,110},1,cv::LINE_AA);cv::drawMarker(canvas,b,blue,cv::MARKER_TILTED_CROSS,6,1,cv::LINE_AA);}else cv::circle(canvas,{X((double)i+1),Y(fabs(p[panel-1]-(panel==1?r.y:r.t)))},2,blue,-1,cv::LINE_AA);}}
 text("Labels: audited commands (.33 -> .32 once); all images retained. Development data, not new-capture validation.",75,1015,.6);
 if(!cv::imwrite((out/(held?"OOF575_overview.png":"full575_overview.png")).u8string(),canvas))throw std::runtime_error("Could not write plot");
}
int wmain(int argc,wchar_t**argv){try{
 if(argc!=4)throw std::runtime_error("Usage: full_train MODULE RESULTS DATA_ROOT");fs::path module=argv[1],out=argv[2],root=argv[3];cv::setNumThreads(1);
 std::ifstream in(out/"features.csv");std::string s;std::getline(in,s);std::map<std::string,Features>features;while(std::getline(in,s)){auto v=split(s);if(v.size()!=13)throw std::runtime_error("Invalid feature row");Features a;for(int j=0;j<10;j++)a[j]=std::stod(v[j+3]);features[v[0]]=a;}
 std::map<std::string,double>repeat;std::ifstream diagnostic(out.parent_path()/"audit/final_repeat_diagnostics.csv");std::getline(diagnostic,s);while(std::getline(diagnostic,s)){auto v=split(s);repeat[v.at(1)]=std::stod(v.at(6));}
 std::vector<Row>rows;std::ifstream log(out/"predictions.jsonl");std::map<std::pair<double,double>,int>groups;std::vector<int>ids;
 while(std::getline(log,s)){cv::FileStorage f(s,cv::FileStorage::READ|cv::FileStorage::MEMORY|cv::FileStorage::FORMAT_JSON);Row r;r.path=(std::string)f["path"];r.hash=(std::string)f["sha256"];r.group=(std::string)f["group"];r.error=(std::string)f["error"];r.ok=(int)f["ok"]!=0;r.oldtrain=(int)f["training"]!=0;r.y=(double)f["Y"];r.t=(double)f["T"];r.fixed={(double)f["predY"],(double)f["predT"]};if(r.path=="Y+0.05 T+0.32/Y+0.05 T+0.33 (3).bmp")r.t=.32;if(repeat.count(r.path))r.repeat=repeat.at(r.path);if(r.ok){r.x=features.at(r.path);auto key=std::make_pair(r.y,r.t);if(!groups.count(key))groups[key]=(int)groups.size();r.fold=groups[key];ids.push_back((int)rows.size());}rows.push_back(r);}
 if(rows.size()!=575)throw std::runtime_error("Expected the inventoried 575 images");if(ids.size()!=rows.size())throw std::runtime_error("Some images failed; cannot claim full575 training");
 auto full=fit(rows,ids);for(auto&r:rows)if(r.ok)r.full=predict(full,r.x);
 cv::FileStorage heldmodels((out/"held_coordinate_models.json").u8string(),cv::FileStorage::WRITE|cv::FileStorage::FORMAT_JSON);heldmodels<<"alpha_fixed_before_evaluation"<<0.<<"models"<<"[";
 for(int g=0;g<(int)groups.size();g++){std::vector<int>train,test;for(int i:ids)(rows[i].fold==g?test:train).push_back(i);auto model=fit(rows,train);heldmodels<<"{"<<"held_group"<<g<<"training_images"<<(int)train.size();matrices(heldmodels,model);heldmodels<<"}";for(int i:test)rows[i].held=predict(model,rows[i].x);if(g%50==0)std::cout<<"Coordinate folds "<<g<<'/'<<groups.size()<<'\n'<<std::flush;}heldmodels<<"]";heldmodels.release();
 auto assets=out/"full_assets";fs::create_directories(assets);for(auto&e:fs::directory_iterator(module/"assets"))if(e.path().extension()==L".png")fs::copy_file(e.path(),assets/e.path().filename(),fs::copy_options::overwrite_existing);
 cv::FileStorage m((assets/"model.json").u8string(),cv::FileStorage::WRITE|cv::FileStorage::FORMAT_JSON);m<<"schema"<<"qrwithbar.lanczos3.10features.v1"<<"method"<<"lanczos3"<<"feature_count"<<10<<"training_images"<<(int)ids.size()<<"training_coordinate_groups"<<(int)groups.size()<<"label_basis"<<"filename commands"<<"regression"<<"10-feature weighted standardized SVD; alpha0 fixed; audit repeat QC";matrices(m,full);
 double loY=0,hiY=0,loT=0,hiT=0;for(auto&r:rows)if(r.ok){loY=std::min(loY,r.y);hiY=std::max(hiY,r.y);loT=std::min(loT,r.t);hiT=std::max(hiT,r.t);}m<<"minimum_y"<<loY<<"maximum_y"<<hiY<<"minimum_t"<<loT<<"maximum_t"<<hiT;m.release();
 auto weight=weights(rows,ids);std::ofstream wf(out/"training_weights.csv");wf<<std::setprecision(17)<<"path,repeat_common_norm_px,weight\n";for(size_t i=0;i<ids.size();i++)wf<<rows[ids[i]].path<<','<<rows[ids[i]].repeat<<','<<weight[i]<<'\n';std::ofstream js(out/"full_data.js"),csv(out/"full_predictions.csv");csv<<std::setprecision(17)<<"path,sha256,group,coordinate_fold,Y_command,yaw_command,full_Y,full_yaw,OOF_Y,OOF_yaw,full_Y_error,full_yaw_error,OOF_Y_error,OOF_yaw_error\n";
 js<<"const rowsFixedAudited=[";for(size_t i=0;i<rows.size();i++){if(i)js<<',';auto fixed=rows[i];fixed.full=fixed.fixed;data(js,fixed,false,true);}js<<"];\n";for(int mode=0;mode<2;mode++){js<<(mode==0?"const rowsFull=[":"const rowsOOF=[");for(size_t i=0;i<rows.size();i++){if(i)js<<',';data(js,rows[i],mode==1);}js<<"];\n";}
 for(auto&r:rows)csv<<r.path<<','<<r.hash<<','<<r.group<<','<<r.fold<<','<<r.y<<','<<r.t<<','<<r.full[0]<<','<<r.full[1]<<','<<r.held[0]<<','<<r.held[1]<<','<<r.full[0]-r.y<<','<<r.full[1]-r.t<<','<<r.held[0]-r.y<<','<<r.held[1]-r.t<<'\n';
 // Confirm exported model against actual runtime on three different subsets.
 qrwithbar::Estimator runtime(assets.u8string());double maxdelta=0;for(int i:{0,(int)rows.size()/2,(int)rows.size()-1}){auto image=qrwithbar::detail::read_gray(root/fs::u8path(rows[i].path));auto v=runtime.estimate(image);maxdelta=std::max(maxdelta,std::max(fabs(v.y_mm-rows[i].full[0]),fabs(v.yaw_deg-rows[i].full[1])));}if(maxdelta>1e-10)throw std::runtime_error("Exported runtime model mismatch");
 cv::FileStorage meta((out/"full_training.json").u8string(),cv::FileStorage::WRITE|cv::FileStorage::FORMAT_JSON);meta<<"training_images"<<(int)ids.size()<<"evaluation_images"<<(int)rows.size()<<"coordinate_groups"<<(int)groups.size()<<"alpha_fixed"<<0.<<"OOF_all_images_evaluated"<<1<<"label_basis"<<"filename commands"<<"runtime_three_images_max_difference"<<maxdelta<<"model_sha256"<<qrwithbar::detail::sha256(assets/"model.json")<<"features_sha256"<<qrwithbar::detail::sha256(out/"features.csv")<<"independent_new_capture_test"<<0;
 std::ofstream metrics(out/"all_metrics.csv");metrics<<std::setprecision(17)<<"method,subset,count,Y_mae,Y_max,Y_rmse,yaw_mae,yaw_max,yaw_rmse,both009,both_tick\n";
 for(int mode=0;mode<3;mode++)for(const std::string group:{"all","Y-axis","yaw-axis","mixed","center"}){int n=0,passes=0,ticks=0;double sum[2]={},mx[2]={},sq[2]={};for(auto&r:rows){if(!r.ok||(group!="all"&&group!=r.group))continue;n++;auto p=mode==0?r.fixed:mode==1?r.full:r.held;double e[2]={p[0]-r.y,p[1]-r.t};passes+=fabs(e[0])<=.09&&fabs(e[1])<=.09;ticks+=fabs(e[0])<=.05&&fabs(e[1])<=.01;for(int k=0;k<2;k++){sum[k]+=fabs(e[k]);mx[k]=std::max(mx[k],fabs(e[k]));sq[k]+=e[k]*e[k];}}metrics<<(mode==0?"fixed57":mode==1?"full575":"OOF575")<<','<<group<<','<<n;for(int k=0;k<2;k++)metrics<<','<<sum[k]/n<<','<<mx[k]<<','<<sqrt(sq[k]/n);metrics<<','<<passes<<','<<ticks<<'\n';}
 plot(out,rows,false);plot(out,rows,true);
 std::cout<<"Full training complete: "<<ids.size()<<" images; grouped OOF "<<rows.size()<<" images; "<<groups.size()<<" folds; runtime delta "<<maxdelta<<'\n';return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
