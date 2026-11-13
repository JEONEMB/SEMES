#include "observation.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <cmath>
namespace fs=std::filesystem;
using Features=std::array<double,10>;
struct Row{std::string path;int fold;double y,t;Features x;};
struct Model{cv::Mat mean,scale,ym,coef;double alpha;};
static std::vector<std::string> split(const std::string& s){std::vector<std::string> v;std::istringstream in(s);std::string x;while(std::getline(in,x,','))v.push_back(x);return v;}
static Model fit(const std::vector<Row>& rows,const std::vector<int>& ids,double alpha){
    Model m;m.alpha=alpha;int n=(int)ids.size();cv::Mat x(n,10,CV_64F),y(n,2,CV_64F);
    for(int r=0;r<n;r++){for(int j=0;j<10;j++)x.at<double>(r,j)=rows[ids[r]].x[j];y.at<double>(r,0)=rows[ids[r]].y/.05;y.at<double>(r,1)=rows[ids[r]].t/.01;}
    cv::reduce(x,m.mean,0,cv::REDUCE_AVG);cv::reduce(y,m.ym,0,cv::REDUCE_AVG);m.scale=cv::Mat(1,10,CV_64F);
    for(int j=0;j<10;j++){double sum=0;for(int r=0;r<n;r++)sum+=pow(x.at<double>(r,j)-m.mean.at<double>(0,j),2);m.scale.at<double>(0,j)=std::max(1e-8,sqrt(sum/n));for(int r=0;r<n;r++)x.at<double>(r,j)=(x.at<double>(r,j)-m.mean.at<double>(0,j))/m.scale.at<double>(0,j);}
    for(int r=0;r<n;r++)for(int j=0;j<2;j++)y.at<double>(r,j)-=m.ym.at<double>(0,j);
    cv::SVD svd(x);cv::Mat projected=svd.u.t()*y;
    for(int r=0;r<svd.w.rows;r++){double s=svd.w.at<double>(r),factor=alpha?s/(s*s+alpha):(s>svd.w.at<double>(0)*1e-10?1/s:0);projected.row(r)*=factor;}
    m.coef=svd.vt.t()*projected;return m;
}
static cv::Vec2d predict(const Model& m,const Features& x){cv::Vec2d y(m.ym.at<double>(0,0),m.ym.at<double>(0,1));for(int j=0;j<10;j++)for(int k=0;k<2;k++)y[k]+=(x[j]-m.mean.at<double>(0,j))/m.scale.at<double>(0,j)*m.coef.at<double>(j,k);return {y[0]*.05,y[1]*.01};}
static double tune(const std::vector<Row>& rows,const std::vector<int>& ids){double best=1e300,result=0;for(double a:{0.,.001,.01,.1,1.,10.,100.}){double cost=0;for(int g=0;g<19;g++){std::vector<int> train,test;for(int i:ids)(rows[i].fold==g?test:train).push_back(i);if(test.empty())continue;auto model=fit(rows,train,a);for(int i:test){auto p=predict(model,rows[i].x);cost+=pow((p[0]-rows[i].y)/.05,2)+pow((p[1]-rows[i].t)/.01,2);}}if(cost<best){best=cost;result=a;}}return result;}
static void write_model(cv::FileStorage& out,const Model& model){out<<"mean"<<model.mean<<"scale"<<model.scale<<"ymean_ticks"<<model.ym<<"coef"<<model.coef<<"alpha"<<model.alpha;}
int wmain(int argc,wchar_t** argv){try{
    if(argc!=3)throw std::invalid_argument("Usage: qrwithbar_train MODULE_DIRECTORY IMAGE_ROOT");
    fs::path module=argv[1],images=argv[2],fixtures=module/"tests/fixtures";cv::setNumThreads(1);
    std::vector<Row> rows;std::ifstream input(fixtures/"manifest.csv");std::string text;std::getline(input,text);
    while(std::getline(input,text)){auto v=split(text);Row r;r.path=v.at(0);r.fold=std::stoi(v.at(1));r.y=std::stod(v.at(12));r.t=std::stod(v.at(13));rows.push_back(r);}
    if(rows.size()!=57)throw std::runtime_error("Expected manifest57");
    input.close();input.open(fixtures/"comparison_features.csv");std::getline(input,text);int found=0;
    while(std::getline(input,text)){auto v=split(text);if(v.at(0)!="lanczos3")continue;if(std::stoi(v.at(2))!=0)throw std::runtime_error("Failed training measurement");auto r=std::find_if(rows.begin(),rows.end(),[&](const Row& row){return row.path==v.at(1);});if(r==rows.end())throw std::runtime_error("Unmatched feature path");for(int j=0;j<10;j++)r->x[j]=std::stod(v.at(4+j));found++;}
    if(found!=57)throw std::runtime_error("Expected57 Lanczos feature rows");
    cv::FileStorage held((fixtures/"lanczos_validation_models.json").u8string(),cv::FileStorage::WRITE|cv::FileStorage::FORMAT_JSON);held<<"method"<<"lanczos3"<<"models"<<"[";
    for(int g=0;g<19;g++){std::vector<int> ids;for(int i=0;i<57;i++)if(rows[i].fold!=g)ids.push_back(i);auto model=fit(rows,ids,tune(rows,ids));held<<"{"<<"held_group"<<g;write_model(held,model);held<<"}";}held<<"]";held.release();
    std::vector<int> all(57);std::iota(all.begin(),all.end(),0);auto model=fit(rows,all,tune(rows,all));
    cv::FileStorage file((module/"assets/model.json").u8string(),cv::FileStorage::WRITE|cv::FileStorage::FORMAT_JSON);
    file<<"schema"<<"qrwithbar.lanczos3.10features.v1"<<"method"<<"lanczos3"<<"feature_count"<<10<<"training_images"<<57<<"training_coordinate_groups"<<19;
    file<<"feature_order"<<"["<<"L_dx"<<"L_dy"<<"C_dx"<<"C_dy"<<"R_dx"<<"R_dy"<<"upper_position"<<"upper_slope"<<"lower_position"<<"lower_slope"<<"]";
    write_model(file,model);auto range=[&](bool theta,bool maximum){double value=theta?rows[0].t:rows[0].y;for(auto& r:rows){double x=theta?r.t:r.y;value=maximum?std::max(value,x):std::min(value,x);}return value;};
    file<<"minimum_y"<<range(false,false)<<"maximum_y"<<range(false,true)<<"minimum_t"<<range(true,false)<<"maximum_t"<<range(true,true);file.release();
    auto reference=qrwithbar::detail::read_gray(images/fs::u8path(u8"기준 위치/center1.bmp"));
    if(qrwithbar::detail::sha256(images/fs::u8path(u8"기준 위치/center1.bmp"))!="e50e9579a787bf1dfa2f82b6f5efa6036026c6195986d7855ce2989e9d890c6c")throw std::runtime_error("Unexpected reference bytes");
    if(!cv::imwrite((module/"assets/reference.png").u8string(),reference))throw std::runtime_error("Cannot write calibration reference");
    if(cv::norm(reference,qrwithbar::detail::read_gray(module/"assets/reference.png"),cv::NORM_INF)!=0)throw std::runtime_error("Reference PNG changed grayscale pixels");
    std::cout<<"Exported Lanczos deployment model(alpha="<<model.alpha<<"),19 validation-only models and lossless reference PNG\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
