#include "six.hpp"
#include "reference_data.h"
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <stdexcept>
namespace six {
cv::Mat describe(const Frame& f,const Frame& r){
 cv::Mat x=cv::Mat::zeros(1,dimensions,CV_64F);for(int j=0;j<10;j++)x.at<double>(j)=f.observation.features[j];
 for(int q=0;q<3;q++){
  double x0=1e9,y0=1e9,x1=-1e9,y1=-1e9;std::vector<cv::Point2d> a,b;
  for(int i=0;i<630;i++)if(r.observation.points[i].qr==q){auto p=r.observation.points[i];x0=std::min(x0,p.x);x1=std::max(x1,p.x);y0=std::min(y0,p.y);y1=std::max(y1,p.y);}
  int count[9]={};
  for(int i=0;i<630;i++)if(r.observation.points[i].qr==q){auto p=r.observation.points[i],v=f.observation.points[i];int col=std::min(2,(int)(3*(p.x-x0)/(x1-x0+1e-9))),row=std::min(2,(int)(3*(p.y-y0)/(y1-y0+1e-9))),cell=row*3+col;count[cell]++;x.at<double>(10+q*18+cell*2)+=v.x-p.x;x.at<double>(11+q*18+cell*2)+=v.y-p.y;a.push_back({(p.x-(x0+x1)/2)/100,(p.y-(y0+y1)/2)/100});b.push_back({(v.x-(x0+x1)/2)/100,(v.y-(y0+y1)/2)/100});}
  for(int cell=0;cell<9;cell++){if(!count[cell])throw std::runtime_error("Empty fixed QR cell");x.at<double>(10+q*18+cell*2)/=count[cell];x.at<double>(11+q*18+cell*2)/=count[cell];}
  auto h=cv::findHomography(a,b,0);if(h.empty())throw std::runtime_error("Projective QR fit failed");h/=h.at<double>(2,2);for(int j=0;j<8;j++)x.at<double>(72+q*8+j)=h.at<double>(j/3,j%3)-((j==0||j==4)?1.:0.);
 }
 for(int e=0;e<2;e++)for(int side=0;side<2;side++){
  cv::Mat a(30,2,CV_64F),b(30,1,CV_64F),coef;
  for(int k=0;k<30;k++){auto s=f.observation.edges[e*60+side*30+k];double u=(s.x-1920)/1920.,baseline=qc_reference_edges[e*2]+r.observation.features[e*2+6]+(qc_reference_edges[e*2+1]+r.observation.features[e*2+7])*u;double w=sqrt((s.valid?1.:.02)/pow(std::max(1.5,s.rmse),2));a.at<double>(k,0)=w;a.at<double>(k,1)=w*u;b.at<double>(k)=(s.y-baseline)*w;}
  if(!cv::solve(a,b,coef,cv::DECOMP_SVD))throw std::runtime_error("Split line fit failed");x.at<double>(64+e*4+side*2)=coef.at<double>(0);x.at<double>(65+e*4+side*2)=coef.at<double>(1);
 }
 std::vector<cv::Point2d> globalReference,globalCurrent;for(int i=0;i<630;i++){auto a=r.observation.points[i],b=f.observation.points[i];globalReference.push_back({(a.x-1920)/1000,(a.y-1080)/1000});globalCurrent.push_back({(b.x-1920)/1000,(b.y-1080)/1000});}
 auto global=cv::findHomography(globalReference,globalCurrent,0);if(global.empty())throw std::runtime_error("Shared QR-plane homography failed");global/=global.at<double>(2,2);for(int j=0;j<8;j++)x.at<double>(96+j)=global.at<double>(j/3,j%3)-((j==0||j==4)?1.:0.);
 if(!cv::checkRange(x))throw std::runtime_error("Nonfinite rich measurement");return x;
}
void save_frame(const Frame& f,const std::filesystem::path& path){cv::FileStorage out(path.u8string(),cv::FileStorage::WRITE);cv::Mat p(630,3,CV_64F),e(120,6,CV_64F),w(3,6,CV_64F),z(1,10,CV_64F);for(int i=0;i<630;i++){p.at<double>(i,0)=f.observation.points[i].x;p.at<double>(i,1)=f.observation.points[i].y;p.at<double>(i,2)=f.observation.points[i].qr;}for(int i=0;i<120;i++){auto s=f.observation.edges[i];double values[]={s.x,s.y,s.sigma,s.contrast,s.rmse,(double)s.valid};for(int j=0;j<6;j++)e.at<double>(i,j)=values[j];}for(int i=0;i<3;i++)for(int j=0;j<6;j++)w.at<double>(i,j)=f.warps[i].val[j];for(int j=0;j<10;j++)z.at<double>(j)=f.observation.features[j];out<<"points"<<p<<"edges"<<e<<"warps"<<w<<"features"<<z;}
Frame load_frame(const std::filesystem::path& path){cv::FileStorage in(path.u8string(),cv::FileStorage::READ);if(!in.isOpened())throw std::runtime_error("Missing frame cache");cv::Mat p,e,w,z;in["points"]>>p;in["edges"]>>e;in["warps"]>>w;in["features"]>>z;if(p.rows!=630||p.cols!=3||e.rows!=120||e.cols!=6||w.rows!=3||w.cols!=6||z.cols!=10)throw std::runtime_error("Invalid frame cache");Frame f;for(int i=0;i<630;i++)f.observation.points[i]={p.at<double>(i,0),p.at<double>(i,1),(int)p.at<double>(i,2)};for(int i=0;i<120;i++)f.observation.edges[i]={e.at<double>(i,0),e.at<double>(i,1),e.at<double>(i,2),e.at<double>(i,3),e.at<double>(i,4),(int)e.at<double>(i,5)};for(int i=0;i<3;i++)for(int j=0;j<6;j++)f.warps[i].val[j]=w.at<double>(i,j);for(int j=0;j<10;j++)f.observation.features[j]=z.at<double>(j);return f;}
std::vector<Row> read_rows(const std::filesystem::path& path){std::ifstream f(path);if(!f)throw std::runtime_error("Missing rich observations");std::string line;std::getline(f,line);std::vector<Row> rows;while(std::getline(f,line)){std::istringstream stream(line);std::vector<std::string> v;std::string s;while(std::getline(stream,s,','))v.push_back(s);if(v.size()!=104&&v.size()!=112)throw std::runtime_error("Unexpected rich feature column count");Row r;r.path=v.at(0);r.cohort=v.at(1);r.y=std::stod(v.at(2));r.t=std::stod(v.at(3));r.time=std::stod(v.at(4));r.session=std::stoi(v.at(5));r.status=std::stoi(v.at(6));r.index=std::stoi(v.at(7));r.x=cv::Mat(1,dimensions,CV_64F);for(int j=0;j<dimensions;j++)r.x.at<double>(j)=(8+j<(int)v.size())?std::stod(v.at(8+j)):0;std::string key=std::to_string(std::lround(r.y*100))+":"+std::to_string(std::lround(r.t*100));unsigned h=2166136261u;for(unsigned char c:key)h=(h^c)*16777619u;r.fold=h%5;rows.push_back(r);}return rows;}
}
