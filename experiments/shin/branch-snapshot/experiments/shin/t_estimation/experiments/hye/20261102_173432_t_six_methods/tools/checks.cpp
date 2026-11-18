#include "six.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <stdexcept>
int main(){try{cv::setNumThreads(1);std::vector<six::Row> rows;for(int y=-2;y<=2;y++)for(int t=-2;t<=2;t++)for(int repeat=0;repeat<3;repeat++){six::Row r;r.y=y*.3;r.t=t*.2;r.time=1000+rows.size();r.session=y+2;r.x=cv::Mat(1,six::dimensions,CV_64F);for(int j=0;j<six::dimensions;j++)r.x.at<double>(j)=(1.+.03*j)*r.y+(2.-.013*j)*r.t+.00001*(repeat-1)*(1+j%3);rows.push_back(r);}std::vector<int> train;for(int i=0;i<60;i++)train.push_back(i);
 for(int method=0;method<=6;method++){auto configs=six::candidates(method);auto cfg=configs[0];if(method==0)cfg.alpha=.001;auto first=six::fit(rows,train,cfg);auto before=six::predict(first,rows[65].x,rows[65].time).t;auto changed=rows;for(int i=60;i<(int)changed.size();i++){changed[i].t=1e9;changed[i].y=-1e9;}auto after=six::fit(changed,train,cfg);if(std::abs(before-six::predict(after,rows[65].x,rows[65].time).t)>1e-10)throw std::runtime_error("Held labels affected fitting");if(!std::isfinite(before))throw std::runtime_error("Nonfinite synthetic inference");if((method==1||method==2||method==4)&&std::abs(before-rows[65].t)>.01)throw std::runtime_error("Forward Y/T separation failed");}
 for(int method:{1,2}){auto cfg=six::candidates(method).back();auto model=six::fit(rows,train,cfg);double before=six::predict(model,rows[65].x).t;auto changed=rows;changed[65].t=1e9;changed[65].y=-1e9;auto again=six::fit(changed,train,cfg);if(std::abs(before-six::predict(again,rows[65].x).t)>1e-10)throw std::runtime_error("Extended representation label leak");}
 six::Frame r,f;for(int i=0;i<630;i++){int q=i/210;double x=1200+q*400+(i%14)*10.,y=1000+((i/14)%15)*12.;r.observation.points[i]={x,y,q};f.observation.points[i]={x+7,y-4,q};}
 for(int i=0;i<120;i++){double x=(i%60<30)?250.+(i%30)*15.:3100.+(i%30)*15.;r.observation.edges[i]=f.observation.edges[i]={x,500.+(i/60)*500,1,100,2,1};}
 auto rich=six::describe(f,r);std::cout<<std::setprecision(17)<<"Synthetic homography tx="<<rich.at<double>(98)<<" ty="<<rich.at<double>(101)<<"\n";if(std::abs(rich.at<double>(98)-.007)>1e-7||std::abs(rich.at<double>(101)+.004)>1e-7)throw std::runtime_error("Shared homography translation recovery failed");for(int j=10;j<64;j+=2)if(std::abs(rich.at<double>(j)-7)>1e-10||std::abs(rich.at<double>(j+1)+4)>1e-10)throw std::runtime_error("Fixed QR spatial cells changed");
 six::Model session=six::fit(rows,train,six::candidates(5)[0]);if(six::predict(session,rows[65].x,1e8).known_session)throw std::runtime_error("Unknown timestamp used session correction");
 std::cout<<"All six families: held-label isolation, forward separation, unknown-session fallback passed\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
