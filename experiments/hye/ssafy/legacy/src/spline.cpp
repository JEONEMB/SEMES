#include "control_internal.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace legacy_control {
static void prefilter_line(std::vector<double>& line){
    if(line.size()<2)return;
    constexpr double pole=-0.267949192431122706472553658494127633;
    for(auto& v:line)v*=(1-pole)*(1-1/pole);
    int n=int(line.size());double pn=std::pow(pole,n-1),initial=line[0]+pn*line[n-1],power=pole;
    for(int i=1;i<n-1;++i){initial+=power*(line[i]+pn*line[n-1-i]);power*=pole;}
    line[0]=initial/(1-pn*pn);
    for(int i=1;i<n;++i)line[i]+=pole*line[i-1];
    line[n-1]=(pole*line[n-2]+line[n-1])*pole/(pole*pole-1);
    for(int i=n-2;i>=0;--i)line[i]=pole*(line[i+1]-line[i]);
}
cv::Mat spline_coefficients(const cv::Mat& input){
    cv::Mat out;input.convertTo(out,CV_64F);std::vector<double> line(out.rows);
    for(int x=0;x<out.cols;++x){for(int y=0;y<out.rows;++y)line[y]=out.at<double>(y,x);prefilter_line(line);for(int y=0;y<out.rows;++y)out.at<double>(y,x)=line[y];}
    line.resize(out.cols);
    for(int y=0;y<out.rows;++y){for(int x=0;x<out.cols;++x)line[x]=out.at<double>(y,x);prefilter_line(line);for(int x=0;x<out.cols;++x)out.at<double>(y,x)=line[x];}
    return out;
}
cv::Mat nearest_prefilter(const cv::Mat& image){
    // scipy's nearest+prefilter pads12 edge samples; run() explicit coefficients do not.
    cv::Mat padded;cv::copyMakeBorder(image,padded,12,12,12,12,cv::BORDER_REPLICATE);return spline_coefficients(padded);
}
static void weights(double coordinate,double* w){
    double t=coordinate-std::floor(coordinate),r=1-t;
    w[1]=(t*t*(t-2)*3+4)/6;w[2]=(r*r*(r-2)*3+4)/6;w[0]=r*r*r/6;w[3]=1-w[0]-w[1]-w[2];
}
double spline_sample(const cv::Mat& c,double x,double y){
    if(!std::isfinite(x)||!std::isfinite(y)||std::abs(x)>1e8||std::abs(y)>1e8)throw std::runtime_error("Invalid spline coordinate");
    double wx[4],wy[4],out=0;weights(x,wx);weights(y,wy);int x0=int(std::floor(x))-1,y0=int(std::floor(y))-1;
    for(int j=0;j<4;++j){auto row=c.ptr<double>(std::clamp(y0+j,0,c.rows-1));for(int i=0;i<4;++i)out+=row[std::clamp(x0+i,0,c.cols-1)]*wy[j]*wx[i];}
    return out;
}
}
