#include "observer_internal.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
namespace alignment_detail {
std::vector<double> line_profile(const cv::Mat& image,const cv::Rect& b){
    std::vector<double> p(49),result(49);for(int y=0;y<49;++y){auto row=image.ptr<unsigned char>(b.y+y)+b.x;for(int x=0;x<b.width;++x)p[y]+=row[x];p[y]/=b.width;}
    double kernel[9],sum=0;for(int k=-4;k<=4;++k)sum+=kernel[k+4]=std::exp(-.5*k*k);
    for(int i=0;i<49;++i)for(int k=-4;k<=4;++k){int j=i+k;while(j<0||j>=49)j=j<0?-j-1:98-j-1;result[i]+=p[j]*kernel[k+4]/sum;}return result;
}
LineFit match_line(const std::vector<double>& reference,const std::vector<double>& current){
    cv::Mat a=cv::Mat::zeros(49,49,CV_64F),rhs=cv::Mat::zeros(49,1,CV_64F),answer;
    a.at<double>(0,0)=1;a.at<double>(0,1)=-2;a.at<double>(0,2)=1;a.at<double>(48,46)=1;a.at<double>(48,47)=-2;a.at<double>(48,48)=1;
    for(int i=1;i<48;++i){a.at<double>(i,i-1)=1;a.at<double>(i,i)=4;a.at<double>(i,i+1)=1;rhs.at<double>(i)=6*(current[i+1]-2*current[i]+current[i-1]);}if(!cv::solve(a,rhs,answer,cv::DECOMP_LU))return {0,0,false};
    auto spline=[&](double x){double index=x+24;int k=std::clamp(int(std::floor(index)),0,47);double t=index-k,u=1-t;return u*current[k]+t*current[k+1]+((u*u*u-u)*answer.at<double>(k)+(t*t*t-t)*answer.at<double>(k+1))/6;};
    double target[21],mean=0,ss=0;for(int i=0;i<21;++i)mean+=(target[i]=reference[i+14]);mean/=21;for(double v:target)ss+=(v-mean)*(v-mean);
    struct Fit{double cost,gain,correlation,contrast;};
    auto evaluate=[&](double shift){double v[21],vm=0,vv=0,product=0;for(int i=0;i<21;++i)vm+=(v[i]=spline(i-10+shift));vm/=21;for(int i=0;i<21;++i){vv+=(v[i]-vm)*(v[i]-vm);product+=(target[i]-mean)*(v[i]-vm);}if(vv<1e-12)return Fit{1e9,0,0,0};double gain=product/vv,offset=mean-gain*vm,cost=0;for(int i=0;i<21;++i)cost+=std::pow(target[i]-gain*v[i]-offset,2);std::sort(v,v+21);return Fit{cost/21,gain,product/std::sqrt(std::max(ss*vv,1e-24)),v[18]-v[2]};};
    double best=-12;for(int i=-11;i<=12;++i)if(evaluate(i).cost<evaluate(best).cost)best=i;
    double low=std::max(-12.,best-1),high=std::min(12.,best+1),golden=.5*(3-std::sqrt(5.)),xf=low+golden*(high-low),nfc=xf,fulc=xf,fx=evaluate(xf).cost,fnfc=fx,ffulc=fx,rat=0,e=0;
    int count=1;bool success=true;
    while(true){double middle=.5*(low+high),tol1=std::sqrt(2.2e-16)*std::abs(xf)+1e-7/3,tol2=2*tol1;if(std::abs(xf-middle)<=tol2-.5*(high-low))break;
        bool g=true;if(std::abs(e)>tol1){double r=(xf-nfc)*(fx-ffulc),q=(xf-fulc)*(fx-fnfc),p=(xf-fulc)*q-(xf-nfc)*r;q=2*(q-r);if(q>0)p=-p;q=std::abs(q);r=e;e=rat;if(std::abs(p)<std::abs(.5*q*r)&&p>q*(low-xf)&&p<q*(high-xf)){rat=p/q;double x=xf+rat;g=false;if(x-low<tol2||high-x<tol2)rat=tol1*(middle-xf>=0?1:-1);}}
        if(g){e=xf>=middle?low-xf:high-xf;rat=golden*e;}double x=xf+(rat>=0?1:-1)*std::max(std::abs(rat),tol1),fu=evaluate(x).cost;++count;
        if(fu<=fx){if(x>=xf)low=xf;else high=xf;fulc=nfc;ffulc=fnfc;nfc=xf;fnfc=fx;xf=x;fx=fu;}else{if(x<xf)low=x;else high=x;if(fu<=fnfc||nfc==xf){fulc=nfc;ffulc=fnfc;nfc=x;fnfc=fu;}else if(fu<=ffulc||fulc==xf||fulc==nfc){fulc=x;ffulc=fu;}}
        if(count>=60){success=false;break;}
    }
    auto f=evaluate(xf);return {xf,f.correlation,success&&std::isfinite(xf)&&std::isfinite(f.cost)&&f.correlation>=.9&&f.contrast>=5&&f.gain>=.25&&f.gain<=4&&std::abs(xf)<=11.5};
}
}
