#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include "../../qrwithbar-lanczos-src/modules/qrwithbar_lanczos/src/numeric.h"
struct Gray {const unsigned char* p;int w,h;};
static double sinc(double t){return std::abs(t)<1e-8?1.:std::sin(QC_PI*t)/(QC_PI*t);}
static double dsinc(double t){return std::abs(t)<1e-6?-QC_PI*QC_PI*t/3:(QC_PI*t*std::cos(QC_PI*t)-std::sin(QC_PI*t))/(QC_PI*t*t);}
static double sample(const Gray& im,double x,double y,double& gx,double& gy){
 int ix=(int)std::floor(x),iy=(int)std::floor(y);gx=gy=0;
 if(ix<2||iy<2||ix+3>=im.w||iy+3>=im.h)return std::numeric_limits<double>::quiet_NaN();
 double wx[6],wy[6],dx[6],dy[6],sx=0,sy=0,sdx=0,sdy=0;
 for(int k=0;k<6;k++){double tx=x-(ix-2+k),ty=y-(iy-2+k);
  wx[k]=sinc(tx)*sinc(tx/3);wy[k]=sinc(ty)*sinc(ty/3);
  dx[k]=dsinc(tx)*sinc(tx/3)+sinc(tx)*dsinc(tx/3)/3;
  dy[k]=dsinc(ty)*sinc(ty/3)+sinc(ty)*dsinc(ty/3)/3;
  sx+=wx[k];sy+=wy[k];sdx+=dx[k];sdy+=dy[k];}
 for(int k=0;k<6;k++){dx[k]=(dx[k]*sx-wx[k]*sdx)/(sx*sx);dy[k]=(dy[k]*sy-wy[k]*sdy)/(sy*sy);wx[k]/=sx;wy[k]/=sy;}
 double v=0;for(int j=0;j<6;j++)for(int i=0;i<6;i++){double pix=im.p[(iy-2+j)*im.w+ix-2+i];v+=pix*wx[i]*wy[j];gx+=pix*dx[i]*wy[j];gy+=pix*wx[i]*dy[j];}return v;
}
struct Context {Gray im;double cx,cy;std::vector<double> target,ox,oy;};
static double evaluate(const double* p,int i,void* ptr,double* jac){auto& c=*(Context*)ptr;double gx,gy;
 double v=sample(c.im,c.cx+c.ox[i]+p[0],c.cy+c.oy[i]+p[1],gx,gy);
 jac[0]=gx;jac[1]=gy;jac[2]=-c.target[i];jac[3]=-1;return v-p[2]*c.target[i]-p[3];}
extern "C" __declspec(dllexport) int fit_batch(const unsigned char* reference,const unsigned char* current,int w,int h,const double* reference_xy,const double* initial_xy,int n,double* output){
 Gray ref{reference,w,h},im{current,w,h};
 #pragma omp parallel for num_threads(4)
 for(int k=0;k<n;k++){
  Context c{im,initial_xy[k*2],initial_xy[k*2+1],{},{},{}};bool finite=true;double sum=0,sum2=0;
  for(int y=-20;y<=20;y+=2)for(int x=-20;x<=20;x+=2){double gx,gy,v=sample(ref,reference_xy[k*2]+x,reference_xy[k*2+1]+y,gx,gy);finite=finite&&std::isfinite(v);c.ox.push_back(x);c.oy.push_back(y);c.target.push_back(v);sum+=v;sum2+=v*v;}
  double stddev=std::sqrt(std::max(0.,sum2/c.target.size()-std::pow(sum/c.target.size(),2)));
  double p[4]={0,0,1,0},lo[4]={-4,-4,.2,-255},hi[4]={4,4,5,255};
  int ok=finite&&stddev>3?qc_optimize(p,4,(int)c.target.size(),lo,hi,1,evaluate,&c):0;
  double ss=0;for(int i=0;i<(int)c.target.size();i++){double jac[6],r=evaluate(p,i,&c,jac);ss+=r*r;}
  output[k*8]=c.cx+p[0];output[k*8+1]=c.cy+p[1];output[k*8+2]=std::sqrt(ss/c.target.size());
  output[k*8+3]=stddev;output[k*8+4]=p[2];output[k*8+5]=p[3];output[k*8+6]=ok;output[k*8+7]=std::max(std::abs(p[0]),std::abs(p[1]));
 }
 return 0;
}
extern "C" __declspec(dllexport) int check_kernel(){
 std::vector<unsigned char> pix(128*128);for(int y=0;y<128;y++)for(int x=0;x<128;x++)pix[y*128+x]=(37*x+17*y+3*x*y)%256;
 Gray im{pix.data(),128,128};double gx,gy,unused1,unused2;
 for(int i=0;i<20;i++){double x=50+.037*(i+.13),y=55+.023*(i+.17),h=1e-5;sample(im,x,y,gx,gy);
  double dx=(sample(im,x+h,y,unused1,unused2)-sample(im,x-h,y,unused1,unused2))/(2*h);
  double dy=(sample(im,x,y+h,unused1,unused2)-sample(im,x,y-h,unused1,unused2))/(2*h);
  if(std::abs(dx-gx)>1e-5||std::abs(dy-gy)>1e-5)return 1;
  if(std::abs(sample(im,50,55,unused1,unused2)-pix[55*128+50])>1e-9)return 2;
 }
 return 0;
}
extern "C" __declspec(dllexport) int quality_batch(const unsigned char* reference,const unsigned char* current,int w,int h,const double* reference_xy,const double* current_xy,const double* rmse,int n,double* output){
 Gray ref{reference,w,h},im{current,w,h};
 #pragma omp parallel for num_threads(4)
 for(int k=0;k<n;k++){
  double count=0,sT=0,sTT=0,sx=0,sy=0,sxx=0,sxy=0,syy=0,sTx=0,sTy=0;bool good=true;
  for(int y=-20;y<=20;y+=2)for(int x=-20;x<=20;x+=2){
   double unused1,unused2,gx,gy,T=sample(ref,reference_xy[k*2]+x,reference_xy[k*2+1]+y,unused1,unused2);
   double v=sample(im,current_xy[k*2]+x,current_xy[k*2+1]+y,gx,gy);
   good=good&&std::isfinite(T)&&std::isfinite(v)&&std::isfinite(gx)&&std::isfinite(gy);
   count++;sT+=T;sTT+=T*T;sx+=gx;sy+=gy;sxx+=gx*gx;sxy+=gx*gy;syy+=gy*gy;sTx+=T*gx;sTy+=T*gy;
  }
  double tt=sTT-sT*sT/count,tx=sTx-sT*sx/count,ty=sTy-sT*sy/count;
  double a=sxx-sx*sx/count-tx*tx/std::max(tt,1e-9),b=sxy-sx*sy/count-tx*ty/std::max(tt,1e-9),c=syy-sy*sy/count-ty*ty/std::max(tt,1e-9);
  double disc=std::sqrt(std::max(0.,(a-c)*(a-c)+4*b*b));double low=(a+c-disc)/2,high=(a+c+disc)/2,det=a*c-b*b;
  double variance=std::pow(std::max(1.,rmse[k]),2);
  if(!good||low<=1e-8||det<=1e-12){for(int j=0;j<8;j++)output[k*8+j]=std::numeric_limits<double>::quiet_NaN();continue;}
  output[k*8]=variance*c/det;output[k*8+1]=-variance*b/det;output[k*8+2]=variance*a/det;
  output[k*8+3]=low;output[k*8+4]=high/low;output[k*8+5]=std::sqrt(variance/low);output[k*8+6]=std::sqrt(variance/high);
  output[k*8+7]=.5*std::atan2(2*b,a-c);
 }
 return 0;
}
