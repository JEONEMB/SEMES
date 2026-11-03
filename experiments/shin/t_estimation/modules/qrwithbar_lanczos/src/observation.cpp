#include "observation.hpp"
#include "reference_data.h"
#include "numeric.h"
#include <windows.h>
#include <bcrypt.h>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <limits>
extern "C" int qc_reference_profile(const qc_gray*,double,double,int,qc_edge_sample*);
extern "C" int qc_summarize_profile(const qc_edge_sample*,int,int,double*);
namespace qrwithbar { namespace detail {
cv::Mat read_gray(const fs::path& p){std::ifstream f(p,std::ios::binary);if(!f)throw std::runtime_error("Missing image: "+p.u8string());std::vector<unsigned char> bytes{std::istreambuf_iterator<char>(f),{}};auto im=cv::imdecode(bytes,cv::IMREAD_GRAYSCALE);if(im.empty())throw std::runtime_error("Decode error: "+p.u8string());return im;}
std::string sha256(const fs::path& p){std::ifstream f(p,std::ios::binary);std::vector<unsigned char> b{std::istreambuf_iterator<char>(f),{}};BCRYPT_ALG_HANDLE alg=nullptr;BCRYPT_HASH_HANDLE hash=nullptr;DWORD n=0,done=0;BCryptOpenAlgorithmProvider(&alg,BCRYPT_SHA256_ALGORITHM,nullptr,0);BCryptGetProperty(alg,BCRYPT_OBJECT_LENGTH,(PUCHAR)&n,sizeof(n),&done,0);std::vector<unsigned char> obj(n),digest(32);BCryptCreateHash(alg,&hash,obj.data(),n,nullptr,0,0);BCryptHashData(hash,b.data(),(ULONG)b.size(),0);BCryptFinishHash(hash,digest.data(),32,0);BCryptDestroyHash(hash);BCryptCloseAlgorithmProvider(alg,0);std::ostringstream out;out<<std::hex<<std::setfill('0');for(auto v:digest)out<<std::setw(2)<<(int)v;return out.str();}
Warps align_qrs(const cv::Mat& image,const fs::path& cal){Warps result;
 for(int q=0;q<3;q++){int x=qc_patch_bounds[q][0],y=qc_patch_bounds[q][1],w=204,h=216;auto templ=read_gray(cal/("template_"+std::to_string(q)+".png"));int left=std::max(0,x-300),top=std::max(0,y-120);auto search=image(cv::Rect(left,top,std::min(image.cols,x+w+300)-left,std::min(image.rows,y+h+120)-top));cv::Mat scores;cv::matchTemplate(search,templ,scores,cv::TM_CCOEFF_NORMED);double score;cv::Point loc;cv::minMaxLoc(scores,nullptr,&score,nullptr,&loc);if(score<.5)throw std::runtime_error("NCC failed");int cx=left+loc.x,cy=top+loc.y;auto patch=image(cv::Rect(cx,cy,w,h));cv::Mat warp=cv::Mat::eye(2,3,CV_32F);double corr=cv::findTransformECC(templ,patch,warp,cv::MOTION_AFFINE,cv::TermCriteria(3,100,1e-7),cv::noArray(),1);if(corr<.5)throw std::runtime_error("ECC failed");auto& a=result[q];for(int i=0;i<6;i++)a.val[i]=warp.at<float>(i/3,i%3);a(0,2)+=cx-a(0,0)*x-a(0,1)*y;a(1,2)+=cy-a(1,0)*x-a(1,1)*y;}
 return result;
}
Observation analytic_observe(const cv::Mat& image,const Warps& warps){Observation out;qc_gray gray{image.data,image.cols,image.rows,(int)image.step};int failures=0;int counts[3]={0,0,0};
 #pragma omp parallel for num_threads(4) reduction(+:failures)
 for(int i=0;i<630;i++){auto& s=qc_corner_specs[i];auto& a=warps[s.qr];double x=a(0,0)*s.seed[0]+a(0,1)*s.seed[1]+a(0,2),y=a(1,0)*s.seed[0]+a(1,1)*s.seed[1]+a(1,2),rmse=0;double basis[4]={a(0,0)*s.basis[0]+a(0,1)*s.basis[2],a(0,0)*s.basis[1]+a(0,1)*s.basis[3],a(1,0)*s.basis[0]+a(1,1)*s.basis[2],a(1,0)*s.basis[1]+a(1,1)*s.basis[3]};for(int j=0;j<2;j++){double n=std::hypot(basis[j],basis[j+2]);basis[j]/=n;basis[j+2]/=n;}for(int pass=0;pass<3;pass++){double nx,ny;if(qc_fit_corner(&gray,x,y,s.bits,basis,&nx,&ny,&rmse)!=0){failures++;break;}x=nx;y=ny;}out.points[i]={x,y,s.qr};}
 for(int i=0;i<630;i++){auto& s=qc_corner_specs[i];counts[s.qr]++;out.features[s.qr*2]+=out.points[i].x-s.reference[0];out.features[s.qr*2+1]+=out.points[i].y-s.reference[1];}
 for(int q=0;q<3;q++){out.features[q*2]/=counts[q];out.features[q*2+1]/=counts[q];}qc_result diagnostics{};int status=qc_extract_edges(&gray,out.points.data(),630,out.features.data()+6,out.edges.data(),&diagnostics);if(status!=0)failures++;out.failures=failures;return out;
}
static double sinc(double t){return fabs(t)<1e-8?1:sin(QC_PI*t)/(QC_PI*t);}
static double dsinc(double t){return fabs(t)<1e-6?-QC_PI*QC_PI*t/3:(QC_PI*t*cos(QC_PI*t)-sin(QC_PI*t))/(QC_PI*t*t);}
static void kernel(double t,int method,double& w,double& d){double r=fabs(t),sgn=t<0?-1:1;w=d=0;
 if(method!=2)throw std::invalid_argument("Only Lanczos-3 is supported");
 if(r<3){w=sinc(t)*sinc(t/3);d=dsinc(t)*sinc(t/3)+sinc(t)*dsinc(t/3)/3;}
}
static double interpolate(const cv::Mat& im,double x,double y,int method,double& gx,double& gy){int radius=3,ix=(int)floor(x),iy=(int)floor(y);double wx[6],wy[6],dx[6],dy[6],sx=0,sy=0,sdx=0,sdy=0;int n=2*radius;if(ix-radius+1<0||iy-radius+1<0||ix+radius>=im.cols||iy+radius>=im.rows){gx=gy=0;return std::numeric_limits<double>::quiet_NaN();}
 for(int k=0;k<n;k++){kernel(x-(ix-radius+1+k),method,wx[k],dx[k]);kernel(y-(iy-radius+1+k),method,wy[k],dy[k]);sx+=wx[k];sy+=wy[k];sdx+=dx[k];sdy+=dy[k];}
 for(int k=0;k<n;k++){dx[k]=(dx[k]*sx-wx[k]*sdx)/(sx*sx);dy[k]=(dy[k]*sy-wy[k]*sdy)/(sy*sy);wx[k]/=sx;wy[k]/=sy;}double v=0;gx=gy=0;
 for(int j=0;j<n;j++){const auto* row=im.ptr<unsigned char>(iy-radius+1+j);for(int i=0;i<n;i++){double pix=row[ix-radius+1+i];v+=pix*wx[i]*wy[j];gx+=pix*dx[i]*wy[j];gy+=pix*wx[i]*dy[j];}}return v;
}
double sample_lanczos_gray(const cv::Mat& image,double x,double y,double& gx,double& gy){return interpolate(image,x,y,2,gx,gy);}
struct Match {const cv::Mat* image;int method;bool edge;double cx,cy;std::vector<cv::Point2d> offsets;std::vector<double> targets;};
static double match_eval(const double* p,int i,void* v,double* jac){auto& c=*(Match*)v;double gx,gy,x=c.cx+c.offsets[i].x+(c.edge?0:p[0]),y=c.cy+c.offsets[i].y+(c.edge?p[0]:p[1]);double value=interpolate(*c.image,x,y,c.method,gx,gy);if(c.edge){jac[0]=gy;jac[1]=-c.targets[i];jac[2]=-1;return value-p[1]*c.targets[i]-p[2];}jac[0]=gx;jac[1]=gy;jac[2]=-c.targets[i];jac[3]=-1;return value-p[2]*c.targets[i]-p[3];}
Observation interpolated_observe(const cv::Mat& image,const cv::Mat& ref,const Warps& warps,const Warps& rw,const Observation& initial,const Observation& ro,int method){Observation out=initial;out.features.fill(0);int failures=0;cv::Matx22d transforms[3];for(int q=0;q<3;q++){auto a=warps[q],b=rw[q];transforms[q]=cv::Matx22d(a(0,0),a(0,1),a(1,0),a(1,1))*cv::Matx22d(b(0,0),b(0,1),b(1,0),b(1,1)).inv();}
 #pragma omp parallel for num_threads(4) reduction(+:failures)
 for(int i=0;i<630;i++){auto r=ro.points[i],s=initial.points[i];int ix=qc_round(r.x),iy=qc_round(r.y);Match c{&image,method,false,s.x,s.y,{},{}};for(int y=-4;y<=4;y++)for(int x=-4;x<=4;x++){cv::Vec2d delta=transforms[s.qr]*cv::Vec2d(ix+x-r.x,iy+y-r.y);c.offsets.push_back({delta[0],delta[1]});c.targets.push_back(ref.at<unsigned char>(iy+y,ix+x));}double p[4]={0,0,1,0},lo[4]={-3.6,-3.6,.2,-255},hi[4]={3.6,3.6,5,255};int ok=qc_optimize(p,4,81,lo,hi,0,match_eval,&c);if(!ok)failures++;out.points[i]={s.x+p[0],s.y+p[1],s.qr};}
 int counts[3]={};for(int i=0;i<630;i++){int q=out.points[i].qr;counts[q]++;out.features[q*2]+=out.points[i].x-ro.points[i].x;out.features[q*2+1]+=out.points[i].y-ro.points[i].y;}for(int q=0;q<3;q++){out.features[q*2]/=counts[q];out.features[q*2+1]/=counts[q];}
 qc_gray rg{ref.data,ref.cols,ref.rows,(int)ref.step};
 #pragma omp parallel for num_threads(4) reduction(+:failures)
 for(int i=0;i<120;i++){auto s=initial.edges[i];int edge=i/60,polarity=edge?-1:1;double ref_y=qc_reference_edges[edge*2]+qc_reference_edges[edge*2+1]*(s.x-ref.cols/2.)/(ref.cols/2.);qc_edge_sample rs;int good=qc_reference_profile(&rg,s.x,ref_y,polarity,&rs);int cy=qc_round(rs.y),ix=qc_round(s.x);Match c{&image,method,true,s.x,s.y,{},{}};for(int y=-7;y<=7;y++)for(int x=-3;x<=3;x++){c.offsets.push_back({(double)x,cy+y-rs.y});c.targets.push_back(ref.at<unsigned char>(cy+y,ix+x));}double p[3]={0,1,0},lo[3]={-4,.2,-255},hi[3]={4,5,255};int ok=qc_optimize(p,3,105,lo,hi,1,match_eval,&c);double ss=0;for(int k=0;k<105;k++){double j[6],r=match_eval(p,k,&c,j);ss+=r*r;}out.edges[i].y=s.y+p[0];out.edges[i].rmse=sqrt(ss/105);out.edges[i].valid=good&&s.valid&&ok&&fabs(p[0])<3.9&&out.edges[i].rmse<15;if(!ok)failures++;}
 for(int e=0;e<2;e++){if(!qc_summarize_profile(out.edges.data()+e*60,image.cols,e,out.features.data()+6+e*2))failures++;out.features[6+e*2]-=ro.features[6+e*2];out.features[7+e*2]-=ro.features[7+e*2];}out.failures=failures;return out;
}

void check_lanczos_kernel() {
 cv::Mat constant(128,128,CV_8U,cv::Scalar(117)),noise(128,128,CV_8U);
 for(int y=0;y<128;y++)for(int x=0;x<128;x++)noise.at<unsigned char>(y,x)=(unsigned char)((37*x+17*y+3*x*y)%256);
 for(int i=0;i<20;i++) {
  double x=50+.037*(i+.13),y=55+.023*(i+.17),gx,gy;
  if(fabs(interpolate(constant,x,y,2,gx,gy)-117)>1e-9)throw std::runtime_error("Lanczos constant reproduction");
  if(fabs(interpolate(noise,50,55,2,gx,gy)-noise.at<unsigned char>(55,50))>1e-9)throw std::runtime_error("Lanczos integer reproduction");
  interpolate(noise,x,y,2,gx,gy);double a,b,c,d,h=1e-5;
  double dx=(interpolate(noise,x+h,y,2,a,b)-interpolate(noise,x-h,y,2,c,d))/(2*h);
  double dy=(interpolate(noise,x,y+h,2,a,b)-interpolate(noise,x,y-h,2,c,d))/(2*h);
  if(fabs(dx-gx)>1e-5||fabs(dy-gy)>1e-5)throw std::runtime_error("Lanczos analytic gradient");
 }
}
} } // namespace qrwithbar::detail
