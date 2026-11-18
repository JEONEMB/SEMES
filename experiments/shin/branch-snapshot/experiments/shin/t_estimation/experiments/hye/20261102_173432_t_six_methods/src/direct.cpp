#include "six.hpp"
#include "reference_data.h"
#include <algorithm>
#include <cmath>
#include <limits>
extern "C" int qc_reference_profile(const qc_gray*,double,double,int,qc_edge_sample*);
namespace six {
namespace {
struct Pixel {double x,y,target,u,reliability;int ix,iy,group;};
struct Evaluation {double cost=1e300;cv::Mat h,g;};
}
Prediction direct_refine(const Model& model,const Row& row,const Frame& frame,const Frame& ref,const cv::Mat& image,const cv::Mat& reference){
 auto initial=predict(model,row.x,row.time);std::vector<Pixel> pixels;pixels.reserve(18300);double bounds[3][4]={{1e9,1e9,-1e9,-1e9},{1e9,1e9,-1e9,-1e9},{1e9,1e9,-1e9,-1e9}};
 for(auto p:ref.observation.points){auto& b=bounds[p.qr];b[0]=std::min(b[0],p.x);b[1]=std::min(b[1],p.y);b[2]=std::max(b[2],p.x);b[3]=std::max(b[3],p.y);}
 cv::Matx22d shape[3];for(int q=0;q<3;q++){auto a=frame.warps[q],b=ref.warps[q];shape[q]=cv::Matx22d(a(0,0),a(0,1),a(1,0),a(1,1))*cv::Matx22d(b(0,0),b(0,1),b(1,0),b(1,1)).inv();}
 for(int i=0;i<630;i++){auto r=ref.observation.points[i],s=frame.observation.points[i];auto& b=bounds[r.qr];int col=std::min(2,(int)(3*(r.x-b[0])/(b[2]-b[0]+1e-9))),rr=std::min(2,(int)(3*(r.y-b[1])/(b[3]-b[1]+1e-9))),base=10+r.qr*18+2*(rr*3+col),rx=cvRound(r.x),ry=cvRound(r.y);
  for(int dy=-4;dy<=4;dy+=2)for(int dx=-4;dx<=4;dx+=2){auto offset=shape[r.qr]*cv::Vec2d(rx+dx-r.x,ry+dy-r.y);pixels.push_back({s.x+offset[0],s.y+offset[1],(double)reference.at<unsigned char>(ry+dy,rx+dx),0,1.,base,base+1,i});}
 }
 qc_gray gray{reference.data,reference.cols,reference.rows,(int)reference.step};
 int lowQuality=0;for(int i=0;i<120;i++){auto s=frame.observation.edges[i];int e=i/60,side=i%60/30,base=64+e*4+side*2;double u=(s.x-1920)/1920.,prior=qc_reference_edges[e*2]+ref.observation.features[6+e*2]+(qc_reference_edges[e*2+1]+ref.observation.features[7+e*2])*u;qc_edge_sample rs{};int good=qc_reference_profile(&gray,s.x,prior,e?-1:1,&rs);if(!good||!s.valid)lowQuality++;if(!good)rs.y=prior;int rx=cvRound(s.x),ry=cvRound(rs.y);
  for(int dy=-3;dy<=3;dy++)for(int dx=-2;dx<=2;dx+=2)pixels.push_back({s.x+dx,s.y+ry+dy-rs.y,(double)reference.at<unsigned char>(ry+dy,rx+dx),u,(good&&s.valid)?1.:.02,base,base+1,630+i});
 }
 auto evaluate=[&](cv::Vec2d state){
  Evaluation out;int n=(int)pixels.size();cv::Mat expected=forward(model,state),derivative(2,expected.cols,CV_64F);
  for(int k=0;k<2;k++){auto a=state,b=state;a[k]+=1e-4;b[k]-=1e-4;cv::Mat d=(forward(model,a)-forward(model,b))/(2e-4);d.copyTo(derivative.row(k));}
  std::vector<double> value(n),j0(n),j1(n);double sums[750][5]={};
  for(int i=0;i<n;i++){auto& p=pixels[i];double x=p.x,y=p.y,dx[2]={},dy[2]={};if(p.ix<64){x+=expected.at<double>(p.ix)-row.x.at<double>(p.ix);y+=expected.at<double>(p.iy)-row.x.at<double>(p.iy);for(int k=0;k<2;k++){dx[k]=derivative.at<double>(k,p.ix);dy[k]=derivative.at<double>(k,p.iy);}}
   else {y+=expected.at<double>(p.ix)-row.x.at<double>(p.ix)+p.u*(expected.at<double>(p.iy)-row.x.at<double>(p.iy));for(int k=0;k<2;k++)dy[k]=derivative.at<double>(k,p.ix)+p.u*derivative.at<double>(k,p.iy);}
   double gx,gy;value[i]=qrwithbar::detail::sample_lanczos_gray(image,x,y,gx,gy);if(!std::isfinite(value[i]))return out;j0[i]=gx*dx[0]+gy*dy[0];j1[i]=gx*dx[1]+gy*dy[1];auto& s=sums[p.group];s[0]+=p.reliability;s[1]+=p.reliability*p.target;s[2]+=p.reliability*value[i];s[3]+=p.reliability*p.target*p.target;s[4]+=p.reliability*p.target*value[i];
  }
  double gain[750],offset[750],meanTarget[750],varTarget[750],jm[750][2]={},jt[750][2]={};
  for(int g=0;g<750;g++){auto& s=sums[g];meanTarget[g]=s[1]/s[0];varTarget[g]=std::max(1e-6,s[3]-s[1]*s[1]/s[0]);gain[g]=std::max(.2,std::min(5.,(s[4]-s[1]*s[2]/s[0])/varTarget[g]));offset[g]=s[2]/s[0]-gain[g]*meanTarget[g];}
  for(int i=0;i<n;i++){auto& p=pixels[i];jm[p.group][0]+=p.reliability*j0[i];jm[p.group][1]+=p.reliability*j1[i];jt[p.group][0]+=p.reliability*(p.target-meanTarget[p.group])*j0[i];jt[p.group][1]+=p.reliability*(p.target-meanTarget[p.group])*j1[i];}
  out.h=cv::Mat::zeros(2,2,CV_64F);out.g=cv::Mat::zeros(2,1,CV_64F);out.cost=0;
  for(int i=0;i<n;i++){auto& p=pixels[i];int g=p.group;double residual=value[i]-gain[g]*p.target-offset[g],a=std::abs(residual),w=p.reliability*std::min(1.,3./std::max(1e-12,a));double j[2]={j0[i]-jm[g][0]/sums[g][0]-(p.target-meanTarget[g])*jt[g][0]/varTarget[g],j1[i]-jm[g][1]/sums[g][0]-(p.target-meanTarget[g])*jt[g][1]/varTarget[g]};
   out.cost+=p.reliability*(a<=3?residual*residual:6*a-9);for(int k=0;k<2;k++){out.g.at<double>(k)+=w*j[k]*residual;for(int l=0;l<2;l++)out.h.at<double>(k,l)+=w*j[k]*j[l];}
  }out.cost/=n;return out;
 };
 cv::Vec2d state=initial.latent;auto current=evaluate(state);if(!std::isfinite(current.cost)||current.cost>=1e299)throw std::runtime_error("Direct initialization outside image");double damping=.001;
 for(int iter=0;iter<15;iter++){auto h=current.h.clone();for(int k=0;k<2;k++)h.at<double>(k,k)+=damping*std::max(1.,h.at<double>(k,k));cv::Mat step;if(!cv::solve(h,-current.g,step,cv::DECOMP_SVD))break;cv::Vec2d delta(step.at<double>(0),step.at<double>(1));double limit=std::max(1.,std::max(std::abs(delta[0])/.4,std::abs(delta[1])/.2));delta/=limit;auto trial=evaluate(state+delta);if(trial.cost<current.cost){state+=delta;current=trial;damping=std::max(1e-8,damping/3);if(cv::norm(delta)<1e-7)break;}else{damping*=10;if(damping>1e8)break;}}
 Prediction result;result.t=state[1];result.latent=state;result.cost=current.cost;result.low_quality_line_samples=lowQuality;return result;
}
}
