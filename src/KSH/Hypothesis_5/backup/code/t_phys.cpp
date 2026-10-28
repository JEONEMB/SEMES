#include "../src/camera_model.hpp"
#include <cstdio>
int main(){ mm::CameraModel c; cv::Point2d p=c.plate_center; cv::Point2d w(1010,1677);
 auto s=[&](double y,double t){return c.displacement(p,c.z_plate,{y,t}).x;};
 auto d=[&](double y,double t){return s(y,t)-c.displacement(w,c.z_bg,{y,t}).x;};
 // fit quadratic on +-1 deg (y=0) and evaluate at 2
 double A[3][3]={0},B[3]={0}; for(int i=-100;i<=100;i++){double t=i*0.01,v=s(0,t),r[3]={1,t,t*t}; for(int a=0;a<3;a++){B[a]+=r[a]*v; for(int b=0;b<3;b++)A[a][b]+=r[a]*r[b];}}
 // solve 3x3
 cv::Matx33d M(A[0][0],A[0][1],A[0][2],A[1][0],A[1][1],A[1][2],A[2][0],A[2][1],A[2][2]); cv::Vec3d q=M.inv()*cv::Vec3d(B[0],B[1],B[2]);
 for(double t: {1.0,2.0}) printf("theta %.1f exact s %.4f quadfit %.4f diff %.4f px (= %.4f deg)\n",t,s(0,t),q[0]+q[1]*t+q[2]*t*t,s(0,t)-(q[0]+q[1]*t+q[2]*t*t),(s(0,t)-(q[0]+q[1]*t+q[2]*t*t))/80.06);
 for(double y: {0.05,0.5,1.0}) for(double t:{1.0,2.0}) printf("cross Y%.2f T%.1f : s %.4f  s(Y)+s(T) %.4f  diff %.4f | d diff %.4f\n",y,t,s(y,t),s(y,0)+s(0,t),s(y,t)-s(y,0)-s(0,t),d(y,t)-d(y,0)-d(0,t));
}
