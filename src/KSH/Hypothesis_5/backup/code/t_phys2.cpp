#include "../src/camera_model.hpp"
#include <cstdio>
int main(){ mm::CameraModel c; cv::Point2d p=c.plate_center; cv::Point2d w2(2867,1357),w3(1010,1677);
 auto s=[&](double y,double t){return c.displacement(p,c.z_plate,{y,t}).x;};
 auto d=[&](double y,double t){return s(y,t)-0.5*(c.displacement(w2,c.z_bg,{y,t}).x+c.displacement(w3,c.z_bg,{y,t}).x);};
 double ds=(s(0,0.01)-s(0,-0.01))/0.02, dd=(d(0,0.01)-d(0,-0.01))/0.02, sy=(s(0.01,0)-s(-0.01,0))/0.02, dy=(d(0.01,0)-d(-0.01,0))/0.02;
 printf("sens s: Y %.3f T %.3f | d: Y %.3f T %.3f\n",sy,ds,dy,dd);
 for(double y:{0.05,0.5,1.0}) for(double t:{1.0,2.0}){ double es=s(y,t)-(sy*y+ds*t), ed=d(y,t)-(dy*y+dd*t);
   // invert linear model error
   double det=sy*dd-ds*dy; double ey=( dd*es - ds*ed)/det, et=(-dy*es + sy*ed)/det;
   printf("Y%.2f T%.1f nonlin s %.3f d %.3f -> est err Y %.1f um T %.4f deg\n",y,t,es,ed,ey*1000,et);}
}
