#include "t_features.hpp"
#include <algorithm>
#include <stdexcept>
namespace t_only {
Features enriched_features(const qrwithbar::detail::Observation& current,const qrwithbar::detail::Observation& reference){
 Features result{};std::copy(current.features.begin(),current.features.end(),result.begin());
 for(int q=0;q<3;q++){
  double mx=0,my=0;int n=0;
  for(int i=0;i<630;i++)if(reference.points[i].qr==q){mx+=reference.points[i].x;my+=reference.points[i].y;n++;}
  mx/=n;my/=n;cv::Mat a(n,3,CV_64F),b(n,2,CV_64F),coef;int r=0;
  for(int i=0;i<630;i++)if(reference.points[i].qr==q){a.at<double>(r,0)=1;a.at<double>(r,1)=(reference.points[i].x-mx)/100.;a.at<double>(r,2)=(reference.points[i].y-my)/100.;b.at<double>(r,0)=current.points[i].x-reference.points[i].x;b.at<double>(r,1)=current.points[i].y-reference.points[i].y;r++;}
  if(!cv::solve(a,b,coef,cv::DECOMP_SVD)||!cv::checkRange(coef))throw std::runtime_error("Invalid QR spatial slope fit");
  result[10+q*4]=coef.at<double>(1,0);result[11+q*4]=coef.at<double>(2,0);result[12+q*4]=coef.at<double>(1,1);result[13+q*4]=coef.at<double>(2,1);
 }return result;
}
}
