#include "six.hpp"
#include <iostream>
extern "C" __declspec(dllexport) int v3_feature_extract(const unsigned char* pixels,int rows,int cols,const wchar_t* assets_path,double* output){try{
 cv::setNumThreads(1);if(rows!=2160||cols!=3840)return 1;
 cv::Mat image(rows,cols,CV_8U,const_cast<unsigned char*>(pixels));std::filesystem::path assets(assets_path);
 auto reference=qrwithbar::detail::read_gray(assets/"reference.png");six::Frame r;
 r.warps=qrwithbar::detail::align_qrs(reference,assets);r.observation=qrwithbar::detail::analytic_observe(reference,r.warps);
 six::Frame frame;frame.warps=qrwithbar::detail::align_qrs(image,assets);auto initial=qrwithbar::detail::analytic_observe(image,frame.warps);
 if(r.observation.failures||initial.failures)return 2;
 frame.observation=qrwithbar::detail::interpolated_observe(image,reference,frame.warps,r.warps,initial,r.observation,2);
 if(frame.observation.failures)return 3;
 cv::Mat x=six::describe(frame,r);for(int j=0;j<104;j++)output[j]=x.at<double>(j);return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 4;}}
