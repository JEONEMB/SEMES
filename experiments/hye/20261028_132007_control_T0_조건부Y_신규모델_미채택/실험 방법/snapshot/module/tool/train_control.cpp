#include "observer_internal.hpp"
#include <iostream>
#include <regex>
#include <algorithm>
#include <cmath>
using namespace alignment_detail;
int wmain(int argc,wchar_t** argv){try {
 cv::setNumThreads(1);
 if(argc!=4)throw std::runtime_error("train_control CLEAN_20261014 JOINT_ASSETS NEW_MODEL.yml");
 fs::path clean(argv[1]),assets(argv[2]),output(argv[3]);if(fs::exists(output))throw std::runtime_error("Refusing to overwrite existing model");
 initialize_alignment(assets.u8string());std::vector<fs::path> files{clean/"center.bmp"};
 for(const auto& e:fs::directory_iterator(clean/"Y"))if(e.path().extension()==".bmp")files.push_back(e.path());
 std::sort(files.begin()+1,files.end());if(files.size()!=81)throw std::runtime_error("Expected center1 plus Y-only80 with T fixed0");
 cv::Mat X(81,204,CV_64F),y(81,1,CV_64F);std::regex pattern("^Y([+-][0-9]+(?:\\.[0-9]+)?)$");
 for(int i=0;i<81;++i){std::smatch match;auto name=files[i].stem().u8string();double command=0;if(i){if(!std::regex_match(name,match,pattern))throw std::runtime_error("Invalid controlled-Y filename");command=std::stod(match[1]);}y.at<double>(i)=command;auto image=read_gray(files[i]);feature_map(observe(image).values).copyTo(X.row(i));if(i%10==0)std::cout<<"measured "<<i+1<<"/81\n";}
 cv::Mat A(81,2,CV_64F);A.col(0).setTo(1);y.copyTo(A.col(1));cv::Mat coef;
 if(!cv::solve(A,X,coef,cv::DECOMP_SVD))throw std::runtime_error("Conditional fit failed");
 cv::Mat origin=coef.row(0).clone(),response=coef.row(1),residual=X-A*coef,R=residual.clone(),V=response.t(),stddev(1,204,CV_64F);
 for(int k=0;k<204;++k){double sum=0;for(int i=0;i<81;++i)sum+=residual.at<double>(i,k)*residual.at<double>(i,k);double s=std::max(std::sqrt(sum/81),1e-6);stddev.at<double>(k)=s;R.col(k)/=s;V.at<double>(k)/=s;}
 double a=.01,b=.99/81;cv::Mat system=a*cv::Mat::eye(81,81,CV_64F)+b*R*R.t(),solved;
 if(!cv::solve(system,R*V,solved,cv::DECOMP_CHOLESKY))throw std::runtime_error("Covariance solve failed");
 cv::Mat Q=(V-b*R.t()*solved)/a,projector=Q.clone();for(int k=0;k<204;++k)projector.at<double>(k)/=stddev.at<double>(k);
 cv::Mat norm=response*projector;double denominator=norm.at<double>(0);if(!std::isfinite(denominator)||denominator<=0)throw std::runtime_error("Unobservable Y");projector/=denominator;
 cv::Mat centered=X.clone();for(int i=0;i<81;++i)centered.row(i)-=origin;cv::Mat error=cv::abs(centered*projector-y);double maximum;cv::minMaxLoc(error,nullptr,&maximum);
 fs::create_directories(output.parent_path());cv::FileStorage f(output.u8string(),cv::FileStorage::WRITE);
 f<<"origin"<<origin<<"projector"<<projector<<"minimum_y"<<-2.<<"maximum_y"<<2.<<"training_images"<<81<<"fixed_T"<<0.<<"shrink"<<.01<<"training_Y_MAE"<<cv::mean(error)[0]<<"training_Y_max"<<maximum<<"scope"<<"Controlled T0 development fit; not independent test or robot result";
 std::cout<<"Training reapplication Ymean="<<cv::mean(error)[0]<<" Ymax="<<maximum<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
