#pragma once
#include "numeric.hpp"
#include "workspace.hpp"
#include <windows.h>
namespace semes::lapack {
inline double pairwise_sum(const double* values,size_t count){
    if(count<8){double total=-0.;for(size_t i=0;i<count;++i)total+=values[i];return total;}
    if(count<=128){double r[8];for(int i=0;i<8;++i)r[i]=values[i];size_t i=8;for(;i+7<count;i+=8)for(int k=0;k<8;++k)r[k]+=values[i+k];double total=((r[0]+r[1])+(r[2]+r[3]))+((r[4]+r[5])+(r[6]+r[7]));for(;i<count;++i)total+=values[i];return total;}
    size_t half=count/2;half-=half%8;return pairwise_sum(values,half)+pairwise_sum(values+half,count-half);
}
inline HMODULE library(){static HMODULE handle=[](){auto directory=workspace_root()/"third_party/openblas";SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);AddDllDirectory(directory.c_str());for(auto& file:fs::directory_iterator(directory))if(file.path().filename().u8string().rfind("libscipy_openblas64_",0)==0){auto h=LoadLibraryExW(file.path().c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);if(h){auto set_threads=reinterpret_cast<void(*)(int)>(GetProcAddress(h,"scipy_openblas_set_num_threads64_"));if(!set_threads)throw std::runtime_error("Pinned OpenBLAS thread control missing");set_threads(1);return h;}}throw std::runtime_error("Pinned native OpenBLAS runtime unavailable");}();return handle;}
struct SVD{cv::Mat u,w,vt;};
inline SVD svd(const cv::Mat& input){
    using Function=long long(*)(int,char,long long,long long,double*,long long,double*,double*,long long,double*,long long);
    auto function=reinterpret_cast<Function>(GetProcAddress(library(),"scipy_LAPACKE_dgesdd64_"));if(!function)throw std::runtime_error("Native LAPACKE dgesdd missing");
    auto a=input.clone();int k=std::min(a.rows,a.cols);SVD r{cv::Mat(a.rows,k,CV_64F),cv::Mat(k,1,CV_64F),cv::Mat(k,a.cols,CV_64F)};
    auto status=function(101,'S',a.rows,a.cols,a.ptr<double>(),a.cols,r.w.ptr<double>(),r.u.ptr<double>(),k,r.vt.ptr<double>(),a.cols);if(status)throw std::runtime_error("Native LAPACK SVD failed: "+std::to_string(status));return r;
}
inline cv::Mat least_squares(const cv::Mat& a,const cv::Mat& b,double relative_cutoff=-1){
    using Function=long long(*)(int,long long,long long,long long,double*,long long,double*,long long,double*,double,long long*);
    auto function=reinterpret_cast<Function>(GetProcAddress(library(),"scipy_LAPACKE_dgelsd64_"));if(!function)throw std::runtime_error("Native LAPACKE dgelsd missing");
    auto copy=a.clone();cv::Mat rhs=cv::Mat::zeros(std::max(a.rows,a.cols),b.cols,CV_64F);b.copyTo(rhs.rowRange(0,b.rows));std::vector<double> singular(std::min(a.rows,a.cols));long long rank=0;
    auto status=function(101,a.rows,a.cols,b.cols,copy.ptr<double>(),a.cols,rhs.ptr<double>(),b.cols,singular.data(),relative_cutoff<0?std::numeric_limits<double>::epsilon()*std::max(a.rows,a.cols):relative_cutoff,&rank);if(status)throw std::runtime_error("Native LAPACK least squares failed");return rhs.rowRange(0,a.cols).clone();
}
inline cv::Mat solve(const cv::Mat& a,const cv::Mat& b){using Function=long long(*)(int,long long,long long,double*,long long,long long*,double*,long long);auto f=reinterpret_cast<Function>(GetProcAddress(library(),"scipy_LAPACKE_dgesv64_"));if(!f||a.rows!=a.cols||a.rows!=b.rows)throw std::runtime_error("Invalid native linear solve");auto copy=a.clone(),rhs=b.clone();std::vector<long long> pivots(a.rows);auto status=f(101,a.rows,b.cols,copy.ptr<double>(),a.cols,pivots.data(),rhs.ptr<double>(),b.cols);if(status)throw std::runtime_error("Native linear solve failed");return rhs;}
inline cv::Mat pinv(const cv::Mat& a,double cutoff=1e-15){auto s=svd(a);cv::Mat diagonal=cv::Mat::zeros(s.w.rows,s.w.rows,CV_64F);for(int i=0;i<s.w.rows;++i)if(s.w.at<double>(i)>cutoff*s.w.at<double>(0))diagonal.at<double>(i,i)=1/s.w.at<double>(i);return s.vt.t()*diagonal*s.u.t();}
inline cv::Mat multiply(const cv::Mat& a,const cv::Mat& b){using Function=void(*)(int,int,int,long long,long long,long long,double,const double*,long long,const double*,long long,double,double*,long long);auto f=reinterpret_cast<Function>(GetProcAddress(library(),"scipy_cblas_dgemm64_"));if(!f||a.cols!=b.rows||a.type()!=CV_64F||b.type()!=CV_64F)throw std::runtime_error("Invalid native BLAS matrix product");auto left=a.clone(),right=b.clone();cv::Mat result(a.rows,b.cols,CV_64F);f(101,111,111,a.rows,b.cols,a.cols,1.,left.ptr<double>(),a.cols,right.ptr<double>(),b.cols,0.,result.ptr<double>(),b.cols);return result;}
}
