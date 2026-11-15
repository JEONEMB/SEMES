#include "qr_pose.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace qr {
namespace {
constexpr double sx=340., sy=88.;

double clamp(double a,double min,double max) {return std::min(std::max(a,min),max);}

// Cubic B-spline prefilter. Mirror conditions double the first and last
// off-diagonal weights in the one-dimensional interpolation equations.
void spline_rows(cv::Mat& image) {
    const int n=image.cols;
    std::vector<double> upper(n),diag(n);
    diag[0]=4.;upper[0]=2./4.;
    for (int i=1;i<n;++i) {
        const double lower=i==n-1?2.:1.;
        diag[i]=4.-lower*upper[i-1];upper[i]=1./diag[i];
    }
    for (int row=0;row<image.rows;++row) {
        auto data=image.ptr<double>(row);
        data[0]*=6./diag[0];
        for (int i=1;i<n;++i) data[i]=(6.*data[i]-(i==n-1?2.:1.)*data[i-1])/diag[i];
        for (int i=n-2;i>=0;--i) data[i]-=upper[i]*data[i+1];
    }
}

std::array<double,4> spline_weight(double x) {
    const double a=1.-x,b=x*x,d=b*x;
    return {{a*a*a/6.,(4.-6.*b+3.*d)/6.,(1.+3.*x+3.*b-3.*d)/6.,d/6.}};
}

double sample(const cv::Mat& coefficients,double x,double y) {
    if (!(x>=1 && x<coefficients.cols-2 && y>=1 && y<coefficients.rows-2))
        throw std::runtime_error("Joint QR warp left the search crop");
    const auto ix=int(std::floor(x)),iy=int(std::floor(y));
    const auto wx=spline_weight(x-ix),wy=spline_weight(y-iy);
    double value=0.;
    for(int r=0;r<4;++r) {
        const auto col=coefficients.ptr<double>(iy+r-1)+ix-1;
        value+=wy[r]*(wx[0]*col[0]+wx[1]*col[1]+wx[2]*col[2]+wx[3]*col[3]);
    }
    return value;
}

struct Pixels { cv::Mat points,intensity; };
Pixels samples(const cv::Mat& reference,const std::vector<cv::Point>& qr_centers) {
    Pixels out{cv::Mat(3072,2,CV_64F),cv::Mat(3,1024,CV_64F)};
    auto cut=[](int index,int length) {return 3+index*(length/8)+std::min(index,length%8);};
    for(int i=0;i<3;++i) {
        const auto p=qr_centers[i]-cv::Point(74,80);
        cv::Mat region;
        reference(cv::Rect(p.x,p.y,149,161)).convertTo(region,CV_64F);
        cv::GaussianBlur(region,region,{0,0},.75);
        int k=0;
        for(int row=0;row<8;++row) for(int col=0;col<8;++col) {
            std::vector<std::pair<double,cv::Point>> block;
            for(int y=cut(row,155);y<cut(row+1,155);++y) for(int x=cut(col,143);x<cut(col+1,143);++x) {
                const double gx=(region.at<double>(y,x+1)-region.at<double>(y,x-1))*.5;
                const double gy=(region.at<double>(y+1,x)-region.at<double>(y-1,x))*.5;
                block.emplace_back(gx*gx+gy*gy,cv::Point(x,y));
            }
            std::stable_sort(block.begin(),block.end(),[](const auto& a,const auto& b){return a.first<b.first;});
            for(auto it=block.end()-16;it!=block.end();++it,++k) {
                const auto q=it->second;
                out.points.at<double>(i*1024+k,0)=p.x+q.x;out.points.at<double>(i*1024+k,1)=p.y+q.y;
                out.intensity.at<double>(i,k)=region.at<double>(q);
            }
        }
    }
    return out;
}
}

DirectPose::DirectPose(const std::string& path,const std::string& domain,const cv::Mat& reference) {
    const auto file=open_model(path);
    if (std::string(file["domain"])!=domain) throw std::runtime_error("Joint model domain mismatch");
    centers_=centers(matrix(file,"centers"));coefficients_=matrix(file,"coefficients");
    initializer_=matrix(file,"initializer");intercept_=matrix(file,"intercept");
    const auto model_limit=matrix(file,"limit");limit_={model_limit.at<double>(0),model_limit.at<double>(1)};
    if (coefficients_.size()!=cv::Size(8,6) || initializer_.size()!=cv::Size(2,4) || intercept_.total()!=2)
        throw std::runtime_error("Invalid joint model coefficients");
    for (const auto p:centers_) origin_+=cv::Point2d(p);
    origin_*=1./3.;
    cv::Mat points;
    if (reference.empty()) {
        if (domain=="rig") throw std::runtime_error("Rig requires a center image");
        geometry_=Geometry(file,centers_);points=matrix(file,"points");intensity_=matrix(file,"intensity");
    } else {
        geometry_=Geometry(reference,centers_);
        const auto px=samples(reference,centers_);points=px.points;intensity_=px.intensity;
    }
    if (points.size()!=cv::Size(2,3072) || intensity_.size()!=cv::Size(1024,3))
        throw std::runtime_error("Invalid direct QR samples");
    for(int i=0;i<points.rows;++i)
        normalized_.emplace_back((points.at<double>(i,0)-origin_.x)/sx,(points.at<double>(i,1)-origin_.y)/sy);
    for(int i=0;i<3;++i) {
        const double m=cv::mean(intensity_.row(i))[0];
        intensity_.row(i)-=m;energy_[i]=intensity_.row(i).dot(intensity_.row(i));
        if (!(energy_[i]>0.)) throw std::runtime_error("Empty QR photometric reference");
    }
}

Pose DirectPose::initial(const cv::Mat& frame) const {
    const auto x=geometry_.predict(frame);
    const cv::Mat a=cv::Mat(x).t()*initializer_;
    return {a.at<double>(0)+intercept_.at<double>(0),a.at<double>(1)+intercept_.at<double>(1)};
}

bool DirectPose::ssafy_supported(Pose p) {return p[0]>=-.05 && p[0]<=.15 && p[1]>.20 && p[1]<=.82;}

Fit DirectPose::refine(const cv::Mat& frame,Pose initial) const {
    check_frame(frame);
    const auto shift=geometry_.seed(frame);
    int minx=centers_[0].x,miny=centers_[0].y,maxx=minx,maxy=miny;
    for(const auto p:centers_) {minx=std::min(minx,p.x);maxx=std::max(maxx,p.x);miny=std::min(miny,p.y);maxy=std::max(maxy,p.y);}
    const cv::Rect roi(minx+shift.x-113,miny+shift.y-112,maxx-minx+227,maxy-miny+225);
    cv::Mat pixels,tmp;
    frame(roi).convertTo(pixels,CV_64F);cv::GaussianBlur(pixels,pixels,{0,0},.75);
    spline_rows(pixels);cv::transpose(pixels,tmp);spline_rows(tmp);cv::transpose(tmp,pixels);
    auto residual=[&](const cv::Vec3d& pose){
        const double y=pose[0],t=pose[1];
        const std::array<double,6> basis{{1.,y,t,y*y,y*t,t*t}};
        std::array<double,8> a{};
        for(int i=0;i<6;++i) for(int j=0;j<8;++j) a[j]+=basis[i]*coefficients_.at<double>(i,j);
        std::vector<double> r(3073);
        for(int q=0;q<3;++q) {
            double mean=0.,product=0.;
            for(int i=q*1024;i<(q+1)*1024;++i) {
                const auto p=normalized_[i];
                const double d=1.+p.x*a[6]+p.y*a[7];
                const double x=(p.x+a[0]+p.x*a[1]+p.y*a[2])*sx/d+origin_.x-roi.x;
                const double y=(p.y+a[3]+p.x*a[4]+p.y*a[5])*sy/d+origin_.y-roi.y+pose[2];
                r[i]=sample(pixels,x,y);mean+=r[i];
            }
            mean/=1024.;
            const auto templ=intensity_.ptr<double>(q);
            for(int i=0;i<1024;++i) {
                r[q*1024+i]-=mean;product+=r[q*1024+i]*templ[i];
            }
            const double gain=std::max(.4,product/energy_[q]);
            for(int i=0;i<1024;++i) r[q*1024+i]=r[q*1024+i]/gain-templ[i];
        }
        r.back()=.5*pose[2];return r;
    };

    cv::Vec3d pose(clamp(initial[0],-limit_[0]+.001,limit_[0]-.001),
                   clamp(initial[1],-limit_[1]+.001,limit_[1]-.001),0.);
    const cv::Vec3d low(std::max(pose[0]-.9,-limit_[0]),std::max(pose[1]-.16,-limit_[1]),-4.);
    const cv::Vec3d high(std::min(pose[0]+.9,limit_[0]),std::min(pose[1]+.16,limit_[1]),4.);
    const cv::Vec3d h(1e-4,2e-5,1e-3);
    auto cost=[](const std::vector<double>& r){double out=0.;for(auto v:r) out+=std::sqrt(1.+v*v/9.)-1.;return out;};
    auto error=residual(pose);double value=cost(error),lambda=.003;
    int iteration=0;
    for(;iteration<24;++iteration) {
        cv::Matx33d normal=cv::Matx33d::zeros();cv::Vec3d gradient(0.,0.,0.);
        std::array<std::vector<double>,3> jac;
        for(int j=0;j<3;++j) {
            auto step=pose;step[j]+=h[j];jac[j]=residual(step);
            for(size_t i=0;i<error.size();++i) jac[j][i]=(jac[j][i]-error[i])/h[j];
        }
        for(size_t i=0;i<error.size();++i) {
            const double weight=1./std::sqrt(1.+error[i]*error[i]/9.);
            for(int j=0;j<3;++j) {
                gradient[j]+=weight*jac[j][i]*error[i];
                for(int k=0;k<3;++k) normal(j,k)+=weight*jac[j][i]*jac[k][i];
            }
        }
        for(int j=0;j<3;++j) normal(j,j)+=lambda*(normal(j,j)+1e-8);
        cv::Vec3d step=normal.inv(cv::DECOMP_SVD)*(-gradient);
        auto next=pose+step;
        for(int j=0;j<3;++j) next[j]=clamp(next[j],low[j],high[j]);
        if (cv::norm(next-pose)<1e-8) break;
        auto trial=residual(next);const double score=cost(trial);
        if (score<value) {
            const double change=cv::norm(next-pose);
            pose=next;error=std::move(trial);value=score;lambda=std::max(1e-8,lambda*.3);
            if (change<1e-7) break;
        } else lambda*=6.;
    }
    const double rmse=std::sqrt(std::inner_product(error.begin(),error.end(),error.begin(),0.)/error.size());
    if (!std::isfinite(rmse)) throw std::runtime_error("Invalid joint pose result");
    return {{pose[0],pose[1]},rmse,iteration+1};
}
}
