#include "control_internal.hpp"
#include "site.hpp"
#include "sha256.hpp"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace legacy_control {
static std::vector<unsigned char> bytes(const fs::path& p){std::ifstream f(p,std::ios::binary);if(!f)throw std::runtime_error("Cannot read "+p.u8string());return {std::istreambuf_iterator<char>(f),std::istreambuf_iterator<char>()};}
cv::Mat read_gray(const fs::path& p){auto r=cv::imdecode(bytes(p),cv::IMREAD_GRAYSCALE);if(r.empty())throw std::runtime_error("Cannot decode "+p.u8string());return r;}
std::string sha256(const fs::path& p){return final00_sha::sha256_hex(bytes(p));}   // 운영체제 라이브러리 없는 SHA-256 (리눅스 빌드)
cv::Mat checked_gray(const cv::Mat& input){
    if(input.empty()||input.depth()!=CV_8U||input.size()!=cv::Size(3840,2160))throw std::invalid_argument("control expects3840x2160 uint8 image");
    cv::Mat gray;if(input.channels()==1)gray=input;else if(input.channels()==3)cv::cvtColor(input,gray,cv::COLOR_BGR2GRAY);else if(input.channels()==4)cv::cvtColor(input,gray,cv::COLOR_BGRA2GRAY);else throw std::invalid_argument("control expects1/3/4 channels");return gray;
}
static std::vector<PointModel> reference_models(const cv::Mat& template8,const std::vector<cv::Point2f>& points){
    cv::Mat image;template8.convertTo(image,CV_64F);cv::GaussianBlur(image,image,cv::Size(),.8);
    cv::Mat gx(image.size(),CV_64F),gy(image.size(),CV_64F);
    for(int y=0;y<image.rows;++y)for(int x=0;x<image.cols;++x){
        gx.at<double>(y,x)=x==0?image.at<double>(y,1)-image.at<double>(y,0):x==image.cols-1?image.at<double>(y,x)-image.at<double>(y,x-1):.5*(image.at<double>(y,x+1)-image.at<double>(y,x-1));
        gy.at<double>(y,x)=y==0?image.at<double>(1,x)-image.at<double>(0,x):y==image.rows-1?image.at<double>(y,x)-image.at<double>(y-1,x):.5*(image.at<double>(y+1,x)-image.at<double>(y-1,x));
    }
    auto tc=nearest_prefilter(image),xc=nearest_prefilter(gx),yc=nearest_prefilter(gy);std::vector<PointModel> out;
    for(auto point:points){PointModel model;model.point=point;model.reference=cv::Mat(289,1,CV_64F);cv::Mat design(289,4,CV_64F);int i=0;
        for(int y=-8;y<=8;++y)for(int x=-8;x<=8;++x,++i){double xx=double(point.x)+x+12,yy=double(point.y)+y+12;double value=spline_sample(tc,xx,yy);model.reference.at<double>(i)=value;
            auto d=design.ptr<double>(i);d[0]=spline_sample(xc,xx,yy);d[1]=spline_sample(yc,xx,yy);d[2]=value;d[3]=1;}
        cv::invert(design,model.pinverse,cv::DECOMP_SVD);out.push_back(model);
    }return out;
}
Tracker::Tracker(const fs::path& assets){
    auto center=assets/"center.bmp";
    if(sha256(center)!=site::get().center_sha256)throw std::runtime_error("Legacy fixed center changed");
    reference_=read_gray(center);checked_gray(reference_);
    cv::FileStorage file((assets/"points.json").u8string(),cv::FileStorage::READ);if(!file.isOpened())throw std::runtime_error("Missing fixed630/selected536 points");
    auto rows=file["points"];if(!rows.isSeq()||rows.size()!=630)throw std::runtime_error("Expected630 fixed points");
    for(auto it=rows.begin();it!=rows.end();++it){auto row=*it;if(!row.isSeq()||row.size()!=4)throw std::runtime_error("Expected[x,y,region,selected] point row");
        double x=double(row[0]),y=double(row[1]);int k=int(row[2]),selected=int(row[3]);
        if(!std::isfinite(x)||!std::isfinite(y)||k<0||k>2||(selected!=0&&selected!=1)||!site::get().qr[k].contains(cv::Point(int(x),int(y))))throw std::runtime_error("Invalid fixed reference point");
        int id=int(points_.size());points_.emplace_back(x,y);regions_.push_back(k);selected_.push_back(static_cast<unsigned char>(selected));
        // Match Python's global->crop float32 cast before local QR subtraction.
        cv::Point2f crop(float(x-site::get().full_crop.x),float(y-site::get().full_crop.y));auto local=site::get().qr[k].tl()-site::get().full_crop.tl();
        panels_[k].points.emplace_back(crop.x-float(local.x),crop.y-float(local.y));panels_[k].ids.push_back(id);
    }
    if(std::count(selected_.begin(),selected_.end(),1)!=536)throw std::runtime_error("Expected fixed536 interior selection");
    for(int k=0;k<3;++k){auto original=reference_(site::get().qr[k]).clone();cv::GaussianBlur(original,panels_[k].blurred8,cv::Size(),.8);panels_[k].models=reference_models(original,panels_[k].points);}
    // Preserve source Predictor.__init__: independently track center, then subtract its D0.
    zero_=measure(reference_).displacement.clone();
}
int Tracker::valid_selected(const Tracks& tracks)const{int count=0;for(int i=0;i<630;++i)if(selected_[i]&&tracks.valid[i])++count;return count;}
Tracks Tracker::measure(const cv::Mat& gray)const{
    checked_gray(gray);Tracks out{cv::Mat::zeros(630,2,CV_32F),cv::Mat::zeros(630,5,CV_32F),std::vector<unsigned char>(630,0),{}};
    auto ref=reference_(site::get().full_crop),cur=gray(site::get().full_crop);
    for(int k=0;k<3;++k){auto b=site::get().qr[k];b.x-=site::get().full_crop.x;b.y-=site::get().full_crop.y;const int sx=site::get().search_x,sy=site::get().search_y;cv::Rect search(b.x-sx,b.y-sy,b.width+2*sx,b.height+2*sy);
        cv::Mat correlation;cv::matchTemplate(cur(search),ref(b),correlation,cv::TM_CCOEFF_NORMED);cv::Point location;cv::minMaxLoc(correlation,nullptr,&out.ncc[k],nullptr,&location);
        cv::Point shift=location-cv::Point(sx,sy);cv::Rect target_box=b+shift;
        if((target_box&cv::Rect(0,0,cur.cols,cur.rows))!=target_box)throw std::runtime_error("Legacy QR crop outside image");
        cv::Mat target=cur(target_box).clone(),blurred;cv::GaussianBlur(target,blurred,cv::Size(),.8);const auto& panel=panels_[k];
        std::vector<cv::Point2f> next=panel.points,back=panel.points;std::vector<unsigned char> fwd,rev;std::vector<float> e1,e2;
        auto criteria=cv::TermCriteria(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,60,.0001);
        cv::calcOpticalFlowPyrLK(panel.blurred8,blurred,panel.points,next,fwd,e1,cv::Size(15,15),0,criteria,cv::OPTFLOW_USE_INITIAL_FLOW,1e-5);
        cv::calcOpticalFlowPyrLK(blurred,panel.blurred8,next,back,rev,e2,cv::Size(15,15),0,criteria,cv::OPTFLOW_USE_INITIAL_FLOW,1e-5);
        cv::Mat double_target;target.convertTo(double_target,CV_64F);cv::GaussianBlur(double_target,double_target,cv::Size(),.8);auto coefficients=spline_coefficients(double_target);
        size_t n=panel.points.size();std::vector<cv::Point2d> d(n),initial(n),step(n);std::vector<cv::Mat> values(n),parameters(n);
        for(size_t i=0;i<n;++i){auto delta=next[i]-panel.points[i];d[i]=initial[i]=cv::Point2d(delta.x,delta.y);values[i]=cv::Mat(289,1,CV_64F);}
        for(int iteration=0;iteration<12;++iteration){double largest=0;
            for(size_t i=0;i<n;++i){auto point=panel.points[i];int pixel=0;
                for(int y=-8;y<=8;++y)for(int x=-8;x<=8;++x)values[i].at<double>(pixel++)=spline_sample(coefficients,double(point.x)+x+d[i].x,double(point.y)+y+d[i].y);
                parameters[i]=panel.models[i].pinverse*values[i];double gain=std::max(parameters[i].at<double>(2),.25);
                step[i]={std::clamp(parameters[i].at<double>(0)/gain,-.5,.5),std::clamp(parameters[i].at<double>(1)/gain,-.5,.5)};d[i]-=step[i];largest=std::max(largest,cv::norm(step[i]));
            }if(largest<2e-5)break;
        }
        for(size_t i=0;i<n;++i){int id=panel.ids[i];float dx=float(d[i].x),dy=float(d[i].y);out.displacement.at<float>(id,0)=dx+float(shift.x);out.displacement.at<float>(id,1)=dy+float(shift.y);
            auto fb=back[i]-panel.points[i];double fb_error=std::sqrt(double(fb.x*fb.x+fb.y*fb.y)),last=cv::norm(step[i]);
            bool finite=std::isfinite(d[i].x)&&std::isfinite(d[i].y);out.valid[id]=fwd[i]&&rev[i]&&fb_error<.5&&finite&&cv::norm(d[i]-initial[i])<1.5&&last<.02;
            cv::Mat residual=values[i]-parameters[i].at<double>(2)*panel.models[i].reference-parameters[i].at<double>(3);
            auto q=out.quality.ptr<float>(id);q[0]=float(cv::norm(residual)/17.);q[1]=float(last);q[2]=float(parameters[i].at<double>(2));q[3]=float(parameters[i].at<double>(3));q[4]=float(fb_error);
        }
    }if(!cv::checkRange(out.displacement))throw std::runtime_error("Nonfinite legacy tracks");return out;
}
cv::Mat Tracker::features(const Tracks& tracks)const{
    cv::Mat basis(536,6,CV_64F),difference(536,2,CV_64F);int row=0;
    for(int i=0;i<630;++i)if(selected_[i]){double u=(points_[i].x-1906)/486,v=(points_[i].y-1070)/122;
        double values[]={1,u,v,u*u,u*v,v*v};for(int j=0;j<6;++j)basis.at<double>(row,j)=values[j];
        for(int j=0;j<2;++j)difference.at<double>(row,j)=double(tracks.displacement.at<float>(i,j))-double(zero_.at<float>(i,j));++row;}
    cv::Mat coefficients;if(!cv::solve(basis,difference,coefficients,cv::DECOMP_SVD))throw std::runtime_error("Legacy P12 solve failed");return coefficients.reshape(1,1).clone();
}
}
