#include "direct_observer.hpp"
#include "ecc_compat.hpp"
#include <chrono>
namespace semes {
static const std::array<cv::Rect,3> boxes={cv::Rect(1552,969,184,198),cv::Rect(1809,971,184,198),cv::Rect(2067,973,185,198)};
static cv::Mat translation(double x,double y){return (cv::Mat_<double>(3,3)<<1,0,x,0,1,y,0,0,1);}
static cv::Mat numeric_array(const Npy& a){if(a.shape.size()!=2)throw std::runtime_error("Reference array must have rank2");cv::Mat m(int(a.shape[0]),int(a.shape[1]),CV_64F);for(size_t i=0;i<a.count();++i)m.ptr<double>()[i]=a.number(i);return m;}
cv::Point2d DirectObserver::project(const cv::Mat& H,cv::Point2d p){double z=H.at<double>(2,0)*p.x+H.at<double>(2,1)*p.y+H.at<double>(2,2);return {(H.at<double>(0,0)*p.x+H.at<double>(0,1)*p.y+H.at<double>(0,2))/z,(H.at<double>(1,0)*p.x+H.at<double>(1,1)*p.y+H.at<double>(1,2))/z};}
DirectObserver::DirectObserver(const fs::path& center,const fs::path& reference){
    if(sha256(center)!="01a69df6771b840d2fcf9cf727760592b5fb199c12277bfc173af514e26fd685")throw std::runtime_error("Fixed center hash changed");center_=loadimage(center);auto a=npz(reference);points_=numeric_array(a.at("points"));if(points_.cols!=2||points_.rows!=630)throw std::runtime_error("Expected the original 630-point projection geometry");for(int i=0;i<points_.rows;++i){ids_.push_back(int(a.at("ids").number(i)));regions_.push_back(int(a.at("regions").number(i)));if(regions_.back()<0||regions_.back()>2)throw std::runtime_error("Invalid QR reference region");}
    for(int k=0;k<3;++k){templates_[k]=center_(boxes[k]).clone();masks_[k]=cv::Mat::zeros(boxes[k].size(),CV_8U);masks_[k](cv::Rect(6,6,boxes[k].width-12,boxes[k].height-12)).setTo(255);}
}
Json DirectObserver::extract(const std::vector<unsigned char>& current,bool all)const{
    auto began=std::chrono::steady_clock::now();auto image=cv::imdecode(current,cv::IMREAD_GRAYSCALE);if(image.empty()||image.size()!=center_.size())throw std::runtime_error("Expected decodable fixed-center image dimensions");
    std::array<cv::Point,3> shifts;std::array<double,3> scores;int selected_pad=0;Json attempts=Json::list();auto coarse_start=std::chrono::steady_clock::now();
    for(int pad:{130,260,512}){Json attempt=Json::dict();attempt["pad"]=pad;attempt["scores"]=Json::list();for(int k=0;k<3;++k){auto box=boxes[k];cv::Rect window(std::max(0,box.x-pad),std::max(0,box.y-pad),0,0);window.width=std::min(image.cols,box.x+box.width+pad)-window.x;window.height=std::min(image.rows,box.y+box.height+pad)-window.y;cv::Mat correlation;cv::matchTemplate(image(window),templates_[k],correlation,cv::TM_CCOEFF_NORMED);cv::Point location;cv::minMaxLoc(correlation,nullptr,&scores[k],nullptr,&location);shifts[k]=location+window.tl()-box.tl();attempt["scores"].push(scores[k]);}attempts.push(attempt);selected_pad=pad;if(*std::min_element(scores.begin(),scores.end())>=.6)break;}
    Json result=Json::dict();result["image_sha256"]=sha256(current);result["ncc_pad"]=selected_pad;result["ncc_attempts"]=attempts;result["ncc_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-coarse_start).count();result["ncc_shifts"]=Json::list();result["ncc_scores"]=Json::list();for(int k=0;k<3;++k){Json p=Json::list();p.push(shifts[k].x);p.push(shifts[k].y);result["ncc_shifts"].push(p);result["ncc_scores"].push(scores[k]);}result["candidates"]=Json::dict();
    struct Variant{const char* name;int motion,gaussian;};
    for(auto variant:{Variant{"affine_g3",cv::MOTION_AFFINE,3},Variant{"homography_g3",cv::MOTION_HOMOGRAPHY,3},Variant{"affine_g5",cv::MOTION_AFFINE,5},Variant{"homography_g5",cv::MOTION_HOMOGRAPHY,5}}){
        if(!all&&std::string(variant.name)!="homography_g5")continue;auto start=std::chrono::steady_clock::now();Json candidate=Json::dict();for(auto key:{"H","ecc","photometric_rmse","brightness_gain","brightness_offset","valid_pixels","qr_success","reasons"})candidate[key]=Json::list();cv::Mat displacement(points_.size(),CV_64F,cv::Scalar(std::numeric_limits<double>::quiet_NaN()));bool success=true;
        for(int k=0;k<3;++k){Json reasons=Json::list();bool good=false;cv::Mat H(3,3,CV_64F,cv::Scalar(std::numeric_limits<double>::quiet_NaN()));double cc=std::numeric_limits<double>::quiet_NaN(),rmse=cc,gain=cc,offset=cc;int pixels=0;
            try{
                if(scores[k]<.6)throw std::runtime_error("NCC below fixed threshold after expanded search");auto box=boxes[k];cv::Rect read(box.x+shifts[k].x-12,box.y+shifts[k].y-12,box.width+24,box.height+24);if((read&cv::Rect(0,0,image.cols,image.rows))!=read)throw std::runtime_error("Current crop outside image");auto crop=image(read).clone();cv::Mat local;translation(12,12).convertTo(local,CV_32F);if(variant.motion!=cv::MOTION_HOMOGRAPHY)local=local.rowRange(0,2).clone();
                cc=ecc5::findTransformECCWithMask(templates_[k],crop,masks_[k],cv::Mat(crop.size(),CV_8U,cv::Scalar(255)),local,variant.motion,cv::TermCriteria(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,200,1e-8),variant.gaussian);
                cv::Mat localH=cv::Mat::eye(3,3,CV_64F);cv::Mat d;local.convertTo(d,CV_64F);d.copyTo(localH.rowRange(0,d.rows));H=translation(read.x,read.y)*localH*translation(-box.x,-box.y);H/=H.at<double>(2,2);double refinement=0;
                for(int i=0;i<points_.rows;++i)if(regions_[i]==k){cv::Point2d p(points_.at<double>(i,0),points_.at<double>(i,1)),mapped=project(H,p);double dx=mapped.x-p.x-shifts[k].x,dy=mapped.y-p.y-shifts[k].y;refinement=std::max({refinement,std::abs(dx),std::abs(dy)});if(!std::isfinite(mapped.x)||!std::isfinite(mapped.y))reasons.push("nonfinite");}
                if(!cv::checkRange(H))reasons.push("nonfinite");if(cc<.8)reasons.push("ECC below fixed0.8");if(refinement>10)reasons.push("refinement more than10px from NCC");
                cv::Mat mapx(box.size(),CV_32F),mapy(box.size(),CV_32F),floating,aligned;crop.convertTo(floating,CV_32F);for(int y=0;y<box.height;++y)for(int x=0;x<box.width;++x){auto p=project(localH,{double(x),double(y)});mapx.at<float>(y,x)=float(p.x);mapy.at<float>(y,x)=float(p.y);}cv::remap(floating,aligned,mapx,mapy,cv::INTER_LINEAR,cv::BORDER_CONSTANT);pixels=cv::countNonZero(masks_[k]);cv::Mat A(pixels,2,CV_64F),target(pixels,1,CV_64F);int row=0;
                for(int y=0;y<box.height;++y)for(int x=0;x<box.width;++x)if(masks_[k].at<unsigned char>(y,x)){A.at<double>(row,0)=aligned.at<float>(y,x);A.at<double>(row,1)=1;target.at<double>(row)=templates_[k].at<unsigned char>(y,x);++row;}auto fit=numeric::least_squares(A,target);gain=fit.at<double>(0);offset=fit.at<double>(1);rmse=numeric::rms(A*fit-target);good=reasons.array.empty();
                if(good)for(int i=0;i<points_.rows;++i)if(regions_[i]==k){cv::Point2d p(points_.at<double>(i,0),points_.at<double>(i,1)),mapped=project(H,p);displacement.at<double>(i,0)=mapped.x-p.x;displacement.at<double>(i,1)=mapped.y-p.y;}
            }catch(const std::exception& error){H.setTo(std::numeric_limits<double>::quiet_NaN());cc=rmse=gain=offset=std::numeric_limits<double>::quiet_NaN();pixels=0;reasons=Json::list();reasons.push(std::string("native_error: ")+error.what());}
            candidate["H"].push(jsonmatrix(H));candidate["ecc"].push(cc);candidate["photometric_rmse"].push(rmse);candidate["brightness_gain"].push(gain);candidate["brightness_offset"].push(offset);candidate["valid_pixels"].push(pixels);candidate["qr_success"].push(good);candidate["reasons"].push(reasons);success&=good;
        }
        candidate["D"]=jsonmatrix(displacement);candidate["success"]=success;candidate["seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();result["candidates"][variant.name]=candidate;
    }
    result["decoded_gray_sha256"]=sha256(std::vector<unsigned char>(image.datastart,image.dataend));result["center_gray_sha256"]=sha256(std::vector<unsigned char>(center_.datastart,center_.dataend));
    result["seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count();return result;
}
cv::Mat DirectObserver::qr_grid(const Json& candidate){
    if(candidate.at("H").size()!=3)throw std::runtime_error("Three QR homographies required");cv::Mat x(1,54,CV_64F);int index=0;
    for(int k=0;k<3;++k){auto H=matrix(candidate.at("H").at(k));auto box=boxes[k];for(double b:{.2,.5,.8})for(double a:{.2,.5,.8}){cv::Point2d p(box.x+a*box.width,box.y+b*box.height);auto mapped=project(H,p);x.at<double>(index++)=mapped.x-p.x;x.at<double>(index++)=mapped.y-p.y;}}
    return x;
}
}
