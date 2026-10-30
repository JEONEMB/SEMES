#pragma once
#include "direct_observer.hpp"
#include "line_observer.hpp"
namespace semes {
class ImageObserver {
    DirectObserver qr_; LineObserver lines_;
public:
    ImageObserver(const fs::path& center,const fs::path& geometry,const fs::path& protocol):qr_(center,geometry),lines_(center,protocol){}
    Json extract(const std::vector<unsigned char>& current,const std::string& condition)const {
        if(condition!="qr"&&condition!="upper"&&condition!="lower"&&condition!="both")throw std::runtime_error("Unknown observation condition");
        auto qr=qr_.extract(current,false).at("candidates").at("homography_g5");
        auto features=DirectObserver::qr_grid(qr);Json x=Json::list();
        for(int i=0;i<features.cols;++i)x.push(features.at<double>(i));
        bool good=qr.at("success").number!=0;
        if(condition!="qr"){
            auto line=lines_.extract(current);
            for(const std::string part:{"upper","lower"})if(condition==part||condition=="both"){
                for(const auto& v:line.at("X_"+part).array)x.push(v);
                good=good&&(line.at("success_"+part).number!=0);
            }
        }
        Json result=Json::dict();result["X"]=x;result["success"]=good;result["condition"]=condition;result["image_sha256"]=sha256(current);return result;
    }
};
}
