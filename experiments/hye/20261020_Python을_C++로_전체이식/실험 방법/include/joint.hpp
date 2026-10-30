#pragma once
#include "alignment.hpp"
namespace semes {
    // Frozen training results only. Inference takes observable features, never labels or trial IDs.
    class JointOrigins {
        public: explicit JointOrigins(const fs::path& model);
        Json predict(const cv::Mat& feature)const;
        const std::string& feature_name()const{
            return feature_;
        }
        private: cv::Mat A_,W_,centers_,Aw_,cw_,linear_pinv_;
        std::vector<std::string> ids_;
        std::vector<cv::Mat> local_aw_,local_pinv_;
        cv::Vec2d scale_,lower_,upper_;
        int degree_=2;
        std::string feature_;
    };
}
