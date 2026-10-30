#pragma once
#include "io.hpp"
#include <opencv2/opencv.hpp>
#include <array>
namespace semes {
    struct Corner {
        int id=0,region=0;
        cv::Point2d reference,initial;
        cv::Matx22d inverse;
        std::array<double,4> patch;
    };
    struct Measurement {
        cv::Mat displacement,current,quality;
        std::vector<int> valid,flags;
        std::array<double,3> ncc{
        };
        double seconds=0;
    };
    struct Prediction {
        std::string axis,branch;
        double value=0,y=0,t=0,cost=0;
        Json json()const;
    };
    class Engine {
        public: Engine(const fs::path& assets,const fs::path& center);
        Measurement measure(const cv::Mat& image,int search=130)const;
        cv::Mat features(const cv::Mat& displacement,const std::string& name="P12")const;
        Prediction predict(const cv::Mat& feature)const;
        Json infer(const std::vector<unsigned char>& image,int search=130,Measurement* measured=nullptr)const;
        const std::vector<Corner>& corners()const{
            return corners_;
        }
        const cv::Mat& reference()const{
            return center_;
        }
        const Json& params()const{
            return model_;
        }
        private: cv::Mat center_,basis_,pinv_;
        std::vector<Corner> corners_;
        Json model_;
    };
    cv::Mat matrix(const Json& j);
    Json jsonmatrix(const cv::Mat& m);
    cv::Mat loadimage(const fs::path& p);
    void selftest();
    Prediction predict_model(const cv::Mat& feature,const Json& model,bool sign_gate=false);
    Json train_axis_model(const cv::Mat& X,const std::vector<std::string>& names,const std::vector<int>& training,const std::string& mode="center",bool quadratic=true,double shrink=.001,int covariance_dof=0);
}
