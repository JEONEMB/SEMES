#include "common.hpp"
#include <iostream>
#include <regex>
#include <limits>
using namespace semes;
int wmain(int argc,wchar_t** wide_argv){
    semes::Utf8Arguments unicode_arguments(argc,wide_argv);
    char** argv=unicode_arguments.data();
    try {
        if(argc!=3)throw std::runtime_error("historical_measure_qr existing-20261014-data new-output");
        fs::path data=fs::u8path(argv[1]),out=fs::u8path(argv[2]);
        if(fs::exists(out))throw std::runtime_error("New output required");
        fs::create_directories(out);
        auto center=loadimage(data/"center.bmp");
        struct Shot{
            std::string axis;
            double value;
            fs::path path;
        };
        std::vector<Shot> shots{
            {
                "center",0,data/"center.bmp"
            }
        };
        std::regex label("([TY])([+-][0-9]+\\.[0-9]+)\\.bmp");
        for(auto folder:{
            "Theta","Y"
        }) {
            std::vector<fs::path> paths;
            for(auto& entry:fs::directory_iterator(data/folder))if(entry.is_regular_file())paths.push_back(entry.path());
            std::sort(paths.begin(),paths.end());
            for(auto& path:paths){
                std::smatch m;
                auto name=path.filename().u8string();
                if(std::regex_match(name,m,label))shots.push_back({
                    m[1],std::stod(m[2]),path
                });
            }
        }
        const cv::Rect boxes[]={
            {
                1554,972,180,192
            },{
                1815,972,180,192
            },{
                2073,972,180,192
            }
        };
        std::ofstream rows(out/"qr_measurements.csv");
        rows<<"axis,value,file";
        for(int k=0; k<3; ++k)for(auto key:{
            "dx","dy","a00","a01","a10","a11","cc","match"
        })rows<<",q"<<k<<'_'<<key;
        rows<<'\n';
        int failed=0;
        for(size_t i=0; i<shots.size(); ++i) {
            auto& shot=shots[i];
            auto image=shot.axis=="center"?center:loadimage(shot.path);
            if(image.size()!=center.size())throw std::runtime_error("Original dimensions required");
            rows<<shot.axis<<','<<std::setprecision(17)<<shot.value<<','<<csv(shot.path.filename().u8string());
            for(auto box:boxes) {
                cv::Rect search(box.x-130,box.y-130,box.width+260,box.height+260);
                cv::Mat correlation;
                cv::matchTemplate(image(search),center(box),correlation,cv::TM_CCOEFF_NORMED);
                double score;
                cv::Point location;
                cv::minMaxLoc(correlation,nullptr,&score,nullptr,&location);
                cv::Point shift=location-cv::Point(130,130);
                cv::Mat warp=cv::Mat::eye(2,3,CV_32F);
                double cc=std::numeric_limits<double>::quiet_NaN();
                try{
                    cc=cv::findTransformECC(center(box),image(box+shift),warp,cv::MOTION_AFFINE,cv::TermCriteria(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,150,1e-7),cv::noArray(),3);
                }
                catch(const cv::Exception&){
                    ++failed;
                }
                rows<<','<<shift.x+warp.at<float>(0,2)<<','<<shift.y+warp.at<float>(1,2)<<','<<warp.at<float>(0,0)<<','<<warp.at<float>(0,1)<<','<<warp.at<float>(1,0)<<','<<warp.at<float>(1,1)<<','<<cc<<','<<score;
            }
            rows<<'\n';
            rows.flush();
            if(i%30==0)std::cout<<"Measured "<<i+1<<'/'<<shots.size()<<std::endl;
        }
        Json summary=Json::dict();
        summary["n"]=int(shots.size());
        summary["ecc_failures"]=failed;
        summary["center_sha256"]=sha256(data/"center.bmp");
        summary["opencv"]=CV_VERSION;
        summary["measurement"]="Historical three QR affine ECC, 180x192 crops, NCC130, ECC150 EPS1e-7 filter3";
        summary["changes"]=std::string("Sorted input filenames and explicit original data path; OpenCV ")+CV_VERSION+" C++ instead of the original Python OpenCV runtime. No claim of 630-point measurement.";
        summary["created_kst"]=now();
        write(out/"summary.json",summary.dump());
        return 0;
    }
    catch(const std::exception&e){
        std::cerr<<e.what()<<'\n';
        return 1;
    }
}
