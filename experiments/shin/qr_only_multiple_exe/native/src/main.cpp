#include "qr_pose.hpp"
#include <opencv2/imgcodecs.hpp>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace {
using Clock=std::chrono::steady_clock;
double ms(Clock::time_point start) {return std::chrono::duration<double,std::milli>(Clock::now()-start).count();}

struct Args {
    std::map<std::string,std::string> flags;
    std::vector<std::string> files;
    Args(int argc,char** argv) {
        for(int i=2;i<argc;++i) {
            std::string arg=argv[i];
            if (arg.compare(0,2,"--")==0) {
                if (i+1==argc || std::string(argv[i+1]).compare(0,2,"--")==0) throw std::runtime_error("Missing value for "+arg);
                flags[arg]=argv[++i];
            } else files.push_back(arg);
        }
    }
    std::string value(const std::string& key,const std::string& fallback={})const {
        auto it=flags.find(key);return it==flags.end()?fallback:it->second;
    }
};

bool absolute_path(const std::string& p) {return !p.empty() && (p.front()=='/' || p.front()=='\\' || (p.size()>1 && p[1]==':'));}

void ssafy(const Args& args) {
    const auto config_path=args.value("--config","config/ssafy.json");
    const auto config=qr::open_model(config_path);
    const cv::FileNode y=config["y_mm"],t=config["theta_deg"];
    if (!(y.isInt()||y.isReal()) || !(t.isInt()||t.isReal()) || !std::isfinite(double(y)) || !std::isfinite(double(t)))
        throw std::runtime_error("A finite numeric y_mm and theta_deg are required");
    const qr::Pose target{double(y),double(t)};
    std::string model=config["model"];
    if (!absolute_path(model)) {
        const auto end=config_path.find_last_of("/\\");
        if (end!=std::string::npos) model=config_path.substr(0,end+1)+model;
    }
    qr::FeaturePose feature(model);std::unique_ptr<qr::DirectPose> direct;
    if (!args.value("--direct").empty()) direct.reset(new qr::DirectPose(args.value("--direct"),"ssafy"));
    for(const auto& p:args.files) {
        auto frame=qr::read_frame(p);auto start=Clock::now();qr::Pose pose;
        bool refined=false;
        if (direct) {
            try {
                const auto init=direct->initial(frame);
                if (qr::DirectPose::ssafy_supported(init)) {
                    const auto fit=direct->refine(frame,init);pose=(init+fit.pose)*.5;refined=true;
                }
            } catch (const std::exception&) { /* The established feature path remains available. */ }
        }
        if (!refined) pose=feature.predict(frame);
        pose-=target;
        std::cout<<"{\"y_mm\":"<<pose[0]<<",\"theta_deg\":"<<pose[1]<<",\"method\":\""
                 <<(refined?"joint_pixel_refined":"feature_fit")<<"\",\"inference_ms\":"<<ms(start)<<"}\n";
    }
}

void rig_frame(const Args& args) {
    auto reference=qr::read_frame(args.value("--reference"));
    qr::Cancellation model(args.value("--model","models/rig-cancellation.xml.gz"),reference);
    std::unique_ptr<qr::DirectPose> direct;
    if (!args.value("--direct").empty()) direct.reset(new qr::DirectPose(args.value("--direct"),"rig",reference));
    for(const auto& path:args.files) {
        auto frame=qr::read_frame(path);auto start=Clock::now();auto pose=model.predict(frame);
        const auto dx=model.horizontal(frame);
        std::ostringstream optional;
        if (direct) {
            try {
                auto result=direct->refine(frame,pose);
                optional<<",\"joint_y_mm\":"<<result.pose[0]<<",\"joint_theta_deg\":"<<result.pose[1]
                        <<",\"joint_pixel_rmse\":"<<result.pixel_rmse;
            } catch(const std::exception&) { optional<<",\"joint_fit_available\":false"; }
        }
        std::cout<<"{\"dx_px\":"<<dx<<",\"y_mm\":"<<pose[0]<<",\"theta_deg\":"<<pose[1]
                 <<optional.str()<<",\"inference_ms\":"<<ms(start)<<"}\n";
    }
}

void direct_check(const Args& args) {
    const qr::DirectPose model(args.value("--direct"),"ssafy");
    for (const auto& path:args.files) {
        auto frame=qr::read_frame(path);const auto start=Clock::now();
        const auto init=model.initial(frame);
        std::cout<<"{\"initial_y_mm\":"<<init[0]<<",\"initial_theta_deg\":"<<init[1];
        if (qr::DirectPose::ssafy_supported(init)) {
            try {
                const auto fit=model.refine(frame,init);const auto pose=(init+fit.pose)*.5;
                std::cout<<",\"y_mm\":"<<pose[0]<<",\"theta_deg\":"<<pose[1];
            } catch (const std::exception&) {std::cout<<",\"joint_fit_available\":false";}
        }
        std::cout<<",\"inference_ms\":"<<ms(start)<<"}\n";
    }
}
}

int main(int argc,char** argv) {
    cv::setNumThreads(1);
    try {
        const std::string mode=argc>1?argv[1]:"--help";
        if (mode=="--help") {
            std::cout<<"qr_pose_cli ssafy --config config/ssafy.json [--direct models/ssafy-direct.xml.gz] FRAME...\n"
                       "qr_pose_cli rig-frame --reference CENTER.bmp --model models/rig-cancellation.xml.gz [--direct MODEL] FRAME...\n"
                       "qr_pose_cli direct-check --direct SSAFY_MODEL FRAME...\n";
            return 0;
        }
        const Args args(argc,argv);
        if (args.files.empty()) throw std::runtime_error("Supply at least one image");
        std::cout<<std::setprecision(12);
        if (mode=="ssafy") ssafy(args);
        else if (mode=="rig-frame") rig_frame(args);
        else if (mode=="direct-check") direct_check(args);
        else throw std::runtime_error("Unknown mode; use --help");
        return 0;
    } catch (const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
