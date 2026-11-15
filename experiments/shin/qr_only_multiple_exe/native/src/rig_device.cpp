#include "qr_pose.hpp"
#include <stage/stage.hpp>
#include <stage_camera/camera.hpp>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>

namespace {
using Clock=std::chrono::steady_clock;
double millis(Clock::time_point start) {return std::chrono::duration<double,std::milli>(Clock::now()-start).count();}

class DeviceLock {
public:
    DeviceLock() {
        const char* home=std::getenv("HOME");
        if (!home) throw std::runtime_error("HOME is required for the original stage lock");
        const auto file=std::string(home)+"/.alignment-stage.lock";
        fd_=open(file.c_str(),O_CREAT|O_RDWR|O_CLOEXEC,0600);
        if (fd_<0 || flock(fd_,LOCK_EX|LOCK_NB)<0) {
            if (fd_>=0) close(fd_);
            throw std::runtime_error("Stage is in use or its lock cannot be opened");
        }
    }
    ~DeviceLock() {close(fd_);}
    DeviceLock(const DeviceLock&)=delete;
    DeviceLock& operator=(const DeviceLock&)=delete;
private:
    int fd_;
};

class RigStage final : public qr::Stage {
public:
    qr::Position position() override {const auto p=stage_.status().pos;return {p.y_um,p.t_mdeg};}
    std::array<int,4> limits() override {
        const auto l=stage_.limits();return {{l.y_min_um,l.y_max_um,l.t_min_mdeg,l.t_max_mdeg}};
    }
    qr::Position move_absolute(qr::Position p) override {
        try {
            const auto r=stage_.moveTo({p[0],p[1]},std::chrono::milliseconds(60000));
            if(!r.ok()) throw std::runtime_error(stage::toString(r.result));
            return {r.pos.y_um,r.pos.t_mdeg};
        } catch (...) {
            try{stage_.stop();}catch(...){ }
            throw;
        }
    }
private:
    stage::Stage stage_;
};

double number(const char* arg) {
    size_t used=0;const auto v=std::stod(arg,&used);
    if (arg[used]!='\0' || !std::isfinite(v)) throw std::runtime_error("Expected a finite number");
    return v;
}
}

int main(int argc,char** argv) {
    if(argc!=8 && argc!=9) {
        std::cerr<<"qr_rig CAMERA_CALIB CENTER.bmp GEOMETRY.xml.gz CENTER_Y_MM CENTER_THETA_DEG ZOOM LOG.jsonl [JOINT.xml.gz]\n";
        return 1;
    }
    cv::setNumThreads(1);
    try {
        const auto center=stage::Position::fromMmDeg(number(argv[4]),number(argv[5]));
        stage_camera::CameraOptions options;
        options.calib_path=argv[1];options.zoom=number(argv[6]);options.analogue_gain=3.0f;
        options.settle.max_shift_px=.25;options.settle.timeout_ms=6000;
        auto reference=qr::read_frame(argv[2]);qr::Cancellation model(argv[3],reference);
        std::unique_ptr<qr::DirectPose> direct;
        if(argc==9) direct.reset(new qr::DirectPose(argv[8],"rig",reference));
        const int log_fd=open(argv[7],O_CREAT|O_EXCL|O_WRONLY|O_CLOEXEC,0644);
        if(log_fd<0) throw std::runtime_error("Cannot create a new log file");
        close(log_fd);
        std::ofstream log(argv[7],std::ios::app);
        if (!log) throw std::runtime_error("Cannot open log");
        DeviceLock lock;RigStage stage;stage_camera::Camera camera(options);
        auto frame=[&](){
            auto shot=camera.captureSettled(4);
            if(!shot.settle.settled) throw std::runtime_error("Camera did not settle");
            return shot.image;
        };
        qr::Settings settings;settings.center={center.y_um,center.t_mdeg};
        const auto start=Clock::now();
        const auto result=qr::run_correction(stage,settings,[&](){
            const auto im=frame();const auto begin=Clock::now();const double dx=model.horizontal(im);
            log<<"{\"kind\":\"qr_frame\",\"dx_px\":"<<dx<<",\"inference_ms\":"<<millis(begin)<<"}\n";
            return dx;
        },[&](){
            const auto im=frame();const auto begin=Clock::now();
            qr::Pose estimate;
            try { estimate=model.predict(im); }
            catch(const std::exception&) {const double nan=std::numeric_limits<double>::quiet_NaN();return qr::Pose(nan,nan);}
            log<<"{\"kind\":\"qr_geometry_time\",\"inference_ms\":"<<millis(begin)<<"}\n";
            if(direct) {
                try {
                    const auto begin_joint=Clock::now();const auto result=direct->refine(im,estimate);
                    log<<"{\"kind\":\"joint_pixel_fit\",\"y_mm\":"<<result.pose[0]<<",\"theta_deg\":"<<result.pose[1]
                       <<",\"pixel_rmse\":"<<result.pixel_rmse<<",\"inference_ms\":"<<millis(begin_joint)<<"}\n";
                } catch(const std::exception&) {log<<"{\"kind\":\"joint_pixel_fit\",\"available\":false}\n";}
            }
            return estimate;
        },[&](const char* kind,qr::Position reported,qr::Pose value){
            log<<"{\"kind\":\""<<kind<<"\",\"reported_y_um\":"<<reported[0]<<",\"reported_theta_mdeg\":"<<reported[1];
            if (std::isfinite(value[0]) && std::isfinite(value[1])) log<<",\"y_mm\":"<<value[0]<<",\"theta_deg\":"<<value[1];
            log<<"}\n";log.flush();
        });
        log<<"{\"kind\":\"result\",\"result\":\""<<result.reason<<"\",\"moves\":"<<result.moves
           <<",\"corrections\":"<<result.corrections<<",\"elapsed_ms\":"<<millis(start)<<"}\n";
        return result.reason=="qr_center_candidate" ? 2 : 1;
    } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
