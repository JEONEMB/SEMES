#include "qrwithbar_lanczos.hpp"
#include "observation.hpp"
#include <windows.h>
#include <cmath>
#include <stdexcept>
namespace qrwithbar {
namespace fs=std::filesystem;
static void matrix(const cv::FileNode& node,const char* key,cv::Mat& out,int rows,int columns) {
    node[key]>>out;
    if(out.type()!=CV_64F||out.rows!=rows||out.cols!=columns||!cv::checkRange(out))
        throw std::runtime_error(std::string("Invalid Lanczos calibration matrix: ")+key);
}
struct Estimator::Implementation {
    fs::path assets;
    cv::Mat reference,mean,scale,ymean,coef;
    detail::Warps reference_warps;
    detail::Observation reference_observation;
    double minimum_y,maximum_y,minimum_t,maximum_t;
    explicit Implementation(const std::string& path):assets(fs::u8path(path)) {
        cv::FileStorage file((assets/"model.json").u8string(),cv::FileStorage::READ);
        if(!file.isOpened()||(std::string)file["method"]!="lanczos3"||(int)file["feature_count"]!=10)
            throw std::runtime_error("Missing or incompatible Lanczos model.json");
        matrix(file.root(),"mean",mean,1,10);matrix(file.root(),"scale",scale,1,10);
        matrix(file.root(),"ymean_ticks",ymean,1,2);matrix(file.root(),"coef",coef,10,2);
        for(int j=0;j<10;j++)if(scale.at<double>(0,j)<=0)throw std::runtime_error("Invalid feature scale");
        minimum_y=(double)file["minimum_y"];maximum_y=(double)file["maximum_y"];
        minimum_t=(double)file["minimum_t"];maximum_t=(double)file["maximum_t"];
        reference=detail::read_gray(assets/"reference.png");
        if(reference.cols!=3840||reference.rows!=2160)throw std::runtime_error("Wrong calibration image size");
        reference_warps=detail::align_qrs(reference,assets);
        reference_observation=detail::analytic_observe(reference,reference_warps);
        if(reference_observation.failures)throw std::runtime_error("Reference measurement failed");
    }
};
Estimator::Estimator(const std::string& path):implementation_(std::make_shared<Implementation>(path)) {}
Estimator::~Estimator()=default;
Estimate Estimator::estimate(const cv::Mat& source) const {
    if(source.empty()||source.depth()!=CV_8U||source.cols!=3840||source.rows!=2160)
        throw std::invalid_argument("Lanczos estimator expects a3840x2160 uint8 image");
    cv::Mat image;
    if(source.channels()==1)image=source;
    else if(source.channels()==3)cv::cvtColor(source,image,cv::COLOR_BGR2GRAY);
    else if(source.channels()==4)cv::cvtColor(source,image,cv::COLOR_BGRA2GRAY);
    else throw std::invalid_argument("Expected1,3 or4 image channels");
    const auto& c=*implementation_;
    auto warps=detail::align_qrs(image,c.assets);
    auto initial=detail::analytic_observe(image,warps);
    if(initial.failures)throw std::runtime_error("QR/line initialization failed");
    auto observation=detail::interpolated_observe(image,c.reference,warps,c.reference_warps,
        initial,c.reference_observation,2);
    if(observation.failures)throw std::runtime_error("Lanczos QR/line registration failed");
    double pose[2]={c.ymean.at<double>(0,0),c.ymean.at<double>(0,1)};
    for(int j=0;j<10;j++) {
        double x=observation.features[j];
        if(!std::isfinite(x))throw std::runtime_error("Nonfinite measured feature");
        for(int k=0;k<2;k++)pose[k]+=(x-c.mean.at<double>(0,j))/c.scale.at<double>(0,j)*c.coef.at<double>(j,k);
    }
    Estimate result{};result.y_mm=pose[0]*.05;result.yaw_deg=pose[1]*.01;
    if(!std::isfinite(result.y_mm)||!std::isfinite(result.yaw_deg))throw std::runtime_error("Nonfinite pose");
    result.features=observation.features;result.qr_points=630;
    for(int i=0;i<60;i++){result.upper_valid+=observation.edges[i].valid;result.lower_valid+=observation.edges[60+i].valid;}
    result.outside_training_range=result.y_mm<c.minimum_y||result.y_mm>c.maximum_y||
        result.yaw_deg<c.minimum_t||result.yaw_deg>c.maximum_t;
    return result;
}
std::string default_assets_directory() {
    fs::path local=fs::current_path()/"qrwithbar_assets";
    if(fs::exists(local/"model.json"))return local.u8string();
    wchar_t name[32768];DWORD length=GetModuleFileNameW(nullptr,name,32768);
    if(length&&length<32768) {
        auto beside=fs::path(std::wstring(name,length)).parent_path()/"qrwithbar_assets";
        if(fs::exists(beside/"model.json"))return beside.u8string();
    }
    return (fs::current_path()/"experiments/shin/qrwithbar_lanczos/assets").u8string();
}
}
