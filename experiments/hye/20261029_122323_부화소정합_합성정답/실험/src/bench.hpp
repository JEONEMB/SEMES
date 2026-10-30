#pragma once
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>
#include <vector>
namespace bench {
namespace fs=std::filesystem;
// 채택 control 측정기와 같은 영역 정의 (tracker.cpp의 full_crop, qr)
extern const cv::Rect kCrop;
extern const cv::Rect kQr[3];
extern const cv::Point2d kBoardCenter;

// 알려진 변형. 내용이 p에서 s·R(θ)·(p−c)+c+t 로 옮겨진 영상을 만든다. c는 판 중심(1906,1070).
struct Condition {
    std::string name,group;
    double tx=0,ty=0,angle_deg=0,scale=1;
    double gain=1,offset=0,blur=0,noise=0;
    int occlusions=0;unsigned seed=0;
};
struct FixedPoints {std::vector<cv::Point2d> p;std::vector<int> region;std::vector<unsigned char> selected;};
FixedPoints load_points(const fs::path& file);
cv::Point2d warp_point(const Condition& c,cv::Point2d p);
// center(3840×2160 uint8)의 kCrop 영역만 변형해 같은 크기 영상을 돌려준다. 평행이동만이면 FFT 위상 이동, 회전·배율이 있으면 Lanczos4.
cv::Mat synthesize(const cv::Mat& center,const Condition& c);
// 같은 자세로 반복 촬영한 두 영상의 평탄부 차이로 촬영 잡음 σ를 추정한다 (MAD 기반, 차이의 σ/√2).
double estimate_noise(const cv::Mat& a,const cv::Mat& b);

// d: 점 변위 추정. ok: 수치 결과가 나왔는지. own_valid: 그 방법 자체의 유효성 검사(있으면) 통과 여부. fb: 정·역방향 오차(LK 계열만).
struct Estimate {std::vector<cv::Point2d> d;std::vector<unsigned char> ok,own_valid;std::vector<double> fb;};
class Methods {
public:
    Methods(const cv::Mat& center,const FixedPoints& points,const fs::path& control_assets);
    ~Methods();
    static int count();
    static const char* name(int method);
    Estimate run(int method,const cv::Mat& image)const;
private:
    struct Impl;Impl* impl_;
};
}
