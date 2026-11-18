#pragma once
#include <opencv2/opencv.hpp>
#include <functional>
#include <string>
#include <vector>
namespace bench {
// 입력은 kCrop 영역 8비트 영상과 같은 크기의 마스크. 출력 좌표는 kCrop 기준.
struct Detector {
    std::string name;
    bool descriptor=false;   // true면 기술자 짝짓기 경로도 평가
    bool trackable=false;    // true면 center 검출점을 LK로 추적하는 경로도 평가
    int norm=cv::NORM_HAMMING;
    std::function<void(const cv::Mat& image,const cv::Mat& mask,std::vector<cv::KeyPoint>& keypoints,cv::Mat* descriptors)> run;
};
std::vector<Detector> detectors();
// Förstner 연산자: 구조 텐서 w=det/trace, 원형도 q=4det/trace² 로 후보를 고르고, 창 안 기울기 직선들의 최소제곱 교점으로 부화소 위치를 구한다.
std::vector<cv::Point2f> forstner(const cv::Mat& image,const cv::Mat& mask,int max_points,double min_distance);
}
