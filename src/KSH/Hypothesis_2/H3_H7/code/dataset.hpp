// =====================================================================================
// dataset.hpp — 영상 목록 · 이름 해석 · 학습/검증 분할 (H3 · H6 · H7 공통)
//
//  img/ 한 폴더의 영상을 이름으로 구분한다.
//    center.bmp               : 1차 세트 기준 (10/04)          — 대응점 기준 영상
//    center1~3.bmp            : 2차 세트 기준 (10/15)          — 2차 세트 원점 (center1)
//    Y±a.bmp, T±b.bmp         : 1차 세트 (한 축씩)             — 값 순서 짝수 번째 학습, 홀수 번째 검증
//    Y+a T+b (n).bmp          : 2차 세트 (Y·Θ 동시, 같은 자세 3장)
//  한글·공백 경로를 위해 std::filesystem::path 와 바이트 읽기 + imdecode 를 쓴다.
// =====================================================================================
#ifndef __H3_DATASET_HPP__
#define __H3_DATASET_HPP__

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>
#include <vector>

namespace h3 {

namespace fs = std::filesystem;

struct Item {
    fs::path    path;
    std::string name;          // 파일 이름 (UTF-8)
    int    set = 1;            // 1 : 1차 세트, 2 : 2차 세트
    int    shot = 0;           // 2차 세트 (1)(2)(3), centerN 은 N
    bool   isCenter = false;
    double y_mm = 0, t_deg = 0;     // 지령값 (파일 이름)
    bool   train = false;      // 1차 세트 짝수 번째
    bool   test = false;       // 1차 세트 홀수 번째
};

std::string u8(const fs::path& p);
cv::Mat     readGray(const fs::path& p);
bool        parseName(const std::string& stem, Item& it);
std::vector<Item> listImages(const fs::path& imgDir);   // 정렬 + 학습/검증 표시
int         findItem(const std::vector<Item>& v, const std::string& name);
// 같은 세트·자세·촬영 번호로 찾기 (이름 표기 차이 "Y+0.1" / "Y+0.10" 무시)
int         findPose(const std::vector<Item>& v, int set, double y_mm, double t_deg, int shot, bool isCenter);

} // namespace h3

#endif
