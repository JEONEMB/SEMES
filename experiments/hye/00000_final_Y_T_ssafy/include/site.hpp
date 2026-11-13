#pragma once
#include <opencv2/core.hpp>
#include <filesystem>
#include <string>

// 장비(촬영 환경)마다 다른 화면 위치를 한곳에 모은다.
// 원본 모듈(experiments/hye/00000_final_Y_T)은 이 값들을 SEMES 센터 사진 기준 상수로 코드에 적어 두었다.
// 이 복사본은 assets/site.json 에서 읽는다. site.json 이 없으면 원본 상수(SEMES)를 그대로 쓴다.
namespace site {
struct Geometry {
    std::string center_sha256;   // assets/center.bmp 의 SHA-256
    cv::Rect full_crop;          // 536점 추적이 보는 영역
    cv::Rect qr[3];              // 센터 사진에서 QR 3개의 상자
    cv::Point screws[4];         // 센터 사진에서 나사 4개의 중심
    int search_x, search_y;      // 536점 추적의 QR 정수 이동 탐색 반폭(px)
};
// assets: center.bmp 가 있는 폴더. 스레드를 만들기 전에 한 번 부른다.
void load(const std::filesystem::path& assets);
const Geometry& get();
}
