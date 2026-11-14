#pragma once
// 기준 사진을 바꿔 "다른 환경에서 카메라가 볼 화면"을 만드는 함수들.
// make_environment_images, make_site_images, check_new_installation 이 같이 쓴다. 모두 회색 8비트 사진을 받아 같은 크기로 돌려준다.
#include <opencv2/core.hpp>
#include <filesystem>
#include <string>

namespace scene {

const cv::Rect kView(700, 800, 2440, 1050);   // 문서 그림에 쓰는 판 둘레 (사진 3840×2160 기준)
const unsigned kSeed = 20261115;              // 무작위 무늬를 만들 때 쓰는 씨앗 (누가 돌려도 같은 그림)

struct Layout {
    cv::Rect qr[3];        // QR 상자, 왼쪽부터
    cv::Point screw[4];    // 나사 중심
};

// ---- 읽고 쓰기 ----
cv::Mat read_gray(const std::filesystem::path& path);
void save_png(const std::filesystem::path& path, const cv::Mat& picture);
Layout read_layout(const std::filesystem::path& data_folder);   // data/reference.json
// 판 둘레를 잘라 절반 크기로 줄이고 왼쪽 위에 제목을 쓴다 (컬러 3채널로 돌려줌)
cv::Mat labeled_view(const cv::Mat& picture, const std::string& title);

// ---- 카메라 위치, 거리, 렌즈 ----
cv::Mat scale_rotate(const cv::Mat& gray, double scale, double angle_deg);   // 사진 가운데 둘레로 배율(거리), 회전
cv::Mat shift(const cv::Mat& gray, double right_px, double down_px);
cv::Mat tilt(const cv::Mat& gray, double tilt_deg, double focal_px);          // 가로축 둘레로 기울어져 원근이 생김
cv::Mat distort(const cv::Mat& gray, double k1);                              // 렌즈 왜곡 (k1 > 0 술통형)

// ---- 대상물 ----
cv::Mat replace_qr(const cv::Mat& gray, const cv::Rect& box, unsigned seed);  // QR 안쪽 칸을 무작위 무늬로 (위치 사각형 3개는 남김)
cv::Mat scale_qr(const cv::Mat& gray, const cv::Rect& box, double factor);   // QR 하나를 자기 가운데 둘레로 factor 배 (인쇄 크기). 원래 자리는 판 색으로 덮음
cv::Rect scaled_box(const cv::Rect& box, double factor);                     // scale_qr 뒤 QR 상자
cv::Mat remove_screw(const cv::Mat& gray, cv::Point center);
cv::Mat different_screw(const cv::Mat& gray, cv::Point center);

// ---- 조명 ----
cv::Mat brightness(const cv::Mat& gray, double gain, double offset);
cv::Mat gamma(const cv::Mat& gray, double g);
cv::Mat side_light(const cv::Mat& gray, double left, double right);          // 왼쪽에서 오른쪽으로 밝기 배율이 바뀜
cv::Mat glare(const cv::Mat& gray, cv::Point center, double radius_px, double strength);
cv::Mat banding(const cv::Mat& gray, double depth, double period_px);        // 형광등 깜빡임 줄무늬

// ---- 공정 환경 ----
cv::Mat defocus(const cv::Mat& gray, double sigma_px);
cv::Mat motion_blur(const cv::Mat& gray, int length_px, bool horizontal);
cv::Mat heat_shimmer(const cv::Mat& gray, double amplitude_px, double cell_px, unsigned seed);
cv::Mat fog(const cv::Mat& gray, double amount, double sigma_px);
cv::Mat spots(const cv::Mat& gray, int count, double min_radius, double max_radius, bool droplets, unsigned seed);
cv::Mat ghost(const cv::Mat& gray, double right_px, double down_px, double weight);   // 유리, 창의 이중 상
cv::Mat glow_from_top(const cv::Mat& gray, double strength, double center_y, double width);
cv::Mat vignette(const cv::Mat& gray, cv::Point center, double radius_px);   // 둥근 창 밖이 어두워짐
cv::Mat noise(const cv::Mat& gray, double sigma, unsigned seed);
cv::Mat jpeg(const cv::Mat& gray, int quality);

}   // namespace scene
