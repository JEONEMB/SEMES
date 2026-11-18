#pragma once
// 환경 변화 실험용 합성 영상
// 영상 변환: 8비트 회색 입력, 같은 크기 출력(labeled_view 제외)
#include <opencv2/core.hpp>
#include <filesystem>
#include <string>

namespace scene {

const cv::Rect kView(700, 800, 2440, 1050);   // 문서 그림용 판 주변 영역
const unsigned kSeed = 20261115;              // 같은 무작위 무늬를 다시 만들기 위한 시작값
const double kToleranceYmm = 0.045, kToleranceTdeg = 0.01;   // 통과 기준 (tools/common/coefficient_fit.hpp와 같은 값)

struct Layout {
    cv::Rect qr[3];        // QR 사각 영역, 왼쪽부터
    cv::Point screw[4];    // 나사 중심
};

// save_png: PNG 저장
//   입력: path (파일 경로), picture (사진)
//   출력: 없음 (파일 저장)
void save_png(const std::filesystem::path& path, const cv::Mat& picture);

// read_layout: 기준 좌표 읽기
//   입력: data_folder (reference.json 포함)
//   출력: QR 영역과 나사 중심
Layout read_layout(const std::filesystem::path& data_folder);

// labeled_view: 제목 포함 사례 그림 생성
//   입력: picture (사진), title (제목)
//   출력: 판 주변을 절반 크기로 줄인 컬러 사진(BGR)
cv::Mat labeled_view(const cv::Mat& picture, const std::string& title);

// scale_rotate: 영상 중심 기준 배율과 회전
//   입력: gray (회색 사진), scale (크기 배율), angle_deg (회전 각도)
//   출력: 변환 영상
cv::Mat scale_rotate(const cv::Mat& gray, double scale, double angle_deg);

// shift: 영상 평행이동
//   입력: gray (회색 사진), right_px와 down_px (오른쪽, 아래쪽 이동 px)
//   출력: 이동 영상
cv::Mat shift(const cv::Mat& gray, double right_px, double down_px);

// tilt: 판을 앞뒤로 기울여 본 모습 만들기
//   입력: gray (회색 사진), tilt_deg (기울기 각도), focal_px (카메라 초점거리 px)
//   출력: 기울어진 영상
cv::Mat tilt(const cv::Mat& gray, double tilt_deg, double focal_px);

// distort: 렌즈 왜곡 합성
//   입력: gray (회색 사진), k1 (휘어짐 정도, 양수이면 바깥으로 볼록)
//   출력: 왜곡 영상
cv::Mat distort(const cv::Mat& gray, double k1);

// replace_qr: QR 내부 무늬 변경
//   입력: gray (회색 사진), box (QR 영역), seed (무작위 시작값)
//   출력: QR 모서리의 위치 표시 사각형 3개를 유지한 사진
cv::Mat replace_qr(const cv::Mat& gray, const cv::Rect& box, unsigned seed);

// scale_qr: QR 인쇄 크기 변경
//   입력: gray (회색 사진), box (QR 영역), factor (배율)
//   출력: 원래 자리를 판 색으로 덮은 변환 영상
cv::Mat scale_qr(const cv::Mat& gray, const cv::Rect& box, double factor);

// scaled_box: 배율 적용 QR 영역 계산
//   입력: box (QR 영역), factor (크기 배율)
//   출력: 변경 영역
cv::Rect scaled_box(const cv::Rect& box, double factor);

// remove_screw: 나사 제거 합성
//   입력: gray (회색 사진), center (중심 위치)
//   출력: 나사 영역을 덮은 영상
cv::Mat remove_screw(const cv::Mat& gray, cv::Point center);

// different_screw: 나사 모양 변경 합성
//   입력: gray (회색 사진), center (중심 위치)
//   출력: 변경 영상
cv::Mat different_screw(const cv::Mat& gray, cv::Point center);

// brightness: 밝기를 곱하고 더하여 조정
//   입력: gray (회색 사진), gain (밝기 배율), offset (더할 밝기)
//   출력: 밝기 조정 사진. 밝기값은 0~255로 제한
cv::Mat brightness(const cv::Mat& gray, double gain, double offset);

// gamma: 어두운 부분과 밝은 부분의 밝기 비율 변경
//   입력: gray (회색 사진), g (밝기 비율 조정값)
//   출력: 밝기 변환 영상
cv::Mat gamma(const cv::Mat& gray, double g);

// side_light: 좌우 밝기 차이 합성
//   입력: gray (회색 사진), left (왼쪽 밝기 배율), right (오른쪽 밝기 배율)
//   출력: 조명 변화 영상
cv::Mat side_light(const cv::Mat& gray, double left, double right);

// glare: 일부 위치의 빛 반사 만들기
//   입력: gray (회색 사진), center (중심 위치), radius_px (반지름 px), strength (빛 강도)
//   출력: 반사광 영상
cv::Mat glare(const cv::Mat& gray, cv::Point center, double radius_px, double strength);

// banding: 조명 줄무늬 합성
//   입력: gray (회색 사진), depth (줄무늬 강도), period_px (줄 간격 px)
//   출력: 줄무늬 영상
cv::Mat banding(const cv::Mat& gray, double depth, double period_px);

// defocus: 초점 흐림 합성
//   입력: gray (회색 사진), sigma_px (흐림 정도 px)
//   출력: 주변 밝기를 섞어 흐리게 만든 사진
cv::Mat defocus(const cv::Mat& gray, double sigma_px);

// motion_blur: 이동 흐림 합성
//   입력: gray (회색 사진), length_px (흐려질 길이 px), horizontal (가로 방향 여부)
//   출력: 이동 방향으로 흐린 영상
cv::Mat motion_blur(const cv::Mat& gray, int length_px, bool horizontal);

// heat_shimmer: 열에 의한 영상 흔들림 합성
//   입력: gray (회색 사진), amplitude_px (흔들림 크기 px), cell_px (흔들림 간격 px), seed (무작위 시작값)
//   출력: 변형 영상
cv::Mat heat_shimmer(const cv::Mat& gray, double amplitude_px, double cell_px, unsigned seed);

// fog: 안개 합성
//   입력: gray (회색 사진), amount (안개 농도), sigma_px (흐림 정도 px)
//   출력: 흐림과 밝기 변화 영상
cv::Mat fog(const cv::Mat& gray, double amount, double sigma_px);

// spots: 오염 또는 물방울 합성
//   입력: gray (회색 사진), count (개수), min_radius (최소 반지름), max_radius (최대 반지름), droplets (물방울 여부), seed (무작위 시작값)
//   출력: 얼룩 영상
cv::Mat spots(const cv::Mat& gray, int count, double min_radius, double max_radius, bool droplets, unsigned seed);

// ghost: 사진이 두 겹으로 보이는 모습 만들기
//   입력: gray, right_px와 down_px (오른쪽, 아래쪽 이동 px), weight (겹칠 비중)
//   출력: 이동한 상을 겹친 영상
cv::Mat ghost(const cv::Mat& gray, double right_px, double down_px, double weight);

// glow_from_top: 위쪽 빛 번짐 합성
//   입력: gray (회색 사진), strength (빛 강도), center_y (세로 중심), width (빛의 폭)
//   출력: 빛 번짐 영상
cv::Mat glow_from_top(const cv::Mat& gray, double strength, double center_y, double width);

// vignette: 원형 창 밖 어둡게 처리
//   입력: gray (회색 사진), center (중심 위치), radius_px (반지름 px)
//   출력: 원 밖이 어두운 사진
cv::Mat vignette(const cv::Mat& gray, cv::Point center, double radius_px);

// noise: 밝기 잡음 합성
//   입력: gray (회색 사진), sigma (잡음 크기), seed (무작위 시작값)
//   출력: 잡음 영상
cv::Mat noise(const cv::Mat& gray, double sigma, unsigned seed);

// jpeg: JPEG 압축 손실 합성
//   입력: gray (회색 사진), quality (압축 품질)
//   출력: 압축 후 복원 영상
cv::Mat jpeg(const cv::Mat& gray, int quality);

}   // scene
