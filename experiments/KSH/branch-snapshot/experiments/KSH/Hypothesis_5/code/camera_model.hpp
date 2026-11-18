// =====================================================================================
// camera_model.hpp — 운동 모델 : 로봇 자세 (Y, Θ) 가 바뀔 때 영상이 어떻게 움직이는가
//
//  좌표 (기준 카메라 = 세계 좌표) : x 오른쪽, y 아래, z 앞 (mm)
//  카메라 내부 : K = [fx 0 cx; 0 fy cy; 0 0 1]   (가로 4096 → 3840 축소 → fx = fy × 15/16)
//              방사 왜곡 (정규화 좌표) : x_d = x·(1 + k1·r²), y_d = y·(1 + k1·r²),  r² = x² + y²
//
//  자세 → 카메라 움직임
//    Y  : 카메라 중심이 −x 로 Y mm 이동            → 장면은 +x 로 fx·Y/Z 만큼 이동 (깊이 Z 에 반비례)
//    Θ  : 광학 중심에서 앞쪽(+z) a mm 에 있는 세로축(y) 둘레로 회전
//         R = R_y(−Θ),  C = A − R·A,  A = (0, 0, a)
//         → 장면 이동 ≈ fx·Θ·(1 + u²/fx²) − fx·a·Θ/Z   (앞항: 순수 회전, 뒷항: 축이 어긋나 생기는 평행 이동)
//         → 세로 성분 Δv ≈ Θ·u·v/fx  (회전에만 있음)
//    현재 카메라 좌표 : Xc = Rᵀ (X − C),  투영 : u = fx·Xc.x/Xc.z + cx,  v = fy·Xc.y/Xc.z + cy
//
//  깊이 Z 의 정면 평면 → 평면 호모그래피 : H = K · Rᵀ · (Z·I − C·e3ᵀ) · K⁻¹  (기준 화소 → 현재 화소, k1 = 0 일 때)
//  왜곡이 있으면 화소마다 mapToReference 로 역사상 → cv::remap 으로 영상 합성
// =====================================================================================
#ifndef MM_CAMERA_MODEL_HPP
#define MM_CAMERA_MODEL_HPP

#include <opencv2/core.hpp>
#include <array>
#include <string>

namespace mm {

// 로봇 자세 (기준 = teaching 위치 대비)
struct Pose {
    double y_mm = 0.0;   // Y 이동 (mm)
    double t_deg = 0.0;  // Θ 회전 (°)
};

struct CameraModel {
    // ---- 카메라 (config/camera_model.yml 로 덮어쓸 수 있음) ----
    double fx = 5486.0;            // 가로 초점거리 (px) = 14.25 px/mm × 385 mm
    double fy = 5852.0;            // 세로 초점거리 (px) = fx × 16/15 (가로 축소 보정)
    double cx = 1920.0, cy = 1080.0;
    double k1 = 0.0;               // 방사 왜곡 (음수 = 술통형). 제원 미상 → batch 의 데이터 맞춤 값 참고
    int width = 3840, height = 2160;
    // ---- 장면 ----
    double z_plate = 385.0;        // 카메라–QR 플레이트 거리 (mm, 렌즈 앞면 기준 제원)
    double z_bg = 494.0;           // 배경(하단 벽·door 쪽) 대표 깊이 (mm) = fx / 11.1 px/mm (실측 Y 감도)
    double z_wall_upper = 418.0;   // 위쪽 벽 깊이 (mm) = fx / 13.1 px/mm
    double axis_a = 63.1;          // Θ 회전축 위치 : 광학 중심에서 플레이트 쪽 (mm). 실측 Θ 감도 80.06 px/° 로 역산
    cv::Point2d plate_center{1901.3, 1068.4}; // 기준 영상의 플레이트 중심 (px, EDA C-1)
    double qr_pitch_mm = 18.0;     // QR 중심 간격 = 13 + 5 mm
    double qr_size_mm = 13.0;      // QR 한 변

    bool load(const std::string& path);        // 없으면 false (기본값 유지)
    bool save(const std::string& path) const;

    cv::Matx33d K() const;
    // 자세 → 현재 카메라 회전 R (카메라→세계) 과 중심 C (세계 좌표, mm)
    void motion(const Pose& pose, cv::Matx33d& R, cv::Vec3d& C) const;
    // 기준 영상의 화소 p 를 깊이 Z 평면으로 역투영
    cv::Vec3d backProject(const cv::Point2d& p, double Z) const;
    // 세계 점 → 현재 자세의 화소
    cv::Point2d project(const cv::Vec3d& Xw, const Pose& pose) const;
    // 현재 영상의 화소 q 가 보는 깊이 Z 평면의 점이 기준 영상에서 찍힌 화소 (영상 합성용 역사상)
    cv::Point2d mapToReference(const cv::Point2d& q, double Z, const Pose& pose) const;
    // 정규화 좌표 ↔ 화소 (왜곡 포함)
    cv::Point2d toPixel(double xn, double yn) const;
    cv::Point2d toNormalized(const cv::Point2d& p) const;
    // 기준 화소 p (깊이 Z) 가 현재 영상에서 이동한 양 — 정확한 3D 계산
    cv::Point2d displacement(const cv::Point2d& p, double Z, const Pose& pose) const;
    // 같은 양의 작은 각 근사식 (설명용 : 위 헤더의 식 그대로)
    cv::Point2d displacementLinear(const cv::Point2d& p, double Z, const Pose& pose) const;
    // 깊이 Z 정면 평면의 호모그래피 (기준 화소 → 현재 화소) — warpPerspective 로 영상 합성
    cv::Matx33d planeHomography(double Z, const Pose& pose) const;
    // QR 3개(L, C, R)의 꼭짓점 (TL, TR, BR, BL) — 현재 자세에서
    std::array<std::array<cv::Point2d, 4>, 3> qrCorners(const Pose& pose) const;
    // 기준 영상에서 QR k (0=L, 1=C, 2=R) 중심 화소
    cv::Point2d qrCenterRef(int k) const;
};

// QR 특징 예측 (현재 자세)
struct QrPrediction {
    double plate_cx = 0, plate_cy = 0;  // QR 3개 중심 평균 (px)
    double w[3] = {0, 0, 0};            // 가로 폭 (위·아래 변 평균, px)
    double h[3] = {0, 0, 0};            // 세로 높이 (왼·오른 변 평균, px)
    double w_diff = 0;                  // w_R − w_L (px) : Θ 만의 원근 단서
    double h_diff = 0;                  // h_R − h_L (px)
};
QrPrediction predictQr(const CameraModel& cam, const Pose& pose);

// 감도 (자세 0 근처 중앙 차분) — 실측값과 비교하는 표에 사용
struct Sensitivity {
    double plate_per_mm = 0, plate_per_deg = 0;   // 플레이트 중심 x 이동 (px/mm, px/°)
    double wdiff_per_mm = 0, wdiff_per_deg = 0;   // 폭 차 R−L (px/mm, px/°)
    double hdiff_per_mm = 0, hdiff_per_deg = 0;   // 높이 차 R−L
    double bg_per_mm = 0, bg_per_deg = 0;         // 배경 깊이에서의 이동 (px/mm, px/°)
    double equal_shift_deg_per_mm = 0;            // 플레이트에서 Y 1 mm 와 같은 이동을 만드는 Θ (°)
};
Sensitivity sensitivities(const CameraModel& cam);

// 파일 이름 → 라벨. set_id : 1 = 1차 (center, Y±a, T±b), 2 = 2차 (centerN, "Y+a T+b (n)")
bool parseLabel(const std::string& stem, Pose& pose, int& set_id);

} // namespace mm

#endif
