// =====================================================================================
// center_feature.hpp — 기준(center) 영상 특징 + 깊이별 랜드마크 시차(parallax)
//
//  관찰 (2026-10-21, 전체 화면 3840×2160 분석)
//    - 카메라가 로봇과 함께 움직이고, QR 플레이트(스테이션)와 뒤쪽 벽은 고정되어 있음
//    - Y 평행이동 : 화면 이동이 깊이에 반비례  → QR 플레이트 14.2 px/mm, 아래 벽 나사 11.1 px/mm (시차)
//    - Θ 회전     : 화면 이동이 깊이와 무관     → 플레이트 80.0 px/°, 벽 81.2 px/°
//    → 깊이가 다른 랜드마크 두 묶음만 있으면 Y 와 Θ 가 두 개의 독립 방정식으로 바로 분리됨
//       (QR 중심 좌표 하나만으로는 Y 1 mm ↔ Θ 0.177° 를 구분할 수 없음)
//
//  근거
//    - Hartley & Zisserman, Multiple View Geometry : 순수 회전은 깊이와 무관한 호모그래피,
//      평행이동은 깊이에 반비례하는 시차를 만든다
//    - Guizar-Sicairos, Thurman, Fienup (2008), Opt. Lett. 33(2) : 위상 상관 + 행렬곱 DFT 업샘플링으로
//      1/100 px 이동 추정 (본 코드 xcorrShift)
//
//  처리
//    1) 랜드마크 : 영상 좌표 ROI (landmarks.yml 로 수정 가능) — 플레이트 QR 3개·라벨, 스테이션 너트·브래킷,
//                  위쪽 벽 나사, 아래쪽 벽 나사 (반사가 심한 플레이트 나사 구멍은 제외)
//    2) 이동 측정 : 플레이트 띠로 거친 이동 → 랜드마크마다 정수 이동 후 위상 상관 업샘플링 (반복 3회)
//    3) 모델      : 랜드마크 k 의 가로 이동 dx_k = a_k·Y + b_k·Θ + q_k·Θ² + c_k,  잔차 공분산 Σ
//    4) 추정      : 일반화 최소제곱 (Gauss-Newton)
//
//  OpenCV 3.4.15 (dft, mulSpectrums, gemm, FileStorage) 만 사용
// =====================================================================================
#ifndef __CENTER_FEATURE_HPP__
#define __CENTER_FEATURE_HPP__

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace cf {

// ---------------------------------------------------------------------------------
// 랜드마크 (영상 좌표, 기준 영상 center.bmp 기준)
// ---------------------------------------------------------------------------------
struct Landmark {
    std::string name;
    std::string group;      // plate / station / wall_upper / wall_lower
    int cx = 0, cy = 0;     // ROI 중심
    int hw = 60, hh = 60;   // ROI 반폭·반높이
    bool use = true;        // 모델에 사용할지
};

std::vector<Landmark> defaultLandmarks();
bool saveLandmarks(const std::string& path, const std::vector<Landmark>& lm);
bool loadLandmarks(const std::string& path, std::vector<Landmark>& lm);

// ---------------------------------------------------------------------------------
// 서브픽셀 이동 : b(x) = a(x − s) 일 때 s = (dx, dy) 를 1/upsample px 단위로
// ---------------------------------------------------------------------------------
struct Shift { double dx = 0, dy = 0, peak = 0; bool ok = false; };
Shift xcorrShift(const cv::Mat& a64, const cv::Mat& b64, int upsample = 100);

// 영상 한 장의 랜드마크 이동 (기준 영상 대비)
struct Measure {
    bool ok = false;
    double coarseDx = 0, coarseDy = 0;
    std::vector<Shift> s;   // 랜드마크 순서
    double ms = 0;
};

class LandmarkTracker {
public:
    bool setReference(const cv::Mat& gray8, const std::vector<Landmark>& lm);
    Measure measure(const cv::Mat& gray8) const;
    const std::vector<Landmark>& landmarks() const { return lm_; }
private:
    cv::Mat ref64_;
    std::vector<Landmark> lm_;
    std::vector<cv::Mat> tpl_;          // 랜드마크별 기준 조각
};

// ---------------------------------------------------------------------------------
// 영상 품질 (기준 영상 특징용)
// ---------------------------------------------------------------------------------
struct Quality {
    double mean = 0;        // 평균 밝기
    double sharp = 0;       // 라플라시안 분산 (초점)
    double noise = 0;       // 잡음 σ 추정 (Immerkær 1996, 평탄 영역 라플라시안 MAD)
    double satFrac = 0;     // 포화 화소 비율 (≥ 250)
};
Quality quality(const cv::Mat& gray8, const cv::Rect& roi);

// ---------------------------------------------------------------------------------
// 시차 모델 : dx_k = a_k Y + b_k Θ + q_k Θ² + c_k
// ---------------------------------------------------------------------------------
struct Sample { std::vector<double> dx; double y = 0, t = 0; };
struct Estimate { double y = 0, t = 0, sdY = 0, sdT = 0; bool valid = false; };

struct ParallaxModel {
    std::vector<std::string> names;     // 사용한 랜드마크 이름
    std::vector<int> idx;               // 트래커 랜드마크 번호
    std::vector<double> a, b, q, c;
    cv::Mat cov;                        // 잔차 공분산 (K×K)
    int nTrain = 0;
    std::string note;
    bool ok = false;

    bool save(const std::string& path) const;
    bool load(const std::string& path);
};

ParallaxModel fitModel(const std::vector<Sample>& train, const std::vector<int>& idx,
                       const std::vector<Landmark>& lm, const std::string& note);
// dx : 트래커 전체 랜드마크 이동 (모델이 idx 로 골라 씀), refDx : 같은 세션 기준 영상 이동 (없으면 nullptr)
Estimate estimate(const ParallaxModel& m, const std::vector<double>& dx, const std::vector<double>* refDx = nullptr);

} // namespace cf

#endif // __CENTER_FEATURE_HPP__
