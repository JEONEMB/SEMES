// =====================================================================================
// phys_model.hpp — 가설 H3 (완전 물리 모델) : 카메라 + 플레이트 평면 + Y 이동 방향 + Θ 회전축
//
//  기준 카메라 좌표 (x 오른쪽, y 아래, z 앞)
//    카메라   : 초점 f, 주점 (cx, cy), 방사 왜곡 k1
//    플레이트 : 법선 n(na, nb), 거리 d (고정 323 mm — f/d 만 관측됨)
//    Y 이동   : 방향 e(e1, e2) — 기본 −x (Y+ 이면 영상이 +x 로 이동)
//    Θ 회전   : 축 방향 ω(w1, w2) — 기본 −y, 축 위치 A = (ax, 0, az) mm
//    라벨 보정 : 원점 차 (Y0, T0), 접근 방향 차 (bY, bT) — 라벨 = 지령값 + 원점 + 부호 × 절반 백래시
//  카메라 중심 c = Y·e + (I − R(Θ))·A (Θ 단이 Y 단 위, TonY), 점 투영 x = K·dist(Rᵀ(X − c))
//  기준 영상 화소를 평면으로 역투영해 3D 점을 만들고, 학습 영상에서 재투영 오차를 줄이도록 전역 매개변수를 맞춘다.
//  최적화 : Levenberg–Marquardt + soft-L1 가중(IRLS), 전역은 수치 미분, 영상별 세로 이동 ν 는 해석 미분.
// =====================================================================================
#ifndef __H3_PHYS_MODEL_HPP__
#define __H3_PHYS_MODEL_HPP__

#include "qr_track.hpp"
#include <map>
#include <string>
#include <vector>

namespace h3 {

struct PhysParams {
    std::map<std::string, double> v = {
        {"f", 4584.0}, {"cx", 1920.0}, {"cy", 1080.0}, {"k1", 0.0}, {"na", 0.0}, {"nb", 0.0}, {"d", 323.0},
        {"e1", 0.0}, {"e2", 0.0}, {"w1", 0.0}, {"w2", 0.0}, {"ax", 0.0}, {"az", 0.0},
        {"bY", 0.0}, {"bT", 0.0}, {"Y0", 0.0}, {"T0", 0.0} };
    double operator[](const std::string& k) const { return v.at(k); }
};

// 기준 화소 → 평면 위 3D 점
std::vector<cv::Vec3d> platePoints(const std::vector<cv::Point2d>& P0, const PhysParams& g);
// (Y, Θ, 세로 이동 ν) 에서 투영
std::vector<cv::Point2d> projectPoints(const std::vector<cv::Vec3d>& X, double y_mm, double t_deg, double nu_px, const PhysParams& g);

struct PhysFitResult {
    PhysParams g;
    std::vector<double> nu;
    double rmsX = 0, rmsY = 0, cost = 0;
    int iters = 0;
    double seconds = 0;
};

// 학습 : imgIdx 영상들의 지령값 (y, t) 와 대응점으로 전역 매개변수 free 를 맞춤
PhysFitResult fitPhysModel(const std::vector<cv::Point2d>& P0, const std::vector<Track>& tracks,
                           const std::vector<int>& imgIdx, const std::vector<double>& y, const std::vector<double>& t,
                           const std::vector<std::string>& freeNames, const PhysParams& init, int maxIter = 100);

// 진단 : 영상별 잔차 dx 를 [1, u, u²] 로 분해 (평균, 기울기, 곡률, 영상 안 표준편차)
struct ResidualRow { double y_mm, t_deg, mean, slope, curv, within; };
std::vector<ResidualRow> residualDiag(const std::vector<cv::Point2d>& P0, const std::vector<Track>& tracks,
                                      const std::vector<int>& imgIdx, const std::vector<double>& y,
                                      const std::vector<double>& t, const PhysParams& g);

} // namespace h3

#endif
