// #50~#55 가 함께 쓰는 측정 묶음: 사진 281장 crop, center 630점, v1 경계 모델 추적, v2 GN17 패치 추적.
#pragma once
#include <cstdio>
#include <fstream>
#include "yt630.hpp"

namespace yt {

struct Measure630 {
    std::vector<Img> imgs;
    std::vector<cv::Mat> crops;
    std::vector<QRInfo> qrs;
    std::vector<Corner> all;      // 755 구조 모서리
    std::vector<int> use;         // black_neighbors != 2 (630)
    std::vector<double> X, Y;     // center 맞춤 위치 (corners_280.csv center 행)
    std::vector<int> ids, gxs, gys;
    std::vector<std::vector<Tracked>> edge;  // [img][630] v1 경계 모델 추적
};

// v1 경계 추적까지 할지(with_edge). 시간이 오래 걸린다(281장 x 630점 x 최대 4회 TRF 적합).
inline Measure630 measure630(bool with_edge) {
    Measure630 M;
    M.imgs = images_281();
    std::printf("read %zu images\n", M.imgs.size());
    M.crops = read_all_crops(M.imgs);
    const cv::Mat& cen = M.crops[0];
    M.qrs = detect_qr(cen);
    if (M.qrs.size() != 3) throw std::runtime_error("QR 3개를 못 찾음");
    M.all = structural_corners(M.qrs);
    refine_center(cen, M.all);
    for (auto& c : M.all) if (c.black != 2) M.use.push_back(c.id);
    // corners_280.csv 의 center 행 = center 사진에 같은 추적을 한 결과
    auto ct = track_corners(cen, cen, M.all, M.use);
    for (size_t u = 0; u < M.use.size(); ++u) {
        const auto& c = M.all[M.use[u]];
        M.X.push_back(ct[u].x); M.Y.push_back(ct[u].y);
        M.ids.push_back(c.id); M.gxs.push_back(c.gx); M.gys.push_back(c.gy);
    }
    std::printf("corners: %zu structural, %zu used\n", M.all.size(), M.use.size());
    if (with_edge) {
        M.edge.resize(M.imgs.size());
        M.edge[0] = ct;
#pragma omp parallel for schedule(dynamic) num_threads(4)
        for (int i = 1; i < (int)M.imgs.size(); ++i) M.edge[i] = track_corners(M.crops[i], cen, M.all, M.use);
        std::printf("edge tracking done\n");
    }
    return M;
}

// D(중심 뺀 값, float32 -> float64)
inline void center_relative(const Tracks& t, std::vector<std::vector<double>>& Dx, std::vector<std::vector<double>>& Dy) {
    size_t M = t.Dx.size(), N = t.Dx[0].size();
    Dx.assign(M, std::vector<double>(N));
    Dy.assign(M, std::vector<double>(N));
    for (size_t i = 0; i < M; ++i)
        for (size_t p = 0; p < N; ++p) {
            Dx[i][p] = (double)t.Dx[i][p] - (double)t.Dx[0][p];
            Dy[i][p] = (double)t.Dy[i][p] - (double)t.Dy[0][p];
        }
}

// run.py 품질 가중: 1/(1+(RMSE/6)^2+(FB/.15)^2), 무효 x .05, 하한 .001 (float32 로 계산되던 그대로)
inline std::vector<std::vector<double>> quality_weights(const Tracks& t) {
    std::vector<std::vector<double>> W(t.Q.size());
    for (size_t i = 0; i < t.Q.size(); ++i) {
        W[i].resize(t.Q[i].size());
        for (size_t p = 0; p < t.Q[i].size(); ++p) {
            float a = t.Q[i][p][0] / 6.0f, b = t.Q[i][p][4] / 0.15f;
            float w = 1.0f / (1.0f + a * a + b * b);
            w = (float)((double)w * (t.V[i][p] ? 1.0 : 0.05));
            w = std::max(w, 0.001f);
            W[i][p] = w;
        }
    }
    return W;
}

// 개발용: 중간값을 CSV 로 남긴다 (환경변수 YT_DUMP 가 있을 때만, 결과/ 밖)
inline void dump_csv(const std::string& name, const std::vector<std::vector<double>>& rows) {
    const char* d = std::getenv("YT_DUMP");
    if (!d) return;
    std::ofstream f(std::string(d) + "/" + name);
    f.precision(17);
    for (auto& r : rows) {
        for (size_t k = 0; k < r.size(); ++k) f << (k ? "," : "") << r[k];
        f << "\n";
    }
}

}  // namespace yt
