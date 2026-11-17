// 반복 촬영에서 점 측정 잡음 공분산을 만든다. A1 과 같은 정의이며 A2·A6·A8 이 함께 쓴다.
//   raw 잔차   = 묶음 평균을 뺀 점 변위           (측정 잡음 + 반복 사이 실제 자세 차이)
//   affine 잔차 = raw 에서 QR별 아핀 1개를 더 뺀 것 (측정 잡음만)
// 표본이 282장뿐이라 1260차원 공분산은 특이하다. 그래서 표본 고유공간 + 바닥값으로 역행렬을 만든다.
#pragma once
#include "cache.hpp"
#include <opencv2/core.hpp>

namespace a0 {

const cv::Rect kQrBox[3] = {cv::Rect(1552, 969, 184, 198), cv::Rect(1809, 971, 184, 198), cv::Rect(2067, 973, 185, 198)};

// r(630) 에서 QR별 아핀 d = B[1,u,v] 를 뺀다. 1 px 넘는 점은 빼고 한 번 더 맞춘다.
inline void remove_qr_affine(const std::vector<Point>& pts, std::vector<cv::Point2d>& r,
                             const std::vector<unsigned char>& use, double outlier = 1.0) {
    for (int q = 0; q < 3; ++q) {
        std::vector<int> idx;
        for (int i = 0; i < kPoints; ++i)
            if (pts[i].region == q && use[i] && std::isfinite(r[i].x) && std::isfinite(r[i].y)) idx.push_back(i);
        if (idx.size() < 6) continue;
        cv::Point2d c(kQrBox[q].x + kQrBox[q].width / 2.0, kQrBox[q].y + kQrBox[q].height / 2.0);
        auto fit = [&](const std::vector<int>& rows) {
            cv::Mat X(int(rows.size()), 3, CV_64F), Y(int(rows.size()), 2, CV_64F), B;
            for (int k = 0; k < int(rows.size()); ++k) {
                X.at<double>(k, 0) = 1;
                X.at<double>(k, 1) = (pts[rows[k]].x - c.x) / 100;
                X.at<double>(k, 2) = (pts[rows[k]].y - c.y) / 100;
                Y.at<double>(k, 0) = r[rows[k]].x;
                Y.at<double>(k, 1) = r[rows[k]].y;
            }
            cv::solve(X, Y, B, cv::DECOMP_SVD);
            return B;
        };
        auto model = [&](const cv::Mat& B, int i) {
            double u = (pts[i].x - c.x) / 100, v = (pts[i].y - c.y) / 100;
            return cv::Point2d(B.at<double>(0, 0) + B.at<double>(1, 0) * u + B.at<double>(2, 0) * v,
                               B.at<double>(0, 1) + B.at<double>(1, 1) * u + B.at<double>(2, 1) * v);
        };
        cv::Mat B = fit(idx);
        std::vector<int> in;
        for (int i : idx) if (cv::norm(r[i] - model(B, i)) <= outlier) in.push_back(i);
        if (in.size() >= 6 && in.size() < idx.size()) B = fit(in);
        for (int i : idx) r[i] -= model(B, i);
    }
}

// 점 1260차원 + 선 48차원의 잡음 표본. 행 하나가 사진 한 장의 잔차다.
struct NoiseSample {
    cv::Mat point_rows;  // N x 1260 (x,y 교대)
    cv::Mat line_rows;   // N x 48
    int groups = 0, shots = 0, dropped = 0;
};

// shot_ok 가 true 인 사진만 쓴다 (예: 학습 fold 안의 사진). 630점이 모두 유효한 묶음만 쓴다.
// remove_affine=false 로 부르면 QR별 아핀을 빼지 않는다. 그 경우 잔차에는 반복 사이의 실제 자세 차이가 들어 있다.
inline NoiseSample measure_noise(const Cache& c, const std::function<bool(const Shot&)>& shot_ok, bool remove_affine = true) {
    std::map<int, std::vector<int>> group;
    for (auto& s : c.shots) if (s.measured && shot_ok(s)) group[s.group].push_back(s.idx);
    std::vector<std::vector<double>> prow, lrow;
    NoiseSample out;
    for (auto& g : group) {
        const auto& ids = g.second;
        const int m = int(ids.size());
        if (m < 2) continue;
        bool full = true;
        for (int i : ids) for (int p = 0; p < kPoints; ++p) if (!c.v(i)[p]) full = false;
        for (int i : ids) for (int b = 0; b < kLines; ++b) if (!c.line_ok(i)[b]) full = false;
        if (!full) { ++out.dropped; continue; }
        ++out.groups;
        std::vector<cv::Point2d> mean(kPoints, {0, 0});
        std::vector<double> lmean(kLines, 0);
        for (int i : ids) {
            for (int p = 0; p < kPoints; ++p) mean[p] += c.dz(i, p);
            for (int b = 0; b < kLines; ++b) lmean[b] += c.line(i)[b * 2];
        }
        for (auto& v : mean) v *= 1.0 / m;
        for (auto& v : lmean) v /= m;
        const double scale = std::sqrt(double(m) / (m - 1));  // 묶음 평균을 뺀 보정
        std::vector<unsigned char> use(kPoints, 1);
        for (int i : ids) {
            std::vector<cv::Point2d> r(kPoints);
            for (int p = 0; p < kPoints; ++p) r[p] = c.dz(i, p) - mean[p];
            if (remove_affine) remove_qr_affine(c.points, r, use);
            std::vector<double> v(kPoints * 2);
            for (int p = 0; p < kPoints; ++p) { v[p * 2] = r[p].x * scale; v[p * 2 + 1] = r[p].y * scale; }
            prow.push_back(std::move(v));
            std::vector<double> l(kLines);
            double lm = 0;
            if (remove_affine) {  // 선 전체의 공통 이동은 실제 자세 차이로 보고 뺀다 (점의 아핀 제거와 같은 취지)
                for (int b = 0; b < kLines; ++b) lm += c.line(i)[b * 2] - lmean[b];
                lm /= kLines;
            }
            for (int b = 0; b < kLines; ++b) l[b] = (c.line(i)[b * 2] - lmean[b] - lm) * scale;
            lrow.push_back(std::move(l));
            ++out.shots;
        }
    }
    out.point_rows = cv::Mat(int(prow.size()), kPoints * 2, CV_64F);
    for (int i = 0; i < int(prow.size()); ++i)
        for (int j = 0; j < kPoints * 2; ++j) out.point_rows.at<double>(i, j) = prow[i][j];
    out.line_rows = cv::Mat(int(lrow.size()), kLines, CV_64F);
    for (int i = 0; i < int(lrow.size()); ++i)
        for (int j = 0; j < kLines; ++j) out.line_rows.at<double>(i, j) = lrow[i][j];
    return out;
}

// 표본 잔차 R(N x D) 에서 Sigma^-1 을 쓰는 연산자를 만든다.
//   Sigma = U diag(lam) U^T + floor * (I - U U^T)
// U 는 R^T R 의 주고유벡터 (그람 행렬로 구한다). floor 는 설명되지 않은 방향의 분산 바닥값이다.
struct InvSigma {
    cv::Mat U;              // D x k
    std::vector<double> lam;  // k
    double floor_var = 0;
    int D = 0, k = 0;

    // Sigma^-1 * X  (X: D x m)
    cv::Mat apply(const cv::Mat& X) const {
        cv::Mat UtX = U.t() * X;                       // k x m
        cv::Mat scaled = UtX.clone();
        for (int i = 0; i < k; ++i)
            for (int j = 0; j < scaled.cols; ++j)
                scaled.at<double>(i, j) *= (1.0 / lam[i] - 1.0 / floor_var);
        return X / floor_var + U * scaled;
    }
};

// keep: 쓸 차원의 번호. shrink: 고유값에 더하는 바닥 비율 (0.05 = 바닥값의 5%)
inline InvSigma build_inv_sigma(const cv::Mat& rows, const std::vector<int>& keep, double floor_quantile = 0.5) {
    const int N = rows.rows, D = int(keep.size());
    cv::Mat R(N, D, CV_64F);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < D; ++j) R.at<double>(i, j) = rows.at<double>(i, keep[j]);
    cv::Mat G = R * R.t() / std::max(1, N - 1);  // N x N 그람
    cv::Mat eval, evec;
    cv::eigen(G, eval, evec);                     // 내림차순
    // 설명되지 않은 방향의 바닥값: 차원별 분산의 분위수
    std::vector<double> dv(D, 0);
    for (int j = 0; j < D; ++j) {
        double s = 0;
        for (int i = 0; i < N; ++i) s += R.at<double>(i, j) * R.at<double>(i, j);
        dv[j] = s / std::max(1, N - 1);
    }
    InvSigma out;
    out.D = D;
    out.floor_var = std::max(1e-12, quantile(dv, floor_quantile));
    std::vector<int> use;
    for (int i = 0; i < eval.rows; ++i) if (eval.at<double>(i) > out.floor_var * 1.0) use.push_back(i);
    out.k = int(use.size());
    out.U = cv::Mat::zeros(D, std::max(1, out.k), CV_64F);
    out.lam.assign(std::max(1, out.k), out.floor_var);
    for (int c2 = 0; c2 < out.k; ++c2) {
        int i = use[c2];
        cv::Mat w = R.t() * evec.row(i).t();  // D x 1
        double n = cv::norm(w);
        if (n < 1e-15) { out.lam[c2] = out.floor_var; continue; }
        w /= n;
        w.copyTo(out.U.col(c2));
        out.lam[c2] = eval.at<double>(i);
    }
    if (out.k == 0) { out.k = 1; out.U = cv::Mat::zeros(D, 1, CV_64F); }
    return out;
}

}  // namespace a0
