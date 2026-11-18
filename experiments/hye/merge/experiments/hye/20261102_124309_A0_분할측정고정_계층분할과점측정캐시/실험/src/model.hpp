// A3·A6·A7·A8·A9 가 함께 쓰는 "기준 모델".
//
// 이것은 팀에 전달한 joint/control 모델이 아니다. 진단을 위해 같은 측정에서 다시 세운 대리 모델이며,
// fold 마다 다시 학습할 수 있어야 하므로 따로 만들었다. 여기서 나오는 성적을 전달 모델의 성적으로 쓰지 않는다.
//
// 특징 66차원
//   QR별 아핀 6개 x 3 = 18   : d = B[1, u, v] 의 계수 (판 전체 이동 + 1차 구조)
//   선 48블록 이동량     = 48
// 추정  (Y, T) = W f + c   (fold 안의 학습 사진에서만 W, c 를 정한다)
#pragma once
#include "cache.hpp"
#include "noise.hpp"
#include <opencv2/core.hpp>

namespace a0 {

constexpr int kFeatQr = 18, kFeatLine = kLines, kFeat = kFeatQr + kFeatLine;  // 18 + 48 = 66

// QR q 의 선택점에서 d = B[1,u,v] 를 맞춘다. 반환은 [bx0,bx1,bx2,by0,by1,by2].
inline std::array<double, 6> qr_affine(const Cache& c, int shot, int q, bool selected_only = true) {
    std::vector<int> idx;
    for (int p = 0; p < kPoints; ++p)
        if (c.points[p].region == q && c.v(shot)[p] && (!selected_only || c.points[p].selected)) idx.push_back(p);
    std::array<double, 6> out{};
    if (idx.size() < 6) return out;
    cv::Point2d ctr(kQrBox[q].x + kQrBox[q].width / 2.0, kQrBox[q].y + kQrBox[q].height / 2.0);
    cv::Mat X(int(idx.size()), 3, CV_64F), Y(int(idx.size()), 2, CV_64F), B;
    for (int k = 0; k < int(idx.size()); ++k) {
        X.at<double>(k, 0) = 1;
        X.at<double>(k, 1) = (c.points[idx[k]].x - ctr.x) / 100;
        X.at<double>(k, 2) = (c.points[idx[k]].y - ctr.y) / 100;
        auto d = c.dz(shot, idx[k]);
        Y.at<double>(k, 0) = d.x;
        Y.at<double>(k, 1) = d.y;
    }
    cv::solve(X, Y, B, cv::DECOMP_SVD);
    for (int j = 0; j < 3; ++j) { out[j] = B.at<double>(j, 0); out[3 + j] = B.at<double>(j, 1); }
    return out;
}

// 특징 66차원. use_lines=false 면 선 48칸을 0 으로 둔다 (관측 조건 비교용).
inline std::vector<double> features(const Cache& c, int shot, bool use_lines = true) {
    std::vector<double> f(kFeat, 0);
    for (int q = 0; q < 3; ++q) {
        auto a = qr_affine(c, shot, q);
        for (int j = 0; j < 6; ++j) f[q * 6 + j] = a[j];
    }
    if (use_lines) for (int b = 0; b < kLines; ++b) f[kFeatQr + b] = c.line(shot)[b * 2];
    return f;
}

// 선형 모델. ridge lambda 는 표준화한 특징에 건다. lambda=0 이면 보통 최소제곱이다.
struct LinearModel {
    cv::Mat W;                 // (d+1) x 2
    std::vector<double> mu, sd_;
    int d = 0;

    void fit(const std::vector<std::vector<double>>& X, const std::vector<cv::Point2d>& Y, double lambda) {
        d = int(X[0].size());
        const int n = int(X.size());
        mu.assign(d, 0);
        sd_.assign(d, 1);
        for (int j = 0; j < d; ++j) {
            double s = 0;
            for (int i = 0; i < n; ++i) s += X[i][j];
            mu[j] = s / n;
            double v = 0;
            for (int i = 0; i < n; ++i) v += (X[i][j] - mu[j]) * (X[i][j] - mu[j]);
            sd_[j] = std::sqrt(v / std::max(1, n - 1));
            if (sd_[j] < 1e-12) sd_[j] = 1;
        }
        cv::Mat A(n + (lambda > 0 ? d : 0), d + 1, CV_64F, 0.0);
        cv::Mat B(A.rows, 2, CV_64F, 0.0);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < d; ++j) A.at<double>(i, j) = (X[i][j] - mu[j]) / sd_[j];
            A.at<double>(i, d) = 1;
            B.at<double>(i, 0) = Y[i].x;
            B.at<double>(i, 1) = Y[i].y;
        }
        if (lambda > 0)
            for (int j = 0; j < d; ++j) A.at<double>(n + j, j) = std::sqrt(lambda);
        cv::solve(A, B, W, cv::DECOMP_SVD);
    }
    cv::Point2d predict(const std::vector<double>& x) const {
        double a = W.at<double>(d, 0), b = W.at<double>(d, 1);
        for (int j = 0; j < d; ++j) {
            double z = (x[j] - mu[j]) / sd_[j];
            a += W.at<double>(j, 0) * z;
            b += W.at<double>(j, 1) * z;
        }
        return {a, b};
    }
};

// fold 배정에 따른 out-of-fold 예측. fold_of(shot) < 0 이면 학습에만 쓴다 (center).
struct Oof {
    std::vector<int> shot;
    std::vector<cv::Point2d> truth, pred;
    std::vector<int> fold;
};

inline Oof run_oof(const Cache& c, const std::vector<std::vector<double>>& feat,
                   const std::function<int(const Shot&)>& fold_of, int n_fold, double lambda) {
    Oof out;
    for (int f = 0; f < n_fold; ++f) {
        std::vector<std::vector<double>> X;
        std::vector<cv::Point2d> Y;
        for (auto& s : c.shots) {
            if (!s.measured) continue;
            int g = fold_of(s);
            if (g == f) continue;           // 시험 fold 는 학습에서 뺀다
            X.push_back(feat[s.idx]);
            Y.push_back({s.y, s.t});
        }
        LinearModel m;
        m.fit(X, Y, lambda);
        for (auto& s : c.shots) {
            if (!s.measured || fold_of(s) != f) continue;
            out.shot.push_back(s.idx);
            out.truth.push_back({s.y, s.t});
            out.pred.push_back(m.predict(feat[s.idx]));
            out.fold.push_back(f);
        }
    }
    return out;
}

struct Score {
    double mean_y = 0, mean_t = 0, max_y = 0, max_t = 0, pass = 0, pass09 = 0;
    int n = 0, npass = 0;
};

inline Score score(const Oof& o, const std::vector<unsigned char>* keep = nullptr) {
    Score s;
    std::vector<double> ey, et;
    for (size_t i = 0; i < o.shot.size(); ++i) {
        if (keep && !(*keep)[i]) continue;
        double a = o.pred[i].x - o.truth[i].x, b = o.pred[i].y - o.truth[i].y;
        ey.push_back(a);
        et.push_back(b);
        ++s.n;
        if (std::abs(a) <= kTol && std::abs(b) <= kTol) ++s.npass;
        if (std::abs(a) <= 0.09 && std::abs(b) <= 0.09) s.pass09 += 1;
    }
    std::vector<double> ay, at;
    for (double v : ey) ay.push_back(std::abs(v));
    for (double v : et) at.push_back(std::abs(v));
    s.mean_y = mean(ay);
    s.mean_t = mean(at);
    s.max_y = max_abs(ey);
    s.max_t = max_abs(et);
    s.pass = s.n ? double(s.npass) / s.n : 0;
    s.pass09 = s.n ? s.pass09 / s.n : 0;
    return s;
}

// 오차 (eY, eT) 의 상관계수와 주축 각도(도). 주축은 공분산의 큰 고유벡터.
struct ErrorAxis {
    double r = 0, angle_to_null = 0;
    cv::Point2d axis{0, 0};
};
inline ErrorAxis error_axis(const Oof& o, const std::vector<unsigned char>* keep = nullptr) {
    std::vector<double> ey, et;
    for (size_t i = 0; i < o.shot.size(); ++i) {
        if (keep && !(*keep)[i]) continue;
        ey.push_back(o.pred[i].x - o.truth[i].x);
        et.push_back(o.pred[i].y - o.truth[i].y);
    }
    ErrorAxis out;
    if (ey.size() < 3) return out;
    double my = mean(ey), mt = mean(et), sy = 0, st = 0, sc = 0;
    for (size_t i = 0; i < ey.size(); ++i) {
        sy += (ey[i] - my) * (ey[i] - my);
        st += (et[i] - mt) * (et[i] - mt);
        sc += (ey[i] - my) * (et[i] - mt);
    }
    out.r = (sy > 0 && st > 0) ? sc / std::sqrt(sy * st) : 0;
    cv::Matx22d C(sy, sc, sc, st);
    cv::Mat ev, evec;
    cv::eigen(cv::Mat(C), ev, evec);
    out.axis = {evec.at<double>(0, 0), evec.at<double>(0, 1)};
    cv::Point2d nd(-kTtoY, 1.0);
    nd *= 1.0 / cv::norm(nd);
    double cosang = std::abs(out.axis.dot(nd)) / (cv::norm(out.axis) * cv::norm(nd));
    out.angle_to_null = std::acos(std::min(1.0, cosang)) * 180 / CV_PI;
    return out;
}

}  // namespace a0
