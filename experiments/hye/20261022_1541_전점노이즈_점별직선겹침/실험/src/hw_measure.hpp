// 묶음 EDA 측정 (hw_eda 의 measure.cpp, measurements.cpp 를 이 실험 안으로 옮긴 것)
//   원본: experiments/hye/20261019_1400_측정별판정비교_동일GLS판정기/종합/20261020_1752_C++이식_파일별이식검증/tool/hw/eda/src/measure.cpp
//   1. center 사진에서 밝은 QR 라벨 3개를 찾는다 (오츠 이진화)
//   2. ECC 패치: 라벨 + 바깥 8px. 템플릿 매칭(가로 ±320px) -> ECC 아핀(흐림 5px), 패치 가운데 점의 이동
//   3. 모서리: 라벨 안 코너 검출 -> 4칸 종류 판정 -> 흐린 2x2 모듈 밝기 모델(soft-L1 LM)로 서브픽셀
//      사진마다 ECC 로 예측한 자리에서 다시 맞추고, 대각선이 아닌 '모든 사진에서 추적된' 점의 평균 이동 = corner_dx
// 원래 결과(shots.csv)가 OpenCV 3.4.15 C++ 프로그램으로 나왔으므로 ECC 도 그때와 같은 cv::findTransformECC 를 쓴다.
#pragma once
#include <hye/report.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <array>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace hwm {

constexpr int kQr = 3;
constexpr int kPatchMargin = 8;

struct QrRef {
    cv::Rect label, patch;
};

struct EccMotion {
    double dx = 0, dy = 0, cc = 0;
    cv::Matx23d warp;
    bool ok = false;
};

struct Corner {
    cv::Point2f p;
    int qr = 0;
    bool diagonal = false;
    std::array<int, 4> black{};
};

struct CornerFit {
    cv::Point2d p;
    double rmse = 0, contrast = 0, blur = 0;
    bool ok = false;
    std::string reason;
};

// 한글 경로도 읽히게 바이트로 읽어서 디코드. 3채널이면 0번 채널 (원래 readGray 와 같음)
inline cv::Mat read_gray0(const std::string& path) {
    std::ifstream in(hye::fs::u8path(path), std::ios::binary);
    std::vector<uchar> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    cv::Mat img = cv::imdecode(bytes, cv::IMREAD_UNCHANGED);
    if (img.empty()) throw std::runtime_error("읽기 실패: " + path);
    if (img.channels() == 1) return img;
    cv::Mat ch;
    cv::extractChannel(img, ch, 0);
    return ch;
}

inline std::array<QrRef, kQr> find_labels(const cv::Mat& gray) {
    const cv::Rect band(gray.cols / 4, gray.rows / 4, gray.cols / 2, gray.rows / 2);
    cv::Mat roi, bin;
    cv::GaussianBlur(gray(band), roi, cv::Size(0, 0), 1.0);
    cv::threshold(roi, bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::Mat labels, stats, centroids;
    const int n = cv::connectedComponentsWithStats(bin, labels, stats, centroids, 8, CV_32S);
    std::vector<cv::Rect> cand;
    for (int i = 1; i < n; ++i) {
        const int w = stats.at<int>(i, cv::CC_STAT_WIDTH), h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        const int area = stats.at<int>(i, cv::CC_STAT_AREA);
        const double aspect = (double)w / h;
        if (w < 100 || h < 100 || w > 400 || h > 400) continue;
        if (aspect < 0.75 || aspect > 1.33) continue;
        if (area < 0.5 * w * h) continue;
        cand.emplace_back(stats.at<int>(i, cv::CC_STAT_LEFT) + band.x, stats.at<int>(i, cv::CC_STAT_TOP) + band.y, w, h);
    }
    std::sort(cand.begin(), cand.end(), [](const cv::Rect& a, const cv::Rect& b) { return a.x < b.x; });
    std::array<QrRef, kQr> qrs{};
    double best = 1e9;
    for (size_t a = 0; a < cand.size(); ++a)
        for (size_t b = a + 1; b < cand.size(); ++b)
            for (size_t c = b + 1; c < cand.size(); ++c) {
                const double ya = cand[a].y + cand[a].height / 2.0, yb = cand[b].y + cand[b].height / 2.0, yc = cand[c].y + cand[c].height / 2.0;
                const double spread = std::max({ya, yb, yc}) - std::min({ya, yb, yc});
                const double ss = std::max({cand[a].area(), cand[b].area(), cand[c].area()}) /
                                  (double)std::min({cand[a].area(), cand[b].area(), cand[c].area()});
                const double score = spread + 100 * (ss - 1);
                if (score < best) { best = score; qrs[0].label = cand[a]; qrs[1].label = cand[b]; qrs[2].label = cand[c]; }
            }
    if (best > 60) throw std::runtime_error("QR 라벨 3개를 찾지 못함");
    for (auto& q : qrs)
        q.patch = cv::Rect(q.label.x - kPatchMargin, q.label.y - kPatchMargin, q.label.width + 2 * kPatchMargin, q.label.height + 2 * kPatchMargin);
    return qrs;
}

struct EccMeasurer {
    std::array<cv::Mat, kQr> tpl;
    std::array<QrRef, kQr> qrs;
    EccMeasurer(const cv::Mat& center, const std::array<QrRef, kQr>& q) : qrs(q) {
        for (int i = 0; i < kQr; ++i) center(q[i].patch).convertTo(tpl[i], CV_32F);
    }
    EccMotion measure(const cv::Mat& gray, int q) const {
        EccMotion m;
        const cv::Rect patch = qrs[q].patch;
        const int padX = 320, padY = 60;
        cv::Rect search(patch.x - padX, patch.y - padY, patch.width + 2 * padX, patch.height + 2 * padY);
        search &= cv::Rect(0, 0, gray.cols, gray.rows);
        cv::Mat sf, score;
        gray(search).convertTo(sf, CV_32F);
        cv::matchTemplate(sf, tpl[q], score, cv::TM_CCOEFF_NORMED);
        cv::Point loc;
        cv::minMaxLoc(score, nullptr, nullptr, nullptr, &loc);
        const int ox = search.x + loc.x - patch.x, oy = search.y + loc.y - patch.y;
        cv::Rect target(patch.x + ox, patch.y + oy, patch.width, patch.height);
        if ((target & cv::Rect(0, 0, gray.cols, gray.rows)) != target) return m;
        cv::Mat tf;
        gray(target).convertTo(tf, CV_32F);
        cv::Mat W = cv::Mat::eye(2, 3, CV_32F);
        try {
            m.cc = cv::findTransformECC(tpl[q], tf, W, cv::MOTION_AFFINE, cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 200, 1e-8),
                                        cv::noArray(), 5);
        } catch (const cv::Exception&) {
            return m;
        }
        const double cu = (patch.width - 1) / 2.0, cv_ = (patch.height - 1) / 2.0;
        const double mu = W.at<float>(0, 0) * cu + W.at<float>(0, 1) * cv_ + W.at<float>(0, 2);
        const double mv = W.at<float>(1, 0) * cu + W.at<float>(1, 1) * cv_ + W.at<float>(1, 2);
        m.dx = ox + mu - cu;
        m.dy = oy + mv - cv_;
        m.warp = cv::Matx23d(W.at<float>(0, 0), W.at<float>(0, 1), W.at<float>(0, 2) + patch.x + ox, W.at<float>(1, 0), W.at<float>(1, 1),
                             W.at<float>(1, 2) + patch.y + oy);
        m.ok = true;
        return m;
    }
};

inline int corner_kind(const cv::Mat& gray, cv::Point2f p, double thr, std::array<int, 4>& black) {
    const int d = 4, r = 2;
    auto dark = [&](int sx, int sy) {
        const cv::Rect box(cvRound(p.x) + sx * d - r, cvRound(p.y) + sy * d - r, 2 * r + 1, 2 * r + 1);
        if ((box & cv::Rect(0, 0, gray.cols, gray.rows)) != box) return -1;
        return cv::mean(gray(box))[0] < thr ? 1 : 0;
    };
    black = {dark(-1, -1), dark(1, -1), dark(-1, 1), dark(1, 1)};
    if (black[0] < 0 || black[1] < 0 || black[2] < 0 || black[3] < 0) return 0;
    const int count = black[0] + black[1] + black[2] + black[3];
    if (count == 1 || count == 3) return 1;
    if (count == 2 && black[0] == black[3]) return 2;
    return 0;
}

struct FitResult { double p[6]; double rmse; bool converged; };

// 흐린 2x2 모듈 밝기 모델: I = low + contrast (1 - B(Φ((X-x)/σx), Φ((Y-y)/σy)))  soft-L1(f=6) LM, 범위 제한
inline FitResult fit_window(const cv::Mat& gray, const Corner& c, cv::Point2d center, cv::Point2d start, int radius) {
    const int x0 = cvRound(center.x), y0 = cvRound(center.y);
    std::vector<double> X, Y, I;
    for (int y = y0 - radius; y <= y0 + radius; ++y)
        for (int x = x0 - radius; x <= x0 + radius; ++x)
            if (x >= 0 && y >= 0 && x < gray.cols && y < gray.rows) { X.push_back(x); Y.push_back(y); I.push_back(gray.at<uchar>(y, x)); }
    const int n = (int)I.size();
    std::vector<double> sorted = I;
    std::sort(sorted.begin(), sorted.end());
    const double low0 = sorted[n / 10], high0 = sorted[n * 9 / 10];
    double p[6] = {start.x, start.y, 1.0, 1.0, low0, std::max(high0 - low0, 20.0)};
    const double lo[6] = {start.x - 3.8, start.y - 3.8, 0.35, 0.35, 0, 10};
    const double hi[6] = {start.x + 3.8, start.y + 3.8, 2.7, 2.7, 240, 255};
    const double tl = c.black[0], tr = c.black[1], bl = c.black[2], br = c.black[3];
    const double fS = 6.0, kInvSqrt2 = 0.70710678118654752, kInvSqrt2Pi = 0.39894228040143268;
    auto residuals = [&](const double* q, std::vector<double>& r, cv::Mat* J) {
        for (int k = 0; k < n; ++k) {
            const double a = X[k] - q[0], b = Y[k] - q[1];
            const double U = 0.5 * (1 + std::erf(a / q[2] * kInvSqrt2)), V = 0.5 * (1 + std::erf(b / q[3] * kInvSqrt2));
            const double B = tl * (1 - U) * (1 - V) + tr * U * (1 - V) + bl * (1 - U) * V + br * U * V;
            r[k] = q[4] + q[5] * (1 - B) - I[k];
            if (!J) continue;
            const double gU = kInvSqrt2Pi / q[2] * std::exp(-0.5 * a * a / (q[2] * q[2]));
            const double gV = kInvSqrt2Pi / q[3] * std::exp(-0.5 * b * b / (q[3] * q[3]));
            const double dBdU = (tr - tl) * (1 - V) + (br - bl) * V, dBdV = (bl - tl) * (1 - U) + (br - tr) * U;
            double* row = J->ptr<double>(k);
            row[0] = q[5] * dBdU * gU;
            row[1] = q[5] * dBdV * gV;
            row[2] = q[5] * dBdU * gU * a / q[2];
            row[3] = q[5] * dBdV * gV * b / q[3];
            row[4] = 1;
            row[5] = 1 - B;
        }
    };
    auto cost = [&](const std::vector<double>& res) {
        double s = 0;
        for (double v : res) s += 2 * fS * fS * (std::sqrt(1 + (v / fS) * (v / fS)) - 1);
        return s;
    };
    std::vector<double> r(n), rNew(n);
    cv::Mat J(n, 6, CV_64F);
    residuals(p, r, &J);
    double lambda = 1e-3, cur = cost(r);
    bool converged = false;
    double lastChange = 1e9;
    for (int it = 0; it < 80 && !converged; ++it) {
        cv::Matx66d A = cv::Matx66d::zeros();
        cv::Vec6d g = cv::Vec6d::all(0);
        for (int k = 0; k < n; ++k) {
            const double w = 1.0 / std::sqrt(1 + (r[k] / fS) * (r[k] / fS));
            const double* row = J.ptr<double>(k);
            for (int a = 0; a < 6; ++a) {
                g[a] += w * row[a] * r[k];
                for (int b = 0; b < 6; ++b) A(a, b) += w * row[a] * row[b];
            }
        }
        for (int a = 0; a < 6; ++a) {
            const bool atLo = p[a] <= lo[a] + 1e-12 && g[a] > 0, atHi = p[a] >= hi[a] - 1e-12 && g[a] < 0;
            if (!atLo && !atHi) continue;
            for (int b = 0; b < 6; ++b) { A(a, b) = 0; A(b, a) = 0; }
            A(a, a) = 1;
            g[a] = 0;
        }
        bool improved = false;
        for (int tries = 0; tries < 8 && !improved; ++tries) {
            cv::Matx66d Ad = A;
            for (int a = 0; a < 6; ++a) Ad(a, a) *= (1 + lambda);
            cv::Vec6d step;
            cv::solve(Ad, -g, step, cv::DECOMP_SVD);
            double q[6], change2 = 0, norm2 = 0;
            for (int a = 0; a < 6; ++a) {
                q[a] = std::min(hi[a], std::max(lo[a], p[a] + step[a]));
                change2 += (q[a] - p[a]) * (q[a] - p[a]);
                norm2 += p[a] * p[a];
            }
            const double change = std::sqrt(change2), pnorm = std::sqrt(norm2);
            residuals(q, rNew, nullptr);
            const double nc = cost(rNew);
            if (nc <= cur) {
                improved = true;
                std::copy(q, q + 6, p);
                const double rel = (cur - nc) / std::max(cur, 1e-12);
                cur = nc;
                lambda = std::max(lambda * 0.3, 1e-7);
                residuals(p, r, &J);
                lastChange = change;
                if (change < 1e-7 * (1e-7 + pnorm) || rel < 1e-7) converged = true;
            } else {
                lambda *= 10;
            }
        }
        if (!improved) converged = true;
    }
    if (!converged && lastChange < 1e-3) converged = true;
    FitResult fr;
    std::copy(p, p + 6, fr.p);
    double ss = 0;
    for (double v : r) ss += v * v;
    fr.rmse = std::sqrt(ss / n);
    fr.converged = converged;
    return fr;
}

inline CornerFit fit_corner(const cv::Mat& gray, const Corner& c, cv::Point2d start) {
    cv::Point2d est = start;
    FitResult f1{};
    for (int k = 0; k < 3; ++k) {
        f1 = fit_window(gray, c, est, start, 4);
        const cv::Point2d got(f1.p[0], f1.p[1]);
        const bool moved = cvRound(got.x) != cvRound(est.x) || cvRound(got.y) != cvRound(est.y);
        est = got;
        if (!moved) break;
    }
    const FitResult f2 = fit_window(gray, c, est, start, 5);
    CornerFit out;
    out.p = cv::Point2d(f1.p[0], f1.p[1]);
    out.rmse = f1.rmse;
    out.contrast = f1.p[5];
    out.blur = std::max(f1.p[2], f1.p[3]);
    const double stability = cv::norm(out.p - cv::Point2d(f2.p[0], f2.p[1]));
    if (!f1.converged || !f2.converged) out.reason = "optimizer";
    else if (out.rmse > (c.diagonal ? 35 : 15)) out.reason = "rmse";
    else if (out.contrast < 45) out.reason = "contrast";
    else if (stability > 0.65) out.reason = "window";
    else if (std::fabs(out.p.x - start.x) > 3.6 || std::fabs(out.p.y - start.y) > 3.6) out.reason = "search";
    else if (out.blur > 2.6) out.reason = "blur";
    out.ok = out.reason.empty();
    return out;
}

inline std::vector<Corner> find_corners(const cv::Mat& gray, const std::array<QrRef, kQr>& qrs) {
    std::vector<Corner> out;
    for (int q = 0; q < kQr; ++q) {
        const cv::Rect inner(qrs[q].label.x + 5, qrs[q].label.y + 5, qrs[q].label.width - 10, qrs[q].label.height - 10);
        cv::Mat bin;
        const double thr = cv::threshold(gray(inner), bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        std::vector<cv::Point2f> pts;
        cv::goodFeaturesToTrack(gray(inner), pts, 1500, 0.01, 4, cv::noArray(), 5);
        for (const auto& p0 : pts) {
            Corner c;
            c.p = p0 + cv::Point2f((float)inner.x, (float)inner.y);
            c.qr = q;
            const int kind = corner_kind(gray, c.p, thr, c.black);
            if (kind == 0) continue;
            c.diagonal = kind == 2;
            const CornerFit f = fit_corner(gray, c, cv::Point2d(c.p));
            if (!f.ok) continue;
            c.p = cv::Point2f((float)f.p.x, (float)f.p.y);
            const bool dup = std::any_of(out.begin(), out.end(), [&](const Corner& o) { return o.qr == q && cv::norm(o.p - c.p) < 2.0; });
            if (!dup) out.push_back(c);
        }
    }
    return out;
}

inline std::vector<cv::Point2f> track_corners(const cv::Mat& gray, const std::vector<Corner>& corners, const std::array<QrRef, kQr>& qrs,
                                             const std::array<EccMotion, kQr>& ecc) {
    const float nanf = (float)hye::kNaN;
    std::vector<cv::Point2f> out(corners.size(), cv::Point2f(nanf, nanf));
    for (size_t i = 0; i < corners.size(); ++i) {
        const int q = corners[i].qr;
        if (!ecc[q].ok) continue;
        const cv::Point2d local(corners[i].p.x - qrs[q].patch.x, corners[i].p.y - qrs[q].patch.y);
        const cv::Matx23d& W = ecc[q].warp;
        const cv::Point2d pred(W(0, 0) * local.x + W(0, 1) * local.y + W(0, 2), W(1, 0) * local.x + W(1, 1) * local.y + W(1, 2));
        const CornerFit f = fit_corner(gray, corners[i], pred);
        if (f.ok) out[i] = cv::Point2f((float)f.p.x, (float)f.p.y);
    }
    return out;
}

struct ShotMeasure {
    std::array<EccMotion, kQr> ecc;
    double ecc_dx = hye::kNaN, ecc_dy = hye::kNaN, corner_dx = hye::kNaN, corner_dy = hye::kNaN;
    std::vector<cv::Point2f> pos;
};

struct SetMeasure {
    std::array<QrRef, kQr> qrs;
    std::vector<Corner> corners;
    std::vector<bool> usable;
    std::vector<ShotMeasure> shots;
};

// paths[ref] 가 기준 사진. 사진은 한 장씩 읽고 버린다 (병렬 4).
inline SetMeasure measure_set(const std::vector<std::string>& paths, size_t ref) {
    SetMeasure m;
    const cv::Mat refGray = read_gray0(paths[ref]);
    m.qrs = find_labels(refGray);
    m.corners = find_corners(refGray, m.qrs);
    const EccMeasurer ecc(refGray, m.qrs);
    m.shots.resize(paths.size());
#pragma omp parallel for schedule(dynamic) num_threads(4)
    for (int i = 0; i < (int)paths.size(); ++i) {
        const cv::Mat gray = (size_t)i == ref ? refGray : read_gray0(paths[i]);
        ShotMeasure& s = m.shots[i];
        double sx = 0, sy = 0;
        for (int q = 0; q < kQr; ++q) {
            s.ecc[q] = ecc.measure(gray, q);
            sx += s.ecc[q].dx;
            sy += s.ecc[q].dy;
        }
        s.ecc_dx = sx / kQr;
        s.ecc_dy = sy / kQr;
        s.pos = track_corners(gray, m.corners, m.qrs, s.ecc);
    }
    m.usable.assign(m.corners.size(), true);
    for (const auto& s : m.shots)
        for (size_t k = 0; k < m.corners.size(); ++k)
            if (!std::isfinite(s.pos[k].x)) m.usable[k] = false;
    for (auto& s : m.shots) {
        double sx = 0, sy = 0;
        int n = 0;
        for (size_t k = 0; k < m.corners.size(); ++k) {
            if (!m.usable[k] || m.corners[k].diagonal) continue;
            sx += s.pos[k].x - m.corners[k].p.x;
            sy += s.pos[k].y - m.corners[k].p.y;
            ++n;
        }
        if (n) { s.corner_dx = sx / n; s.corner_dy = sy / n; }
    }
    return m;
}

}  // namespace hwm
