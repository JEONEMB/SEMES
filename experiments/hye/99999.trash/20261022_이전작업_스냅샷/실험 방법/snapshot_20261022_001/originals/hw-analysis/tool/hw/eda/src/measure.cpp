#include "measure.hpp"

#include "core.hpp"

namespace hw {

// -------------------------------------------------------------------------------------
// 1. QR 라벨 찾기
// -------------------------------------------------------------------------------------

std::array<QrRef, kQrCount> findQrLabels(const cv::Mat &gray) {
    // 사진 가운데 띠에서만 찾는다 (QR 판이 화면 가운데에 있다)
    const cv::Rect band(gray.cols / 4, gray.rows / 4, gray.cols / 2, gray.rows / 2);
    cv::Mat roi, bin;
    cv::GaussianBlur(gray(band), roi, cv::Size(0, 0), 1.0);
    cv::threshold(roi, bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);   // 밝은 라벨 = 255

    cv::Mat labels, stats, centroids;
    const int n = cv::connectedComponentsWithStats(bin, labels, stats, centroids, 8, CV_32S);

    // 라벨 후보: 정사각형에 가깝고 크기가 비슷한 밝은 덩어리
    std::vector<cv::Rect> cand;
    for (int i = 1; i < n; ++i) {
        const int w = stats.at<int>(i, cv::CC_STAT_WIDTH), h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        const int area = stats.at<int>(i, cv::CC_STAT_AREA);
        const double aspect = static_cast<double>(w) / h;
        if (w < 100 || h < 100 || w > 400 || h > 400) continue;
        if (aspect < 0.75 || aspect > 1.33) continue;
        if (area < 0.5 * w * h) continue;   // 속이 찬 사각형 (테두리만 있는 건 제외)
        cand.emplace_back(stats.at<int>(i, cv::CC_STAT_LEFT) + band.x, stats.at<int>(i, cv::CC_STAT_TOP) + band.y, w, h);
    }
    // 같은 높이에 나란히 있는 3개를 고른다: 세로 중심이 가장 가까운 3개 묶음
    std::sort(cand.begin(), cand.end(), [](const cv::Rect &a, const cv::Rect &b) { return a.x < b.x; });
    std::array<QrRef, kQrCount> qrs{};
    double best = 1e9;
    for (size_t a = 0; a < cand.size(); ++a)
        for (size_t b = a + 1; b < cand.size(); ++b)
            for (size_t c = b + 1; c < cand.size(); ++c) {
                const double ya = cand[a].y + cand[a].height / 2.0, yb = cand[b].y + cand[b].height / 2.0, yc = cand[c].y + cand[c].height / 2.0;
                const double spread = std::max({ya, yb, yc}) - std::min({ya, yb, yc});
                const double sizeSpread = std::max({cand[a].area(), cand[b].area(), cand[c].area()}) /
                                          static_cast<double>(std::min({cand[a].area(), cand[b].area(), cand[c].area()}));
                const double score = spread + 100 * (sizeSpread - 1);
                if (score < best) {
                    best = score;
                    qrs[0].label = cand[a]; qrs[1].label = cand[b]; qrs[2].label = cand[c];
                }
            }
    if (best > 60) fail("center 사진에서 나란한 QR 라벨 3개를 찾지 못했습니다 (후보 " + std::to_string(cand.size()) + "개)");

    for (auto &q : qrs) {
        q.patch = cv::Rect(q.label.x - kPatchMargin, q.label.y - kPatchMargin, q.label.width + 2 * kPatchMargin,
                           q.label.height + 2 * kPatchMargin);
        q.labelCenter = cv::Point2d(q.label.x + q.label.width / 2.0, q.label.y + q.label.height / 2.0);
        q.labelMean = cv::mean(gray(q.label))[0];
    }
    return qrs;
}

// -------------------------------------------------------------------------------------
// 2. ECC 패치
// -------------------------------------------------------------------------------------

EccMeasurer::EccMeasurer(const cv::Mat &centerGray, const std::array<QrRef, kQrCount> &qrs) : qrs_(qrs) {
    for (int q = 0; q < kQrCount; ++q) centerGray(qrs[q].patch).convertTo(templates_[q], CV_32F);
}

EccMotion EccMeasurer::measure(const cv::Mat &gray, int q) const {
    EccMotion m;
    const cv::Rect patch = qrs_[q].patch;
    // 대략 위치: 가로로 크게(±320px = Y 1mm + Θ 2 정도), 세로로 작게 찾는다
    const int padX = 320, padY = 60;
    cv::Rect search(patch.x - padX, patch.y - padY, patch.width + 2 * padX, patch.height + 2 * padY);
    search &= cv::Rect(0, 0, gray.cols, gray.rows);
    cv::Mat searchF, score;
    gray(search).convertTo(searchF, CV_32F);
    cv::matchTemplate(searchF, templates_[q], score, cv::TM_CCOEFF_NORMED);
    cv::Point loc;
    cv::minMaxLoc(score, nullptr, nullptr, nullptr, &loc);
    const int ox = search.x + loc.x - patch.x, oy = search.y + loc.y - patch.y;   // 정수 오프셋

    // 그 자리의 같은 크기 패치를 center 패치에 ECC 아핀으로 맞춘다
    cv::Rect target(patch.x + ox, patch.y + oy, patch.width, patch.height);
    if ((target & cv::Rect(0, 0, gray.cols, gray.rows)) != target) return m;
    cv::Mat targetF;
    gray(target).convertTo(targetF, CV_32F);
    cv::Mat W = cv::Mat::eye(2, 3, CV_32F);
    try {
        m.cc = cv::findTransformECC(templates_[q], targetF, W, cv::MOTION_AFFINE,
                                    cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 200, 1e-8), cv::noArray(), 5);
    } catch (const cv::Exception &) {
        return m;
    }
    // 패치 좌표 (u, v) -> 사진 좌표: W * (u, v, 1) + 패치 원점 + 정수 오프셋
    const double cu = (patch.width - 1) / 2.0, cv_ = (patch.height - 1) / 2.0;
    const double mu = W.at<float>(0, 0) * cu + W.at<float>(0, 1) * cv_ + W.at<float>(0, 2);
    const double mv = W.at<float>(1, 0) * cu + W.at<float>(1, 1) * cv_ + W.at<float>(1, 2);
    m.dx = ox + mu - cu;
    m.dy = oy + mv - cv_;
    m.rotDeg = std::atan2(W.at<float>(1, 0), W.at<float>(0, 0)) * 180.0 / CV_PI;
    m.scale = std::sqrt(std::fabs(W.at<float>(0, 0) * W.at<float>(1, 1) - W.at<float>(0, 1) * W.at<float>(1, 0)));
    m.warp = cv::Matx23d(W.at<float>(0, 0), W.at<float>(0, 1), W.at<float>(0, 2) + patch.x + ox,
                         W.at<float>(1, 0), W.at<float>(1, 1), W.at<float>(1, 2) + patch.y + oy);
    m.ok = true;
    return m;
}

// -------------------------------------------------------------------------------------
// 3. 모서리 특징점
// -------------------------------------------------------------------------------------

// 모서리 주변 4칸(왼위, 오른위, 왼아래, 오른아래)이 검정인지 본다.
//   검정 1칸 또는 3칸 = 보통 모서리 (L 자)       -> 1
//   검정 2칸이 대각선 = 대각선 모서리             -> 2
//   그 외(검정 0, 4칸, 나란한 2칸 = 변) = 모서리 아님 -> 0
static int cornerKind(const cv::Mat &gray, cv::Point2f p, double thr, std::array<int, 4> &black) {
    const int d = 4, r = 2;   // 꼭짓점에서 4px 떨어진 곳의 5x5 평균 (QR 모듈 한 칸 약 6.5px)
    auto dark = [&](int sx, int sy) {
        const cv::Rect box(cvRound(p.x) + sx * d - r, cvRound(p.y) + sy * d - r, 2 * r + 1, 2 * r + 1);
        if ((box & cv::Rect(0, 0, gray.cols, gray.rows)) != box) return -1;
        return cv::mean(gray(box))[0] < thr ? 1 : 0;
    };
    black = {dark(-1, -1), dark(1, -1), dark(-1, 1), dark(1, 1)};
    if (black[0] < 0 || black[1] < 0 || black[2] < 0 || black[3] < 0) return 0;
    const int count = black[0] + black[1] + black[2] + black[3];
    if (count == 1 || count == 3) return 1;
    if (count == 2 && black[0] == black[3]) return 2;   // 대각선
    return 0;
}

// -------------------------------------------------------------------------------------
// 밝기 모델로 꼭짓점 맞추기
//   꼭짓점 (x, y) 를 기준으로 4칸이 검정/흰색인 이상적인 무늬를, 가로·세로 경계를 가우스 σx, σy 로 흐린 것.
//     U = Φ((X - x) / σx),  V = Φ((Y - y) / σy)          (Φ = 표준정규 누적분포)
//     검정 비율 B = TL(1-U)(1-V) + TR U(1-V) + BL (1-U)V + BR UV
//     밝기 I = low + contrast × (1 - B)
//   매개변수 6개 (x, y, σx, σy, low, contrast) 를 창 안 픽셀 밝기에 최소제곱으로 맞춘다 (soft-L1, 범위 제한).
//   이전 Python 분석(analysis_grid_corners_20261019/track_corners.py)과 같은 모델, 같은 통과 기준.
// -------------------------------------------------------------------------------------

namespace {

struct FitResult { double p[6]; double rmse; bool converged; };

FitResult fitWindow(const cv::Mat &gray, const Corner &c, cv::Point2d center, cv::Point2d start, int radius) {
    // 창 안 픽셀
    const int x0 = cvRound(center.x), y0 = cvRound(center.y);
    std::vector<double> X, Y, I;
    for (int y = y0 - radius; y <= y0 + radius; ++y)
        for (int x = x0 - radius; x <= x0 + radius; ++x)
            if (x >= 0 && y >= 0 && x < gray.cols && y < gray.rows) { X.push_back(x); Y.push_back(y); I.push_back(gray.at<uchar>(y, x)); }
    const int n = static_cast<int>(I.size());
    std::vector<double> sorted = I;
    std::sort(sorted.begin(), sorted.end());
    const double low0 = sorted[n / 10], high0 = sorted[n * 9 / 10];

    double p[6] = {start.x, start.y, 1.0, 1.0, low0, std::max(high0 - low0, 20.0)};
    const double lo[6] = {start.x - 3.8, start.y - 3.8, 0.35, 0.35, 0, 10};
    const double hi[6] = {start.x + 3.8, start.y + 3.8, 2.7, 2.7, 240, 255};
    const double tl = c.black[0], tr = c.black[1], bl = c.black[2], br = c.black[3];
    const double fScale = 6.0, kInvSqrt2 = 0.70710678118654752, kInvSqrt2Pi = 0.39894228040143268;

    // 잔차와 (필요하면) 야코비안
    auto residuals = [&](const double *q, std::vector<double> &r, cv::Mat *J) {
        for (int k = 0; k < n; ++k) {
            const double a = X[k] - q[0], b = Y[k] - q[1];
            const double U = 0.5 * (1 + std::erf(a / q[2] * kInvSqrt2)), V = 0.5 * (1 + std::erf(b / q[3] * kInvSqrt2));
            const double B = tl * (1 - U) * (1 - V) + tr * U * (1 - V) + bl * (1 - U) * V + br * U * V;
            r[k] = q[4] + q[5] * (1 - B) - I[k];
            if (!J) continue;
            const double gU = kInvSqrt2Pi / q[2] * std::exp(-0.5 * a * a / (q[2] * q[2]));   // dU/da
            const double gV = kInvSqrt2Pi / q[3] * std::exp(-0.5 * b * b / (q[3] * q[3]));
            const double dBdU = (tr - tl) * (1 - V) + (br - bl) * V, dBdV = (bl - tl) * (1 - U) + (br - tr) * U;
            double *row = J->ptr<double>(k);
            row[0] = q[5] * dBdU * gU;             // d/dx  (a = X - x 라서 부호가 바뀜)
            row[1] = q[5] * dBdV * gV;             // d/dy
            row[2] = q[5] * dBdU * gU * a / q[2];  // d/dσx
            row[3] = q[5] * dBdV * gV * b / q[3];  // d/dσy
            row[4] = 1;                            // d/dlow
            row[5] = 1 - B;                        // d/dcontrast
        }
    };
    // soft-L1 비용 (큰 잔차의 영향을 줄임)
    auto cost = [&](const std::vector<double> &res) {
        double s = 0;
        for (double v : res) s += 2 * fScale * fScale * (std::sqrt(1 + (v / fScale) * (v / fScale)) - 1);
        return s;
    };

    std::vector<double> r(n), rNew(n);
    cv::Mat J(n, 6, CV_64F);
    residuals(p, r, &J);
    double lambda = 1e-3, cur = cost(r);
    bool converged = false;
    double lastChange = 1e9;
    for (int it = 0; it < 80 && !converged; ++it) {
        // 가중 정규방정식 (soft-L1 가중치 w = 1 / sqrt(1 + (r/f)^2)), 레벤버그-마쿼트
        cv::Matx66d A = cv::Matx66d::zeros();
        cv::Vec6d g = cv::Vec6d::all(0);
        for (int k = 0; k < n; ++k) {
            const double w = 1.0 / std::sqrt(1 + (r[k] / fScale) * (r[k] / fScale));
            const double *row = J.ptr<double>(k);
            for (int a = 0; a < 6; ++a) {
                g[a] += w * row[a] * r[k];
                for (int b = 0; b < 6; ++b) A(a, b) += w * row[a] * row[b];
            }
        }
        // 범위 끝에 붙어 있고 더 바깥으로 가려는 매개변수는 이번 단계에서 고정 (활성 집합)
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
                // scipy least_squares 와 같은 종료 기준: 이동 폭 < 1e-7 × (1e-7 + |p|), 비용 변화 < 1e-7 × 비용
                if (change < 1e-7 * (1e-7 + pnorm) || rel < 1e-7) converged = true;
            } else {
                lambda *= 10;
            }
        }
        if (!improved) converged = true;   // 더 줄일 수 없음 = 최소점
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

}  // namespace

CornerFit fitCorner(const cv::Mat &gray, const Corner &c, cv::Point2d start) {
    // 9x9 창으로 맞추고, 맞춘 위치의 가장 가까운 픽셀로 창을 옮겨 다시 맞춘다 (최대 3번).
    // 창이 시작점에 따라 1px 밀리면 결과가 달라지는 문제를 없앤다.
    cv::Point2d est = start;
    FitResult f1{};
    for (int k = 0; k < 3; ++k) {
        f1 = fitWindow(gray, c, est, start, 4);
        const cv::Point2d got(f1.p[0], f1.p[1]);
        const bool moved = cvRound(got.x) != cvRound(est.x) || cvRound(got.y) != cvRound(est.y);
        est = got;
        if (!moved) break;
    }
    const FitResult f2 = fitWindow(gray, c, est, start, 5);   // 11x11 창: 안정성 확인
    CornerFit out;
    out.p = cv::Point2d(f1.p[0], f1.p[1]);
    out.rmse = f1.rmse;
    out.contrast = f1.p[5];
    out.blur = std::max(f1.p[2], f1.p[3]);
    const double stability = cv::norm(out.p - cv::Point2d(f2.p[0], f2.p[1]));
    // 통과 기준: 수렴, RMSE <= 15 (대각선 35), 대비 >= 45, 두 창 차이 <= 0.65px, 시작점에서 3.6px 이내, 흐림 <= 2.6
    if (!f1.converged || !f2.converged) out.reason = "수렴 안 됨";
    // 대각선 모서리는 모듈 사이 밝은 틈 때문에 모델이 덜 맞아서(RMSE 20~30) 35 까지 허용.
    // 대각선은 사진 이동 평균에는 쓰지 않고, "왜 빼는지" 흔들림 비교에만 쓴다.
    else if (out.rmse > (c.diagonal ? 35 : 15)) out.reason = "모델과 밝기 차이 큼";
    else if (out.contrast < 45) out.reason = "대비 낮음";
    else if (stability > 0.65) out.reason = "창 크기에 따라 위치가 바뀜";
    else if (std::fabs(out.p.x - start.x) > 3.6 || std::fabs(out.p.y - start.y) > 3.6) out.reason = "시작점에서 너무 멂";
    else if (out.blur > 2.6) out.reason = "흐림 큼";
    out.ok = out.reason.empty();
    return out;
}

std::vector<Corner> findCorners(const cv::Mat &gray, const std::array<QrRef, kQrCount> &qrs) {
    std::vector<Corner> out;
    std::map<std::string, int> dropped;   // 탈락 이유별 개수 (로그용)
    for (int q = 0; q < kQrCount; ++q) {
        // 라벨 안쪽만 (라벨 테두리 5px 는 뺌)
        const cv::Rect inner(qrs[q].label.x + 5, qrs[q].label.y + 5, qrs[q].label.width - 10, qrs[q].label.height - 10);
        cv::Mat bin;
        const double thr = cv::threshold(gray(inner), bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

        std::vector<cv::Point2f> pts;
        cv::goodFeaturesToTrack(gray(inner), pts, 1500, 0.01, 4, cv::noArray(), 5);
        for (const auto &p0 : pts) {
            Corner c;
            c.p = p0 + cv::Point2f(static_cast<float>(inner.x), static_cast<float>(inner.y));
            c.qr = q;
            const int kind = cornerKind(gray, c.p, thr, c.black);
            if (kind == 0) continue;
            c.diagonal = kind == 2;
            // center 에서도 같은 밝기 모델로 꼭짓점을 맞춘다. 통과 못 하면 버림
            const CornerFit f = fitCorner(gray, c, cv::Point2d(c.p));
            if (!f.ok) { ++dropped[f.reason]; continue; }
            c.p = cv::Point2f(static_cast<float>(f.p.x), static_cast<float>(f.p.y));
            // 같은 꼭짓점을 두 번 잡은 것은 버림
            const bool dup = std::any_of(out.begin(), out.end(), [&](const Corner &o) { return o.qr == q && cv::norm(o.p - c.p) < 2.0; });
            if (dup) ++dropped["같은 꼭짓점 중복"];
            else out.push_back(c);
        }
    }
    std::string msg = "center 모서리 맞춤 탈락:";
    for (const auto &[reason, n] : dropped) msg += " " + reason + " " + std::to_string(n) + ",";
    log(msg);
    return out;
}

std::vector<cv::Point2f> trackCorners(const cv::Mat &gray, const std::vector<Corner> &corners,
                                      const std::array<QrRef, kQrCount> &qrs,
                                      const std::array<EccMotion, kQrCount> &ecc) {
    const float nanf = static_cast<float>(kNaN);
    std::vector<cv::Point2f> out(corners.size(), cv::Point2f(nanf, nanf));
    for (size_t i = 0; i < corners.size(); ++i) {
        const int q = corners[i].qr;
        if (!ecc[q].ok) continue;
        // ECC 변환으로 이 사진에서의 자리를 예측하고, 그 자리에서 밝기 모델을 맞춘다
        const cv::Point2d local(corners[i].p.x - qrs[q].patch.x, corners[i].p.y - qrs[q].patch.y);
        const cv::Matx23d &W = ecc[q].warp;
        const cv::Point2d pred(W(0, 0) * local.x + W(0, 1) * local.y + W(0, 2), W(1, 0) * local.x + W(1, 1) * local.y + W(1, 2));
        const CornerFit f = fitCorner(gray, corners[i], pred);
        if (f.ok) out[i] = cv::Point2f(static_cast<float>(f.p.x), static_cast<float>(f.p.y));
    }
    return out;
}

}  // namespace hw
