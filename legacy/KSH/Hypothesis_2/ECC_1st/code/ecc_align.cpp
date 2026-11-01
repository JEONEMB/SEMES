// =====================================================================================
// ecc_align.cpp — 가설 H2 : ECC 전역 원근 정합 구현 (흐름 설명은 ecc_align.hpp 머리말)
// =====================================================================================
#include "ecc_align.hpp"
#include <cmath>
#include <chrono>
#include <algorithm>
#include <climits>
#include <cstdio>

namespace ea {

namespace {

cv::Mat toGray8(const cv::Mat& src) {
    cv::Mat g;
    if (src.channels() == 3) cv::cvtColor(src, g, cv::COLOR_BGR2GRAY);
    else if (src.channels() == 4) cv::cvtColor(src, g, cv::COLOR_BGRA2GRAY);
    else g = src;
    if (g.depth() != CV_8U) g.convertTo(g, CV_8U);
    return g;
}

struct Candidate { cv::Rect rect; cv::Point2f c; };

// 밝은 사각형(QR 바탕) 후보 : 임계값 130 → 닫힘 → 크기·정사각형·충전율 조건
std::vector<Candidate> findCandidates(const cv::Mat& gray8) {
    cv::Mat bin;
    cv::GaussianBlur(gray8, bin, cv::Size(5, 5), 0);
    cv::threshold(bin, bin, 130, 255, cv::THRESH_BINARY);
    cv::morphologyEx(bin, bin, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9)));
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bin, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    std::vector<Candidate> out;
    for (size_t i = 0; i < contours.size(); ++i) {
        cv::Rect r = cv::boundingRect(contours[i]);
        double ar = (double)r.width / r.height;
        if (r.width < 120 || r.width > 300 || ar < 0.8 || ar > 1.25) continue;
        if (cv::contourArea(contours[i]) < 0.85 * r.area()) continue;
        cv::Moments m = cv::moments(contours[i]);
        Candidate c;
        c.rect = r;
        c.c = cv::Point2f((float)(m.m10 / m.m00), (float)(m.m01 / m.m00));
        out.push_back(c);
    }
    return out;
}

int findNeighbor(const std::vector<Candidate>& cs, const cv::Point2f& p, double dx, double tol) {
    int best = -1;
    double bestErr = tol;
    for (size_t i = 0; i < cs.size(); ++i) {
        double ex = std::fabs((cs[i].c.x - p.x) - dx), ey = std::fabs(cs[i].c.y - p.y);
        if (ey < tol && ex < bestErr) { bestErr = ex; best = (int)i; }
    }
    return best;
}

// 호모그래피로 점 p 를 옮긴 위치
cv::Point2d applyH(const cv::Mat& H, double x, double y) {
    const double* h = H.ptr<double>(0);
    double w = h[6] * x + h[7] * y + h[8];
    return cv::Point2d((h[0] * x + h[1] * y + h[2]) / w, (h[3] * x + h[4] * y + h[5]) / w);
}

// double 정밀도 bicubic 보간 (Keys, a = −0.5) : 값과 x·y 미분을 함께 계산
//   OpenCV remap / warpPerspective 는 내부에서 좌표를 1/32 px 로 양자화해서(INTER_BITS = 5)
//   0.01 px 보다 작은 이동을 다루는 반복 정합이 0.02 px 크기로 진동함 → 직접 보간
inline void keys(double t, double w[4], double dw[4]) {
    // 표본 위치 −1, 0, 1, 2 에 대한 가중치와 t 미분
    const double a = -0.5;
    const double x0 = 1 + t, x1 = t, x2 = 1 - t, x3 = 2 - t;
    w[0] = ((a * x0 - 5 * a) * x0 + 8 * a) * x0 - 4 * a;
    w[1] = ((a + 2) * x1 - (a + 3)) * x1 * x1 + 1;
    w[2] = ((a + 2) * x2 - (a + 3)) * x2 * x2 + 1;
    w[3] = ((a * x3 - 5 * a) * x3 + 8 * a) * x3 - 4 * a;
    dw[0] = (3 * a * x0 - 10 * a) * x0 + 8 * a;              // d/dt (x0 = 1 + t)
    dw[1] = (3 * (a + 2) * x1 - 2 * (a + 3)) * x1;           // d/dt (x1 = t)
    dw[2] = -((3 * (a + 2) * x2 - 2 * (a + 3)) * x2);        // d/dt (x2 = 1 − t)
    dw[3] = -((3 * a * x3 - 10 * a) * x3 + 8 * a);           // d/dt (x3 = 2 − t)
}

inline bool sampleBicubic(const cv::Mat& img, double x, double y, double& v, double& gx, double& gy) {
    const int ix = (int)std::floor(x), iy = (int)std::floor(y);
    if (ix < 1 || iy < 1 || ix + 2 >= img.cols || iy + 2 >= img.rows) return false;
    double wx[4], dwx[4], wy[4], dwy[4];
    keys(x - ix, wx, dwx);
    keys(y - iy, wy, dwy);
    v = gx = gy = 0;
    for (int j = 0; j < 4; ++j) {
        const float* row = img.ptr<float>(iy - 1 + j) + (ix - 1);
        double r = 0, rd = 0;
        for (int k = 0; k < 4; ++k) { r += wx[k] * row[k]; rd += dwx[k] * row[k]; }
        v += wy[j] * r; gx += wy[j] * rd; gy += dwy[j] * r;
    }
    return true;
}

} // namespace

// =====================================================================================
// QR 찾기 (기준 영상에서만)
// =====================================================================================
bool findQRBoxes(const cv::Mat& src, QRBox boxes[NUM_QR], double pitch) {
    cv::Mat gray8 = toGray8(src);
    std::vector<Candidate> cs = findCandidates(gray8);
    int bc = -1, bl = -1, br = -1;
    double bestDist = 1e18;
    for (size_t i = 0; i < cs.size(); ++i) {
        int l = findNeighbor(cs, cs[i].c, -pitch, 50.0);
        int r = findNeighbor(cs, cs[i].c, +pitch, 50.0);
        if (l < 0 || r < 0) continue;
        double d = std::fabs(cs[i].c.x - gray8.cols * 0.5);
        if (d < bestDist) { bestDist = d; bc = (int)i; bl = l; br = r; }
    }
    if (bc < 0) return false;
    int idx[NUM_QR] = { bl, bc, br };
    for (int q = 0; q < NUM_QR; ++q) { boxes[q].rect = cs[idx[q]].rect; boxes[q].centroid = cs[idx[q]].c; }
    return true;
}

// =====================================================================================
// EccAligner
// =====================================================================================
bool EccAligner::setReference(const cv::Mat& image) {
    hasRef_ = false;
    cv::Mat g8 = toGray8(image);
    if (!findQRBoxes(g8, box_, p_.qrPitch)) return false;

    // 띠 : QR 3개를 감싸는 사각형 + 여유
    int x0 = box_[0].rect.x - p_.bandPadX, x1 = box_[2].rect.br().x + p_.bandPadX;
    int y0 = INT_MAX, y1 = 0;
    for (int q = 0; q < NUM_QR; ++q) { y0 = std::min(y0, box_[q].rect.y); y1 = std::max(y1, box_[q].rect.br().y); }
    y0 -= p_.bandPadY; y1 += p_.bandPadY;
    band_ = cv::Rect(x0, y0, x1 - x0, y1 - y0) & cv::Rect(0, 0, g8.cols, g8.rows);
    if (band_.width < 100 || band_.height < 50) return false;

    g8(band_).convertTo(ref32_, CV_32F);
    if (p_.blurSigma > 0) cv::GaussianBlur(ref32_, ref32_, cv::Size(0, 0), p_.blurSigma);

    // 마스크 : QR 사각형 + maskPad (판 무늬·LED 번짐 영역 제외)
    mask_ = cv::Mat::zeros(band_.size(), CV_8U);
    for (int q = 0; q < NUM_QR; ++q) {
        cv::Rect r = box_[q].rect;
        r.x -= band_.x + p_.maskPad; r.y -= band_.y + p_.maskPad;
        r.width += 2 * p_.maskPad;   r.height += 2 * p_.maskPad;
        r &= cv::Rect(0, 0, band_.width, band_.height);
        mask_(r).setTo(255);
    }
    cv::createHanningWindow(hann_, band_.size(), CV_32F);

    // method 1 준비 : 마스크 화소 목록, 정규화 좌표, 기준 밝기(평균 0, 노름 1)
    xc_ = 0.5 * (band_.width - 1); yc_ = 0.5 * (band_.height - 1); S_ = 0.5 * band_.width;
    px_.clear(); xh_.clear(); yh_.clear(); tn_.clear();
    for (int y = 0; y < band_.height; ++y) {
        const uchar* mk = mask_.ptr<uchar>(y);
        const float* rv = ref32_.ptr<float>(y);
        for (int x = 0; x < band_.width; ++x) {
            if (!mk[x]) continue;
            px_.push_back(cv::Point2f((float)x, (float)y));
            xh_.push_back((x - xc_) / S_);
            yh_.push_back((y - yc_) / S_);
            tn_.push_back(rv[x]);
        }
    }
    double mean = 0, nrm = 0;
    for (double v : tn_) mean += v;
    mean /= std::max<size_t>(1, tn_.size());
    for (double& v : tn_) { v -= mean; nrm += v * v; }
    nrm = std::sqrt(nrm);
    if (nrm < 1e-9) return false;
    for (double& v : tn_) v /= nrm;

    hasRef_ = true;
    return true;
}

cv::Point2f EccAligner::qrCenter(int q) const {
    const cv::Rect& r = box_[q].rect;
    return cv::Point2f(r.x + 0.5f * (r.width - 1) - band_.x, r.y + 0.5f * (r.height - 1) - band_.y);
}

Measure EccAligner::measure(const cv::Mat& image) const {
    Measure m;
    if (!hasRef_) return m;
    auto t0 = std::chrono::steady_clock::now();
    cv::Mat g8 = toGray8(image);
    if (band_.br().x > g8.cols || band_.br().y > g8.rows) return m;

    // 1) 거친 단계 : 같은 위치의 띠끼리 위상 상관
    cv::Mat cur32;
    g8(band_).convertTo(cur32, CV_32F);
    cv::Point2d sh = cv::phaseCorrelate(ref32_, cur32, hann_);
    m.coarseDx = sh.x; m.coarseDy = sh.y;

    // 2) 정밀 단계 : 현재 영상은 띠보다 넓게 잘라 와서 (큰 이동도 영역 안에 들어오게) 정합
    cv::Rect sr(band_.x - p_.searchX, band_.y - p_.searchY, band_.width + 2 * p_.searchX, band_.height + 2 * p_.searchY);
    sr &= cv::Rect(0, 0, g8.cols, g8.rows);
    const double ox = band_.x - sr.x, oy = band_.y - sr.y;   // 띠 원점이 검색 영역 안에서 있는 위치
    cv::Mat in32;
    g8(sr).convertTo(in32, CV_32F);
    if (p_.blurSigma > 0) cv::GaussianBlur(in32, in32, cv::Size(0, 0), p_.blurSigma);

    const bool ok = (p_.method == 0) ? measureEcc(in32, ox, oy, m) : measureGN(in32, ox, oy, m);
    if (!ok) return m;
    m.ok = true;
    m.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return m;
}

// method 0 : OpenCV findTransformECC 호모그래피 → 특징
bool EccAligner::measureEcc(const cv::Mat& in32, double ox, double oy, Measure& m) const {
    // ECC 마스크는 "현재 영상" 기준 (OpenCV : inputMask = 입력 영상의 유효 화소).
    //   기준 QR 사각형을 거친 이동만큼 옮겨 표시 (+ maskPad, 정밀 이동·원근 여유)
    cv::Mat inMask = cv::Mat::zeros(in32.size(), CV_8U);
    for (int q = 0; q < NUM_QR; ++q) {
        cv::Rect r = box_[q].rect;
        r.x = (int)std::floor(r.x - band_.x + ox + m.coarseDx) - p_.maskPad;
        r.y = (int)std::floor(r.y - band_.y + oy + m.coarseDy) - p_.maskPad;
        r.width += 2 * p_.maskPad; r.height += 2 * p_.maskPad;
        r &= cv::Rect(0, 0, in32.cols, in32.rows);
        if (r.area() > 0) inMask(r).setTo(255);
    }
    cv::Mat W = cv::Mat::eye(3, 3, CV_32F);
    W.at<float>(0, 2) = (float)(m.coarseDx + ox);
    W.at<float>(1, 2) = (float)(m.coarseDy + oy);
    try {
        m.cc = cv::findTransformECC(ref32_, in32, W, cv::MOTION_HOMOGRAPHY,
                                    cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, p_.maxIter, p_.eps),
                                    inMask);
    } catch (const cv::Exception& ex) {
#ifdef ECC_DEBUG
        std::fprintf(stderr, "ECC exception: %s\n", ex.what());
#endif
        (void)ex;
        return false;   // 수렴 실패
    }
    // 검색 영역 좌표 → 띠 좌표 :  H = T(−ox, −oy) · W
    cv::Mat W64;
    W.convertTo(W64, CV_64F);
    cv::Mat T = (cv::Mat_<double>(3, 3) << 1, 0, -ox, 0, 1, -oy, 0, 0, 1);
    m.H = T * W64;
    // 특징 : QR 중심의 가로 이동 t, 가로 배율 s = 100 · ∂u/∂x (중심 차분)
    const double e = p_.derivStep;
    for (int q = 0; q < NUM_QR; ++q) {
        cv::Point2f c = qrCenter(q);
        cv::Point2d p0 = applyH(m.H, c.x, c.y);
        cv::Point2d pl = applyH(m.H, c.x - e, c.y), pr = applyH(m.H, c.x + e, c.y);
        m.f[2 * q]     = p0.x - c.x;
        m.f[2 * q + 1] = 100.0 * ((pr.x - (c.x + e)) - (pl.x - (c.x - e))) / (2 * e);
        if (q == 1) m.vC = p0.y - c.y;
    }
    return true;
}

// method 1 : ZNCC(= ECC 기준) Gauss-Newton, 원근 다항 워프 (double 정밀도)
//   I(x + u, y + v) 를 기준 T 와 정규화 상관이 최대가 되게 맞춤.  워프가 매개변수에 선형이므로
//   J = ∇I · ∂W/∂p,  정규화 잔차 r = Î − T̂ ,  (J̃ᵀJ̃) Δp = −J̃ᵀ r  를 반복
bool EccAligner::measureGN(const cv::Mat& in32, double ox, double oy, Measure& m) const {
    const int N = (int)px_.size();
    if (N < 1000) return false;
    double p[9] = { m.coarseDx, 0, 0, 0, 0, m.coarseDy, 0, 0, 0 };
    std::vector<float> iwv(N), gxv(N), gyv(N);
    double cc = 0;
    int it = 0;
    for (it = 1; it <= p_.maxIter; ++it) {
        // 현재 매개변수로 마스크 화소 위치를 옮겨 밝기·기울기를 bicubic 으로 읽음
        for (int i = 0; i < N; ++i) {
            const double xh = xh_[i], yh = yh_[i];
            const double u = p[0] + p[1] * xh + p[2] * xh * xh + p[3] * yh + p[4] * xh * yh;
            const double v = p[5] + p[6] * yh + p[7] * xh + p[8] * xh * yh;
            double val, dgx, dgy;
            if (!sampleBicubic(in32, px_[i].x + u + ox, px_[i].y + v + oy, val, dgx, dgy)) return false;
            iwv[i] = (float)val; gxv[i] = (float)dgx; gyv[i] = (float)dgy;
        }
        const float* iw = iwv.data();
        const float* ggx = gxv.data();
        const float* ggy = gyv.data();

        // 정규화 (평균 0, 노름 1)
        double mI = 0;
        for (int i = 0; i < N; ++i) mI += iw[i];
        mI /= N;
        double nI = 0;
        for (int i = 0; i < N; ++i) nI += (iw[i] - mI) * (iw[i] - mI);
        nI = std::sqrt(nI);
        if (nI < 1e-9) return false;

        // 정규화된 현재 밝기 Î = (I − mI) / nI 의 매개변수 미분 (정규화 항까지 포함한 정확한 야코비안)
        //   ∂Î/∂p = (J − J̄)/nI − Î · Σ[Î · (J − J̄)/nI]
        double Jm[9] = {0}, cI[9] = {0}, A[81] = {0}, bvec[9] = {0};
        auto jac = [&](int i, double* J9) {
            const double xh = xh_[i], yh = yh_[i], gxv = ggx[i], gyv = ggy[i];
            J9[0] = gxv; J9[1] = gxv * xh; J9[2] = gxv * xh * xh; J9[3] = gxv * yh; J9[4] = gxv * xh * yh;
            J9[5] = gyv; J9[6] = gyv * yh; J9[7] = gyv * xh; J9[8] = gyv * xh * yh;
        };
        double J9[9];
        for (int i = 0; i < N; ++i) { jac(i, J9); for (int k = 0; k < 9; ++k) Jm[k] += J9[k]; }
        for (int k = 0; k < 9; ++k) Jm[k] /= N;
        for (int i = 0; i < N; ++i) {
            jac(i, J9);
            const double ih = (iw[i] - mI) / nI;
            for (int k = 0; k < 9; ++k) cI[k] += ih * (J9[k] - Jm[k]) / nI;
        }
        double ss = 0, jt[9];
        for (int i = 0; i < N; ++i) {
            jac(i, J9);
            const double ih = (iw[i] - mI) / nI;
            const double r = ih - tn_[i];
            ss += r * r;
            for (int k = 0; k < 9; ++k) jt[k] = (J9[k] - Jm[k]) / nI - ih * cI[k];
            for (int a = 0; a < 9; ++a) {
                bvec[a] += jt[a] * r;
                for (int b = a; b < 9; ++b) A[a * 9 + b] += jt[a] * jt[b];
            }
        }
        for (int a = 0; a < 9; ++a) for (int b = 0; b < a; ++b) A[a * 9 + b] = A[b * 9 + a];
        cc = 1.0 - 0.5 * ss;
        cv::Mat Am(9, 9, CV_64F, A), bm(9, 1, CV_64F, bvec), d;
        if (!cv::solve(Am, -bm, d, cv::DECOMP_CHOLESKY) && !cv::solve(Am, -bm, d, cv::DECOMP_SVD)) return false;
        for (int k = 0; k < 9; ++k) p[k] += d.at<double>(k);

        // 수렴 : QR 중심 세 곳의 가로 이동 변화가 eps(px) 이하
        double maxDu = 0;
        for (int q = 0; q < NUM_QR; ++q) {
            const cv::Point2f c = qrCenter(q);
            const double xh = (c.x - xc_) / S_, yh = (c.y - yc_) / S_;
            const double du = d.at<double>(0) + d.at<double>(1) * xh + d.at<double>(2) * xh * xh +
                              d.at<double>(3) * yh + d.at<double>(4) * xh * yh;
            maxDu = std::max(maxDu, std::fabs(du));
        }
        if (maxDu < p_.eps) break;
    }
    m.cc = cc;
    m.iters = std::min(it, p_.maxIter);
    for (int k = 0; k < 9; ++k) m.p[k] = p[k];
    // 특징 : QR 중심의 가로 이동 t = u,  가로 배율 s = 100 · ∂u/∂x  (해석적 미분)
    for (int q = 0; q < NUM_QR; ++q) {
        const cv::Point2f c = qrCenter(q);
        const double xh = (c.x - xc_) / S_, yh = (c.y - yc_) / S_;
        m.f[2 * q]     = p[0] + p[1] * xh + p[2] * xh * xh + p[3] * yh + p[4] * xh * yh;
        m.f[2 * q + 1] = 100.0 * (p[1] + 2 * p[2] * xh + p[4] * yh) / S_;
        if (q == 1) m.vC = p[5] + p[6] * yh + p[7] * xh + p[8] * xh * yh;
    }
    return true;
}

// =====================================================================================
// 학습 / 추정
// =====================================================================================
Model fitModel(const std::vector<Sample>& train, const EccParams& p, bool quad, const std::string& note) {
    Model md;
    md.p = p;
    md.quad = quad;
    md.note = note;
    const int n = (int)train.size(), U = quad ? 4 : 3;   // 미지수 : a, b, (q), c
    if (n < U + 2) return md;

    cv::Mat X(n, U, CV_64F), F(n, NUM_FEAT, CV_64F);
    for (int i = 0; i < n; ++i) {
        X.at<double>(i, 0) = train[i].y;
        X.at<double>(i, 1) = train[i].t;
        if (quad) X.at<double>(i, 2) = train[i].t * train[i].t;
        X.at<double>(i, U - 1) = 1.0;
        for (int k = 0; k < NUM_FEAT; ++k) F.at<double>(i, k) = train[i].f[k];
    }
    cv::Mat B;   // U × NUM_FEAT
    cv::solve(X, F, B, cv::DECOMP_SVD);
    for (int k = 0; k < NUM_FEAT; ++k) {
        md.a[k] = B.at<double>(0, k);
        md.b[k] = B.at<double>(1, k);
        md.q[k] = quad ? B.at<double>(2, k) : 0.0;
        md.c[k] = B.at<double>(U - 1, k);
    }
    cv::Mat R = F - X * B;
    md.cov = (R.t() * R) / std::max(1, n - U) + cv::Mat::eye(NUM_FEAT, NUM_FEAT, CV_64F) * 1e-12;
    md.nTrain = n;
    md.ok = true;
    return md;
}

Estimate estimate(const Model& m, const double f[NUM_FEAT]) {
    Estimate est;
    if (!m.ok) return est;
    cv::Mat Si = m.cov.inv(cv::DECOMP_SVD);
    cv::Mat r0(NUM_FEAT, 1, CV_64F);
    for (int k = 0; k < NUM_FEAT; ++k) r0.at<double>(k) = f[k] - m.c[k];

    double y = 0, t = 0;
    cv::Mat J(NUM_FEAT, 2, CV_64F), N;
    for (int it = 0; it < 8; ++it) {          // Θ² 항 때문에 Gauss-Newton (선형이면 1회로 끝)
        cv::Mat r(NUM_FEAT, 1, CV_64F);
        for (int k = 0; k < NUM_FEAT; ++k) {
            r.at<double>(k) = r0.at<double>(k) - (m.a[k] * y + m.b[k] * t + m.q[k] * t * t);
            J.at<double>(k, 0) = m.a[k];
            J.at<double>(k, 1) = m.b[k] + 2.0 * m.q[k] * t;
        }
        N = J.t() * Si * J;
        cv::Mat d = N.inv(cv::DECOMP_SVD) * (J.t() * Si * r);
        y += d.at<double>(0);
        t += d.at<double>(1);
        if (std::fabs(d.at<double>(0)) < 1e-9 && std::fabs(d.at<double>(1)) < 1e-9) break;
    }
    cv::Mat Ni = N.inv(cv::DECOMP_SVD);
    est.y = y; est.t = t;
    est.sdY = std::sqrt(std::max(0.0, Ni.at<double>(0, 0)));
    est.sdT = std::sqrt(std::max(0.0, Ni.at<double>(1, 1)));
    est.valid = std::isfinite(y) && std::isfinite(t);
    return est;
}

// =====================================================================================
// 모델 파일 (yml) : MFC 앱도 이 파일 하나로 같은 측정·추정을 재현
// =====================================================================================
bool Model::save(const std::string& path) const {
    cv::FileStorage fs(path, cv::FileStorage::WRITE);
    if (!fs.isOpened()) return false;
    fs << "model_type" << "ECC_BAND_v1";
    fs << "note" << note << "nTrain" << nTrain << "quad" << (int)quad;
    fs << "method" << p.method;
    fs << "bandPadX" << p.bandPadX << "bandPadY" << p.bandPadY << "searchX" << p.searchX << "searchY" << p.searchY
       << "maskPad" << p.maskPad << "blurSigma" << p.blurSigma << "maxIter" << p.maxIter << "eps" << p.eps
       << "derivStep" << p.derivStep << "qrPitch" << p.qrPitch;
    fs << "features" << "tL sL tC sC tR sR (t px, s px per 100px)";
    fs << "model_form" << "f = a Y + b Theta + q Theta2 + c (Y mm, Theta deg)";
    fs << "a" << std::vector<double>(a, a + NUM_FEAT) << "b" << std::vector<double>(b, b + NUM_FEAT)
       << "q" << std::vector<double>(q, q + NUM_FEAT) << "c" << std::vector<double>(c, c + NUM_FEAT);
    fs << "cov" << cov;
    return true;
}

bool Model::load(const std::string& path) {
    cv::FileStorage fs;
    try { fs.open(path, cv::FileStorage::READ); } catch (const cv::Exception&) { return false; }
    if (!fs.isOpened()) return false;
    std::string type;
    fs["model_type"] >> type;
    if (type != "ECC_BAND_v1") return false;
    int qd = 1;
    fs["note"] >> note; fs["nTrain"] >> nTrain; fs["quad"] >> qd; quad = (qd != 0);
    if (!fs["method"].empty()) fs["method"] >> p.method;
    fs["bandPadX"] >> p.bandPadX; fs["bandPadY"] >> p.bandPadY; fs["searchX"] >> p.searchX; fs["searchY"] >> p.searchY;
    fs["maskPad"] >> p.maskPad; fs["blurSigma"] >> p.blurSigma; fs["maxIter"] >> p.maxIter; fs["eps"] >> p.eps;
    fs["derivStep"] >> p.derivStep; fs["qrPitch"] >> p.qrPitch;
    std::vector<double> va, vb, vq, vc;
    fs["a"] >> va; fs["b"] >> vb; fs["q"] >> vq; fs["c"] >> vc; fs["cov"] >> cov;
    if (va.size() != NUM_FEAT || vb.size() != NUM_FEAT || vq.size() != NUM_FEAT || vc.size() != NUM_FEAT ||
        cov.rows != NUM_FEAT || cov.cols != NUM_FEAT) return false;
    for (int k = 0; k < NUM_FEAT; ++k) { a[k] = va[k]; b[k] = vb[k]; q[k] = vq[k]; c[k] = vc[k]; }
    ok = true;
    return true;
}

} // namespace ea
