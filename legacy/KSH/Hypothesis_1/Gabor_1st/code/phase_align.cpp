// =====================================================================================
// phase_align.cpp — 위상 기반 Y·Θ 분리 알고리즘 구현 (흐름 설명은 phase_align.hpp 머리말)
// =====================================================================================
#include "phase_align.hpp"
#include <cmath>
#include <cstdio>
#include <climits>
#include <algorithm>

namespace pa {

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

bool inside(const cv::Rect& r, const cv::Mat& m) {
    return r.x >= 0 && r.y >= 0 && r.x + r.width <= m.cols && r.y + r.height <= m.rows && r.width > 0 && r.height > 0;
}

} // namespace

// =====================================================================================
// 기준 영상 QR 검출
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
// Calibration 저장 / 읽기
// =====================================================================================
bool Calibration::save(const std::string& path, const PhaseParams& p) const {
    cv::FileStorage fs(path, cv::FileStorage::WRITE);
    if (!fs.isOpened()) return false;
    fs << "lambda" << p.lambda << "sigmaX" << p.sigmaX << "sigmaY" << p.sigmaY
       << "roiPadX" << p.roiPadX << "roiInsetY" << p.roiInsetY << "ampPercentile" << p.ampPercentile
       << "maxGradErr" << p.maxGradErr << "maxResidualFrac" << p.maxResidualFrac
       << "minValidRatio" << p.minValidRatio << "maxFitStd" << p.maxFitStd;
    fs << "a" << std::vector<double>(a, a + NUM_FEAT);
    fs << "b" << std::vector<double>(b, b + NUM_FEAT);
    fs << "c" << std::vector<double>(c, c + NUM_FEAT);
    fs << "cov" << cov;
    return true;
}

bool Calibration::load(const std::string& path, PhaseParams* p) {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) return false;
    std::vector<double> va, vb, vc;
    fs["a"] >> va; fs["b"] >> vb; fs["c"] >> vc; fs["cov"] >> cov;
    if (va.size() != NUM_FEAT || vb.size() != NUM_FEAT || vc.size() != NUM_FEAT ||
        cov.rows != NUM_FEAT || cov.cols != NUM_FEAT) return false;
    for (int k = 0; k < NUM_FEAT; ++k) { a[k] = va[k]; b[k] = vb[k]; c[k] = vc[k]; }
    if (p) {
        fs["lambda"] >> p->lambda; fs["sigmaX"] >> p->sigmaX; fs["sigmaY"] >> p->sigmaY;
        fs["roiPadX"] >> p->roiPadX; fs["roiInsetY"] >> p->roiInsetY; fs["ampPercentile"] >> p->ampPercentile;
        fs["maxGradErr"] >> p->maxGradErr; fs["maxResidualFrac"] >> p->maxResidualFrac;
        fs["minValidRatio"] >> p->minValidRatio; fs["maxFitStd"] >> p->maxFitStd;
    }
    ok = true;
    return true;
}

// =====================================================================================
// PhaseAligner
// =====================================================================================
void PhaseAligner::buildKernel() {
    // 가로 : 가우시안 × (cos, sin) 복소 물결.  세로 : 가우시안.  (분리형 필터 → 빠름)
    int hx = params_.halfX(), hy = params_.halfY();
    kx_cos_.create(1, 2 * hx + 1, CV_64F);
    kx_sin_.create(1, 2 * hx + 1, CV_64F);
    std::vector<double> g(2 * hx + 1);
    double sumG = 0, sumC = 0;
    for (int k = -hx; k <= hx; ++k) {
        g[k + hx] = std::exp(-(k * k) / (2.0 * params_.sigmaX * params_.sigmaX));
        kx_cos_.at<double>(k + hx) = g[k + hx] * std::cos(2 * CV_PI * k / params_.lambda);
        kx_sin_.at<double>(k + hx) = g[k + hx] * std::sin(2 * CV_PI * k / params_.lambda);
        sumG += g[k + hx]; sumC += kx_cos_.at<double>(k + hx);
    }
    // 실수부 DC 제거 : 밝기 오프셋·완만한 배경(LED 번짐)에 반응하지 않게 함
    for (int k = 0; k < 2 * hx + 1; ++k) kx_cos_.at<double>(k) -= sumC * g[k] / sumG;

    ky_.create(2 * hy + 1, 1, CV_64F);
    double sy = 0;
    for (int k = -hy; k <= hy; ++k) {
        ky_.at<double>(k + hy) = std::exp(-(k * k) / (2.0 * params_.sigmaY * params_.sigmaY));
        sy += ky_.at<double>(k + hy);
    }
    ky_ /= sy;
}

void PhaseAligner::response(const cv::Mat& g32, cv::Rect r, cv::Mat& re, cv::Mat& im) const {
    // 출력 : r.height × (r.width + 2)  — 열 r.x−1 … r.x+r.width
    int hx = params_.halfX(), hy = params_.halfY();
    cv::Rect src(r.x - 1 - hx, r.y - hy, r.width + 2 + 2 * hx, r.height + 2 * hy);
    cv::Mat tmpRe, tmpIm;
    cv::sepFilter2D(g32(src), tmpRe, CV_64F, kx_cos_, ky_, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
    cv::sepFilter2D(g32(src), tmpIm, CV_64F, kx_sin_, ky_, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
    cv::Rect in(hx, hy, r.width + 2, r.height);
    re = tmpRe(in).clone();
    im = tmpIm(in).clone();
}

bool PhaseAligner::setReference(const cv::Mat& src) {
    hasRef_ = false;
    cv::Mat gray8 = toGray8(src);
    if (!findQRBoxes(gray8, refBox_)) return false;
    gray8.convertTo(ref32_, CV_32F);
    buildKernel();

    int top = INT_MAX, bot = 0;
    for (int q = 0; q < NUM_QR; ++q) {
        const cv::Rect& b = refBox_[q].rect;
        roi_[q] = cv::Rect(b.x - params_.roiPadX, b.y + params_.roiInsetY,
                           b.width + 2 * params_.roiPadX, b.height - 2 * params_.roiInsetY);
        top = std::min(top, b.y); bot = std::max(bot, b.y + b.height);
    }
    // 거친 단계용 띠 : QR 3개를 감싸는 가로로 긴 영역
    int x0 = std::max(0, refBox_[0].rect.x - 512), x1 = std::min(ref32_.cols, refBox_[2].rect.br().x + 512);
    int y0 = std::max(0, top - 32), y1 = std::min(ref32_.rows, bot + 32);
    band_ = cv::Rect(x0, y0, x1 - x0, y1 - y0);
    ref32_(band_).copyTo(refBand_);
    cv::createHanningWindow(hann_, band_.size(), CV_32F);

    int margin = params_.halfX() + 2, marginY = params_.halfY() + 1;
    for (int q = 0; q < NUM_QR; ++q) {
        cv::Rect need(roi_[q].x - margin, roi_[q].y - marginY, roi_[q].width + 2 * margin, roi_[q].height + 2 * marginY);
        if (!inside(need, ref32_)) return false;
        response(ref32_, roi_[q], refRe_[q], refIm_[q]);
    }
    hasRef_ = true;
    return true;
}

void PhaseAligner::setParams(const PhaseParams& p) {
    params_ = p;
    buildKernel();
    if (!ref32_.empty()) {
        cv::Mat r8;
        ref32_.convertTo(r8, CV_8U);
        setReference(r8);
    }
}

QRMeasure PhaseAligner::measureQR(const cv::Mat& g32, int q, int ix, int iy) const {
    QRMeasure m;
    const cv::Rect& r0 = roi_[q];
    cv::Rect r(r0.x + ix, r0.y + iy, r0.width, r0.height);
    int margin = params_.halfX() + 2, marginY = params_.halfY() + 1;
    cv::Rect need(r.x - margin, r.y - marginY, r.width + 2 * margin, r.height + 2 * marginY);
    if (!inside(need, g32)) return m;

    cv::Mat cRe, cIm;
    response(g32, r, cRe, cIm);
    const cv::Mat& aRe = refRe_[q];
    const cv::Mat& aIm = refIm_[q];
    const double ideal = 2 * CV_PI / params_.lambda;
    const int H = r.height, W = r.width;

    // 1차 : 화소별 위상 변화·기울기·응답 크기
    std::vector<double> dx(H * W), amp(H * W), gr(H * W);
    for (int y = 0; y < H; ++y) {
        const double *ar = aRe.ptr<double>(y), *ai = aIm.ptr<double>(y);
        const double *br = cRe.ptr<double>(y), *bi = cIm.ptr<double>(y);
        for (int x = 0; x < W; ++x) {
            int j = x + 1;                                    // 응답 배열은 왼쪽 1열 여유
            // 위상 변화 Δφ = arg(b · conj(a))
            double dphi = std::atan2(bi[j] * ar[j] - br[j] * ai[j], br[j] * ar[j] + bi[j] * ai[j]);
            // 위상 기울기 = arg(z(x+1) · conj(z(x−1))) / 2,  기준·현재 평균 (논문 Eq. 35)
            double ga = std::atan2(ai[j + 1] * ar[j - 1] - ar[j + 1] * ai[j - 1], ar[j + 1] * ar[j - 1] + ai[j + 1] * ai[j - 1]) / 2;
            double gb = std::atan2(bi[j + 1] * br[j - 1] - br[j + 1] * bi[j - 1], br[j + 1] * br[j - 1] + bi[j + 1] * bi[j - 1]) / 2;
            double g = 0.5 * (ga + gb);
            int k = y * W + x;
            gr[k]  = g;
            amp[k] = std::sqrt((ar[j] * ar[j] + ai[j] * ai[j]) * (br[j] * br[j] + bi[j] * bi[j]));
            dx[k]  = (std::fabs(g) > 1e-9) ? ix - dphi / g : 1e9;
        }
    }
    // 신뢰도 척도 ① : 응답 크기 하위 ampPercentile % 제외
    std::vector<double> tmp(amp);
    size_t nth = (size_t)std::min<double>((double)tmp.size() - 1, tmp.size() * params_.ampPercentile / 100.0);
    std::nth_element(tmp.begin(), tmp.begin() + nth, tmp.end());
    double ampTh = tmp[nth];

    // 2차 : 유효 화소로 가중 최소제곱  dx = t + s·(x − xc)/100
    double xc = r0.x + 0.5 * (W - 1);
    double S00 = 0, S01 = 0, S11 = 0, T0 = 0, T1 = 0, sg = 0, sgg = 0;
    std::vector<int> ok;
    ok.reserve(H * W);
    for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
        int k = y * W + x;
        double g = std::fabs(gr[k]);
        if (amp[k] <= ampTh) continue;                                                   // ① 약한 응답
        if (std::fabs(g - ideal) / ideal > params_.maxGradErr) continue;                // ② 비선형 위상
        if (std::fabs(dx[k] - ix) > params_.maxResidualFrac * params_.lambda) continue;   // 위상 감김
        double u = (r0.x + x - xc) / 100.0, w = amp[k];
        S00 += w; S01 += w * u; S11 += w * u * u; T0 += w * dx[k]; T1 += w * u * dx[k];
        sg += g / ideal; sgg += (g / ideal) * (g / ideal);
        ok.push_back(k);
    }
    m.nValid = (int)ok.size();
    m.validRatio = (double)ok.size() / (H * W);
    double det = S00 * S11 - S01 * S01;
    if (ok.size() < 50 || std::fabs(det) < 1e-12) return m;
    m.shift = (S11 * T0 - S01 * T1) / det;
    m.scale = (S00 * T1 - S01 * T0) / det;

    double ss = 0;
    for (size_t i = 0; i < ok.size(); ++i) {
        int k = ok[i], x = k % W;
        double e = dx[k] - (m.shift + m.scale * (r0.x + x - xc) / 100.0);
        ss += amp[k] * e * e;
    }
    m.fitStd = std::sqrt(ss / S00);
    double n = (double)ok.size();
    m.gradSpread = std::sqrt(std::max(0.0, sgg / n - (sg / n) * (sg / n)));
    m.valid = (m.validRatio >= params_.minValidRatio) && (m.fitStd <= params_.maxFitStd);
    return m;
}

Features PhaseAligner::measure(const cv::Mat& src) const {
    Features f;
    if (!hasRef_) return f;
    cv::Mat gray8 = toGray8(src), g32;
    if (gray8.size() != ref32_.size()) return f;
    gray8.convertTo(g32, CV_32F);

    // 거친 단계 : QR 띠 위상 상관 → 정수 이동 (QR 검출이 필요 없어 LED 번짐에도 안정적)
    cv::Mat curBand;
    g32(band_).copyTo(curBand);
    cv::Point2d sh = cv::phaseCorrelate(refBand_, curBand, hann_);
    f.coarseDx = sh.x; f.coarseDy = sh.y;
    int ix = (int)std::lround(sh.x), iy = (int)std::lround(sh.y);

    // 정밀 단계 : QR 별 위상 변위장 → (t, s)
    for (int q = 0; q < NUM_QR; ++q) {
        f.qr[q] = measureQR(g32, q, ix, iy);
        if (f.qr[q].valid) f.numValidQR++;
        f.f[2 * q]     = f.qr[q].shift;
        f.f[2 * q + 1] = f.qr[q].scale;
    }
    f.valid = (f.qr[0].nValid > 0 && f.qr[1].nValid > 0 && f.qr[2].nValid > 0);
    return f;
}

Estimate PhaseAligner::estimate(const Features& f) const {
    Estimate est;
    if (!f.valid || !calib_.ok) return est;
    // 일반화 최소제곱 : min (f − c − A u)ᵀ Σ⁻¹ (f − c − A u),  u = [Y, Θ]
    cv::Mat A(NUM_FEAT, 2, CV_64F), r(NUM_FEAT, 1, CV_64F);
    for (int k = 0; k < NUM_FEAT; ++k) {
        A.at<double>(k, 0) = calib_.a[k];
        A.at<double>(k, 1) = calib_.b[k];
        r.at<double>(k)    = f.f[k] - calib_.c[k];
    }
    cv::Mat Si = calib_.cov.inv(cv::DECOMP_SVD);
    cv::Mat Mi = cv::Mat(A.t() * Si * A).inv(cv::DECOMP_SVD);
    cv::Mat u = Mi * (A.t() * Si * r);
    est.y = u.at<double>(0);
    est.t = u.at<double>(1);
    est.sdY = std::sqrt(std::max(0.0, Mi.at<double>(0, 0)));
    est.sdT = std::sqrt(std::max(0.0, Mi.at<double>(1, 1)));
    est.valid = true;
    return est;
}

// =====================================================================================
// 학습·평가 도우미
// =====================================================================================
Calibration fitCalibration(const std::vector<Sample>& samples) {
    Calibration cal;
    cv::Mat X, F;
    for (size_t i = 0; i < samples.size(); ++i) {
        const Sample& s = samples[i];
        if (!s.train || !s.feat.valid) continue;
        cv::Mat xrow = (cv::Mat_<double>(1, 3) << s.trueY, s.trueT, 1.0);
        cv::Mat frow(1, NUM_FEAT, CV_64F);
        for (int k = 0; k < NUM_FEAT; ++k) frow.at<double>(k) = s.feat.f[k];
        X.push_back(xrow);
        F.push_back(frow);
    }
    if (X.rows < NUM_FEAT + 3) return cal;
    cv::Mat coef;                                            // 3 × NUM_FEAT
    cv::solve(X, F, coef, cv::DECOMP_SVD);
    for (int k = 0; k < NUM_FEAT; ++k) {
        cal.a[k] = coef.at<double>(0, k);
        cal.b[k] = coef.at<double>(1, k);
        cal.c[k] = coef.at<double>(2, k);
    }
    cv::Mat R = F - X * coef;                                // 학습 잔차 → 잡음 공분산
    cal.cov = (R.t() * R) / std::max(1, R.rows - 3);
    cal.cov += cv::Mat::eye(NUM_FEAT, NUM_FEAT, CV_64F) * 1e-10;
    cal.ok = true;
    return cal;
}

std::vector<Sample> buildSampleList(const std::string& dir, double maxY, double stepY, double maxT, double stepT) {
    std::vector<Sample> v;
    char buf[256];
    Sample c; c.path = dir + "/center.bmp"; c.sweep = 'C'; c.train = true;
    v.push_back(c);
    int nY = (stepY > 0) ? (int)std::lround(maxY / stepY) : 0;
    for (int i = -nY; i <= nY; ++i) {
        if (i == 0) continue;
        double y = i * stepY;
        std::snprintf(buf, sizeof(buf), "%s/Y/Y%+.2f.bmp", dir.c_str(), y);
        Sample s; s.path = buf; s.sweep = 'Y'; s.trueY = y; s.train = ((i + nY) % 2 == 0);
        v.push_back(s);
    }
    int nT = (stepT > 0) ? (int)std::lround(maxT / stepT) : 0;
    for (int i = -nT; i <= nT; ++i) {
        if (i == 0) continue;
        double t = i * stepT;
        std::snprintf(buf, sizeof(buf), "%s/Theta/T%+.2f.bmp", dir.c_str(), t);
        Sample s; s.path = buf; s.sweep = 'T'; s.trueT = t; s.train = ((i + nT) % 2 == 0);
        v.push_back(s);
    }
    return v;
}

} // namespace pa
