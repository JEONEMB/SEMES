// =====================================================================================
// report_media.cpp — 보고서용 설명 영상 프레임 생성 (실제 사진 사용)
//
//  V1 parallax   : 플레이트 QR 과 하단 문 홀을 Y 이동 / Θ 이동 사진에서 깜빡여 비교 (시차)
//  V2 phasecorr  : Θ 스윕 사진을 차례로 넣으며 위상 상관 봉우리가 움직이는 모습
//  V3 subpixel   : QR 모서리를 0~1 px 부분 이동 → 화소 밝기가 조금씩 변하는 모습과 측정값
//  V4 gaussnewton: 가우스–뉴턴 반복으로 어긋남이 줄어드는 모습
//  프레임은 outDir/<이름>/0000.png ... 로 저장 → ffmpeg 로 mp4 (tools/encode_videos.sh)
//  화면 글자는 OpenCV 기본 글꼴(영문)만 가능 → 한국어 설명은 보고서 본문에 둠
// =====================================================================================
#include "report_media.hpp"
#include "../yt/m1_phase_ecc.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <iostream>

namespace yt {
namespace media {

namespace {
const cv::Scalar kRed(60, 60, 230), kYel(0, 220, 255), kWhite(255, 255, 255), kGreen(80, 200, 80), kCyan(230, 200, 40);

cv::Mat to8(const cv::Mat& m32) { cv::Mat o; m32.convertTo(o, CV_8U); cv::cvtColor(o, o, cv::COLOR_GRAY2BGR); return o; }
cv::Mat crop(const cv::Mat& img, cv::Point c, int hw) { return img(cv::Rect(c.x - hw, c.y - hw, 2 * hw, 2 * hw)).clone(); }
void text(cv::Mat& m, const std::string& s, cv::Point p, double sc = 0.8, cv::Scalar c = kWhite, int th = 2) {
    cv::putText(m, s, p, cv::FONT_HERSHEY_SIMPLEX, sc, cv::Scalar(0, 0, 0), th + 3, cv::LINE_AA);
    cv::putText(m, s, p, cv::FONT_HERSHEY_SIMPLEX, sc, c, th, cv::LINE_AA);
}
class Frames {
public:
    Frames(const std::string& dir) : dir_(dir) { ensureDir(dir); }
    void add(const cv::Mat& f, int repeat = 1) {
        for (int i = 0; i < repeat; ++i) cv::imwrite(dir_ + fmt("/%04d.png", n_++), f);
    }
    int count() const { return n_; }
private:
    std::string dir_;
    int n_ = 0;
};
cv::Mat heat(const cv::Mat& m) {
    cv::Mat n, c;
    cv::normalize(m, n, 0, 255, cv::NORM_MINMAX);
    n.convertTo(n, CV_8U);
    cv::applyColorMap(n, c, cv::COLORMAP_INFERNO);
    return c;
}
// 위상 상관 면 (가운데 = 이동 0), 실수
cv::Mat corrSurface(const cv::Mat& a, const cv::Mat& b) {
    cv::Mat win, A, B, fa, fb, R, r;
    cv::createHanningWindow(win, a.size(), CV_32F);
    A = a.mul(win); B = b.mul(win);
    cv::dft(A, fa, cv::DFT_COMPLEX_OUTPUT);
    cv::dft(B, fb, cv::DFT_COMPLEX_OUTPUT);
    cv::mulSpectrums(fb, fa, R, 0, true);
    std::vector<cv::Mat> ch;
    cv::split(R, ch);
    cv::Mat mag;
    cv::magnitude(ch[0], ch[1], mag);
    mag += 1e-9f;
    ch[0] /= mag; ch[1] /= mag;
    cv::merge(ch, R);
    cv::idft(R, r, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
    // fftshift
    const int cx = r.cols / 2, cy = r.rows / 2;
    cv::Mat o = r.clone();
    r(cv::Rect(0, 0, cx, cy)).copyTo(o(cv::Rect(cx, cy, cx, cy)));
    r(cv::Rect(cx, cy, cx, cy)).copyTo(o(cv::Rect(0, 0, cx, cy)));
    r(cv::Rect(cx, 0, cx, cy)).copyTo(o(cv::Rect(0, cy, cx, cy)));
    r(cv::Rect(0, cy, cx, cy)).copyTo(o(cv::Rect(cx, 0, cx, cy)));
    return o;
}
} // namespace

bool makeAll(const std::vector<Sample>& ds, const std::string& outDir) {
    const Sample* r = findStem(ds, "center");
    if (!r) return false;
    const cv::Mat ref = loadGray32(r->path);
    auto img = [&](const char* st) { const Sample* s = findStem(ds, st); return s ? loadGray32(s->path) : cv::Mat(); };
    const cv::Point qr(regions()[1].cx, regions()[1].cy);           // QR_C
    const cv::Point hole(1010, 1677);                                 // 하단 문 홀 W3

    // ---------------- V1 시차 : 같은 플레이트 이동(≈14 px)을 만드는 Y +1.00 mm 와 Θ +0.18°
    {
        Frames F(outDir + "/v1_parallax");
        const cv::Mat y1 = img("Y+1.00"), t1 = img("T+0.18");
        const int hw = 70, sc = 3;   // 140 px 를 3 배로
        auto panel = [&](const cv::Mat& src, cv::Point c, const std::string& lab) {
            cv::Mat p;
            cv::resize(to8(crop(src, c, hw)), p, cv::Size(), sc, sc, cv::INTER_NEAREST);
            const int m = hw * sc;
            cv::line(p, {m, 0}, {m, p.rows}, kRed, 1);
            cv::line(p, {0, m}, {p.cols, m}, kRed, 1);
            text(p, lab, {10, 30}, 0.75);
            return p;
        };
        for (int cyc = 0; cyc < 6; ++cyc)
            for (int ph = 0; ph < 2; ++ph) {
                const cv::Mat& a = ph ? y1 : ref;
                const cv::Mat& b = ph ? t1 : ref;
                const std::string s = ph ? "MOVED" : "REFERENCE";
                cv::Mat top, bot, all;
                cv::hconcat(panel(a, qr, "PLATE QR  (Y +1.00 mm)"), panel(a, hole, "DOOR HOLE (Y +1.00 mm)"), top);
                cv::hconcat(panel(b, qr, "PLATE QR  (THETA +0.18 deg)"), panel(b, hole, "DOOR HOLE (THETA +0.18 deg)"), bot);
                cv::vconcat(top, bot, all);
                cv::line(all, {0, top.rows}, {all.cols, top.rows}, kWhite, 3);
                cv::line(all, {top.cols / 2, 0}, {top.cols / 2, all.rows}, kWhite, 3);
                text(all, s, {all.cols / 2 - 90, all.rows - 20}, 1.0, ph ? kYel : kWhite);
                F.add(all, 8);
            }
        std::cout << "  v1 " << F.count() << " frames\n";
    }

    // ---------------- V2 위상 상관 : Θ +0.01 ~ +0.30
    {
        Frames F(outDir + "/v2_phasecorr");
        const int hw = 128;
        const cv::Mat a = crop(ref, qr, hw);
        for (int i = 1; i <= 30; ++i) {
            const cv::Mat cur = img(fmt("T+%.2f", 0.01 * i).c_str());
            if (cur.empty()) continue;
            const cv::Mat b = crop(cur, qr, hw);
            const cv::Mat S = corrSurface(a, b);
            cv::Point pk;
            double mx;
            cv::minMaxLoc(S, nullptr, &mx, nullptr, &pk);
            const cv::Point2d sub = m1::phaseShift(a, b);
            // 봉우리 주변 ±40 px 만 확대
            const int Z = 16;
            cv::Mat zoom = cv::max(S(cv::Rect(hw - Z, hw - Z, 2 * Z, 2 * Z)), 0.0);
            cv::sqrt(zoom, zoom);   // 약한 봉우리도 보이게
            cv::Mat hz;
            cv::resize(heat(zoom), hz, cv::Size(512, 512), 0, 0, cv::INTER_NEAREST);
            const double k = 512.0 / (2 * Z);
            cv::line(hz, {256, 0}, {256, 512}, kWhite, 1);
            cv::line(hz, {0, 256}, {512, 256}, kWhite, 1);
            cv::circle(hz, cv::Point(int((pk.x - (hw - Z) + 0.5) * k), int((pk.y - (hw - Z) + 0.5) * k)), 14, kCyan, 2);
            text(hz, "CORRELATION  (+-16 px, center = no move)", {10, 30}, 0.6);
            cv::Mat pa, pb;
            cv::resize(to8(a), pa, cv::Size(512, 512));
            cv::resize(to8(b), pb, cv::Size(512, 512));
            text(pa, "REFERENCE", {10, 30}, 0.7);
            text(pb, fmt("THETA +%.2f deg", 0.01 * i), {10, 30}, 0.7, kYel);
            cv::Mat row;
            cv::hconcat(std::vector<cv::Mat>{pa, pb, hz}, row);
            cv::Mat band(70, row.cols, CV_8UC3, cv::Scalar(30, 30, 30));
            text(band, fmt("peak at %+d px  ->  sub-pixel shift %+.3f px", pk.x - hw, sub.x), {20, 48}, 1.0, kCyan);
            cv::vconcat(row, band, row);
            F.add(row, 4);
        }
        std::cout << "  v2 " << F.count() << " frames\n";
    }

    // ---------------- V3 서브픽셀 : QR 모서리 부분 이동
    {
        Frames F(outDir + "/v3_subpixel");
        const cv::Point e(qr.x - 64, qr.y);                            // QR 왼쪽 모듈 경계 근처
        const cv::Mat big = crop(ref, e, 64);
        const int hw = 8, sc = 36;                                      // 16 × 16 화소를 36 배
        for (int i = 0; i <= 40; ++i) {
            const double dx = (i <= 20 ? i : 40 - i) * 0.05;
            const cv::Mat sh = fourierShift(big, dx, 0.0);
            const cv::Mat p = sh(cv::Rect(64 - hw, 64 - hw, 2 * hw, 2 * hw));
            cv::Mat px;
            cv::resize(to8(p), px, cv::Size(), sc, sc, cv::INTER_NEAREST);
            for (int k = 0; k <= 2 * hw; ++k) {
                cv::line(px, {k * sc, 0}, {k * sc, px.rows}, cv::Scalar(90, 90, 90), 1);
                cv::line(px, {0, k * sc}, {px.cols, k * sc}, cv::Scalar(90, 90, 90), 1);
            }
            const int rowY = hw;                                        // 가운데 줄
            cv::rectangle(px, cv::Rect(0, rowY * sc, px.cols, sc), kYel, 2);
            // 밝기 그래프 : 가운데 줄 16 화소
            cv::Mat g(px.rows, 560, CV_8UC3, cv::Scalar(25, 25, 25));
            const cv::Mat b0 = big(cv::Rect(64 - hw, 64, 2 * hw, 1));
            const cv::Mat bi = sh(cv::Rect(64 - hw, 64, 2 * hw, 1));
            auto Y = [&](float v) { return int(g.rows - 60 - v / 255.0 * (g.rows - 120)); };
            auto X = [&](int k) { return 40 + k * 30; };
            for (int k = 0; k < 2 * hw; ++k) {
                cv::circle(g, {X(k), Y(b0.at<float>(0, k))}, 5, cv::Scalar(140, 140, 140), -1);
                cv::circle(g, {X(k), Y(bi.at<float>(0, k))}, 6, kYel, -1);
                if (k) cv::line(g, {X(k - 1), Y(bi.at<float>(0, k - 1))}, {X(k), Y(bi.at<float>(0, k))}, kYel, 2);
            }
            const Region loc{"e", 0, 64, 64, 40};
            const m1::Shift m = m1::measureRegion(big, sh, loc, cv::Point2d(0, 0), false);
            text(g, "brightness along yellow row", {20, 30}, 0.6);
            text(g, "grey = before, yellow = after", {20, 60}, 0.6, cv::Scalar(200, 200, 200));
            text(g, fmt("true move   %.2f px", dx), {20, g.rows - 70}, 0.8, kWhite);
            text(g, fmt("measured    %.3f px", m.d.x), {20, g.rows - 30}, 0.8, kCyan);
            cv::Mat row;
            cv::hconcat(px, g, row);
            F.add(row, i == 0 || i == 20 ? 12 : 3);
        }
        std::cout << "  v3 " << F.count() << " frames\n";
    }

    // ---------------- V4 가우스–뉴턴 : 평행이동 2 변수, 일부러 틀린 시작값에서 출발
    {
        Frames F(outDir + "/v4_gaussnewton");
        const cv::Mat cur = img("T+0.05");
        const int hw = 90;
        const cv::Mat T0 = crop(ref, qr, hw);
        cv::Point2d p(0.0, 0.0);                                         // 시작값 : 움직이지 않았다고 가정
        const cv::Point2d truth = m1::measureRegion(ref, cur, regions()[1], m1::phaseShift(T0, crop(cur, qr, hw)), false).d;
        // 거친 → 고운 : 처음 4 걸음은 흐리게(σ 3) 해서 멀리서도 방향을 찾고, 그다음 원본으로 정밀하게
        cv::Mat curBlur;
        cv::GaussianBlur(cur, curBlur, cv::Size(0, 0), 3.0);
        for (int it = 0; it <= 10; ++it) {
            const bool coarse = it < 4;
            cv::Mat T = T0.clone();
            if (coarse) cv::GaussianBlur(T0, T, cv::Size(0, 0), 3.0);
            const cv::Mat& src = coarse ? curBlur : cur;
            cv::Mat gx, gy;
            cv::Sobel(T, gx, CV_32F, 1, 0, 3, 1.0 / 8);
            cv::Sobel(T, gy, CV_32F, 0, 1, 3, 1.0 / 8);
            double H00 = 0, H01 = 0, H11 = 0;
            for (int y = 0; y < T.rows; ++y)
                for (int x = 0; x < T.cols; ++x) {
                    const double a = gx.at<float>(y, x), b = gy.at<float>(y, x);
                    H00 += a * a; H01 += a * b; H11 += b * b;
                }
            const cv::Matx22d Hinv = cv::Matx22d(H00, H01, H01, H11).inv();
            // 현재 추정 p 로 현재 영상을 되돌려 기준과 비교
            cv::Mat mapx(T.size(), CV_32F), mapy(T.size(), CV_32F);
            for (int y = 0; y < T.rows; ++y)
                for (int x = 0; x < T.cols; ++x) {
                    mapx.at<float>(y, x) = float(qr.x - hw + x + p.x);
                    mapy.at<float>(y, x) = float(qr.y - hw + y + p.y);
                }
            cv::Mat W;
            cv::remap(src, W, mapx, mapy, cv::INTER_CUBIC);
            const cv::Mat E = W - T;
            const double rms = std::sqrt(cv::mean(E.mul(E))[0]);
            cv::Mat ed = cv::abs(E);
            cv::Mat he;
            ed.convertTo(he, CV_8U, 2.0);
            cv::applyColorMap(he, he, cv::COLORMAP_INFERNO);
            cv::Mat pw, pe;
            cv::resize(to8(W), pw, cv::Size(450, 450));
            cv::resize(he, pe, cv::Size(450, 450));
            text(pw, "MOVED IMAGE, UNDONE BY GUESS", {10, 30}, 0.6);
            text(pe, "DIFFERENCE (bright = mismatch)", {10, 30}, 0.6);
            cv::Mat row;
            cv::hconcat(pw, pe, row);
            cv::Mat band(110, row.cols, CV_8UC3, cv::Scalar(30, 30, 30));
            text(band, fmt("step %d %s  guess dx = %+.3f  dy = %+.3f px", it, coarse ? "(blurred)" : "(sharp)  ", p.x, p.y), {20, 42}, 0.85, kYel);
            text(band, fmt("mismatch %.2f   (M1 full result dx = %+.3f)", rms, truth.x), {20, 90}, 0.85, kCyan);
            cv::vconcat(row, band, row);
            F.add(row, it == 0 ? 16 : 10);
            // 가우스–뉴턴 한 걸음 : Δp = H⁻¹ Σ ∇T · (T − W)
            double b0 = 0, b1 = 0;
            for (int y = 0; y < T.rows; ++y)
                for (int x = 0; x < T.cols; ++x) {
                    const double e = W.at<float>(y, x) - T.at<float>(y, x);
                    b0 += gx.at<float>(y, x) * e; b1 += gy.at<float>(y, x) * e;
                }
            const cv::Vec2d d = Hinv * cv::Vec2d(b0, b1);
            // W(x + p + Δ) ≈ W + ∇T·Δ = T  →  Δ = −H⁻¹ Σ ∇T·(W − T)
            p -= cv::Point2d(d[0], d[1]);
        }
        std::cout << "  v4 " << F.count() << " frames\n";
    }
    return true;
}

} // namespace media
} // namespace yt
