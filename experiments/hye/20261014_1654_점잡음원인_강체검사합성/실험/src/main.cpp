// 특징점 잡음의 원인 (원본: analysis_v8/r1_rigid.py, r4_struct.py, r5_phase.py, r6_pairdiff.py, r8_pairdiff2.py,
// r11_noise_model.py, r3_synth.py, r12_synth2.py, r13_crlb.py. 추적은 analysis_v6/common.py, t11_qr_features.py, analysis_v7/e4_track.py)
#include <hye/dataset.hpp>
#include <hye/plot.hpp>
#include <hye/report.hpp>
#include <opencv2/calib3d.hpp>
#include <omp.h>
#include <array>
#include <complex>
#include <iostream>
#include <map>
#include <random>
#include <tuple>
#include "track.hpp"

using namespace hye;

namespace {

struct Img {
    std::string name;  // T-1.00.bmp 처럼 원래 NAMES 이름
    char axis;         // 'C', 'T', 'Y'
    double val;
    std::string ses;
    const Shot* shot;
};

cv::Mat read_f32(const Shot& s) {
    cv::Mat g = read_gray(s), f;
    g.convertTo(f, CV_32F);
    return f;
}
cv::Mat read_f64(const Shot& s) {
    cv::Mat g = read_gray(s), f;
    g.convertTo(f, CV_64F);
    return f;
}

double std0(const std::vector<double>& v) {
    double m = mean(v), s = 0;
    for (double x : v) s += (x - m) * (x - m);
    return std::sqrt(s / v.size());
}
double median(std::vector<double> v) { return percentile(std::move(v), 0.5); }
std::pair<double, double> polyfit1(const double* x, const double* y, size_t n) {
    double mx = 0, my = 0;
    for (size_t i = 0; i < n; ++i) { mx += x[i]; my += y[i]; }
    mx /= n; my /= n;
    double sxy = 0, sxx = 0;
    for (size_t i = 0; i < n; ++i) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) * (x[i] - mx); }
    double k = sxy / sxx;
    return {k, my - k * mx};
}
// np.gradient (가운데 차분, 가장자리 한쪽 차분)
cv::Mat gradient(const cv::Mat& a, int axis) {
    cv::Mat g(a.size(), CV_64F);
    for (int r = 0; r < a.rows; ++r)
        for (int c = 0; c < a.cols; ++c) {
            if (axis == 1) {
                int n = a.cols;
                g.at<double>(r, c) = c == 0 ? a.at<double>(r, 1) - a.at<double>(r, 0)
                                   : c == n - 1 ? a.at<double>(r, n - 1) - a.at<double>(r, n - 2)
                                                : (a.at<double>(r, c + 1) - a.at<double>(r, c - 1)) / 2;
            } else {
                int n = a.rows;
                g.at<double>(r, c) = r == 0 ? a.at<double>(1, c) - a.at<double>(0, c)
                                   : r == n - 1 ? a.at<double>(n - 1, c) - a.at<double>(n - 2, c)
                                                : (a.at<double>(r + 1, c) - a.at<double>(r - 1, c)) / 2;
            }
        }
    return g;
}
cv::Mat sobel_mag(const cv::Mat& a, double sg) {  // hypot(Sobel x, Sobel y)/8 of GaussianBlur(a, sg)
    cv::Mat gm, sx, sy, m;
    cv::GaussianBlur(a, gm, cv::Size(0, 0), sg);
    cv::Sobel(gm, sx, CV_64F, 1, 0);
    cv::Sobel(gm, sy, CV_64F, 0, 1);
    cv::magnitude(sx, sy, m);
    return m / 8;
}
std::vector<double> masked(const cv::Mat& a, const cv::Mat& m) {
    std::vector<double> v;
    for (int r = 0; r < a.rows; ++r)
        for (int c = 0; c < a.cols; ++c)
            if (m.at<uchar>(r, c)) v.push_back(a.at<double>(r, c));
    return v;
}
std::vector<double> flat(const cv::Mat& a) {
    cv::Mat c = a.isContinuous() ? a : a.clone();
    return std::vector<double>((const double*)c.data, (const double*)c.data + c.total());
}

// ---- 강체 검사: 사진마다 QR 점에 호모그래피(8 변수) 최소제곱 ----
void H_apply(const double* h, double x, double y, double& X, double& Y) {
    double w = h[6] * x + h[7] * y + 1;
    X = (h[0] * x + h[1] * y + h[2]) / w;
    Y = (h[3] * x + h[4] * y + h[5]) / w;
}
// P: 기준 점(정규화), Q: 현재 점(정규화). 반환: 모델 - 측정 (정규화 단위)
std::vector<cv::Point2d> fit_h(const std::vector<cv::Point2d>& P, const std::vector<cv::Point2d>& Q) {
    std::vector<cv::Point2f> pf(P.begin(), P.end()), qf(Q.begin(), Q.end());
    cv::Mat H0 = cv::findHomography(pf, qf, 0);
    double h[8];
    for (int i = 0; i < 8; ++i) h[i] = H0.at<double>(i / 3, i % 3) / H0.at<double>(2, 2);
    auto cost = [&](const double* hh) {
        double s = 0;
        for (size_t i = 0; i < P.size(); ++i) {
            double X, Y;
            H_apply(hh, P[i].x, P[i].y, X, Y);
            s += (X - Q[i].x) * (X - Q[i].x) + (Y - Q[i].y) * (Y - Q[i].y);
        }
        return s;
    };
    double lam = 1e-3, c0 = cost(h);
    for (int it = 0; it < 200; ++it) {
        cv::Mat A = cv::Mat::zeros(8, 8, CV_64F), g = cv::Mat::zeros(8, 1, CV_64F);
        for (size_t i = 0; i < P.size(); ++i) {
            double x = P[i].x, y = P[i].y, X, Y;
            H_apply(h, x, y, X, Y);
            double w = h[6] * x + h[7] * y + 1;
            double Jx[8] = {x / w, y / w, 1 / w, 0, 0, 0, -X * x / w, -X * y / w};
            double Jy[8] = {0, 0, 0, x / w, y / w, 1 / w, -Y * x / w, -Y * y / w};
            double rx = X - Q[i].x, ry = Y - Q[i].y;
            for (int p = 0; p < 8; ++p) {
                g.at<double>(p) += Jx[p] * rx + Jy[p] * ry;
                for (int q = 0; q < 8; ++q) A.at<double>(p, q) += Jx[p] * Jx[q] + Jy[p] * Jy[q];
            }
        }
        bool done = false;
        for (int tries = 0; tries < 30; ++tries) {
            cv::Mat Ad = A.clone(), d;
            for (int p = 0; p < 8; ++p) Ad.at<double>(p, p) *= (1 + lam);
            cv::solve(Ad, -g, d, cv::DECOMP_CHOLESKY);
            double hn[8];
            for (int p = 0; p < 8; ++p) hn[p] = h[p] + d.at<double>(p);
            double c1 = cost(hn);
            if (c1 <= c0) {
                double step = cv::norm(d);
                std::copy(hn, hn + 8, h);
                lam = std::max(lam / 10, 1e-12);
                if (c0 - c1 <= 1e-15 * c0 || step < 1e-13) done = true;
                c0 = c1;
                break;
            }
            lam *= 10;
            if (tries == 29) done = true;
        }
        if (done) break;
    }
    std::vector<cv::Point2d> r(P.size());
    for (size_t i = 0; i < P.size(); ++i) {
        double X, Y;
        H_apply(h, P[i].x, P[i].y, X, Y);
        r[i] = {X - Q[i].x, Y - Q[i].y};
    }
    return r;
}

const std::vector<std::string> kSES = {"Y1+", "Y1-", "Y2+", "Y2-", "T+", "T-"};

// 구간마다 값 v 에 대한 2차를 빼서 무작위 부분만 남김. 반환: center 를 뺀 280 행
std::vector<std::vector<cv::Point2d>> session_detrend(const std::vector<std::vector<cv::Point2d>>& R, const std::vector<Img>& im) {
    size_t P = R[0].size();
    std::vector<std::vector<cv::Point2d>> out(R.size(), std::vector<cv::Point2d>(P));
    for (auto& s : kSES) {
        std::vector<size_t> idx;
        for (size_t i = 0; i < im.size(); ++i) if (im[i].ses == s) idx.push_back(i);
        cv::Mat A((int)idx.size(), 3, CV_64F), X((int)idx.size(), (int)P * 2, CV_64F);
        for (int r = 0; r < (int)idx.size(); ++r) {
            double v = im[idx[r]].val;
            A.at<double>(r, 0) = 1; A.at<double>(r, 1) = v; A.at<double>(r, 2) = v * v;
            for (size_t j = 0; j < P; ++j) { X.at<double>(r, (int)j * 2) = R[idx[r]][j].x; X.at<double>(r, (int)j * 2 + 1) = R[idx[r]][j].y; }
        }
        cv::Mat C;
        cv::solve(A, X, C, cv::DECOMP_SVD);
        cv::Mat E = X - A * C;
        for (int r = 0; r < (int)idx.size(); ++r)
            for (size_t j = 0; j < P; ++j) out[idx[r]][j] = {E.at<double>(r, (int)j * 2), E.at<double>(r, (int)j * 2 + 1)};
    }
    std::vector<std::vector<cv::Point2d>> o;
    for (size_t i = 0; i < im.size(); ++i) if (im[i].axis != 'C') o.push_back(out[i]);
    return o;
}
// 점마다 사진들에 걸친 표준편차(ddof 0)의 중앙값
std::pair<double, double> point_jitter(const std::vector<std::vector<cv::Point2d>>& R, std::vector<double>* sx_out = nullptr,
                                       std::vector<double>* sy_out = nullptr) {
    size_t P = R[0].size();
    std::vector<double> sx(P), sy(P);
    for (size_t j = 0; j < P; ++j) {
        std::vector<double> a, b;
        for (auto& r : R) { a.push_back(r[j].x); b.push_back(r[j].y); }
        sx[j] = std0(a); sy[j] = std0(b);
    }
    if (sx_out) *sx_out = sx;
    if (sy_out) *sy_out = sy;
    return {median(sx), median(sy)};
}

// ---- 합성: center 를 푸리에로 정확히 옮긴 사진 (원본 r3_synth.render) ----
constexpr int RX0 = 1150, RX1 = 2670, RY0 = 760, RY1 = 1380;
constexpr double SN = 0.65, B0 = 0.04;
std::vector<double> fftfreq(int n) {
    std::vector<double> f(n);
    for (int k = 0; k < n; ++k) f[k] = (k < (n + 1) / 2 ? k : k - n) / (double)n;
    return f;
}
// reg(float64) 를 (tx, ty) 만큼 옮기고 가우스 흐림 sx2, sy2 를 넣는다. 결과 = ifft 의 실수부
cv::Mat fourier_shift(const cv::Mat& reg, double tx, double ty, double sx2, double sy2) {
    cv::Mat F;
    cv::dft(reg, F, cv::DFT_COMPLEX_OUTPUT);
    auto fx = fftfreq(reg.cols), fy = fftfreq(reg.rows);
    for (int r = 0; r < F.rows; ++r) {
        auto* p = F.ptr<std::complex<double>>(r);
        for (int c = 0; c < F.cols; ++c) {
            double ph = -2 * CV_PI * (fx[c] * tx + fy[r] * ty);
            double g = std::exp(-2 * CV_PI * CV_PI * (sx2 * fx[c] * fx[c] + sy2 * fy[r] * fy[r]));
            p[c] *= std::complex<double>(std::cos(ph), std::sin(ph)) * g;
        }
    }
    cv::Mat inv, ch[2];
    cv::dft(F, inv, cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_COMPLEX_OUTPUT);
    cv::split(inv, ch);
    return ch[0];
}

struct NoiseModel {
    std::vector<double> x, y;  // 밝기 중심, 한 장 잡음 sd (nan 은 뺌)
    double at(double v) const {
        if (v <= x.front()) return y.front();
        if (v >= x.back()) return y.back();
        size_t i = std::upper_bound(x.begin(), x.end(), v) - x.begin();
        double t = (v - x[i - 1]) / (x[i] - x[i - 1]);
        return y[i - 1] + t * (y[i] - y[i - 1]);
    }
};

}  // namespace

static int run();
int main() {
    try {
        return run();
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}

static int run() {
    begin_results();
    omp_set_num_threads(4);
    auto shots = load_1014();
    // 원래 NAMES 순서: T 오름차순(center 는 T 0 자리), 그다음 Y 오름차순
    std::vector<Img> im;
    for (auto& s : shots) {
        Img a;
        a.shot = &s;
        a.axis = s.condition == "center" ? 'C' : s.condition == "T" ? 'T' : 'Y';
        a.val = a.axis == 'T' ? s.t : a.axis == 'Y' ? s.y : 0;
        a.name = s.rel.substr(s.rel.find('/') == std::string::npos ? 0 : s.rel.find('/') + 1);
        if (a.axis == 'C') a.ses = "C";
        else if (a.axis == 'T') a.ses = a.val > 0 ? "T+" : "T-";
        else a.ses = std::string(((long long)std::llround(std::abs(a.val / 0.05)) % 2 == 1) ? "Y2" : "Y1") + (a.val > 0 ? "+" : "-");
        im.push_back(a);
    }
    std::stable_sort(im.begin(), im.end(), [](const Img& a, const Img& b) {
        int ka = a.axis == 'Y', kb = b.axis == 'Y';
        return ka != kb ? ka < kb : a.val < b.val;
    });
    const int NI = (int)im.size();
    int ic = 0;
    for (int i = 0; i < NI; ++i) if (im[i].axis == 'C') ic = i;
    auto idx_of = [&](const std::string& n) {
        for (int i = 0; i < NI; ++i) if (im[i].name == n) return i;
        throw std::runtime_error("no image " + n);
    };

    // ---- 특징점과 추적 ----
    const cv::Mat cen = read_f32(*im[ic].shot);
    auto F = v6::pick_qr(cen, 8);
    int cnt[4] = {};
    for (auto& f : F) ++cnt[f.region];
    std::vector<int> qr;
    for (int k = 0; k < (int)F.size(); ++k) if (F[k].region < 3) qr.push_back(k);
    std::cout << "features " << F.size() << " QR " << qr.size() << "\n";
    v6::Pipe p17{8, 1.0}, p33{16, 2.0};
    p17.setup(cen, F);
    p33.setup(cen, F);
    auto big = v6::big_templates(cen, F);
    std::vector<std::vector<cv::Point2d>> D17(NI), D33(NI);
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < NI; ++i) {
        cv::Mat img = read_f32(*im[i].shot);
        auto c0 = v6::coarse(img, big, F);
        D17[i] = p17.run(img, c0, F);
        D33[i] = p33.run(img, c0, F);
    }
    std::cout << "tracked " << NI << "\n";
    auto qr_only = [&](const std::vector<std::vector<cv::Point2d>>& D) {
        std::vector<std::vector<cv::Point2d>> o(D.size());
        for (size_t i = 0; i < D.size(); ++i) for (int k : qr) o[i].push_back(D[i][k]);
        return o;
    };
    auto Q17 = qr_only(D17), Q33 = qr_only(D33);
    const int NP = (int)qr.size();
    std::vector<cv::Point2d> PQ;
    for (int k : qr) PQ.push_back({(double)F[k].x, (double)F[k].y});

    // ---- r1 강체 검사 ----
    auto rigid = [&](const std::vector<std::vector<cv::Point2d>>& Q) {
        cv::Point2d c(0, 0);
        for (auto& p : PQ) c += p;
        c /= (double)NP;
        std::vector<cv::Point2d> Pn;
        for (auto& p : PQ) Pn.push_back((p - c) / 500.0);
        std::vector<std::vector<cv::Point2d>> R(NI);
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < NI; ++i) {
            std::vector<cv::Point2d> Qn;
            for (int j = 0; j < NP; ++j) Qn.push_back((PQ[j] + Q[i][j] - c) / 500.0);
            R[i] = fit_h(Pn, Qn);
            for (auto& r : R[i]) r *= 500.0;
        }
        return R;
    };
    auto R17 = rigid(Q17), R33 = rigid(Q33);
    auto drop_c = [&](const std::vector<std::vector<cv::Point2d>>& R) {
        std::vector<std::vector<cv::Point2d>> o;
        for (int i = 0; i < NI; ++i) if (i != ic) o.push_back(R[i]);
        return o;
    };
    auto Rr17 = session_detrend(R17, im), Rr33 = session_detrend(R33, im);
    std::vector<double> ptsx17, ptsy17;
    auto raw17 = point_jitter(drop_c(R17)), rnd17 = point_jitter(Rr17, &ptsx17, &ptsy17);
    auto raw33 = point_jitter(drop_c(R33)), rnd33 = point_jitter(Rr33);
    auto sys_rms = [&](const std::vector<std::vector<cv::Point2d>>& R) {
        auto X = drop_c(R);
        double sx = 0, sy = 0;
        for (int j = 0; j < NP; ++j) {
            double mx = 0, my = 0;
            for (auto& r : X) { mx += r[j].x; my += r[j].y; }
            mx /= X.size(); my /= X.size();
            sx += mx * mx; sy += my * my;
        }
        return cv::Point2d(std::sqrt(sx / NP), std::sqrt(sy / NP));
    };
    auto sys17 = sys_rms(R17), sys33 = sys_rms(R33);
    Table t_rigid{{"측정", "구분", "점 흔들림 x (px)", "점 흔들림 y (px)"}};
    t_rigid.add({"17px σ1", "강체 잔차 그대로", fmt(raw17.first, 4), fmt(raw17.second, 4)});
    t_rigid.add({"17px σ1", "강체 잔차 중 무작위 부분", fmt(rnd17.first, 4), fmt(rnd17.second, 4)});
    t_rigid.add({"33px σ2", "강체 잔차 그대로", fmt(raw33.first, 4), fmt(raw33.second, 4)});
    t_rigid.add({"33px σ2", "강체 잔차 중 무작위 부분", fmt(rnd33.first, 4), fmt(rnd33.second, 4)});
    t_rigid.csv(out("r1_rigid.csv"));
    std::cout << "rigid 17 " << rnd17.first << " " << rnd17.second << " 33 " << rnd33.first << " " << rnd33.second << "\n";

    // ---- r4 (a) 잡음 공간 구조 (33px 무작위 부분) ----
    Table t_corr{{"성분", "세로 거리 (px)", "가로 거리", "평균 상관", "점 쌍 수"}};
    std::map<std::string, double> corr_v;
    std::vector<double> band_ratio(2);
    for (int c = 0; c < 2; ++c) {
        int N = (int)Rr33.size();
        cv::Mat Z(N, NP, CV_64F);
        for (int i = 0; i < N; ++i) for (int j = 0; j < NP; ++j) Z.at<double>(i, j) = c ? Rr33[i][j].y : Rr33[i][j].x;
        for (int j = 0; j < NP; ++j) {
            cv::Mat col = Z.col(j);
            cv::Scalar m, s;
            cv::meanStdDev(col, m, s);
            col = (col - m[0]) / s[0];
        }
        cv::Mat C = Z.t() * Z / N;
        for (auto [lo, hi] : std::vector<std::pair<int, int>>{{0, 5}, {5, 20}, {20, 60}, {60, 250}})
            for (auto [xlo, xhi, xn] : std::vector<std::tuple<int, int, std::string>>{{0, 60, "가까움(<60)"}, {200, 1000, "멂(>200, 다른 QR)"}}) {
                double s = 0;
                int n = 0;
                for (int a = 0; a < NP; ++a)
                    for (int b = a + 1; b < NP; ++b) {
                        double dx = std::abs(PQ[a].x - PQ[b].x), dy = std::abs(PQ[a].y - PQ[b].y);
                        if (dy >= lo && dy < hi && dx >= xlo && dx < xhi) { s += C.at<double>(a, b); ++n; }
                    }
                if (n > 20) {
                    t_corr.add({c ? "y" : "x", std::to_string(lo) + "~" + std::to_string(hi), xn, fmts(s / n, 3), std::to_string(n)});
                    corr_v[std::string(c ? "y" : "x") + std::to_string(lo) + (xlo ? "far" : "near")] = s / n;
                }
            }
        // 같은 줄(10px) 묶음 평균 분산 / 독립 기대
        std::map<int, std::vector<int>> rows;
        for (int j = 0; j < NP; ++j) rows[(int)std::floor(PQ[j].y / 10)].push_back(j);
        double zz = 0;
        for (auto& r : Rr33) for (int j = 0; j < NP; ++j) { double v = c ? r[j].y : r[j].x; zz += v * v; }
        zz /= (double)Rr33.size() * NP;
        std::vector<double> ratio;
        for (auto& [k, js] : rows) {
            double obs = 0;
            for (auto& r : Rr33) {
                double g = 0;
                for (int j : js) g += c ? r[j].y : r[j].x;
                g /= js.size();
                obs += g * g;
            }
            obs /= Rr33.size();
            ratio.push_back(obs / (zz / js.size()));
        }
        band_ratio[c] = median(ratio);
    }

    // ---- r4 (b) 평평한 벽 블록과 센서 잡음 ----
    std::vector<cv::Point> cands;
    {
        cv::Mat b = v6::blur(cen, 3);
        for (int y = 100; y < 2000; y += 64)
            for (int x = 200; x < 3600; x += 64) {
                cv::Rect r(x, y, 64, 64);
                cv::Scalar m, s;
                cv::meanStdDev(b(r), m, s);
                double mx;
                cv::minMaxLoc(cen(r), nullptr, &mx);
                if (s[0] < 1.0 && mx < 250) cands.push_back({x, y});
            }
    }
    double s1med, s2med;
    std::vector<std::array<double, 2>> band_pow;
    {
        cv::Mat A = read_f64(*im[idx_of("Y+1.00.bmp")].shot), B = read_f64(*im[idx_of("Y+1.05.bmp")].shot);
        auto hp = [](const cv::Mat& a) { cv::Mat g; cv::GaussianBlur(a, g, cv::Size(0, 0), 4); return cv::Mat(a - g); };
        std::vector<double> s1, s2;
        cv::Mat ps1 = cv::Mat::zeros(64, 64, CV_64F), ps2 = ps1.clone();
        int nb = std::min<int>(200, (int)cands.size());
        auto pw = [](const cv::Mat& a) {
            cv::Mat Fq, ch[2], m;
            cv::dft(a, Fq, cv::DFT_COMPLEX_OUTPUT);
            cv::split(Fq, ch);
            m = ch[0].mul(ch[0]) + ch[1].mul(ch[1]);
            return m;
        };
        for (int k = 0; k < nb; ++k) {
            cv::Rect r(cands[k].x, cands[k].y, 64, 64);
            cv::Mat a = hp(A(r).clone()), bb = hp(B(r).clone());
            cv::Scalar m, s;
            cv::meanStdDev(a(cv::Rect(8, 8, 48, 48)), m, s);
            s1.push_back(s[0]);
            cv::Mat d = a - bb;
            cv::meanStdDev(d(cv::Rect(8, 8, 48, 48)), m, s);
            s2.push_back(s[0] / std::sqrt(2.0));
            ps1 += pw(a);
            ps2 += pw(d / std::sqrt(2.0));
        }
        s1med = median(s1);
        s2med = median(s2);
        auto f = fftfreq(64);
        for (auto [lo, hi] : std::vector<std::pair<double, double>>{{0.02, 0.1}, {0.1, 0.2}, {0.2, 0.3}, {0.3, 0.5}}) {
            double a1 = 0, a2 = 0;
            int n = 0;
            for (int r = 0; r < 64; ++r)
                for (int c = 0; c < 64; ++c) {
                    double fr = std::hypot(f[r], f[c]);
                    if (fr >= lo && fr < hi) { a1 += ps1.at<double>(r, c); a2 += ps2.at<double>(r, c); ++n; }
                }
            band_pow.push_back({a1 / n / nb / 4096, a2 / n / nb / 4096});
        }
    }

    // ---- r5 픽셀 위상: 잔차가 이동 소수부의 주기 함수인가 (짝/홀 사진 교차검증) ----
    Table t_phase{{"측정", "차수", "x 설명 비율 (%)", "y 설명 비율 (%)"}};
    double phase_max = 0;
    std::vector<double> phase_v;
    for (int pi = 0; pi < 2; ++pi) {
        auto& Rr = pi ? Rr33 : Rr17;
        auto& Q = pi ? Q33 : Q17;
        std::vector<int> rows_i, fold;
        for (int i = 0; i < NI; ++i) if (i != ic) rows_i.push_back(i);
        { std::mt19937 fr(0); for (int i = 0; i < NI - 1; ++i) fold.push_back((int)(fr() & 1u)); }
        int N = (int)rows_i.size();
        for (int K : {1, 2}) {
            double ex[2];
            for (int c = 0; c < 2; ++c) {
                double num = 0, den = 0;
                for (int j = 0; j < NP; ++j) {
                    int nc = 1 + 4 * K + 4;
                    cv::Mat Bm(N, nc, CV_64F), y(N, 1, CV_64F);
                    for (int r = 0; r < N; ++r) {
                        double a = PQ[j].x + Q[rows_i[r]][j].x, b = PQ[j].y + Q[rows_i[r]][j].y;
                        a -= std::floor(a); b -= std::floor(b);
                        int q = 0;
                        Bm.at<double>(r, q++) = 1;
                        for (int k = 1; k <= K; ++k) {
                            Bm.at<double>(r, q++) = std::sin(2 * CV_PI * k * a); Bm.at<double>(r, q++) = std::cos(2 * CV_PI * k * a);
                            Bm.at<double>(r, q++) = std::sin(2 * CV_PI * k * b); Bm.at<double>(r, q++) = std::cos(2 * CV_PI * k * b);
                        }
                        Bm.at<double>(r, q++) = std::sin(2 * CV_PI * (a + b)); Bm.at<double>(r, q++) = std::cos(2 * CV_PI * (a + b));
                        Bm.at<double>(r, q++) = std::sin(2 * CV_PI * (a - b)); Bm.at<double>(r, q++) = std::cos(2 * CV_PI * (a - b));
                        y.at<double>(r) = c ? Rr[r][j].y : Rr[r][j].x;
                    }
                    for (int k = 0; k < 2; ++k) {
                        std::vector<int> tr, te;
                        for (int r = 0; r < N; ++r) (fold[r] != k ? tr : te).push_back(r);
                        cv::Mat Bt((int)tr.size(), nc, CV_64F), yt((int)tr.size(), 1, CV_64F), co;
                        double ym = 0;
                        for (size_t q = 0; q < tr.size(); ++q) { Bm.row(tr[q]).copyTo(Bt.row((int)q)); yt.at<double>((int)q) = y.at<double>(tr[q]); ym += y.at<double>(tr[q]); }
                        ym /= tr.size();
                        cv::solve(Bt, yt, co, cv::DECOMP_SVD);
                        for (int r : te) {
                            double pr = Bm.row(r).dot(co.t());
                            num += (y.at<double>(r) - pr) * (y.at<double>(r) - pr);
                            den += (y.at<double>(r) - ym) * (y.at<double>(r) - ym);
                        }
                    }
                }
                ex[c] = 1 - num / den;
                phase_max = std::max(phase_max, ex[c]);
                phase_v.push_back(ex[c] * 100);
            }
            t_phase.add({pi ? "33px" : "17px", std::to_string(K), fmt(ex[0] * 100, 1), fmt(ex[1] * 100, 1)});
        }
    }

    // ---- 보간 없는 사진 쌍: 판 이동 차이가 정수 픽셀에 가까운 Y 사진 쌍 ----
    std::vector<cv::Point2d> M33(NI);
    for (int i = 0; i < NI; ++i) {
        cv::Point2d s(0, 0);
        for (auto& p : Q33[i]) s += p;
        M33[i] = s / (double)NP;
    }
    struct Pair { double e; int a, b; cv::Point2d d; };
    std::vector<Pair> best;
    for (int a = 0; a < NI; ++a)
        for (int b = a + 1; b < NI; ++b) {
            if (im[a].axis != 'Y' || im[b].axis != 'Y') continue;
            cv::Point2d d = M33[b] - M33[a];
            best.push_back({std::abs(d.x - std::nearbyint(d.x)) + std::abs(d.y - std::nearbyint(d.y)), a, b, d});
        }
    std::sort(best.begin(), best.end(), [](const Pair& p, const Pair& q) { return std::tie(p.e, p.a, p.b) < std::tie(q.e, q.a, q.b); });
    std::map<int, cv::Mat> cache;
    auto photo = [&](int i) {
        cv::Mat m;
#pragma omp critical(photo_cache)
        {
            auto it = cache.find(i);
            if (it == cache.end()) it = cache.emplace(i, read_f64(*im[i].shot)).first;
            m = it->second;
        }
        return m;
    };
    // r6
    Table t_r6{{"쌍", "남은 소수 이동", "평평 영역 차이 sd", "경계 영역 차이 sd", "소수 이동 예상", "센서 잡음 예상"}};
    std::vector<std::array<double, 3>> r6v;
    for (int q = 0; q < 4; ++q) {
        auto& p = best[q];
        int ix = (int)std::nearbyint(p.d.x), iy = (int)std::nearbyint(p.d.y);
        const int X0 = 1430, X1 = 2390, Y0 = 950, Y1 = 1190;
        cv::Mat A = photo(p.a), B = photo(p.b);
        cv::Mat ra = A(cv::Range(Y0, Y1), cv::Range(X0, X1)).clone(), rb = B(cv::Range(Y0 + iy, Y1 + iy), cv::Range(X0 + ix, X1 + ix));
        cv::Mat diff = rb - ra;
        diff -= median(flat(diff));
        cv::Mat g = sobel_mag(ra, 1);
        auto gv = flat(g);
        double q85 = percentile(gv, 0.85), q30 = percentile(gv, 0.3);
        cv::Mat edge = g > q85, flt = g < q30;
        double res = std::hypot(p.d.x - ix, p.d.y - iy), pr = 0;
        auto ge = masked(g, edge);
        for (double v : ge) pr += v * res * v * res;
        pr = std::sqrt(pr / ge.size());
        double sf = std0(masked(diff, flt)), se = std0(masked(diff, edge));
        r6v.push_back({sf, se, pr});
        t_r6.add({im[p.a].name + " vs " + im[p.b].name, "(" + fmts(p.d.x - ix, 3) + "," + fmts(p.d.y - iy, 3) + ")", fmt(sf, 3), fmt(se, 3), fmt(pr, 3),
                  fmt(0.665 * std::sqrt(2.0), 2)});
    }
    // r8 밝기 맞춤 + 고역통과
    Table t_r8{{"쌍", "밝기비", "벽(판 밖 위쪽)", "판 어두운 평평", "QR 흰 평평", "QR 경계(기울기 상위)"}};
    std::vector<std::array<double, 5>> r8v;
    cv::Mat show_hp;
    for (int q = 0; q < 6; ++q) {
        auto& p = best[q];
        int ix = (int)std::nearbyint(p.d.x), iy = (int)std::nearbyint(p.d.y);
        const int X0 = 1300, X1 = 2520, Y0 = 700, Y1 = 1260;
        cv::Mat A = photo(p.a)(cv::Range(Y0, Y1), cv::Range(X0, X1)).clone();
        cv::Mat B = photo(p.b)(cv::Range(Y0 + iy, Y1 + iy), cv::Range(X0 + ix, X1 + ix)).clone();
        cv::Mat plate = cv::Mat::zeros(A.size(), CV_8U);
        for (int r = 0; r < A.rows; ++r)
            for (int c = 0; c < A.cols; ++c) {
                int xx = c + X0, yy = r + Y0;
                plate.at<uchar>(r, c) = (xx > 1440 && xx < 2380 && yy > 960 && yy < 1180) ? 255 : 0;
            }
        auto av = masked(A, plate), bv = masked(B, plate);
        auto co = polyfit1(av.data(), bv.data(), av.size());
        cv::Mat diff = B - (co.first * A + co.second), gd;
        cv::GaussianBlur(diff, gd, cv::Size(0, 0), 8);
        cv::Mat hpd = diff - gd;
        cv::Mat g = sobel_mag(A, 1);
        double q85 = percentile(masked(g, plate), 0.85);
        cv::Mat wall = cv::Mat::zeros(A.size(), CV_8U), dark = wall.clone(), white = wall.clone(), edge = wall.clone();
        for (int r = 0; r < A.rows; ++r)
            for (int c = 0; c < A.cols; ++c) {
                double gg = g.at<double>(r, c), a = A.at<double>(r, c);
                bool pl = plate.at<uchar>(r, c);
                wall.at<uchar>(r, c) = (r + Y0 < 900 && gg < 1 && a < 245) ? 255 : 0;
                dark.at<uchar>(r, c) = (pl && gg < 1 && a < 150) ? 255 : 0;
                white.at<uchar>(r, c) = (pl && gg < 1 && a >= 150) ? 255 : 0;
                edge.at<uchar>(r, c) = (pl && gg > q85) ? 255 : 0;
            }
        std::array<double, 5> v{co.first, std0(masked(hpd, wall)) / std::sqrt(2.0), std0(masked(hpd, dark)) / std::sqrt(2.0),
                                std0(masked(hpd, white)) / std::sqrt(2.0), std0(masked(hpd, edge)) / std::sqrt(2.0)};
        r8v.push_back(v);
        t_r8.add({im[p.a].name + "-" + im[p.b].name, fmt(v[0], 4), fmt(v[1], 3), fmt(v[2], 3), fmt(v[3], 3), fmt(v[4], 3)});
        if (q == 0) show_hp = hpd(cv::Range(950 - Y0, 1190 - Y0), cv::Range(1430 - X0, 2390 - X0)).clone();
    }
    // r11 밝기별 판 잡음 (국소 이동 보정 뒤)
    const int NB = 15;
    std::vector<double> bs(NB), bss(NB);
    std::vector<long long> bn(NB);
    std::array<double, 4> gs{}, gss{};
    std::array<long long, 4> gn{};
    double sxs = 0, sxss = 0;
    long long sxn = 0;
    for (int q = 0; q < 8; ++q) {
        auto& p = best[q];
        int ix = (int)std::nearbyint(p.d.x), iy = (int)std::nearbyint(p.d.y);
        const int X0 = 1440, X1 = 2380, Y0 = 960, Y1 = 1180;
        cv::Mat A = photo(p.a)(cv::Range(Y0, Y1), cv::Range(X0, X1)).clone();
        cv::Mat B = photo(p.b)(cv::Range(Y0 + iy, Y1 + iy), cv::Range(X0 + ix, X1 + ix)).clone();
        auto av = flat(A), bv = flat(B);
        auto co = polyfit1(av.data(), bv.data(), av.size());
        cv::Mat dd = B - (co.first * A + co.second), br = (A + B) / 2, g;
        cv::GaussianBlur(br, g, cv::Size(0, 0), 0.7);
        cv::Mat gx = gradient(g, 1), gy = gradient(g, 0);
        cv::Mat Sxx, Syy, Sxy, Sxd, Syd;
        cv::Size k(9, 9);
        cv::blur(gx.mul(gx), Sxx, k); cv::blur(gy.mul(gy), Syy, k); cv::blur(gx.mul(gy), Sxy, k);
        cv::blur(gx.mul(dd), Sxd, k); cv::blur(gy.mul(dd), Syd, k);
        for (int r = 0; r < A.rows; ++r)
            for (int c = 0; c < A.cols; ++c) {
                double sxx = Sxx.at<double>(r, c), syy = Syy.at<double>(r, c), sxy = Sxy.at<double>(r, c);
                double det = sxx * syy - sxy * sxy + 1e-6;
                double sx = (syy * Sxd.at<double>(r, c) - sxy * Syd.at<double>(r, c)) / det;
                double sy = (sxx * Syd.at<double>(r, c) - sxy * Sxd.at<double>(r, c)) / det;
                double gxx = gx.at<double>(r, c), gyy = gy.at<double>(r, c);
                double rr = dd.at<double>(r, c) - gxx * sx - gyy * sy, b = br.at<double>(r, c), gm = std::hypot(gxx, gyy);
                int bi = (int)std::ceil(b / 16) - 1;
                if (b > 0 && bi >= 0 && bi < NB) { bs[bi] += rr; bss[bi] += rr * rr; ++bn[bi]; }
                const double ge[5] = {0, 2, 8, 30, 200};
                for (int t = 0; t < 4; ++t)
                    if (gm > ge[t] && gm <= ge[t + 1]) { gs[t] += rr; gss[t] += rr * rr; ++gn[t]; }
                sxs += sx; sxss += sx * sx; ++sxn;
            }
    }
    auto var1 = [](double s, double ss, long long n) { return n > 1 ? (ss - s * s / n) / (n - 1) : kNaN; };
    NoiseModel nm;
    Table t_nb{{"밝기 구간", "픽셀 수", "한 장 잡음 sd"}};
    std::vector<double> nb_sd(NB);
    for (int b = 0; b < NB; ++b) {
        nb_sd[b] = bn[b] > 1 ? std::sqrt(var1(bs[b], bss[b], bn[b]) / 2) : kNaN;
        t_nb.add({"(" + std::to_string(b * 16) + ", " + std::to_string(b * 16 + 16) + "]", std::to_string(bn[b]), fmt(nb_sd[b], 3)});
        if (!std::isnan(nb_sd[b])) { nm.x.push_back(b * 16 + 8); nm.y.push_back(nb_sd[b]); }
    }
    Table t_ng{{"기울기 구간", "픽셀 수", "한 장 잡음 sd"}};
    const char* gl[4] = {"(0, 2]", "(2, 8]", "(8, 30]", "(30, 200]"};
    for (int t = 0; t < 4; ++t) t_ng.add({gl[t], std::to_string(gn[t]), fmt(std::sqrt(var1(gs[t], gss[t], gn[t]) / 2), 3)});
    double sx_sd = std::sqrt(var1(sxs, sxss, sxn));
    std::vector<double> wall_br;
    for (auto& c : cands) wall_br.push_back(cv::mean(cen(cv::Rect(c.x, c.y, 64, 64)))[0]);
    double wall_med = median(wall_br);
    {
        Table nb{{"brightness", "sd"}};
        for (size_t i = 0; i < nm.x.size(); ++i) nb.add({fmt(nm.x[i], 0), fmt(nm.y[i], 5)});
        nb.csv(out("r11_noise_by_brightness.csv"));
    }

    // ---- r13 이론 하한 ----
    Table t_crlb{{"점 창 크기", "같은 무게 (px)", "잡음 가중 최적 (px)", "개선 배수"}};
    std::map<int, std::array<double, 3>> crlb;
    {
        cv::Mat c64;
        cen.convertTo(c64, CV_64F);
        cv::Rect roi(1300, 850, 1200, 450);  // 점 창이 모두 들어가는 범위 (gradient 는 내부에서 가운데 차분이라 전체와 같다)
        cv::Mat sub = c64(roi);
        cv::Mat gx = gradient(sub, 1);
        for (int hp : {8, 16, 24}) {
            std::vector<double> u, o, rt;
            for (int k : qr) {
                double sgg = 0, sggv = 0, sg_v = 0;
                for (int y = F[k].y - hp; y <= F[k].y + hp; ++y)
                    for (int x = F[k].x - hp; x <= F[k].x + hp; ++x) {
                        double g = gx.at<double>(y - roi.y, x - roi.x), s = nm.at(sub.at<double>(y - roi.y, x - roi.x)), v = s * s;
                        sgg += g * g; sggv += g * g * v; sg_v += g * g / v;
                    }
                double uu = std::sqrt(sggv / (sgg * sgg)), oo = std::sqrt(1 / sg_v);
                u.push_back(uu); o.push_back(oo); rt.push_back(uu / oo);
            }
            crlb[hp] = {median(u), median(o), median(rt)};
            t_crlb.add({std::to_string(2 * hp + 1) + "px", fmt(median(u), 4), fmt(median(o), 4), fmt(median(rt), 2)});
        }
    }

    // ---- 합성 사다리 (r3 일부 + r12) ----
    cv::Mat CEN;
    cen(cv::Range(RY0, RY1), cv::Range(RX0, RX1)).convertTo(CEN, CV_64F);
    std::vector<v6::Feat> Fc;
    for (int k : qr) { auto f = F[k]; f.x -= RX0; f.y -= RY0; Fc.push_back(f); }
    auto add_sensor = [&](cv::Mat out64, std::mt19937_64& rng) {
        std::normal_distribution<double> nd(0, SN);
        for (int r = 0; r < out64.rows; ++r) {
            double* p = out64.ptr<double>(r);
            for (int c = 0; c < out64.cols; ++c) p[c] += nd(rng);
        }
        cv::Mat f;
        out64.convertTo(f, CV_32F);
        return f;
    };
    cv::Mat TMPL;
    {
        std::mt19937_64 rng(999);
        TMPL = add_sensor(fourier_shift(CEN, 0, 0, B0, B0), rng);
    }
    v6::Pipe s17{8, 1.0}, s33{16, 2.0};
    s17.setup(TMPL, Fc);
    s33.setup(TMPL, Fc);
    auto sbig = v6::big_templates(TMPL, Fc);
    const std::vector<std::string> conds = {"잡음만", "잡음+조명 기울기", "판 잡음(밝기별, 픽셀 독립)", "판 잡음 절반 크기", "무늬 결 변화(고주파 무늬 x 무작위)"};
    const std::vector<std::string> conds_en = {"sensor only", "sensor+light", "plate noise", "plate x0.5", "pattern"};
    const int NS = 40;
    struct SynthRow { double bias_x, err_x, err_y, jit_x, jit_y; };
    std::map<std::pair<int, int>, SynthRow> syn;  // (cond, pipe)
    for (int ci = 0; ci < (int)conds.size(); ++ci) {
        std::vector<std::vector<cv::Point2d>> E[2];
        E[0].resize(NS); E[1].resize(NS);
#pragma omp parallel for schedule(dynamic)
        for (int k = 0; k < NS; ++k) {
            std::mt19937_64 rng((ci < 2 ? 1000 : 5000) + k + 100000ULL * ci);
            std::uniform_real_distribution<double> ux(-20, 20), uy(-1, 1);
            double tx = ux(rng), ty = uy(rng);
            cv::Mat o = fourier_shift(CEN, tx, ty, B0, B0);
            if (ci == 1) {
                std::normal_distribution<double> g01(0, 0.01);
                double g0 = g01(rng), g1 = g01(rng), g2 = g01(rng);
                double mx = (o.cols - 1) / 2.0, my = (o.rows - 1) / 2.0;
                double sx = std::sqrt((o.cols * (double)o.cols - 1) / 12), sy = std::sqrt((o.rows * (double)o.rows - 1) / 12);
                for (int r = 0; r < o.rows; ++r)
                    for (int c = 0; c < o.cols; ++c)
                        o.at<double>(r, c) *= 1 + g0 + g1 * (c - mx) / sx * 0.3 + g2 * (r - my) / sy * 0.3;
            } else if (ci == 2 || ci == 3) {
                std::normal_distribution<double> n1(0, 1);
                double sc = ci == 3 ? 0.5 : 1.0;
                for (int r = 0; r < o.rows; ++r)
                    for (int c = 0; c < o.cols; ++c) { double& v = o.at<double>(r, c); v = v + n1(rng) * nm.at(v) * sc; }
            } else if (ci == 4) {
                cv::Mat bl, hf;
                cv::GaussianBlur(o, bl, cv::Size(0, 0), 1.5);
                hf = o - bl;
                std::normal_distribution<double> n15(0, 0.15);
                for (int r = 0; r < o.rows; ++r)
                    for (int c = 0; c < o.cols; ++c) o.at<double>(r, c) += hf.at<double>(r, c) * n15(rng);
            }
            cv::Mat img = add_sensor(o, rng);
            auto c0 = v6::coarse(img, sbig, Fc);
            for (int pi = 0; pi < 2; ++pi) {
                auto d = (pi ? s33 : s17).run(img, c0, Fc);
                for (auto& v : d) v -= cv::Point2d(tx, ty);
                E[pi][k] = d;
            }
        }
        for (int pi = 0; pi < 2; ++pi) {
            SynthRow s{};
            double bx = 0;
            for (auto& e : E[pi]) for (auto& v : e) bx += v.x;
            s.bias_x = bx / (NS * (double)Fc.size());
            auto ej = point_jitter(E[pi]);
            s.err_x = ej.first; s.err_y = ej.second;
            auto Ec = E[pi];
            for (auto& e : Ec) {
                cv::Point2d m(0, 0);
                for (auto& v : e) m += v;
                m /= (double)e.size();
                for (auto& v : e) v -= m;
            }
            auto cj = point_jitter(Ec);
            s.jit_x = cj.first; s.jit_y = cj.second;
            syn[{ci, pi}] = s;
        }
        std::cout << "synth " << ci << " " << syn[{ci, 0}].jit_x << " " << syn[{ci, 0}].jit_y << "\n";
    }
    Table t_syn{{"조건", "측정", "편향 x (px)", "점 오차 x", "점 오차 y", "점 흔들림 x", "점 흔들림 y"}};
    for (int ci = 0; ci < (int)conds.size(); ++ci)
        for (int pi = 0; pi < 2; ++pi) {
            auto& s = syn[{ci, pi}];
            t_syn.add({conds[ci], pi ? "33px σ2" : "17px σ1", fmts(s.bias_x, 5), fmt(s.err_x, 5), fmt(s.err_y, 5), fmt(s.jit_x, 5), fmt(s.jit_y, 5)});
        }
    t_syn.add({"실제 사진 (강체 검사)", "17px σ1", "", "", "", fmt(rnd17.first, 5), fmt(rnd17.second, 5)});
    t_syn.add({"실제 사진 (강체 검사)", "33px σ2", "", "", "", fmt(rnd33.first, 5), fmt(rnd33.second, 5)});
    t_syn.csv(out("synth.csv"));

    // ---- 튀는 값: 강체 잔차(17px, 그대로)가 가장 큰 사진·점 ----
    struct Out { int i, j; double v; };
    auto worst_of = [&](int c, int n) {
        std::vector<Out> all;
        for (int i = 0; i < NI; ++i) {
            if (i == ic) continue;
            for (int j = 0; j < NP; ++j) all.push_back({i, j, c ? R17[i][j].y : R17[i][j].x});
        }
        std::stable_sort(all.begin(), all.end(), [](const Out& a, const Out& b) { return std::abs(a.v) > std::abs(b.v); });
        all.resize(n);
        return all;
    };
    auto wx = worst_of(0, 5), wy = worst_of(1, 5);
    std::vector<double> ptsx_raw, ptsy_raw;
    point_jitter(drop_c(R17), &ptsx_raw, &ptsy_raw);
    Table t_out{{"구분", "사진", "정답 Y (mm)", "정답 T (deg)", "점 번호", "영역", "점 위치 (px)", "측정 이동 (px)", "강체 잔차 x / y (px)", "이 점의 평소 흔들림 x / y (px)"}};
    for (auto [lab, lst] : std::vector<std::pair<std::string, std::vector<Out>>>{{"x 잔차 상위", wx}, {"y 잔차 상위", wy}})
        for (auto& o : lst) {
            auto& s = *im[o.i].shot;
            t_out.add({lab, im[o.i].name, fmt(s.y, 2), fmt(s.t, 2), std::to_string(o.j), v6::region_name(F[qr[o.j]].region),
                       "(" + std::to_string((int)PQ[o.j].x) + ", " + std::to_string((int)PQ[o.j].y) + ")",
                       "(" + fmts(Q17[o.i][o.j].x, 3) + ", " + fmts(Q17[o.i][o.j].y, 3) + ")", fmts(R17[o.i][o.j].x, 4) + " / " + fmts(R17[o.i][o.j].y, 4),
                       fmt(ptsx_raw[o.j], 4) + " / " + fmt(ptsy_raw[o.j], 4)});
        }
    auto outlier_fig = [&](const Out& o, const std::string& file) {
        cv::Mat img = read_color(*im[o.i].shot);
        cv::Point2d meas = PQ[o.j] + Q17[o.i][o.j], model = meas + R17[o.i][o.j];
        cv::Rect roi((int)meas.x - 24, (int)meas.y - 24, 49, 49);
        auto tile = zoom(img, roi, 490, {{meas, R17[o.i][o.j], "measured", kRed}, {model, {0, 0}, "rigid fit", kGreen}}, 150,
                         im[o.i].name + " pt " + std::to_string(o.j) + " resid (" + fmts(R17[o.i][o.j].x, 3) + ", " + fmts(R17[o.i][o.j].y, 3) + ") px",
                         {{cv::Rect2d(meas.x - 8.5, meas.y - 8.5, 17, 17), kRed, "17px window"}});
        cv::Mat c3 = read_color(*im[ic].shot);
        auto ref = zoom(c3, cv::Rect(F[qr[o.j]].x - 24, F[qr[o.j]].y - 24, 49, 49), 490, {{PQ[o.j], {0, 0}, "reference", kGreen}}, 0, "center.bmp same point");
        save_png(file, hstack({ref, tile}));
    };
    std::string fx_name = "06_튀는값_x_" + im[wx[0].i].name.substr(0, im[wx[0].i].name.size() - 4) + ".png";
    std::string fy_name = "07_튀는값_y_" + im[wy[0].i].name.substr(0, im[wy[0].i].name.size() - 4) + ".png";
    outlier_fig(wx[0], fx_name);
    outlier_fig(wy[0], fy_name);

    // ---- 그림 ----
    {
        std::vector<std::string> cats;
        std::vector<double> vx, vy;
        for (int ci = 0; ci < (int)conds.size(); ++ci) { cats.push_back(conds_en[ci]); vx.push_back(syn[{ci, 0}].jit_x); vy.push_back(syn[{ci, 0}].jit_y); }
        cats.push_back("real (rigid)");
        vx.push_back(rnd17.first);
        vy.push_back(rnd17.second);
        bar_chart("01_점잡음_가로_17px.png", "Point jitter x, 17px sigma1 (px)", cats, vx, "px", (int)cats.size() - 1, rnd17.first);
        bar_chart("02_점잡음_세로_17px.png", "Point jitter y, 17px sigma1 (px)", cats, vy, "px", (int)cats.size() - 1, rnd17.second);
    }
    {
        Chart c("Per-image noise sd by brightness (plate, integer-shift pairs)", "brightness", "noise sd (gray level)");
        c.line(nm.x, nm.y, color(0), "plate (pair diff)", true).hline(s2med, kRed).note(wall_med, s2med, "wall " + fmt(s2med, 2));
        c.save("03_밝기별_판잡음.png");
    }
    {
        cv::Mat v;
        show_hp.convertTo(v, CV_8U, 10, 128);
        cv::Mat c3;
        cv::cvtColor(v, c3, cv::COLOR_GRAY2BGR);
        cv::resize(c3, c3, cv::Size(), 1.25, 1.25, cv::INTER_NEAREST);
        cv::rectangle(c3, {0, 0}, {text_w(im[best[0].a].name + " - " + im[best[0].b].name + " (integer shift, x10)", 0.55) + 12, 26}, cv::Scalar(255, 255, 255), -1);
        text(c3, im[best[0].a].name + " - " + im[best[0].b].name + " (integer shift, x10)", {6, 19}, 0.55);
        save_png("04_보간없는_쌍차이.png", c3);
    }
    {
        std::vector<std::string> cats;
        std::vector<double> v;
        for (int hp : {8, 16, 24}) {
            cats.push_back(std::to_string(2 * hp + 1) + "px uniform");
            v.push_back(crlb[hp][0]);
            cats.push_back(std::to_string(2 * hp + 1) + "px weighted");
            v.push_back(crlb[hp][1]);
        }
        cats.push_back("17px synth plate");
        v.push_back(syn[{2, 0}].jit_x);
        bar_chart("05_이론하한.png", "Cramer-Rao bound per point (px)", cats, v, "px", (int)cats.size() - 1);
    }
    {
        std::vector<double> x;
        for (int j = 0; j < NP; ++j) x.push_back(PQ[j].x);
        Chart c("Rigid residual (random part) per point, 17px", "point x in center image (px)", "jitter (px)");
        c.scatter(x, ptsx17, color(0), "x").scatter(x, ptsy17, color(1), "y");
        c.save("00_점별_강체잔차.png");
    }

    // ---- 원래 결과와 대조 ----
    Parity par;
    // 특징점 수: analysis_v6/features_qr_hp8.csv
    par.add("특징점 수 (QR+나사)", 250, (double)F.size(), 0, 0);
    par.add("QR 좌 / 중 / 우 점 수 (81/73/78)", 817378, cnt[0] * 10000.0 + cnt[1] * 100 + cnt[2], 0, 0);
    par.add("판 나사 점 수", 18, cnt[3], 0, 0);
    // r1_rigid.csv
    par.add("17px 강체 잔차 그대로 x", 0.0239, raw17.first, 0.0005);
    par.add("17px 강체 잔차 그대로 y", 0.0172, raw17.second, 0.0005);
    par.add("17px 무작위 부분 x", 0.0222, rnd17.first, 0.0005);
    par.add("17px 무작위 부분 y", 0.0136, rnd17.second, 0.0005);
    par.add("33px 강체 잔차 그대로 x", 0.0162, raw33.first, 0.0005);
    par.add("33px 강체 잔차 그대로 y", 0.0110, raw33.second, 0.0005);
    par.add("33px 무작위 부분 x", 0.0149, rnd33.first, 0.0005);
    par.add("33px 무작위 부분 y", 0.0089, rnd33.second, 0.0005);
    // r4_flat.npy 크기
    par.add("평평한 벽 블록 수", 643, (double)cands.size(), 0, 0);
    // r6_pairdiff.csv
    const std::vector<std::string> r6n = {"Y-0.20.bmp vs Y+0.40.bmp", "Y-0.70.bmp vs Y+0.60.bmp", "Y-0.80.bmp vs Y+0.50.bmp", "Y-1.45.bmp vs Y-0.75.bmp"};
    const double r6o[4][2] = {{2.843, 8.188}, {3.434, 9.37}, {3.45, 9.467}, {3.064, 8.959}};
    for (int q = 0; q < 4; ++q) {
        std::string nn = im[best[q].a].name + " vs " + im[best[q].b].name;
        par.add_text("r6 쌍 " + std::to_string(q + 1), r6n[q], nn, nn == r6n[q]);
        par.add("r6 쌍 " + std::to_string(q + 1) + " 평평 영역 차이 sd", r6o[q][0], r6v[q][0], 0.01, 3);
        par.add("r6 쌍 " + std::to_string(q + 1) + " 경계 영역 차이 sd", r6o[q][1], r6v[q][1], 0.02, 3);
    }
    // r8_pairdiff2.csv
    const double r8o[6][4] = {{1.3112, 1.5944, 2.333, 5.9533}, {1.5474, 1.7765, 2.9787, 6.6769}, {1.55, 1.8054, 2.9632, 6.7439},
                              {1.349, 1.6152, 2.6414, 6.4169}, {1.8793, 1.7716, 3.1859, 7.3864}, {1.6828, 1.8236, 3.0714, 6.8367}};
    const char* r8l[4] = {"벽", "판 어두운 평평", "QR 흰 평평", "QR 경계"};
    for (int q = 0; q < 6; ++q)
        for (int c = 0; c < 4; ++c) par.add("r8 쌍 " + std::to_string(q + 1) + " " + r8l[c], r8o[q][c], r8v[q][c + 1], 0.02, 3);
    // r11_noise_by_brightness.npy
    const double nbo[15] = {kNaN, kNaN, 1.4241, 2.4869, 2.4769, 2.6844, 4.5776, 6.3401, 6.2853, 5.2514, 4.7082, 4.9115, 3.9869, 8.0886, 11.9427};
    for (int b = 2; b < 15; ++b) par.add("밝기 " + std::to_string(b * 16 + 8) + " 판 잡음 sd", nbo[b], nb_sd[b], 0.02, 3);
    // r13 (00_v8결과리포트.md 6절)
    const double co[3][2] = {{0.0124, 0.0101}, {0.0066, 0.0054}, {0.0045, 0.0036}};
    int hi = 0;
    for (int hp : {8, 16, 24}) {
        par.add("CRLB " + std::to_string(2 * hp + 1) + "px 같은 무게", co[hi][0], crlb[hp][0], 0.0002);
        par.add("CRLB " + std::to_string(2 * hp + 1) + "px 잡음 가중", co[hi][1], crlb[hp][1], 0.0002);
        ++hi;
    }
    // r3_synth.csv, r12_synth2.csv: 난수열이 numpy 와 달라 40회 통계 오차만큼(상대 15%) 허용
    const double so[5][2][2] = {{{0.00208, 0.00197}, {0.00155, 0.00149}}, {{0.00207, 0.00202}, {0.00153, 0.0016}},
                                {{0.01379, 0.01364}, {0.01014, 0.01006}}, {{0.00712, 0.00705}, {0.00526, 0.00519}},
                                {{0.01503, 0.01547}, {0.01052, 0.01067}}};
    for (int ci = 0; ci < 5; ++ci)
        for (int pi = 0; pi < 2; ++pi) {
            auto& s = syn[{ci, pi}];
            std::string nm2 = "합성 " + conds[ci] + " " + (pi ? "33px" : "17px");
            par.add(nm2 + " 흔들림 x", so[ci][pi][0], s.jit_x, so[ci][pi][0] * 0.15, 5);
            par.add(nm2 + " 흔들림 y", so[ci][pi][1], s.jit_y, so[ci][pi][1] * 0.15, 5);
        }
    // r5: 원래 r5_phase.py 를 원래 저장 배열(r1_resid_*.npy, tracks)에 다시 돌려 얻은 출력 (리포트의 "0~9%"는 33px 와 x 만 해당, 17px y 는 원래도 27%).
    // 반 나누기 난수가 numpy 와 달라 4%p 허용
    const double pho[8] = {-2.1, 26.9, -6.0, 26.0, 9.2, 9.2, 4.4, 7.6};
    const char* phn[8] = {"17px 1차 x", "17px 1차 y", "17px 2차 x", "17px 2차 y", "33px 1차 x", "33px 1차 y", "33px 2차 x", "33px 2차 y"};
    for (int q = 0; q < 8; ++q) par.add(std::string("픽셀 위상 설명 비율 (%) ") + phn[q], pho[q], phase_v[q], 4.0, 1);

    // ---- 결과.md ----
    double rep_y = syn[{2, 0}].jit_y / rnd17.second, rep_x = syn[{2, 0}].jit_x / rnd17.first;
    std::ostringstream R;
    R << "# 결과: 특징점 잡음의 원인 (강체 검사·합성 사다리·쌍 차이·이론 하한)\n\n"
      << "입력: `data/clean/clean_20261014` 281장. 기준 = center.bmp. 특징점은 흐린 center(σ1)의 QR 판 영역에서 `goodFeaturesToTrack`(간격 17px)으로 고르고 QR 점 "
      << NP << "개 + 판 나사 " << cnt[3] << "개를 썼다. 추적은 81x81 큰 템플릿 정수 탐색 → 정규화 상관 + 포물선 → 흐린 영상에서 (dx, dy, a, b) 가우스-뉴턴(3차 스플라인)이다. "
      << "두 가지 설정: 17px σ1 (원래 v6), 33px σ2 (원래 v7 `tr_fs8_hp16_s2`).\n\n"
      << "## 1. 강체 검사 (r1)\n\n사진마다 QR 점 " << NP << "개에 호모그래피를 맞추고 남은 양이 점 오차다. 무작위 부분은 촬영 구간 6개마다 값에 대한 2차를 뺀 나머지다.\n\n"
      << t_rigid.md() << "\n계통 성분(점마다 평균 잔차) RMS: 17px (" << fmt(sys17.x, 4) << ", " << fmt(sys17.y, 4) << "), 33px (" << fmt(sys33.x, 4) << ", " << fmt(sys33.y, 4) << ") px.\n\n"
      << "## 2. 잡음 공간 구조 (r4 a, 33px 무작위 부분)\n\n" << t_corr.md()
      << "\n같은 줄(10px) 묶음 평균의 분산 / 독립 기대: x " << fmt(band_ratio[0], 2) << ", y " << fmt(band_ratio[1], 2) << " (1이면 줄 공통 흔들림 없음).\n\n"
      << "## 3. 센서 잡음 (r4 b)\n\n평평한 벽 블록 " << cands.size() << "개 (64x64, 흐린 center 표준편차 < 1, 최대 < 250). Y+1.00, Y+1.05 두 장으로 처음 200개를 쟀다.\n\n"
      << "| 지표 | 값 |\n|---|---|\n| 한 장 잡음 (중앙) | " << fmt(s1med, 3) << " |\n| 두 장 차이로 본 시간 잡음 (중앙) | " << fmt(s2med, 3) << " |\n"
      << "| 고정 패턴 몫 | " << fmt(std::sqrt(std::max(s1med * s1med - s2med * s2med, 0.0)), 3) << " |\n| 벽 블록 밝기 (중앙) | " << fmt(wall_med, 1) << " |\n\n"
      << "| 주파수 | 한 장 잡음 파워 | 시간 잡음 파워 |\n|---|---|---|\n";
    {
        const char* fl[4] = {"0.02~0.1", "0.1~0.2", "0.2~0.3", "0.3~0.5"};
        for (int i = 0; i < 4; ++i) R << "| " << fl[i] << " | " << fmt(band_pow[i][0], 3) << " | " << fmt(band_pow[i][1], 3) << " |\n";
    }
    R << "\n## 4. 픽셀 위상 (r5)\n\n점마다 강체 잔차를 현재 위치 소수부의 sin/cos 로 회귀한 교차검증 설명 비율. 원래는 numpy 난수(시드 0)로 반을 나눴고 여기서는 std::mt19937(0)으로 나눴다. 원래 리포트는 \"0~9%\"라고 적었지만 원래 코드를 원래 저장 배열에 다시 돌리면 17px y 는 27%다(아래 대조표). 17px y 잔차의 약 1/4이 세로 위상과 관련 있다.\n\n"
      << t_phase.md() << "\n## 5. 보간 없는 사진 쌍 차이 (r6, r8)\n\n33px 추적의 QR 평균 이동 차이가 정수에 가장 가까운 Y 사진 쌍을 정수만큼 옮겨 바로 뺐다.\n\n" << t_r6.md()
      << "\n밝기 맞춤(a·A+b) + 고역통과(σ8) 뒤 영역별 한 장 기준 크기 (차이/√2):\n\n" << t_r8.md()
      << "\n## 6. 밝기별 판 잡음 (r11)\n\n정수 이동 쌍 8개, 판 영역(1440~2380, 960~1180). 9x9 국소 최소제곱으로 남은 이동을 뺀 뒤 차이의 분산/2.\n\n" << t_nb.md()
      << "\n기울기별:\n\n" << t_ng.md() << "\n국소 이동 보정량 sd: " << fmt(sx_sd, 4) << " px. 벽 평평 블록 밝기 중앙 " << fmt(wall_med, 1) << ".\n\n"
      << "## 7. 합성 사다리 (r3 일부, r12)\n\ncenter 의 판 주변(" << RX0 << "~" << RX1 << ", " << RY0 << "~" << RY1 << ")을 푸리에로 정확히 옮기고(tx ∈ ±20, ty ∈ ±1 px, 기본 흐림 Δσ² "
      << B0 << ") 센서 잡음 " << SN << "을 넣었다. 조건마다 " << NS << "장, 고정 시드 std::mt19937_64. 점 흔들림 = 사진마다 점 평균을 뺀 오차의 점별 표준편차 중앙값.\n\n" << t_syn.md()
      << "\n## 8. 이론 하한 (r13)\n\n점 창 안 픽셀 잡음 sd = 6절 밝기별 모델, 기울기 = center 의 가로 기울기.\n\n" << t_crlb.md()
      << "\n## 옮기지 않은 부분\n\n- r2 (사진마다 흐림 차이 스펙트럼)와 그것을 쓰는 r3 흐림 조건 2개, r3 일렁임 조건(5차 스플라인 재표본)은 옮기지 않았다.\n"
      << "- r7 (시점 의존성), r9 (압축·색 보간 흔적), r10 (차이 영상 확대 그림)은 옮기지 않았다. 결론에는 4절·5절 수치만 쓴다.\n- build_ppt_v8.js (발표 파일)는 옮기지 않았다.\n\n"
      << "## 원래 결과와 대조 (" << par.ok << "/" << par.total << " 일치)\n\n" << par.t.md()
      << "\n## 파일\n\n- `r1_rigid.csv`: 강체 검사 점 흔들림\n- `r11_noise_by_brightness.csv`: 밝기별 판 잡음 모델\n- `synth.csv`: 합성 사다리\n- `이미지/`: 그림 8장\n";
    write_text(out("결과.md"), R.str());

    // ---- 보고서.md ----
    auto& ox = wx[0];
    auto& oy = wy[0];
    std::ostringstream P;
    P << "# 보고서: 특징점 잡음의 원인 (10-14 16:54)\n\n"
      << "## 문제\n\n특징점 위치 잡음이 어디서 오는지 몰라 판정 100%를 막는 원인을 알 수 없었다.\n\n"
      << "## 해결 방법\n\n강체 검사로 실제 점 잡음을 재고, 보간 없는 사진 쌍 차이로 판 잡음을 밝기별로 잰 뒤, 그 잡음을 넣은 합성 사진과 이론 하한으로 원인을 갈랐다.\n\n"
      << "## 결과\n\n실측 판 잡음만 넣은 합성이 실제 점 잡음의 세로 " << fmt(rep_y * 100, 0) << "%, 가로 " << fmt(rep_x * 100, 0)
      << "%를 재현했고, 센서 잡음·조명만으로는 " << fmt(syn[{0, 0}].jit_x / rnd17.first * 100, 0) << "%에 그쳐 주원인은 찍을 때마다 바뀌는 판 무늬다.\n\n"
      << "| 지표 (17px σ1) | x (px) | y (px) |\n|---|---|---|\n"
      << "| 실제 사진 강체 잔차 무작위 부분 | " << fmt(rnd17.first, 4) << " | " << fmt(rnd17.second, 4) << " |\n"
      << "| 합성: 센서 잡음만 (" << SN << ") | " << fmt(syn[{0, 0}].jit_x, 4) << " | " << fmt(syn[{0, 0}].jit_y, 4) << " |\n"
      << "| 합성: + 조명 기울기 | " << fmt(syn[{1, 0}].jit_x, 4) << " | " << fmt(syn[{1, 0}].jit_y, 4) << " |\n"
      << "| 합성: 실측 판 잡음 (밝기별, 픽셀 독립) | " << fmt(syn[{2, 0}].jit_x, 4) << " | " << fmt(syn[{2, 0}].jit_y, 4) << " |\n"
      << "| 합성: 무늬 결 변화 | " << fmt(syn[{4, 0}].jit_x, 4) << " | " << fmt(syn[{4, 0}].jit_y, 4) << " |\n"
      << "| 이론 하한 (같은 무게 / 잡음 가중) | " << fmt(crlb[8][0], 4) << " | " << fmt(crlb[8][1], 4) << " |\n\n"
      << "| 영역 (정수 이동 쌍, 한 장 기준) | 변화 크기 |\n|---|---|\n"
      << "| 매끈한 벽 (시간 잡음) | " << fmt(s2med, 2) << " |\n| 판 어두운 평평 (쌍 6개 중앙) | " << fmt(median({r8v[0][2], r8v[1][2], r8v[2][2], r8v[3][2], r8v[4][2], r8v[5][2]}), 2) << " |\n"
      << "| QR 흰 평평 | " << fmt(median({r8v[0][3], r8v[1][3], r8v[2][3], r8v[3][3], r8v[4][3], r8v[5][3]}), 2) << " |\n"
      << "| QR 경계 | " << fmt(median({r8v[0][4], r8v[1][4], r8v[2][4], r8v[3][4], r8v[4][4], r8v[5][4]}), 2) << " |\n\n"
      << img_md("01_점잡음_가로_17px.png", "가로 점 잡음") << "\n\n" << img_md("02_점잡음_세로_17px.png", "세로 점 잡음") << "\n\n"
      << img_md("03_밝기별_판잡음.png", "밝기별 판 잡음") << "\n\n" << img_md("04_보간없는_쌍차이.png", "보간 없는 쌍 차이") << "\n\n"
      << img_md("05_이론하한.png", "이론 하한") << "\n\n" << img_md("00_점별_강체잔차.png", "점별 강체 잔차") << "\n\n"
      << "## 튀는 값\n\n강체 잔차가 가장 큰 점은 가로 " << im[ox.i].name << "의 " << ox.j << "번 점(" << fmts(ox.v, 3) << " px), 세로 " << im[oy.i].name << "의 " << oy.j
      << "번 점(" << fmts(oy.v, 3) << " px)이고, 그 점들의 평소 흔들림은 " << fmt(ptsx_raw[ox.j], 3) << " / " << fmt(ptsy_raw[oy.j], 3)
      << " px로 전체 중앙값 " << fmt(raw17.first, 3) << " / " << fmt(raw17.second, 3)
      << " px의 몇 배라서, 튀는 값은 한 장의 사고가 아니라 늘 크게 흔들리는 소수의 점이 여러 사진에서 반복해 만든다.\n\n" << t_out.md() << "\n" << img_md(fx_name, "x 튀는 값") << "\n\n" << img_md(fy_name, "y 튀는 값")
      << "\n\n왼쪽은 center.bmp의 같은 점(초록 십자 = 기준 위치), 오른쪽은 그 사진(빨간 십자 = 측정 위치, 초록 십자 = 강체 변환이 예측한 위치, 화살표 = 잔차 x150, 빨간 상자 = 17px 창)이다.\n\n"
      << "## 한줄평\n\n판의 무늬 있는 곳은 찍을 때마다 한 장 기준 " << fmt(nm.y.front(), 1) << "~" << fmt(nm.y.back(), 1) << " 밝기만큼 픽셀마다 바뀌어 벽의 센서 잡음 " << fmt(s2med, 2)
      << "보다 몇 배 크고, 이 잡음이 17px 창의 이론 하한 " << fmt(crlb[8][0], 4) << " px를 만들어서, 매칭을 바꿔도 점 잡음은 " << fmt(syn[{2, 0}].jit_y, 4)
      << " px 아래로 내려가기 어렵다. 가로만 실제가 합성보다 " << fmt(rnd17.first / syn[{2, 0}].jit_x, 1) << "배 커서 가로 전용 성분이 따로 남는다.\n\n"
      << "## 원래 결과와 대조\n\n원래 값은 `analysis_v8/r1_rigid.csv`, `r4_flat.npy`(블록 수), `r6_pairdiff.csv`, `r8_pairdiff2.csv`, `r11_noise_by_brightness.npy`, `r3_synth.csv`, `r12_synth2.csv`, "
      << "`00_v8결과리포트.md`(CRLB), r5_phase.py 재실행 출력(위상), `analysis_v6/features_qr_hp8.csv`(점 수)에서 가져왔다. 합성은 numpy 난수열을 C++에서 재현할 수 없어 40장 통계 오차만큼(상대 15%) 허용했다. "
      << par.ok << "/" << par.total << " 일치.\n\n" << par.t.md();
    write_text(out("보고서.md"), P.str());
    std::cout << "parity " << par.ok << "/" << par.total << "\n";
    return 0;
}
