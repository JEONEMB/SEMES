// direct_alignment/direct_pair.py DenseRegistration 의 C++ 이식.
// QR 3개 안쪽(경계에서 inset px) 픽셀만으로 QR별 gain/bias 를 빼고 P12 변형 12계수를 Gauss-Newton 으로 맞춘다.
#pragma once
#include <opencv2/imgproc.hpp>
#include <array>
#include <map>
#include <cmath>
#include <vector>
#include "track.hpp"

namespace dr {

const std::array<cv::Vec4i, 3> QR_BOUNDS{{{1563, 982, 1721, 1153}, {1821, 983, 1980, 1154}, {2079, 984, 2239, 1157}}};
const double XC = 1906, YC = 1070, XS = 486, YS = 122;

inline void basis(double px, double py, double b[6]) {
    double x = (px - XC) / XS, y = (py - YC) / YS;
    b[0] = 1; b[1] = x; b[2] = y; b[3] = x * x; b[4] = x * y; b[5] = y * y;
}

struct Region {
    std::vector<double> x, y, T;
    std::vector<std::array<double, 6>> B;
};

struct Result {
    std::array<double, 12> X{}, initial{};
    double mse_initial = 0, mse_final = 0, condition = 0;
    int iterations = 0;
    std::array<std::array<double, 2>, 3> gains{};
};

struct Dense {
    int inset = 10;
    int x0, y0, x1, y1;
    std::array<cv::Vec4i, 3> rects;
    std::map<int, std::vector<Region>> geom;
    int pixel_count = 0;

    explicit Dense(const cv::Mat& ref_gray, int ins = 10) : inset(ins) {
        x0 = 1 << 30; y0 = 1 << 30; x1 = 0; y1 = 0;
        for (int k = 0; k < 3; ++k) {
            auto q = QR_BOUNDS[k];
            rects[k] = {q[0] + inset, q[1] + inset, q[2] - inset, q[3] - inset};
            x0 = std::min(x0, q[0]); y0 = std::min(y0, q[1]); x1 = std::max(x1, q[2]); y1 = std::max(y1, q[3]);
        }
        x0 -= 120; y0 -= 24; x1 += 121; y1 += 25;
        cv::Mat ref;
        ref_gray(cv::Range(y0, y1), cv::Range(x0, x1)).convertTo(ref, CV_64F);
        cv::GaussianBlur(ref, ref, cv::Size(0, 0), 1.0);
        for (int stride : {2, 1}) {
            std::vector<Region> regs;
            for (auto& r : rects) {
                Region g;
                for (int yy = r[1]; yy < r[3]; yy += stride)
                    for (int xx = r[0]; xx < r[2]; xx += stride) {
                        g.x.push_back(xx);
                        g.y.push_back(yy);
                        std::array<double, 6> b;
                        basis(xx, yy, b.data());
                        g.B.push_back(b);
                        g.T.push_back(ref.at<double>(yy - y0, xx - x0));
                    }
                regs.push_back(g);
            }
            geom[stride] = regs;
        }
        for (auto& g : geom[1]) pixel_count += (int)g.x.size();
    }

    // im, gx, gy 의 스플라인 계수 (mirror)
    std::array<cv::Mat, 3> coefficients(const cv::Mat& current) const {
        cv::Mat im;
        current(cv::Range(y0, y1), cv::Range(x0, x1)).convertTo(im, CV_64F);
        cv::GaussianBlur(im, im, cv::Size(0, 0), 1.0);
        return {trk::spl::filter(im, false), trk::spl::filter(trk::np_gradient(im, 1), false), trk::spl::filter(trk::np_gradient(im, 0), false)};
    }

    struct Eval {
        double H[12][12], g[12], mse;
        std::array<std::array<double, 2>, 3> gains;
    };
    Eval evaluate(const std::array<cv::Mat, 3>& cf, const std::array<double, 12>& p, int stride, bool deriv) const {
        Eval e{};
        double sse = 0;
        long long count = 0;
        const auto& regs = geom.at(stride);
        for (size_t ri = 0; ri < regs.size(); ++ri) {
            const Region& g = regs[ri];
            const size_t n = g.x.size();
            std::vector<double> I(n), rowc(n), colc(n);
            for (size_t i = 0; i < n; ++i) {
                double dx = 0, dy = 0;
                for (int k = 0; k < 6; ++k) { dx += g.B[i][k] * p[2 * k]; dy += g.B[i][k] * p[2 * k + 1]; }
                rowc[i] = g.y[i] + dy - y0;
                colc[i] = g.x[i] + dx - x0;
                I[i] = trk::spl::eval(cf[0], rowc[i], colc[i]);
            }
            double mI = 0, mT = 0;
            for (size_t i = 0; i < n; ++i) { mI += I[i]; mT += g.T[i]; }
            mI /= n; mT /= n;
            double ii = 0, it = 0;
            for (size_t i = 0; i < n; ++i) { ii += (I[i] - mI) * (I[i] - mI); it += (I[i] - mI) * (g.T[i] - mT); }
            double denom = std::max(ii, 1e-12), a = it / denom, b = mT - a * mI;
            std::vector<double> r(n);
            for (size_t i = 0; i < n; ++i) { r[i] = g.T[i] - a * I[i] - b; sse += r[i] * r[i]; }
            count += (long long)n;
            e.gains[ri] = {a, b};
            if (!deriv) continue;
            std::vector<std::array<double, 12>> J(n);
            double mJ[12] = {0};
            for (size_t i = 0; i < n; ++i) {
                double Ix = trk::spl::eval(cf[1], rowc[i], colc[i]) * a, Iy = trk::spl::eval(cf[2], rowc[i], colc[i]) * a;
                for (int k = 0; k < 6; ++k) { J[i][2 * k] = Ix * g.B[i][k]; J[i][2 * k + 1] = Iy * g.B[i][k]; }
                for (int k = 0; k < 12; ++k) mJ[k] += J[i][k];
            }
            for (int k = 0; k < 12; ++k) mJ[k] /= n;
            double v[12] = {0}, JJ[12][12] = {{0}}, Jr[12] = {0}, icr = 0;
            for (size_t i = 0; i < n; ++i) {
                double ic = I[i] - mI, jc[12];
                for (int k = 0; k < 12; ++k) jc[k] = J[i][k] - mJ[k];
                for (int k = 0; k < 12; ++k) {
                    v[k] += jc[k] * ic;
                    Jr[k] += jc[k] * r[i];
                    for (int l = k; l < 12; ++l) JJ[k][l] += jc[k] * jc[l];
                }
                icr += ic * r[i];
            }
            for (int k = 0; k < 12; ++k) {
                for (int l = k; l < 12; ++l) {
                    double h = JJ[k][l] - v[k] * v[l] / denom;
                    e.H[k][l] += h;
                    if (l != k) e.H[l][k] += h;
                }
                e.g[k] += Jr[k] - v[k] * icr / denom;
            }
        }
        e.mse = sse / count;
        return e;
    }

    Result refine_coef(const std::array<cv::Mat, 3>& cf, const std::array<double, 12>& initial) const {
        Result res;
        res.initial = initial;
        auto p = initial;
        res.mse_initial = evaluate(cf, p, 1, false).mse;
        int niter = 0;
        double cond = std::nan("");
        for (auto sm : std::vector<std::pair<int, int>>{{2, 12}, {1, 10}}) {
            int stride = sm.first;
            for (int it = 0; it < sm.second; ++it) {
                Eval e = evaluate(cf, p, stride, true);
                cv::Mat H(12, 12, CV_64F), g(12, 1, CV_64F);
                for (int k = 0; k < 12; ++k) {
                    g.at<double>(k) = e.g[k];
                    for (int l = 0; l < 12; ++l) H.at<double>(k, l) = e.H[k][l];
                }
                cv::Mat w;
                cv::SVD::compute(H, w, cv::SVD::NO_UV);
                cond = w.at<double>(0) / w.at<double>(11);
                cv::Mat dpm;
                cv::solve(H + cv::Mat::eye(12, 12, CV_64F) * 1e-8, g, dpm, cv::DECOMP_LU);
                std::array<double, 12> dp;
                for (int k = 0; k < 12; ++k) dp[k] = dpm.at<double>(k);
                double maxdelta = 0;
                for (auto& rg : geom.at(stride))
                    for (auto& B : rg.B) {
                        double dx = 0, dy = 0;
                        for (int k = 0; k < 6; ++k) { dx += B[k] * dp[2 * k]; dy += B[k] * dp[2 * k + 1]; }
                        maxdelta = std::max(maxdelta, std::hypot(dx, dy));
                    }
                if (maxdelta > 0.5)
                    for (auto& v : dp) v *= 0.5 / maxdelta;
                bool accepted = false;
                double rate = 1;
                for (double rt : {1.0, 0.5, 0.25, 0.125}) {
                    rate = rt;
                    std::array<double, 12> cand;
                    for (int k = 0; k < 12; ++k) cand[k] = p[k] + rt * dp[k];
                    if (evaluate(cf, cand, stride, false).mse <= e.mse + 1e-10) { p = cand; accepted = true; break; }
                }
                ++niter;
                if (!accepted || maxdelta * rate < 2e-5) break;
            }
        }
        Eval f = evaluate(cf, p, 1, false);
        res.X = p;
        res.mse_final = f.mse;
        res.iterations = niter;
        res.condition = cond;
        res.gains = f.gains;
        return res;
    }
};

}  // namespace dr
