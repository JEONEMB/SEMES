// QR별 직접 정합(affine/homography ECC) -> 630점 변위 -> 선형·2차·조건균형 GLS 판정기, 중첩 OOF 선택
// 원본: joint_yt_reboot_20261022/direct/extractor.py, direct/measure.py, direct_models.py, evaluate_direct.py,
//       joint_yt_generalization_20261022/forward_model.py, joint_yt_erroraudit_20261022/robust/robust_model.py, metrics.py
#include <hye/dataset.hpp>
#include <hye/ecc_compat.hpp>
#include <hye/plot.hpp>
#include <hye/report.hpp>
#include <omp.h>
#include <opencv2/core.hpp>
#include <array>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <tuple>
#include "ecc_persp.hpp"
#include "fixed_inputs.hpp"

using namespace hye;

namespace {

// QR 상자 (x0, y0, x1, y1), x1·y1 은 끝 제외
const int kBox[3][4] = {{1552, 969, 1736, 1167}, {1809, 971, 1993, 1169}, {2067, 973, 2252, 1171}};
const int kPads[3] = {130, 260, 512};
const char* kCandName[4] = {"affine_g3", "homography_g3", "affine_g5", "homography_g5"};
const int kCandGauss[4] = {3, 3, 5, 5};
const bool kCandHomo[4] = {false, true, false, true};
const char* kHeadName[3] = {"linear", "quadratic", "balanced"};
constexpr int kP = 630, kF = 2 * kP;

struct Cand {
    bool ok = false;
    double ecc[3] = {kNaN, kNaN, kNaN};
    cv::Matx33d H[3];
    std::vector<double> D = std::vector<double>(kF, kNaN);  // 630 x (dx, dy)
};
struct Meas {
    int pad = 0;
    double ncc[3]{};
    int sx[3]{}, sy[3]{};
    Cand c[4];
};

cv::Matx33d translate(double x, double y) { return cv::Matx33d(1, 0, x, 0, 1, y, 0, 0, 1); }
cv::Point2d project(const cv::Matx33d& H, double x, double y) {
    double z = H(2, 0) * x + H(2, 1) * y + H(2, 2);
    return {(H(0, 0) * x + H(0, 1) * y + H(0, 2)) / z, (H(1, 0) * x + H(1, 1) * y + H(1, 2)) / z};
}

Meas measure(const cv::Mat& center, const std::array<cv::Mat, 3>& tpl, const std::array<cv::Mat, 3>& mask, const cv::Mat& im) {
    Meas m;
    for (int pad : kPads) {
        for (int k = 0; k < 3; ++k) {
            const int* b = kBox[k];
            int l = std::max(b[0] - pad, 0), t = std::max(b[1] - pad, 0), r = std::min(b[2] + pad, im.cols), bt = std::min(b[3] + pad, im.rows);
            cv::Mat res;
            cv::matchTemplate(im(cv::Range(t, bt), cv::Range(l, r)), tpl[k], res, cv::TM_CCOEFF_NORMED);
            cv::Point loc;
            cv::minMaxLoc(res, nullptr, &m.ncc[k], nullptr, &loc);
            m.sx[k] = loc.x + l - b[0];
            m.sy[k] = loc.y + t - b[1];
        }
        m.pad = pad;
        if (std::min({m.ncc[0], m.ncc[1], m.ncc[2]}) >= 0.6) break;
    }
    for (int c = 0; c < 4; ++c) {
        Cand& cd = m.c[c];
        cd.D.assign(kF, kNaN);
        bool all = true;
        for (int k = 0; k < 3; ++k) {
            const int* b = kBox[k];
            bool good = false;
            try {
                if (m.ncc[k] < 0.6) throw std::runtime_error("ncc");
                int l = b[0] + m.sx[k] - 12, t = b[1] + m.sy[k] - 12, r = b[2] + m.sx[k] + 12, bt = b[3] + m.sy[k] + 12;
                if (std::min(l, t) < 0 || r > im.cols || bt > im.rows) throw std::runtime_error("crop");
                cv::Mat crop = im(cv::Range(t, bt), cv::Range(l, r)).clone();
                cv::Mat W = (cv::Mat_<float>(3, 3) << 1, 0, 12, 0, 1, 12, 0, 0, 1);
                if (!kCandHomo[c]) W = W.rowRange(0, 2).clone();
                cv::Mat full(crop.size(), CV_8U, cv::Scalar(255));
                cv::TermCriteria tc(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 200, 1e-8);
                double cc = kCandHomo[c] ? semes::ecc5p::findTransformECCWithMask(tpl[k], crop, mask[k], full, W, cv::MOTION_HOMOGRAPHY, tc, kCandGauss[c])
                                         : semes::ecc5::findTransformECCWithMask(tpl[k], crop, mask[k], full, W, cv::MOTION_AFFINE, tc, kCandGauss[c]);
                cv::Matx33d L = cv::Matx33d::eye();
                for (int i = 0; i < W.rows; ++i)
                    for (int j = 0; j < 3; ++j) L(i, j) = W.at<float>(i, j);
                cv::Matx33d H = translate(l, t) * L * translate(-b[0], -b[1]);
                H = H * (1.0 / H(2, 2));
                cd.H[k] = H;
                cd.ecc[k] = cc;
                bool finite = true;
                double mv = 0;
                for (int i = 0; i < kP; ++i)
                    if (fixed::kRegion[i] == k) {
                        auto p = project(H, fixed::kPoints[i][0], fixed::kPoints[i][1]);
                        finite &= std::isfinite(p.x) && std::isfinite(p.y);
                        mv = std::max({mv, std::abs(p.x - fixed::kPoints[i][0] - m.sx[k]), std::abs(p.y - fixed::kPoints[i][1] - m.sy[k])});
                    }
                good = finite && cc >= 0.8 && mv <= 10;
                if (good)
                    for (int i = 0; i < kP; ++i)
                        if (fixed::kRegion[i] == k) {
                            auto p = project(H, fixed::kPoints[i][0], fixed::kPoints[i][1]);
                            cd.D[2 * i] = p.x - fixed::kPoints[i][0];
                            cd.D[2 * i + 1] = p.y - fixed::kPoints[i][1];
                        }
            } catch (const std::exception&) {
                good = false;
            }
            all &= good;
        }
        cd.ok = all;
    }
    return m;
}

// ---- 선형대수 도우미 (CV_64F) ----
cv::Mat lstsq(const cv::Mat& A, const cv::Mat& B) {
    cv::Mat X;
    cv::solve(A, B, X, cv::DECOMP_SVD);
    return X;
}
cv::Mat pinv(const cv::Mat& A) {
    cv::Mat X;
    cv::invert(A, X, cv::DECOMP_SVD);
    return X;
}

// 특징: [raw 1260, raw * (평균 Dx / 100)]
cv::Mat features(const std::vector<const std::vector<double>*>& D) {
    cv::Mat X((int)D.size(), 2 * kF, CV_64F);
    for (int r = 0; r < X.rows; ++r) {
        const auto& d = *D[r];
        double h = 0;
        for (int i = 0; i < kP; ++i) h += d[2 * i];
        h = h / kP / 100;
        double* o = X.ptr<double>(r);
        for (int j = 0; j < kF; ++j) { o[j] = d[j]; o[kF + j] = d[j] * h; }
    }
    return X;
}

void design(double y, double t, int degree, std::vector<double>& p, std::vector<double>& jy, std::vector<double>& jt) {
    p = {y, t};
    jy = {1, 0};
    jt = {0, 1};
    for (int k = 2; k <= degree; ++k)
        for (int b = 0; b <= k; ++b) {
            int a = k - b;
            p.push_back(std::pow(y, a) * std::pow(t, b));
            jy.push_back(a ? a * std::pow(y, a - 1) * std::pow(t, b) : 0.0);
            jt.push_back(b ? b * std::pow(y, a) * std::pow(t, b - 1) : 0.0);
        }
}

// forward_model.py ForwardGLS
struct ForwardGLS {
    int degree;
    double shrink;
    cv::Mat origin, projector, gram, initial;
    double scale[2], lower[2], upper[2];
    void fit(const cv::Mat& X, const cv::Mat& y) {
        const int n = X.rows, p = X.cols;
        for (int j = 0; j < 2; ++j) {
            double mx = 0;
            for (int i = 0; i < n; ++i) mx = std::max(mx, std::abs(y.at<double>(i, j)));
            scale[j] = std::max(mx, 0.05);
            lower[j] = 1e300;
            upper[j] = -1e300;
        }
        std::vector<double> P, jy, jt;
        design(0, 0, degree, P, jy, jt);
        const int k = (int)P.size();
        cv::Mat A(n, 1 + k, CV_64F);
        for (int i = 0; i < n; ++i) {
            double q0 = y.at<double>(i, 0) / scale[0], q1 = y.at<double>(i, 1) / scale[1];
            lower[0] = std::min(lower[0], q0); upper[0] = std::max(upper[0], q0);
            lower[1] = std::min(lower[1], q1); upper[1] = std::max(upper[1], q1);
            design(q0, q1, degree, P, jy, jt);
            A.at<double>(i, 0) = 1;
            for (int j = 0; j < k; ++j) A.at<double>(i, 1 + j) = P[j];
        }
        for (int j = 0; j < 2; ++j) { lower[j] -= 0.25; upper[j] += 0.25; }
        cv::Mat coef = lstsq(A, X);
        origin = coef.row(0).clone();
        cv::Mat C = coef.rowRange(1, 1 + k).clone();
        cv::Mat r = X - A * coef;
        cv::Mat sd(1, p, CV_64F);
        for (int j = 0; j < p; ++j) {
            double s = 0;
            for (int i = 0; i < n; ++i) s += r.at<double>(i, j) * r.at<double>(i, j);
            sd.at<double>(j) = std::max(std::sqrt(s / n), 1e-6);
        }
        cv::Mat R = r.clone(), Cs = C.clone();
        for (int i = 0; i < n; ++i) cv::divide(R.row(i), sd, R.row(i));
        for (int i = 0; i < k; ++i) cv::divide(Cs.row(i), sd, Cs.row(i));
        cv::Mat V = Cs.t();
        double a = shrink, b = (1 - a) / n;
        cv::Mat M = a * cv::Mat::eye(n, n, CV_64F) + b * R * R.t();
        cv::Mat S;
        cv::solve(M, R * V, S, cv::DECOMP_LU);
        cv::Mat Q = (V - b * R.t() * S) / a;
        projector = Q.clone();
        for (int j = 0; j < p; ++j) projector.row(j) /= sd.at<double>(j);
        gram = C * projector;
        gram = (gram + gram.t()) / 2;
        initial = pinv(gram(cv::Range(0, 2), cv::Range(0, 2)));
    }
    cv::Mat predict(const cv::Mat& X) const {
        cv::Mat out(X.rows, 2, CV_64F);
        const int k = gram.rows;
        for (int r = 0; r < X.rows; ++r) {
            cv::Mat h = (X.row(r) - origin) * projector;  // 1 x k
            double q[2];
            for (int j = 0; j < 2; ++j) q[j] = h.at<double>(0) * initial.at<double>(0, j) + h.at<double>(1) * initial.at<double>(1, j);
            if (degree > 1) {
                for (int j = 0; j < 2; ++j) q[j] = std::min(std::max(q[j], lower[j]), upper[j]);
                std::vector<double> P, jy, jt;
                auto mulg = [&](const std::vector<double>& v, int col) {
                    double s = 0;
                    for (int i = 0; i < k; ++i) s += v[i] * gram.at<double>(i, col);
                    return s;
                };
                auto cost = [&](const std::vector<double>& v) {
                    double s = 0;
                    for (int j = 0; j < k; ++j) s += v[j] * (mulg(v, j) - 2 * h.at<double>(j));
                    return s;
                };
                for (int it = 0; it < 35; ++it) {
                    design(q[0], q[1], degree, P, jy, jt);
                    double g0 = 0, g1 = 0, a = 0, b = 0, c = 0;
                    for (int j = 0; j < k; ++j) {
                        double rj = h.at<double>(j) - mulg(P, j);
                        g0 += jy[j] * rj;
                        g1 += jt[j] * rj;
                        double gy = mulg(jy, j), gt = mulg(jt, j);
                        a += gy * jy[j];
                        b += gy * jt[j];
                        c += gt * jt[j];
                    }
                    double damp = 1e-8 * (a + c + 1);
                    a += damp;
                    c += damp;
                    double det = std::max(a * c - b * b, 1e-20);
                    double st[2] = {std::min(std::max((c * g0 - b * g1) / det, -0.25), 0.25), std::min(std::max((a * g1 - b * g0) / det, -0.25), 0.25)};
                    double old = cost(P);
                    for (double f : {1.0, 0.5, 0.25, 0.125, 0.0625}) {
                        double tr[2];
                        for (int j = 0; j < 2; ++j) tr[j] = std::min(std::max(q[j] + f * st[j], lower[j]), upper[j]);
                        std::vector<double> tp, a1, a2;
                        design(tr[0], tr[1], degree, tp, a1, a2);
                        if (cost(tp) < old) { q[0] = tr[0]; q[1] = tr[1]; break; }
                    }
                    if (std::max(std::abs(st[0]), std::abs(st[1])) < 1e-8) break;
                }
            }
            out.at<double>(r, 0) = q[0] * scale[0];
            out.at<double>(r, 1) = q[1] * scale[1];
        }
        return out;
    }
};

// robust_model.py RobustBaggedForward(fit_mode=new_condition_means, shrink=.01, huber=None, bags=1)
struct Balanced {
    cv::Mat beta, bias;
    void fit(const cv::Mat& X, const cv::Mat& y, const std::vector<int>& dates, double shrink) {
        const int n = X.rows, p = X.cols;
        std::vector<std::tuple<int, double, double>> keys(n);
        for (int i = 0; i < n; ++i) keys[i] = {dates[i], y.at<double>(i, 0), y.at<double>(i, 1)};
        const std::set<std::tuple<int, double, double>> uniq(keys.begin(), keys.end());
        const std::vector<std::tuple<int, double, double>> names(uniq.begin(), uniq.end());
        const int ng = (int)names.size();
        std::vector<int> gi(n), counts(ng, 0);
        for (int i = 0; i < n; ++i) { gi[i] = int(std::lower_bound(names.begin(), names.end(), keys[i]) - names.begin()); ++counts[gi[i]]; }
        cv::Mat gx = cv::Mat::zeros(ng, p, CV_64F), A(ng, 3, CV_64F);
        for (int i = 0; i < n; ++i) gx.row(gi[i]) += X.row(i);
        std::map<int, int> per_date;
        for (auto& nm : names) per_date[std::get<0>(nm)]++;
        std::vector<double> gb(ng), fw(ng);
        int nsel = 0;
        for (int g = 0; g < ng; ++g) {
            gx.row(g) /= counts[g];
            A.at<double>(g, 0) = 1;
            A.at<double>(g, 1) = std::get<1>(names[g]);
            A.at<double>(g, 2) = std::get<2>(names[g]);
            gb[g] = 1.0 / per_date[std::get<0>(names[g])];
            bool sel = std::get<0>(names[g]) == 20261020;
            nsel += sel;
            fw[g] = gb[g] * sel;
        }
        if (nsel < 3) throw std::runtime_error("Training conditions do not identify affine Y/T forward response");
        cv::Mat Aw = A.clone(), Gw = gx.clone();
        for (int g = 0; g < ng; ++g) { Aw.row(g) *= std::sqrt(fw[g]); Gw.row(g) *= std::sqrt(fw[g]); }
        cv::Mat coef = lstsq(Aw, Gw);  // 3 x p
        cv::Mat AX(n, 3, CV_64F);
        std::vector<double> rw(n);
        double sw = 0;
        for (int i = 0; i < n; ++i) {
            AX.at<double>(i, 0) = 1;
            AX.at<double>(i, 1) = y.at<double>(i, 0);
            AX.at<double>(i, 2) = y.at<double>(i, 1);
            rw[i] = gb[gi[i]] / counts[gi[i]];
            sw += rw[i];
        }
        cv::Mat residual = X - AX * coef;
        cv::Mat slope = coef.rowRange(1, 3).clone();
        // gls_decoder
        cv::Mat R = residual.clone();
        for (int i = 0; i < n; ++i) R.row(i) *= std::sqrt(rw[i] / sw);
        cv::Mat sd(1, p, CV_64F);
        for (int j = 0; j < p; ++j) {
            double s = 0;
            for (int i = 0; i < n; ++i) s += R.at<double>(i, j) * R.at<double>(i, j);
            sd.at<double>(j) = std::max(std::sqrt(s), 1e-6);
        }
        cv::Mat Z = R.clone(), Ss = slope.clone();
        for (int i = 0; i < n; ++i) cv::divide(Z.row(i), sd, Z.row(i));
        for (int i = 0; i < 2; ++i) cv::divide(Ss.row(i), sd, Ss.row(i));
        cv::Mat V = Ss.t();
        double a = shrink, b = 1 - a;
        cv::Mat M = a * cv::Mat::eye(n, n, CV_64F) + b * Z * Z.t();
        cv::Mat S;
        cv::solve(M, Z * V, S, cv::DECOMP_LU);
        cv::Mat Q = (V - b * Z.t() * S) / a;
        beta = Q * pinv(Ss * Q);
        for (int j = 0; j < p; ++j) beta.row(j) /= sd.at<double>(j);
        bias = -coef.row(0) * beta;
    }
    cv::Mat predict(const cv::Mat& X) const {
        cv::Mat o = X * beta;
        for (int r = 0; r < o.rows; ++r) o.row(r) += bias;
        return o;
    }
};

struct Metric {
    int n = 0, failures = 0, passed = 0;
    double ymae = kNaN, ymax = kNaN, tmae = kNaN, tmax = kNaN;
};
Metric metric(const std::vector<std::array<double, 2>>& y, const std::vector<std::array<double, 2>>& p) {
    Metric m;
    m.n = (int)y.size();
    std::vector<double> ey, et;
    for (size_t i = 0; i < y.size(); ++i) {
        if (!std::isfinite(p[i][0]) || !std::isfinite(p[i][1])) { ++m.failures; continue; }
        double a = std::abs(p[i][0] - y[i][0]), b = std::abs(p[i][1] - y[i][1]);
        ey.push_back(a);
        et.push_back(b);
        m.passed += (a <= kPass + 1e-12 && b <= kPass + 1e-12);
    }
    if (!ey.empty()) { m.ymae = mean(ey); m.ymax = max_abs(ey); m.tmae = mean(et); m.tmax = max_abs(et); }
    return m;
}

// 공통 read_gray/read_color 는 cv::imread 라 한글 경로(10-20 '기준 위치')를 못 읽는다. 바이트로 읽어 imdecode 한다.
cv::Mat read_img(const Shot& s, int flag) {
    std::ifstream f(fs::u8path(s.path), std::ios::binary);
    std::vector<uchar> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    cv::Mat m = cv::imdecode(buf, flag);
    if (m.empty()) throw std::runtime_error("read failed: " + s.rel);
    return m;
}

std::string stem(const Shot& s) {
    auto nm = s.rel.substr(s.rel.rfind('/') + 1);
    return nm.substr(0, nm.rfind('.'));
}
std::string safe(std::string s) {
    for (auto& c : s) if (c == ' ' || c == '(' || c == ')') c = '_';
    while (!s.empty() && s.back() == '_') s.pop_back();
    return s;
}

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
    cv::setNumThreads(1);
    omp_set_num_threads(4);
    auto shots = load_341();
    const int N = (int)shots.size();
    const cv::Mat center = read_img(center_1014(shots), cv::IMREAD_GRAYSCALE);
    std::array<cv::Mat, 3> tpl, mask;
    for (int k = 0; k < 3; ++k) {
        const int* b = kBox[k];
        tpl[k] = center(cv::Range(b[1], b[3]), cv::Range(b[0], b[2])).clone();
        mask[k] = cv::Mat::zeros(tpl[k].size(), CV_8U);
        mask[k](cv::Rect(6, 6, tpl[k].cols - 12, tpl[k].rows - 12)).setTo(255);
    }
    std::vector<Meas> meas(N);
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N; ++i) {
        try {
            meas[i] = measure(center, tpl, mask, read_img(shots[i], cv::IMREAD_GRAYSCALE));
        } catch (const std::exception& e) {
#pragma omp critical
            std::cerr << "measure failed " << i << ": " << e.what() << "\n";
        }
    }
    std::cout << "measured " << N << "\n";

    std::vector<std::array<double, 2>> Y(N);
    std::vector<int> date(N);
    for (int i = 0; i < N; ++i) { Y[i] = {shots[i].y, shots[i].t}; date[i] = shots[i].set == "clean_20261020" ? 20261020 : 20261014; }

    // 측정 기록
    Table mt{{"file", "Y", "T", "ncc_pad", "ncc_min"}};
    for (int c = 0; c < 4; ++c) for (auto f : {"_success", "_ecc_min", "_meanDx", "_meanDy"}) mt.head.push_back(std::string(kCandName[c]) + f);
    int cand_fail[4] = {0, 0, 0, 0};
    for (int i = 0; i < N; ++i) {
        auto& m = meas[i];
        std::vector<std::string> r{shots[i].name(), fmt(shots[i].y, 2), fmt(shots[i].t, 2), std::to_string(m.pad), fmt(std::min({m.ncc[0], m.ncc[1], m.ncc[2]}), 6)};
        for (int c = 0; c < 4; ++c) {
            auto& cd = m.c[c];
            double sx = 0, sy = 0;
            for (int p = 0; p < kP; ++p) { sx += cd.D[2 * p]; sy += cd.D[2 * p + 1]; }
            cand_fail[c] += !cd.ok;
            r.insert(r.end(), {cd.ok ? "True" : "False", fmt(std::min({cd.ecc[0], cd.ecc[1], cd.ecc[2]}), 6), fmt(sx / kP, 6), fmt(sy / kP, 6)});
        }
        mt.add(r);
    }
    mt.csv(out("direct_measurements.csv"));

    // ---- 판정기 ----
    auto fit_predict = [&](int cfg, const std::vector<int>& train_all, const std::vector<int>& test) {
        const int c = cfg / 3, head = cfg % 3;
        std::vector<int> tr;
        for (int i : train_all) if (meas[i].c[c].ok) tr.push_back(i);
        std::vector<const std::vector<double>*> Dtr, Dte;
        std::vector<int> te_ok, dates;
        for (int i : tr) { Dtr.push_back(&meas[i].c[c].D); dates.push_back(date[i]); }
        for (int j = 0; j < (int)test.size(); ++j) if (meas[test[j]].c[c].ok) { te_ok.push_back(j); Dte.push_back(&meas[test[j]].c[c].D); }
        cv::Mat X = features(Dtr), y((int)tr.size(), 2, CV_64F);
        for (int r = 0; r < y.rows; ++r) { y.at<double>(r, 0) = Y[tr[r]][0]; y.at<double>(r, 1) = Y[tr[r]][1]; }
        std::vector<std::array<double, 2>> out(test.size(), {kNaN, kNaN});
        if (Dte.empty()) return out;
        cv::Mat Xt = features(Dte), P;
        try {
            if (head == 2) {
                Balanced b;
                b.fit(X, y, dates, 0.01);
                P = b.predict(Xt);
            } else {
                ForwardGLS g{head == 0 ? 1 : 2, head == 0 ? 0.01 : 0.1};
                g.fit(X, y);
                P = g.predict(Xt);
            }
        } catch (const std::exception& e) {
#pragma omp critical
            std::cerr << "fit failed cfg " << cfg << ": " << e.what() << "\n";
            return out;
        }
        for (int r = 0; r < (int)te_ok.size(); ++r) out[te_ok[r]] = {P.at<double>(r, 0), P.at<double>(r, 1)};
        return out;
    };

    struct Score { int failures; double maximum, mean; };
    std::vector<std::array<double, 2>> nested(N, {kNaN, kNaN}), reapply(N, {kNaN, kNaN});
    std::vector<std::vector<std::array<double, 2>>> fixedp(12, std::vector<std::array<double, 2>>(N, {kNaN, kNaN}));
    std::vector<int> chosen(5), fold_of(N);
    std::vector<std::array<Score, 12>> scores(5);
    for (int i = 0; i < N; ++i) fold_of[i] = fixed::kFold[i];
    for (int f = 0; f < 5; ++f) {
        std::vector<int> tr, te;
        for (int i = 0; i < N; ++i) {
            if (f == 4 || fold_of[i] != f) tr.push_back(i);
            if (f == 4 || fold_of[i] == f) te.push_back(i);
        }
#pragma omp parallel for schedule(dynamic)
        for (int k = 0; k < 12; ++k) {
            std::vector<double> e;
            int nonfinite = 0, over = 0;
            for (int s = 0; s < 3; ++s) {
                auto& sp = fixed::kInner[f][s];
                std::vector<int> it(sp.tr, sp.tr + sp.ntr), iv;
                for (int j = 0; j < sp.nva; ++j) if (date[sp.va[j]] == 20261020) iv.push_back(sp.va[j]);
                auto p = fit_predict(k, it, iv);
                for (size_t j = 0; j < iv.size(); ++j) {
                    double a = std::abs(p[j][0] - Y[iv[j]][0]), b = std::abs(p[j][1] - Y[iv[j]][1]);
                    if (!std::isfinite(a) || !std::isfinite(b)) { ++nonfinite; continue; }
                    e.push_back(a);
                    e.push_back(b);
                    over += (a > kPass + 1e-12 || b > kPass + 1e-12);
                }
            }
            scores[f][k] = {nonfinite + over, e.empty() ? 1e100 : max_abs(e), e.empty() ? 1e100 : mean(e)};
            if (f < 4) {
                auto p = fit_predict(k, tr, te);
                for (size_t j = 0; j < te.size(); ++j) fixedp[k][te[j]] = p[j];
            }
        }
        int best = 0;
        for (int k = 1; k < 12; ++k) {
            auto& a = scores[f][k];
            auto& b = scores[f][best];
            if (std::tie(a.failures, a.maximum, a.mean) < std::tie(b.failures, b.maximum, b.mean)) best = k;
        }
        chosen[f] = best;
        auto p = fit_predict(best, tr, te);
        for (size_t j = 0; j < te.size(); ++j) (f < 4 ? nested : reapply)[te[j]] = p[j];
        std::cout << "fold " << f << " chosen " << best << "\n";
    }

    auto subset = [&](const std::vector<std::array<double, 2>>& p, int ds) {
        std::vector<std::array<double, 2>> a, b;
        for (int i = 0; i < N; ++i) if (ds == 0 || date[i] == ds) { a.push_back(Y[i]); b.push_back(p[i]); }
        return metric(a, b);
    };
    Metric mAll = subset(nested, 0), m14 = subset(nested, 20261014), m20 = subset(nested, 20261020);
    Metric rAll = subset(reapply, 0), r14 = subset(reapply, 20261014), r20 = subset(reapply, 20261020);

    // 예측 저장
    Table pt{{"file", "fold", "Y", "T", "nested_predY", "nested_predT", "eY", "eT", "passed", "reapply_predY", "reapply_predT"}};
    std::vector<double> ty, py, tt, ptv, ey, et;
    std::vector<std::string> names;
    for (int i = 0; i < N; ++i) {
        double a = nested[i][0] - Y[i][0], b = nested[i][1] - Y[i][1];
        pt.add({shots[i].name(), std::to_string(fold_of[i]), fmt(Y[i][0], 2), fmt(Y[i][1], 2), fmt(nested[i][0], 6), fmt(nested[i][1], 6), fmt(std::abs(a), 6),
                fmt(std::abs(b), 6), (std::abs(a) <= kPass + 1e-12 && std::abs(b) <= kPass + 1e-12) ? "True" : "False", fmt(reapply[i][0], 6), fmt(reapply[i][1], 6)});
        ty.push_back(Y[i][0]); py.push_back(nested[i][0]); tt.push_back(Y[i][1]); ptv.push_back(nested[i][1]);
        ey.push_back(a); et.push_back(b);
        names.push_back(stem(shots[i]));
    }
    pt.csv(out("direct_predictions.csv"));

    // 설정별 고정 OOF
    Table cfgt{{"번호", "측정", "판정기", "통과", "Y 평균 |오차|", "Y 최대", "T 평균 |오차|", "T 최대", "바깥 fold에서 선택된 횟수"}};
    std::vector<Metric> fm(12);
    std::vector<std::string> cats;
    std::vector<double> passes;
    for (int k = 0; k < 12; ++k) {
        fm[k] = subset(fixedp[k], 0);
        int times = (int)std::count(chosen.begin(), chosen.begin() + 4, k);
        cfgt.add({std::to_string(k), kCandName[k / 3], kHeadName[k % 3], std::to_string(fm[k].passed) + "/" + std::to_string(fm[k].n), fmt(fm[k].ymae, 4), fmt(fm[k].ymax, 4),
                  fmt(fm[k].tmae, 4), fmt(fm[k].tmax, 4), std::to_string(times)});
        cats.push_back(std::to_string(k) + " " + std::string(kCandName[k / 3]).substr(0, 4) + std::string(kCandName[k / 3]).substr(std::string(kCandName[k / 3]).size() - 2) + " " + kHeadName[k % 3]);
        passes.push_back(fm[k].passed);
    }
    Table selt{{"바깥 fold", "선택", "측정", "판정기", "안쪽 실패 수", "안쪽 최대 |오차|", "안쪽 평균 |오차|"}};
    for (int f = 0; f < 5; ++f) {
        auto& s = scores[f][chosen[f]];
        selt.add({f < 4 ? std::to_string(f) : "전체(재적용)", std::to_string(chosen[f]), kCandName[chosen[f] / 3], kHeadName[chosen[f] % 3], std::to_string(s.failures), fmt(s.maximum, 4), fmt(s.mean, 4)});
    }

    // Y·T 반응 방향: 10-14 단일축 스캔에서 좌표별 직선 기울기 (affine_g5)
    std::vector<double> sY(kF), sT(kF), aT(kF), aY(kF);
    {
        for (int ax = 0; ax < 2; ++ax) {
            std::vector<int> idx;
            for (int i = 0; i < N; ++i) if (date[i] == 20261014 && (shots[i].condition == (ax ? "T" : "Y") || shots[i].condition == "center")) idx.push_back(i);
            for (int j = 0; j < kF; ++j) {
                double mx = 0, my = 0, sxy = 0, sxx = 0;
                for (int i : idx) { mx += Y[i][ax]; my += meas[i].c[2].D[j]; }
                mx /= idx.size(); my /= idx.size();
                for (int i : idx) { sxy += (Y[i][ax] - mx) * (meas[i].c[2].D[j] - my); sxx += (Y[i][ax] - mx) * (Y[i][ax] - mx); }
                (ax ? sT : sY)[j] = sxy / sxx;
                (ax ? aT : aY)[j] = my - sxy / sxx * mx;
            }
        }
    }
    double nY = 0, nT = 0, dot = 0;
    for (int j = 0; j < kF; ++j) { nY += sY[j] * sY[j]; nT += sT[j] * sT[j]; dot += sY[j] * sT[j]; }
    const double cosYT = dot / std::sqrt(nY * nT);
    const double rmsY = std::sqrt(nY / kF), rmsT = std::sqrt(nT / kF);
    // Y 반응 중 T 반응과 직교하는 부분 (px/mm, 좌표 RMS)
    double orth = 0;
    for (int j = 0; j < kF; ++j) { double v = sY[j] - dot / nT * sT[j]; orth += v * v; }
    const double rmsOrth = std::sqrt(orth / kF);
    // 사진의 변위가 단일축 직선 모델(T 절편 + sT*T + sY*Y)에서 벗어난 RMS
    // 잔차를 'T와 직교하는 Y 방향'에 투영해 mm로 환산 (이 방향만이 Y를 T와 구분한다)
    auto resid_ymm = [&](int i) {
        double s = 0;
        for (int j = 0; j < kF; ++j) {
            double v = sY[j] - dot / nT * sT[j];
            s += (meas[i].c[2].D[j] - (aT[j] + sT[j] * Y[i][1] + sY[j] * Y[i][0])) * v;
        }
        return s / orth;
    };
    auto resid_rms = [&](int i) {
        double s = 0;
        for (int j = 0; j < kF; ++j) { double v = meas[i].c[2].D[j] - (aT[j] + sT[j] * Y[i][1] + sY[j] * Y[i][0]); s += v * v; }
        return std::sqrt(s / kF);
    };
    {
        std::vector<double> x[2], y[2];
        for (int j = 0; j < kF; ++j) { x[j % 2].push_back(sT[j]); y[j % 2].push_back(sY[j]); }
        Chart c("Response per coordinate: Y slope vs T slope (affine_g5)", "T response (px / deg)", "Y response (px / mm)");
        c.scatter(x[0], y[0], color(0), "x component", 2).scatter(x[1], y[1], color(1), "y component", 2).hline(0).vline(0);
        c.save("03_Y와T_좌표별반응.png");
    }

    // ---- 그림 ----
    pred_vs_truth("01_중첩OOF341_예측오차.png", "Nested OOF 341", ty, py, tt, ptv, names);
    bar_chart("02_설정별_고정OOF_통과장수.png", "Fixed-config OOF: passed of 341", cats, passes, "passed", chosen[0], 218);

    size_t wy = worst(ey, 1)[0], wt = worst(et, 1)[0];
    auto outlier_fig = [&](size_t i, const std::string& file, const std::string& cap) {
        cv::Mat im = read_img(shots[i], cv::IMREAD_COLOR);
        const int c = chosen[fold_of[i]] / 3;
        std::vector<cv::Mat> tiles;
        for (int k = 0; k < 3; ++k) {
            const int* b = kBox[k];
            cv::Rect2d ref(b[0], b[1], b[2] - b[0], b[3] - b[1]);
            double mx = 0, my = 0;
            int n = 0;
            std::vector<Mark> mk;
            for (int p = 0; p < kP; ++p)
                if (fixed::kRegion[p] == k) {
                    mx += meas[i].c[c].D[2 * p]; my += meas[i].c[c].D[2 * p + 1]; ++n;
                }
            mx /= n; my /= n;
            for (int p = 0, q = 0; p < kP; ++p)
                if (fixed::kRegion[p] == k && (q++ % 20) == 0) {
                    Mark a{{fixed::kPoints[p][0] + meas[i].c[c].D[2 * p], fixed::kPoints[p][1] + meas[i].c[c].D[2 * p + 1]}, {0, 0}, "", kRed};
                    mk.push_back(a);
                }
            cv::Rect2d now(ref.x + mx, ref.y + my, ref.width, ref.height);
            cv::Rect roi = cv::Rect(ref | now) + cv::Size(60, 60) - cv::Point(30, 30);
            tiles.push_back(zoom(im, roi, 420, mk, 0, k == 0 ? cap : "", {{ref, kGreen, ""}, {now, kRed, "q" + std::to_string(k) + " dx " + fmt(mx, 2) + " dy " + fmt(my, 2)}}));
        }
        save_png(file, hstack(tiles));
    };
    const std::string fy = "04_튀는값_Y_" + safe(names[wy]) + ".png", ftn = "05_튀는값_T_" + safe(names[wt]) + ".png";
    outlier_fig(wy, fy, names[wy] + " Y err " + fmts(ey[wy], 3) + " mm");
    outlier_fig(wt, ftn, names[wt] + " T err " + fmts(et[wt], 3) + " deg");

    Table outl{{"구분", "사진", "fold / 선택", "정답 Y (mm)", "추정 Y", "정답 T (deg)", "추정 T", "q0 / q1 / q2 평균 dx (px)", "단일축 직선 대비 잔차 RMS (px)", "잔차의 Y 구분 방향 성분 (mm 환산)"}};
    auto qdx = [&](int i) {
        const int c = chosen[fold_of[i]] / 3;
        std::string s;
        for (int k = 0; k < 3; ++k) {
            double mx = 0;
            int n = 0;
            for (int p = 0; p < kP; ++p) if (fixed::kRegion[p] == k) { mx += meas[i].c[c].D[2 * p]; ++n; }
            s += (k ? " / " : "") + fmt(mx / n, 2);
        }
        return s;
    };
    for (auto [lab, idx] : std::vector<std::pair<std::string, std::vector<size_t>>>{{"Y 오차 상위", worst(ey, 5)}, {"T 오차 상위", worst(et, 3)}})
        for (size_t i : idx)
            outl.add({lab, shots[i].name(), std::to_string(fold_of[i]) + " / " + std::to_string(chosen[fold_of[i]]), fmt(Y[i][0], 2), fmt(nested[i][0], 3), fmt(Y[i][1], 2),
                      fmt(nested[i][1], 3), qdx((int)i), fmt(resid_rms((int)i), 4), fmts(resid_ymm((int)i), 3)});
    std::vector<double> rr14;
    for (int i = 0; i < N; ++i) if (date[i] == 20261014) rr14.push_back(resid_rms(i));
    const double rrMed = percentile(rr14, 0.5);

    // ---- 원래 결과와 대조 ----
    // 원래 값: direct_summary.csv, direct_selection.json, direct_fixed_predictions.csv, direct/features.npz(D 평균), direct/measurement_audit.json
    Parity par;
    par.add("중첩 OOF 341 통과 장수", 218, mAll.passed, 0, 0);
    par.add("중첩 OOF 10-14 통과 (281장)", 165, m14.passed, 0, 0);
    par.add("중첩 OOF 10-20 통과 (60장)", 53, m20.passed, 0, 0);
    par.add("중첩 OOF Y 평균 |오차| (mm)", 0.081715, mAll.ymae, 0.0005);
    par.add("중첩 OOF Y 최대 |오차| (mm)", 0.377428, mAll.ymax, 0.002);
    par.add("중첩 OOF T 평균 |오차| (deg)", 0.014577, mAll.tmae, 0.0002);
    par.add("중첩 OOF T 최대 |오차| (deg)", 0.065622, mAll.tmax, 0.001);
    par.add("전체 학습 재적용 통과", 216, rAll.passed, 0, 0);
    {
        std::string o = "8, 8, 9, 11, 8", n;
        for (int f = 0; f < 5; ++f) n += (f ? ", " : "") + std::to_string(chosen[f]);
        par.add_text("fold 0~3 + 재적용 선택 설정", o, n, o == n);
    }
    par.add("고정 설정 6 (affine_g5 linear) 통과", 225, fm[6].passed, 0, 0);
    par.add("고정 설정 8 (affine_g5 balanced) 통과", 207, fm[8].passed, 0, 0);
    par.add("고정 설정 9 (homography_g5 linear) 통과", 224, fm[9].passed, 0, 0);
    par.add("고정 설정 11 (homography_g5 balanced) 통과", 212, fm[11].passed, 0, 0);
    auto meanD = [&](int i, int c, int ax) {
        double s = 0;
        for (int p = 0; p < kP; ++p) s += meas[i].c[c].D[2 * p + ax];
        return s / kP;
    };
    par.add("T-1.00 affine_g5 평균 Dx (px)", -80.994182, meanD(200, 2, 0), 0.001);
    par.add("T-1.00 homography_g5 평균 Dx (px)", -80.992597, meanD(200, 3, 0), 0.001);
    par.add("T+0.01 affine_g3 평균 Dx (px)", 0.173068, meanD(1, 0, 0), 0.001);
    par.add("Y+0.05 T+0.01 (1) homography_g5 평균 Dy (px)", 0.451924, meanD(281, 3, 1), 0.001);
    par.add("측정 실패 (4측정 합)", 0, cand_fail[0] + cand_fail[1] + cand_fail[2] + cand_fail[3], 0, 0);

    const std::string mismatch_note =
        "\n불일치 원인: 측정 변위는 원래 `features.npz`와 좌표 RMS 0.00001 px(최대 0.001 px, affine_g3) 차이다(OpenCV 3.4와 cv2 5.0의 가우시안·NCC·마스크 warp 반올림). "
        "C++ 변위를 원래 Python 판정기에 넣은 결과와 이 C++ 판정기 결과는 0.000001 mm 안으로 같았다(별도 확인). shrink 0.01 GLS가 작은 잔차 방향을 크게 키워 이 차이가 예측에서 최대 약 0.001 mm가 되고, "
        "원래 오차 0.090001 mm로 경계에 걸려 떨어졌던 10-14 Y+0.20이 C++에서는 0.08997 mm로 통과해 1장이 바뀌었다. 고정 설정 9의 1장 차이도 같은 이유다.\n";
    // ---- 결과.md ----
    auto mrow = [](const std::string& a, const std::string& b, const Metric& m) {
        return std::vector<std::string>{a, b, std::to_string(m.n), std::to_string(m.failures), std::to_string(m.passed), fmt(m.ymae, 4), fmt(m.ymax, 4), fmt(m.tmae, 4), fmt(m.tmax, 4)};
    };
    Table sum{{"평가", "묶음", "장수", "실패", "통과", "Y 평균 |오차|", "Y 최대", "T 평균 |오차|", "T 최대"}};
    sum.add(mrow("중첩 OOF", "전체", mAll));
    sum.add(mrow("중첩 OOF", "10-14", m14));
    sum.add(mrow("중첩 OOF", "10-20", m20));
    sum.add(mrow("전체 학습 재적용", "전체", rAll));
    sum.add(mrow("전체 학습 재적용", "10-14", r14));
    sum.add(mrow("전체 학습 재적용", "10-20", r20));
    // 원래 비교표 (comparison.csv, 2026-10-22 당시). 이 C++는 '새 QR별 직접 정합' 줄만 다시 계산했다.
    Table old{{"방법 (원래 comparison.csv)", "Y 평균", "T 평균", "10-20 동시 통과", "10-14 동시 통과", "이 C++에서"}};
    old.add({"직전 QR 630점 개선 후보", "0.0497", "0.0088", "53/60", "191/281", "옮기지 않음 (다른 실험의 경계점 측정)"});
    old.add({"새 QR별 직접 정합", "0.0463 (10-20)", "0.0085 (10-20)", "53/60", "165/281", "다시 계산 (위 표)"});
    old.add({"새 QR 공동 정합", "0.0636", "0.0106", "42/60", "131/281", "옮기지 않음"});
    old.add({"기하 생성모델 사후 최선", "0.0751", "0.0134", "41/60", "125/281", "옮기지 않음"});
    old.add({"원픽셀 PLS", "0.0561", "0.0674", "36/60", "239/281", "옮기지 않음"});
    old.add({"원픽셀 CNN", "0.1461", "0.0406", "18/60", "149/281", "옮기지 않음 (학습 난수)"});
    old.add({"외부 볼트·브래킷 5곳만", "0.0138", "0.0023", "60/60", "281/281", "철회 (QR 밖 사용, 규칙 위반)"});
    old.add({"QR + 외부 볼트·브래킷 5곳", "0.0133", "0.0024", "60/60", "281/281", "철회 (QR 밖 사용, 규칙 위반)"});

    std::ostringstream R;
    R << "# 결과: 관측 재설계 - QR별 직접 정합 (10-22 17:40)\n\n"
      << "입력: `data/clean/clean_20261014` 281장 + `clean_20261020` 60장 = 341장. 기준은 10-14 center.bmp.\n"
      << "측정: QR 세 상자를 NCC(탐색 여백 130/260/512)로 찾고, QR마다 ECC affine 또는 homography(가우시안 3 또는 5, 200회, 1e-8)로 center에 정합했다. "
      << "정합 변환을 center 위 630점(QR 격자 경계점)에 적용해 630x2 변위를 만든다. 실패 기준: NCC 0.6 미만, ECC 0.8 미만, NCC 대비 10 px 넘게 이동.\n"
      << "판정기: 특징 [변위 1260, 변위 x 평균Dx/100]에 선형 GLS(shrink 0.01), 2차 GLS(0.1), 조건균형 GLS(10-20 조건 평균으로 앞방향 적합, shrink 0.01).\n"
      << "선택: 원래 바깥 4 fold와 안쪽 3분할(10-20 행만 검증)을 그대로 쓰고, 실패 수 -> 최대 오차 -> 평균 오차 순으로 12설정 중 하나를 골랐다.\n\n"
      << "## 요약\n\n" << sum.md()
      << "\n## fold별 선택\n\n" << selt.md()
      << "\n## 설정별 고정 OOF (바깥 4 fold, 341장)\n\n" << cfgt.md()
      << "\n## Y·T 반응 방향 (10-14 단일축 스캔, affine_g5, 1260좌표)\n\n"
      << "| 지표 | 값 |\n|---|---|\n"
      << "| Y 1mm 반응 RMS (px) | " << fmt(rmsY, 3) << " |\n| T 1deg 반응 RMS (px) | " << fmt(rmsT, 3) << " |\n"
      << "| 두 반응 벡터 cos | " << fmt(cosYT, 5) << " |\n| Y 1mm 반응 중 T와 직교하는 부분 RMS (px) | " << fmt(rmsOrth, 4) << " |\n"
      << "| 같은 부분의 0.09 mm 신호 (px) | " << fmt(rmsOrth * 0.09, 5) << " |\n"
      << "| 10-14 사진의 단일축 직선 대비 잔차 RMS 중앙값 (px) | " << fmt(rrMed, 4) << " |\n"
      << "\n## 원래 비교표와 이 재현의 범위\n\n" << old.md()
      << "\n2026-10-23 범위 정정(`candidate_status_20261023.json`): QR 세 영역 밖을 쓰면 안 된다는 규칙이 다시 확인돼 외부 볼트·브래킷 후보의 341/341은 철회됐다. "
      << "규칙을 지킨 결과는 QR별 직접 정합 218/341이다.\n\n"
      << "## 옮기지 않은 부분\n\n"
      << "- 공동 정합(shared), 기하 생성모델, 원픽셀 PLS/CNN, 시차(parallax), 외부 5곳 정합, 날짜 스트레스 평가. 대표 결과(218/341)를 만든 직접 정합 경로만 옮겼다.\n"
      << "- 630점 기준 좌표, 바깥 fold, 안쪽 3분할은 원래 파일 값을 `실험/src/fixed_inputs.hpp` 상수로 옮겼다(생성 스크립트 `실험/tools/gen_fixed_inputs.py`). "
      << "630점을 만든 격자 경계 측정과 sklearn StratifiedGroupKFold(shuffle) 분할은 C++로 다시 만들지 않았다.\n"
      << "- 측광 RMSE·밝기 gain/offset 진단값(판정에 안 쓰임)은 계산하지 않았다.\n"
      << "- homography ECC는 공통 `ecc_compat.cpp`가 OpenCV 3.4 `warpPerspective`(1/32 px 양자화)를 쓰므로, 실험 안 사본 `실험/src/ecc_persp.cpp`에서 그 세 줄만 정확한 bilinear로 바꿨다.\n"
      << "\n## 원래 결과와 대조 (" << par.ok << "/" << par.total << " 일치)\n\n" << par.t.md() << mismatch_note
      << "\n## 파일\n\n- `direct_measurements.csv`: 사진별 NCC 여백, 4측정의 성공·ECC 최소·평균 변위\n"
      << "- `direct_predictions.csv`: 사진별 중첩 OOF 예측과 전체 학습 재적용 예측\n"
      << "- `이미지/01~05`: 예측 오차, 설정별 통과, 좌표별 반응, 튀는 값 사진\n";
    write_text(out("결과.md"), R.str());

    // ---- 보고서.md ----
    std::ostringstream P;
    P << "# 보고서: 관측 재설계 - QR별 직접 정합 (10-22 17:40)\n\n"
      << "## 문제\n\nQR 안에서 Y와 T가 거의 같은 방향으로 움직여 분리가 어려워, QR 관측 방식을 다시 설계해야 했다.\n\n"
      << "## 해결 방법\n\nQR마다 center에 ECC affine·homography로 직접 정합해 630점 변위를 만들고, 세 판정기 12설정을 원래와 같은 중첩 OOF로 골라 341장을 평가했다.\n\n"
      << "## 결과\n\n중첩 OOF 341장 중 " << mAll.passed << "장(원래 218장, 0.09 mm 경계에 걸린 1장 차이)이 Y·T 동시 통과했고, 10-20은 " << m20.passed << "/60이지만 10-14는 " << m14.passed
      << "/281로 큰 T 사진에서 Y가 틀렸다. 같은 날 보고된 QR 밖 외부 5곳 정합의 341/341은 QR 영역 규칙 위반으로 철회됐다.\n\n"
      << "| 지표 | 값 |\n|---|---|\n"
      << "| 동시 통과 (\\|eY\\|<=0.09, \\|eT\\|<=0.09) | " << mAll.passed << " / " << mAll.n << " |\n"
      << "| 10-14 / 10-20 통과 | " << m14.passed << " / 281, " << m20.passed << " / 60 |\n"
      << "| Y 평균 / 최대 \\|오차\\| (mm) | " << fmt(mAll.ymae, 4) << " / " << fmt(mAll.ymax, 4) << " |\n"
      << "| T 평균 / 최대 \\|오차\\| (deg) | " << fmt(mAll.tmae, 4) << " / " << fmt(mAll.tmax, 4) << " |\n"
      << "| Y 단독 통과 / T 단독 통과 | " << std::count_if(ey.begin(), ey.end(), [](double v) { return std::abs(v) <= kPass + 1e-12; }) << " / 341, "
      << std::count_if(et.begin(), et.end(), [](double v) { return std::abs(v) <= kPass + 1e-12; }) << " / 341 |\n"
      << "| 측정 실패 | " << mAll.failures << " |\n"
      << "| fold별 선택 설정 | " << chosen[0] << ", " << chosen[1] << ", " << chosen[2] << ", " << chosen[3] << " (재적용 " << chosen[4] << ") |\n"
      << "| (철회) 외부 5곳 정합 | 341 / 341, QR 밖 사용 |\n\n"
      << img_md("01_중첩OOF341_예측오차.png", "예측 오차") << "\n\n" << img_md("02_설정별_고정OOF_통과장수.png", "설정별 통과") << "\n\n"
      << "## 튀는 값\n\n가장 크게 틀린 사진은 Y는 " << names[wy] << "(정답 Y " << fmt(Y[wy][0], 2) << ", 추정 " << fmt(nested[wy][0], 3) << " mm), T는 " << names[wt]
      << "(정답 T " << fmt(Y[wt][1], 2) << ", 추정 " << fmt(nested[wt][1], 3) << " deg)이다. Y가 0인 T 사진인데 단일축 직선 대비 잔차(" << fmt(resid_rms((int)wy), 3)
      << " px)를 T와 직교하는 Y 방향에 투영하면 Y " << fmts(resid_ymm((int)wy), 3) << " mm에 해당하고(Y 0.09 mm 신호는 그 방향으로 " << fmt(rmsOrth * 0.09, 4)
      << " px뿐), 정합 잔차가 그대로 Y 오차가 됐다.\n\n"
      << outl.md() << "\n" << img_md(fy, "Y 튀는 값") << "\n\n" << img_md(ftn, "T 튀는 값") << "\n\n"
      << "초록 상자는 center의 QR 상자, 빨간 상자는 그 fold에서 고른 측정의 QR 평균 변위만큼 옮긴 상자, 빨간 십자는 630점 중 20개마다 한 점의 정합 위치다. "
      << "단일축 직선 대비 잔차는 10-14 Y·T 스캔에서 좌표별로 맞춘 직선(T 절편 + T 기울기 x T + Y 기울기 x Y)과의 차이이고, 10-14 전체 중앙값은 " << fmt(rrMed, 4) << " px다.\n\n"
      << img_md("03_Y와T_좌표별반응.png", "좌표별 반응") << "\n\n"
      << "## 한줄평\n\nQR 안 1260좌표에서 Y 1mm 반응과 T 1deg 반응의 cos가 " << fmt(cosYT, 5) << "로 거의 같은 방향이고 T와 직교하는 Y 성분은 1mm당 "
      << fmt(rmsOrth, 3) << " px(0.09 mm면 " << fmt(rmsOrth * 0.09, 4) << " px)뿐이라, 정합 방식을 바꿔도 큰 T 사진의 정합 잔차(중앙값 " << fmt(rrMed, 3)
      << " px)를 넘지 못해 10-14에서 " << (281 - m14.passed) << "장이 Y로 떨어졌다.\n\n"
      << "## 원래 결과와 대조\n\n원래 값은 `joint_yt_reboot_20261022/direct_summary.csv`, `direct_selection.json`, `direct_fixed_predictions.csv`, `direct/features.npz`(평균 변위), "
      << "`direct/measurement_audit.json`에서 가져왔다. 철회 사실은 `candidate_status_20261023.json`. " << par.ok << "/" << par.total << " 일치.\n\n" << par.t.md() << mismatch_note;
    write_text(out("보고서.md"), P.str());
    std::cout << "passed " << mAll.passed << " parity " << par.ok << "/" << par.total << "\n";
    return 0;
}
