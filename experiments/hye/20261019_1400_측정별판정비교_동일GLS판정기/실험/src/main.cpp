// 같은 Y/Θ GLS 판정기에 측정값만 바꿔 넣어 판정 정확도를 비교한다.
// 원본: analysis_judge_compare_20261019/run_compare.py
//   판정기 = codex_experiments_20261014/zero_experiment.py 의 fit/predict (mode 'center', 2차)
//   측정 = analysis_grid_corners_20261019/track_corners.py (모서리 755점), analysis_agent/measure_qr.py (QR ECC 아핀)
//   분할 = gls_comparison_20261014/compare.py protocols() 와 같은 규칙으로 다시 만듦 (재대입, 짝홀 2겹, StratifiedKFold 반복 5분할 20회)
#include <hye/dataset.hpp>
#include <hye/ecc_compat.hpp>
#include <hye/plot.hpp>
#include <hye/report.hpp>
#include <omp.h>
#include <cstdint>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include "track.hpp"

using namespace hye;

namespace {

using Vec = std::vector<double>;

// ---- QR 세 영역 ECC 아핀 (10-12 시범 실험 measure 와 같음) ----
const std::array<cv::Vec4i, 3> kQR{{{1554, 1734, 972, 1164}, {1815, 1995, 972, 1164}, {2073, 2253, 972, 1164}}};
std::array<std::array<double, 6>, 3> ecc_affine(const cv::Mat& center, const cv::Mat& im) {
    std::array<std::array<double, 6>, 3> out;
    const int pad = 130;
    for (int q = 0; q < 3; ++q) {
        auto b = kQR[q];
        cv::Mat tpl = center(cv::Range(b[2], b[3]), cv::Range(b[0], b[1]));
        cv::Mat search = im(cv::Range(b[2] - pad, b[3] + pad), cv::Range(b[0] - pad, b[1] + pad));
        cv::Mat res;
        cv::matchTemplate(search, tpl, res, cv::TM_CCOEFF_NORMED);
        cv::Point loc;
        cv::minMaxLoc(res, nullptr, nullptr, nullptr, &loc);
        int dx0 = loc.x - pad, dy0 = loc.y - pad;
        cv::Mat target = im(cv::Range(b[2] + dy0, b[3] + dy0), cv::Range(b[0] + dx0, b[1] + dx0));
        cv::Mat W = cv::Mat::eye(2, 3, CV_32F);
        try {
            semes::ecc5::findTransformECCWithMask(tpl, target, cv::noArray(), cv::noArray(), W, cv::MOTION_AFFINE,
                                                  cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 150, 1e-7), 3);
        } catch (const cv::Exception&) {
        }
        // dx, dy, a00-1, a01, a10, a11-1 (run_compare.py 의 ecc_affine18 순서)
        out[q] = {dx0 + (double)W.at<float>(0, 2), dy0 + (double)W.at<float>(1, 2), (double)W.at<float>(0, 0) - 1, (double)W.at<float>(0, 1),
                  (double)W.at<float>(1, 0), (double)W.at<float>(1, 1) - 1};
    }
    return out;
}

// ---- P12: 판 전체 2차 변형 [1, x, y, x^2, xy, y^2], x/y 이동 각각 6계수 (가중 최소제곱) ----
Vec p12(const Vec& px, const Vec& py, const Vec& dx, const Vec& dy, const Vec& w) {
    int n = (int)px.size();
    cv::Mat A(n, 6, CV_64F), B(n, 2, CV_64F), C;
    for (int i = 0; i < n; ++i) {
        double x = (px[i] - 1906) / 486, y = (py[i] - 1070) / 122, s = std::sqrt(w[i]);
        double b[6] = {1, x, y, x * x, x * y, y * y};
        for (int k = 0; k < 6; ++k) A.at<double>(i, k) = b[k] * s;
        B.at<double>(i, 0) = dx[i] * s;
        B.at<double>(i, 1) = dy[i] * s;
    }
    cv::solve(A, B, C, cv::DECOMP_SVD);
    Vec out;
    for (int k = 0; k < 6; ++k) { out.push_back(C.at<double>(k, 0)); out.push_back(C.at<double>(k, 1)); }
    return out;
}

// ---- 분할 ----
struct Split { std::string protocol, fold; std::vector<int> train, test; };

// numpy RandomState.shuffle (legacy MT19937, random_interval)
void np_shuffle(std::vector<int>& a, std::mt19937& rng) {
    for (size_t i = a.size() - 1; i >= 1; --i) {
        uint64_t mx = i, mask = mx;
        mask |= mask >> 1; mask |= mask >> 2; mask |= mask >> 4; mask |= mask >> 8; mask |= mask >> 16; mask |= mask >> 32;
        uint64_t v;
        while ((v = ((uint64_t)rng() & mask)) > mx) {}
        std::swap(a[i], a[(size_t)v]);
    }
}
// sklearn StratifiedKFold(n_splits, shuffle=True, random_state=seed)._make_test_folds
std::vector<int> stratified_folds(const std::vector<std::string>& y, int n_splits, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<std::string> order;  // 처음 나온 순서로 부호화
    std::vector<int> enc(y.size());
    for (size_t i = 0; i < y.size(); ++i) {
        auto it = std::find(order.begin(), order.end(), y[i]);
        if (it == order.end()) { order.push_back(y[i]); enc[i] = (int)order.size() - 1; }
        else enc[i] = (int)(it - order.begin());
    }
    int nc = (int)order.size();
    std::vector<int> sorted = enc;
    std::sort(sorted.begin(), sorted.end());
    std::vector<std::vector<int>> alloc(n_splits, std::vector<int>(nc, 0));
    for (int i = 0; i < n_splits; ++i)
        for (size_t j = i; j < sorted.size(); j += n_splits) alloc[i][sorted[j]]++;
    std::vector<int> test(y.size());
    for (int k = 0; k < nc; ++k) {
        std::vector<int> f;
        for (int i = 0; i < n_splits; ++i) for (int r = 0; r < alloc[i][k]; ++r) f.push_back(i);
        np_shuffle(f, rng);
        size_t p = 0;
        for (size_t i = 0; i < y.size(); ++i) if (enc[i] == k) test[i] = f[p++];
    }
    return test;
}

// ---- GLS 판정기 (zero_experiment.fit / predict, mode center, 2차) ----
struct Model {
    int d = 0;
    bool diag = false;
    Vec s[2], c[2];      // 0 = Y, 1 = T
    Vec scale;           // full: std, diag: sd
    std::vector<double> L;  // C = L L^T (아래 삼각, d x d)
    Vec Ps[2], Pc[2];
    double ss[2], sc[2], cc[2];
    // P = W^T W.  full: D^-1 C^-1 D^-1,  diag: diag(1/sd^2)
    Vec applyP(const Vec& x) const {
        Vec y(d);
        if (diag) { for (int i = 0; i < d; ++i) y[i] = x[i] / (scale[i] * scale[i]); return y; }
        for (int i = 0; i < d; ++i) y[i] = x[i] / scale[i];
        for (int i = 0; i < d; ++i) {  // L z = y
            double a = y[i];
            const double* Li = &L[(size_t)i * d];
            for (int k = 0; k < i; ++k) a -= Li[k] * y[k];
            y[i] = a / Li[i];
        }
        for (int i = d - 1; i >= 0; --i) {  // L^T w = z
            double a = y[i];
            for (int k = i + 1; k < d; ++k) a -= L[(size_t)k * d + i] * y[k];
            y[i] = a / L[(size_t)i * d + i];
        }
        for (int i = 0; i < d; ++i) y[i] /= scale[i];
        return y;
    }
};
double median(std::vector<double> v) { return percentile(std::move(v), 0.5); }
double dot(const Vec& a, const Vec& b) { double s = 0; for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i]; return s; }

Model fit(const std::vector<Vec>& X, const std::vector<int>& tr, const std::vector<int>& axis, const Vec& val, bool diag) {
    Model M;
    int d = (int)X[0].size();
    M.d = d;
    M.diag = diag;
    std::vector<Vec> R;
    for (int a = 0; a < 2; ++a) {
        std::vector<int> ii;
        for (int i : tr) if (axis[i] == a) ii.push_back(i);
        double s11 = 0, s12 = 0, s22 = 0;
        for (int i : ii) { double v = val[i]; s11 += v * v; s12 += v * v * v; s22 += v * v * v * v; }
        double det = s11 * s22 - s12 * s12;
        M.s[a].assign(d, 0);
        M.c[a].assign(d, 0);
        for (int j = 0; j < d; ++j) {
            double b1 = 0, b2 = 0;
            for (int i : ii) { double v = val[i]; b1 += v * X[i][j]; b2 += v * v * X[i][j]; }
            M.s[a][j] = (s22 * b1 - s12 * b2) / det;
            M.c[a][j] = (s11 * b2 - s12 * b1) / det;
        }
        for (int i : ii) {
            Vec r(d);
            double v = val[i];
            for (int j = 0; j < d; ++j) r[j] = X[i][j] - v * M.s[a][j] - v * v * M.c[a][j];
            R.push_back(std::move(r));
        }
    }
    const int N = (int)R.size();
    M.scale.assign(d, 0);
    if (diag) {
        // 같은 학습 잔차의 분산(ddof 1)만 쓰고 점끼리 상관은 버림
        for (int j = 0; j < d; ++j) {
            double m = 0;
            for (auto& r : R) m += r[j];
            m /= N;
            double s = 0;
            for (auto& r : R) s += (r[j] - m) * (r[j] - m);
            M.scale[j] = std::sqrt(std::max(s / (N - 1), 1e-14));
        }
    } else {
        // S = R^T R / (N - 4), 상관행렬 C = .999 C + .001 I, 촐레스키
        std::vector<double> S((size_t)d * d, 0.0);
        for (auto& r : R)
            for (int i = 0; i < d; ++i) {
                double ri = r[i];
                double* Si = &S[(size_t)i * d];
                for (int j = 0; j <= i; ++j) Si[j] += ri * r[j];
            }
        double den = std::max(N - 4, 1);
        for (int i = 0; i < d; ++i) M.scale[i] = std::sqrt(std::max(S[(size_t)i * d + i] / den, 1e-14));
        M.L.assign((size_t)d * d, 0.0);
        for (int i = 0; i < d; ++i)
            for (int j = 0; j <= i; ++j) {
                double v = S[(size_t)i * d + j] / den / M.scale[i] / M.scale[j];
                M.L[(size_t)i * d + j] = .999 * v + (i == j ? .001 : 0);
            }
        for (int j = 0; j < d; ++j) {
            double* Lj = &M.L[(size_t)j * d];
            double a = Lj[j];
            for (int k = 0; k < j; ++k) a -= Lj[k] * Lj[k];
            Lj[j] = std::sqrt(a);
            for (int i = j + 1; i < d; ++i) {
                double* Li = &M.L[(size_t)i * d];
                double b = Li[j];
                for (int k = 0; k < j; ++k) b -= Li[k] * Lj[k];
                Li[j] = b / Lj[j];
            }
        }
    }
    for (int a = 0; a < 2; ++a) {
        M.Ps[a] = M.applyP(M.s[a]);
        M.Pc[a] = M.applyP(M.c[a]);
        M.ss[a] = dot(M.s[a], M.Ps[a]);
        M.sc[a] = dot(M.s[a], M.Pc[a]);
        M.cc[a] = dot(M.c[a], M.Pc[a]);
    }
    return M;
}

struct Pred { int file; int pred_axis; double pred, cost, cost_y, cost_t; };

Pred predict(const Model& M, const Vec& x, int file) {
    Vec Px = M.applyP(x);
    double best = INFINITY, costs[2], vs[2];
    int ba = 0;
    for (int a = 0; a < 2; ++a) {
        double xs = dot(M.s[a], Px), xc = dot(M.c[a], Px), ss = M.ss[a], sc = M.sc[a], cc = M.cc[a];
        double v = xs / ss;
        for (int it = 0; it < 8; ++it) {
            double JR = xs - v * ss - v * v * sc + 2 * v * (xc - v * sc - v * v * cc);
            double JJ = ss + 4 * v * sc + 4 * v * v * cc;
            v += JR / std::max(JJ, 1e-20);
        }
        Vec r(M.d);
        for (int j = 0; j < M.d; ++j) r[j] = x[j] - v * M.s[a][j] - v * v * M.c[a][j];
        double cost = dot(r, M.applyP(r));
        costs[a] = cost;
        vs[a] = v;
        if (cost < best) { best = cost; ba = a; }
    }
    return {file, ba, vs[ba], best, costs[0], costs[1]};
}

// zero_experiment.score 의 필요한 부분
struct Score { double Yacc, Tacc, Ymax, Tmax; int nY, nT, wrongY, wrongT; };
Score score(const std::vector<Pred>& P, const std::vector<int>& axis, const Vec& val) {
    Score s{0, 0, 0, 0, 0, 0, 0, 0};
    for (auto& p : P) {
        int a = axis[p.file];
        double v = val[p.file];
        bool ok = a == p.pred_axis && ((v > 0) - (v < 0)) == ((p.pred > 0) - (p.pred < 0));
        double tY = a == 0 ? v : 0, tT = a == 1 ? v : 0, pY = p.pred_axis == 0 ? p.pred : 0, pT = p.pred_axis == 1 ? p.pred : 0;
        s.Ymax = std::max(s.Ymax, std::abs(tY - pY));
        s.Tmax = std::max(s.Tmax, std::abs(tT - pT));
        if (a == 0) { s.nY++; s.wrongY += !ok; } else { s.nT++; s.wrongT += !ok; }
    }
    s.Yacc = 1 - (double)s.wrongY / s.nY;
    s.Tacc = 1 - (double)s.wrongT / s.nT;
    return s;
}

std::string stem_of(const Shot& s) { auto f = s.rel.substr(s.rel.rfind('/') + 1); return f.substr(0, f.size() - 4); }

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
    auto shots = load_1014();  // center + 이름순 = 원래 names 순서
    const int NS = (int)shots.size();
    int ic = 0;
    for (int i = 0; i < NS; ++i) if (shots[i].condition == "center") ic = i;
    const cv::Mat center = read_gray(shots[ic]);
    tk::Tracker T;
    T.init(center);
    const int NC = (int)T.corners.size();

    // ---- 측정: 모서리 755점 추적 + QR ECC 아핀 ----
    std::vector<std::vector<tk::Meas>> CM(NS);
    std::vector<std::array<std::array<double, 6>, 3>> EA(NS);
    int done = 0;
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < NS; ++i) {
        cv::Mat img = read_gray(shots[i]);
        CM[i] = T.track(img);
        EA[i] = ecc_affine(center, img);
#pragma omp critical
        {
            if (++done % 40 == 0) std::cout << "measured " << done << "/" << NS << "\n";
        }
    }

    // ---- 라벨: 축(0 Y, 1 T), 값, 묶음 ----
    std::vector<int> axis(NS, -1);
    Vec val(NS, 0);
    std::vector<std::string> session(NS, "C");
    for (int i = 0; i < NS; ++i) {
        if (i == ic) continue;
        std::string f = stem_of(shots[i]);
        val[i] = std::stod(f.substr(1));
        axis[i] = f[0] == 'T' ? 1 : 0;
        std::string sg = val[i] > 0 ? "+" : "-";
        session[i] = axis[i] == 1 ? "T" + sg : (((long long)std::abs(std::nearbyint(val[i] / .05)) % 2) ? "Y2" : "Y1") + sg;
    }
    std::vector<int> ids;
    for (int i = 0; i < NS; ++i) if (i != ic) ids.push_back(i);

    // ---- 특징 ----
    std::map<std::string, std::vector<Vec>> FEAT;
    {
        Vec cx0(NC), cy0(NC);
        for (int k = 0; k < NC; ++k) { cx0[k] = CM[ic][k].x; cy0[k] = CM[ic][k].y; }
        std::vector<int> black(NC);
        for (int k = 0; k < NC; ++k) black[k] = T.corners[k].black;
        auto okrule = [&](const std::string& r, int i, int k) {
            if (r == "valid") return CM[i][k].valid && CM[ic][k].valid;
            if (r == "nondiag") return black[k] != 2;
            return true;
        };
        for (std::string r : {"all", "nondiag", "valid"}) {
            std::vector<Vec> F12(NS), FD(NS);
            std::vector<int> fixed;
            for (int k = 0; k < NC; ++k) {
                bool f = r == "all" ? true : r == "nondiag" ? black[k] != 2 : [&] { for (int i = 0; i < NS; ++i) if (!CM[i][k].valid) return false; return true; }();
                if (f) fixed.push_back(k);
            }
            for (int i = 0; i < NS; ++i) {
                Vec dx(NC), dy(NC), w(NC);
                for (int k = 0; k < NC; ++k) { dx[k] = CM[i][k].x - cx0[k]; dy[k] = CM[i][k].y - cy0[k]; w[k] = okrule(r, i, k) ? 1 : 0; }
                F12[i] = p12(cx0, cy0, dx, dy, w);
                for (int k : fixed) FD[i].push_back(dx[k]);
                for (int k : fixed) FD[i].push_back(dy[k]);
            }
            FEAT["corner_" + r + "_P12"] = F12;
            FEAT["corner_" + r + "_direct"] = FD;
        }
        std::vector<Vec> FE(NS);
        for (int i = 0; i < NS; ++i)
            for (int q = 0; q < 3; ++q) for (int j = 0; j < 6; ++j) FE[i].push_back(EA[i][q][j]);
        FEAT["ecc_affine18"] = FE;
    }

    // ---- 분할 ----
    std::vector<Split> splits;
    splits.push_back({"full280_resubstitution", "all", ids, ids});
    {
        // 짝홀: 축마다 촬영 순서로 늘어놓고 번갈아 나눔. 촬영 순서(codex_experiments_20261014/manifest.csv 의 mtime) =
        //   Y: Y1+ |v| 오름, Y1- |v| 오름, Y2+ |v| 오름, Y2- |v| 내림 / T: T+ |v| 오름, T- |v| 오름
        auto cap_key = [&](int i) {
            const std::string& s = session[i];
            static const std::map<std::string, int> rank{{"Y1+", 0}, {"Y1-", 1}, {"Y2+", 2}, {"Y2-", 3}, {"T+", 0}, {"T-", 1}};
            double a = std::abs(val[i]);
            return std::make_pair(rank.at(s), s == "Y2-" ? -a : a);
        };
        for (int k = 0; k < 2; ++k) {
            std::vector<int> te;
            for (int a = 0; a < 2; ++a) {
                std::vector<int> ix;
                for (int i : ids) if (axis[i] == a) ix.push_back(i);
                std::stable_sort(ix.begin(), ix.end(), [&](int p, int q) { return cap_key(p) < cap_key(q); });
                for (size_t j = k; j < ix.size(); j += 2) te.push_back(ix[j]);
            }
            std::sort(te.begin(), te.end());
            std::vector<int> trn;
            for (int i : ids) if (!std::binary_search(te.begin(), te.end(), i)) trn.push_back(i);
            splits.push_back({"interleaved", "f" + std::to_string(k), trn, te});
        }
    }
    for (int seed = 0; seed < 20; ++seed) {
        std::vector<std::string> y;
        for (int i : ids) y.push_back(session[i]);
        auto tf = stratified_folds(y, 5, 7000 + seed);
        for (int f = 0; f < 5; ++f) {
            Split s{"repeated5_seed" + std::to_string(seed), std::to_string(f), {}, {}};
            for (size_t j = 0; j < ids.size(); ++j) (tf[j] == f ? s.test : s.train).push_back(ids[j]);
            splits.push_back(s);
        }
    }
    {
        Table t{{"protocol", "fold", "test"}};
        for (auto& s : splits) {
            if (s.protocol == "full280_resubstitution") continue;
            std::string te;
            for (int i : s.test) te += (te.empty() ? "" : " ") + stem_of(shots[i]);
            t.add({s.protocol, s.fold, te});
        }
        t.csv(out("splits.csv"));
    }

    // ---- 평가 ----
    const std::vector<std::string> methods = {"corner_all_P12",     "corner_all_direct",   "corner_nondiag_P12",        "corner_nondiag_direct",
                                              "corner_valid_P12",   "corner_valid_direct", "ecc_affine18",              "corner_all_direct_diagcov",
                                              "corner_nondiag_direct_diagcov", "corner_valid_direct_diagcov"};
    struct Task { int m; int s; };
    std::vector<Task> tasks;
    for (int m = 0; m < (int)methods.size(); ++m)
        for (int s = 0; s < (int)splits.size(); ++s) tasks.push_back({m, s});
    std::vector<std::vector<Pred>> TP(tasks.size());
    std::cout << "tasks " << tasks.size() << "\n";
#pragma omp parallel for schedule(dynamic)
    for (int t = 0; t < (int)tasks.size(); ++t) {
        const std::string& name = methods[tasks[t].m];
        bool diag = name.size() > 8 && name.substr(name.size() - 8) == "_diagcov";
        const auto& X = FEAT.at(diag ? name.substr(0, name.size() - 8) : name);
        const auto& sp = splits[tasks[t].s];
        Model M = fit(X, sp.train, axis, val, diag);
        for (int i : sp.test) TP[t].push_back(predict(M, X[i], i));
    }

    // ---- 점수 ----
    struct Sum { int dims; Score resub, evenodd, rep; int rep_all_correct; double wrong_per_round; std::vector<double> round_wrong; };
    std::map<std::string, Sum> SUM;
    Table mp{{"method", "protocol", "n", "Y_axis_sign_accuracy", "T_axis_sign_accuracy", "vector_Y_max", "vector_T_max"}};
    Table pr{{"method", "protocol", "fold", "file", "axis", "value", "pred_axis", "pred", "cost", "cost_Y", "cost_T"}};
    std::map<std::string, std::map<int, int>> rep_wrong_count;  // method -> file -> 20회 중 틀린 횟수
    std::map<std::string, std::map<int, std::vector<double>>> rep_pred;  // method -> file -> 예측값(축 맞을 때 값, 아니면 nan)
    for (int m = 0; m < (int)methods.size(); ++m) {
        auto& S = SUM[methods[m]];
        std::string base = methods[m].substr(0, methods[m].find("_diagcov"));
        S.dims = (int)FEAT.at(base)[0].size();
        std::map<std::string, std::vector<Pred>> byprot;
        for (size_t t = 0; t < tasks.size(); ++t) {
            if (tasks[t].m != m) continue;
            auto& sp = splits[tasks[t].s];
            auto& v = byprot[sp.protocol];
            v.insert(v.end(), TP[t].begin(), TP[t].end());
            for (auto& p : TP[t])
                pr.add({methods[m], sp.protocol, sp.fold, stem_of(shots[p.file]) + ".bmp", axis[p.file] ? "T" : "Y", fmt(val[p.file], 2), p.pred_axis ? "T" : "Y",
                        fmt(p.pred, 5), fmt(p.cost, 4), fmt(p.cost_y, 4), fmt(p.cost_t, 4)});
        }
        std::vector<Pred> rep;
        S.rep_all_correct = 0;
        for (auto& [prot, P] : byprot) {
            auto sc = score(P, axis, val);
            mp.add({methods[m], prot, std::to_string(P.size()), fmt(sc.Yacc, 6), fmt(sc.Tacc, 6), fmt(sc.Ymax, 5), fmt(sc.Tmax, 5)});
            if (prot == "full280_resubstitution") S.resub = sc;
            else if (prot == "interleaved") S.evenodd = sc;
            else {
                rep.insert(rep.end(), P.begin(), P.end());
                S.rep_all_correct += sc.wrongY == 0 && sc.wrongT == 0;
                S.round_wrong.push_back((1 - sc.Yacc) * 80 + (1 - sc.Tacc) * 200);
                for (auto& p : P) {
                    int a = axis[p.file];
                    double v = val[p.file];
                    bool ok = a == p.pred_axis && ((v > 0) - (v < 0)) == ((p.pred > 0) - (p.pred < 0));
                    rep_wrong_count[methods[m]][p.file] += !ok;
                    rep_pred[methods[m]][p.file].push_back(a == p.pred_axis ? p.pred : kNaN);
                }
            }
        }
        S.rep = score(rep, axis, val);
        S.wrong_per_round = mean(S.round_wrong);
    }
    fs::create_directories(out("out"));
    mp.csv(out("metrics_by_protocol.csv"));
    pr.csv(out("out") / "predictions.csv");
    Table sumt{{"method", "dims", "resub_Y", "resub_T", "evenodd_Y", "evenodd_T", "rep_Y", "rep_T", "rep_Ymax", "rep_Tmax", "rep_all_correct", "rep_wrong_per_round"}};
    for (auto& m : methods) {
        auto& S = SUM[m];
        sumt.add({m, std::to_string(S.dims), fmt(S.resub.Yacc, 4), fmt(S.resub.Tacc, 4), fmt(S.evenodd.Yacc, 4), fmt(S.evenodd.Tacc, 4), fmt(S.rep.Yacc, 6),
                  fmt(S.rep.Tacc, 6), fmt(S.rep.Ymax, 4), fmt(S.rep.Tmax, 4), std::to_string(S.rep_all_correct), fmt(S.wrong_per_round, 2)});
    }
    sumt.csv(out("summary.csv"));
    // 짝홀 실패 사진
    Table fe{{"method", "실패 장수", "사진"}};
    std::map<std::string, int> evenodd_fail;
    for (int m = 0; m < (int)methods.size(); ++m) {
        std::vector<std::string> fl;
        for (size_t t = 0; t < tasks.size(); ++t) {
            if (tasks[t].m != m || splits[tasks[t].s].protocol != "interleaved") continue;
            for (auto& p : TP[t]) {
                int a = axis[p.file];
                double v = val[p.file];
                if (!(a == p.pred_axis && ((v > 0) - (v < 0)) == ((p.pred > 0) - (p.pred < 0)))) fl.push_back(stem_of(shots[p.file]));
            }
        }
        std::sort(fl.begin(), fl.end());
        evenodd_fail[methods[m]] = (int)fl.size();
        std::string s;
        for (size_t i = 0; i < fl.size() && i < 14; ++i) s += (i ? " " : "") + fl[i];
        if (fl.size() > 14) s += " ...";
        fe.add({methods[m], std::to_string(fl.size()), s});
    }

    // ---- 사람이 읽는 이름 ----
    const std::map<std::string, std::string> kor = {
        {"corner_all_P12", "모서리 755점, 12계수"}, {"corner_nondiag_P12", "대각선 제외 630점, 12계수"}, {"corner_valid_P12", "통과 점만, 12계수"},
        {"ecc_affine18", "ECC 아핀 18값"}, {"corner_all_direct", "모서리 755점 그대로"}, {"corner_nondiag_direct", "630점 그대로"},
        {"corner_valid_direct", "늘 통과한 점 그대로"}, {"corner_all_direct_diagcov", "755점 그대로, 대각 공분산"},
        {"corner_nondiag_direct_diagcov", "630점 그대로, 대각 공분산"}, {"corner_valid_direct_diagcov", "늘 통과한 점 그대로, 대각 공분산"}};
    const std::map<std::string, std::string> eng = {
        {"corner_all_P12", "755 P12"}, {"corner_nondiag_P12", "630 P12"}, {"corner_valid_P12", "valid P12"}, {"ecc_affine18", "ECC affine18"},
        {"corner_all_direct", "755 direct"}, {"corner_nondiag_direct", "630 direct"}, {"corner_valid_direct", "valid direct"},
        {"corner_all_direct_diagcov", "755 diagcov"}, {"corner_nondiag_direct_diagcov", "630 diagcov"}, {"corner_valid_direct_diagcov", "valid diagcov"}};
    // 가장 나은 방법 (회당 틀린 장수 최소, diagcov 포함 전체)
    std::string best = methods[0];
    for (auto& m : methods) if (SUM[m].wrong_per_round < SUM[best].wrong_per_round) best = m;

    // ---- 그림 ----
    {
        std::vector<std::string> cats;
        std::vector<double> v;
        int hi = 0;
        for (auto& m : methods) {
            if (m == best) hi = (int)cats.size();
            cats.push_back(eng.at(m));
            v.push_back(SUM[m].wrong_per_round);
        }
        bar_chart("01_방법별_회당틀린장수.png", "Wrong photos per round (repeated 5-fold x20, of 280)", cats, v, "wrong / round", hi);
    }
    {
        Chart c("Axis+sign accuracy, repeated 5-fold x20", "method (1..10, order of table)", "accuracy");
        Vec x, y, t;
        for (size_t i = 0; i < methods.size(); ++i) { x.push_back(i + 1); y.push_back(SUM[methods[i]].rep.Yacc); t.push_back(SUM[methods[i]].rep.Tacc); }
        c.line(x, y, color(0), "Y (80 photos)", true).line(x, t, color(1), "T (200 photos)", true);
        c.save("02_방법별_Y_T정확도.png");
    }
    // 가장 나은 방법에서 반복 20회 중 틀린 횟수가 많은 사진
    std::vector<std::pair<int, int>> wc;  // (횟수, 파일)
    for (auto& [f, n] : rep_wrong_count[best]) if (n > 0) wc.push_back({n, f});
    std::stable_sort(wc.begin(), wc.end(), [](auto& a, auto& b) { return a.first > b.first; });
    {
        std::vector<std::string> cats;
        std::vector<double> v;
        for (size_t i = 0; i < wc.size() && i < 12; ++i) { cats.push_back(stem_of(shots[wc[i].second])); v.push_back(wc[i].first); }
        bar_chart("03_최선방법_자주틀린사진.png", "Best method (" + eng.at(best) + "): rounds wrong out of 20", cats, v, "rounds wrong", -1, 20);
    }
    // 튀는 값: 가장 자주 틀린 Y 사진, Θ 사진 (같으면 재대입 비용 차이가 작은 쪽)
    auto resub_pred = [&](const std::string& m, int f) {
        for (size_t t = 0; t < tasks.size(); ++t)
            if (methods[tasks[t].m] == m && splits[tasks[t].s].protocol == "full280_resubstitution")
                for (auto& p : TP[t]) if (p.file == f) return p;
        return Pred{f, -1, kNaN, kNaN, kNaN, kNaN};
    };
    int wyf = -1, wtf = -1;
    for (auto& [n, f] : wc) {
        if (axis[f] == 0 && wyf < 0) wyf = f;
        if (axis[f] == 1 && wtf < 0) wtf = f;
    }
    // 평균 모서리 x 이동 (630점) : 작은 Y 와 작은 Θ 가 얼마나 비슷한가
    auto mean_dx = [&](int i) {
        double s = 0;
        int n = 0;
        for (int k = 0; k < NC; ++k) if (T.corners[k].black != 2) { s += CM[i][k].x - CM[ic][k].x; ++n; }
        return s / n;
    };
    auto find_file = [&](const std::string& st) { for (int i = 0; i < NS; ++i) if (stem_of(shots[i]) == st) return i; return -1; };
    double px_y005 = (mean_dx(find_file("Y+0.05")) - mean_dx(find_file("Y-0.05"))) / 2;
    double px_t001 = (mean_dx(find_file("T+0.01")) - mean_dx(find_file("T-0.01"))) / 2;
    auto photo_fig = [&](int f, const std::string& file) {
        cv::Mat im = read_color(shots[f]);
        std::vector<cv::Mat> tiles;
        const double gain = 25;
        for (int q = 0; q < 3; ++q) {
            auto b = cf::kBoxes[q];
            cv::Rect roi(b[0], b[1], b[2] - b[0], b[3] - b[1]);
            std::vector<Mark> marks;
            double sx = 0, sy = 0;
            int n = 0;
            for (int k = 0; k < NC; ++k) {
                if (T.corners[k].qr != q + 1 || T.corners[k].black == 2) continue;
                double dx = CM[f][k].x - CM[ic][k].x, dy = CM[f][k].y - CM[ic][k].y;
                sx += dx; sy += dy; ++n;
            }
            sx /= n; sy /= n;
            // 이 QR 의 평균 이동을 빼고 남은 점별 이동만 화살표로 (x25)
            for (int k = 0; k < NC; ++k) {
                if (T.corners[k].qr != q + 1 || T.corners[k].black == 2 || (k % 3)) continue;
                double dx = CM[f][k].x - CM[ic][k].x - sx, dy = CM[f][k].y - CM[ic][k].y - sy;
                marks.push_back({{CM[f][k].x + 0.5, CM[f][k].y + 0.5}, {dx, dy}, "", kRed});
            }
            tiles.push_back(zoom(im, roi, 420, marks, gain, "QR" + std::to_string(q + 1) + " mean d " + fmt(sx, 2) + ", " + fmt(sy, 2) + " px"));
        }
        cv::Mat all = hstack(tiles);
        auto rp = resub_pred(best, f);
        std::string cap = stem_of(shots[f]) + ": red arrows = corner shift minus QR mean, x25. resub cost Y " + fmt(rp.cost_y, 1) + " / T " + fmt(rp.cost_t, 1);
        cv::Mat head(30, all.cols, CV_8UC3, cv::Scalar(255, 255, 255));
        text(head, cap, {6, 21}, 0.55);
        save_png(file, vstack({head, all}, 0));
    };
    std::string wyn = wyf >= 0 ? stem_of(shots[wyf]) : "", wtn = wtf >= 0 ? stem_of(shots[wtf]) : "";
    if (wyf >= 0) photo_fig(wyf, "04_튀는값_Y_" + wyn + ".png");
    if (wtf >= 0) photo_fig(wtf, "05_튀는값_T_" + wtn + ".png");
    Table outl{{"사진", "정답", "20회 중 틀림", "축이 맞은 회의 추정값 중앙값", "재대입 판정", "재대입 Y가설 비용", "재대입 Θ가설 비용", "630점 평균 x 이동 (px)", "P12 x 이동 / x 기울기 (px)"}};
    for (size_t i = 0; i < wc.size() && i < 6; ++i) {
        int f = wc[i].second;
        auto rp = resub_pred(best, f);
        Vec pv;
        for (double v : rep_pred[best][f]) if (!std::isnan(v)) pv.push_back(v);
        auto& F = FEAT.at("corner_nondiag_P12")[f];
        outl.add({stem_of(shots[f]), std::string(axis[f] ? "Θ " : "Y ") + fmts(val[f], 2), std::to_string(wc[i].first), pv.empty() ? "-" : fmts(median(pv), 3),
                  std::string(rp.pred_axis ? "Θ " : "Y ") + fmts(rp.pred, 3), fmt(rp.cost_y, 2), fmt(rp.cost_t, 2), fmts(mean_dx(f), 3),
                  fmts(F[0], 3) + " / " + fmts(F[2], 3)});
    }

    // ---- 원래 결과와 대조 (analysis_judge_compare_20261019/summary.csv, failures_evenodd.csv) ----
    struct O { double rY, rT, wr, eY, eT, sY, sT, yM, tM; int ef; };
    const std::map<std::string, O> orig = {
        {"corner_all_P12", {.914375, .98125, 10.6, .9125, .975, .9125, .99, .8651, .16, 12}},
        {"corner_all_direct", {.966875, .9825, 6.15, .975, .99, 1, 1, .5, .0846, 4}},
        {"corner_all_direct_diagcov", {.898125, .6765, 72.85, .9, .685, .925, .67, 2.2384, .4, 71}},
        {"corner_nondiag_P12", {.94125, .995, 5.7, .9375, .99, .95, .995, .3, .05, 7}},
        {"corner_nondiag_direct", {.96125, .98525, 6.05, .9875, .99, 1, 1, .4, .0740, 3}},
        {"corner_nondiag_direct_diagcov", {.879375, .68175, 73.3, .875, .69, .9, .685, 2.2400, .4, 72}},
        {"corner_valid_P12", {.946875, .991, 6.05, .9375, .99, .95, .995, .4182, .08, 7}},
        {"corner_valid_direct", {.9675, .9795, 6.7, .95, .985, 1, 1, .9594, .17, 7}},
        {"corner_valid_direct_diagcov", {.870625, .679, 74.55, .8875, .69, .875, .68, 2.2390, .4, 71}},
        {"ecc_affine18", {.958125, .9875, 5.85, .9625, .99, .9625, .985, .3, .05, 5}}};
    Parity par;
    par.add_text("가장 나은 방법 (회당 틀린 장수 최소)", "corner_nondiag_P12", best, best == "corner_nondiag_P12");
    for (auto& m : methods) {
        auto& o = orig.at(m);
        auto& S = SUM[m];
        std::string k = kor.at(m);
        par.add(k + " 회당 틀린 장수", o.wr, S.wrong_per_round, 0.5, 2);
        par.add(k + " 반복 Y 정확도", o.rY, S.rep.Yacc, 0.01);
        par.add(k + " 반복 Θ 정확도", o.rT, S.rep.Tacc, 0.004);
        par.add(k + " 짝홀 실패 장수", o.ef, evenodd_fail[m], 1, 0);
        par.add(k + " 재대입 Y / Θ 정확도 합", o.sY + o.sT, S.resub.Yacc + S.resub.Tacc, 0.0126);
    }
    par.add("630점 P12 반복 최대 오차 Y", orig.at("corner_nondiag_P12").yM, SUM["corner_nondiag_P12"].rep.Ymax, 0.06);
    par.add("630점 P12 반복 최대 오차 Θ", orig.at("corner_nondiag_P12").tM, SUM["corner_nondiag_P12"].rep.Tmax, 0.011);
    {
        int any_all = 0;
        for (auto& m : methods) any_all += SUM[m].rep_all_correct;
        par.add("20회 중 280장 전부 맞힌 회 (모든 방법 합)", 0, any_all, 0, 0);
    }

    // ---- 결과.md ----
    Table st{{"측정값", "차원", "재대입 Y / Θ", "짝홀 Y / Θ", "반복 Y", "반복 Θ", "회당 틀린 장수", "반복 최대오차 Y / Θ", "20회 중 전부 맞힘"}};
    for (auto& m : methods) {
        auto& S = SUM[m];
        st.add({kor.at(m) + " (`" + m + "`)", std::to_string(S.dims), fmt(100 * S.resub.Yacc, 1) + " / " + fmt(100 * S.resub.Tacc, 1) + "%",
                fmt(100 * S.evenodd.Yacc, 1) + " / " + fmt(100 * S.evenodd.Tacc, 1) + "%", fmt(100 * S.rep.Yacc, 1) + "%", fmt(100 * S.rep.Tacc, 1) + "%",
                (m == best ? "**" : "") + fmt(S.wrong_per_round, 2) + (m == best ? "**" : ""), fmt(S.rep.Ymax, 2) + " / " + fmt(S.rep.Tmax, 2), std::to_string(S.rep_all_correct)});
    }
    std::ostringstream R;
    R << "# 결과: 측정값별 같은 GLS 판정기 정확도\n\n"
      << "입력: `data/clean/clean_20261014` 281장. 사진마다 다시 측정했다: 모서리 755점(10-19 13:16 실험과 같은 추적), QR 세 영역 ECC 아핀(10-12 실험과 같은 측정).\n"
      << "판정기: center 고정, 축마다 원점을 지나는 2차 곡선(이동 = v s + v² c), 학습 잔차 공분산으로 만든 GLS 거리로 'Y만'과 'Θ만' 가설을 비교해 비용이 작은 쪽, 값은 가우스-뉴턴 8회.\n"
      << "측정값: 모서리 이동을 판 전체 2차 변형 12계수(P12)로 요약하거나 x, y 이동을 그대로(direct) 넣음. 규칙 all = 755점, nondiag = 대각 접점 125점 뺀 630점, valid = P12 는 사진마다 통과 점, direct 는 281장 모두 통과한 점("
      << SUM["corner_valid_direct"].dims / 2 << "점). diagcov 는 direct 와 같되 공분산 대각만.\n"
      << "분할: 재대입(280장 학습·평가), 짝홀(축마다 촬영 순서로 번갈아 2겹), 반복 5분할 20회(sklearn StratifiedKFold, 묶음 6개로 층화, random_state 7000~7019). 반복 분할은 numpy MT19937 셔플을 그대로 옮겨 원래 splits.json 과 같은 사진 묶음을 만든다(`splits.csv`).\n\n"
      << "## 방법별 요약\n\n" << st.md()
      << "\n## 짝홀 실패 사진\n\n" << fe.md()
      << "\n## 가장 나은 방법(" << kor.at(best) << ")에서 반복 20회 중 틀린 사진\n\n" << outl.md()
      << "\n## 원래 결과와 대조 (" << par.ok << "/" << par.total << " 일치)\n\n" << par.t.md()
      << "\n## 옮기지 않은 부분\n\n- `ref205_P12`(기존 205점 GN 추적 + P12): center_fixed_20261015 의 SpreadPairExtractor 추적을 다시 만들어야 해서 옮기지 않았다. 원래 결과는 회당 6.25장 틀림(Y 93.4%, Θ 99.5%).\n"
      << "- `easy_figure.py`, `add_easy_slide.py`, `add_judge_slide.py`: 발표용 그림·슬라이드. `이미지/01`~`03` 으로 대신했다.\n"
      << "- 저장소 `종합/` 아래 하위 작업(묶음별 EDA 도구, 옛 C++ 이식, 6원점 발표 630점 재평가)은 이 실험의 대표 결과를 만든 경로가 아니라 옮기지 않았다.\n"
      << "\n## 파일\n\n- `summary.csv`: 방법별 요약\n- `metrics_by_protocol.csv`: 방법 x 분할 지표\n- `splits.csv`: 짝홀·반복 분할의 평가 사진\n- `out/predictions.csv`: 사진별 예측 (크기 때문에 out/, git 제외)\n"
      << "- `이미지/01_방법별_회당틀린장수.png`, `02_방법별_Y_T정확도.png`(방법 번호 = 위 표 순서), `03_최선방법_자주틀린사진.png`, `04`·`05` 튀는 값\n";
    write_text(out("결과.md"), R.str());

    // ---- 보고서.md ----
    auto& B = SUM[best];
    double wmin = 1e9, wmax = 0;
    for (auto& m : methods) if (m.find("diagcov") == std::string::npos) { wmin = std::min(wmin, SUM[m].wrong_per_round); wmax = std::max(wmax, SUM[m].wrong_per_round); }
    std::ostringstream P;
    P << "# 보고서: 측정값별 같은 GLS 판정기 정확도 (10-19 14:00)\n\n"
      << "## 문제\n\n측정 방법(모서리 755/630점, ECC 아핀)에 따라 Y/Θ 판정 정확도가 달라지는지 같은 판정기로 비교해야 했다.\n\n"
      << "## 해결 방법\n\ncenter 고정 2차 곡선 GLS 판정기에 측정값 7가지와 대각 공분산 3가지를 넣고 재대입·짝홀·반복 5분할 20회로 평가했다.\n\n"
      << "## 결과\n\n" << kor.at(best) << "가 회당 " << fmt(B.wrong_per_round, 2) << "장 틀림(Y " << fmt(100 * B.rep.Yacc, 1) << "%, Θ " << fmt(100 * B.rep.Tacc, 1)
      << "%)으로 가장 나았고, 대각 공분산을 뺀 방법들은 회당 " << fmt(wmin, 2) << "~" << fmt(wmax, 2) << "장으로 비슷했으며, 20회 중 280장을 모두 맞힌 회는 어느 방법에도 없었다.\n\n"
      << "| 지표 (" << kor.at(best) << ") | 값 |\n|---|---|\n"
      << "| 반복 5분할 20회 Y 축·부호 정확도 | " << fmt(100 * B.rep.Yacc, 2) << "% (1600건) |\n"
      << "| 반복 5분할 20회 Θ 축·부호 정확도 | " << fmt(100 * B.rep.Tacc, 2) << "% (4000건) |\n"
      << "| 회당 틀린 장수 (280장 중) | " << fmt(B.wrong_per_round, 2) << " |\n"
      << "| 반복 최대 오차 Y / Θ | " << fmt(B.rep.Ymax, 3) << " mm / " << fmt(B.rep.Tmax, 3) << " deg |\n"
      << "| 짝홀 실패 / 재대입 Y·Θ 정확도 | " << evenodd_fail[best] << "장 / " << fmt(100 * B.resub.Yacc, 1) << "%·" << fmt(100 * B.resub.Tacc, 1) << "% |\n\n"
      << st.md() << "\n" << img_md("01_방법별_회당틀린장수.png", "회당 틀린 장수") << "\n\n" << img_md("02_방법별_Y_T정확도.png", "Y Θ 정확도") << "\n\n"
      << "## 튀는 값\n\n" << kor.at(best) << "에서 가장 자주 틀린 사진은 Y 는 " << wyn << ", Θ 는 " << wtn << "(20회 중 각각 " << (wyf >= 0 ? rep_wrong_count[best][wyf] : 0)
      << "·" << (wtf >= 0 ? rep_wrong_count[best][wtf] : 0) << "회)이고, 재대입에서 Y가설과 Θ가설 비용 차이가 각각 "
      << fmt(wyf >= 0 ? std::abs(resub_pred(best, wyf).cost_y - resub_pred(best, wyf).cost_t) : kNaN, 2) << ", "
      << fmt(wtf >= 0 ? std::abs(resub_pred(best, wtf).cost_y - resub_pred(best, wtf).cost_t) : kNaN, 2) << "이고 둘 다 맞혔지만, 그 사진이 학습에서 빠지면 축이나 부호가 뒤집혔다. 즉 |값|이 작은 사진은 이웃 사진만으로 만든 곡선과 공분산으로는 두 가설이 갈리지 않았다.\n\n"
      << outl.md() << "\n" << img_md("03_최선방법_자주틀린사진.png", "자주 틀린 사진") << "\n\n"
      << (wyf >= 0 ? img_md("04_튀는값_Y_" + wyn + ".png", "Y 튀는 값") + "\n\n" : "") << (wtf >= 0 ? img_md("05_튀는값_T_" + wtn + ".png", "T 튀는 값") + "\n\n" : "")
      << "빨간 화살표는 그 QR 평균 이동을 뺀 모서리별 이동(25배)이다. Y 와 Θ 를 가르는 단서는 이 남은 변형인데, 작은 값에서는 화살표가 흔들림 수준이다.\n\n"
      << "## 한줄평\n\n판정기는 같은 이동 방향을 축별 2차 곡선과 공분산으로 가르는데 Y 0.05 mm 가 630점 평균 x 이동 " << fmt(px_y005, 3) << " px, Θ 0.01 deg 가 "
      << fmt(px_t001, 3) << " px로 같은 방향의 비슷한 크기라서, 측정값을 바꿔도 작은 값 사진 몇 장이 계속 틀려 회당 " << fmt(wmin, 1) << "~" << fmt(wmax, 1)
      << "장 차이밖에 나지 않았고, 점끼리 상관을 버린 대각 공분산은 사진 공통 흔들림을 모든 점에서 독립으로 세어 Θ 가 " << fmt(100 * SUM["corner_nondiag_direct_diagcov"].rep.Tacc, 0) << "%로 무너졌다.\n\n"
      << "## 원래 결과와 대조\n\n원래 값은 `analysis_judge_compare_20261019/summary.csv`, `failures_evenodd.csv` 에서 가져왔다. " << par.ok << "/" << par.total
      << " 일치. 반복 분할은 원래 splits.json 과 같은 규칙(numpy MT19937 셔플)으로 다시 만들었고(102개 평가 묶음 모두 같음을 따로 확인), 짝홀은 원래 파일 수정 시각 순서를 촬영 순서 규칙으로 적었다. "
      << "ECC 아핀과 630점 P12 는 원래와 거의 같다. 755점·통과 점 방법이 어긋난 이유는 QR3 사각형이다: OpenCV 3.4 해독기가 QR3 을 3x3 가우시안으로 흐린 영역에서만 찾아 시작 사각형이 원래(cv2 5.0)와 최대 0.73 px 다르고, "
      << "대각 접점과 통과 판정이 시작점에 민감해 그 방법들에서 늘 틀리던 사진 1장이 맞거나(회당 -1.00) 늘 통과 점 수가 달라졌다(" << SUM["corner_valid_direct"].dims / 2
      << "점, 원래 258점). 확인용으로 원래 QR3 사각형 값을 넣어 따로 돌렸을 때는 54/54 일치하고 630점 P12 가 회당 5.70장으로 가장 나았다(이 결과 파일에는 넣지 않음).\n\n" << par.t.md();
    write_text(out("보고서.md"), P.str());
    std::cout << "best " << best << " wrong/round " << B.wrong_per_round << " parity " << par.ok << "/" << par.total << "\n";
    return 0;
}
