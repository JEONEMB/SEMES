// 10-21 4-fold 실험군 공통 판정부: 6원점 후보(T+,T-,Y1+,Y1-,Y2+,Y2-) 이차곡선 + GLS 비용, 4-fold 바깥 / 3-fold 안쪽 선택.
// 원본: codex_experiments_20261014/zero_experiment.py (manifest, fit, predict),
//       multiref_4fold_20261021/experiment.py (fit 축소, enrich, stats, StratifiedKFold 분할).
// sklearn StratifiedKFold(shuffle=True, random_state) 는 numpy RandomState(MT19937) 의 shuffle 을 그대로 옮겨 같은 분할을 만든다.
#pragma once
#include <hye/dataset.hpp>
#include <hye/report.hpp>
#include <opencv2/core.hpp>
#include <omp.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <vector>
#include "g8_measure.hpp"

namespace g8 {

// ---- 사진 목록: 원본 names 순서 (center 먼저, 나머지 파일명 사전순) ----
struct Item {
    std::string file;  // T+0.01.bmp
    char axis;         // C, T, Y
    double value;
    std::string session;  // C, T+, T-, Y1+, Y1-, Y2+, Y2-
    int shot;             // load_1014() 인덱스
};
struct Manifest {
    std::vector<Item> it;
    std::vector<int> ids;  // center 제외
    int n() const { return (int)it.size(); }
};
inline Manifest make_manifest(const std::vector<hye::Shot>& shots) {
    Manifest m;
    for (int i = 0; i < (int)shots.size(); ++i) {
        std::string f = shots[i].rel.substr(shots[i].rel.find_last_of('/') + 1);
        Item t;
        t.file = f;
        t.shot = i;
        if (f == "center.bmp") { t.axis = 'C'; t.value = 0; t.session = "C"; }
        else {
            t.axis = f[0];
            t.value = std::stod(f.substr(1, f.size() - 5));
            if (t.axis == 'T') t.session = "T";
            else t.session = (std::abs(std::lround(t.value / 0.05)) % 2) ? "Y2" : "Y1";
            t.session += t.value > 0 ? "+" : "-";
        }
        m.it.push_back(t);
    }
    std::sort(m.it.begin(), m.it.end(), [](const Item& a, const Item& b) {
        bool ca = a.file != "center.bmp", cb = b.file != "center.bmp";
        return ca != cb ? ca < cb : a.file < b.file;
    });
    for (int i = 0; i < m.n(); ++i) if (m.it[i].axis != 'C') m.ids.push_back(i);
    return m;
}

// ---- sklearn StratifiedKFold(shuffle=True) ----
inline uint32_t np_interval(std::mt19937& g, uint32_t mx) {
    if (mx == 0) return 0;
    uint32_t mask = mx;
    mask |= mask >> 1; mask |= mask >> 2; mask |= mask >> 4; mask |= mask >> 8; mask |= mask >> 16;
    uint32_t v;
    while ((v = (uint32_t)g() & mask) > mx) {}
    return v;
}
// y: 각 표본의 범주. 반환: (train 위치, test 위치) n_splits 개, 위치는 오름차순
inline std::vector<std::pair<std::vector<int>, std::vector<int>>> skfold(const std::vector<std::string>& y, int ns, uint32_t seed) {
    std::mt19937 g(seed);
    std::vector<std::string> order;
    std::vector<int> enc(y.size());
    for (size_t i = 0; i < y.size(); ++i) {
        auto f = std::find(order.begin(), order.end(), y[i]);
        if (f == order.end()) { order.push_back(y[i]); enc[i] = (int)order.size() - 1; }
        else enc[i] = (int)(f - order.begin());
    }
    const int nc = (int)order.size();
    std::vector<int> yo = enc;
    std::sort(yo.begin(), yo.end());
    std::vector<std::vector<int>> alloc(ns, std::vector<int>(nc, 0));
    for (int i = 0; i < ns; ++i)
        for (size_t j = i; j < yo.size(); j += ns) alloc[i][yo[j]]++;
    std::vector<int> tf(y.size());
    for (int k = 0; k < nc; ++k) {
        std::vector<int> ff;
        for (int i = 0; i < ns; ++i) for (int r = 0; r < alloc[i][k]; ++r) ff.push_back(i);
        for (int i = (int)ff.size() - 1; i >= 1; --i) std::swap(ff[i], ff[np_interval(g, (uint32_t)i)]);
        size_t p = 0;
        for (size_t i = 0; i < y.size(); ++i) if (enc[i] == k) tf[i] = ff[p++];
    }
    std::vector<std::pair<std::vector<int>, std::vector<int>>> out(ns);
    for (int f = 0; f < ns; ++f)
        for (int i = 0; i < (int)y.size(); ++i) (tf[i] == f ? out[f].second : out[f].first).push_back(i);
    return out;
}
struct Fold { std::vector<int> tr, te; };
// multiref_4fold_20261021: StratifiedKFold(4, shuffle, 20261021) on ids by session
inline std::vector<Fold> outer_folds(const Manifest& m) {
    std::vector<std::string> y;
    for (int i : m.ids) y.push_back(m.it[i].session);
    std::vector<Fold> o;
    for (auto& s : skfold(y, 4, 20261021u)) {
        Fold f;
        for (int a : s.first) f.tr.push_back(m.ids[a]);
        for (int b : s.second) f.te.push_back(m.ids[b]);
        o.push_back(f);
    }
    return o;
}
inline std::vector<Fold> inner_folds(const Manifest& m, const std::vector<int>& tr, int f) {
    std::vector<std::string> y;
    for (int i : tr) y.push_back(m.it[i].session);
    std::vector<Fold> o;
    for (auto& s : skfold(y, 3, (uint32_t)(100 + f))) {
        Fold q;
        for (int a : s.first) q.tr.push_back(tr[a]);
        for (int b : s.second) q.te.push_back(tr[b]);
        o.push_back(q);
    }
    return o;
}

// ---- 선형대수 ----
inline cv::Mat rows(const cv::Mat& X, const std::vector<int>& ix) {
    cv::Mat o((int)ix.size(), X.cols, CV_64F);
    for (int i = 0; i < (int)ix.size(); ++i) X.row(ix[i]).copyTo(o.row(i));
    return o;
}
inline cv::Mat lstsq(const cv::Mat& A, const cv::Mat& B) {
    cv::Mat X;
    cv::solve(A, B, X, cv::DECOMP_SVD);
    return X;
}
inline int matrix_rank(const cv::Mat& A) {
    cv::Mat w;
    cv::SVD::compute(A, w, cv::SVD::NO_UV);
    double mx = 0;
    for (int i = 0; i < w.rows; ++i) mx = std::max(mx, w.at<double>(i));
    double tol = mx * std::max(A.rows, A.cols) * std::numeric_limits<double>::epsilon();
    int r = 0;
    for (int i = 0; i < w.rows; ++i) r += w.at<double>(i) > tol;
    return r;
}
inline cv::Mat cholesky_lower(const cv::Mat& S) {
    const int n = S.rows;
    cv::Mat L = cv::Mat::zeros(n, n, CV_64F);
    for (int j = 0; j < n; ++j) {
        double s = S.at<double>(j, j);
        for (int k = 0; k < j; ++k) s -= L.at<double>(j, k) * L.at<double>(j, k);
        L.at<double>(j, j) = std::sqrt(s);
        for (int i = j + 1; i < n; ++i) {
            double t = S.at<double>(i, j);
            for (int k = 0; k < j; ++k) t -= L.at<double>(i, k) * L.at<double>(j, k);
            L.at<double>(i, j) = t / L.at<double>(j, j);
        }
    }
    return L;
}
// np.linalg.cholesky(inv(C)).T / std[None,:]
inline cv::Mat whitener(const cv::Mat& C, const std::vector<double>& sd) {
    cv::Mat Ci;
    cv::invert(C, Ci, cv::DECOMP_LU);
    Ci = (Ci + Ci.t()) * 0.5;
    cv::Mat W = cholesky_lower(Ci).t();
    for (int j = 0; j < W.cols; ++j) W.col(j) /= sd[j];
    return W;
}

// ---- 6 원점 후보 곡선 ----
struct Branch {
    std::string key;
    char axis;
    cv::Mat b, s, c;  // 1 x d
};
using Branches = std::vector<Branch>;
inline const Branch& find_branch(const Branches& B, const std::string& k) {
    for (auto& b : B) if (b.key == k) return b;
    throw std::runtime_error("branch " + k);
}

// zero_experiment.fit(mode='inferred_session'): 축마다 [v, v^2, 시행 절편] 최소제곱. resid_rows 는 축 Y, T 순
inline Branches zfit(const Manifest& m, const cv::Mat& X, const std::vector<int>& tr, bool quad, cv::Mat* R = nullptr, int* rank = nullptr) {
    Branches br;
    std::vector<cv::Mat> res;
    int pr = 0;
    for (char a : {'Y', 'T'}) {
        std::vector<int> ii;
        for (int i : tr) if (m.it[i].axis == a) ii.push_back(i);
        std::vector<std::string> ses;
        for (int i : ii) ses.push_back(m.it[i].session);
        std::sort(ses.begin(), ses.end());
        ses.erase(std::unique(ses.begin(), ses.end()), ses.end());
        const int nb = quad ? 2 : 1;
        cv::Mat A = cv::Mat::zeros((int)ii.size(), nb + (int)ses.size(), CV_64F);
        for (int r = 0; r < (int)ii.size(); ++r) {
            double v = m.it[ii[r]].value;
            A.at<double>(r, 0) = v;
            if (quad) A.at<double>(r, 1) = v * v;
            A.at<double>(r, nb + (int)(std::find(ses.begin(), ses.end(), m.it[ii[r]].session) - ses.begin())) = 1;
        }
        cv::Mat Z = rows(X, ii), co = lstsq(A, Z);
        if (rank) pr += matrix_rank(A);
        if (R) res.push_back(Z - A * co);
        for (int j = 0; j < (int)ses.size(); ++j)
            br.push_back({ses[j], a, co.row(nb + j).clone(), co.row(0).clone(), quad ? co.row(1).clone() : cv::Mat::zeros(1, X.cols, CV_64F)});
    }
    if (R) cv::vconcat(res, *R);
    if (rank) *rank = pr;
    return br;
}
// 학습 순서대로 잔차 (multiref experiment.fit 의 R)
inline cv::Mat residual_rows(const Manifest& m, const cv::Mat& X, const std::vector<int>& tr, const Branches& br) {
    cv::Mat R((int)tr.size(), X.cols, CV_64F);
    for (int r = 0; r < (int)tr.size(); ++r) {
        auto& b = find_branch(br, m.it[tr[r]].session);
        double v = m.it[tr[r]].value;
        cv::Mat x = X.row(tr[r]) - b.b - b.s * v - b.c * (v * v);
        x.copyTo(R.row(r));
    }
    return R;
}
inline std::vector<double> col_std(const cv::Mat& S) {
    std::vector<double> sd(S.rows);
    for (int i = 0; i < S.rows; ++i) sd[i] = std::sqrt(std::max(S.at<double>(i, i), 1e-14));
    return sd;
}
inline cv::Mat corr(const cv::Mat& S, const std::vector<double>& sd) {
    cv::Mat C = S.clone();
    for (int i = 0; i < S.rows; ++i)
        for (int j = 0; j < S.cols; ++j) C.at<double>(i, j) /= sd[i] * sd[j];
    return C;
}
// multiref experiment.fit: inferred_session, 공분산 상관 축소
struct GLSModel { Branches br; cv::Mat W; };
inline GLSModel mfit(const Manifest& m, const cv::Mat& X, const std::vector<int>& tr, bool quad, double shrink) {
    GLSModel g;
    g.br = zfit(m, X, tr, quad);
    cv::Mat R = residual_rows(m, X, tr, g.br);
    cv::Mat S = R.t() * R / std::max((int)tr.size() - 16, 1);
    auto sd = col_std(S);
    cv::Mat C = corr(S, sd) * (1 - shrink) + cv::Mat::eye(S.rows, S.cols, CV_64F) * shrink;
    g.W = whitener(C, sd);
    return g;
}

// ---- 예측 ----
struct Pred {
    int i;  // manifest 인덱스
    char axis, pred_axis;
    double value, pred, cost;
    std::string session, branch;
    std::vector<double> bv, bc;  // 후보별 값·비용 (후보 순서)
    double true_v = 0, true_c = 0;  // 정답 시행 후보의 값·비용
    bool correct = false, pass = false;
    double eY = 0, eT = 0;
};
// 한 후보: z = 변환된 (x - b), s, c. 8번 GN 으로 명령값 v 를 풀고 비용을 낸다
inline void branch_solve(const cv::Mat& z, const cv::Mat& s, const cv::Mat& c, std::vector<double>& v, std::vector<double>& cost) {
    const int n = z.rows, d = z.cols;
    v.assign(n, 0);
    cost.assign(n, 0);
    const double* sp = s.ptr<double>();
    const double* cp = c.ptr<double>();
    double ss = 0;
    for (int j = 0; j < d; ++j) ss += sp[j] * sp[j];
    for (int r = 0; r < n; ++r) {
        const double* zp = z.ptr<double>(r);
        double a = 0;
        for (int j = 0; j < d; ++j) a += zp[j] * sp[j];
        double x = a / ss;
        for (int it = 0; it < 8; ++it) {
            double num = 0, den = 0;
            for (int j = 0; j < d; ++j) {
                double J = sp[j] + 2 * x * cp[j], rr = zp[j] - x * sp[j] - x * x * cp[j];
                num += J * rr;
                den += J * J;
            }
            x += num / std::max(den, 1e-20);
        }
        double q = 0;
        for (int j = 0; j < d; ++j) {
            double rr = zp[j] - x * sp[j] - x * x * cp[j];
            q += rr * rr;
        }
        v[r] = x;
        cost[r] = q;
    }
}
inline bool sign_ok(const std::string& key, double v) { return v * (key.back() == '+' ? 1 : -1) >= 0; }
// 후보별 (v, cost) 로 최소 비용 후보를 고른다
inline std::vector<Pred> choose(const Manifest& m, const std::vector<int>& te, const std::vector<std::string>& keys, const std::vector<char>& axes,
                                const std::vector<std::vector<double>>& V, const std::vector<std::vector<double>>& Cst) {
    std::vector<Pred> out;
    for (int t = 0; t < (int)te.size(); ++t) {
        int best = 0;
        for (int k = 1; k < (int)keys.size(); ++k) if (Cst[k][t] < Cst[best][t]) best = k;
        Pred p;
        p.i = te[t];
        p.axis = m.it[te[t]].axis;
        p.value = m.it[te[t]].value;
        p.session = m.it[te[t]].session;
        p.pred_axis = axes[best];
        p.pred = V[best][t];
        p.branch = keys[best];
        p.cost = Cst[best][t];
        for (size_t k = 0; k < keys.size(); ++k) {
            p.bv.push_back(V[k][t]);
            p.bc.push_back(Cst[k][t]);
            if (keys[k] == p.session) { p.true_v = V[k][t]; p.true_c = Cst[k][t]; }
        }
        double ty = p.axis == 'Y' ? p.value : 0, tt = p.axis == 'T' ? p.value : 0;
        double py = p.pred_axis == 'Y' ? p.pred : 0, pt = p.pred_axis == 'T' ? p.pred : 0;
        auto sgn = [](double x) { return (x > 0) - (x < 0); };
        p.correct = p.axis == p.pred_axis && sgn(p.value) == sgn(p.pred);
        p.eY = std::abs(ty - py);
        p.eT = std::abs(tt - pt);
        p.pass = p.correct && p.eY <= 0.09 && p.eT <= 0.09;
        out.push_back(p);
    }
    return out;
}
// zero_experiment.predict (inferred_session): W 로 백색화한 GLS 비용
inline std::vector<Pred> zpredict(const Manifest& m, const cv::Mat& X, const std::vector<int>& te, const GLSModel& g,
                                  const std::vector<double>* penalty = nullptr, const std::vector<cv::Mat>* Ws = nullptr) {
    std::vector<std::string> keys;
    std::vector<char> axes;
    std::vector<std::vector<double>> V, C;
    cv::Mat Xt = rows(X, te);
    for (size_t k = 0; k < g.br.size(); ++k) {
        auto& b = g.br[k];
        const cv::Mat& W = Ws ? (*Ws)[k] : g.W;
        cv::Mat Z = (Xt - cv::repeat(b.b, Xt.rows, 1)) * W.t(), sw = b.s * W.t(), cw = b.c * W.t();
        std::vector<double> v, c;
        branch_solve(Z, sw, cw, v, c);
        for (size_t t = 0; t < c.size(); ++t) {
            if (penalty) c[t] += (*penalty)[k];
            if (!sign_ok(b.key, v[t])) c[t] = std::numeric_limits<double>::infinity();
        }
        keys.push_back(b.key);
        axes.push_back(b.axis);
        V.push_back(v);
        C.push_back(c);
    }
    return choose(m, te, keys, axes, V, C);
}

struct Stats {
    int n = 0, Yc = 0, Yn = 0, Tc = 0, Tn = 0, fail = 0;
    double Ymax = 0, Tmax = 0;
};
inline Stats stats(const std::vector<Pred>& p) {
    Stats s;
    for (auto& r : p) {
        ++s.n;
        if (r.axis == 'Y') { ++s.Yn; s.Yc += r.correct; }
        if (r.axis == 'T') { ++s.Tn; s.Tc += r.correct; }
        s.fail += !r.pass;
        s.Ymax = std::max(s.Ymax, r.eY);
        s.Tmax = std::max(s.Tmax, r.eT);
    }
    return s;
}

// ---- 바깥 4-fold x 안쪽 3-fold 선택, 전체 재대입 ----
struct Selection {
    int fold, chosen;
    std::vector<std::tuple<int, double, int>> rank;  // (실패, 최대오차, 설정) 오름차순
};
struct Nested {
    std::vector<std::vector<Pred>> oof;    // [설정] 바깥 4 fold 시험 예측 280
    std::vector<std::vector<int>> oof_fold;
    std::vector<std::vector<Pred>> resub;  // [설정] 전체 280 재대입
    std::vector<Pred> nested;
    std::vector<int> nested_fold;
    std::vector<Selection> sel;
};
template <class PF>
Nested run_nested(const Manifest& m, int K, PF pf, bool verbose = true) {
    Nested N;
    N.oof.resize(K);
    N.oof_fold.resize(K);
    N.resub.resize(K);
    auto folds = outer_folds(m);
    for (int f = 0; f < (int)folds.size(); ++f) {
        auto inner = inner_folds(m, folds[f].tr, f);
        std::vector<std::tuple<int, double, int>> rank(K);
        std::vector<std::vector<Pred>> local(K);
#pragma omp parallel for schedule(dynamic)
        for (int k = 0; k < K; ++k) {
            std::vector<Pred> ip;
            for (auto& q : inner) {
                auto r = pf(k, q.tr, q.te);
                ip.insert(ip.end(), r.begin(), r.end());
            }
            Stats s = stats(ip);
            rank[k] = std::make_tuple(s.fail, std::max(s.Ymax, s.Tmax), k);
            local[k] = pf(k, folds[f].tr, folds[f].te);
        }
        auto best = *std::min_element(rank.begin(), rank.end());
        int chosen = std::get<2>(best);
        for (int k = 0; k < K; ++k) {
            N.oof[k].insert(N.oof[k].end(), local[k].begin(), local[k].end());
            N.oof_fold[k].insert(N.oof_fold[k].end(), local[k].size(), f);
        }
        N.nested.insert(N.nested.end(), local[chosen].begin(), local[chosen].end());
        N.nested_fold.insert(N.nested_fold.end(), local[chosen].size(), f);
        std::sort(rank.begin(), rank.end());
        N.sel.push_back({f, chosen, rank});
        if (verbose) std::printf("fold %d chosen %d\n", f, chosen), std::fflush(stdout);
    }
#pragma omp parallel for schedule(dynamic)
    for (int k = 0; k < K; ++k) N.resub[k] = pf(k, m.ids, m.ids);
    return N;
}

// ---- 측정 (281 장) ----
struct Measured {
    Manifest m;
    std::vector<cv::Point2d> pts;
    std::map<std::string, std::vector<Track>> tr;  // 변형 -> manifest 순서
    std::vector<Track> screw;
    ScrewSet screw_set;
};
struct Variant { std::string key; int window; double sigma; int robust; double delta; };
inline Measured measure_all(const std::vector<hye::Shot>& shots, const std::vector<Variant>& vars, bool screws) {
    Measured M;
    M.m = make_manifest(shots);
    M.pts = points630();
    cv::Mat center = read_crop(shots[M.m.it[0].shot]);
    std::vector<QRExtractor> ex;
    for (auto& v : vars) {
        ex.emplace_back(center, v.window, v.sigma, v.robust, v.delta);
        M.tr[v.key].resize(M.m.n());
    }
    std::unique_ptr<ScrewExtractor> sx;
    if (screws) {
        M.screw_set = screw_features(center);
        sx.reset(new ScrewExtractor(center, M.screw_set));
        M.screw.resize(M.m.n());
    }
    int done = 0;
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < M.m.n(); ++i) {
        cv::Mat cur = read_crop(shots[M.m.it[i].shot]);
        for (size_t v = 0; v < vars.size(); ++v) M.tr[vars[v].key][i] = ex[v].extract(cur);
        if (screws) M.screw[i] = sx->extract(cur);
#pragma omp critical
        {
            if (++done % 40 == 0) std::printf("measured %d/%d\n", done, M.m.n()), std::fflush(stdout);
        }
    }
    return M;
}
// center 를 뺀 이동 (double)
inline std::vector<std::vector<cv::Point2d>> rel_D(const std::vector<Track>& T) {
    std::vector<std::vector<cv::Point2d>> D(T.size());
    for (size_t n = 0; n < T.size(); ++n)
        for (size_t i = 0; i < T[n].D.size(); ++i)
            D[n].push_back({(double)T[n].D[i].x - (double)T[0].D[i].x, (double)T[n].D[i].y - (double)T[0].D[i].y});
    return D;
}
// center_patch_v2 의 고정 품질 가중: 1/(1+(rmse/6)^2+(FB/.15)^2), 무효 x .05, 하한 .001
inline std::vector<std::vector<double>> quality_weights(const std::vector<Track>& T) {
    std::vector<std::vector<double>> w(T.size());
    for (size_t n = 0; n < T.size(); ++n)
        for (size_t i = 0; i < T[n].q.size(); ++i) {
            double a = (double)T[n].q[i][0] / 6, b = (double)T[n].q[i][4] / 0.15;
            double v = 1 / (1 + a * a + b * b) * (T[n].valid[i] ? 1 : 0.05);
            w[n].push_back(std::max(v, 0.001));
        }
    return w;
}
inline cv::Mat full_field(const std::vector<std::vector<cv::Point2d>>& D) {
    cv::Mat X((int)D.size(), (int)D[0].size() * 2, CV_64F);
    for (int n = 0; n < X.rows; ++n)
        for (int i = 0; i < (int)D[n].size(); ++i) { X.at<double>(n, 2 * i) = D[n][i].x; X.at<double>(n, 2 * i + 1) = D[n][i].y; }
    return X;
}
inline double valid_rate(const std::vector<Track>& T) {
    double s = 0, n = 0;
    for (size_t k = 1; k < T.size(); ++k) for (char v : T[k].valid) { s += v; ++n; }
    return s / n;
}

// 진단용 덤프 (원본 npz 와 비교할 때만): REPO 밖 경로에 CSV
inline void dump_tracks(const std::string& path, const Measured& M, const std::string& key) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return;
    auto& T = key == "screw" ? M.screw : M.tr.at(key);
    for (int n = 0; n < M.m.n(); ++n)
        for (size_t i = 0; i < T[n].D.size(); ++i)
            std::fprintf(f, "%s,%d,%.9g,%.9g,%.9g,%.9g,%d\n", M.m.it[n].file.c_str(), (int)i, T[n].D[i].x, T[n].D[i].y, T[n].q[i][0], T[n].q[i][4], (int)T[n].valid[i]);
    std::fclose(f);
}

}  // namespace g8
