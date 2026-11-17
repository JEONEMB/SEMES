// 6후보(촬영 구간별 0점) GLS 판정기와 검증 분할 (원본: codex_experiments_20261014/zero_experiment.py fit/predict/splits/score,
// gls_comparison_20261014/compare.py protocols/score). sklearn StratifiedKFold(shuffle) 는 numpy RandomState(MT19937)와 같은 난수열로 옮겼다.
#pragma once
#include <hye/dataset.hpp>
#include <hye/report.hpp>
#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace six {

// 사진 목록: center, Theta 정렬, Y 정렬 (measure.py paths 순서)
struct Meta { std::string file, axis, session; double value; size_t shot; };
inline std::string session_of(const std::string& axis, double v) {
    if (axis == "C") return "C";
    if (axis == "T") return v > 0 ? "T+" : "T-";
    return std::string(std::abs(std::lround(v / 0.05)) % 2 ? "Y2" : "Y1") + (v > 0 ? "+" : "-");
}
inline std::vector<Meta> manifest(const std::vector<hye::Shot>& shots) {
    std::vector<Meta> m;
    for (size_t i = 0; i < shots.size(); ++i) {
        auto& s = shots[i];
        std::string f = s.rel.substr(s.rel.find('/') + 1), ax = s.condition == "center" ? "C" : s.condition == "Y" ? "Y" : "T";
        double v = ax == "Y" ? s.y : ax == "T" ? s.t : 0;
        m.push_back({f, ax, session_of(ax, v), v, i});
    }
    auto rank = [](const Meta& a) { return a.axis == "C" ? 0 : a.axis == "T" ? 1 : 2; };
    std::stable_sort(m.begin(), m.end(), [&](const Meta& a, const Meta& b) { return rank(a) != rank(b) ? rank(a) < rank(b) : a.file < b.file; });
    return m;
}

// 촬영 순서 (data/clean 에 시각이 없음): 구간마다 |값| 오름차순, Y2- 만 내림차순. 원본 BMP 파일 시각으로 확인한 방향이다.
inline int capture_index(const Meta& a) {
    if (a.axis == "T") return (int)std::lround(std::abs(a.value) * 100) - 1;
    int st = (int)std::lround(std::abs(a.value) / 0.05), i = st % 2 ? (st - 1) / 2 : st / 2 - 1;
    return a.session == "Y2-" ? 19 - i : i;
}

struct Split { std::string protocol, fold; std::vector<int> tr, te; };

// numpy RandomState.shuffle (legacy random_interval)
inline void np_shuffle(std::vector<int>& x, std::mt19937& g) {
    for (int i = (int)x.size() - 1; i >= 1; --i) {
        uint32_t mask = (uint32_t)i;
        for (int s : {1, 2, 4, 8, 16}) mask |= mask >> s;
        uint32_t v;
        while ((v = (uint32_t)g() & mask) > (uint32_t)i) {}
        std::swap(x[i], x[v]);
    }
}
// sklearn StratifiedKFold(n, shuffle=True, random_state=seed)._make_test_folds
inline std::vector<int> strat_folds(const std::vector<std::string>& y, int n_splits, uint32_t seed) {
    std::vector<std::string> cls;  // 처음 나온 순서
    std::vector<int> enc(y.size());
    for (size_t i = 0; i < y.size(); ++i) {
        auto it = std::find(cls.begin(), cls.end(), y[i]);
        if (it == cls.end()) { cls.push_back(y[i]); it = cls.end() - 1; }
        enc[i] = (int)(it - cls.begin());
    }
    int nc = (int)cls.size();
    std::vector<int> order = enc;
    std::sort(order.begin(), order.end());
    std::vector<std::vector<int>> alloc(n_splits, std::vector<int>(nc, 0));
    for (int i = 0; i < n_splits; ++i)
        for (size_t j = i; j < order.size(); j += n_splits) alloc[i][order[j]]++;
    std::mt19937 g(seed);
    std::vector<int> tf(y.size());
    for (int k = 0; k < nc; ++k) {
        std::vector<int> f;
        for (int i = 0; i < n_splits; ++i) for (int r = 0; r < alloc[i][k]; ++r) f.push_back(i);
        np_shuffle(f, g);
        size_t j = 0;
        for (size_t i = 0; i < y.size(); ++i) if (enc[i] == k) tf[i] = f[j++];
    }
    return tf;
}

inline std::vector<int> diff(const std::vector<int>& ids, const std::vector<int>& te) {
    std::vector<int> o;
    for (int i : ids) if (!std::binary_search(te.begin(), te.end(), i)) o.push_back(i);
    return o;
}

// compare.protocols (25 프로토콜, 119 분할)
inline std::vector<Split> protocols(const std::vector<Meta>& m) {
    std::vector<int> ids;
    for (int i = 0; i < (int)m.size(); ++i) if (m[i].axis != "C") ids.push_back(i);
    std::vector<Split> out;
    out.push_back({"full280", "all", ids, ids});
    for (int k = 0; k < 2; ++k) {  // interleaved: 축마다 촬영 순서로 번갈아
        std::vector<int> te;
        for (std::string a : {"Y", "T"}) {
            std::vector<int> ix;
            for (int i : ids) if (m[i].axis == a) ix.push_back(i);
            std::vector<std::pair<std::pair<std::string, int>, int>> key;  // (구간 순번, 구간 안 순번) -> 홀짝 (구간 크기 모두 짝수)
            for (int i : ix) if (capture_index(m[i]) % 2 == k) te.push_back(i);
        }
        std::sort(te.begin(), te.end());
        out.push_back({"interleaved", "f" + std::to_string(k), diff(ids, te), te});
    }
    std::vector<std::string> ys;
    for (int i : ids) ys.push_back(m[i].session);
    {
        auto tf = strat_folds(ys, 5, 20261014u);
        for (int k = 0; k < 5; ++k) {
            std::vector<int> te;
            for (size_t j = 0; j < ids.size(); ++j) if (tf[j] == k) te.push_back(ids[j]);
            out.push_back({"stratified5", "f" + std::to_string(k), diff(ids, te), te});
        }
    }
    {
        std::vector<std::string> order;  // m.session[...].unique() 순서
        for (int i : ids) if (std::find(order.begin(), order.end(), m[i].session) == order.end()) order.push_back(m[i].session);
        for (int k = 0; k < 5; ++k) {
            std::vector<int> te;
            for (auto& s : order) {
                std::vector<int> ii;
                for (int i = 0; i < (int)m.size(); ++i) if (m[i].session == s) ii.push_back(i);
                std::stable_sort(ii.begin(), ii.end(), [&](int a, int b) { return m[a].value < m[b].value; });
                int n = (int)ii.size(), base = n / 5, rem = n % 5, st = 0;
                for (int q = 0; q < k; ++q) st += base + (q < rem);
                for (int q = st; q < st + base + (k < rem); ++q) te.push_back(ii[q]);
            }
            std::sort(te.begin(), te.end());
            out.push_back({"blocked_within_session5", std::to_string(k), diff(ids, te), te});
        }
    }
    for (int seed = 0; seed < 20; ++seed) {
        auto tf = strat_folds(ys, 5, 7000u + seed);
        for (int k = 0; k < 5; ++k) {
            std::vector<int> te;
            for (size_t j = 0; j < ids.size(); ++j) if (tf[j] == k) te.push_back(ids[j]);
            out.push_back({"repeated5_seed" + std::to_string(seed), std::to_string(k), diff(ids, te), te});
        }
    }
    for (std::string s : {"T+", "T-", "Y1+", "Y1-", "Y2+", "Y2-"}) {
        std::vector<int> te;
        for (int i : ids) if (m[i].session == s) te.push_back(i);
        out.push_back({"session_out_missing_origin", s, diff(ids, te), te});
    }
    return out;
}

// 분할 정의 지문 (FNV-1a 64): "protocol|fold|test 파일들;" 를 이어 붙인 것. 원래 splits.json 과 비교한다.
inline uint64_t fingerprint(const std::vector<Split>& sp, const std::vector<Meta>& m) {
    uint64_t h = 1469598103934665603ull;
    auto add = [&](const std::string& s) { for (unsigned char c : s) { h ^= c; h *= 1099511628211ull; } };
    for (auto& s : sp) {
        add(s.protocol + "|" + s.fold + "|");
        for (int i : s.te) add(m[i].file + ",");
        add(";");
    }
    return h;
}

inline cv::Mat lstsq(const cv::Mat& A, const cv::Mat& B) { cv::Mat X; cv::solve(A, B, X, cv::DECOMP_SVD); return X; }
inline int rank_of(const cv::Mat& A) {
    cv::Mat w;
    cv::SVD::compute(A, w);
    double tol = w.at<double>(0) * std::max(A.rows, A.cols) * 2.220446049250313e-16;
    int r = 0;
    for (int i = 0; i < w.rows; ++i) r += w.at<double>(i) > tol;
    return r;
}
// 상삼각 W, W^T W = inv(C)
inline cv::Mat chol_inv_root(const cv::Mat& C) {
    cv::Mat Q = C.inv(cv::DECOMP_SVD);
    int n = Q.rows;
    cv::Mat L = cv::Mat::zeros(n, n, CV_64F);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j <= i; ++j) {
            double s = Q.at<double>(i, j);
            for (int k = 0; k < j; ++k) s -= L.at<double>(i, k) * L.at<double>(j, k);
            L.at<double>(i, j) = i == j ? std::sqrt(s) : s / L.at<double>(j, j);
        }
    return L.t();
}
// 방해 무늬 U 를 뺀 화이트닝: W - (WU) pinv(WU) W  (null_space 투영과 같은 이차형식)
inline cv::Mat project_out(const cv::Mat& W, const cv::Mat& U) {
    cv::Mat A = W * U, P;
    cv::invert(A, P, cv::DECOMP_SVD);
    return W - A * P * W;
}

struct Branch { std::string name, axis; cv::Mat base, s, c; };  // 각 1 x 12
struct SixModel { std::vector<Branch> br; cv::Mat W; };

// mode: "inferred_session"(구간마다 0점 6후보), "oracle_zero"(시험 사진의 정답 구간 0점, 진단용), "free"(축마다 절편), "center"(절편 0)
inline SixModel fit(const std::vector<Meta>& m, const std::vector<std::vector<double>>& X, const std::vector<int>& tr, const std::vector<double>& lab,
                    const std::string& mode, bool quad, const cv::Mat* U = nullptr) {
    SixModel md;
    std::vector<cv::Mat> res;
    int prank = 0;
    for (std::string a : {"Y", "T"}) {
        std::vector<int> ii;
        for (int i : tr) if (m[i].axis == a) ii.push_back(i);
        std::vector<std::string> ses;
        for (int i : ii) ses.push_back(m[i].session);
        std::sort(ses.begin(), ses.end());
        ses.erase(std::unique(ses.begin(), ses.end()), ses.end());
        bool use = mode == "inferred_session" || mode == "oracle_zero";
        int nb = quad ? 2 : 1, nc = nb + (use ? (int)ses.size() : mode == "free" ? 1 : 0);
        cv::Mat A((int)ii.size(), nc, CV_64F, 0.0), Z((int)ii.size(), 12, CV_64F);
        for (int r = 0; r < A.rows; ++r) {
            double v = lab[ii[r]];
            A.at<double>(r, 0) = v;
            if (quad) A.at<double>(r, 1) = v * v;
            if (use) A.at<double>(r, nb + (int)(std::find(ses.begin(), ses.end(), m[ii[r]].session) - ses.begin())) = 1;
            else if (mode == "free") A.at<double>(r, nb) = 1;
            for (int j = 0; j < 12; ++j) Z.at<double>(r, j) = X[ii[r]][j];
        }
        cv::Mat co = lstsq(A, Z);
        prank += rank_of(A);
        res.push_back(Z - A * co);
        cv::Mat s = co.row(0).clone(), c = quad ? co.row(1).clone() : cv::Mat::zeros(1, 12, CV_64F);
        if (use)
            for (size_t j = 0; j < ses.size(); ++j) md.br.push_back({ses[j], a, co.row(nb + (int)j).clone(), s, c});
        else
            md.br.push_back({a, a, mode == "free" ? co.row(nb).clone() : cv::Mat::zeros(1, 12, CV_64F), s, c});
    }
    cv::Mat R;
    cv::vconcat(res, R);
    cv::Mat S = R.t() * R / std::max(R.rows - prank, 1);
    cv::Mat sd(12, 1, CV_64F), C(12, 12, CV_64F);
    for (int i = 0; i < 12; ++i) sd.at<double>(i) = std::sqrt(std::max(S.at<double>(i, i), 1e-14));
    for (int i = 0; i < 12; ++i)
        for (int j = 0; j < 12; ++j) C.at<double>(i, j) = 0.999 * S.at<double>(i, j) / sd.at<double>(i) / sd.at<double>(j) + (i == j ? 0.001 : 0);
    cv::Mat W = chol_inv_root(C);
    for (int i = 0; i < 12; ++i)
        for (int j = 0; j < 12; ++j) W.at<double>(i, j) /= sd.at<double>(j);
    if (U) W = project_out(W, *U);
    md.W = W;
    return md;
}

struct Cand { std::string branch, axis; double v, cost; bool allowed; };
inline Cand solve_branch(const Branch& b, const cv::Mat& W, const std::vector<double>& x, bool sign_rule) {
    cv::Mat f(1, 12, CV_64F);
    for (int j = 0; j < 12; ++j) f.at<double>(j) = x[j];
    cv::Mat Z = (f - b.base) * W.t(), sw = b.s * W.t(), cw = b.c * W.t();
    double v = Z.dot(sw) / sw.dot(sw);
    for (int it = 0; it < 8; ++it) {
        cv::Mat J = sw + 2 * v * cw, r = Z - v * sw - v * v * cw;
        v += J.dot(r) / std::max(J.dot(J), 1e-20);
    }
    cv::Mat r = Z - v * sw - v * v * cw;
    double cost = r.dot(r);
    bool ok = true;
    if (sign_rule && b.name.size() > 1) ok = v * (b.name.back() == '+' ? 1 : -1) >= 0;
    return {b.name, b.axis, v, cost, ok};
}

struct Pred { int i; std::string axis, pred_axis, branch; double value, pred, cost, costY, costT; };
inline Pred predict_one(const std::vector<Meta>& m, const std::vector<double>& x, int i, const SixModel& md, const std::string& mode,
                        std::vector<Cand>* all = nullptr) {
    double best = std::numeric_limits<double>::infinity(), bY = best, bT = best;
    Cand pick{"", "", 0, best, false};
    bool first = true;
    for (auto& b : md.br) {
        Cand c = solve_branch(b, md.W, x, mode == "inferred_session");
        double cost = c.allowed ? c.cost : std::numeric_limits<double>::infinity();
        if (all) all->push_back(c);
        if (first || cost < best) { best = cost; pick = c; first = false; }
        if (c.axis == "Y") bY = std::min(bY, cost); else bT = std::min(bT, cost);
    }
    return {i, m[i].axis, pick.axis, pick.branch, m[i].value, pick.v, best, bY, bT};
}

// compare.score 의 핵심 지표
struct Score {
    int n = 0, correct = 0, failed = 0, Ywrong = 0, Twrong = 0, Ycorrect = 0, Tcorrect = 0;
    double YMAE = hye::kNaN, Ymax = hye::kNaN, TMAE = hye::kNaN, Tmax = hye::kNaN;
};
inline Score score(const std::vector<Pred>& P) {
    Score s;
    std::vector<double> eY, eT;
    for (auto& p : P) {
        ++s.n;
        bool ok = p.axis == p.pred_axis, sign = (p.value > 0) - (p.value < 0) == (p.pred > 0) - (p.pred < 0);
        double e = std::abs(p.value - p.pred);
        double tY = p.axis == "Y" ? p.value : 0, tT = p.axis == "T" ? p.value : 0, qY = p.pred_axis == "Y" ? p.pred : 0, qT = p.pred_axis == "T" ? p.pred : 0;
        bool corr = ok && sign;
        s.correct += corr;
        s.failed += !corr || std::abs(qY - tY) > 0.09 || std::abs(qT - tT) > 0.09;
        if (p.axis == "Y") { s.Ywrong += !ok; s.Ycorrect += corr; if (ok) eY.push_back(e); }
        else { s.Twrong += !ok; s.Tcorrect += corr; if (ok) eT.push_back(e); }
    }
    if (!eY.empty()) { s.YMAE = hye::mean(eY); s.Ymax = hye::max_abs(eY); }
    if (!eT.empty()) { s.TMAE = hye::mean(eT); s.Tmax = hye::max_abs(eT); }
    return s;
}
inline bool pass09(const Pred& p) {
    bool ok = p.axis == p.pred_axis, sign = (p.value > 0) - (p.value < 0) == (p.pred > 0) - (p.pred < 0);
    double tY = p.axis == "Y" ? p.value : 0, tT = p.axis == "T" ? p.value : 0, qY = p.pred_axis == "Y" ? p.pred : 0, qT = p.pred_axis == "T" ? p.pred : 0;
    return ok && sign && std::abs(qY - tY) <= 0.09 && std::abs(qT - tT) <= 0.09;
}

}  // namespace six
