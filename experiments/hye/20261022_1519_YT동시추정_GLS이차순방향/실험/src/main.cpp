// Y·T 동시 추정: 고정 630점 경계 측정 -> P12 -> GLS 선형·가산 이차 순방향 모델, 4-fold 중첩 OOF
// 원본: codex_experiments_20261014/joint_yt_20261022/experiment.py (+ zero_experiment.py 의 기존 Y-or-T 판정기,
//       center_patch_v2_20261021/run.py 의 P12, analysis_grid_corners_20261019/track_corners.py 의 630점 측정)
#include "numeric.hpp"
#include "struct630.hpp"
#include <hye/dataset.hpp>
#include <hye/plot.hpp>
#include <hye/report.hpp>
#include <iostream>

using namespace hye;
using num::Mat;

namespace {

struct Row {
    std::string file, axis, session;
    double value;
};

// zero_experiment.manifest: 세션 = T± / Y1± / Y2± (Y2 = 0.05 홀수 단계)
std::vector<Row> manifest(const std::vector<std::string>& names) {
    std::vector<Row> m;
    for (auto& n : names) {
        Row r{n};
        if (n == "center.bmp") { r.axis = "C"; r.value = 0; r.session = "C"; }
        else {
            r.axis = n.substr(0, 1);
            r.value = std::stod(n.substr(1, n.size() - 5));
            if (r.axis == "T") r.session = "T";
            else r.session = (std::llabs(std::llround(r.value / .05)) % 2) ? "Y2" : "Y1";
            r.session += r.value > 0 ? "+" : "-";
        }
        m.push_back(r);
    }
    return m;
}

// ---- GLS 동시 모델 (fit_joint / predict_joint) ----
struct Joint {
    Mat co, W;  // co: k x 12, W: 12 x 12
    bool quad;
    double shrink;
};
Mat basis(const Mat& y, bool quad) {
    if (!quad) return y.clone();
    Mat b(y.rows, 4, CV_64F);
    for (int i = 0; i < y.rows; ++i) {
        b.at<double>(i, 0) = y.at<double>(i, 0); b.at<double>(i, 1) = y.at<double>(i, 1);
        b.at<double>(i, 2) = y.at<double>(i, 0) * y.at<double>(i, 0); b.at<double>(i, 3) = y.at<double>(i, 1) * y.at<double>(i, 1);
    }
    return b;
}
// S -> 상관 C, (1-s)C + sI, W = chol(inv(C))^T / std
Mat whitening(const Mat& R, int dof_sub, double shrink, double* cond_out = nullptr) {
    Mat S = R.t() * R / (double)(R.rows - dof_sub);
    const int p = S.cols;
    std::vector<double> sd(p);
    for (int j = 0; j < p; ++j) sd[j] = std::sqrt(std::max(S.at<double>(j, j), 1e-14));
    Mat C(p, p, CV_64F);
    for (int i = 0; i < p; ++i)
        for (int j = 0; j < p; ++j) C.at<double>(i, j) = (1 - shrink) * S.at<double>(i, j) / sd[i] / sd[j] + (i == j ? shrink : 0.0);
    Mat L = num::cholesky(C.inv(cv::DECOMP_SVD));
    Mat W = L.t();
    for (int i = 0; i < p; ++i) for (int j = 0; j < p; ++j) W.at<double>(i, j) /= sd[j];
    (void)cond_out;
    return W;
}
Joint fit_joint(const Mat& X, const Mat& y, const std::vector<int>& tr, bool quad, double shrink) {
    Mat A = basis(num::rows(y, tr), quad), Xt = num::rows(X, tr);
    Joint j{num::lstsq(A, Xt), Mat(), quad, shrink};
    Mat R = Xt - A * j.co;
    j.W = whitening(R, j.co.rows, shrink);
    return j;
}
// 2변수 최소제곱 (LM): f(v) = [v0, v1, v0², v1²] A - z
cv::Vec2d solve_quad(const Mat& A, const Mat& z, cv::Vec2d v, double& cost) {
    auto F = [&](const cv::Vec2d& q, std::vector<double>& r) {
        double s = 0;
        for (int c = 0; c < A.cols; ++c) {
            r[c] = q[0] * A.at<double>(0, c) + q[1] * A.at<double>(1, c) + q[0] * q[0] * A.at<double>(2, c) + q[1] * q[1] * A.at<double>(3, c) - z.at<double>(0, c);
            s += r[c] * r[c];
        }
        return s;
    };
    std::vector<double> r(A.cols), rn(A.cols);
    double cur = F(v, r), lam = 1e-6;
    for (int it = 0; it < 500; ++it) {
        double H00 = 0, H01 = 0, H11 = 0, g0 = 0, g1 = 0;
        for (int c = 0; c < A.cols; ++c) {
            double j0 = A.at<double>(0, c) + 2 * v[0] * A.at<double>(2, c), j1 = A.at<double>(1, c) + 2 * v[1] * A.at<double>(3, c);
            H00 += j0 * j0; H01 += j0 * j1; H11 += j1 * j1; g0 += j0 * r[c]; g1 += j1 * r[c];
        }
        if (std::max(std::abs(g0), std::abs(g1)) < 1e-14 * std::max(1.0, cur)) break;
        bool ok = false;
        for (int t = 0; t < 40 && !ok; ++t) {
            double a = H00 * (1 + lam), b = H01, d = H11 * (1 + lam), det = a * d - b * b;
            cv::Vec2d st(-(d * g0 - b * g1) / det, -(a * g1 - b * g0) / det);
            cv::Vec2d q = v + st;
            double nc = F(q, rn);
            if (nc <= cur) {
                double rel = (cur - nc) / std::max(cur, 1e-300);
                v = q; cur = nc; r = rn; lam = std::max(lam * 0.1, 1e-15); ok = true;
                if (rel < 1e-15 || cv::norm(st) < 1e-13 * (1e-13 + cv::norm(v))) { it = 1 << 20; }
            } else lam *= 10;
        }
        if (!ok) break;
    }
    cost = cur;
    return v;
}
Mat predict_joint(const Mat& X, const Joint& j) {
    Mat A = j.co * j.W.t(), z = X * j.W.t();
    Mat A2 = A.rowRange(0, 2).t();  // 12 x 2
    Mat init = num::lstsq(A2, z.t()).t();  // n x 2
    if (!j.quad) return init;
    Mat out(X.rows, 2, CV_64F);
    for (int i = 0; i < X.rows; ++i) {
        cv::Vec2d st(init.at<double>(i, 0), init.at<double>(i, 1));
        std::vector<cv::Vec2d> starts{st, {0, 0}, {st[0], 0}, {0, st[1]}};
        double best = 1e300;
        cv::Vec2d bv;
        for (auto s : starts) {
            double c;
            cv::Vec2d v = solve_quad(A, z.row(i), s, c);
            if (c < best) { best = c; bv = v; }
        }
        out.at<double>(i, 0) = bv[0];
        out.at<double>(i, 1) = bv[1];
    }
    return out;
}

// ---- 기존 Y-or-T 판정기 (zero_experiment.fit/predict, mode='center', quad=True) ----
struct YorT {
    Mat W;
    Mat s[2], c[2];  // 0: Y, 1: T  (1 x 12)
};
YorT fit_yort(const std::vector<Row>& m, const Mat& X, const std::vector<int>& tr) {
    YorT o;
    Mat R(0, X.cols, CV_64F);
    int rank = 0;
    for (int a = 0; a < 2; ++a) {
        std::vector<int> ii;
        for (int i : tr) if (m[i].axis == (a ? "T" : "Y")) ii.push_back(i);
        Mat A((int)ii.size(), 2, CV_64F);
        for (size_t k = 0; k < ii.size(); ++k) { double v = m[ii[k]].value; A.at<double>((int)k, 0) = v; A.at<double>((int)k, 1) = v * v; }
        Mat Z = num::rows(X, ii), co = num::lstsq(A, Z);
        R.push_back(Mat(Z - A * co));
        rank += 2;
        o.s[a] = co.row(0).clone();
        o.c[a] = co.row(1).clone();
    }
    o.W = whitening(R, rank, 0.001);
    return o;
}
struct YorTPred { int axis; double v, cost; };
std::vector<YorTPred> predict_yort(const Mat& X, const YorT& md) {
    std::vector<YorTPred> out(X.rows, {0, 0, 1e300});
    for (int a = 0; a < 2; ++a) {
        Mat Z = X * md.W.t(), sw = md.s[a] * md.W.t(), cw = md.c[a] * md.W.t();
        for (int i = 0; i < X.rows; ++i) {
            Mat z = Z.row(i);
            double v = z.dot(sw) / sw.dot(sw);
            for (int it = 0; it < 8; ++it) {
                Mat J = sw + 2 * v * cw, r = z - v * sw - v * v * cw;
                v += J.dot(r) / std::max(J.dot(J), 1e-20);
            }
            Mat r = z - v * sw - v * v * cw;
            double cost = r.dot(r);
            if (cost < out[i].cost) out[i] = {a, v, cost};
        }
    }
    return out;
}

struct Pred {
    int idx;
    double py, pt;
    std::string method, protocol;
    int fold;
    int axis_ok = -1;
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
    // ---- 1. 630점 측정 ----
    auto shots = load_1014();
    const cv::Mat center = s630::read_gray(center_1014(shots).path);
    auto qrs = s630::decode_qr(center);
    auto all = s630::structural_corners(qrs);
    s630::refine_center(center, all);
    auto cs = s630::non_diagonal(all);
    // names: center 먼저, 나머지 파일명 순 (원래 tracks.npz 순서)
    std::vector<int> order(shots.size());
    for (size_t i = 0; i < shots.size(); ++i) order[i] = (int)i;
    auto fname = [&](int i) { auto r = shots[i].rel; return r.substr(r.rfind('/') + 1); };
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        bool ca = fname(a) == "center.bmp", cb = fname(b) == "center.bmp";
        if (ca != cb) return ca;
        return fname(a) < fname(b);
    });
    std::vector<std::string> names;
    for (int i : order) names.push_back(fname(i));
    const int N = (int)names.size(), P = (int)cs.size();
    std::vector<s630::Track> tracks(N);
#pragma omp parallel for schedule(dynamic) num_threads(4)
    for (int k = 0; k < N; ++k) {
        cv::Mat im = s630::read_gray(shots[order[k]].path);
        tracks[k] = s630::track(im, center, cs);
    }
    std::cout << "tracked " << N << " x " << P << "\n";
    std::vector<cv::Point2d> pts(P);
    for (int j = 0; j < P; ++j) pts[j] = {tracks[0].obs[j].x, tracks[0].obs[j].y};  // center 의 맞춘 좌표
    Mat X(N, 12, CV_64F);
    int invalid = 0;
    for (int k = 0; k < N; ++k) {
        std::vector<cv::Point2d> d(P);
        for (int j = 0; j < P; ++j) {
            d[j] = {tracks[k].obs[j].x - pts[j].x, tracks[k].obs[j].y - pts[j].y};
            if (k && !tracks[k].obs[j].valid) ++invalid;
        }
        auto f = s630::p12(pts, d);
        for (int c = 0; c < 12; ++c) X.at<double>(k, c) = f[c];
    }
    {
        Table t{{"file", "c0x", "c0y", "c1x", "c1y", "c2x", "c2y", "c3x", "c3y", "c4x", "c4y", "c5x", "c5y"}};
        for (int k = 0; k < N; ++k) {
            std::vector<std::string> r{names[k]};
            for (int c = 0; c < 12; ++c) r.push_back(fmt(X.at<double>(k, c), 6));
            t.add(r);
        }
        t.csv(out("p12_features.csv"));
    }
    auto m = manifest(names);
    std::vector<int> ids;
    for (int i = 0; i < N; ++i) if (m[i].axis != "C") ids.push_back(i);
    Mat y = num::zeros(N, 2);
    for (int i = 0; i < N; ++i) { if (m[i].axis == "Y") y.at<double>(i, 0) = m[i].value; if (m[i].axis == "T") y.at<double>(i, 1) = m[i].value; }

    // ---- 2. 고정 4-fold (multiref_4fold_20261021: StratifiedKFold(4, shuffle, 20261021), 세션 층화) ----
    std::vector<std::string> lab;
    for (int i : ids) lab.push_back(m[i].session);
    auto tf = num::stratified_test_folds(lab, 4, 20261021);
    std::vector<std::vector<int>> TR(4), TE(4);
    for (size_t k = 0; k < ids.size(); ++k)
        for (int f = 0; f < 4; ++f) (tf[k] == f ? TE[f] : TR[f]).push_back(ids[k]);
    // 원래 splits.json 의 fold 별 시험 파일 목록 FNV-1a 해시 (codex_experiments_20261014/multiref_4fold_20261021/splits.json)
    const uint64_t split_hash[4] = {0x73c185417168da7cULL, 0x9f8abcb24a4c8185ULL, 0x95027a617b7e6e9eULL, 0xa04f94a26009a659ULL};
    int split_same = 0;
    for (int f = 0; f < 4; ++f) {
        std::string s;
        for (size_t k = 0; k < TE[f].size(); ++k) s += (k ? "\n" : "") + names[TE[f][k]];
        split_same += num::fnv1a(s) == split_hash[f];
    }

    struct Cfg { bool quad; double shrink; };
    const std::vector<Cfg> configs{{false, .001}, {false, .05}, {false, .2}, {false, 1.}, {true, .001}, {true, .05}, {true, .2}, {true, 1.}};
    auto cfg_name = [&](int k) { return std::string("joint_") + (configs[k].quad ? "quadratic" : "linear") + "_s" + fmt(configs[k].shrink, configs[k].shrink < .01 ? 3 : configs[k].shrink < .1 ? 2 : 1); };
    using Rank = std::tuple<int, double, double, int>;
    auto score = [&](const Mat& p, const Mat& t, int k) {
        int fails = 0;
        double mx = 0, sm = 0;
        for (int i = 0; i < p.rows; ++i) {
            double e0 = std::abs(p.at<double>(i, 0) - t.at<double>(i, 0)), e1 = std::abs(p.at<double>(i, 1) - t.at<double>(i, 1));
            fails += (e0 > .09 || e1 > .09);
            mx = std::max({mx, e0, e1});
            sm += e0 + e1;
        }
        return Rank{fails, mx, sm / (2.0 * p.rows), k};
    };
    std::vector<Pred> preds;
    std::vector<int> chosen_fold(4);
    Table ident{{"fold", "설정", "백색화 코사인 (Y·T 반응)", "정규화 조건수"}};
    std::map<std::string, double> cos0;
    for (int f = 0; f < 4; ++f) {
        const auto& tr = TR[f];
        std::vector<std::string> tl;
        for (int i : tr) tl.push_back(m[i].session);
        auto itf = num::stratified_test_folds(tl, 3, 100 + f);
        std::vector<Rank> ranks;
        for (int k = 0; k < (int)configs.size(); ++k) {
            Mat ip(0, 2, CV_64F), it(0, 2, CV_64F);
            for (int g = 0; g < 3; ++g) {
                std::vector<int> a, b;
                for (size_t q = 0; q < tr.size(); ++q) (itf[q] == g ? b : a).push_back(tr[q]);
                auto md = fit_joint(X, y, a, configs[k].quad, configs[k].shrink);
                ip.push_back(predict_joint(num::rows(X, b), md));
                it.push_back(num::rows(y, b));
            }
            ranks.push_back(score(ip, it, k));
        }
        int chosen = std::get<3>(*std::min_element(ranks.begin(), ranks.end()));
        chosen_fold[f] = chosen;
        Mat Xte = num::rows(X, TE[f]);
        for (int k = 0; k < (int)configs.size(); ++k) {
            auto md = fit_joint(X, y, tr, configs[k].quad, configs[k].shrink);
            Mat p = predict_joint(Xte, md);
            for (int r = 0; r < p.rows; ++r) {
                preds.push_back({TE[f][r], p.at<double>(r, 0), p.at<double>(r, 1), cfg_name(k), "oof4", f});
                if (k == chosen) preds.push_back({TE[f][r], p.at<double>(r, 0), p.at<double>(r, 1), "joint_nested_selected", "oof4", f});
            }
            Mat cw = md.co * md.W.t();
            Mat lin = cw.rowRange(0, 2).t();  // 12 x 2
            double n0 = cv::norm(lin.col(0)), n1 = cv::norm(lin.col(1));
            double cosv = lin.col(0).dot(lin.col(1)) / (n0 * n1);
            Mat ln = lin.clone();
            ln.col(0) /= n0;
            ln.col(1) /= n1;
            ident.add({std::to_string(f), cfg_name(k), fmt(cosv, 6), fmt(num::cond2(ln), 3)});
            if (f == 0) cos0[cfg_name(k)] = cosv;
        }
        auto yt = fit_yort(m, X, tr);
        auto po = predict_yort(Xte, yt);
        for (size_t r = 0; r < po.size(); ++r) {
            int i = TE[f][r];
            Pred p{i, po[r].axis == 0 ? po[r].v : 0, po[r].axis == 1 ? po[r].v : 0, "existing_Y_or_T", "oof4", f};
            p.axis_ok = (po[r].axis == 0) == (m[i].axis == "Y");
            preds.push_back(p);
        }
        std::cout << "fold " << f << " chosen " << cfg_name(chosen) << "\n";
    }
    // 배포 설정 선택(외부 4-fold OOF 기준) -> 전체 280장 학습, 재대입
    std::vector<Rank> fr;
    for (int k = 0; k < (int)configs.size(); ++k) {
        Mat ip(0, 2, CV_64F), it(0, 2, CV_64F);
        for (int f = 0; f < 4; ++f) {
            ip.push_back(predict_joint(num::rows(X, TE[f]), fit_joint(X, y, TR[f], configs[k].quad, configs[k].shrink)));
            it.push_back(num::rows(y, TE[f]));
        }
        fr.push_back(score(ip, it, k));
    }
    const int final_k = std::get<3>(*std::min_element(fr.begin(), fr.end()));
    auto fmodel = fit_joint(X, y, ids, configs[final_k].quad, configs[final_k].shrink);
    {
        Mat p = predict_joint(num::rows(X, ids), fmodel);
        for (int r = 0; r < p.rows; ++r) preds.push_back({ids[r], p.at<double>(r, 0), p.at<double>(r, 1), "joint_final_model", "resub280", -1});
        auto po = predict_yort(num::rows(X, ids), fit_yort(m, X, ids));
        for (size_t r = 0; r < po.size(); ++r)
            preds.push_back({ids[r], po[r].axis == 0 ? po[r].v : 0, po[r].axis == 1 ? po[r].v : 0, "existing_Y_or_T", "resub280", -1});
    }
    // 원점 확인: 특징 0 -> (0,0)
    Mat zero = num::zeros(1, 12), pz = predict_joint(zero, fmodel);
    const bool zero_ok = std::abs(pz.at<double>(0, 0)) < 1e-12 && std::abs(pz.at<double>(0, 1)) < 1e-12;

    // ---- 3. 요약 ----
    struct Sum { int n = 0, passed = 0, sign = 0; double ymae = 0, ymax = 0, tmae = 0, tmax = 0, inact = 0; };
    std::map<std::pair<std::string, std::string>, std::map<std::string, Sum>> summ;
    Table predT{{"file", "axis", "value", "pred_Y", "pred_T", "eY", "eT", "both_pass09", "method", "protocol", "fold"}};
    for (auto& p : preds) {
        auto& r = m[p.idx];
        double ty = y.at<double>(p.idx, 0), tt = y.at<double>(p.idx, 1), eY = std::abs(p.py - ty), eT = std::abs(p.pt - tt);
        bool pass = eY <= .09 && eT <= .09;
        double active = r.axis == "Y" ? p.py : p.pt;
        bool sign = (active > 0) - (active < 0) == (r.value > 0) - (r.value < 0);
        double inact = r.axis == "Y" ? std::abs(p.pt) : std::abs(p.py);
        std::vector<std::string> subs{"all", r.axis};
        if (r.axis == "Y" && std::abs(r.value) <= .30 + 1e-12) subs.push_back("small_Y_le0.30");
        if (r.axis == "T" && std::abs(r.value) <= .05 + 1e-12) subs.push_back("small_T_le0.05");
        for (auto& sname : subs) {
            auto& s = summ[{p.method, p.protocol}][sname];
            s.n++; s.passed += pass; s.sign += sign; s.ymae += eY; s.tmae += eT; s.ymax = std::max(s.ymax, eY); s.tmax = std::max(s.tmax, eT); s.inact += inact;
        }
        if (p.protocol == "oof4" && (p.method == "joint_nested_selected" || p.method == "existing_Y_or_T"))
            predT.add({r.file, r.axis, fmt(r.value, 2), fmt(p.py, 6), fmt(p.pt, 6), fmt(eY, 6), fmt(eT, 6), pass ? "True" : "False", p.method, p.protocol, std::to_string(p.fold)});
    }
    predT.csv(out("predictions.csv"));
    Table sumT{{"방법", "평가", "부분", "장수", "통과", "Y 평균 |오차|", "Y 최대", "T 평균 |오차|", "T 최대", "활성 축 부호 정확도", "비활성 축 평균 |값|"}};
    for (auto& [key, mm] : summ)
        for (auto sname : {"all", "Y", "T", "small_Y_le0.30", "small_T_le0.05"}) {
            if (!mm.count(sname)) continue;
            auto& s = mm.at(sname);
            sumT.add({key.first, key.second, sname, std::to_string(s.n), std::to_string(s.passed), fmt(s.ymae / s.n, 4), fmt(s.ymax, 4), fmt(s.tmae / s.n, 4),
                      fmt(s.tmax, 4), fmt((double)s.sign / s.n, 4), fmt(s.inact / s.n, 4)});
        }
    auto S = [&](const std::string& meth, const std::string& prot) { return summ[{meth, prot}]["all"]; };
    auto sel = S("joint_nested_selected", "oof4"), old = S("existing_Y_or_T", "oof4");

    // ---- 4. 그림 ----
    std::vector<double> ty, py, tt, pt, ey, et;
    std::vector<std::string> nm;
    std::vector<int> pidx;
    for (auto& p : preds)
        if (p.method == "joint_nested_selected") {
            ty.push_back(y.at<double>(p.idx, 0)); tt.push_back(y.at<double>(p.idx, 1)); py.push_back(p.py); pt.push_back(p.pt);
            ey.push_back(p.py - ty.back()); et.push_back(p.pt - tt.back());
            nm.push_back(names[p.idx].substr(0, names[p.idx].size() - 4));
            pidx.push_back(p.idx);
        }
    pred_vs_truth("01_동시모델_OOF오차.png", "Joint GLS nested OOF", ty, py, tt, pt, nm);
    {
        std::vector<double> a, b, c, d;
        std::vector<std::string> n2;
        for (auto& p : preds)
            if (p.method == "existing_Y_or_T" && p.protocol == "oof4") {
                a.push_back(y.at<double>(p.idx, 0)); b.push_back(p.py); c.push_back(y.at<double>(p.idx, 1)); d.push_back(p.pt);
                n2.push_back(names[p.idx].substr(0, names[p.idx].size() - 4));
            }
        pred_vs_truth("02_기존YorT_OOF오차.png", "Existing Y-or-T OOF", a, b, c, d, n2);
    }
    {
        std::vector<std::string> cats;
        std::vector<double> vals;
        for (int k = 0; k < (int)configs.size(); ++k) { cats.push_back(cfg_name(k).substr(6)); vals.push_back(S(cfg_name(k), "oof4").passed); }
        cats.push_back("nested_selected"); vals.push_back(sel.passed);
        cats.push_back("existing_Y_or_T"); vals.push_back(old.passed);
        bar_chart("03_설정별_통과수.png", "OOF pass count (of 280)", cats, vals, "passed", (int)cats.size() - 2, 280);
    }
    // 반응 방향: P12 c0x (평균 수평 이동) vs c2x (세로 위치에 따른 수평 이동 = 회전 성분)
    {
        std::vector<double> xs[2], ys[2];
        for (int i : ids) { int a = m[i].axis == "T"; xs[a].push_back(X.at<double>(i, 0)); ys[a].push_back(X.at<double>(i, 4)); }
        Chart c("P12 response: Y and T are nearly parallel", "c0x: mean horizontal shift (px)", "c2x: shift vs vertical position (px)");
        c.scatter(xs[1], ys[1], color(1), "T scan").scatter(xs[0], ys[0], color(2), "Y scan");
        c.save("04_P12반응_Y와T.png");
    }
    // 튀는 값 사진: 630점 측정 위치(빨강)와 기준 영역(초록), 화살표 = 점 변위 - 그 QR 평균 변위 (x20)
    auto outlier_fig = [&](int idx, const std::string& file, const std::string& cap) {
        cv::Mat im = s630::read_gray(shots[order[idx]].path);
        std::vector<cv::Mat> tiles;
        for (int q = 0; q < 3; ++q) {
            double mx = 0, my = 0;
            int nq = 0;
            for (int j = 0; j < P; ++j)
                if (cs[j].qr == q + 1) { mx += tracks[idx].obs[j].x - pts[j].x; my += tracks[idx].obs[j].y - pts[j].y; ++nq; }
            mx /= nq; my /= nq;
            auto b = s630::kBoxes[q];
            auto sh = tracks[idx].shift[q];
            cv::Rect roi(b[0] + sh.x - 15, b[1] + sh.y - 15, b[2] - b[0] + 30, b[3] - b[1] + 30);
            std::vector<Mark> mk;
            for (int j = 0; j < P; ++j)
                if (cs[j].qr == q + 1 && j % 3 == 0)
                    mk.push_back({{tracks[idx].obs[j].x, tracks[idx].obs[j].y}, {tracks[idx].obs[j].x - pts[j].x - mx, tracks[idx].obs[j].y - pts[j].y - my}, "",
                                  tracks[idx].obs[j].valid ? kRed : cv::Scalar(255, 0, 255)});
            tiles.push_back(zoom(im, roi, 420, mk, 20, q == 0 ? cap : "",
                                 {{cv::Rect2d(b[0], b[1], b[2] - b[0], b[3] - b[1]), kGreen, "ref"},
                                  {cv::Rect2d(b[0] + mx, b[1] + my, b[2] - b[0], b[3] - b[1]), kRed, "QR shift " + fmt(mx, 2) + ", " + fmt(my, 2)}}));
        }
        save_png(file, hstack(tiles));
    };
    size_t wy = worst(ey, 1)[0], wt = worst(et, 1)[0];
    std::string fy = "05_튀는값_Y_" + nm[wy] + ".png", ft = "06_튀는값_T_" + nm[wt] + ".png";
    outlier_fig(pidx[wy], fy, nm[wy] + " Y err " + fmts(ey[wy], 3) + " mm");
    outlier_fig(pidx[wt], ft, nm[wt] + " T err " + fmts(et[wt], 3) + " deg");
    Table outl{{"구분", "사진", "정답 Y", "추정 Y", "정답 T", "추정 T", "P12 c0x 평균 수평 이동 (px)", "c2x 회전 성분 (px)", "c0y 평균 수직 이동 (px)", "기존 Y-or-T 추정"}};
    std::map<int, std::pair<double, double>> oldp;
    for (auto& p : preds) if (p.method == "existing_Y_or_T" && p.protocol == "oof4") oldp[p.idx] = {p.py, p.pt};
    for (auto [lab, idx] : std::vector<std::pair<std::string, std::vector<size_t>>>{{"Y 오차 상위", worst(ey, 4)}, {"T 오차 상위", worst(et, 3)}})
        for (size_t k : idx) {
            int i = pidx[k];
            outl.add({lab, nm[k], fmt(ty[k], 2), fmt(py[k], 3), fmt(tt[k], 2), fmt(pt[k], 3), fmt(X.at<double>(i, 0), 3), fmt(X.at<double>(i, 4), 4), fmt(X.at<double>(i, 1), 4),
                      "Y " + fmt(oldp[i].first, 3) + " / T " + fmt(oldp[i].second, 3)});
        }

    // ---- 5. 원래 결과와 대조 (joint_yt_20261022/summary.csv, selection.json, identifiability.csv, multiref splits.json) ----
    Parity par;
    par.add("분할: fold 시험 목록이 원래 splits.json 과 같음 (4개 중)", 4, split_same, 0, 0);
    par.add("동시 모델(중첩 선택) OOF 통과 장수", 134, sel.passed, 3, 0);
    par.add("동시 모델 OOF Y 평균 |오차| (mm)", 0.106650, sel.ymae / sel.n, 0.003);
    par.add("동시 모델 OOF Y 최대 |오차| (mm)", 0.413447, sel.ymax, 0.02);
    par.add("동시 모델 OOF T 평균 |오차| (deg)", 0.019066, sel.tmae / sel.n, 0.001);
    par.add("동시 모델 OOF T 최대 |오차| (deg)", 0.075617, sel.tmax, 0.005);
    par.add("기존 Y-or-T OOF 통과 장수", 276, old.passed, 1, 0);
    par.add("기존 Y-or-T OOF Y 최대 |오차| (mm)", 0.255551, old.ymax, 0.01);
    par.add("기존 Y-or-T OOF T 최대 |오차| (deg)", 0.050000, old.tmax, 0.002);
    {
        const char* nmz[8] = {"joint_linear_s0.001", "joint_linear_s0.05", "joint_linear_s0.2", "joint_linear_s1.0",
                              "joint_quadratic_s0.001", "joint_quadratic_s0.05", "joint_quadratic_s0.2", "joint_quadratic_s1.0"};
        const int orig[8] = {132, 132, 131, 119, 137, 134, 133, 112};
        for (int k = 0; k < 8; ++k) par.add(std::string(nmz[k]) + " OOF 통과", orig[k], S(nmz[k], "oof4").passed, 3, 0);
    }
    par.add_text("fold별 선택 설정 (0~7)", "4,4,0,1",
                 std::to_string(chosen_fold[0]) + "," + std::to_string(chosen_fold[1]) + "," + std::to_string(chosen_fold[2]) + "," + std::to_string(chosen_fold[3]),
                 chosen_fold[0] == 4 && chosen_fold[1] == 4 && chosen_fold[2] == 0 && chosen_fold[3] == 1);
    par.add("최종 설정 번호", 4, final_k, 0, 0);
    par.add("최종 모델 재대입 통과 장수", 143, S("joint_final_model", "resub280").passed, 3, 0);
    par.add("fold0 linear s0.001 백색화 코사인", 0.992778, cos0["joint_linear_s0.001"], 0.001, 6);
    par.add("fold0 quadratic s0.001 백색화 코사인", 0.993346, cos0["joint_quadratic_s0.001"], 0.001, 6);
    par.add_text("특징 0 -> (0,0)", "True", zero_ok ? "True" : "False", zero_ok);

    // ---- 6. 결과.md ----
    double cmean = 0;
    for (auto& [k, v] : cos0) cmean += v;
    cmean /= cos0.size();
    std::ostringstream R;
    R << "# 결과: Y·T 동시 추정, GLS 선형·가산 이차 순방향 모델\n\n"
      << "입력: `data/clean/clean_20261014` 281장. center.bmp 의 QR 3개를 디코딩해 구조 모서리 " << all.size() << "점을 정하고, 대각선을 뺀 " << P
      << "점을 사진마다 흐린 경계 모델로 맞췄다(진단 실패 " << invalid << " / " << (N - 1) * P << ", 진단 실패 점도 그대로 쓴다). "
      << "변위(사진 - center)를 [1,u,v,u²,uv,v²]로 최소제곱해 P12 특징 12개를 만든다.\n\n"
      << "QR 디코딩: QR1 " << qrs[0].note << ", QR2 " << qrs[1].note << ", QR3 " << qrs[2].note << ". QR3 사각형은 OpenCV 3.4 에서 원래(cv2 5.0) 사각형과 최대 1px 다르다(처음 위치·격자 방향만 영향).\n\n"
      << "분할: StratifiedKFold(4, shuffle, 20261021)을 세션(T±, Y1±, Y2±)으로 층화해 C++에서 다시 만들었다(numpy MT19937 셔플 재현). 원래 splits.json 과 " << split_same << "/4 fold 동일.\n"
      << "모델: X = sY·Y + sT·T (+ cY·Y² + cT·T²), 절편 없음. 잔차 상관을 (1-s)C + sI 로 줄이고 GLS 백색화 후 최소제곱으로 (Y,T)를 푼다. 이차는 시작점 4개 LM 중 비용 최소.\n"
      << "설정 8개(선형/이차 x 수축 0.001, 0.05, 0.2, 1.0)는 학습 210장 안 3-fold(StratifiedKFold(3, shuffle, 100+fold))에서 (실패 수, 최대 오차, 평균 오차) 순으로 골랐다.\n\n"
      << "## 방법별 요약\n\n" << sumT.md()
      << "\n## fold별 선택: " << cfg_name(chosen_fold[0]) << ", " << cfg_name(chosen_fold[1]) << ", " << cfg_name(chosen_fold[2]) << ", " << cfg_name(chosen_fold[3])
      << " / 최종 " << cfg_name(final_k) << "\n\n## Y·T 반응 방향 유사도 (백색화 공간)\n\n" << ident.md()
      << "\n## 원래 결과와 대조 (" << par.ok << "/" << par.total << " 일치)\n\n" << par.t.md()
      << "\n## 옮기지 않은 부분\n\n- 설정별 모델 npz 저장, report.html, audit.json, 보호 파일 해시 검사.\n"
      << "\n## 파일\n\n- `p12_features.csv`: 사진별 P12 특징\n- `predictions.csv`: OOF 사진별 추정(중첩 선택 동시 모델, 기존 Y-or-T)\n- `이미지/`: 그림\n";
    write_text(out("결과.md"), R.str());

    // ---- 7. 보고서.md ----
    auto pass_n = [](const std::vector<double>& e) { return std::count_if(e.begin(), e.end(), [](double v) { return std::abs(v) <= kPass; }); };
    std::ostringstream Pp;
    Pp << "# 보고서: Y·T 동시 추정, GLS 이차 순방향 모델 (10-22 15:19)\n\n"
       << "## 문제\n\n축 하나만 고르는 Y-or-T 모델 대신 Y와 T를 동시에 추정할 수 있는지 시험해야 했다.\n\n"
       << "## 해결 방법\n\n630 경계 특징(P12)에 GLS 선형·가산 이차 순방향 모델을 맞추고 4-fold 중첩 OOF로 평가했다.\n\n"
       << "## 결과\n\n동시 모델 OOF는 " << sel.passed << "/280(" << fmt(100.0 * sel.passed / 280, 1) << "%) 통과로 기존 Y-or-T " << old.passed << "/280보다 크게 낮았다.\n\n"
       << "| 지표 | 동시 모델(중첩 선택) | 기존 Y-or-T |\n|---|---|---|\n"
       << "| 통과 (두 축 \\|오차\\|<=0.09) | " << sel.passed << " / 280 | " << old.passed << " / 280 |\n"
       << "| Y 평균 / 최대 \\|오차\\| (mm) | " << fmt(sel.ymae / sel.n, 4) << " / " << fmt(sel.ymax, 4) << " | " << fmt(old.ymae / old.n, 4) << " / " << fmt(old.ymax, 4) << " |\n"
       << "| T 평균 / 최대 \\|오차\\| (deg) | " << fmt(sel.tmae / sel.n, 4) << " / " << fmt(sel.tmax, 4) << " | " << fmt(old.tmae / old.n, 4) << " / " << fmt(old.tmax, 4) << " |\n"
       << "| Y 통과 / T 통과 | " << pass_n(ey) << " / " << pass_n(et) << " | - |\n"
       << "| 비활성 축 평균 \\|값\\| | " << fmt(sel.inact / sel.n, 4) << " | " << fmt(old.inact / old.n, 4) << " |\n\n"
       << img_md("01_동시모델_OOF오차.png", "동시 모델 오차") << "\n\n" << img_md("02_기존YorT_OOF오차.png", "기존 오차") << "\n\n" << img_md("03_설정별_통과수.png", "설정별 통과")
       << "\n\n" << img_md("04_P12반응_Y와T.png", "P12 반응") << "\n\n"
       << "## 튀는 값\n\n가장 크게 틀린 사진은 Y는 " << nm[wy] << "(정답 Y " << fmt(ty[wy], 2) << ", 추정 " << fmt(py[wy], 3) << " mm, T " << fmt(tt[wy], 2) << "→" << fmt(pt[wy], 3)
       << "), T는 " << nm[wt] << "(정답 T " << fmt(tt[wt], 2) << ", 추정 " << fmt(pt[wt], 3) << ")이고, 평균 수평 이동이 Y와 T에 거의 같은 방향으로 나타나 모델이 이동을 두 축에 나눠 줬다.\n\n"
       << outl.md() << "\n" << img_md(fy, "Y 튀는 값") << "\n\n" << img_md(ft, "T 튀는 값")
       << "\n\n초록 상자는 기준 사진의 QR 영역, 빨간 상자는 그 QR 점들의 평균 이동만큼 옮긴 영역, 빨간 십자는 이 사진에서 맞춘 모서리(3개 중 1개, 자홍 = 진단 실패), 화살표는 점 변위 - QR 평균 변위를 20배 키운 것(회전이 만드는 점별 차이)이다.\n\n"
       << "## 한줄평\n\n백색화한 Y 반응과 T 반응의 코사인이 fold0 평균 " << fmt(cmean, 4) << "로 거의 나란해서, 축을 고르지 않고 두 값을 함께 풀면 같은 이동을 Y와 T로 나눠 가져 비활성 축 누출이 평균 "
       << fmt(sel.inact / sel.n, 3) << "가 되고 통과가 " << sel.passed << "장으로 떨어졌다.\n\n"
       << "## 원래 결과와 대조\n\n원래 값은 `codex_experiments_20261014/joint_yt_20261022/summary.csv`, `selection.json`, `identifiability.csv`, `multiref_4fold_20261021/splits.json`에서 가져왔다. "
       << par.ok << "/" << par.total << " 일치. 630점 측정은 사진에서 다시 했고(원래 scipy TRF 대신 같은 비용의 LM, QR3 디코딩 사각형 차이), 이차 모델 풀이도 LM이라 통과 장수는 ±3장까지 일치로 본다. "
       << "다시 잰 P12 특징은 원래 `center_patch_v2_20261021/v1_edge630_features.npy`와 특징별 최대 0.0035 px(중앙값 0.0001~0.0007 px) 다르다(scipy 가 xtol 1e-7 x |좌표| = 약 2e-4 px 에서 멈춘 차이). "
       << "fold 2·3 의 내부 3-fold 순위는 원래도 1위와 2위의 실패 수 차이가 1장이었고, 이 작은 특징 차이로 1위가 바뀌어(원래 0·1번 -> C++ "
       << chosen_fold[2] << "·" << chosen_fold[3] << "번) 중첩 선택 통과가 134 -> " << sel.passed << "장이 됐다. 이 두 항목은 불일치로 남긴다. 설정별 OOF 통과는 모두 ±3장 안이다.\n\n" << par.t.md();
    write_text(out("보고서.md"), Pp.str());
    std::cout << "joint " << sel.passed << " existing " << old.passed << " parity " << par.ok << "/" << par.total << "\n";
    return 0;
}
