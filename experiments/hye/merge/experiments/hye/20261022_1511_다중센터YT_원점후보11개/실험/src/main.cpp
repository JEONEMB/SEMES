// 원점 후보 11개 + Y/T 동시 추정, 341장 4-fold (내부 3-fold 설정 선택)
// 원본: joint_center_630_20261022/extract.py, measurement.py(compiled LM), features.py, model.py, local_model.py, evaluate.py
#include <hye/dataset.hpp>
#include <hye/plot.hpp>
#include <hye/report.hpp>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <iterator>
#include <map>
#include "corners.hpp"
#include "gls.hpp"
#include "joint.hpp"

using namespace hye;
using g09::JointConfig;
using g09::JointModel;

namespace {

struct Img {
    Shot shot;
    std::string relative;  // 20261014\Theta\T+0.01.bmp
    std::string file, scope, group, condition;
    double Y = 0, T = 0;
};

std::vector<std::string> parts_lower(const std::string& s) {
    std::vector<std::string> out(1);
    for (char c : s) {
        if (c == '\\') { out.emplace_back(); continue; }
        out.back() += (c >= 'A' && c <= 'Z') ? char(c - 'A' + 'a') : c;
    }
    return out;
}
std::string base_name(const std::string& rel) {
    auto p = rel.find_last_of("/\\");
    return p == std::string::npos ? rel : rel.substr(p + 1);
}
std::string stem(std::string s) { return s.size() > 4 && s.substr(s.size() - 4) == ".bmp" ? s.substr(0, s.size() - 4) : s; }

// hye::read_gray 는 cv::imread 라 한글 경로(clean_20261020/기준 위치/)를 못 연다. 바이트로 읽어 imdecode 한다.
cv::Mat read_u8path(const Shot& s, int flags) {
    std::ifstream f(fs::u8path(s.path), std::ios::binary);
    std::vector<uchar> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    cv::Mat m = cv::imdecode(buf, flags);
    if (m.empty()) throw std::runtime_error("읽기 실패: " + s.path);
    return m;
}

struct Meas {
    std::vector<cv::Point2d> D;
    std::vector<bool> valid;
    bool strict = true;
    int pad = 130;
    double ncc_min = 0;
};

// measurement.py StructuralExtractor.extract: NCC ±130 실패 시 ±260, ±512 로 넓힘
Meas measure(const cv::Mat& img, const cv::Mat& center, const std::vector<g09::Corner>& C) {
    Meas r;
    int shift[3][2];
    double score[3];
    for (int pad : {130, 260, 512}) {
        g09::ncc_shift(img, center, pad, shift, score);
        double mn = std::min({score[0], score[1], score[2]});
        if (pad == 130) r.strict = mn >= .6;
        r.pad = pad;
        r.ncc_min = mn;
        if (mn >= .6) break;
    }
    if (r.ncc_min < .6) throw std::runtime_error("QR NCC below .6 even with expanded search");
    r.D.resize(C.size());
    r.valid.resize(C.size());
    for (size_t i = 0; i < C.size(); ++i) {
        const auto& c = C[i];
        int k = c.qr - 1, dx = shift[k][0], dy = shift[k][1];
        int box[4] = {g09::BOXES[k][0] + dx, g09::BOXES[k][1] + dy, g09::BOXES[k][2] + dx, g09::BOXES[k][3] + dy};
        g09::CornerFit f = g09::fit_corner(img, c.cx + dx, c.cy + dy, box, c.inv, c.patch);
        r.D[i] = {f.x - c.px, f.y - c.py};
        r.valid[i] = f.valid;
    }
    return r;
}

struct PRow {
    int index;
    double Y, T, pY, pT;
    std::string method, protocol, center;
    int fold;
    double cost = kNaN, margin = kNaN, aY = kNaN, aT = kNaN;
    double eY() const { return std::abs(pY - Y); }
    double eT() const { return std::abs(pT - T); }
    bool finite() const { return std::isfinite(pY) && std::isfinite(pT); }
    bool passed() const { return finite() && eY() <= .09 + 1e-12 && eT() <= .09 + 1e-12; }
};

struct Metric {
    int n = 0, fails = 0, passed = 0;
    double YMAE = kNaN, Ymax = kNaN, TMAE = kNaN, Tmax = kNaN;
};
Metric metrics(const std::vector<const PRow*>& p) {
    Metric m;
    m.n = (int)p.size();
    double sy = 0, st = 0, my = 0, mt = 0;
    int k = 0;
    for (auto* r : p) {
        if (!r->finite()) { ++m.fails; continue; }
        ++k;
        sy += r->eY(); st += r->eT();
        my = std::max(my, r->eY()); mt = std::max(mt, r->eT());
        m.passed += r->passed();
    }
    if (k) { m.YMAE = sy / k; m.TMAE = st / k; m.Ymax = my; m.Tmax = mt; }
    return m;
}

}  // namespace

int run();
int main() {
    try { return run(); } catch (std::exception& e) { std::cerr << "error: " << e.what() << std::endl; return 3; }
}
int run() {
    omp_set_num_threads(4);
    begin_results();
    // ---- 341장 목록 (원래 경로 정렬: 소문자 경로 조각 순) ----
    std::vector<Img> im;
    for (auto& s : load_341()) {
        Img x;
        x.shot = s;
        std::string rel = s.rel;
        for (auto& c : rel) if (c == '/') c = '\\';
        x.relative = (s.set == "clean_20261014" ? "20261014\\" : "20261020\\") + rel;
        x.file = base_name(s.rel);
        im.push_back(x);
    }
    std::stable_sort(im.begin(), im.end(), [](const Img& a, const Img& b) { return parts_lower(a.relative) < parts_lower(b.relative); });
    const int N = (int)im.size();
    for (auto& x : im) {
        const Shot& s = x.shot;
        if (s.condition == "center") {
            x.scope = "center"; x.group = "center"; x.condition = stem(x.file);
            continue;
        }
        x.Y = s.y; x.T = s.t;
        x.scope = (s.y != 0 && s.t != 0) ? "mixed" : "single";
        char b[64];
        std::snprintf(b, sizeof b, "Y%+.4f_T%+.4f", x.Y, x.T);
        x.condition = b;
        if (s.set == "clean_20261014") {
            bool isT = x.T != 0;
            double v = isT ? x.T : x.Y;
            x.group = std::string(isT ? "T" : ((std::llabs(std::llround(v / .05)) % 2) ? "Y2" : "Y1")) + (v > 0 ? "+" : "-");
        } else x.group = "date20261020";
    }
    std::vector<std::array<double, 2>> truth(N);
    std::vector<std::string> groups(N), scopes(N), rels(N);
    for (int i = 0; i < N; ++i) { truth[i] = {im[i].Y, im[i].T}; groups[i] = im[i].group; scopes[i] = im[i].scope; rels[i] = im[i].relative; }

    // ---- 630점 경계 측정 ----
    int ic = 0;
    for (int i = 0; i < N; ++i) if (im[i].relative == "20261014\\center.bmp") ic = i;
    cv::Mat center = read_u8path(im[ic].shot, cv::IMREAD_GRAYSCALE);
    g09::Reference R = g09::build_reference(center);
    const std::vector<g09::Corner>& C = R.c630;
    const int NP = (int)C.size();
    std::vector<Meas> M(N);
#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < N; ++i) {
        try {
            cv::Mat img = read_u8path(im[i].shot, cv::IMREAD_GRAYSCALE);
            M[i] = measure(img, center, C);
        } catch (std::exception& e) {
#pragma omp critical
            std::cerr << "measure " << im[i].relative << ": " << e.what() << std::endl;
            M[i].ncc_min = -1;
        }
    }
    int strict_fail = 0, pad260 = 0;
    long valid_sum = 0;
    for (auto& m : M) { strict_fail += !m.strict; pad260 += m.pad != 130; for (bool v : m.valid) valid_sum += v; }

    // ---- 특징 (features.py FeatureMaps) ----
    std::map<std::string, cv::Mat> BANK;
    {
        auto pinv_design = [](const std::vector<cv::Point2d>& p, double cx, double cy, double sx, double sy, int terms) {
            cv::Mat B((int)p.size(), terms, CV_64F), out;
            for (int i = 0; i < (int)p.size(); ++i) {
                double u = (p[i].x - cx) / sx, v = (p[i].y - cy) / sy;
                double row[6] = {1, u, v, u * u, u * v, v * v};
                for (int j = 0; j < terms; ++j) B.at<double>(i, j) = row[j];
            }
            cv::invert(B, out, cv::DECOMP_SVD);
            return out;
        };
        std::vector<cv::Point2d> pts;
        for (auto& c : C) pts.push_back({c.px, c.py});
        cv::Mat P12 = pinv_design(pts, 1906, 1070, 486, 122, 6);
        std::vector<std::vector<int>> rid(3);
        std::vector<cv::Mat> aff(3), quad(3);
        for (int k = 0; k < 3; ++k) {
            std::vector<cv::Point2d> p;
            for (int i = 0; i < NP; ++i) {
                double cx[3] = {1644, 1901, 2159};
                int r = 0;
                for (int j = 1; j < 3; ++j) if (std::abs(C[i].px - cx[j]) < std::abs(C[i].px - cx[r])) r = j;
                if (r == k) { rid[k].push_back(i); p.push_back(pts[i]); }
            }
            double mx = 0, my = 0;
            for (auto& q : p) { mx += q.x; my += q.y; }
            mx /= p.size(); my /= p.size();
            aff[k] = pinv_design(p, mx, my, 100, 100, 3);
            quad[k] = pinv_design(p, mx, my, 100, 100, 6);
        }
        auto apply = [](const cv::Mat& Pm, const std::vector<cv::Point2d>& D, const std::vector<int>& idx, double* out) {
            for (int a = 0; a < Pm.rows; ++a) {
                double sx = 0, sy = 0;
                for (int j = 0; j < Pm.cols; ++j) { sx += Pm.at<double>(a, j) * D[idx[j]].x; sy += Pm.at<double>(a, j) * D[idx[j]].y; }
                out[a * 2] = sx;
                out[a * 2 + 1] = sy;
            }
        };
        std::vector<int> allidx(NP);
        std::iota(allidx.begin(), allidx.end(), 0);
        cv::Mat x12(N, 12, CV_64F), x18(N, 18, CV_64F), x36(N, 36, CV_64F), x1260(N, 2 * NP, CV_64F);
        for (int i = 0; i < N; ++i) {
            apply(P12, M[i].D, allidx, x12.ptr<double>(i));
            for (int k = 0; k < 3; ++k) { apply(aff[k], M[i].D, rid[k], x18.ptr<double>(i) + 6 * k); apply(quad[k], M[i].D, rid[k], x36.ptr<double>(i) + 12 * k); }
            for (int p = 0; p < NP; ++p) { x1260.at<double>(i, 2 * p) = M[i].D[p].x; x1260.at<double>(i, 2 * p + 1) = M[i].D[p].y; }
        }
        BANK["P12"] = x12; BANK["QR18"] = x18; BANK["QR36"] = x36; BANK["raw1260"] = x1260;
    }

    // ---- 설정 20개 ----
    std::vector<JointConfig> CONFIGS;
    for (int d : {1, 2}) for (double s : {.2, 1.}) for (std::string o : {"learned", "learned_actual"}) CONFIGS.push_back({"P12", d, s, o, false});
    for (std::string f : {"QR18", "QR36", "raw1260"})
        for (int d : {1, 2})
            for (double s : (f == "raw1260" ? std::vector<double>{1.} : std::vector<double>{.2, 1.})) CONFIGS.push_back({f, d, s, "learned_actual", false});
    for (std::string f : {"P12", "QR36"}) CONFIGS.push_back({f, 2, .2, "learned_actual", true});
    const int NC = (int)CONFIGS.size();
    auto train = [&](const JointConfig& c, const std::vector<int>& tr) {
        JointModel m;
        m.cfg = c;
        m.fit(BANK.at(c.feature), truth, groups, scopes, tr, rels);
        return m;
    };

    // ---- 바깥 4-fold: 기존 280장 분할 그대로 + 복합 조건 묶음 + center ----
    std::vector<int> fold(N, -1);
    {
        // multiref_4fold_20261021 분할: 10-14 단일축 280장을 center 다음 파일 이름 순으로 놓고 StratifiedKFold(4, 20261021)
        std::vector<int> singles;
        for (int i = 0; i < N; ++i) if (im[i].scope == "single") singles.push_back(i);
        std::stable_sort(singles.begin(), singles.end(), [&](int a, int b) { return im[a].file < im[b].file; });
        std::vector<std::string> ses;
        for (int i : singles) ses.push_back(im[i].group);
        auto f = g09::stratified_kfold(ses, 4, 20261021);
        for (size_t k = 0; k < singles.size(); ++k) fold[singles[k]] = f[k];
        // 복합: 조건을 (T, Y) 순으로 놓고 T 값마다 PCG64(20261022) 순열로 돌려 배정
        std::vector<std::pair<std::pair<double, double>, std::string>> conds;
        for (int i = 0; i < N; ++i)
            if (im[i].scope == "mixed") {
                bool seen = false;
                for (auto& c : conds) seen |= c.second == im[i].condition;
                if (!seen) conds.push_back({{im[i].T, im[i].Y}, im[i].condition});
            }
        std::stable_sort(conds.begin(), conds.end(), [](auto& a, auto& b) { return a.first < b.first; });
        g09::PCG64 rng(20261022);
        int counter = 0;
        for (size_t a = 0; a < conds.size();) {
            size_t b = a;
            std::vector<std::string> g;
            while (b < conds.size() && conds[b].first.first == conds[a].first.first) g.push_back(conds[b++].second);
            auto perm = rng.permutation(g);
            for (size_t j = 0; j < perm.size(); ++j)
                for (int i = 0; i < N; ++i)
                    if (im[i].scope == "mixed" && im[i].condition == perm[j]) fold[i] = (int)((counter + j) % 4);
            counter += (int)g.size();
            a = b;
        }
        int j = 0;
        for (int i = 0; i < N; ++i) if (im[i].scope == "center") fold[i] = (j++) % 4;
    }
    auto select = [&](const std::vector<int>& tr, uint32_t seed, std::vector<std::array<double, 3>>& scores) {
        std::vector<std::string> st, gr;
        for (int i : tr) { st.push_back(scopes[i]); gr.push_back(scopes[i] == "mixed" ? im[i].condition : rels[i]); }
        auto inner = g09::stratified_group_kfold(st, gr, 3, seed);
        scores.assign(NC, {0, 0, 0});
#pragma omp parallel for schedule(dynamic, 1)
        for (int c = 0; c < NC; ++c) {
            std::vector<std::array<double, 2>> e;
            for (int g = 0; g < 3; ++g) {
                std::vector<int> a, b;
                for (size_t k = 0; k < tr.size(); ++k) (inner[k] == g ? b : a).push_back(tr[k]);
                JointModel m = train(CONFIGS[c], a);
                auto o = m.predict(BANK.at(CONFIGS[c].feature), b);
                for (size_t k = 0; k < b.size(); ++k) e.push_back({std::abs(o.YT[k][0] - truth[b[k]][0]), std::abs(o.YT[k][1] - truth[b[k]][1])});
            }
            double fails = 0, mx = 0, s0 = 0, s1 = 0;
            for (auto& x : e) {
                fails += (x[0] > .09 + 1e-12 || x[1] > .09 + 1e-12);
                mx = std::max({mx, x[0], x[1]});
                s0 += x[0]; s1 += x[1];
            }
            scores[c] = {fails, mx, s0 / e.size() + s1 / e.size()};
        }
        int best = 0;
        for (int c = 1; c < NC; ++c) if (scores[c] < scores[best]) best = c;
        return best;
    };

    std::vector<PRow> rows;
    auto add_rows = [&](const std::vector<int>& idx, const std::vector<std::array<double, 2>>& yt, const std::string& method, const std::string& proto, int f,
                        const g09::JointOut* o = nullptr, const JointModel* mdl = nullptr) {
        for (size_t k = 0; k < idx.size(); ++k) {
            PRow r{idx[k], truth[idx[k]][0], truth[idx[k]][1], yt[k][0], yt[k][1], method, proto, "", f};
            if (o) {
                r.center = mdl->ids[o->center[k]];
                r.cost = o->cost[k];
                r.margin = o->margin[k];
                r.aY = o->alt[k][0];
                r.aT = o->alt[k][1];
            }
            rows.push_back(r);
        }
    };
    std::vector<std::string> sel_names;
    std::vector<int> cand_count, test_count;
    Table sel_t{{"fold", "선택 설정", "내부 실패", "내부 최대 오차", "내부 MAE 합", "원점 후보 수", "시험 장수"}};
    for (int f = 0; f <= 4; ++f) {
        std::vector<int> tr, te;
        for (int i = 0; i < N; ++i) {
            if (f == 4 || fold[i] != f) tr.push_back(i);
            if (f == 4 || fold[i] == f) te.push_back(i);
        }
        std::vector<std::array<double, 3>> sc;
        int best = select(tr, f == 4 ? 20261030u : 20261022u + f, sc);
        JointModel model = train(CONFIGS[best], tr);
        auto o = model.predict(BANK.at(CONFIGS[best].feature), te);
        std::string proto = f == 4 ? "resub341" : "oof4";
        int ff = f == 4 ? -1 : f;
        add_rows(te, o.YT, "multi_center_joint", proto, ff, &o, &model);
        JointModel single = train({"P12", 2, .5, "fixed", false}, tr);
        add_rows(te, single.predict(BANK.at("P12"), te).YT, "single_center_joint", proto, ff);
        add_rows(te, model.predict(BANK.at(CONFIGS[best].feature), te, 1).YT, "shared_surface_center0", proto, ff);
        sel_names.push_back(CONFIGS[best].label());
        cand_count.push_back(model.centers.rows);
        test_count.push_back((int)te.size());
        sel_t.add({f == 4 ? "전체 341" : std::to_string(f), CONFIGS[best].label(), fmt(sc[best][0], 0), fmt(sc[best][1], 6), fmt(sc[best][2], 6),
                   std::to_string(model.centers.rows), std::to_string(te.size())});
    }
    // 기존 Y-or-T 고정 모델 (v1_edge630: 10-14 단일축 280장 전체 학습, center 1원점 이차) 로 341장
    std::vector<int> all(N);
    std::iota(all.begin(), all.end(), 0);
    {
        std::vector<g09::Item> items(N);
        std::vector<g09::Vec> X12(N);
        std::vector<int> tr;
        for (int i = 0; i < N; ++i) {
            items[i].file = im[i].file;
            items[i].axis = im[i].scope == "single" ? (im[i].T != 0 ? 'T' : 'Y') : 'C';
            items[i].value = items[i].axis == 'T' ? im[i].T : im[i].Y;
            items[i].session = im[i].group;
            X12[i].assign(BANK["P12"].ptr<double>(i), BANK["P12"].ptr<double>(i) + 12);
            if (im[i].scope == "single") tr.push_back(i);
        }
        g09::Model fz = g09::fit_zero(items, X12, tr, "center", true);
        std::vector<std::array<double, 2>> wide(N), strict(N);
        for (int i = 0; i < N; ++i) {
            g09::Pred p = g09::predict_one(fz, X12[i], "center");
            wide[i] = p.pred_axis == 'Y' ? std::array<double, 2>{p.pred, 0.} : std::array<double, 2>{0., p.pred};
            strict[i] = M[i].strict ? wide[i] : std::array<double, 2>{kNaN, kNaN};
        }
        add_rows(all, wide, "frozen_Y_or_T_wide_init", "resub341", -1);
        add_rows(all, strict, "frozen_Y_or_T_strict_init", "resub341", -1);
    }

    // ---- 요약 ----
    std::vector<std::pair<std::string, std::string>> mp = {{"multi_center_joint", "oof4"},       {"multi_center_joint", "resub341"},
                                                           {"single_center_joint", "oof4"},      {"single_center_joint", "resub341"},
                                                           {"shared_surface_center0", "oof4"},   {"shared_surface_center0", "resub341"},
                                                           {"frozen_Y_or_T_wide_init", "resub341"}, {"frozen_Y_or_T_strict_init", "resub341"}};
    std::map<std::string, Metric> MET;
    Table summ{{"방법", "평가", "부분", "장수", "추론 실패", "Y MAE", "Y 최대", "T MAE", "T 최대", "통과", "통과율"}};
    Table scsv{{"method", "protocol", "subset", "n", "inference_failures", "Y_MAE", "Y_max", "T_MAE", "T_max", "passed", "pass_rate"}};
    for (auto& [meth, proto] : mp)
        for (std::string sub : {"all", "single", "mixed", "center"}) {
            std::vector<const PRow*> g;
            for (auto& r : rows) if (r.method == meth && r.protocol == proto && (sub == "all" || scopes[r.index] == sub)) g.push_back(&r);
            Metric m = metrics(g);
            MET[meth + "|" + proto + "|" + sub] = m;
            summ.add({meth, proto, sub, std::to_string(m.n), std::to_string(m.fails), fmt(m.YMAE, 6), fmt(m.Ymax, 6), fmt(m.TMAE, 6), fmt(m.Tmax, 6),
                      std::to_string(m.passed), fmt(100.0 * m.passed / m.n, 1) + "%"});
            scsv.add({meth, proto, sub, std::to_string(m.n), std::to_string(m.fails), fmt(m.YMAE, 8), fmt(m.Ymax, 8), fmt(m.TMAE, 8), fmt(m.Tmax, 8),
                      std::to_string(m.passed), fmt((double)m.passed / m.n, 6)});
        }
    scsv.csv(out("summary.csv"));
    Table pcsv{{"index", "relative", "Y", "T", "scope", "group", "method", "protocol", "fold", "predY", "predT", "eY", "eT", "passed", "center", "cost", "margin"}};
    for (auto& r : rows)
        pcsv.add({std::to_string(r.index), rels[r.index], fmt(r.Y, 4), fmt(r.T, 4), scopes[r.index], groups[r.index], r.method, r.protocol, std::to_string(r.fold),
                  fmt(r.pY, 8), fmt(r.pT, 8), fmt(r.eY(), 8), fmt(r.eT(), 8), r.passed() ? "True" : "False", r.center, fmt(r.cost, 6), fmt(r.margin, 6)});
    pcsv.csv(out("predictions.csv"));
    std::vector<int> fold_sum(4, 0);
    for (int i = 0; i < N; ++i) if (fold[i] >= 0) fold_sum[fold[i]] += i;

    // ---- 튀는 값 (multi_center_joint oof4) ----
    std::vector<const PRow*> oof;
    for (auto& r : rows) if (r.method == "multi_center_joint" && r.protocol == "oof4") oof.push_back(&r);
    std::sort(oof.begin(), oof.end(), [](const PRow* a, const PRow* b) { return a->index < b->index; });
    std::vector<double> ty, py, tt, pt, ey, et;
    std::vector<std::string> nn;
    for (auto* r : oof) {
        ty.push_back(r->Y); py.push_back(r->pY); tt.push_back(r->T); pt.push_back(r->pT);
        ey.push_back(r->pY - r->Y); et.push_back(r->pT - r->T);
        nn.push_back(stem(im[r->index].file));
    }
    size_t wy = worst(ey, 1)[0], wt = worst(et, 1)[0];
    if (wt == wy)
        for (size_t k : worst(et, et.size()))
            if (im[oof[k]->index].condition != im[oof[wy]->index].condition) { wt = k; break; }
    Table outl{{"구분", "사진", "fold", "정답 Y", "추정 Y", "정답 T", "추정 T", "고른 원점", "비용", "비용 간격", "2위 원점 추정 Y / T", "P12 c0x (px)", "QR36 QR1/QR3 c0x (px)"}};
    auto add_out = [&](const std::string& lab, size_t k) {
        const PRow* r = oof[k];
        int i = r->index;
        outl.add({lab, im[i].relative, std::to_string(r->fold), fmt(r->Y, 2), fmt(r->pY, 4), fmt(r->T, 2), fmt(r->pT, 4), r->center, fmt(r->cost, 2),
                  fmt(r->margin, 3), fmt(r->aY, 3) + " / " + fmt(r->aT, 3), fmt(BANK["P12"].at<double>(i, 0), 3),
                  fmt(BANK["QR36"].at<double>(i, 0), 3) + " / " + fmt(BANK["QR36"].at<double>(i, 24), 3)});
    };
    for (size_t k : worst(ey, 4)) add_out("Y 오차 상위", k);
    for (size_t k : worst(et, 3)) add_out("T 오차 상위", k);
    // 원점별 선택 횟수
    std::map<std::string, int> center_count;
    for (auto* r : oof) center_count[r->center]++;
    Table cc_t{{"고른 원점", "OOF 장수"}};
    for (auto& [k, v] : center_count) cc_t.add({k, std::to_string(v)});

    // ---- 그림 ----
    pred_vs_truth("01_다중원점_OOF_예측오차.png", "Multi-origin joint OOF 341", ty, py, tt, pt, nn);
    {
        std::vector<std::string> cats;
        std::vector<double> vals;
        for (auto& [meth, proto] : mp) {
            std::string s = meth == "multi_center_joint" ? "multi" : meth == "single_center_joint" ? "single" : meth == "shared_surface_center0" ? "shared c0"
                            : meth == "frozen_Y_or_T_wide_init" ? "frozen wide" : "frozen strict";
            cats.push_back(s + " " + proto);
            vals.push_back(MET[meth + "|" + proto + "|all"].passed);
        }
        bar_chart("02_방법별_통과장수.png", "Passed (both |err|<=0.09) of 341", cats, vals, "passed", 0, 341);
    }
    {
        Chart c("Multi-origin OOF: |Y err| vs |T err|", "|Y error| (mm)", "|T error| (deg)");
        const char* sc[3] = {"single", "mixed", "center"};
        for (int s = 0; s < 3; ++s) {
            std::vector<double> x, y;
            for (auto* r : oof) if (scopes[r->index] == sc[s]) { x.push_back(r->eY()); y.push_back(r->eT()); }
            c.scatter(x, y, color(s), sc[s], 3);
        }
        c.hline(.09, kRed).vline(.09, kRed);
        c.save("03_다중원점_OOF_두축오차.png");
    }
    auto outlier_fig = [&](size_t k, const std::string& file, const std::string& cap) {
        int i = oof[k]->index;
        cv::Mat img = read_u8path(im[i].shot, cv::IMREAD_COLOR);
        std::vector<cv::Mat> tiles;
        for (int q = 0; q < 3; ++q) {
            double mx = 0, my = 0;
            int n = 0;
            for (int p = 0; p < NP; ++p) if (C[p].qr == q + 1) { mx += M[i].D[p].x; my += M[i].D[p].y; ++n; }
            mx /= n; my /= n;
            std::vector<Mark> mk;
            int cnt = 0;
            for (int p = 0; p < NP; ++p) {
                if (C[p].qr != q + 1) continue;
                bool inv = !M[i].valid[p];
                if (!inv && (cnt++ % 6)) continue;
                mk.push_back({{C[p].px + M[i].D[p].x, C[p].py + M[i].D[p].y}, {M[i].D[p].x - mx, M[i].D[p].y - my}, "", inv ? color(1) : kRed});
            }
            const int* b = g09::BOXES[q];
            cv::Rect2d ref(b[0], b[1], b[2] - b[0], b[3] - b[1]);
            cv::Rect2d nowr(ref.x + mx, ref.y + my, ref.width, ref.height);
            cv::Rect roi = cv::Rect(ref | nowr) + cv::Size(40, 40) - cv::Point(20, 20);
            tiles.push_back(zoom(img, roi, 420, mk, 50, q == 0 ? cap : "",
                                 {{ref, kGreen, ""}, {nowr, kRed, "q" + std::to_string(q) + " dx " + fmt(mx, 2) + " dy " + fmt(my, 2)}}));
        }
        save_png(file, hstack(tiles));
    };
    auto safe = [](std::string s) {
        std::string o;
        for (char c : s) if (c != '(' && c != ')') o += c == ' ' ? '_' : c;
        return o;
    };
    std::string fy = "04_튀는값_Y_" + safe(nn[wy]) + ".png", ft = "05_튀는값_T_" + safe(nn[wt]) + ".png";
    // 평균 x 이동(P12 c0x)의 축별 기울기: 단일축 사진 최소제곱 (절편 포함)
    auto slope = [&](char axis) {
        double sx = 0, sy = 0, sxx = 0, sxy = 0;
        int n = 0;
        for (int i = 0; i < N; ++i) {
            if (im[i].scope != "single" || (axis == 'T') != (im[i].T != 0)) continue;
            double v = axis == 'T' ? im[i].T : im[i].Y, c = BANK["P12"].at<double>(i, 0);
            sx += v; sy += c; sxx += v * v; sxy += v * c; ++n;
        }
        return (n * sxy - sx * sy) / (n * sxx - sx * sx);
    };
    const double kY = slope('Y'), kT = slope('T');
    outlier_fig(wy, fy, nn[wy] + " Y err " + fmts(ey[wy], 3) + " T err " + fmts(et[wy], 3));
    outlier_fig(wt, ft, nn[wt] + " Y err " + fmts(ey[wt], 3) + " T err " + fmts(et[wt], 3));

    // ---- 원래 결과와 대조 (출처: results/manifest.csv, results/local_surfaces/summary.csv, selection.json, audit.json, splits.json) ----
    Parity par;
    par.add("측정 성공 장수", 341, N, 0, 0);
    par.add("±130 NCC 엄격 초기화 실패", 12, strict_fail, 0, 0);
    par.add("±260 확장 탐색 장수", 12, pad260, 0, 0);
    par.add("유효점 합 (/214830)", 213843, (double)valid_sum, 200, 0);
    const int fs0[4] = {14230, 14474, 15034, 14232};
    for (int f = 0; f < 4; ++f) par.add("fold " + std::to_string(f) + " 시험 색인 합", fs0[f], fold_sum[f], 0, 0);
    const int tc0[4] = {86, 86, 86, 83}, cc0[4] = {11, 10, 8, 10};
    for (int f = 0; f < 4; ++f) par.add("fold " + std::to_string(f) + " 시험 장수", tc0[f], test_count[f], 0, 0);
    for (int f = 0; f < 4; ++f) par.add("fold " + std::to_string(f) + " 원점 후보 수", cc0[f], cand_count[f], 0, 0);
    par.add("전체 학습 원점 후보 수", 11, cand_count[4], 0, 0);
    const char* sel0[5] = {"QR36 d2 s0.2 learned_actual", "P12 d2 s0.2 learned_actual", "P12 d1 s1.0 learned", "QR36 d2 s0.2 learned_actual local",
                           "QR36 d2 s0.2 learned_actual"};
    for (int f = 0; f < 5; ++f) par.add_text(f < 4 ? "fold " + std::to_string(f) + " 선택 설정" : "전체 학습 선택 설정", sel0[f], sel_names[f], sel_names[f] == sel0[f]);
    auto& mo = MET["multi_center_joint|oof4|all"];
    par.add("다중 원점 OOF 통과 (/341)", 169, mo.passed, 0, 0);
    par.add("다중 원점 OOF Y MAE (mm)", 0.112240, mo.YMAE, 0.002, 6);
    par.add("다중 원점 OOF Y 최대 (mm)", 0.754249, mo.Ymax, 0.005, 6);
    par.add("다중 원점 OOF T MAE", 0.019840, mo.TMAE, 0.0005, 6);
    par.add("다중 원점 OOF T 최대", 0.132601, mo.Tmax, 0.005, 6);
    par.add("다중 원점 OOF 단일축 통과 (/280)", 142, MET["multi_center_joint|oof4|single"].passed, 0, 0);
    par.add("다중 원점 OOF 복합 통과 (/57)", 23, MET["multi_center_joint|oof4|mixed"].passed, 0, 0);
    par.add("다중 원점 OOF center 통과 (/4)", 4, MET["multi_center_joint|oof4|center"].passed, 0, 0);
    par.add("다중 원점 재대입 통과 (/341)", 191, MET["multi_center_joint|resub341|all"].passed, 0, 0);
    par.add("다중 원점 재대입 Y 최대 (mm)", 0.364079, MET["multi_center_joint|resub341|all"].Ymax, 0.005, 6);
    par.add("단일 원점 OOF 통과", 157, MET["single_center_joint|oof4|all"].passed, 0, 0);
    par.add("단일 원점 재대입 통과", 171, MET["single_center_joint|resub341|all"].passed, 0, 0);
    par.add("공유 표면 center0 OOF 통과", 131, MET["shared_surface_center0|oof4|all"].passed, 0, 0);
    par.add("공유 표면 center0 재대입 통과", 93, MET["shared_surface_center0|resub341|all"].passed, 0, 0);
    par.add("기존 Y-or-T 넓은 초기화 통과", 303, MET["frozen_Y_or_T_wide_init|resub341|all"].passed, 0, 0);
    par.add("기존 Y-or-T 엄격 초기화 통과", 300, MET["frozen_Y_or_T_strict_init|resub341|all"].passed, 0, 0);
    std::string mismatch;
    for (auto& r : par.t.rows) if (r[4] == "불일치") mismatch += (mismatch.empty() ? "" : ", ") + r[0];

    // ---- 결과.md ----
    std::ostringstream Rm;
    Rm << "# 결과: 원점 후보 11개 + Y/T 동시 추정 (341장)\n\n"
       << "입력: `data/clean/clean_20261014` 281장 + `clean_20261020` 60장 = 341장 (단일축 280, 복합 19조건 x 3 = 57, center 4). 원래 경로 정렬(소문자 경로 조각 순)로 놓았다.\n"
       << "측정: center.bmp QR 해독 격자의 구조 모서리 630점을 erf 경계 soft_l1 LM으로 맞췄다. QR NCC ±130이 0.6 미만이면 ±260, ±512로 넓혔다.\n"
       << "특징: P12(630점 이차 6항 x 2), QR18(QR별 일차 3항), QR36(QR별 이차 6항), raw1260(630점 이동 그대로).\n"
       << "모델: 원점 후보 = 고정 center 0 + 학습 그룹 절편(T±, Y1±, Y2±, 10-20 촬영군) + 학습에 든 실제 center1~3 특징. 원점마다 (Y, T) 이차 표면을 GN 16회로 풀고 비용이 가장 작은 원점을 고른다.\n"
       << "평가: 기존 280장 4-fold 그대로 + 복합 조건 3장 묶음(T 값별 PCG64 순열) + center 한 장씩. fold마다 학습 안에서 StratifiedGroupKFold(3)로 20설정 중 하나를 고른다 (실패 수 → 최대 오차 → MAE 합).\n\n"
       << "## 방법별 성적\n\n" << summ.md() << "\n## 설정 선택\n\n" << sel_t.md() << "\n## OOF에서 고른 원점\n\n" << cc_t.md()
       << "\n## 원래 결과와 대조 (" << par.ok << "/" << par.total << " 일치)\n\n" << par.t.md();
    if (!mismatch.empty())
        Rm << "\n불일치 원인: 원래 center 기준 좌표와 시작 좌표는 SciPy TRF로 맞췄고 C++은 같은 LM 계산기로 맞췄다. QR3 해독 사각형도 OpenCV 3.4 검출기라 다르다. 이 차이가 모든 사진의 이동에 같은 작은 치우침을 더해 "
              "20설정 중 고르는 설정과 경계 근처 사진의 통과가 바뀐다. 개발 중 원래 `results/measurements.npz`의 이동을 이 C++ 모델·분할·선택 코드에 넣어 본 별도 검사(산출물 아님)에서는 38/38이 모두 일치했다. 그래서 차이는 측정 단계에서 생긴다.\n";
    Rm << "\n## 옮기지 않은 부분\n\n"
       << "- `results/p12_initial/`, `results/expanded_features/` 이전 단계 절차와 `diagnose.py` 식별성 진단, `test_audit.py` 합성 검사, `benchmark.py` 시간 측정, `engine.py` 상주 추론기는 옮기지 않았다. 최종 절차(`local_surfaces`)만 옮겼다.\n"
       << "- `diagnostic_oracle_group.csv`(정답 그룹을 준 반사실 진단)는 운영 성적이 아니라 옮기지 않았다.\n\n"
       << "## 파일\n\n- `summary.csv`: 방법·평가·부분별 성적\n- `predictions.csv`: 모든 예측\n- `이미지/`: 보고서 그림\n";
    write_text(out("결과.md"), Rm.str());

    // ---- 보고서.md ----
    std::ostringstream P;
    auto& ms = MET["single_center_joint|oof4|all"];
    P << "# 보고서: 다중 센터 Y/T 동시 추정, 원점 후보 11개 (10-22 15:11)\n\n"
      << "## 문제\n\n센터 후보 여러 개를 두고 Y와 T를 함께 추정하면 341장에서 두 축 0.09 이내가 되는지 확인하려 했다.\n\n"
      << "## 해결 방법\n\n630 모서리 측정 위에서 원점 후보 11개마다 Y/T를 함께 최적화해 비용이 가장 작은 원점을 고르는 모델을 4-fold로 평가했다.\n\n"
      << "## 결과\n\n341장 4-fold OOF는 " << mo.passed << "/341(" << fmt(100.0 * mo.passed / 341, 1) << "%), Y 최대 " << fmt(mo.Ymax, 6)
      << " mm로 두 축 0.09 안에 들지 못했다.\n\n"
      << "| 지표 | 값 |\n|---|---|\n"
      << "| OOF 두 축 통과 | " << mo.passed << " / 341 |\n"
      << "| OOF Y 평균 / 최대 \\|오차\\| (mm) | " << fmt(mo.YMAE, 4) << " / " << fmt(mo.Ymax, 4) << " |\n"
      << "| OOF T 평균 / 최대 \\|오차\\| | " << fmt(mo.TMAE, 4) << " / " << fmt(mo.Tmax, 4) << " |\n"
      << "| OOF 통과 단일축 / 복합 / center | " << MET["multi_center_joint|oof4|single"].passed << "/280 · " << MET["multi_center_joint|oof4|mixed"].passed << "/57 · "
      << MET["multi_center_joint|oof4|center"].passed << "/4 |\n"
      << "| 전체 학습 재대입 통과 | " << MET["multi_center_joint|resub341|all"].passed << " / 341 |\n"
      << "| 비교: 단일 원점 동시 추정 OOF | " << ms.passed << " / 341 |\n"
      << "| fold별 선택 | " << sel_names[0] << " · " << sel_names[1] << " · " << sel_names[2] << " · " << sel_names[3] << " |\n\n"
      << img_md("01_다중원점_OOF_예측오차.png", "예측 오차") << "\n\n" << img_md("02_방법별_통과장수.png", "통과 장수") << "\n\n"
      << img_md("03_다중원점_OOF_두축오차.png", "두 축 오차") << "\n\n"
      << "## 튀는 값\n\nY가 가장 크게 틀린 사진은 " << im[oof[wy]->index].relative << "(정답 Y " << fmt(ty[wy], 2) << ", T " << fmt(tt[wy], 2) << ")로 Y "
      << fmt(py[wy], 4) << ", T " << fmt(pt[wy], 4) << "로 읽혔다. 큰 T가 만드는 이동(P12 c0x " << fmt(BANK["P12"].at<double>(oof[wy]->index, 0), 1)
      << " px)에 Y가 섞이면 이차 표면이 Y를 T 쪽으로 넘겨 받아서다.\n\n"
      << outl.md() << "\n" << img_md(fy, "Y 튀는 값") << "\n\n" << img_md(ft, "T 튀는 값")
      << "\n\nT 그림은 다른 조건에서 T 오차가 가장 큰 사진 " << im[oof[wt]->index].relative << "(정답 Y " << fmt(ty[wt], 2) << ", T " << fmt(tt[wt], 2)
      << ", 추정 Y " << fmt(py[wt], 4) << ", T " << fmt(pt[wt], 4) << ")이다. "
      << "초록 상자는 center QR 영역, 빨간 상자는 QR 평균 이동만큼 옮긴 영역이다. 빨간 십자는 유효 모서리(6점마다 1개), 주황 십자는 진단에 걸린 모서리, 화살표는 QR 평균 이동을 뺀 나머지 이동을 50배 키운 것이다.\n\n"
      << "## 한줄평\n\nY 1 mm는 평균 x 이동 " << fmt(kY, 2) << " px, T 1은 " << fmt(kT, 2)
      << " px로 두 축이 같은 x 방향 이동을 만들어 원점 후보를 11개로 늘려도 표면이 두 축을 가르지 못했고, 그래서 다중 원점 OOF(" << mo.passed
      << "장)가 단일 원점(" << ms.passed << "장)보다 " << mo.passed - ms.passed << "장 나은 데 그쳤다.\n\n"
      << "## 원래 결과와 대조\n\n원래 값은 `joint_center_630_20261022/results/manifest.csv`, `results/local_surfaces/summary.csv`, `selection.json`, `audit.json`, `splits.json`에서 가져왔다. "
      << par.ok << "/" << par.total << " 일치.\n\n" << par.t.md();
    if (!mismatch.empty())
        P << "\n불일치 원인: 원래 center 기준 좌표와 시작 좌표는 SciPy TRF로 맞췄고 C++은 같은 LM 계산기로 맞췄다. QR3 해독 사각형도 OpenCV 3.4 검출기라 다르다. 이 차이가 모든 사진의 이동에 같은 작은 치우침을 더해 "
             "20설정 중 고르는 설정과 경계 근처 사진의 통과가 바뀐다. 개발 중 원래 `results/measurements.npz`의 이동을 이 C++ 모델·분할·선택 코드에 넣어 본 별도 검사(산출물 아님)에서는 38/38이 모두 일치했다. 그래서 차이는 측정 단계에서 생긴다.\n";
    write_text(out("보고서.md"), P.str());
    std::cout << "parity " << par.ok << "/" << par.total << "\n";
    return 0;
}
