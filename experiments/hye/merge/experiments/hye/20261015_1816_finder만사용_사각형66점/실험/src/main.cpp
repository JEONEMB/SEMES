// QR finder 사각형만 사용 (원본: codex_experiments_20261014/finder_only_20261015/run.py, verify.py, zero_experiment.py)
// center 에서 Otsu 이진화 + 윤곽선 계층으로 finder 사각형 9개를 찾고, 그 안쪽(3px 안으로)만 잘라 Shi-Tomasi 점을 고른다.
// 추적(NCC -> LK15 -> GN17)도 사각형 crop 안의 픽셀만 쓴다. 모델은 center_fixed 와 같은 절편 0 곡선(1차/2차).
// 비교용으로 같은 프로그램에서 QR 내부 205점(center_fixed 측정)의 center_quadratic 도 다시 돌린다.
#include <hye/dataset.hpp>
#include <hye/plot.hpp>
#include <hye/report.hpp>
#include <omp.h>
#include <iostream>
#include "spread.hpp"
#include "splits.hpp"

using namespace hye;

namespace {
struct Finder {
    int qr, x, y, w, h;  // crop 좌표, 안쪽 3px 들인 뒤
    cv::Mat tmpl, sm;
    std::vector<cv::Point2f> p;
    cx::Refiner model;
};
}  // namespace

static double median_of(std::vector<double> v) { return hye::percentile(std::move(v), 0.5); }
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
    struct Row { cx::Item it; const Shot* s; };
    std::vector<Row> rows;
    for (auto& s : shots) {
        std::string f = s.rel.substr(s.rel.find('/') == std::string::npos ? 0 : s.rel.find('/') + 1);
        char a = s.condition == "center" ? 'C' : s.condition == "T" ? 'T' : 'Y';
        double v = a == 'T' ? s.t : a == 'Y' ? s.y : 0;
        rows.push_back({{f, a, v, cx::session_of(a, v)}, &s});
    }
    std::sort(rows.begin(), rows.end(), [](const Row& p, const Row& q) {
        int ka = p.it.axis == 'C' ? 0 : p.it.axis == 'T' ? 1 : 2, kb = q.it.axis == 'C' ? 0 : q.it.axis == 'T' ? 1 : 2;
        return ka != kb ? ka < kb : p.it.file < q.it.file;
    });
    const int NI = (int)rows.size();
    std::vector<cx::Item> m;
    for (auto& r : rows) m.push_back(r.it);
    int ic = 0;
    for (int i = 0; i < NI; ++i) if (m[i].axis == 'C') ic = i;
    cv::Mat ref = cx::crop(read_gray(*rows[ic].s));

    // ---- finder 사각형 찾기 ----
    std::vector<std::array<int, 5>> boxes;
    {
        cv::Mat bw;
        cv::threshold(ref, bw, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
        std::vector<std::vector<cv::Point>> cont;
        std::vector<cv::Vec4i> h;
        cv::findContours(bw, cont, h, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
        for (int i = 0; i < (int)cont.size(); ++i) {
            int c = h[i][2];
            if (c < 0 || h[c][2] < 0) continue;
            double a = cv::contourArea(cont[i]);
            if (!(500 < a && a < 6000)) continue;
            cv::Rect r = cv::boundingRect(cont[i]);
            int gx = r.x + cx::CROP[0], gy = r.y + cx::CROP[1];
            for (int k = 0; k < 3; ++k)
                if (cx::BOXES[k][0] < gx && gx < cx::BOXES[k][2] && cx::BOXES[k][1] < gy && gy < cx::BOXES[k][3]) { boxes.push_back({k, r.x, r.y, r.width, r.height}); break; }
        }
        std::sort(boxes.begin(), boxes.end());
        if (boxes.size() != 9) throw std::runtime_error("finder count " + std::to_string(boxes.size()));
    }
    std::vector<Finder> fd;
    std::vector<cv::Point2d> pts;
    std::vector<int> pat, preg;
    for (int j = 0; j < 9; ++j) {
        auto b = boxes[j];
        Finder f;
        f.qr = b[0]; f.x = b[1] + 3; f.y = b[2] + 3; f.w = b[3] - 6; f.h = b[4] - 6;
        f.tmpl = ref(cv::Rect(f.x, f.y, f.w, f.h)).clone();
        cv::Mat sm;
        cv::GaussianBlur(f.tmpl, sm, cv::Size(0, 0), 0.65);
        cv::Mat mask = cv::Mat::zeros(f.tmpl.size(), CV_8U);
        mask(cv::Rect(10, 10, f.w - 20, f.h - 20)) = 255;
        std::vector<cv::Point2f> p;
        cv::goodFeaturesToTrack(sm, p, 100, 0.02, 4, mask, 5);
        cv::cornerSubPix(sm, p, cv::Size(2, 2), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 0.001));
        for (auto& q : p)
            if (q.x >= 10 && q.x < f.w - 10 && q.y >= 10 && q.y < f.h - 10) f.p.push_back(q);
        cv::GaussianBlur(f.tmpl, f.sm, cv::Size(0, 0), 0.8);
        f.model = cx::Refiner(f.tmpl, f.p, 17, 0.8);
        for (auto& q : f.p) {
            pts.push_back({(double)q.x + f.x + cx::CROP[0], (double)q.y + f.y + cx::CROP[1]});
            pat.push_back(j);
            preg.push_back(f.qr);
        }
        fd.push_back(f);
    }
    const int NP = (int)pts.size();
    std::cout << "finder points " << NP << "\n";

    // ---- 추적 (finder) + 비교용 205점 ----
    cx::Extractor ex(ref, 14);
    const int NQ = (int)ex.P.p.size();
    std::vector<cv::Point2d> qpts;
    for (int i = 0; i < NQ; ++i) qpts.push_back(ex.global(i));
    std::vector<std::vector<cv::Point2d>> D(NI, std::vector<cv::Point2d>(NP, {0, 0})), DQ(NI, std::vector<cv::Point2d>(NQ, {0, 0}));
    std::vector<std::vector<bool>> V(NI, std::vector<bool>(NP, true));
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < NI; ++i) {
        if (i == ic) continue;
        cv::Mat cur = cx::crop(read_gray(*rows[i].s));
        int o = 0;
        for (auto& f : fd) {
            cv::Mat s = cur(cv::Rect(f.x - 110, f.y - 22, f.w + 220, f.h + 44)), r;
            cv::matchTemplate(s, f.tmpl, r, cv::TM_CCOEFF_NORMED);
            cv::Point pos;
            cv::minMaxLoc(r, nullptr, nullptr, nullptr, &pos);
            int dx = pos.x - 110, dy = pos.y - 22;
            cv::Mat target = cur(cv::Rect(f.x + dx, f.y + dy, f.w, f.h)).clone(), tb;
            cv::GaussianBlur(target, tb, cv::Size(0, 0), 0.8);
            std::vector<cv::Point2f> d, gd;
            std::vector<bool> lv, gv;
            std::vector<float> fb;
            cx::lk_track(f.sm, tb, f.p, 15, d, lv, fb);
            std::vector<std::array<float, 4>> gq;
            f.model.run(target, d, gd, gq, gv);
            for (size_t k = 0; k < f.p.size(); ++k, ++o) {
                D[i][o] = {(double)gd[k].x + dx, (double)gd[k].y + dy};
                V[i][o] = lv[k] && gv[k];
            }
        }
        auto r = ex.extract(cur);
        for (int k = 0; k < NQ; ++k) DQ[i][k] = {r.D[k].x, r.D[k].y};
    }
    Table tf{{"file", "point_id", "pattern"}};
    int invalid = 0;
    for (int i = 0; i < NI; ++i)
        for (int k = 0; k < NP; ++k)
            if (!V[i][k]) { ++invalid; tf.add({m[i].file, std::to_string(k), std::to_string(pat[k] + 1)}); }
    tf.csv(out("tracking_failures.csv"));
    std::vector<std::vector<double>> X(NI), XQ(NI);
    for (int i = 0; i < NI; ++i) {
        std::vector<bool> v = V[i];
        X[i] = cx::p12(pts, D[i], &v);
        XQ[i] = cx::p12(qpts, DQ[i]);
    }
    {
        Table t{{"point_id", "x", "y", "qr", "pattern"}};
        for (int k = 0; k < NP; ++k) t.add({std::to_string(k), fmt(pts[k].x, 4), fmt(pts[k].y, 4), std::to_string(preg[k] + 1), std::to_string(pat[k] + 1)});
        t.csv(out("features.csv"));
    }

    // ---- 분할과 예측 ----
    auto folds = sp::folds(m);
    std::vector<int> ids;
    for (int i = 0; i < NI; ++i) if (i != ic) ids.push_back(i);
    std::vector<sp::Fold> all{{"full280_resubstitution", "all", ids, ids}};
    all.insert(all.end(), folds.begin(), folds.end());
    struct Cfg { std::string method; bool quad; bool finder; };
    const std::vector<Cfg> cfgs = {{"finder_quadratic", true, true}, {"finder_linear", false, true}, {"center_quadratic", true, false}};
    std::vector<sp::PredRow> P;
    std::map<std::string, cx::Model> full_model;
    for (auto& c : cfgs) {
        auto& XX = c.finder ? X : XQ;
        std::vector<std::vector<sp::PredRow>> per(all.size());
#pragma omp parallel for schedule(dynamic)
        for (int f = 0; f < (int)all.size(); ++f) {
            auto md = cx::fit(m, XX, all[f].train, "center", c.quad);
            for (int i : all[f].test) {
                auto p = cx::predict_one(XX[i], md, false);
                per[f].push_back({i, m[i].axis, p.axis, m[i].value, p.value, p.cost, m[i].session, p.branch, c.method, all[f].protocol, all[f].fold});
            }
        }
        for (auto& v : per) P.insert(P.end(), v.begin(), v.end());
        full_model[c.method] = cx::fit(m, XX, ids, "center", c.quad);
    }
    std::vector<std::string> protocols;
    for (auto& f : all) if (std::find(protocols.begin(), protocols.end(), f.protocol) == protocols.end()) protocols.push_back(f.protocol);
    std::map<std::pair<std::string, std::string>, sp::Score> S;
    Table met{{"method", "protocol", "n", "axis_sign_accuracy", "strict_pass09", "Y_axis_sign_accuracy", "T_axis_sign_accuracy", "vector_Y_max", "vector_T_max",
               "Y_matched_MAE", "T_matched_MAE"}};
    for (auto& c : cfgs)
        for (auto& pr : protocols) {
            std::vector<sp::PredRow> d;
            for (auto& r : P) if (r.method == c.method && r.protocol == pr) d.push_back(r);
            auto s = sp::score(d);
            S[{c.method, pr}] = s;
            met.add({c.method, pr, std::to_string(s.n), fmt(s.axis_sign_accuracy, 6), fmt(s.strict_pass09, 6), fmt(s.Y_axis_sign_accuracy, 6), fmt(s.T_axis_sign_accuracy, 6),
                     fmt(s.vector_Y_max, 6), fmt(s.vector_T_max, 6), fmt(s.Y_matched_MAE, 6), fmt(s.T_matched_MAE, 6)});
        }
    met.csv(out("metrics.csv"));
    std::map<std::string, sp::Score> rep;
    for (auto& c : cfgs) {
        std::vector<sp::PredRow> d;
        for (auto& r : P) if (r.method == c.method && r.protocol.rfind("repeated5_", 0) == 0) d.push_back(r);
        rep[c.method] = sp::score(d);
    }
    {
        Table pt{{"file", "axis", "value", "session", "pred_axis", "pred", "branch", "cost", "method", "protocol", "fold"}};
        for (auto& r : P)
            if (r.method != "center_quadratic")
                pt.add({m[r.i].file, std::string(1, r.axis), fmt(r.value, 2), r.session, std::string(1, r.pred_axis), fmt(r.pred, 10), r.branch, fmt(r.cost, 6), r.method, r.protocol, r.fold});
        pt.csv(out("predictions.csv"));
    }

    // 주 결과: finder_quadratic 재대입
    auto full_of = [&](const std::string& meth) {
        std::vector<sp::PredRow> o;
        for (auto& r : P) if (r.method == meth && r.protocol == "full280_resubstitution") o.push_back(r);
        return o;
    };
    auto full = full_of("finder_quadratic");
    std::vector<double> ty, py, tt, pt, eY, eT;
    std::vector<std::string> names;
    int npass = 0, okY = 0, okT = 0;
    std::vector<int> fails;
    for (size_t k = 0; k < full.size(); ++k) {
        auto& r = full[k];
        double a = r.axis == 'Y' ? r.value : 0, b = r.axis == 'T' ? r.value : 0, c = r.pred_axis == 'Y' ? r.pred : 0, d = r.pred_axis == 'T' ? r.pred : 0;
        ty.push_back(a); tt.push_back(b); py.push_back(c); pt.push_back(d);
        eY.push_back(c - a); eT.push_back(d - b);
        names.push_back(m[r.i].file.substr(0, m[r.i].file.size() - 4));
        bool ok = r.axis == r.pred_axis && sp::sgn(r.value) == sp::sgn(r.pred), pass = ok && std::abs(c - a) <= kPass && std::abs(d - b) <= kPass;
        npass += pass;
        (r.axis == 'Y' ? okY : okT) += ok;
        if (!pass) fails.push_back((int)k);
    }
    int npassQ = 0;
    for (auto& r : full_of("center_quadratic")) {
        double a = r.axis == 'Y' ? r.value : 0, b = r.axis == 'T' ? r.value : 0, c = r.pred_axis == 'Y' ? r.pred : 0, d = r.pred_axis == 'T' ? r.pred : 0;
        npassQ += r.axis == r.pred_axis && sp::sgn(r.value) == sp::sgn(r.pred) && std::abs(c - a) <= kPass && std::abs(d - b) <= kPass;
    }
    // 12값 잡음: 절편 0 2차 곡선을 280장에 맞춘 잔차 표준편차 (축마다 따로 맞추고 합침)
    auto resid_sd = [&](const std::vector<std::vector<double>>& XX) {
        std::vector<std::vector<double>> ss(12);
        int n = 0;
        for (char a : {'Y', 'T'}) {
            std::vector<int> ii;
            for (int i : ids) if (m[i].axis == a) ii.push_back(i);
            cv::Mat A((int)ii.size(), 2, CV_64F), Z((int)ii.size(), 12, CV_64F), co;
            for (int r = 0; r < (int)ii.size(); ++r) {
                A.at<double>(r, 0) = m[ii[r]].value; A.at<double>(r, 1) = m[ii[r]].value * m[ii[r]].value;
                for (int c = 0; c < 12; ++c) Z.at<double>(r, c) = XX[ii[r]][c];
            }
            cv::solve(A, Z, co, cv::DECOMP_SVD);
            cv::Mat R = Z - A * co;
            for (int c = 0; c < 12; ++c) for (int r = 0; r < R.rows; ++r) ss[c].push_back(std::abs(R.at<double>(r, c)));
            n += (int)ii.size();
        }
        std::vector<double> o(12);
        for (int c = 0; c < 12; ++c) o[c] = 1.4826 * median_of(ss[c]);
        return o;
    };
    auto sdF = resid_sd(X), sdQ = resid_sd(XQ);
    double rdx_min = 1e9, rdx_max = 0, rdy_min = 1e9, rdy_max = 0;
    for (int c = 2; c < 12; ++c) {
        double r = sdF[c] / sdQ[c];
        if (c % 2 == 0) { rdx_min = std::min(rdx_min, r); rdx_max = std::max(rdx_max, r); }
        else { rdy_min = std::min(rdy_min, r); rdy_max = std::max(rdy_max, r); }
    }
    const char* cn[12] = {"c0x", "c0y", "x·dx", "x·dy", "y·dx", "y·dy", "x²·dx", "x²·dy", "xy·dx", "xy·dy", "y²·dx", "y²·dy"};
    Table tsd{{"12값", "finder 66점 잔차 강건 sd (px)", "QR 내부 205점 잔차 강건 sd (px)", "비 (finder / 205점)"}};
    for (int c = 0; c < 12; ++c) tsd.add({cn[c], fmt(sdF[c], 4), fmt(sdQ[c], 4), fmt(sdF[c] / sdQ[c], 2)});
    // 점 분포: x 퍼짐
    auto spread = [](const std::vector<cv::Point2d>& p) {
        double mx = 0, my = 0;
        for (auto& q : p) { mx += q.x; my += q.y; }
        mx /= p.size(); my /= p.size();
        double sx = 0, sy = 0;
        for (auto& q : p) { sx += (q.x - mx) * (q.x - mx); sy += (q.y - my) * (q.y - my); }
        return cv::Point2d(std::sqrt(sx / p.size()), std::sqrt(sy / p.size()));
    };
    auto spF = spread(pts), spQ = spread(qpts);

    // ---- 그림 ----
    pred_vs_truth("01_재대입_예측오차.png", "finder_quadratic full 280", ty, py, tt, pt, names);
    {
        cv::Mat c;
        cv::cvtColor(ref(cv::Range(150, 390), cv::Range(190, 940)), c, cv::COLOR_GRAY2BGR);
        const double s = 1.6;
        cv::resize(c, c, cv::Size(), s, s, cv::INTER_NEAREST);
        for (auto& f : fd) cv::rectangle(c, cv::Rect((int)std::lround((f.x - 190) * s), (int)std::lround((f.y - 150) * s), (int)std::lround(f.w * s), (int)std::lround(f.h * s)), kGreen, 2);
        for (int k = 0; k < NQ; ++k)
            cv::circle(c, cv::Point((int)std::lround((ex.P.p[k].x - 190) * s), (int)std::lround((ex.P.p[k].y - 150) * s)), 2, kGray, -1, cv::LINE_AA);
        for (int k = 0; k < NP; ++k)
            cv::circle(c, cv::Point((int)std::lround((pts[k].x - cx::CROP[0] - 190) * s), (int)std::lround((pts[k].y - cx::CROP[1] - 150) * s)), 4, kRed, -1, cv::LINE_AA);
        text(c, "red = " + std::to_string(NP) + " finder points, gray = " + std::to_string(NQ) + " QR points, green = finder crops", {8, 24}, 0.6, kBlack, 2);
        save_png("02_finder점66.png", c);
    }
    {
        std::vector<std::string> cats;
        std::vector<double> v;
        for (auto& c : cfgs) {
            std::string sh = c.method == "finder_quadratic" ? "finder quad" : c.method == "finder_linear" ? "finder lin" : "QR205 quad";
            cats.push_back(sh + " Y"); v.push_back(rep[c.method].Y_axis_sign_accuracy * 100);
            cats.push_back(sh + " T"); v.push_back(rep[c.method].T_axis_sign_accuracy * 100);
        }
        bar_chart("03_반복5분할_축부호정확도.png", "Repeated 5-fold x20: axis+sign accuracy (%)", cats, v, "%", 0, 100);
    }
    {
        std::vector<std::string> cats;
        std::vector<double> v;
        for (int c = 0; c < 12; ++c) { cats.push_back(std::string(c < 2 ? "" : "") + "c" + std::to_string(c / 2) + (c % 2 ? "y" : "x")); v.push_back(sdF[c] / sdQ[c]); }
        bar_chart("04_12값_잡음비.png", "Residual sd ratio of 12 values: finder 66 / QR 205", cats, v, "ratio", -1, 1.0);
    }
    auto outlier_fig = [&](int k, const std::string& file, const std::string& cap) {
        auto& r = full[k];
        cv::Mat im = read_color(*rows[r.i].s);
        std::vector<cv::Mat> tiles;
        for (int q = 0; q < 3; ++q) {
            cv::Rect2d qb(cx::BOXES[q][0], cx::BOXES[q][1], cx::BOXES[q][2] - cx::BOXES[q][0], cx::BOXES[q][3] - cx::BOXES[q][1]);
            std::vector<Box> bx;
            std::vector<Mark> mk;
            cv::Point2d qm(0, 0);
            int nq = 0;
            for (int j = 0; j < 9; ++j) {
                if (fd[j].qr != q) continue;
                cv::Point2d md(0, 0);
                int n = 0;
                for (int k2 = 0; k2 < NP; ++k2) if (pat[k2] == j) { md += D[r.i][k2]; ++n; }
                md /= (double)n;
                qm += md; ++nq;
                cv::Rect2d rb(fd[j].x + cx::CROP[0], fd[j].y + cx::CROP[1], fd[j].w, fd[j].h);
                bx.push_back({rb, kGreen, ""});
                bx.push_back({cv::Rect2d(rb.x + md.x, rb.y + md.y, rb.width, rb.height), kRed, ""});
                for (int k2 = 0; k2 < NP; ++k2) if (pat[k2] == j) mk.push_back({pts[k2] + D[r.i][k2], D[r.i][k2] - md, "", kRed});
            }
            qm /= (double)nq;
            cv::Rect roi = cv::Rect(qb) + cv::Size(120, 40) - cv::Point(60, 20);
            roi.x += (int)std::lround(std::min(qm.x, 0.0));
            tiles.push_back(zoom(im, roi, 460, mk, 200, q == 0 ? cap : "", bx));
        }
        save_png(file, hstack(tiles));
    };
    // Y 튀는 값은 정답 Y 사진 중, T 튀는 값은 정답 T 사진 중에서 고른다 (Y 를 T 로 읽은 사진이 두 축 모두 최대가 되지 않게)
    auto worst_axis = [&](char ax) {
        size_t best = 0;
        double bv = -1;
        for (size_t k = 0; k < full.size(); ++k) {
            if (full[k].axis != ax) continue;
            double e = std::max(std::abs(eY[k]), std::abs(eT[k]));
            if (e > bv) { bv = e; best = k; }
        }
        return best;
    };
    size_t wy = worst_axis('Y'), wt = worst_axis('T');
    std::string fy = "05_튀는값_Y_" + names[wy] + ".png", ft = "06_튀는값_T_" + names[wt] + ".png";
    outlier_fig((int)wy, fy, names[wy] + " (" + std::string(1, full[wy].axis) + " " + fmts(full[wy].value, 2) + ") -> " + std::string(1, full[wy].pred_axis) + " " + fmts(full[wy].pred, 3));
    outlier_fig((int)wt, ft, names[wt] + " (" + std::string(1, full[wt].axis) + " " + fmts(full[wt].value, 2) + ") -> " + std::string(1, full[wt].pred_axis) + " " + fmts(full[wt].pred, 3));
    auto& fq = full_model["finder_quadratic"];
    Table ftab{{"사진", "정답", "고른 축", "추정", "고른 축 비용", "맞는 축 비용", "finder c0x (px)", "205점 c0x (px)", "finder x·dx (px)", "205점 x·dx (px)"}};
    for (int k : fails) {
        auto& r = full[k];
        auto p = cx::predict_one(X[r.i], fq, false);
        double right = kNaN;
        for (auto& a : p.all) if (a.first[0] == r.axis) right = a.second.second;
        ftab.add({m[r.i].file, std::string(1, r.axis) + " " + fmts(r.value, 2), std::string(1, r.pred_axis), fmts(r.pred, 4), fmt(p.cost, 2), fmt(right, 2),
                  fmt(X[r.i][0], 3), fmt(XQ[r.i][0], 3), fmts(X[r.i][2], 4), fmts(XQ[r.i][2], 4)});
    }

    // ---- 원래 결과와 대조 (finder_only_20261015/checks.json, metrics.csv, predictions.csv, repeated_cv_summary.csv, tracking_failures.csv; 서명은 gls_comparison_20261014/splits.json) ----
    Parity par;
    par.add("finder 사각형 수", 9, (double)fd.size(), 0, 0);
    par.add("finder 점 수", 66, NP, 0, 0);
    const int per_o[9] = {12, 6, 7, 11, 7, 6, 6, 5, 6};
    const int crop_o[9][4] = {{225, 299, 49, 53}, {226, 183, 48, 53}, {334, 184, 49, 52}, {483, 185, 49, 52}, {483, 301, 48, 52},
                              {592, 186, 48, 51}, {741, 302, 49, 52}, {742, 186, 48, 52}, {851, 187, 48, 52}};
    int per_ok = 0, crop_ok = 0;
    for (int j = 0; j < 9; ++j) {
        per_ok += (int)fd[j].p.size() == per_o[j];
        crop_ok += fd[j].x == crop_o[j][0] && fd[j].y == crop_o[j][1] && fd[j].w == crop_o[j][2] && fd[j].h == crop_o[j][3];
    }
    par.add("사각형별 점 수 일치 (/9)", 9, per_ok, 0, 0);
    par.add("사각형 crop 일치 (/9)", 9, crop_ok, 0, 0);
    par.add("추적 실패 (점 x 사진)", 2, invalid, 0, 0);
    {
        std::string fl;
        for (auto& r : tf.rows) fl += (fl.empty() ? "" : ",") + r[0] + "#" + r[1];
        const std::string o = "T+0.69.bmp#24,T+0.90.bmp#24";
        par.add_text("추적 실패 목록", o, fl, fl == o);
    }
    const std::vector<std::pair<std::string, long long>> sig = {
        {"interleaved", 59500}, {"stratified5", 118343}, {"blocked_within_session5", 118860}, {"repeated5_seed0", 117733}, {"repeated5_seed19", 118285},
        {"session_out_missing_origin", 123150}};
    int sig_ok = 0;
    for (auto& [p, v] : sig) sig_ok += sp::signature(folds, p) == v;
    par.add("분할 서명 일치 (6 protocol 표본)", 6, sig_ok, 0, 0);
    par.add("finder_quadratic 재대입 통과 (/280)", 262, npass, 0, 0);
    par.add("finder_quadratic 재대입 Y 축·부호 (/80)", 65, okY, 0, 0);
    par.add("finder_quadratic 재대입 T 축·부호 (/200)", 197, okT, 0, 0);
    par.add("finder_quadratic 반복 5분할 Y 축·부호 (%)", 81.4375, rep["finder_quadratic"].Y_axis_sign_accuracy * 100, 0.05, 4);
    par.add("finder_quadratic 반복 5분할 T 축·부호 (%)", 98.275, rep["finder_quadratic"].T_axis_sign_accuracy * 100, 0.05, 4);
    par.add("finder_linear 반복 5분할 Y 축·부호 (%)", 80.4375, rep["finder_linear"].Y_axis_sign_accuracy * 100, 0.05, 4);
    par.add("finder_linear 반복 5분할 T 축·부호 (%)", 98.325, rep["finder_linear"].T_axis_sign_accuracy * 100, 0.05, 4);
    par.add("center_quadratic(205점) 반복 5분할 Y 축·부호 (%)", 93.4375, rep["center_quadratic"].Y_axis_sign_accuracy * 100, 0.05, 4);
    par.add("center_quadratic(205점) 재대입 통과 (/280)", 274, npassQ, 0, 0);
    auto& fs = S[{"finder_quadratic", "full280_resubstitution"}];
    par.add("finder_quadratic 재대입 Y 맞은 축 MAE", 0.013078, fs.Y_matched_MAE, 0.0002, 6);
    par.add("finder_quadratic 재대입 T 맞은 축 MAE", 0.002315, fs.T_matched_MAE, 0.00005, 6);
    par.add("finder_quadratic 재대입 T 최대 벡터 오차", 0.190598, fs.vector_T_max, 0.001, 6);
    par.add("finder_quadratic session_out 축·부호", 0.821429, S[{"finder_quadratic", "session_out_missing_origin"}].axis_sign_accuracy, 0.0005, 6);
    par.add("finder_quadratic blocked 축·부호", 0.939286, S[{"finder_quadratic", "blocked_within_session5"}].axis_sign_accuracy, 0.0005, 6);
    par.add("finder_linear 재대입 축·부호", 0.932143, S[{"finder_linear", "full280_resubstitution"}].axis_sign_accuracy, 0.0005, 6);
    {
        std::string fl;
        for (int k : fails) fl += (fl.empty() ? "" : ",") + names[k];
        const std::string o = "T+0.03,T-0.01,T-0.04,Y+0.05,Y+0.10,Y+0.20,Y+0.25,Y+0.30,Y+0.45,Y+0.60,Y+0.70,Y+0.80,Y+1.00,Y-0.15,Y-0.20,Y-0.30,Y-0.70,Y-1.10";
        par.add_text("재대입 실패 사진 (18장)", o, fl, fl == o);
    }
    auto pv = [&](const std::string& f) { for (auto& r : full) if (m[r.i].file == f) return r.pred; return kNaN; };
    const std::vector<std::pair<std::string, double>> sample = {{"T+0.01.bmp", 0.0058223305356267}, {"Y+1.00.bmp", 0.1741195431241043},
                                                                {"T-0.50.bmp", -0.5002592970957954}, {"Y-2.00.bmp", -1.994396260267388},
                                                                {"T+1.00.bmp", 1.0012883590609458}, {"Y-1.10.bmp", -0.190598}};
    for (auto& [f, v] : sample) par.add(f + " 재대입 추정", v, pv(f), 0.0005, 6);

    // ---- 결과.md ----
    Table key{{"method", "protocol", "축·부호 정확도", "엄격 통과", "Y 축·부호", "T 축·부호", "Y 최대 벡터 오차", "T 최대 벡터 오차", "Y 맞은 축 MAE", "T 맞은 축 MAE"}};
    for (auto& c : cfgs)
        for (auto& pr : protocols) {
            if (pr.rfind("repeated5_seed", 0) == 0) continue;
            auto& s = S[{c.method, pr}];
            key.add({c.method, pr, fmt(s.axis_sign_accuracy * 100, 2) + "%", fmt(s.strict_pass09 * 100, 2) + "%", fmt(s.Y_axis_sign_accuracy * 100, 2) + "%",
                     fmt(s.T_axis_sign_accuracy * 100, 2) + "%", fmt(s.vector_Y_max, 4), fmt(s.vector_T_max, 4), fmt(s.Y_matched_MAE, 5), fmt(s.T_matched_MAE, 5)});
        }
    Table rk{{"method", "반복 5분할 x20 Y 축·부호", "T 축·부호", "엄격 통과"}};
    for (auto& c : cfgs)
        rk.add({c.method, fmt(rep[c.method].Y_axis_sign_accuracy * 100, 4) + "%", fmt(rep[c.method].T_axis_sign_accuracy * 100, 4) + "%", fmt(rep[c.method].strict_pass09 * 100, 4) + "%"});
    Table tpat{{"사각형", "QR", "crop (x, y, w, h)", "점 수"}};
    for (int j = 0; j < 9; ++j)
        tpat.add({std::to_string(j + 1), std::to_string(fd[j].qr + 1), "(" + std::to_string(fd[j].x) + ", " + std::to_string(fd[j].y) + ", " + std::to_string(fd[j].w) + ", " + std::to_string(fd[j].h) + ")",
                  std::to_string(fd[j].p.size())});
    std::ostringstream R;
    R << "# 결과: finder 사각형만 사용\n\n"
      << "입력: `data/clean/clean_20261014` 281장. center 의 판 crop 을 Otsu 로 이진화하고 윤곽선 계층(자식·손자가 있는 넓이 500~6000)으로 QR finder 사각형 9개를 찾았다. 사각형을 3px 안으로 들인 crop 에서 Shi-Tomasi(간격 4, 가장자리 10px 제외)로 "
      << NP << "점을 골랐다. 추적은 그 crop 만으로 NCC(±110x22) → LK 15px → GN17 이다. 사진마다 유효 점만으로 판 좌표 2차식 12값을 맞췄다.\n\n"
      << "모델은 center_fixed 와 같은 절편 0 곡선(finder_quadratic, finder_linear). 비교용 center_quadratic 은 같은 프로그램에서 QR 내부 " << NQ << "점으로 다시 계산했다. 분할은 원래 splits.json 과 같은 규칙으로 다시 만들었다.\n\n"
      << "## finder 사각형\n\n" << tpat.md() << "\n추적 실패 " << invalid << "건 (`tracking_failures.csv`).\n\n"
      << "## 반복 5분할 20회\n\n" << rk.md() << "\n## 분할별\n\n" << key.md()
      << "\n## 12값 잡음 (절편 0 2차 곡선 잔차, 강건 sd = 1.4826 x 중앙 |잔차|)\n\n" << tsd.md() << "\n점 퍼짐(표준편차): finder x " << fmt(spF.x, 1) << ", y " << fmt(spF.y, 1) << " px / 205점 x " << fmt(spQ.x, 1) << ", y " << fmt(spQ.y, 1) << " px.\n\n"
      << "## 원래 결과와 대조 (" << par.ok << "/" << par.total << " 일치)\n\n" << par.t.md()
      << "\n## 옮기지 않은 부분\n\n- report.html, 모델 npz 저장, verify.py 의 고정 모델 재현 검사(같은 코드 경로라 생략)는 옮기지 않았다.\n- `종합/` 아래 후보점 지도·QR 구조 모서리 그림은 옮기지 않았다.\n\n"
      << "## 파일\n\n- `features.csv`: finder 점 66개\n- `tracking_failures.csv`: 추적 실패\n- `metrics.csv`: 방법 x 분할 지표 (center_quadratic 비교 포함)\n- `predictions.csv`: finder 두 방법의 모든 분할 예측\n- `이미지/`: 그림 6장\n";
    write_text(out("결과.md"), R.str());

    // ---- 보고서.md ----
    auto& RY = full[wy];
    auto& RT = full[wt];
    std::ostringstream Pm;
    Pm << "# 보고서: finder 사각형만 사용 (10-15 18:16)\n\n"
       << "## 문제\n\nQR의 finder 사각형만 써도 Y/T 판정이 되는지 확인하려 했다.\n\n"
       << "## 해결 방법\n\n9개 finder 사각형 내부 " << NP << "점만 추적해 center_fixed 와 같은 절편 0 곡선 모델로 비교했다.\n\n"
       << "## 결과\n\nfinder_quadratic 은 반복 5분할 20회에서 Y " << fmt(rep["finder_quadratic"].Y_axis_sign_accuracy * 100, 4) << "%·T "
       << fmt(rep["finder_quadratic"].T_axis_sign_accuracy * 100, 4) << "%, 재대입 " << npass << "/280으로 QR 내부 205점(Y " << fmt(rep["center_quadratic"].Y_axis_sign_accuracy * 100, 4)
       << "%, 재대입 " << npassQ << "/280)보다 나빴다.\n\n"
       << "| 지표 | finder_quadratic | finder_linear | center_quadratic (205점) |\n|---|---|---|---|\n";
    auto rowf = [&](const std::string& lab, auto fn) {
        Pm << "| " << lab;
        for (auto& c : cfgs) Pm << " | " << fn(c.method);
        Pm << " |\n";
    };
    rowf("반복 5분할 Y 축·부호", [&](const std::string& k) { return fmt(rep[k].Y_axis_sign_accuracy * 100, 4) + "%"; });
    rowf("반복 5분할 T 축·부호", [&](const std::string& k) { return fmt(rep[k].T_axis_sign_accuracy * 100, 4) + "%"; });
    rowf("재대입 축·부호", [&](const std::string& k) { return fmt(S[{k, "full280_resubstitution"}].axis_sign_accuracy * 100, 2) + "%"; });
    rowf("재대입 Y 맞은 축 MAE (mm)", [&](const std::string& k) { return fmt(S[{k, "full280_resubstitution"}].Y_matched_MAE, 4); });
    rowf("재대입 최대 벡터 오차 Y / T", [&](const std::string& k) { return fmt(S[{k, "full280_resubstitution"}].vector_Y_max, 3) + " / " + fmt(S[{k, "full280_resubstitution"}].vector_T_max, 3); });
    Pm << "\n" << img_md("01_재대입_예측오차.png", "재대입 예측 오차") << "\n\n" << img_md("02_finder점66.png", "finder 점") << "\n\n"
       << img_md("03_반복5분할_축부호정확도.png", "반복 5분할") << "\n\n" << img_md("04_12값_잡음비.png", "12값 잡음비") << "\n\n"
       << "## 튀는 값\n\n재대입에서 " << fails.size() << "장이 틀렸고 가장 크게 틀린 " << m[RY.i].file << "(정답 " << RY.axis << " " << fmts(RY.value, 2) << ")는 " << RY.pred_axis << " "
       << fmts(RY.pred, 4) << "로 읽혔다. 평균 이동 c0x 는 finder " << fmt(X[RY.i][0], 3) << " px, 205점 " << fmt(XQ[RY.i][0], 3)
       << " px로 같지만, Y/T 를 가르는 가로 모양 값(x·dx 등 dx 항)의 잡음이 finder 쪽이 " << fmt(rdx_min, 1) << "~" << fmt(rdx_max, 1) << "배 커서 T 곡선 쪽 비용이 더 작게 나왔다.\n\n"
       << ftab.md() << "\n" << img_md(fy, "Y 튀는 값") << "\n\n" << img_md(ft, "T 튀는 값")
       << "\n\n초록 상자는 center 의 finder crop, 빨간 상자는 그 사각형 점들의 평균 이동만큼 옮긴 crop, 빨간 십자는 측정 점, 화살표는 사각형 평균을 뺀 점 이동 x200이다. T 쪽 최대 오차는 "
       << m[RT.i].file << "(정답 " << RT.axis << " " << fmts(RT.value, 2) << " → " << RT.pred_axis << " " << fmts(RT.pred, 4) << ")다.\n\n"
       << "## 한줄평\n\nfinder 점 " << NP << "개는 작은 사각형 crop(약 49x52 px) 안의 픽셀만으로 추적해서 가로 모양 값(dx 항)의 잡음이 205점보다 " << fmt(rdx_min, 1) << "~"
       << fmt(rdx_max, 1) << "배(세로 dy 항은 " << fmt(rdy_min, 1) << "~" << fmt(rdy_max, 1)
       << "배) 커지고, Y/T 를 가르는 단서가 바로 그 가로 모양 값이라 평균 이동이 비슷한 Y 사진을 T 로 읽는 일이 늘어 Y 축·부호 정확도가 " << fmt(rep["finder_quadratic"].Y_axis_sign_accuracy * 100, 2) << "%로 떨어졌다.\n\n"
       << "## 원래 결과와 대조\n\n원래 값은 `codex_experiments_20261014/finder_only_20261015/checks.json`, `metrics.csv`, `predictions.csv`, `repeated_cv_summary.csv`, `tracking_failures.csv`와 `center_fixed_20261015/metrics.csv`, 분할 서명은 `gls_comparison_20261014/splits.json`에서 가져왔다. "
       << par.ok << "/" << par.total << " 일치.\n\n" << par.t.md();
    write_text(out("보고서.md"), Pm.str());
    std::cout << "pass " << npass << " repY " << rep["finder_quadratic"].Y_axis_sign_accuracy << " parity " << par.ok << "/" << par.total << "\n";
    return 0;
}
