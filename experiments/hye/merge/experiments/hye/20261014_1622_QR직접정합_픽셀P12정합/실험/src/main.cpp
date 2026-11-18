// 점 추적 대신 QR 3개 안쪽 전체 픽셀을 직접 정합해 P12 변형 12계수를 얻는다.
// 원본: codex_experiments_20261014/direct_alignment/direct_pair.py, extract_newinit.py, validate_synthetic.py,
//       measurement/extract_pair.py (GN17 초기화), modeling/new_measurement_models.py·unified.py (분류 평가)
#include <hye/dataset.hpp>
#include <hye/plot.hpp>
#include <hye/report.hpp>
#include <omp.h>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include "dense.hpp"
#include "kfold.hpp"
#include "motion.hpp"
#include "track.hpp"

using namespace hye;

namespace {

std::string file_of(const Shot& s) { return s.rel.substr(s.rel.rfind('/') + 1); }

// scipy map_coordinates(mode='reflect') 의 좌표 접기 (반 화소 대칭). 크롭 가장자리 띠에만 쓰인다
double reflect_coord(double c, int n) {
    double period = 2.0 * n;
    double t = std::fmod(c + 0.5, period);
    if (t < 0) t += period;
    return t < n ? t - 0.5 : period - t - 0.5;
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
    omp_set_num_threads(4);
    auto shots = load_1014();
    const Shot& cs = center_1014(shots);
    const cv::Mat refgray = read_gray(cs);
    const cv::Mat ref = trk::crop(refgray);
    // GN17 점 추적 (extract_pair.PairExtractor gn17, QR 928점): 흐린 LK31 -> 전체 crop GN17
    trk::Features f = trk::features928(ref);
    trk::Refiner gn(ref, f.pts, 17, 0.8);
    cv::Mat refblur;
    cv::GaussianBlur(ref, refblur, cv::Size(0, 0), 0.8);
    auto gpts = f.global();
    dr::Dense dense(refgray, 10);

    const int N = (int)shots.size();
    std::vector<dr::Result> res(N);
    std::vector<int> nvalid(N);
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N; ++i) {
        cv::Mat gray = read_gray(shots[i]), cur = trk::crop(gray), cb;
        auto init = trk::coarse(ref, cur);
        std::vector<cv::Point2f> ip;
        for (int r : f.region) ip.push_back(cv::Point2f((float)init[r].x, (float)init[r].y));
        cv::GaussianBlur(cur, cb, cv::Size(0, 0), 0.8);
        auto lt = trk::lk_track(refblur, cb, f.pts, ip, 31);
        auto g = gn.run(cur, lt.d);
        cv::Mat B, D;
        for (size_t j = 0; j < f.pts.size(); ++j) {
            if (!(lt.valid[j] && g.valid[j])) continue;
            double b[6];
            dr::basis(gpts[j].x, gpts[j].y, b);
            B.push_back(cv::Mat(1, 6, CV_64F, b).clone());
            D.push_back((cv::Mat)(cv::Mat_<double>(1, 2) << g.d[j].x, g.d[j].y));
        }
        nvalid[i] = B.rows;
        cv::Mat co;
        cv::solve(B, D, co, cv::DECOMP_SVD);
        std::array<double, 12> p0;
        for (int k = 0; k < 6; ++k) { p0[2 * k] = co.at<double>(k, 0); p0[2 * k + 1] = co.at<double>(k, 1); }
        res[i] = dense.refine_coef(dense.coefficients(gray), p0);
    }
    std::cout << "aligned " << N << "\n";
    auto idx = [&](const std::string& file) {
        for (int i = 0; i < N; ++i) if (file_of(shots[i]) == file) return i;
        return -1;
    };

    // ---- 합성 검증 (validate_synthetic.py): 알려진 3차 보간 이동 + gain 1.025 / offset 2.3 (+ 잡음 0.6, 8bit) ----
    struct Syn { double dx, dy, noise, rmse, p95, mx, mse; int it; };
    std::vector<Syn> syn;
    {
        cv::Mat raw;
        refgray(cv::Range(dense.y0, dense.y1), cv::Range(dense.x0, dense.x1)).convertTo(raw, CV_64F);
        cv::Mat cr = trk::spl::filter(raw, true);  // mode reflect 의 prefilter
        const std::vector<std::pair<double, double>> motions{{0., 0.}, {.10, .07}, {.25, -.30}, {.5, .5}, {7.7, -1.2}, {28.35, .08}, {80.15, -.45}, {-79.4, .3}};
        std::mt19937 rng(20261014);  // 원래 numpy default_rng(20261014) 의 정규 난수열은 재현할 수 없어 다른 고정 난수열을 쓴다
        std::normal_distribution<double> nd(0.0, 1.0);
        for (double noise : {0.0, 0.6}) {
            std::vector<cv::Mat> curs;
            for (auto& mo : motions) {
                cv::Mat cur(raw.size(), CV_64F);
                for (int y = 0; y < raw.rows; ++y)
                    for (int x = 0; x < raw.cols; ++x)
                        cur.at<double>(y, x) = trk::spl::eval(cr, reflect_coord(y - mo.second, raw.rows), reflect_coord(x - mo.first, raw.cols)) * 1.025 + 2.3;
                if (noise > 0) {
                    for (int y = 0; y < raw.rows; ++y)
                        for (int x = 0; x < raw.cols; ++x) {
                            double v = cur.at<double>(y, x) + noise * nd(rng);
                            cur.at<double>(y, x) = std::min(std::max(std::nearbyint(v), 0.0), 255.0);
                        }
                }
                curs.push_back(cur);
            }
            std::vector<Syn> part(motions.size());
#pragma omp parallel for schedule(dynamic)
            for (int c = 0; c < (int)motions.size(); ++c) {
                cv::Mat canvas;
                refgray.convertTo(canvas, CV_64F);
                curs[c].copyTo(canvas(cv::Range(dense.y0, dense.y1), cv::Range(dense.x0, dense.x1)));
                std::array<double, 12> init{};
                init[0] = motions[c].first + .15;
                init[1] = motions[c].second - .10;
                auto r = dense.refine_coef(dense.coefficients(canvas), init);
                std::array<double, 12> e = r.X;
                e[0] -= motions[c].first;
                e[1] -= motions[c].second;
                std::vector<double> sq, nrm;
                for (auto& rg : dense.geom.at(1))
                    for (auto& B : rg.B) {
                        double ex = 0, ey = 0;
                        for (int k = 0; k < 6; ++k) { ex += B[k] * e[2 * k]; ey += B[k] * e[2 * k + 1]; }
                        sq.push_back(ex * ex);
                        sq.push_back(ey * ey);
                        nrm.push_back(std::hypot(ex, ey));
                    }
                part[c] = {motions[c].first, motions[c].second, noise, std::sqrt(mean(sq)), percentile(nrm, .95), *std::max_element(nrm.begin(), nrm.end()), r.mse_final, r.iterations};
            }
            syn.insert(syn.end(), part.begin(), part.end());
        }
    }
    std::cout << "synthetic done\n";

    // ---- 분류 평가 ----
    // 대상 280장 (축, 값) 순서, 시행 = T± / Y1± / Y2±
    std::vector<int> target;
    int ci = 0;
    for (int i = 0; i < N; ++i) (shots[i].condition == "center" ? (void)(ci = i) : target.push_back(i));
    auto fval = [&](int i) { auto s = file_of(shots[i]); return std::stod(s.substr(1, s.size() - 5)); };
    std::stable_sort(target.begin(), target.end(), [&](int a, int b) {
        char xa = file_of(shots[a])[0], xb = file_of(shots[b])[0];
        return xa != xb ? xa < xb : fval(a) < fval(b);
    });
    const int n = (int)target.size();
    std::vector<char> axes(n);
    std::vector<double> values(n);
    std::vector<std::string> sessions(n), names(n);
    for (int k = 0; k < n; ++k) {
        names[k] = file_of(shots[target[k]]);
        axes[k] = names[k][0];
        values[k] = fval(target[k]);
        std::string s = axes[k] == 'T' ? "T" : ((std::llabs((long long)std::nearbyint(values[k] / .05)) % 2) ? "Y2" : "Y1");
        sessions[k] = s + (values[k] > 0 ? "+" : "-");
    }
    // unified.py independent_data('direct...'): QR 사각형 안 5x5 격자 75점에서 P12 변형 (center 를 뺀 X)
    mm::Data dd;
    for (int r = 0; r < 3; ++r) {
        auto q = dr::QR_BOUNDS[r];
        for (int a = 0; a < 5; ++a)
            for (int b = 0; b < 5; ++b) {
                dd.px.push_back(q[0] + 30 + (q[2] - 30 - (q[0] + 30)) * a / 4.0);
                dd.py.push_back(q[1] + 30 + (q[3] - 30 - (q[1] + 30)) * b / 4.0);
                dd.region.push_back(r);
            }
    }
    for (int k = 0; k < n; ++k) {
        cv::Mat D((int)dd.px.size(), 2, CV_64F);
        for (int j = 0; j < D.rows; ++j) {
            double b[6], x = 0, y = 0;
            dr::basis(dd.px[j], dd.py[j], b);
            for (int c = 0; c < 6; ++c) {
                x += b[c] * (res[target[k]].X[2 * c] - res[ci].X[2 * c]);
                y += b[c] * (res[target[k]].X[2 * c + 1] - res[ci].X[2 * c + 1]);
            }
            D.at<double>(j, 0) = x;
            D.at<double>(j, 1) = y;
        }
        dd.D.push_back(D);
    }
    auto evaluate = [&](const mm::Cand& c, const std::vector<kf::Split>& folds) {
        std::vector<mm::Rec> rec;
        for (auto& fo : folds) {
            mm::Model m;
            m.c = c;
            m.fit(dd, fo.first, axes, values);
            for (int i : fo.second) {
                auto p = m.predict_one(dd.D[i]);
                rec.push_back({i, axes[i], p.axis, values[i], p.value});
            }
        }
        return rec;
    };
    std::vector<int> all(n);
    for (int i = 0; i < n; ++i) all[i] = i;
    auto folds0 = kf::split(all, sessions, 3, 20261014);
    // (1) unified.py 의 direct_p12_newinit 후보 6개 (번호 24~29)
    std::vector<mm::Cand> uc;
    {
        struct S { int deg; double sh; const char* nu; bool grid; };
        const S st[6] = {{1, .1, "none", false}, {2, .1, "none", false}, {2, .1, "dy", false}, {2, .1, "both", false}, {2, 1., "none", false}, {2, .1, "dy", true}};
        for (int k = 0; k < 6; ++k) {
            mm::Cand c;
            c.meas = "direct_p12_newinit";
            c.id = "direct_p12_newinit_" + std::to_string(24 + k);
            c.degree = st[k].deg; c.shrink = st[k].sh; c.nuisance = st[k].nu; c.grid = st[k].grid;
            uc.push_back(c);
        }
    }
    std::vector<mm::Metrics> ucm;
    for (auto& c : uc) ucm.push_back(mm::metrics(evaluate(c, folds0)));
    // (2) new_measurement_models.py nested_run: P12 후보 17개, 바깥 층화 3분할 / 시행 제외, 안쪽 층화 3분할
    std::vector<mm::Cand> cs17;
    {
        struct S { const char* id; int deg; double sh; const char* nu; bool anchor, robust; };
        const S st[17] = {{"c008", 1, 1., "none", 0, 0}, {"c009", 1, .1, "none", 0, 0}, {"c010", 1, .001, "none", 0, 0}, {"c011", 2, 1., "none", 0, 0},
                          {"c012", 2, .1, "none", 0, 0}, {"c013", 2, .001, "none", 0, 0}, {"c040", 1, .1, "dy", 0, 0}, {"c041", 2, .1, "dy", 0, 0},
                          {"c042", 1, .1, "rot", 0, 0}, {"c043", 2, .1, "rot", 0, 0}, {"c044", 1, .1, "both", 0, 0}, {"c045", 2, .1, "both", 0, 0},
                          {"c059", 2, .1, "none", 0, 1}, {"c061", 1, .1, "none", 1, 0}, {"c062", 1, 1., "none", 1, 0}, {"c063", 2, .1, "none", 1, 0},
                          {"c064", 2, 1., "none", 1, 0}};
        for (auto& s : st) {
            mm::Cand c;
            c.meas = "direct_p12_newinit"; c.id = s.id; c.degree = s.deg; c.shrink = s.sh; c.nuisance = s.nu; c.anchor = s.anchor; c.robust = s.robust;
            cs17.push_back(c);
        }
    }
    struct Nested { std::vector<mm::Rec> rec; std::vector<std::string> chosen, test; };
    auto nested = [&](bool strat) {
        Nested out;
        std::vector<kf::Split> outer;
        if (strat) outer = folds0;
        else {
            std::set<std::string> us(sessions.begin(), sessions.end());
            for (auto& s : us) {
                kf::Split sp;
                for (int i = 0; i < n; ++i) (sessions[i] == s ? sp.second : sp.first).push_back(i);
                outer.push_back(sp);
            }
        }
        for (size_t fo = 0; fo < outer.size(); ++fo) {
            auto inner = kf::split(outer[fo].first, sessions, 3, 20261015 + (uint32_t)fo);
            int best = -1;
            mm::Metrics bm;
            for (size_t j = 0; j < cs17.size(); ++j) {
                auto mt = mm::metrics(evaluate(cs17[j], inner));
                if (best < 0 || mm::better(mt, bm)) { best = (int)j; bm = mt; }
            }
            auto r = evaluate(cs17[best], {outer[fo]});
            out.rec.insert(out.rec.end(), r.begin(), r.end());
            out.chosen.push_back(cs17[best].id);
            std::set<std::string> ts;
            for (int i : outer[fo].second) ts.insert(sessions[i]);
            std::string t;
            for (auto& s : ts) t += (t.empty() ? "" : ",") + s;
            out.test.push_back(t);
        }
        return out;
    };
    Nested ns = nested(true), so = nested(false);
    auto mS = mm::metrics(ns.rec), mO = mm::metrics(so.rec);
    std::cout << "stratified Y " << mS.Yok << "/" << mS.Yn << " T " << mS.Tok << "/" << mS.Tn << "  session_out Y " << mO.Yok << "/" << mO.Yn << " T " << mO.Tok << "/" << mO.Tn << "\n";

    // ---- 원래 결과 (direct_alignment/direct_p12_newinit.npz·json, synthetic_validation.csv·json,
    //      modeling/unified_full_cv_selection_metrics.csv, modeling/direct_p12_nested_*_metrics.json) ----
    Parity par;
    par.add("목적함수 픽셀 수 (inset 10)", 63247, dense.pixel_count, 0, 0);
    par.add("GN17 초기화 유효점 최소 (/928)", 928, *std::min_element(nvalid.begin(), nvalid.end()), 0, 0);
    double mse_sum = 0, mse0_sum = 0;
    for (auto& r : res) { mse_sum += r.mse_final; mse0_sum += r.mse_initial; }
    par.add("281장 평균 최종 밝기 잔차 MSE", 3.4422587038673806, mse_sum / N, 1e-5, 6);
    par.add("281장 평균 초기 밝기 잔차 MSE", 3.45575292022364, mse0_sum / N, 1e-4, 6);
    par.add("T+1.00 X[0] 평행 이동 x (px)", 79.35212519, res[idx("T+1.00.bmp")].X[0], 1e-4, 6);
    par.add("T+1.00 X[6] x^2 계수 x (px)", 0.7772479297, res[idx("T+1.00.bmp")].X[6], 1e-4, 6);
    par.add("Y-2.00 X[0] 평행 이동 x (px)", -28.72199632, res[idx("Y-2.00.bmp")].X[0], 1e-4, 6);
    par.add("Y-2.00 X[1] 평행 이동 y (px)", -0.01404389788, res[idx("Y-2.00.bmp")].X[1], 1e-4, 6);
    par.add("T-0.05 X[0] 평행 이동 x (px)", -4.245997212, res[idx("T-0.05.bmp")].X[0], 1e-4, 6);
    // 합성: 잡음 없는 8조건은 같은 입력이라 좌표 RMSE 를 그대로 비교, 잡음 8조건은 난수열이 달라 평균만 넓게 비교
    const double orig_rmse[16] = {5.492198e-08, 9.779946e-06, 9.837863e-05, 1.484505e-04, 5.984421e-05, 3.241097e-05, 1.371930e-04, 1.347094e-04,
                                  2.940109e-04, 2.263284e-04, 2.612923e-04, 3.206710e-04, 3.306878e-04, 2.867206e-04, 2.193246e-04, 4.399761e-04};
    double max_clean_diff = 0, noisy_mean = 0, noisy_mean_o = 0, max_rmse = 0;
    for (int c = 0; c < 16; ++c) {
        if (c < 8) max_clean_diff = std::max(max_clean_diff, std::abs(syn[c].rmse - orig_rmse[c]));
        else { noisy_mean += syn[c].rmse / 8; noisy_mean_o += orig_rmse[c] / 8; }
        max_rmse = std::max(max_rmse, syn[c].rmse);
    }
    par.add("합성 잡음 없는 8조건 좌표 RMSE 최대 차이 (px)", 0, max_clean_diff, 2e-6, 7);
    par.add("합성 잡음 0.6 8조건 평균 좌표 RMSE (px)", noisy_mean_o, noisy_mean, 1e-4, 6);
    par.add("합성 16조건 최대 좌표 RMSE (px)", 0.0004399760985952405, max_rmse, 2e-4, 6);
    const int orig_ue[6] = {5, 7, 8, 7, 7, 5};
    for (int k = 0; k < 6; ++k) par.add(uc[k].id + " 전체 3분할 축 오류", orig_ue[k], ucm[k].axis_err, 0, 0);
    par.add("중첩 층화 Y 정확도 (%)", 93.75, 100.0 * mS.Yok / mS.Yn, 0.01, 2);
    par.add("중첩 층화 T 정확도 (%)", 99.5, 100.0 * mS.Tok / mS.Tn, 0.01, 2);
    par.add("시행 제외 Y 정확도 (%)", 85.0, 100.0 * mO.Yok / mO.Yn, 0.01, 2);
    par.add("시행 제외 T 정확도 (%)", 98.0, 100.0 * mO.Tok / mO.Tn, 0.01, 2);
    // 검산: 원래 Python 코드(experiment.py split_folds·MotionModel, unified.independent_data)를 원래 direct_p12_newinit.npz 에
    // 그대로 돌린 값 (2026-10-30 검산, 옛 코드는 수정하지 않음). 측정 입력이 같으면 C++ 중첩 평가가 원래 코드와 같은지 본다.
    par.add("[검산] newinit 중첩 층화 Y 정확도 (%)", 96.25, 100.0 * mS.Yok / mS.Yn, 0.01, 2);
    par.add("[검산] newinit 중첩 층화 T 정확도 (%)", 99.0, 100.0 * mS.Tok / mS.Tn, 0.01, 2);
    par.add("[검산] newinit 시행 제외 Y 정확도 (%)", 97.5, 100.0 * mO.Yok / mO.Yn, 0.01, 2);
    par.add("[검산] newinit 시행 제외 T 정확도 (%)", 96.5, 100.0 * mO.Tok / mO.Tn, 0.01, 2);
    {
        auto join = [](const std::vector<std::string>& v) { std::string s; for (auto& x : v) s += (s.empty() ? "" : ", ") + x; return s; };
        const std::vector<std::string> os{"c040", "c040", "c045"}, oo{"c042", "c041", "c044", "c010", "c040", "c044"};
        par.add_text("[검산] newinit 층화 선택 후보", join(os), join(ns.chosen), os == ns.chosen);
        par.add_text("[검산] newinit 시행 제외 선택 후보", join(oo), join(so.chosen), oo == so.chosen);
    }

    // ---- 표·CSV ----
    Table xs{{"file", "X0 dx", "X1 dy", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10", "X11", "mse_initial", "mse_final", "iterations", "condition", "valid_points"}};
    for (int i = 0; i < N; ++i) {
        std::vector<std::string> r{shots[i].rel};
        for (double v : res[i].X) r.push_back(fmt(v, 9));
        r.insert(r.end(), {fmt(res[i].mse_initial, 6), fmt(res[i].mse_final, 6), std::to_string(res[i].iterations), fmt(res[i].condition, 1), std::to_string(nvalid[i])});
        xs.add(r);
    }
    xs.csv(out("direct_p12_newinit.csv"));
    Table st{{"dx (px)", "dy (px)", "잡음 sd", "좌표 RMSE (px)", "원래 RMSE (px)", "위치 오차 95% (px)", "최대 (px)", "최종 MSE", "반복"}};
    for (int c = 0; c < 16; ++c)
        st.add({fmt(syn[c].dx, 2), fmt(syn[c].dy, 2), fmt(syn[c].noise, 1), fmt(syn[c].rmse * 1e6, 3) + "e-6", fmt(orig_rmse[c] * 1e6, 3) + "e-6", fmt(syn[c].p95, 6),
                fmt(syn[c].mx, 6), fmt(syn[c].mse, 4), std::to_string(syn[c].it)});
    Table ut{{"후보", "차수", "shrink", "nuisance", "grid", "축 오류", "원래 축 오류", "Y 일치 최대 (mm)", "T 일치 최대 (deg)"}};
    for (int k = 0; k < 6; ++k)
        ut.add({uc[k].id, std::to_string(uc[k].degree), fmt(uc[k].shrink, 1), uc[k].nuisance, uc[k].grid ? "yes" : "no", std::to_string(ucm[k].axis_err),
                std::to_string(orig_ue[k]), fmt(ucm[k].Ymax, 4), fmt(ucm[k].Tmax, 4)});
    Table head{{"평가", "Y 축·부호 정답", "T 축·부호 정답", "통과", "Y 일치 최대 (mm)", "T 일치 최대 (deg)", "바깥 분할별 선택"}};
    auto join = [](const std::vector<std::string>& v) { std::string s; for (auto& x : v) s += (s.empty() ? "" : ", ") + x; return s; };
    for (auto pr : {std::make_pair(std::string("중첩 층화 3분할"), &ns), std::make_pair(std::string("시행 제외 6분할"), &so)}) {
        auto mt = mm::metrics(pr.second->rec);
        head.add({pr.first, std::to_string(mt.Yok) + " / " + std::to_string(mt.Yn) + " (" + fmt(100.0 * mt.Yok / mt.Yn, 2) + "%)",
                  std::to_string(mt.Tok) + " / " + std::to_string(mt.Tn) + " (" + fmt(100.0 * mt.Tok / mt.Tn, 2) + "%)", std::to_string(mt.n - mt.strict_fail) + " / " + std::to_string(mt.n),
                  fmt(mt.Ymax, 4), fmt(mt.Tmax, 4), join(pr.second->chosen)});
    }
    Table bad{{"평가", "사진", "시행", "정답 Y (mm)", "정답 T (deg)", "추정 축", "추정값", "P12 평행 이동 x (px)", "P12 x^2 계수 x (px)"}};
    for (auto* pr : {&ns, &so})
        for (auto& r : pr->rec)
            if (!r.joint()) {
                auto& X = res[target[r.i]].X;
                bad.add({pr == &ns ? "층화" : "시행 제외", names[r.i], sessions[r.i], fmt(r.axis == 'Y' ? r.value : 0, 2), fmt(r.axis == 'T' ? r.value : 0, 2),
                         std::string(1, r.pred_axis), fmts(r.pred, 3), fmt(X[0] - res[ci].X[0], 3), fmts(X[6] - res[ci].X[6], 4)});
            }

    // ---- 그림 ----
    {
        std::vector<double> xT, yT, xY, yY;
        for (int k = 0; k < n; ++k) {
            double v = res[target[k]].X[6] - res[ci].X[6];
            (axes[k] == 'T' ? xT : xY).push_back(values[k]);
            (axes[k] == 'T' ? yT : yY).push_back(v);
        }
        Chart a("P12 x^2 coefficient (bend) vs command", "command (T deg / Y mm)", "X[6] x^2 term dx (px)");
        a.scatter(xT, yT, color(1), "T scan").scatter(xY, yY, color(2), "Y scan").hline(0);
        a.save("01_P12_굽음계수.png");
    }
    {
        std::vector<double> x, y, yo;
        for (int c = 0; c < 16; ++c) { x.push_back(c); y.push_back(syn[c].rmse * 1e4); yo.push_back(orig_rmse[c] * 1e4); }
        Chart c("Synthetic known motion: coordinate RMSE", "case (0-7 no noise, 8-15 noise 0.6)", "RMSE (1e-4 px)");
        c.line(x, y, color(0), "C++", true).scatter(x, yo, kRed, "original", 4);
        c.save("02_합성검증_좌표RMSE.png");
    }
    auto pv = [&](const std::vector<mm::Rec>& r, const std::string& file, const std::string& title) {
        std::vector<double> ty, py, tt, pt;
        std::vector<std::string> nm;
        for (auto& x : r) {
            ty.push_back(x.axis == 'Y' ? x.value : 0); tt.push_back(x.axis == 'T' ? x.value : 0);
            py.push_back(x.pred_axis == 'Y' ? x.pred : 0); pt.push_back(x.pred_axis == 'T' ? x.pred : 0);
            nm.push_back(names[x.i].substr(0, names[x.i].size() - 4));
        }
        pred_vs_truth(file, title, ty, py, tt, pt, nm);
    };
    pv(ns.rec, "03_중첩층화_예측오차.png", "Direct P12 nested stratified");
    pv(so.rec, "04_시행제외_예측오차.png", "Direct P12 leave-session-out");
    // 튀는 값: 시행 제외에서 축·부호를 틀린 Y / T 사진 중 벡터 오차 최대
    auto worst_of = [&](char a) {
        int w = -1;
        double e = -1;
        for (size_t k = 0; k < so.rec.size(); ++k)
            if (so.rec[k].axis == a) {
                double v = std::max(so.rec[k].eY(), so.rec[k].eT());
                if (v > e + 1e-12) { e = v; w = (int)k; }
            }
        return w;
    };
    // 사진 확대: 초록 = 기준 QR 안쪽 영역, 빨강 = P12 변형으로 옮긴 QR 중심 상자. 십자 = 5x5 격자점, 화살표 = 평균 이동을 뺀 P12 변형 x300
    auto photo = [&](const mm::Rec& r, const std::string& png, const std::string& cap) {
        int si = target[r.i];
        cv::Mat im = read_color(shots[si]);
        std::vector<cv::Mat> tiles;
        for (int q = 0; q < 3; ++q) {
            auto b = dr::QR_BOUNDS[q];
            double mx = 0, my = 0;
            int cnt = 0;
            for (size_t j = 0; j < dd.px.size(); ++j)
                if (dd.region[j] == q) { mx += dd.D[r.i].at<double>((int)j, 0); my += dd.D[r.i].at<double>((int)j, 1); ++cnt; }
            mx /= cnt; my /= cnt;
            cv::Rect2d rb(b[0], b[1], b[2] - b[0], b[3] - b[1]), nb(b[0] + mx, b[1] + my, b[2] - b[0], b[3] - b[1]);
            std::vector<Mark> mk;
            for (size_t j = 0; j < dd.px.size(); ++j)
                if (dd.region[j] == q)
                    mk.push_back({{dd.px[j] + mx, dd.py[j] + my}, {dd.D[r.i].at<double>((int)j, 0) - mx, dd.D[r.i].at<double>((int)j, 1) - my}, "", kRed});
            cv::Rect roi = cv::Rect(rb | nb) + cv::Size(40, 40) - cv::Point(20, 20);
            tiles.push_back(zoom(im, roi, 420, mk, 300, q == 0 ? cap : "", {{rb, kGreen, ""}, {nb, kRed, "q" + std::to_string(q) + " dx " + fmt(mx, 2) + " dy " + fmt(my, 3)}}));
        }
        save_png(png, hstack(tiles));
    };
    int wy = worst_of('Y'), wt = worst_of('T');
    auto nm = [&](int k) { return names[so.rec[k].i].substr(0, names[so.rec[k].i].size() - 4); };
    auto cap = [&](int k) { return nm(k) + " -> " + std::string(1, so.rec[k].pred_axis) + " " + fmts(so.rec[k].pred, 3); };
    photo(so.rec[wy], "05_튀는값_Y_" + nm(wy) + ".png", cap(wy));
    photo(so.rec[wt], "06_튀는값_T_" + nm(wt) + ".png", cap(wt));

    // ---- 결과.md ----
    std::ostringstream R;
    R << "# 결과: QR 직접 정합 P12\n\n"
      << "입력: `data/clean/clean_20261014` 281장. 초기값은 QR 928점 GN17(흐린 LK31 후 전체 crop GN17) 이동에 맞춘 2차 변형이고, 그다음 QR 세 사각형 안쪽 10px(" << dense.pixel_count
      << "픽셀)에서 sigma 1 흐림·3차 스플라인으로 QR별 gain/bias를 빼며 P12 12계수를 Gauss-Newton(간격 2 최대 12회, 간격 1 최대 10회)으로 맞췄다.\n"
      << "분류: X(center 뺌)를 QR 안 5x5 격자 75점 이동으로 바꿔 unified.py와 같은 P12 모델에 넣었다. 중첩 평가는 new_measurement_models.py의 P12 후보 17개를 썼다.\n\n"
      << "## 합성 검증 (알려진 이동, 초기값 +0.15/-0.10 px 어긋남)\n\n" << st.md()
      << "\n## unified.py direct_p12_newinit 후보 (전체 3분할)\n\n" << ut.md() << "\n## 중첩 평가 (P12 후보 17개)\n\n" << head.md()
      << "\n## 축·부호를 틀린 사진\n\n" << bad.md() << "\n## 원래 결과와 대조 (" << par.ok << "/" << par.total << " 일치)\n\n" << par.t.md()
      << "\n## 옮기지 않은 부분\n\n- `extract_direct.py`의 예비 측정 direct_p12(inset 3, 옛 analysis_v6 hp8 추적값 초기화): 초기값과 QR 사각형이 옛 결과 CSV(`14_데이터_QR마커위치.csv`, `analysis_v6/*`)에서 와서 옮기지 않았다. 대신 최종 측정인 newinit(inset 10)으로 같은 중첩 평가를 돌렸다.\n"
      << "- 그래서 중첩 정확도는 원래 direct_p12(inset 3 + hp8 점 위치로 nuisance 투영) 값과 다를 수 있다.\n"
      << "- extract_newinit.py의 옛 초기값 비교(X_oldinit_same_support)는 옛 캐시가 필요해 뺐다.\n\n"
      << "## 파일\n\n- `direct_p12_newinit.csv`: 281장의 P12 12계수, 밝기 잔차, 반복 수, 조건수, 초기화 유효점\n";
    write_text(out("결과.md"), R.str());

    // ---- 보고서.md ----
    auto& WY = so.rec[wy];
    auto& WT = so.rec[wt];
    double bendT = 0, bendY = 0;
    {
        // T 와 Y 의 x^2 굽음 계수 기울기 (px / 단위)
        auto slope = [&](char a) {
            double sx = 0, sy = 0, sxx = 0, sxy = 0;
            int c = 0;
            for (int k = 0; k < n; ++k)
                if (axes[k] == a) {
                    double v = res[target[k]].X[6] - res[ci].X[6];
                    sx += values[k]; sy += v; sxx += values[k] * values[k]; sxy += values[k] * v; ++c;
                }
            return (sxy - sx * sy / c) / (sxx - sx * sx / c);
        };
        bendT = slope('T');
        bendY = slope('Y');
    }
    std::ostringstream P;
    P << "# 보고서: QR 직접 정합 픽셀 P12 (10-14 16:22)\n\n"
      << "## 문제\n\n점 추적 대신 QR 3개 안쪽 전체 픽셀을 직접 정합하면 12계수 변형을 더 정밀하게 얻을 수 있는지 확인해야 했다.\n\n"
      << "## 해결 방법\n\nGN17 점 대응으로 초기화한 뒤 QR 안쪽 " << dense.pixel_count << "픽셀에서 QR별 gain/bias를 빼고 P12 변형을 Gauss-Newton으로 맞췄다.\n\n"
      << "## 결과\n\n합성 좌표 RMSE는 최대 " << fmt(max_rmse, 6) << " px였고, 분류는 중첩 층화 Y " << fmt(100.0 * mS.Yok / mS.Yn, 2) << "%·T " << fmt(100.0 * mS.Tok / mS.Tn, 2)
      << "%, 시행 제외 Y " << fmt(100.0 * mO.Yok / mO.Yn, 2) << "%·T " << fmt(100.0 * mO.Tok / mO.Tn, 2) << "%였다.\n\n"
      << "| 지표 | 값 |\n|---|---|\n"
      << "| 목적함수 픽셀 | " << dense.pixel_count << " |\n"
      << "| 합성 16조건 최대 좌표 RMSE (px) | " << fmt(max_rmse, 6) << " |\n"
      << "| 281장 평균 밝기 잔차 MSE (초기 -> 최종) | " << fmt(mse0_sum / N, 4) << " -> " << fmt(mse_sum / N, 4) << " |\n"
      << "| 중첩 층화 Y / T 정답 | " << mS.Yok << "/" << mS.Yn << " / " << mS.Tok << "/" << mS.Tn << " |\n"
      << "| 시행 제외 Y / T 정답 | " << mO.Yok << "/" << mO.Yn << " / " << mO.Tok << "/" << mO.Tn << " |\n"
      << "| unified 후보 6개 전체 3분할 축 오류 | " << ucm[0].axis_err << ", " << ucm[1].axis_err << ", " << ucm[2].axis_err << ", " << ucm[3].axis_err << ", " << ucm[4].axis_err
      << ", " << ucm[5].axis_err << " |\n\n"
      << img_md("01_P12_굽음계수.png", "P12 굽음 계수") << "\n\n" << img_md("02_합성검증_좌표RMSE.png", "합성 검증") << "\n\n" << img_md("03_중첩층화_예측오차.png", "층화 예측 오차")
      << "\n\n" << img_md("04_시행제외_예측오차.png", "시행 제외 예측 오차") << "\n\n"
      << "## 튀는 값\n\n시행 제외에서 가장 크게 틀린 Y 사진은 " << nm(wy) << "(정답 Y " << fmt(WY.value, 2) << " mm를 " << WY.pred_axis << " " << fmts(WY.pred, 3)
      << "로 읽음), T 사진은 " << nm(wt) << "(정답 T " << fmt(WT.value, 2) << " deg를 " << WT.pred_axis << " " << fmts(WT.pred, 3)
      << "로 읽음)이다. 빠진 시행의 0점 이동을 학습에서 못 봐서 작은 이동의 평행 이동 성분을 다른 축 곡선에 붙였다.\n\n"
      << bad.md() << "\n" << img_md("05_튀는값_Y_" + nm(wy) + ".png", "Y 튀는 값") << "\n\n" << img_md("06_튀는값_T_" + nm(wt) + ".png", "T 튀는 값")
      << "\n\n초록 상자는 기준 사진의 QR, 빨간 상자는 P12 변형의 QR 평균 이동만큼 옮긴 QR이다. 빨간 십자는 5x5 격자점, 화살표는 QR 평균을 뺀 P12 변형 x300이다.\n\n"
      << "## 한줄평\n\n직접 정합은 6만 픽셀을 한 번에 맞춰 알려진 이동을 " << fmt(max_rmse, 5) << " px 안으로 재지만, Y와 T를 가르는 x^2 굽음 계수의 기울기가 T " << fmt(bendT, 4)
      << " px/deg, Y " << fmt(bendY, 4) << " px/mm로 작아 시험 시행의 0점이 빠지면 작은 이동에서 축을 바꿔 읽었다(시행 제외 Y " << mO.Yn - mO.Yok << "장·T " << mO.Tn - mO.Tok << "장 틀림).\n\n"
      << "## 원래 결과와 대조\n\n원래 값은 `direct_alignment/direct_p12_newinit.npz`·`.json`, `synthetic_validation.csv`·`.json`, `modeling/unified_full_cv_selection_metrics.csv`, "
         "`modeling/direct_p12_nested_stratified_metrics.json`·`direct_p12_nested_session_out_metrics.json`에서 가져왔다. 합성 잡음 조건은 numpy 난수열을 재현하지 못해 평균만 넓은 허용오차(1e-4 px)로 비교했다. "
         "중첩 정확도의 원래 값은 예비 측정 direct_p12(inset 3, analysis_v6 hp8 점)에서 나온 값이고 C++는 최종 측정 newinit(inset 10, 5x5 격자점)으로 돌렸다. 그래서 그 4줄은 불일치로 남기고, 같은 newinit 측정에 원래 Python 코드를 돌린 값([검산] 줄)과 C++ 값이 같은지 따로 확인했다. "
      << par.ok << "/" << par.total << " 일치.\n\n" << par.t.md();
    write_text(out("보고서.md"), P.str());
    std::cout << "parity " << par.ok << "/" << par.total << "\n";
    return 0;
}
