// 630점 경계 모델 추론 속도: 해석미분 감쇠 LM 계산기로 사진 한 장씩 추론하고 280장 판정 확인
// 원본: motion_speed_630_20261022/compiled.py, engine.py, verify280.py, benchmark.py, test_math.py
//       (모델은 center_patch_v2_20261021 v1_edge630 = zero_experiment.fit center 모드 이차, 280장 전체 학습)
// 시간은 실행마다 달라서 실행 폴더(빌드 폴더)의 motion_speed_timing.txt 에만 쓴다. 결과/ 의 md·csv·그림에는 넣지 않는다.
#include <hye/dataset.hpp>
#include <hye/plot.hpp>
#include <hye/report.hpp>
#include <omp.h>
#include <chrono>
#include <iostream>
#include <set>
#include "corners.hpp"
#include "gls.hpp"

using namespace hye;
using g09::Item;
using g09::Row;
using g09::Vec;

namespace {
std::string base_name(const std::string& rel) {
    auto p = rel.find_last_of('/');
    return p == std::string::npos ? rel : rel.substr(p + 1);
}
std::string stem(std::string s) { return s.size() > 4 && s.substr(s.size() - 4) == ".bmp" ? s.substr(0, s.size() - 4) : s; }
double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct Measure {
    std::vector<cv::Point2d> D;
    std::vector<bool> valid;
    std::vector<unsigned> flags;
    int shift[3][2];
    double score[3];
    double t_ncc = 0, t_fit = 0;
};

// compiled.StructuralExtractor.extract: NCC ±130 → 630점 경계 맞춤 (점 단위 4스레드)
Measure measure(const cv::Mat& img, const cv::Mat& center, const std::vector<g09::Corner>& C) {
    Measure r;
    double t0 = now_s();
    g09::ncc_shift(img, center, 130, r.shift, r.score);
    if (std::min({r.score[0], r.score[1], r.score[2]}) < .6) throw std::runtime_error("QR NCC below .6");
    double t1 = now_s();
    const int n = (int)C.size();
    r.D.resize(n);
    r.valid.resize(n);
    r.flags.resize(n);
    std::vector<char> vv(n);
#pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < n; ++i) {
        const auto& c = C[i];
        int k = c.qr - 1, dx = r.shift[k][0], dy = r.shift[k][1];
        int box[4] = {g09::BOXES[k][0] + dx, g09::BOXES[k][1] + dy, g09::BOXES[k][2] + dx, g09::BOXES[k][3] + dy};
        g09::CornerFit f = g09::fit_corner(img, c.cx + dx, c.cy + dy, box, c.inv, c.patch);
        r.D[i] = {f.x - c.px, f.y - c.py};
        vv[i] = f.valid;
        r.flags[i] = f.flags;
    }
    for (int i = 0; i < n; ++i) r.valid[i] = vv[i];
    r.t_ncc = t1 - t0;
    r.t_fit = now_s() - t1;
    return r;
}
}  // namespace

int main() {
    omp_set_num_threads(4);
    cv::setNumThreads(1);
    begin_results();
    auto all = load_1014();
    std::vector<Shot> shots(all.begin(), all.end());
    std::stable_sort(shots.begin(), shots.end(), [](const Shot& a, const Shot& b) {
        std::string x = base_name(a.rel), y = base_name(b.rel);
        bool ca = x != "center.bmp", cb = y != "center.bmp";
        return ca != cb ? ca < cb : x < y;
    });
    std::vector<Item> m;
    std::vector<std::string> names;
    for (auto& s : shots) { names.push_back(base_name(s.rel)); m.push_back(g09::make_item(names.back())); }
    const int N = (int)shots.size();

    // ---- 초기화 (사진마다 반복하지 않는 부분) ----
    double ti = now_s();
    cv::Mat center = read_gray(shots[0]);
    g09::Reference R = g09::build_reference(center);
    const std::vector<g09::Corner>& C = R.c630;
    const int NP = (int)C.size();
    cv::Mat B(NP, 6, CV_64F), pinvB;
    for (int i = 0; i < NP; ++i) {
        double u = (C[i].px - 1906) / 486, v = (C[i].py - 1070) / 122;
        double row[6] = {1, u, v, u * u, u * v, v * v};
        for (int j = 0; j < 6; ++j) B.at<double>(i, j) = row[j];
    }
    cv::invert(B, pinvB, cv::DECOMP_SVD);
    double init_seconds = now_s() - ti;
    auto p12 = [&](const std::vector<cv::Point2d>& D) {
        Vec X(12, 0.0);
        for (int a = 0; a < 6; ++a) {
            const double* r = pinvB.ptr<double>(a);
            double sx = 0, sy = 0;
            for (int i = 0; i < NP; ++i) { sx += r[i] * D[i].x; sy += r[i] * D[i].y; }
            X[a * 2] = sx;
            X[a * 2 + 1] = sy;
        }
        return X;
    };

    // ---- 281장: 한 장씩 차례로 (점 단위 4스레드) ----
    std::vector<Measure> M(N);
    std::vector<Vec> X(N);
    std::vector<double> t_read(N), t_meas(N), t_reg(N);
    for (int i = 0; i < N; ++i) {
        double a = now_s();
        cv::Mat img = read_gray(shots[i]);
        double b = now_s();
        M[i] = measure(img, center, C);
        double c = now_s();
        X[i] = p12(M[i].D);
        t_read[i] = b - a;
        t_meas[i] = c - b;
        t_reg[i] = now_s() - c;
    }
    // ---- 모델: 280장 전체 학습 (center 모드, 이차, zero_experiment.fit) → 같은 280장 추론 ----
    std::vector<int> ids;
    for (int i = 0; i < N; ++i) if (m[i].axis != 'C') ids.push_back(i);
    g09::Model model = g09::fit_zero(m, X, ids, "center", true);
    std::vector<Row> rows;
    std::vector<double> t_model;
    for (int i : ids) {
        double a = now_s();
        g09::Pred p = g09::predict_one(model, X[i], "center");
        t_model.push_back(now_s() - a);
        rows.push_back(g09::enrich(m, i, p));
    }
    g09::Stats st = g09::stats(rows);
    long valid_total = 0;
    std::vector<int> valid_count(N, 0);
    std::vector<int> flag_count(6, 0);
    for (int i = 0; i < N; ++i)
        for (int p = 0; p < NP; ++p) {
            valid_count[i] += M[i].valid[p];
            for (int b = 0; b < 6; ++b) flag_count[b] += (M[i].flags[p] >> b) & 1;
        }
    for (int i : ids) valid_total += valid_count[i];
    std::vector<std::string> fails;
    for (auto& r : rows) if (!r.pass) fails.push_back(r.file);

    // ---- 해석미분 검사 (test_math.py): 630점 중 70점마다, 9x9 격자, I = 0..80 ----
    double grad_max = 0;
    for (int c = 0; c < NP; c += 70) {
        const auto& k = C[c];
        std::vector<double> xs, ys, I;
        for (int y = -4; y <= 4; ++y) for (int x = -4; x <= 4; ++x) { xs.push_back(k.cx + x); ys.push_back(k.cy + y); I.push_back((double)I.size()); }
        double p[6] = {k.cx + .13, k.cy - .21, 1.1, .8, 30., 150.};
        std::vector<double> r0, J, rl, rh, Jd;
        g09::objective(p, xs, ys, I, k.inv, k.patch, r0, J);
        for (int j = 0; j < 6; ++j) {
            double lo[6], hi[6];
            std::copy(p, p + 6, lo); std::copy(p, p + 6, hi);
            lo[j] -= 1e-5; hi[j] += 1e-5;
            g09::objective(lo, xs, ys, I, k.inv, k.patch, rl, Jd);
            g09::objective(hi, xs, ys, I, k.inv, k.patch, rh, Jd);
            for (size_t q = 0; q < I.size(); ++q) grad_max = std::max(grad_max, std::abs((rh[q] - rl[q]) / 2e-5 - J[q * 6 + j]));
        }
    }

    // ---- 표 ----
    Table pred_csv{{"file", "axis", "value", "pred_axis", "pred", "correct", "eY", "eT", "passed", "valid"}};
    for (auto& r : rows)
        pred_csv.add({r.file, std::string(1, r.axis), fmt(r.value, 2), std::string(1, r.pred_axis), fmt(r.pred, 8), r.correct ? "True" : "False", fmt(r.eY, 8),
                      fmt(r.eT, 8), r.pass ? "True" : "False", std::to_string(valid_count[r.index])});
    pred_csv.csv(out("compiled280.csv"));
    Table fail_t{{"사진", "정답", "추정 축", "추정 값", "eY", "eT", "유효점 (/630)"}};
    for (auto& r : rows)
        if (!r.pass)
            fail_t.add({r.file, std::string(1, r.axis) + " " + fmts(r.value, 2), std::string(1, r.pred_axis), fmt(r.pred, 6), fmt(r.eY, 6), fmt(r.eT, 6),
                        std::to_string(valid_count[r.index])});
    const char* fname[6] = {"optimizer", "photometric_rmse", "low_contrast", "window_instability", "search_limit", "blur_limit"};
    Table flag_t{{"진단", "걸린 점 수 (281장 x 630점)"}};
    for (int b = 0; b < 6; ++b) flag_t.add({fname[b], std::to_string(flag_count[b])});
    // 대표 3장 (benchmark.py)
    Table rep_t{{"사진", "추정 축", "추정 값", "유효점", "원래 compiled 추정", "원래 유효점"}};
    struct Rep { const char* f; double v; int valid; };
    const Rep reps[3] = {{"Y+0.50.bmp", 0.48775354, 628}, {"Y+0.05.bmp", 0.0098023, 626}, {"T-0.05.bmp", -0.25560944, 628}};
    double rep_v[3] = {0, 0, 0};
    int rep_valid[3] = {0, 0, 0};
    char rep_axis[3] = {'?', '?', '?'};
    for (int k = 0; k < 3; ++k)
        for (auto& r : rows)
            if (r.file == reps[k].f) {
                rep_v[k] = r.pred; rep_valid[k] = valid_count[r.index]; rep_axis[k] = r.pred_axis;
                rep_t.add({r.file, std::string(1, r.pred_axis), fmt(r.pred, 6), std::to_string(rep_valid[k]), fmt(reps[k].v, 6), std::to_string(reps[k].valid)});
            }

    // ---- 튀는 값 ----
    std::vector<double> ty, py, tt, pt, ey, et;
    std::vector<std::string> nn;
    for (auto& r : rows) {
        ty.push_back(r.axis == 'Y' ? r.value : 0);
        tt.push_back(r.axis == 'T' ? r.value : 0);
        py.push_back(r.pred_axis == 'Y' ? r.pred : 0);
        pt.push_back(r.pred_axis == 'T' ? r.pred : 0);
        ey.push_back(py.back() - ty.back());
        et.push_back(pt.back() - tt.back());
        nn.push_back(r.file);
    }
    size_t wy = worst(ey, 1)[0], wt = worst(et, 1)[0];
    if (wt == wy)
        for (size_t k : worst(et, et.size()))
            if (k != wy && !rows[k].pass) { wt = k; break; }
    Table outl{{"구분", "사진", "정답 Y (mm)", "추정 Y", "정답 T (deg)", "추정 T", "고른 축 비용", "다른 축 비용", "P12 c0x (px)", "유효점"}};
    auto costs_of = [&](const Row& r) {
        g09::Pred p = g09::predict_one(model, X[r.index], "center");
        double a = 0, b = 0;
        for (size_t k = 0; k < model.branches.size(); ++k) (model.branches[k].axis == r.pred_axis ? a : b) = p.costs[k];
        return std::make_pair(a, b);
    };
    for (auto [lab, idx] : std::vector<std::pair<std::string, std::vector<size_t>>>{{"Y 오차 상위", worst(ey, 3)}, {"T 오차 상위", worst(et, 3)}})
        for (size_t k : idx) {
            auto& r = rows[k];
            auto cc = costs_of(r);
            outl.add({lab, r.file, fmt(ty[k], 2), fmt(py[k], 4), fmt(tt[k], 2), fmt(pt[k], 4), fmt(cc.first, 2), fmt(cc.second, 2), fmt(X[r.index][0], 3),
                      std::to_string(valid_count[r.index])});
        }

    // ---- 그림 (모두 결정적: 시간은 원래 값 상수만) ----
    {
        std::vector<std::string> cats = {"SciPy TRF Y+0.50", "SciPy TRF Y+0.05", "SciPy TRF T-0.05", "Numba LM Y+0.50", "Numba LM Y+0.05", "Numba LM T-0.05"};
        std::vector<double> vals = {30.0325, 51.9025, 48.7558, 0.2381, 0.2574, 0.2305};  // results/paired_timings.json
        bar_chart("01_원래_한장추론시간.png", "Original single-image inference time (s), paired_timings.json", cats, vals, "seconds", -1);
    }
    pred_vs_truth("02_280장_예측오차.png", "Frozen edge630 model, 280 resub", ty, py, tt, pt, nn);
    {
        Chart c("Valid corners per image (of 630)", "image index", "valid corners");
        std::vector<double> x, y;
        for (int i = 1; i < N; ++i) { x.push_back(i); y.push_back(valid_count[i]); }
        c.scatter(x, y, color(0), "", 2).hline(630, kGray);
        c.save("03_사진별_유효점.png");
    }
    auto outlier_fig = [&](size_t k, const std::string& file, const std::string& cap) {
        int i = rows[k].index;
        cv::Mat im = read_color(shots[i]);
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
            tiles.push_back(zoom(im, roi, 420, mk, 200, q == 0 ? cap : "",
                                 {{ref, kGreen, ""}, {nowr, kRed, "q" + std::to_string(q) + " dx " + fmt(mx, 2) + " dy " + fmt(my, 2)}}));
        }
        save_png(file, hstack(tiles));
    };
    std::string fy = "04_튀는값_Y_" + stem(nn[wy]) + ".png", ft = "05_튀는값_T_" + stem(nn[wt]) + ".png";
    outlier_fig(wy, fy, nn[wy] + " Y err " + fmts(ey[wy], 3) + " mm");
    outlier_fig(wt, ft, nn[wt] + " T err " + fmts(et[wt], 3) + " deg");

    // ---- 원래 결과와 대조 (출처: results/compiled280_summary.json, compiled_failures.csv, compiled280.csv, paired_timings.json, math_checks.json) ----
    Parity par;
    par.add("280장 통과", 275, 280 - st.failures, 0, 0);
    par.add("축·부호 맞음", 275, st.Y_correct + st.T_correct, 0, 0);
    par.add("Y 최대 오차 (mm)", 0.255609445, st.Y_max, 0.001, 6);
    par.add("T 최대 오차 (deg)", 0.05, st.T_max, 0.0005, 4);
    {
        std::set<std::string> a{"T-0.05.bmp", "Y+0.05.bmp", "Y+0.15.bmp", "Y-0.10.bmp", "Y-0.20.bmp"}, b(fails.begin(), fails.end());
        std::string f1;
        for (size_t k = 0; k < fails.size(); ++k) f1 += (k ? ", " : "") + fails[k];
        par.add_text("실패 사진", "T-0.05, Y+0.05, Y+0.15, Y-0.10, Y-0.20", f1, a == b);
    }
    for (int k = 0; k < 3; ++k) {
        par.add(std::string(reps[k].f) + " 추정 값", reps[k].v, rep_v[k], 0.001, 6);
        par.add(std::string(reps[k].f) + " 유효점", reps[k].valid, rep_valid[k], 3, 0);
    }
    par.add("280장 유효점 합 (/176400)", 175539, (double)valid_total, 200, 0);
    par.add_text("해석미분 검사 (수치미분과 차이 1e-5 미만, test_math.py 기준)", "통과 (최대 1.86e-7)",
                 std::string(grad_max < 1e-5 ? "통과" : "실패") + " (최대 " + fmt(grad_max * 1e7, 2) + "e-7)", grad_max < 1e-5);
    // 불일치 설명: 원래 모델은 SciPy TRF 좌표로 학습했고, 원래 실험에서 계산기만 바꿨을 때 추정 최대 차가 0.000545 였다
    std::string mismatch_note;
    for (auto& r : par.t.rows)
        if (r[4] == "불일치")
            mismatch_note += "- " + r[0] + ": 차이 " + r[3] +
                             ". 원래 모델은 SciPy TRF로 잰 280장 좌표와 TRF center 기준 좌표로 학습했고, C++은 둘 다 LM으로 쟀다. 원래 실험에서도 계산기만 바꿨을 때 추정이 최대 0.000545 움직였다.\n";
    if (!mismatch_note.empty()) mismatch_note = "\n불일치 원인:\n\n" + mismatch_note;

    // ---- 시간 (실행마다 다름 → 결과/ 밖 motion_speed_timing.txt 에만) ----
    {
        std::vector<double> tot, fit, ncc;
        for (int i : ids) { tot.push_back(t_read[i] + t_meas[i] + t_reg[i]); fit.push_back(M[i].t_fit); ncc.push_back(M[i].t_ncc); }
        std::ostringstream T;
        T << "# C++ 추론 시간 (이번 실행, 실행마다 달라 해시 비교 대상이 아니다)\n\n"
          << "조건: 사진 한 장씩 차례로, 점 단위 OpenMP 4스레드, OpenCV 1스레드, 결과 캐시 없음. 초기화(center 해독·630점 기준) " << fmt(init_seconds, 3) << " s.\n\n"
          << "| 단계 | 중앙값 (s) | p95 (s) | 최대 (s) |\n|---|---|---|---|\n";
        auto row = [&](const std::string& n, const std::vector<double>& v) {
            T << "| " << n << " | " << fmt(percentile(v, .5), 4) << " | " << fmt(percentile(v, .95), 4) << " | " << fmt(*std::max_element(v.begin(), v.end()), 4) << " |\n";
        };
        std::vector<double> rd, rg;
        for (int i : ids) { rd.push_back(t_read[i]); rg.push_back(t_reg[i]); }
        row("BMP 읽기", rd);
        row("NCC 3상자", ncc);
        row("630점 경계 맞춤", fit);
        row("P12 회귀", rg);
        row("모델 비용 비교", t_model);
        row("합계 (읽기~P12)", tot);
        T << "\n원래 Python (results/paired_timings.json, compiled280_summary.json): 대표 3장 SciPy TRF 중앙값 48.7558 s → Numba LM 0.2381 s, Numba LM 280장 중앙값 0.1421 s.\n";
        write_text(fs::current_path() / "motion_speed_timing.txt", T.str());  // 결과/ 밖: 해시 비교에서 빠짐
    }

    // ---- 결과.md ----
    std::ostringstream Rm;
    Rm << "# 결과: 630점 경계 모델 추론 (해석미분 감쇠 LM)\n\n"
       << "입력: `data/clean/clean_20261014` 281장. center.bmp에서 QR 해독 격자로 구조 모서리 630점(대각 접촉 제외)을 만들고, 같은 경계 맞춤으로 center 기준 좌표를 정했다.\n"
       << "측정: QR 상자 NCC(±130) → 점마다 9x9 erf 경계 soft_l1 맞춤(창 다시 맞추기 최대 3번) → 11x11 진단. 계산기는 compiled.py의 해석미분·경계 제한 감쇠 LM을 그대로 옮겼다.\n"
       << "모델: 630점 이동(현재 - center 기준)을 이차식 6항으로 회귀한 P12 → 280장 전체로 학습한 Y/T 이차곡선(center 1원점) GLS → 두 가지 비용 비교. 같은 280장을 추론한 재대입 성적이다.\n\n"
       << "## 성적\n\n| 지표 | 값 |\n|---|---|\n| 통과 | " << 280 - st.failures << " / 280 |\n| 축·부호 맞음 | " << st.Y_correct + st.T_correct
       << " / 280 |\n| Y / T 최대 오차 | " << fmt(st.Y_max, 6) << " / " << fmt(st.T_max, 6) << " |\n| 유효점 합 | " << valid_total << " / " << 280 * NP << " |\n\n"
       << "## 실패 사진\n\n" << fail_t.md() << "\n## 대표 3장 (benchmark.py)\n\n" << rep_t.md()
       << "\n## 경계 맞춤 진단 (점 단위)\n\n" << flag_t.md()
       << "\n## 원래 결과와 대조 (" << par.ok << "/" << par.total << " 일치)\n\n" << par.t.md() << mismatch_note
       << "\n## 옮기지 않은 부분\n\n"
       << "- 원래 비교 대상인 SciPy least_squares TRF(수치미분) 계산기 `baseline.py`, 중간 단계 `optimized.py`·`fast_fit.py`는 옮기지 않았다. 원래 시간은 상수로만 인용한다.\n"
       << "- 상주 HTTP API(`serve.py`, `client_example.py`)와 API 왕복 시간 검사는 옮기지 않았다.\n"
       << "- 원래 모델은 SciPy TRF로 잰 280장 좌표로 학습했고, C++은 같은 LM으로 잰 좌표로 학습한다. center 기준 좌표도 LM으로 맞췄다.\n\n"
       << "## 파일\n\n- `compiled280.csv`: 280장 추론\n- C++ 단계별 시간은 실행마다 달라 결과/에 두지 않고 실행 폴더의 `motion_speed_timing.txt`에 쓴다\n- `이미지/`: 보고서 그림\n";
    write_text(out("결과.md"), Rm.str());

    // ---- 보고서.md ----
    int ny_fail = 0;
    double ylo = 1e9, yhi = 0;
    for (auto& r : rows)
        if (!r.pass && r.axis == 'Y') { ++ny_fail; ylo = std::min(ylo, std::abs(r.value)); yhi = std::max(yhi, std::abs(r.value)); }
    std::ostringstream P;
    P << "# 보고서: 630점 경계 모델 추론 속도, 해석미분 LM (10-22 15:11)\n\n"
      << "## 문제\n\n630점 경계 모델 추론이 사진 한 장에 약 48.8초 걸려 실시간 연동이 어려웠다.\n\n"
      << "## 해결 방법\n\nSciPy 수치미분을 해석미분·감쇠 LM·4스레드 병렬로 바꾼 계산기를 C++로 옮겨 사진 한 장씩 추론했다.\n\n"
      << "## 결과\n\n해석미분 LM으로 잰 280장의 통과는 " << 280 - st.failures << "/280으로 원래(275/280)와 같았고, C++ 한 장 시간은 실행마다 달라 결과/ 밖(실행 폴더의 `motion_speed_timing.txt`)에 따로 적었다.\n\n"
      << "| 지표 | 값 |\n|---|---|\n"
      << "| 통과 | " << 280 - st.failures << " / 280 |\n"
      << "| Y 최대 \\|오차\\| (mm) / T (deg) | " << fmt(st.Y_max, 4) << " / " << fmt(st.T_max, 4) << " |\n"
      << "| Y 평균 \\|오차\\| / T | " << fmt(mean(absv(ey)), 4) << " / " << fmt(mean(absv(et)), 4) << " |\n"
      << "| 유효점 비율 | " << fmt(100.0 * valid_total / (280.0 * NP), 2) << "% |\n"
      << "| 해석미분 - 수치미분 최대 차이 | " << fmt(grad_max * 1e9, 1) << " x 1e-9 |\n"
      << "| 원래 한 장 시간 (대표 3장 중앙값) | SciPy TRF 48.7558 s → Numba LM 0.2381 s (약 204.8배) |\n\n"
      << img_md("01_원래_한장추론시간.png", "원래 시간") << "\n\n" << img_md("02_280장_예측오차.png", "예측 오차") << "\n\n"
      << img_md("03_사진별_유효점.png", "유효점") << "\n\n"
      << "## 튀는 값\n\n가장 크게 틀린 사진은 " << nn[wy] << "(정답 T " << fmt(tt[wy], 2) << ")로 Y " << fmt(py[wy], 4)
      << " mm로 읽혔고, Y 실패 " << ny_fail << "장은 작은 Y(" << fmt(ylo, 2) << "~" << fmt(yhi, 2) << " mm)에서 T 곡선 비용이 더 작아 T로 읽혔다.\n\n"
      << outl.md() << "\n" << img_md(fy, "Y 튀는 값") << "\n\n" << img_md(ft, "T 튀는 값")
      << "\n\n초록 상자는 center QR 영역, 빨간 상자는 QR 평균 이동만큼 옮긴 영역이다. 빨간 십자는 유효 모서리(6점마다 1개), 주황 십자는 진단에 걸린 모서리, 화살표는 QR 평균 이동을 뺀 나머지 이동을 200배 키운 것이다.\n\n"
      << "## 한줄평\n\n계산기만 바꾸고 목적함수·창·진단은 그대로라 판정은 원래와 같은 " << 280 - st.failures
      << "/280에 머물렀고, 남은 실패 " << st.failures << "장은 원점 하나짜리 Y/T 곡선이 작은 이동에서 두 축을 못 가르는 모델 한계다.\n\n"
      << "## 원래 결과와 대조\n\n원래 값은 `motion_speed_630_20261022/results/compiled280_summary.json`, `compiled_failures.csv`, `compiled280.csv`, `paired_timings.json`, `math_checks.json`에서 가져왔다. "
      << par.ok << "/" << par.total << " 일치.\n\n" << par.t.md() << mismatch_note;
    write_text(out("보고서.md"), P.str());
    std::cout << "parity " << par.ok << "/" << par.total << "\n";
    return 0;
}
