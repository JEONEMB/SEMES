// Y 단일축 80장 630점 변위의 지그재그: 공통 직선 / 시행별 절편 / 시행별 직선 비교 (원본: eda_y_trials_20261022/analyze.py)
//   630점 좌표는 track_corners.py 방식(struct630.hpp)으로 사진에서 다시 잰다. 시행 = 파일명 단계(0.1 간격 / 0.05 홀수)와 부호.
#include "struct630.hpp"
#include <hye/dataset.hpp>
#include <hye/plot.hpp>
#include <hye/report.hpp>
#include <iostream>
#include <map>
#include <set>

using namespace hye;
using cv::Mat;

namespace {

struct Poly { std::vector<double> x, y; cv::Scalar c; };
cv::Mat lines_panel(const std::string& title, const std::string& xl, const std::string& yl, const std::vector<Poly>& lines,
                    const std::vector<std::pair<std::string, cv::Scalar>>& legend, double lim = kNaN) {
    double a = 1e300, b = -1e300, c = 1e300, d = -1e300;
    for (auto& l : lines)
        for (size_t i = 0; i < l.x.size(); ++i) { a = std::min(a, l.x[i]); b = std::max(b, l.x[i]); c = std::min(c, l.y[i]); d = std::max(d, l.y[i]); }
    if (std::isfinite(lim)) { c = -lim; d = lim; }
    double px = (b - a) * 0.04, py = (d - c) * 0.04;
    Chart ch(title, xl, yl);
    ch.W = 760; ch.H = 460;
    ch.range(a - px, b + px, c - py, d + py);
    for (auto& [lab, col] : legend) ch.scatter({}, {}, col, lab);
    cv::Mat m = ch.render(), ov = m.clone();
    cv::Rect clip(ch.L, ch.T, ch.W - ch.L - ch.R + 1, ch.H - ch.T - ch.B + 1);
    for (auto& l : lines)
        for (size_t i = 1; i < l.x.size(); ++i) {
            cv::Point p = ch.P(l.x[i - 1], l.y[i - 1]), q = ch.P(l.x[i], l.y[i]);
            if (cv::clipLine(clip, p, q)) cv::line(ov, p, q, l.c, 1, cv::LINE_AA);
        }
    cv::addWeighted(ov, 0.2, m, 0.8, 0, m);
    if (std::isfinite(lim) || (c < 0 && d > 0)) { cv::Point p = ch.P(a - px, 0), q = ch.P(b + px, 0); cv::line(m, p, q, kBlack, 1); }
    return m;
}

Mat lstsq(const Mat& A, const Mat& B) { Mat X; cv::solve(A, B, X, cv::DECOMP_SVD); return X; }
double rms_all(const Mat& E) { return std::sqrt(E.dot(E) / E.total()); }

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
    auto shots = load_1014();
    const cv::Mat center = s630::read_gray(center_1014(shots).path);
    auto all = s630::structural_corners(s630::decode_qr(center));
    s630::refine_center(center, all);
    auto cs = s630::non_diagonal(all);
    const int P = (int)cs.size();
    // Y 80장, Y 명령 순
    std::vector<int> si;
    for (size_t i = 0; i < shots.size(); ++i) if (shots[i].condition == "Y") si.push_back((int)i);
    std::stable_sort(si.begin(), si.end(), [&](int a, int b) { return shots[a].y < shots[b].y; });
    const int N = (int)si.size();
    std::vector<s630::Track> tr(N);
    s630::Track tc = s630::track(center, center, cs);
#pragma omp parallel for schedule(dynamic) num_threads(4)
    for (int k = 0; k < N; ++k) tr[k] = s630::track(s630::read_gray(shots[si[k]].path), center, cs);
    std::cout << "tracked " << N << "\n";
    // 시행: hw_eda 의 group 과 같은 규칙. 0.1 간격 = 짝수 0.05 단계, 0.05 홀수 = 홀수 단계
    const std::vector<std::string> groups{"Y 0.05 홀수 +", "Y 0.05 홀수 -", "Y 0.1 간격 +", "Y 0.1 간격 -"};
    const std::vector<std::string> gen{"0.05 odd +", "0.05 odd -", "0.1 step +", "0.1 step -"};
    std::vector<int> gi(N);
    std::vector<double> x(N);
    std::vector<std::string> nm(N);
    int invalid = 0;
    for (int k = 0; k < N; ++k) {
        double v = shots[si[k]].y;
        x[k] = v;
        bool odd = std::llabs(std::llround(v / 0.05)) % 2;
        gi[k] = (odd ? 0 : 2) + (v > 0 ? 0 : 1);
        auto r = shots[si[k]].rel;
        nm[k] = r.substr(r.rfind('/') + 1, r.size() - r.rfind('/') - 5);
        for (int j = 0; j < P; ++j) invalid += !tr[k].obs[j].valid;
    }
    std::map<std::string, Mat> D;
    for (int comp = 0; comp < 2; ++comp) {
        Mat d(N, P, CV_64F);
        for (int k = 0; k < N; ++k)
            for (int j = 0; j < P; ++j) d.at<double>(k, j) = comp ? tr[k].obs[j].y - tc.obs[j].y : tr[k].obs[j].x - tc.obs[j].x;
        D[comp ? "y" : "x"] = d;
    }
    // 설계 행렬
    Mat G = Mat::zeros(N, 4, CV_64F), X1(N, 1, CV_64F), ONE = Mat::ones(N, 1, CV_64F);
    for (int k = 0; k < N; ++k) { G.at<double>(k, gi[k]) = 1; X1.at<double>(k) = x[k]; }
    Mat Gx = G.clone();
    for (int k = 0; k < N; ++k) Gx.row(k) *= x[k];
    Mat Mp, Mi, Ml;
    cv::hconcat(ONE, X1, Mp);
    cv::hconcat(G, X1, Mi);
    cv::hconcat(G, Gx, Ml);
    Table modelT{{"성분", "모델", "잔차 RMS (px)", "잔차 제곱합", "점당 매개변수"}};
    std::map<std::string, double> mrms;
    struct PosSum { double pooled, inter, line, red_i, red_l, slope_mean; std::vector<double> icpt; };
    std::map<std::string, PosSum> pos;
    Table trialT{{"성분", "시행", "기울기 평균", "절편 평균 (px)", "점 잔차 RMS 중앙값 (px)", "공통 잔차 RMS (px)", "공간 잔차 RMS (px)"}};
    std::map<std::string, std::vector<double>> tslope, ticpt;
    std::map<std::string, std::vector<double>> common_pooled;  // 양수 공통 직선 잔차의 사진 평균
    std::vector<int> posk;
    for (int k = 0; k < N; ++k) if (x[k] > 0) posk.push_back(k);
    std::vector<cv::Mat> figs1, figs2;
    for (auto comp : {"x", "y"}) {
        const Mat& Dm = D[comp];
        for (auto [name, A] : std::vector<std::pair<std::string, Mat>>{{"pooled", Mp}, {"trial_intercept", Mi}, {"trial_line", Ml}}) {
            Mat E = Dm - A * lstsq(A, Dm);
            mrms[std::string(comp) + name] = rms_all(E);
            modelT.add({comp, name, fmt(rms_all(E), 6), fmt(E.dot(E), 4), std::to_string(A.cols)});
        }
        // 양수 구간: 시행 2개 (0.05 홀수 +, 0.1 간격 +)
        const int n = (int)posk.size();
        Mat dp(n, P, CV_64F), ap(n, 2, CV_64F), ai(n, 3, CV_64F), al(n, 4, CV_64F);
        std::vector<double> xp(n);
        std::vector<int> gp(n);
        for (int r = 0; r < n; ++r) {
            int k = posk[r];
            Dm.row(k).copyTo(dp.row(r));
            xp[r] = x[k];
            gp[r] = gi[k] == 0 ? 0 : 1;
            double g0 = gp[r] == 0, g1 = gp[r] == 1;
            double ra[2] = {1, x[k]}, ri[3] = {g0, g1, x[k]}, rl[4] = {g0, g1, g0 * x[k], g1 * x[k]};
            for (int c = 0; c < 2; ++c) ap.at<double>(r, c) = ra[c];
            for (int c = 0; c < 3; ++c) ai.at<double>(r, c) = ri[c];
            for (int c = 0; c < 4; ++c) al.at<double>(r, c) = rl[c];
        }
        Mat bp = lstsq(ap, dp), bi = lstsq(ai, dp), bl = lstsq(al, dp);
        Mat ep = dp - ap * bp, ei = dp - ai * bi, el = dp - al * bl;
        PosSum ps{rms_all(ep), rms_all(ei), rms_all(el), 1 - ei.dot(ei) / ep.dot(ep), 1 - el.dot(el) / ep.dot(ep), cv::mean(bi.row(2))[0], {}};
        ps.icpt = {cv::mean(bi.row(0))[0], cv::mean(bi.row(1))[0]};
        pos[comp] = ps;
        std::vector<double> cp;
        for (int r = 0; r < n; ++r) cp.push_back(cv::mean(ep.row(r))[0]);
        common_pooled[comp] = cp;
        // 그림 1: 양수 구간 네 패널 (같은 세로 범위)
        auto mk = [&](const Mat& E, int only, const cv::Scalar& col) {
            std::vector<Poly> L;
            for (int j = 0; j < P; ++j) {
                for (int g = 0; g < 2; ++g) {
                    if (only >= 0 && g != only) continue;
                    Poly p;
                    p.c = only == -2 ? col : color(g == 0 ? 0 : 2);
                    for (int r = 0; r < n; ++r) if (only == -2 || gp[r] == g) { p.x.push_back(xp[r]); p.y.push_back(E.at<double>(r, j)); }
                    L.push_back(p);
                    if (only == -2) break;
                }
            }
            return L;
        };
        double lim = 0;
        for (int r = 0; r < n; ++r) for (int j = 0; j < P; ++j) lim = std::max(lim, std::abs(ep.at<double>(r, j)));
        std::string cl = std::string(comp) == "x" ? "horizontal" : "vertical";
        std::vector<std::pair<std::string, cv::Scalar>> leg{{gen[0], color(0)}, {gen[2], color(2)}};
        cv::Mat p1 = lines_panel("(1) pooled line, trials mixed | RMS " + fmt(ps.pooled, 4), "Y command", cl + " residual (px)", mk(ep, -2, kGray), {}, lim);
        std::vector<Poly> l2 = mk(ep, 0, {}), l2b = mk(ep, 1, {});
        l2.insert(l2.end(), l2b.begin(), l2b.end());
        cv::Mat p2 = lines_panel("(2) same residual split by trial", "Y command", cl + " residual (px)", l2, leg, lim);
        std::vector<Poly> l3 = mk(ei, 0, {}), l3b = mk(ei, 1, {});
        l3.insert(l3.end(), l3b.begin(), l3b.end());
        cv::Mat p3 = lines_panel("(3) shared slope + trial intercept | RMS " + fmt(ps.inter, 4), "Y command", cl + " residual (px)", l3, leg, lim);
        std::vector<Poly> l4 = mk(el, 0, {}), l4b = mk(el, 1, {});
        l4.insert(l4.end(), l4b.begin(), l4b.end());
        cv::Mat p4 = lines_panel("(4) trial slope + intercept | RMS " + fmt(ps.line, 4), "Y command", cl + " residual (px)", l4, leg, lim);
        save_png(std::string("01_양수구간비교_") + comp + ".png", vstack({hstack({p1, p2}), hstack({p3, p4})}));
        // 시행별 (4개)
        std::vector<cv::Mat> cols;
        for (int g = 0; g < 4; ++g) {
            std::vector<int> ix;
            for (int k = 0; k < N; ++k) if (gi[k] == g) ix.push_back(k);
            const int m = (int)ix.size();
            Mat v(m, P, CV_64F), A(m, 2, CV_64F);
            for (int r = 0; r < m; ++r) { Dm.row(ix[r]).copyTo(v.row(r)); A.at<double>(r, 0) = 1; A.at<double>(r, 1) = x[ix[r]]; }
            Mat b = lstsq(A, v), e = v - A * b;
            std::vector<double> prms, common, spatial2;
            for (int j = 0; j < P; ++j) { Mat c = e.col(j); prms.push_back(std::sqrt(c.dot(c) / m)); tslope[std::string(comp) + groups[g]].push_back(b.at<double>(1, j)); }
            for (int r = 0; r < m; ++r) {
                double cm = cv::mean(e.row(r))[0];
                common.push_back(cm);
                double s = 0;
                for (int j = 0; j < P; ++j) s += (e.at<double>(r, j) - cm) * (e.at<double>(r, j) - cm);
                spatial2.push_back(s / P);
            }
            double smean = cv::mean(b.row(1))[0], imean = cv::mean(b.row(0))[0];
            ticpt[std::string(comp) + groups[g]] = {smean, imean, percentile(prms, .5), rms(common), std::sqrt(mean(spatial2))};
            trialT.add({comp, groups[g], fmt(smean, 4), fmt(imean, 4), fmt(percentile(prms, .5), 4), fmt(rms(common), 4), fmt(std::sqrt(mean(spatial2)), 4)});
            std::vector<Poly> Lv, Le;
            for (int j = 0; j < P; ++j) {
                Poly a{{}, {}, color(g)}, bb{{}, {}, color(g)};
                for (int r = 0; r < m; ++r) { a.x.push_back(x[ix[r]]); a.y.push_back(v.at<double>(r, j)); bb.x.push_back(x[ix[r]]); bb.y.push_back(e.at<double>(r, j)); }
                Lv.push_back(a);
                Le.push_back(bb);
            }
            cols.push_back(vstack({lines_panel(gen[g] + ": " + std::to_string(m) + " shots x 630", "Y command", cl + " shift (px)", Lv, {}),
                                   lines_panel("trial line residual | RMS " + fmt(rms_all(e), 4), "Y command", cl + " residual (px)", Le, {}, lim)}));
        }
        save_png(std::string("02_시행4개_") + comp + ".png", vstack({hstack({cols[0], cols[1]}), hstack({cols[2], cols[3]})}));
    }

    // 튀는 값: 양수 공통 직선에서 사진 평균 잔차가 가장 큰 사진
    auto& cpx = common_pooled["x"];
    size_t wr = worst(cpx, 1)[0];
    int wk = posk[wr];
    Table outl{{"사진", "시행", "Y 명령", "공통 직선 잔차의 630점 평균 (px)", "시행별 절편 적용 후 평균 (px)", "그 시행 절편 평균 - 다른 시행 (px)"}};
    {
        const int n = (int)posk.size();
        Mat dp(n, P, CV_64F), ai(n, 3, CV_64F);
        for (int r = 0; r < n; ++r) {
            D["x"].row(posk[r]).copyTo(dp.row(r));
            ai.at<double>(r, 0) = gi[posk[r]] == 0; ai.at<double>(r, 1) = gi[posk[r]] != 0; ai.at<double>(r, 2) = x[posk[r]];
        }
        Mat ei = dp - ai * lstsq(ai, dp);
        auto& ic = pos["x"].icpt;
        for (size_t r : worst(cpx, 5)) {
            int k = posk[r];
            int g = gi[k] == 0 ? 0 : 1;
            outl.add({nm[k], groups[gi[k]], fmt(x[k], 2), fmts(cpx[r], 4), fmts(cv::mean(ei.row((int)r))[0], 4), fmts(ic[g] - ic[1 - g], 4)});
        }
    }
    std::string fz = "03_튀는값_" + nm[wk] + ".png";
    {
        cv::Mat im = s630::read_gray(shots[si[wk]].path);
        std::vector<cv::Mat> tiles;
        // 공통 직선 잔차(점별)를 화살표로
        const int n = (int)posk.size();
        Mat dp(n, P, CV_64F), ap(n, 2, CV_64F);
        for (int r = 0; r < n; ++r) { D["x"].row(posk[r]).copyTo(dp.row(r)); ap.at<double>(r, 0) = 1; ap.at<double>(r, 1) = x[posk[r]]; }
        Mat ep = dp - ap * lstsq(ap, dp);
        for (int q = 0; q < 3; ++q) {
            auto b = s630::kBoxes[q];
            auto sh = tr[wk].shift[q];
            cv::Rect roi(b[0] + sh.x - 10, b[1] + sh.y - 10, b[2] - b[0] + 20, b[3] - b[1] + 20);
            std::vector<Mark> mk;
            for (int j = 0; j < P; ++j)
                if (cs[j].qr == q + 1 && j % 4 == 0) mk.push_back({{tr[wk].obs[j].x, tr[wk].obs[j].y}, {ep.at<double>((int)wr, j), 0}, "", kRed});
            tiles.push_back(zoom(im, roi, 420, mk, 150, q == 0 ? nm[wk] + " mean resid " + fmts(cpx[wr], 3) + " px" : "",
                                 {{cv::Rect2d(b[0] + sh.x, b[1] + sh.y, b[2] - b[0], b[3] - b[1]), kGreen, "coarse match"}}));
        }
        save_png(fz, hstack(tiles));
    }

    // ---- 원래 결과와 대조 (eda_y_trials_20261022/model_comparison.csv, summary.json, trial_summary.csv) ----
    Parity par;
    const double om[6] = {0.2065041934535171, 0.052260583863562166, 0.05086242129232258, 0.04967396149562422, 0.0390331835185007, 0.03818841542484817};
    const char* mn[3] = {"pooled", "trial_intercept", "trial_line"};
    for (int c = 0; c < 2; ++c)
        for (int k = 0; k < 3; ++k) par.add(std::string(c ? "y " : "x ") + mn[k] + " 잔차 RMS (80장)", om[c * 3 + k], mrms[std::string(c ? "y" : "x") + mn[k]], 0.0005);
    par.add("x 양수 공통 직선 RMS", 0.0783403117117661, pos["x"].pooled, 0.0005);
    par.add("x 양수 시행별 절편 RMS", 0.04955629534507183, pos["x"].inter, 0.0005);
    par.add("x 양수 시행별 직선 RMS", 0.048637754557051965, pos["x"].line, 0.0005);
    par.add("x 양수 절편 허용 제곱합 감소", 0.5998460809479803, pos["x"].red_i, 0.005);
    par.add("x 양수 공통 기울기 평균", 14.373956328936437, pos["x"].slope_mean, 0.002);
    par.add("x 양수 절편 평균 (0.05 홀수 +)", -0.43476987485110696, pos["x"].icpt[0], 0.002);
    par.add("x 양수 절편 평균 (0.1 간격 +)", -0.5549836623683505, pos["x"].icpt[1], 0.002);
    par.add("y 양수 공통 직선 RMS", 0.052938025554710076, pos["y"].pooled, 0.0005);
    par.add("y 양수 시행별 절편 RMS", 0.03792942669059959, pos["y"].inter, 0.0005);
    const double ot[8][2] = {{14.388144620591813, -0.4489581665064848}, {14.363145300340335, -0.4283679823120583}, {14.35976803728107, -0.5400859561302183},
                             {14.392924933752584, 0.11007714368573739}, {0.03343298661974607, 0.07778002805753352}, {0.029503636724724586, 0.08033605347088371},
                             {0.03955645139801511, -0.0009868370283680876}, {0.026085505904061855, 0.03524062389406372}};
    for (int c = 0; c < 2; ++c)
        for (int g = 0; g < 4; ++g) {
            auto& v = ticpt[std::string(c ? "y" : "x") + groups[g]];
            par.add(std::string(c ? "y " : "x ") + groups[g] + " 기울기 평균", ot[c * 4 + g][0], v[0], 0.002);
            par.add(std::string(c ? "y " : "x ") + groups[g] + " 절편 평균 (px)", ot[c * 4 + g][1], v[1], 0.002);
        }
    par.add("진단 실패 수 (80장 x 630점)", 1965, invalid, 0, 0);

    std::ostringstream R;
    R << "# 결과: Y 지그재그, 시행별 절편\n\n"
      << "입력: `data/clean/clean_20261014` Y 80장 + center.bmp. 고정 구조 모서리 630점을 사진마다 흐린 경계 모델로 맞추고 center 좌표를 뺀 변위(Δx, Δy)를 쓴다. 진단 실패 점도 포함.\n"
      << "시행: 파일명 Y 단계가 0.05의 홀수배면 '0.05 홀수', 짝수배면 '0.1 간격', 부호로 ±. 원래 hw_eda shots.csv 의 group 과 같은 규칙이다.\n"
      << "모델(점마다): 공통 직선 [1, Y], 시행별 절편 [시행 4개, Y], 시행별 직선 [시행 4개, 시행 x Y].\n\n"
      << "## 80장 전체 모델 비교\n\n" << modelT.md()
      << "\n## 양수 구간 (시행 2개)\n\n| 성분 | 공통 직선 RMS | 시행별 절편 RMS | 시행별 직선 RMS | 절편 허용 제곱합 감소 | 직선 허용 제곱합 감소 | 공통 기울기 평균 | 절편 평균 (0.05 홀수 + / 0.1 간격 +) |\n|---|---|---|---|---|---|---|---|\n";
    for (auto c : {"x", "y"})
        R << "| " << c << " | " << fmt(pos[c].pooled, 4) << " | " << fmt(pos[c].inter, 4) << " | " << fmt(pos[c].line, 4) << " | " << fmt(pos[c].red_i, 3) << " | "
          << fmt(pos[c].red_l, 3) << " | " << fmt(pos[c].slope_mean, 4) << " | " << fmt(pos[c].icpt[0], 4) << " / " << fmt(pos[c].icpt[1], 4) << " |\n";
    R << "\n## 시행별 직선\n\n" << trialT.md()
      << "\n공통 잔차 = 한 사진에서 630점 잔차의 평균, 공간 잔차 = 각 점에서 그 사진의 공통 잔차를 뺀 값.\n\n"
      << "진단 실패: " << invalid << " / " << N * P << "\n\n"
      << "## 원래 결과와 대조 (" << par.ok << "/" << par.total << " 일치)\n\n" << par.t.md()
      << "\n## 원래와 다른 점\n\n- 원래 그림 02의 셋째 줄(같은 잔차를 촬영 시각 순으로 연결)은 `data/clean`에 촬영 시각이 없어 그리지 않았다.\n"
      << "- 630점 좌표를 옛 corners_280.csv 대신 사진에서 다시 쟀다. 진단 실패 수는 원래 scipy 최적화가 85회 평가 안에 끝나지 않은 경우가 대부분이라 C++ LM 과 다르다.\n"
      << "\n## 파일\n\n- `이미지/`: 그림\n";
    write_text(out("결과.md"), R.str());

    auto& px_ = pos["x"];
    std::ostringstream Pp;
    Pp << "# 보고서: Y 지그재그, 시행별 절편 (10-22 15:48)\n\n"
       << "## 문제\n\nY 단일축 자료에 보이는 지그재그 잔차가 무엇 때문인지 밝혀야 했다.\n\n"
       << "## 해결 방법\n\nY 80장 630점 변위에 공통 직선·시행별 절편·시행별 직선 세 모델을 맞춰 잔차를 비교했다.\n\n"
       << "## 결과\n\n양수 구간에서 시행별 절편을 허용하면 수평 잔차 RMS가 " << fmt(px_.pooled, 4) << "→" << fmt(px_.inter, 4) << " px로 줄어(제곱합 " << fmt(100 * px_.red_i, 0)
       << "% 감소) 지그재그 상당 부분이 시행 간 차이였다.\n\n"
       << "| 지표 | 값 |\n|---|---|\n"
       << "| 양수 x: 공통 직선 / 시행별 절편 / 시행별 직선 RMS (px) | " << fmt(px_.pooled, 4) << " / " << fmt(px_.inter, 4) << " / " << fmt(px_.line, 4) << " |\n"
       << "| 양수 x: 두 시행 절편 평균 차이 (px) | " << fmt(std::abs(px_.icpt[0] - px_.icpt[1]), 4) << " |\n"
       << "| 80장 x: 공통 직선 / 시행별 절편 / 시행별 직선 RMS (px) | " << fmt(mrms["xpooled"], 4) << " / " << fmt(mrms["xtrial_intercept"], 4) << " / " << fmt(mrms["xtrial_line"], 4) << " |\n"
       << "| 양수 y: 공통 직선 / 시행별 절편 RMS (px) | " << fmt(pos["y"].pooled, 4) << " / " << fmt(pos["y"].inter, 4) << " |\n\n"
       << img_md("01_양수구간비교_x.png", "양수 구간 x") << "\n\n" << img_md("02_시행4개_x.png", "시행 4개 x") << "\n\n" << img_md("01_양수구간비교_y.png", "양수 구간 y") << "\n\n"
       << img_md("02_시행4개_y.png", "시행 4개 y") << "\n\n"
       << "## 튀는 값\n\nY·T 추정이 없는 실험이라 양수 공통 직선에서 630점 평균 잔차가 가장 큰 사진을 보였고, " << nm[wk] << "(" << groups[gi[wk]] << ")은 평균 "
       << fmts(cpx[wr], 3) << " px 벗어나지만 시행별 절편을 허용하면 그 차이 대부분이 사라진다.\n\n"
       << outl.md() << "\n" << img_md(fz, "튀는 값")
       << "\n\n빨간 십자는 이 사진에서 맞춘 모서리(4개 중 1개), 화살표는 공통 직선 수평 잔차를 150배 키운 것이다. 화살표가 한 방향으로 모이면 사진 전체가 함께 어긋난 것이다.\n\n"
       << "## 한줄평\n\n두 양수 시행의 점별 절편이 평균 " << fmt(std::abs(px_.icpt[0] - px_.icpt[1]), 3) << " px 달라 명령값 순으로 섞어 연결하면 0.05 간격으로 번갈아 튀는 지그재그가 되고, 절편만 시행별로 두면 잔차 제곱합이 "
       << fmt(100 * px_.red_i, 0) << "% 줄어 남는 것은 시행 안 잔차 " << fmt(px_.inter, 3) << " px다.\n\n"
       << "## 원래 결과와 대조\n\n원래 값은 `eda_y_trials_20261022/model_comparison.csv`, `summary.json`, `trial_summary.csv`에서 가져왔다. " << par.ok << "/" << par.total
       << " 일치. 진단 실패 수는 최적화 수렴 판정(원래 scipy TRF 85회 제한의 'optimizer' 실패)이 달라 불일치로 남긴다.\n\n" << par.t.md();
    write_text(out("보고서.md"), Pp.str());
    std::cout << "parity " << par.ok << "/" << par.total << "\n";
    return 0;
}
