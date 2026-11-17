// QR205 점 이동에 공분산 가중(WLS / Ledoit-Wolf GLS)을 넣으면 기존 205점·6분기 모델보다 나은지 본다.
// 원본: codex_experiments_20261014/gls_comparison_20261014/compare.py, direct410/run_direct410.py,
//       baseline_snapshot/zero_experiment.py, measurement/spread.py·refine.py·measure.py (GN17 spread14 측정)
#include <hye/dataset.hpp>
#include <hye/plot.hpp>
#include <hye/report.hpp>
#include <omp.h>
#include <iostream>
#include "kfold.hpp"
#include "track.hpp"
#include "zero.hpp"

using namespace hye;

namespace {

struct Method {
    std::string name, label, spatial, metric;  // direct410_* 는 spatial = "direct"
};
const std::vector<Method> kMethods{
    {"baseline", "spatial OLS + legacy cov", "ols", "legacy"},
    {"ols_euclidean", "spatial OLS + identity", "ols", "identity"},
    {"ols_wls", "spatial OLS + diagonal WLS", "ols", "diagonal"},
    {"ols_gls_lw", "spatial OLS + LW cov", "ols", "ledoit_wolf"},
    {"spatial_wls", "spatial WLS + legacy cov", "diagonal", "legacy"},
    {"spatial_gls_lw", "spatial GLS(LW) + legacy cov", "ledoit_wolf", "legacy"},
    {"spatial_gls_lw_metric_lw", "spatial GLS(LW) + LW cov", "ledoit_wolf", "ledoit_wolf"},
    {"direct410_identity", "410 direct + identity", "direct", "identity"},
    {"direct410_diagonal", "410 direct + diagonal", "direct", "diagonal"},
    {"direct410_ledoit_wolf", "410 direct + LW", "direct", "ledoit_wolf"},
};
const char* kKor[] = {"기존: 공간 OLS + 기존 공분산", "공간 OLS + 동일 가중", "공간 OLS + 대각 WLS", "공간 OLS + 공분산 LW", "공간 WLS + 기존 공분산",
                      "공간 GLS(LW) + 기존 공분산", "공간 GLS(LW) + 공분산 LW", "410변위 직접 + 동일 가중", "410변위 직접 + 대각 WLS", "410변위 직접 + GLS(LW)"};

// run_direct410.make_metric: 410차원 거리
cv::Mat direct_metric(const cv::Mat& R, int rank, const std::string& kind) {
    const int p = R.cols, dof = std::max(R.rows - rank, 1);
    std::vector<double> sd(p), rms(p);
    for (int j = 0; j < p; ++j) {
        double s = 0;
        for (int i = 0; i < R.rows; ++i) s += R.at<double>(i, j) * R.at<double>(i, j);
        sd[j] = std::sqrt(std::max(s / dof, 1e-14));
        rms[j] = std::sqrt(std::max(s / R.rows, 1e-14));
    }
    if (kind == "identity") return cv::Mat::eye(p, p, CV_64F);
    cv::Mat W = cv::Mat::zeros(p, p, CV_64F);
    if (kind == "diagonal") {
        for (int j = 0; j < p; ++j) W.at<double>(j, j) = 1 / sd[j];
        return W;
    }
    cv::Mat Z(R.size(), CV_64F);
    for (int i = 0; i < R.rows; ++i)
        for (int j = 0; j < p; ++j) Z.at<double>(i, j) = R.at<double>(i, j) / rms[j];
    cv::Mat L = zx::cholesky_lower(zx::ledoit_wolf(Z));
    cv::Mat Li;
    cv::solve(L, cv::Mat::eye(p, p, CV_64F), Li, cv::DECOMP_LU);  // solve_triangular(L, I)
    for (int i = 0; i < p; ++i)
        for (int j = 0; j < p; ++j) Li.at<double>(i, j) /= sd[j];
    return Li;
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
    // 원래 npz 순서: center, Theta 파일명 정렬, Y 파일명 정렬
    std::vector<size_t> order;
    for (size_t i = 0; i < shots.size(); ++i) if (shots[i].condition == "center") order.push_back(i);
    for (const char* c : {"T", "Y"}) {
        std::vector<size_t> v;
        for (size_t i = 0; i < shots.size(); ++i) if (shots[i].condition == c) v.push_back(i);
        std::sort(v.begin(), v.end(), [&](size_t a, size_t b) { return shots[a].rel < shots[b].rel; });
        order.insert(order.end(), v.begin(), v.end());
    }
    const int N = (int)order.size();
    std::vector<zx::Row> m(N);
    for (int k = 0; k < N; ++k) {
        auto& s = shots[order[k]];
        std::string f = s.rel.substr(s.rel.rfind('/') + 1);
        m[k].file = f;
        if (s.condition == "center") { m[k].axis = 'C'; m[k].value = 0; m[k].session = "C"; continue; }
        m[k].axis = f[0];
        m[k].value = std::stod(f.substr(1, f.size() - 5));
        std::string ss = m[k].axis == 'T' ? "T" : ((std::llabs((long long)std::nearbyint(m[k].value / .05)) % 2) ? "Y2" : "Y1");
        m[k].session = ss + (m[k].value > 0 ? "+" : "-");
    }

    // ---- GN17 spread14 측정 (QR 205점) ----
    const cv::Mat ref = trk::crop(read_gray(shots[order[0]]));
    trk::Features f14 = trk::features_spread(ref, 14);
    trk::Spread sp(ref, f14);
    const int P = (int)f14.pts.size();
    std::vector<std::vector<cv::Point2f>> raw(N);
    std::vector<char> allvalid(N, 1);
#pragma omp parallel for schedule(dynamic)
    for (int k = 0; k < N; ++k) {
        auto o = sp.extract(trk::crop(read_gray(shots[order[k]])));
        raw[k] = o.D;
        for (char v : o.valid) allvalid[k] &= v;
    }
    int nvalid = (int)std::count(allvalid.begin(), allvalid.end(), 1);
    std::cout << "measured " << N << " all-valid " << nvalid << "\n";
    // D: n x 410 (점 순서 x,y), center 를 뺀다 (float64)
    cv::Mat D(N, 2 * P, CV_64F);
    for (int k = 0; k < N; ++k)
        for (int j = 0; j < P; ++j) {
            D.at<double>(k, 2 * j) = (double)raw[k][j].x - (double)raw[0][j].x;
            D.at<double>(k, 2 * j + 1) = (double)raw[k][j].y - (double)raw[0][j].y;
        }
    auto g = f14.global();
    cv::Mat B(P, 6, CV_64F), H = cv::Mat::zeros(2 * P, 12, CV_64F);
    for (int j = 0; j < P; ++j) {
        double x = (g[j].x - 1906.0) / 486, y = (g[j].y - 1070.0) / 122;
        double v[6] = {1, x, y, x * x, x * y, y * y};
        for (int c = 0; c < 6; ++c) {
            B.at<double>(j, c) = v[c];
            H.at<double>(2 * j, 2 * c) = v[c];
            H.at<double>(2 * j + 1, 2 * c + 1) = v[c];
        }
    }
    cv::Mat Hp;
    cv::invert(H, Hp, cv::DECOMP_SVD);
    cv::Mat X_ols = D * Hp.t();  // n x 12 = [c0x, c0y, c1x, ...]

    // ---- 평가 묶음 ----
    struct Proto { std::string name; std::vector<zx::Fold> folds; };
    std::vector<Proto> protos;
    auto ids = zx::noncenter(m);
    protos.push_back({"full280", {{"all", ids, ids}}});
    protos.push_back({"stratified5", zx::stratified(m, 5, 20261014)});
    protos.push_back({"blocked_within_session5", zx::blocked5(m)});
    for (int s = 0; s < 20; ++s) protos.push_back({"repeated5_seed" + std::to_string(s), zx::stratified(m, 5, 7000 + s)});
    protos.push_back({"session_out_missing_origin", zx::session_out(m)});

    const int M = (int)kMethods.size();
    // preds[proto][method] = 280 예측
    std::vector<std::vector<std::vector<zx::Pred>>> preds(protos.size(), std::vector<std::vector<zx::Pred>>(M));
    std::vector<double> lw_alpha_spatial;  // full280 의 spatial LW 축소 계수
    // 평가는 한 스레드로 돈다 (결과가 실행마다 같게)
    for (int pi = 0; pi < (int)protos.size(); ++pi) {
        for (auto& f : protos[pi].folds) {
            cv::Mat Rraw;
            int rank_raw;
            zx::Model rawfit = zx::fit_curves(m, D, f.tr, &Rraw, &rank_raw);
            for (int mi = 0; mi < M; ++mi) {
                auto& me = kMethods[mi];
                std::vector<zx::Pred> pr;
                if (me.spatial == "direct") {
                    zx::Model md = rawfit;
                    md.W = direct_metric(Rraw, rank_raw, me.metric);
                    pr = zx::predict(m, D, f.te, md);
                } else {
                    cv::Mat X = X_ols;
                    if (me.spatial != "ols") {
                        double a;
                        cv::Mat Sigma = zx::covariance(Rraw, rank_raw, me.spatial, &a);
                        if (pi == 0 && me.name == "spatial_gls_lw") {
                            lw_alpha_spatial.push_back(a);
                        }
                        cv::Mat PH, proj;
                        cv::solve(Sigma, H, PH, cv::DECOMP_LU);
                        cv::Mat HtPH = H.t() * PH;
                        cv::solve(HtPH, PH.t(), proj, cv::DECOMP_LU);
                        X = cv::Mat(D * proj.t());  // X_ols 버퍼에 덮어쓰지 않게 새 행렬
                    }
                    cv::Mat R;
                    int rank;
                    zx::Model md = zx::fit_curves(m, X, f.tr, &R, &rank);
                    md.W = me.metric == "legacy" ? zx::legacy_W(R, rank) : zx::precision_root(zx::covariance(R, rank, me.metric));
                    pr = zx::predict(m, X, f.te, md);
                }
                preds[pi][mi].insert(preds[pi][mi].end(), pr.begin(), pr.end());
            }
        }
    }
    std::cout << "evaluated" << std::endl;

    // ---- 집계 (comparison_summary 와 같은 정의) ----
    auto pidx = [&](const std::string& n) {
        for (size_t i = 0; i < protos.size(); ++i) if (protos[i].name == n) return (int)i;
        return -1;
    };
    struct Sum {
        zx::Score full;
        int cv_failed = 0, perfect = 0, fmin = 1 << 30, fmax = 0, Yc = 0, Tc = 0, fixed = 0, newfail = 0, uniq = 0;
        double Ymax = 0, Tmax = 0;
        std::map<std::string, int> fail_count;
        zx::Score strat, blocked, sout;
    };
    std::vector<Sum> sum(M);
    for (int mi = 0; mi < M; ++mi) {
        auto& s = sum[mi];
        s.full = zx::score(preds[0][mi]);
        s.strat = zx::score(preds[pidx("stratified5")][mi]);
        s.blocked = zx::score(preds[pidx("blocked_within_session5")][mi]);
        s.sout = zx::score(preds[pidx("session_out_missing_origin")][mi]);
        for (int sd = 0; sd < 20; ++sd) {
            int pi = pidx("repeated5_seed" + std::to_string(sd));
            auto sc = zx::score(preds[pi][mi]);
            s.cv_failed += sc.failed;
            s.perfect += sc.failed == 0;
            s.fmin = std::min(s.fmin, sc.failed);
            s.fmax = std::max(s.fmax, sc.failed);
            s.Yc += sc.Yc;
            s.Tc += sc.Tc;
            s.Ymax = std::max(s.Ymax, sc.Ymax);
            s.Tmax = std::max(s.Tmax, sc.Tmax);
            std::map<int, bool> base_pass;
            for (auto& r : preds[pi][0]) base_pass[r.i] = r.pass();
            for (auto& r : preds[pi][mi]) {
                if (!r.pass()) s.fail_count[m[r.i].file]++;
                s.fixed += !base_pass[r.i] && r.pass();
                s.newfail += base_pass[r.i] && !r.pass();
            }
        }
        s.uniq = (int)s.fail_count.size();
    }

    // ---- 원래 결과 (gls_comparison_20261014/comparison_summary.csv, metrics.csv, direct410/metrics.csv) ----
    struct Orig { int full_failed; double full_Y_max; int cv_failed; int strat_failed, blocked_failed, sout_failed; };
    const Orig orig[] = {
        {0, 0.0362413363547481, 13, 0, 0, 222},  {1, 0.2663473848516616, 74, 3, 2, 212}, {1, 0.2663417974816455, 34, 2, 1, 219},
        {0, 0.0361277832701029, 14, 0, 0, 222},  {0, 0.0086560535336277, 16, 1, 0, 222}, {0, 0.0028805248948673, 66, 2, 3, 210},
        {0, 0.0029053892401209, 63, 2, 4, 211},  {0, 0.0355452513773019, 92, -1, -1, 255}, {9, 1.3339318149601171, 215, -1, -1, 276},
        {0, 0.0030456066044592, 108, -1, -1, 215}};
    Parity par;
    par.add("GN17 spread14 점 수", 205, P, 0, 0);
    par.add("205점 모두 유효한 사진 수", 281, nvalid, 0, 0);
    for (int mi = 0; mi < M; ++mi) {
        auto& o = orig[mi];
        auto& s = sum[mi];
        par.add(kMethods[mi].name + " full280 Y 최대 벡터 오차", o.full_Y_max, s.full.Ymax, 1e-4, 6);
        par.add(kMethods[mi].name + " full280 실패", o.full_failed, s.full.failed, 0, 0);
        par.add(kMethods[mi].name + " 반복 5분할 20회 실패 (/5600)", o.cv_failed, s.cv_failed, 0, 0);
        if (o.strat_failed >= 0) {
            par.add(kMethods[mi].name + " 층화 5분할 실패", o.strat_failed, s.strat.failed, 0, 0);
            par.add(kMethods[mi].name + " 시행 내 블록 5분할 실패", o.blocked_failed, s.blocked.failed, 0, 0);
        }
        par.add(kMethods[mi].name + " 시행 제외 실패", o.sout_failed, mi < 7 ? s.sout.failed : s.sout.n - s.sout.correct, 0, 0);
    }

    // ---- 표 ----
    Table summ{{"방법", "설명", "full280 실패", "full280 Y 최대 (mm)", "full280 T 최대 (deg)", "반복 실패 / 5600", "실패 0 회차 / 20", "회차별 실패 최소~최대",
                "Y 정답 / 1600", "T 정답 / 4000", "반복 Y 최대", "기존 대비 고침", "기존 대비 새 실패", "실패한 사진 수"}};
    for (int mi = 0; mi < M; ++mi) {
        auto& s = sum[mi];
        summ.add({kMethods[mi].name, kKor[mi], std::to_string(s.full.failed), fmt(s.full.Ymax, 7), fmt(s.full.Tmax, 7), std::to_string(s.cv_failed),
                  std::to_string(s.perfect), std::to_string(s.fmin) + "~" + std::to_string(s.fmax), std::to_string(s.Yc), std::to_string(s.Tc), fmt(s.Ymax, 4),
                  std::to_string(s.fixed), std::to_string(s.newfail), std::to_string(s.uniq)});
    }
    summ.csv(out("comparison_summary.csv"));
    Table other{{"방법", "층화 5분할 실패", "시행 내 블록 5분할 실패", "시행 제외 축·부호 오류 / 280"}};
    for (int mi = 0; mi < M; ++mi)
        other.add({kMethods[mi].name, mi < 7 ? std::to_string(sum[mi].strat.failed) : "-", mi < 7 ? std::to_string(sum[mi].blocked.failed) : "-",
                   std::to_string(sum[mi].sout.n - sum[mi].sout.correct)});
    // 사진별 실패 빈도 (반복 20회)
    Table freq{{"방법", "사진", "실패 회차 / 20"}};
    for (int mi = 0; mi < M; ++mi) {
        std::vector<std::pair<std::string, int>> v(sum[mi].fail_count.begin(), sum[mi].fail_count.end());
        std::stable_sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.second > b.second; });
        for (auto& x : v) freq.add({kMethods[mi].name, x.first, std::to_string(x.second)});
    }
    freq.csv(out("failed_image_frequency.csv"));

    // ---- 그림 ----
    {
        std::vector<std::string> cats;
        std::vector<double> v;
        for (int mi = 0; mi < M; ++mi) { cats.push_back(kMethods[mi].name); v.push_back(sum[mi].cv_failed); }
        bar_chart("01_방법별_반복5분할_실패.png", "Failures in 20 x 5-fold CV (5600 predictions)", cats, v, "failed", 0, sum[0].cv_failed);
    }
    {
        Chart c("Resubstitution Y max error vs repeated-CV failures", "log10 full280 Y max vector error (mm)", "repeated CV failures / 5600");
        for (int mi = 0; mi < M; ++mi) {
            c.scatter({std::log10(sum[mi].full.Ymax)}, {(double)sum[mi].cv_failed}, mi == 0 ? kRed : color(mi), "", 6);
            c.note(std::log10(sum[mi].full.Ymax), sum[mi].cv_failed, kMethods[mi].name);
        }
        c.save("02_재대입오차_대_반복실패.png");
    }
    // 튀는 값: spatial_gls_lw 가 반복 20회에서 가장 자주 틀린 Y 사진과 T 사진
    const int GI = 5;
    auto most = [&](char a) {
        std::string best;
        int bc = -1;
        for (auto& kv : sum[GI].fail_count)
            if (kv.first[0] == a && kv.second > bc) { bc = kv.second; best = kv.first; }
        return std::make_pair(best, bc);
    };
    auto [fy, cy] = most('Y');
    auto [ft, ct] = most('T');
    // 그 사진의 대표 예측 (가장 앞 회차에서 틀린 것) 과 기존 모델의 같은 회차 예측
    auto find_pred = [&](const std::string& file, int mi, int* seed_out) {
        for (int sd = 0; sd < 20; ++sd) {
            int pi = pidx("repeated5_seed" + std::to_string(sd));
            for (auto& r : preds[pi][GI])
                if (m[r.i].file == file && !r.pass()) {
                    *seed_out = sd;
                    for (auto& q : preds[pi][mi]) if (q.i == r.i) return q;
                }
        }
        return zx::Pred{};
    };
    Table outl{{"사진", "정답 Y (mm)", "정답 T (deg)", "spatial_gls_lw 실패 회차 / 20", "기존 실패 회차 / 20", "첫 실패 seed", "GLS 추정 (축·값·분기)", "기존 추정 (축·값·분기)",
                "GLS 비용", "기존 비용"}};
    struct OutInfo { std::string file; int seed; zx::Pred g, b; };
    std::vector<OutInfo> oinfo;
    for (auto& fc : std::vector<std::pair<std::string, int>>{{fy, cy}, {ft, ct}}) {
        if (fc.second <= 0) continue;
        int sd = 0;
        auto gp = find_pred(fc.first, GI, &sd);
        auto bp = find_pred(fc.first, 0, &sd);
        oinfo.push_back({fc.first, sd, gp, bp});
        auto bcount = sum[0].fail_count.count(fc.first) ? sum[0].fail_count.at(fc.first) : 0;
        outl.add({fc.first, fmt(gp.axis == 'Y' ? gp.value : 0, 2), fmt(gp.axis == 'T' ? gp.value : 0, 2), std::to_string(fc.second), std::to_string(bcount),
                  std::to_string(7000 + sd), std::string(1, gp.pred_axis) + " " + fmts(gp.pred, 3) + " " + gp.branch,
                  std::string(1, bp.pred_axis) + " " + fmts(bp.pred, 3) + " " + bp.branch, fmt(gp.cost, 1), fmt(bp.cost, 1)});
    }
    // 사진 확대: 초록 = 기준 QR, 빨강 = 205점 평균 이동만큼 옮긴 QR, 점 = QR 평균을 뺀 점별 잔차 x300
    auto photo = [&](const std::string& file, const std::string& png, const std::string& cap) {
        int k = 0;
        for (int i = 0; i < N; ++i) if (m[i].file == file) k = i;
        cv::Mat im = read_color(shots[order[k]]);
        std::vector<cv::Mat> tiles;
        for (int q = 0; q < 3; ++q) {
            auto b = trk::BOXES[q];
            double mx = 0, my = 0;
            int cnt = 0;
            for (int j = 0; j < P; ++j)
                if (f14.region[j] == q) { mx += D.at<double>(k, 2 * j); my += D.at<double>(k, 2 * j + 1); ++cnt; }
            mx /= cnt; my /= cnt;
            cv::Rect2d rb(b[0], b[1], b[2] - b[0], b[3] - b[1]), nb(b[0] + mx, b[1] + my, b[2] - b[0], b[3] - b[1]);
            std::vector<Mark> mk;
            for (int j = 0; j < P; ++j)
                if (f14.region[j] == q) mk.push_back({{g[j].x + mx, g[j].y + my}, {D.at<double>(k, 2 * j) - mx, D.at<double>(k, 2 * j + 1) - my}, "", kRed});
            cv::Rect roi = cv::Rect(rb | nb) + cv::Size(40, 40) - cv::Point(20, 20);
            tiles.push_back(zoom(im, roi, 420, mk, 300, q == 0 ? cap : "", {{rb, kGreen, ""}, {nb, kRed, "q" + std::to_string(q) + " dx " + fmt(mx, 2) + " dy " + fmt(my, 3)}}));
        }
        save_png(png, hstack(tiles));
    };
    for (auto& o : oinfo) {
        bool isY = o.file[0] == 'Y';
        photo(o.file, std::string(isY ? "03_튀는값_Y_" : "04_튀는값_T_") + o.file.substr(0, o.file.size() - 4) + ".png",
              o.file.substr(0, o.file.size() - 4) + " GLS -> " + std::string(1, o.g.pred_axis) + " " + fmts(o.g.pred, 3));
    }

    // ---- 결과.md ----
    std::ostringstream R;
    R << "# 결과: 공분산 가중(GLS/WLS/LW) 비교\n\n"
      << "입력: `data/clean/clean_20261014` 281장. center.bmp 기준 GN17 spread14(QR 205점, QR별 잘라 LK15 후 3차 스플라인 GN17)로 점 이동 410차원을 사진에서 다시 쟀다.\n"
      << "공통: 축마다 [v, v^2, 학습 시행 0점] 회귀, 부호 제약 6분기 중 비용 최소를 고른다. 공간 투영(P12)과 거리 행렬만 바꾼다. 모든 공분산은 학습 분할 잔차로만 추정한다.\n"
      << "평가: full280(재대입), 층화 5분할(seed 20261014), 시행 내 블록 5분할, 반복 5분할 20회(seed 7000~7019), 시행 제외 6분할.\n\n"
      << "## 방법별 요약\n\n" << summ.md() << "\n## 다른 평가\n\n" << other.md()
      << "\n## 원래 결과와 대조 (" << par.ok << "/" << par.total << " 일치)\n\n" << par.t.md()
      << "\n## 옮기지 않은 부분\n\n- `interleaved` 2분할: 촬영 시각(mtime) 순서로 나누는데 `data/clean`에는 촬영 시각이 없어 뺐다.\n"
      << "- 보고서 HTML·패키징·SHA 감사 스크립트(build_report.py, package_results.py, verify_raw_inference.py), 모델 json 저장.\n\n"
      << "## 파일\n\n- `comparison_summary.csv`: 10가지 방법 집계\n- `failed_image_frequency.csv`: 반복 20회 중 사진별 실패 회차\n- `감사.txt`: 원래 실험의 감사 기록(그대로 둠)\n";
    write_text(out("결과.md"), R.str());

    // ---- 보고서.md ----
    auto& b = sum[0];
    auto& gl = sum[GI];
    std::ostringstream Pp;
    Pp << "# 보고서: 공분산 가중 효과 GLS·WLS 비교 (10-14 16:21)\n\n"
       << "## 문제\n\n공간 적합과 판정 거리에 공분산(GLS/WLS/LW)을 넣으면 기존 205점·6분기 모델보다 나아지는지 확인해야 했다.\n\n"
       << "## 해결 방법\n\n같은 QR205 이동을 OLS·WLS·Ledoit-Wolf·공간 GLS·410차원 직접 거리 등 10가지로 바꿔 full280, 반복 5분할 20회, 시행 제외로 비교했다.\n\n"
       << "## 결과\n\n기존은 반복 5600건 중 " << b.cv_failed << "건 실패였고, spatial_gls_lw는 재대입 Y 최대 오차를 " << fmt(b.full.Ymax, 7) << " -> " << fmt(gl.full.Ymax, 7)
       << " mm로 줄였지만 반복 실패가 " << gl.cv_failed << "건으로 늘었다.\n\n"
       << "| 방법 | full280 실패 | full280 Y 최대 (mm) | 반복 실패 / 5600 | Y 정답 / 1600 | T 정답 / 4000 | 실패 0 회차 / 20 |\n|---|---|---|---|---|---|---|\n";
    for (int mi = 0; mi < M; ++mi)
        Pp << "| " << kMethods[mi].name << " | " << sum[mi].full.failed << " | " << fmt(sum[mi].full.Ymax, 7) << " | " << sum[mi].cv_failed << " | " << sum[mi].Yc << " | "
           << sum[mi].Tc << " | " << sum[mi].perfect << " |\n";
    Pp << "\n" << img_md("01_방법별_반복5분할_실패.png", "방법별 반복 실패") << "\n\n" << img_md("02_재대입오차_대_반복실패.png", "재대입 오차 대 반복 실패") << "\n\n";
    Pp << "## 튀는 값\n\nspatial_gls_lw가 반복 20회에서 가장 자주 틀린 사진은 Y는 " << fy << "(" << cy << "회), T는 " << ft << "(" << ct
       << "회)이고, 학습 잔차로 만든 410x410 공분산이 분할마다 달라져 작은 이동 사진의 분기 비용 순서가 바뀌었다.\n\n"
       << outl.md() << "\n";
    for (auto& o : oinfo)
        Pp << img_md(std::string(o.file[0] == 'Y' ? "03_튀는값_Y_" : "04_튀는값_T_") + o.file.substr(0, o.file.size() - 4) + ".png", o.file) << "\n\n";
    Pp << "초록 상자는 기준 사진의 QR, 빨간 상자는 205점 평균 이동만큼 옮긴 QR이다. 빨간 십자는 측정 점, 화살표는 QR 평균을 뺀 점별 잔차 x300이다.\n\n";
    double alpha = lw_alpha_spatial.empty() ? kNaN : lw_alpha_spatial[0];
    Pp << "## 한줄평\n\n공간 GLS(LW)는 224~280장의 잔차로 410차원 공분산을 추정해(축소 계수 " << fmt(alpha, 3)
       << ") 학습 사진의 잔차 구조에 투영을 맞추므로 재대입 Y 최대 오차는 " << fmt(b.full.Ymax / gl.full.Ymax, 1) << "배 줄었지만, 그 구조가 분할 밖 사진에는 맞지 않아 반복 실패가 "
       << b.cv_failed << " -> " << gl.cv_failed << "건으로 늘었다.\n\n"
       << "## 원래 결과와 대조\n\n원래 값은 `gls_comparison_20261014/comparison_summary.csv`, `metrics.csv`, `direct410/metrics.csv`에서 가져왔다. 410차원 직접 방법의 시행 제외는 축·부호 오류 수(joint_errors)로 비교했다. "
       << par.ok << "/" << par.total << " 일치.\n\n" << par.t.md();
    write_text(out("보고서.md"), Pp.str());
    std::cout << "parity " << par.ok << "/" << par.total << "\n";
    return 0;
}
