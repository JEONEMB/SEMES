// 630점 목록: 341장 x 630점 측정의 점별 유효율·품질·Y/T 기울기·YT 코사인·반복 SD (학습·성능 평가 없음)
// 원본: 00000.legacies/eda_630_inventory_20261022/audit.py
// 측정은 원래처럼 joint_center_630_20261022/measurement.py 의 컴파일 LM 변형(results/measurements.npz 를 만든 방법)을 쓴다.
#include <hye/dataset.hpp>
#include <hye/plot.hpp>
#include <hye/report.hpp>
#include <omp.h>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include "pipeline630.hpp"

using namespace hye;
using pipe630::Img;

namespace {
double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}
std::string patch_str(const c630::Corner& c) {
    std::string s;
    for (int k = 0; k < 4; ++k) s += c.patch[k] > 0.5 ? '1' : '0';
    return s;
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
    omp_set_num_threads(4);
    cv::setNumThreads(1);
    begin_results();
    std::vector<cv::Point2d> ref;
    auto imgs = pipe630::load_and_measure(ref, true, "HYE_DEV_CACHE_LM");
    const int N = 341, P = 630;

    // ---- 점별 Y/T 기울기: 단일 축 사진에서 [1, 명령값] 최소제곱 ----
    auto slope = [&](bool isY) {
        std::vector<int> ix;
        for (int i = 0; i < N; ++i) {
            double a = isY ? imgs[i].shot.y : imgs[i].shot.t, b = isY ? imgs[i].shot.t : imgs[i].shot.y;
            if (imgs[i].scope == "single" && a != 0 && b == 0) ix.push_back(i);
        }
        double mx = 0;
        for (int i : ix) mx += isY ? imgs[i].shot.y : imgs[i].shot.t;
        mx /= ix.size();
        double sxx = 0;
        for (int i : ix) { double d = (isY ? imgs[i].shot.y : imgs[i].shot.t) - mx; sxx += d * d; }
        std::vector<cv::Point2d> s(P);
        for (int k = 0; k < P; ++k) {
            double sx = 0, sy = 0;
            for (int i : ix) {
                double d = (isY ? imgs[i].shot.y : imgs[i].shot.t) - mx;
                sx += d * imgs[i].D[2 * k];
                sy += d * imgs[i].D[2 * k + 1];
            }
            s[k] = {sx / sxx, sy / sxx};
        }
        return std::make_pair(s, (int)ix.size());
    };
    auto [sY, nY] = slope(true);
    auto [sT, nT] = slope(false);
    // ---- 같은 명령 반복(20261020 혼합 19조건) SD (ddof 1) 의 RMS ----
    std::map<std::string, std::vector<int>> cond;
    for (int i = 0; i < N; ++i) if (imgs[i].scope == "mixed") cond[imgs[i].group].push_back(i);
    std::vector<cv::Point2d> rep(P, {0, 0});
    for (auto& [g, ix] : cond)
        for (int k = 0; k < P; ++k)
            for (int c = 0; c < 2; ++c) {
                double m = 0, s = 0;
                for (int i : ix) m += imgs[i].D[2 * k + c];
                m /= ix.size();
                for (int i : ix) s += (imgs[i].D[2 * k + c] - m) * (imgs[i].D[2 * k + c] - m);
                (c ? rep[k].y : rep[k].x) += s / (ix.size() - 1);
            }
    for (auto& r : rep) { r.x = std::sqrt(r.x / cond.size()); r.y = std::sqrt(r.y / cond.size()); }

    // ---- 점별 표 ----
    struct PRow { int k, valid; double rate, cos, med[5]; };
    std::vector<PRow> rows(P);
    Table pt{{"id", "qr", "gx", "gy", "x", "y", "patch", "valid_count", "valid_rate", "rmse_median", "stability_median", "contrast_median", "sigma_u_median", "sigma_v_median",
              "Y_dx", "Y_dy", "T_dx", "T_dy", "YT_cosine", "repeat_sd_x", "repeat_sd_y"}};
    std::vector<double> abscos, all_valid_img(N, 0);
    int always = 0, never = 0, ppmin = N, ppmax = 0;
    double vsum = 0;
    for (int k = 0; k < P; ++k) {
        auto& c = c630::kCorners[k];
        PRow& r = rows[k];
        r.k = k;
        r.valid = 0;
        std::vector<double> q[5];
        for (int i = 0; i < N; ++i) {
            r.valid += imgs[i].m.valid[k];
            for (int j = 0; j < 5; ++j) q[j].push_back(imgs[i].m.q[k][j]);
        }
        for (int j = 0; j < 5; ++j) r.med[j] = median(q[j]);
        r.rate = (double)r.valid / N;
        double na = std::hypot(sY[k].x, sY[k].y), nb = std::hypot(sT[k].x, sT[k].y);
        r.cos = (sY[k].x * sT[k].x + sY[k].y * sT[k].y) / (na * nb);
        abscos.push_back(std::abs(r.cos));
        always += r.valid == N;
        never += r.valid == 0;
        ppmin = std::min(ppmin, r.valid);
        ppmax = std::max(ppmax, r.valid);
        vsum += r.valid;
        // gx, gy: 원래 corners.json 격자 번호는 id 와 QR 안 순서로만 쓰였으므로 id 를 그대로 적는다
        pt.add({std::to_string(c.id), std::to_string(c.qr), "", "", fmt(ref[k].x, 4), fmt(ref[k].y, 4), patch_str(c), std::to_string(r.valid), fmt(r.rate, 4), fmt(r.med[0], 4),
                fmt(r.med[1], 4), fmt(r.med[2], 3), fmt(r.med[3], 4), fmt(r.med[4], 4), fmt(sY[k].x, 4), fmt(sY[k].y, 5), fmt(sT[k].x, 4), fmt(sT[k].y, 5), fmt(r.cos, 9),
                fmt(rep[k].x, 5), fmt(rep[k].y, 5)});
    }
    int pimin = P, pimax = 0;
    for (auto& im : imgs) { pimin = std::min(pimin, im.nvalid); pimax = std::max(pimax, im.nvalid); }
    double vfrac = vsum / (double)N / P;
    double med_cos = median(abscos);
    std::vector<double> rx, ry;
    for (auto& r : rep) { rx.push_back(r.x); ry.push_back(r.y); }
    double rep_x = median(rx), rep_y = median(ry);
    std::vector<double> qall[5];
    for (auto& im : imgs) for (int k = 0; k < P; ++k) for (int j = 0; j < 5; ++j) qall[j].push_back(im.m.q[k][j]);
    double qmed[5];
    for (int j = 0; j < 5; ++j) qmed[j] = median(qall[j]);
    // QR 별
    Table qt{{"QR", "점 수", "x 범위", "y 범위", "유효율", "Y 기울기 중앙 (dx, dy px/mm)", "T 기울기 중앙 (dx, dy px/deg)", "패치 종류별 점 수"}};
    double qYdx[3], qTdx[3];
    for (int q = 1; q <= 3; ++q) {
        std::vector<double> xs, ys, ydx, ydy, tdx, tdy;
        std::map<std::string, int> pc;
        double v = 0;
        for (int k = 0; k < P; ++k)
            if (c630::kCorners[k].qr == q) {
                xs.push_back(ref[k].x); ys.push_back(ref[k].y);
                ydx.push_back(sY[k].x); ydy.push_back(sY[k].y); tdx.push_back(sT[k].x); tdy.push_back(sT[k].y);
                pc[patch_str(c630::kCorners[k])]++;
                v += rows[k].valid;
            }
        std::string pcs;
        for (auto& [a, b] : pc) pcs += (pcs.empty() ? "" : " ") + a + ":" + std::to_string(b);
        qYdx[q - 1] = median(ydx);
        qTdx[q - 1] = median(tdx);
        qt.add({std::to_string(q), std::to_string(xs.size()), fmt(*std::min_element(xs.begin(), xs.end()), 1) + "-" + fmt(*std::max_element(xs.begin(), xs.end()), 1),
                fmt(*std::min_element(ys.begin(), ys.end()), 1) + "-" + fmt(*std::max_element(ys.begin(), ys.end()), 1), fmt(v / N / xs.size(), 4),
                fmt(median(ydx), 4) + ", " + fmt(median(ydy), 4), fmt(median(tdx), 4) + ", " + fmt(median(tdy), 4), pcs});
    }
    pt.csv(out("points630.csv"));
    // 유효율 낮은 점 10개
    std::vector<int> order(P);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return rows[a].rate < rows[b].rate; });
    Table wt{{"id", "QR", "x", "y", "패치", "유효 장수", "유효율", "rmse 중앙", "창 안정성 중앙", "대비 중앙", "YT 코사인", "반복 SD x"}};
    for (int j = 0; j < 10; ++j) {
        auto& r = rows[order[j]];
        auto& c = c630::kCorners[r.k];
        wt.add({std::to_string(c.id), std::to_string(c.qr), fmt(ref[r.k].x, 1), fmt(ref[r.k].y, 1), patch_str(c), std::to_string(r.valid), fmt(r.rate, 4), fmt(r.med[0], 2),
                fmt(r.med[1], 4), fmt(r.med[2], 1), fmt(r.cos, 7), fmt(rep[r.k].x, 4)});
    }
    // 무효 이유 집계
    const char* why_name[6] = {"optimizer", "photometric_rmse", "low_contrast", "window_instability", "search_limit", "blur_limit"};
    int why_cnt[6] = {0};
    for (auto& im : imgs) for (int k = 0; k < P; ++k) for (int b = 0; b < 6; ++b) why_cnt[b] += (im.m.why[k] >> b) & 1;
    Table why{{"무효 이유", "점-사진 수"}};
    for (int b = 0; b < 6; ++b) why.add({why_name[b], std::to_string(why_cnt[b])});

    // ---- 그림 ----
    {
        std::vector<double> x[3], y[3];
        for (int k = 0; k < P; ++k) {
            int b = rows[k].rate >= 1 ? 0 : rows[k].rate >= .95 ? 1 : 2;
            x[b].push_back(ref[k].x);
            y[b].push_back(-ref[k].y);
        }
        Chart c("630 points: valid rate over 341 images", "x (px)", "-y (px)");
        c.W = 1100; c.H = 420;
        c.scatter(x[0], y[0], color(0), "always valid", 3).scatter(x[1], y[1], color(1), "0.95-1", 4).scatter(x[2], y[2], kRed, "< 0.95", 5);
        auto& w = rows[order[0]];
        c.note(ref[w.k].x, -ref[w.k].y, "id " + std::to_string(c630::kCorners[w.k].id));
        c.save("01_점별_유효율지도.png");
    }
    {
        std::vector<double> id, oc;
        for (int k = 0; k < P; ++k) { id.push_back(k); oc.push_back((1 - std::abs(rows[k].cos)) * 1e6); }
        Chart c("Y/T response direction: (1 - |cos|) x 1e6 per point", "point index", "(1-|cos|) x 1e6");
        c.scatter(id, oc, color(0)).hline((1 - med_cos) * 1e6, kRed);
        c.save("02_점별_YT코사인.png");
    }
    {
        std::vector<double> a, b;
        for (int k = 0; k < P; ++k) { a.push_back(rep[k].x); b.push_back(rows[k].rate); }
        Chart c("Repeat SD x vs valid rate", "repeat SD x (px)", "valid rate");
        c.scatter(a, b, color(0));
        c.save("03_반복SD_유효율.png");
    }
    {
        std::vector<double> yx, tx, id;
        for (int k = 0; k < P; ++k) { id.push_back(k); yx.push_back(sY[k].x); tx.push_back(sT[k].x / (80.0 / 14.25)); }
        Chart c("Per-point x response: Y (px/mm) and T scaled by 14.25/80", "point index", "px per unit");
        c.scatter(id, yx, color(0), "Y dx/dY").scatter(id, tx, color(1), "T dx/dT x 14.25/80", 2);
        c.save("04_점별_YT기울기.png");
    }
    // 튀는 점: 유효율 최저 점. 무효였던 사진 하나를 골라 그 점 주변을 확대 (초록 = 기준 위치 + NCC 이동, 빨강 = 측정 위치)
    auto& w = rows[order[0]];
    int wi = -1;
    for (int i = 0; i < N && wi < 0; ++i) if (!imgs[i].m.valid[w.k] && imgs[i].scope == "single") wi = i;
    if (wi < 0) wi = 0;
    auto& im = imgs[wi];
    auto& wc = c630::kCorners[w.k];
    cv::Point2d sh(im.m.shift[wc.qr - 1].x, im.m.shift[wc.qr - 1].y);
    cv::Point2d init(wc.cx + sh.x, wc.cy + sh.y), meas = im.m.cur[w.k];
    std::string why_s;
    for (int b = 0; b < 6; ++b) if ((im.m.why[w.k] >> b) & 1) why_s += (why_s.empty() ? "" : ",") + std::string(why_name[b]);
    std::string fz = "05_튀는점_id" + std::to_string(wc.id) + "_" + im.name + ".png";
    {
        cv::Mat col = pipe630::imread_u8(im.shot, cv::IMREAD_COLOR);
        cv::Rect roi((int)init.x - 30, (int)init.y - 30, 60, 60);
        cv::Rect qroi((int)init.x - 120, (int)init.y - 120, 240, 240);
        auto a = zoom(col, qroi, 480, {{init, {0, 0}, "", kGreen}, {meas, {0, 0}, "", kRed}}, 0, "id " + std::to_string(wc.id) + " in " + im.name,
                      {{cv::Rect2d(roi), kGreen, ""}});
        auto b = zoom(col, roi, 480, {{init, {0, 0}, "start", kGreen}, {meas, {0, 0}, "fit", kRed}}, 0, why_s);
        save_png(fz, hstack({a, b}));
    }

    // ---- 대조 (원래 값: eda_630_inventory_20261022/summary.json) ----
    Parity par;
    par.add("유효 비율", 0.995405669599218, vfrac, 0.001, 6);
    par.add("항상 유효한 점 수", 417, always, 10, 0);
    par.add("한 번도 유효하지 않은 점 수", 0, never, 0, 0);
    par.add("사진별 유효점 최소", 620, pimin, 2, 0);
    par.add("사진별 유효점 최대", 630, pimax, 0, 0);
    par.add("점별 유효 장수 최소", 221, ppmin, 3, 0);
    par.add("점별 유효 장수 최대", 341, ppmax, 0, 0);
    par.add_text("유효율 최저 점 id", "658", std::to_string(wc.id), wc.id == 658);
    par.add("Y/T 반응 |코사인| 중앙값", 0.9999997466611823, med_cos, 1e-7, 9);
    par.add("반복 SD 중앙 x (px)", 0.17046997184748675, rep_x, 0.0005, 5);
    par.add("반복 SD 중앙 y (px)", 0.025578638226911123, rep_y, 0.0005, 5);
    const double refY[3] = {14.241384224653984, 14.2535709986705, 14.251863375887497}, refT[3] = {80.13684108114816, 79.9037372804721, 80.13392188895001};
    for (int q = 0; q < 3; ++q) {
        par.add("QR" + std::to_string(q + 1) + " Y 기울기 중앙 dx (px/mm)", refY[q], qYdx[q], 0.005);
        par.add("QR" + std::to_string(q + 1) + " T 기울기 중앙 dx (px/deg)", refT[q], qTdx[q], 0.02);
    }
    par.add("품질 중앙 rmse", 10.758761044459717, qmed[0], 0.01);
    par.add("품질 중앙 창 안정성", 0.040423771537639264, qmed[1], 0.0005, 5);
    par.add("품질 중앙 대비", 122.6236232069265, qmed[2], 0.05, 3);

    // ---- 결과.md ----
    std::ostringstream R;
    R << "# 결과: 630점 목록 - 점별 유효율과 Y/T 반응\n\n"
      << "입력: `data/clean` 341장. 630점은 joint_center_630 의 컴파일 LM 경계 맞춤(초기 위치·축·패치는 `corners630.hpp`)으로 쟀고, 기준 좌표는 center 사진을 scipy TRF 로 잰 값이다.\n"
      << "Y 기울기: 단일 축 Y 사진 " << nY << "장, T 기울기: 단일 축 T 사진 " << nT << "장에 [1, 명령값] 최소제곱. 반복 SD: 20261020 혼합 " << cond.size()
      << "조건의 3장 반복 SD(ddof 1)의 RMS.\n\n"
      << "## 전체\n\n| 항목 | 값 |\n|---|---|\n| 사진 x 점 | " << N << " x " << P << " |\n| 유효 비율 | " << fmt(vfrac, 6) << " |\n| 항상 유효 / 한 번도 유효하지 않음 | "
      << always << " / " << never << " |\n| 사진별 유효점 최소 / 최대 | " << pimin << " / " << pimax << " |\n| 점별 유효 장수 최소 / 최대 | " << ppmin << " / " << ppmax
      << " |\n| Y/T \\|코사인\\| 중앙값 | " << fmt(med_cos, 9) << " |\n| 반복 SD 중앙 (x, y) px | " << fmt(rep_x, 4) << ", " << fmt(rep_y, 4)
      << " |\n| 품질 중앙 (rmse, 창 안정성, 대비, sigma_u, sigma_v) | " << fmt(qmed[0], 3) << ", " << fmt(qmed[1], 4) << ", " << fmt(qmed[2], 2) << ", " << fmt(qmed[3], 4)
      << ", " << fmt(qmed[4], 4) << " |\n\n## QR 별\n\n" << qt.md() << "\n## 무효 이유 (점-사진 수, 겹침 허용)\n\n" << why.md() << "\n## 유효율 낮은 점 10개\n\n" << wt.md()
      << "\n## 원래 결과와 대조 (" << par.ok << "/" << par.total << " 일치)\n\n" << par.t.md()
      << "\n## 옮기지 않은 부분\n\n- 원래 `points630.csv` 의 gx, gy(QR 안 격자 번호)는 corners.json 에만 있는 표시용 번호라 비워 두었다. 나머지 열은 같다.\n"
      << "\n## 파일\n\n- `points630.csv`: 점별 유효율·품질 중앙값·Y/T 기울기·YT 코사인·반복 SD\n";
    write_text(out("결과.md"), R.str());

    std::ostringstream Pm;
    auto& w0 = rows[order[0]];
    Pm << "# 보고서: 630점 목록 - 점별 유효율과 반응 방향 (10-22 17:36)\n\n"
       << "## 문제\n\n341장 x 630점 측정값이 얼마나 유효하고 Y/T가 각 점을 어느 방향으로 움직이는지 점 단위로 정리하려 했다.\n\n"
       << "## 해결 방법\n\n점마다 유효율·경계 맞춤 품질 중앙값·단일 축 사진 Y/T 기울기·두 기울기의 코사인·같은 명령 반복 SD를 계산했다.\n\n"
       << "## 결과\n\n유효 비율은 " << fmt(vfrac, 4) << "이고 Y/T 반응 |코사인| 중앙값이 " << fmt(med_cos, 8) << "로, 모든 점에서 Y와 T가 거의 같은 방향(x)으로 움직였다.\n\n"
       << "| 지표 | 값 |\n|---|---|\n| 유효 비율 | " << fmt(vfrac, 4) << " |\n| 항상 유효한 점 | " << always << " / 630 |\n| 사진별 유효점 최소 | " << pimin
       << " |\n| 점별 유효 장수 최소 | " << ppmin << " / 341 |\n| Y/T \\|코사인\\| 중앙값 | " << fmt(med_cos, 8) << " |\n| Y 기울기 x 중앙 (QR1/2/3, px/mm) | " << fmt(qYdx[0], 2)
       << " / " << fmt(qYdx[1], 2) << " / " << fmt(qYdx[2], 2) << " |\n| T 기울기 x 중앙 (QR1/2/3, px/deg) | " << fmt(qTdx[0], 2) << " / " << fmt(qTdx[1], 2) << " / "
       << fmt(qTdx[2], 2) << " |\n| 반복 SD 중앙 x / y (px) | " << fmt(rep_x, 4) << " / " << fmt(rep_y, 4) << " |\n\n"
       << img_md("01_점별_유효율지도.png", "유효율 지도") << "\n\n" << img_md("02_점별_YT코사인.png", "YT 코사인") << "\n\n" << img_md("04_점별_YT기울기.png", "YT 기울기")
       << "\n\n" << img_md("03_반복SD_유효율.png", "반복 SD") << "\n\n"
       << "## 튀는 값\n\n유효율이 가장 낮은 점은 QR" << c630::kCorners[w0.k].qr << "의 id " << c630::kCorners[w0.k].id << "(" << fmt(ref[w0.k].x, 1) << ", "
       << fmt(ref[w0.k].y, 1) << ")로 341장 중 " << w0.valid << "장만 유효했고, rmse 중앙 " << fmt(w0.med[0], 2) << "가 전체 중앙 " << fmt(qmed[0], 2)
       << "보다 커서 경계 모델이 잘 맞지 않는 자리다.\n\n" << wt.md() << "\n" << img_md(fz, "튀는 점")
       << "\n\n왼쪽은 " << im.name << " 사진에서 그 점 주변 240 px, 오른쪽은 60 px 확대다. 초록 십자는 기준 위치 + NCC 이동(맞춤 시작점), 빨간 십자는 맞춘 위치, 글자는 무효 이유다.\n\n"
       << "## 한줄평\n\n630점은 모두 QR 판 위의 좁은 영역(가로 약 680 px)에 있어 Y 1 mm와 T 1 deg가 각 점을 같은 x 방향으로 " << fmt(qYdx[1], 1) << " px, "
       << fmt(qTdx[1], 1) << " px 옮기므로, 점 하나하나의 반응 방향으로는 두 축을 구분할 수 없고(|코사인| " << fmt(med_cos, 7) << ") 구분은 크기의 미세한 공간 차이에만 기대야 한다.\n\n"
       << "## 원래 결과와 대조\n\n원래 값은 `eda_630_inventory_20261022/summary.json`에서 가져왔다. 원래 입력은 joint_center_630 `results/measurements.npz`(컴파일 LM 측정)였고 C++도 같은 LM 변형으로 다시 쟀다. "
       << par.ok << "/" << par.total << " 일치.\n\n" << par.t.md();
    write_text(out("보고서.md"), Pm.str());
    std::cout << "parity " << par.ok << "/" << par.total << std::endl;
    return 0;
}
