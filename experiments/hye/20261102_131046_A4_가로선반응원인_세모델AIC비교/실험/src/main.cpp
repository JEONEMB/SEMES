// A4: 가로선 반응의 원인 (H04 깊이 시차 / H05 회전 투영 / H06 축 기울기).
//  1014 단일축 자료로 블록별 감도 alphaY, alphaT 를 내고, 그 공간 패턴에 네 모델을 맞춰 AIC 로 고른다.
#include "cache.hpp"
#include "noise.hpp"
#include "hye/plot.hpp"
#include <functional>
#include <iostream>

using namespace a0;
namespace {

struct Block {
    int id;
    double x, y;
    bool upper;
};

std::vector<Block> read_blocks(const fs::path& protocol) {
    cv::FileStorage f(protocol.u8string(), cv::FileStorage::READ);
    if (!f.isOpened()) throw std::runtime_error("line_protocol.json 없음");
    std::vector<Block> out;
    int id = 0;
    for (auto it = f["blocks"].begin(); it != f["blocks"].end(); ++it, ++id) {
        auto n = *it;
        auto b = n["read_box"];
        Block k;
        k.id = id;
        k.x = 0.5 * (double(int(b[0])) + double(int(b[2])));
        k.y = 0.5 * (double(int(b[1])) + double(int(b[3])));
        k.upper = std::string(n["line"]) == "upper";
        out.push_back(k);
    }
    if (out.size() != kLines) throw std::runtime_error("선 블록이 48개가 아니다");
    return out;
}

struct ModelFit {
    std::string name;
    int k = 0;
    double rss = 0, aic = 0;
    std::vector<double> coef;
    std::vector<double> pred;
};

ModelFit fit_model(const std::string& name, const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
    const int n = int(y.size()), k = int(X[0].size());
    cv::Mat A(n, k, CV_64F), B(n, 1, CV_64F), C;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) A.at<double>(i, j) = X[i][j];
        B.at<double>(i) = y[i];
    }
    cv::solve(A, B, C, cv::DECOMP_SVD);
    cv::Mat R = B - A * C;
    ModelFit f;
    f.name = name;
    f.k = k;
    f.rss = 0;
    for (int i = 0; i < n; ++i) f.rss += R.at<double>(i) * R.at<double>(i);
    f.aic = n * std::log(std::max(1e-300, f.rss / n)) + 2.0 * k;
    for (int j = 0; j < k; ++j) f.coef.push_back(C.at<double>(j));
    cv::Mat P = A * C;
    for (int i = 0; i < n; ++i) f.pred.push_back(P.at<double>(i));
    return f;
}

}  // namespace

int main() {
    try {
        const fs::path result = fs::u8path(EXPERIMENT_DIR);
        const fs::path results = result / "results";
        fs::create_directories(results);
        hye::begin_results();
        Cache c = load(cache_dir());
        auto blocks = read_blocks(fs::u8path(REPO_DIR) / fs::u8path(u8"experiments/hye/00000_final_Y_T_joint/assets/line_protocol.json"));

        // ---- 블록별 감도 (1014 단일축) ----
        std::vector<int> ids;
        for (auto& s : c.shots) if (s.measured && s.set == "clean_20261014") ids.push_back(s.idx);
        const int n = int(ids.size());
        cv::Mat X(n, 3, CV_64F), Y(n, kLines, CV_64F);
        for (int i = 0; i < n; ++i) {
            X.at<double>(i, 0) = c.shots[ids[i]].y;
            X.at<double>(i, 1) = c.shots[ids[i]].t;
            X.at<double>(i, 2) = 1;
            for (int b = 0; b < kLines; ++b) Y.at<double>(i, b) = c.line(ids[i])[b * 2];
        }
        cv::Mat B;
        cv::solve(X, Y, B, cv::DECOMP_SVD);
        cv::Mat R = Y - X * B;
        std::vector<double> aY(kLines), aT(kLines), resid(kLines);
        for (int b = 0; b < kLines; ++b) {
            aY[b] = B.at<double>(0, b);
            aT[b] = B.at<double>(1, b);
            double s = 0;
            for (int i = 0; i < n; ++i) s += R.at<double>(i, b) * R.at<double>(i, b);
            resid[b] = std::sqrt(s / n);
        }

        // 선 측정 잡음 (1020/1029 반복)
        auto ns = measure_noise(c, [](const Shot&) { return true; }, true);
        std::vector<double> lnoise(kLines, 0);
        for (int b = 0; b < kLines; ++b) {
            double s = 0;
            for (int i = 0; i < ns.line_rows.rows; ++i) s += ns.line_rows.at<double>(i, b) * ns.line_rows.at<double>(i, b);
            lnoise[b] = std::sqrt(s / std::max(1, ns.line_rows.rows));
        }
        const double noise_med = median(lnoise);

        Table bt;
        bt.head = {"block", "line", "x", "y", "alphaY px per mm", "alphaT px per deg", "적합 잔차 px"};
        for (int b = 0; b < kLines; ++b)
            bt.add({std::to_string(b), blocks[b].upper ? "upper" : "lower", num(blocks[b].x, 1), num(blocks[b].y, 1),
                    num(aY[b], 5), num(aT[b], 5), num(resid[b], 5)});
        bt.csv(results / "block_sensitivity.csv");

        // ---- 네 모델 ----
        Table mt;
        mt.head = {"축", "모델", "뜻", "모수", "RSS", "AIC", "AIC - 최소", "계수"};
        auto run_models = [&](const std::string& axis, const std::vector<double>& a) {
            std::vector<std::vector<double>> X1, X2, X3, X4;
            const double xs = 1000.0, ys = 1000.0;  // 좌표 축척 (수치 안정)
            for (int b = 0; b < kLines; ++b) {
                double x = blocks[b].x / xs, y = blocks[b].y / ys;
                double up = blocks[b].upper ? 1 : 0, lo = 1 - up;
                X1.push_back({up, lo});
                X2.push_back({1, x});
                X3.push_back({up, lo, up * x, lo * x});
                X4.push_back({1, x, y, x * y});
            }
            std::vector<ModelFit> fits = {
                fit_model("M1 선별 상수", X1, a),
                fit_model("M2 공통 기울기", X2, a),
                fit_model("M3 선별 상수+기울기", X3, a),
                fit_model("M4 쌍선형", X4, a),
            };
            double best = 1e300;
            for (auto& f : fits) best = std::min(best, f.aic);
            const char* meaning[] = {"깊이 시차 (H04)", "롤 (H06)", "회전 투영 (H05)", "일반 1차 변환"};
            for (size_t i = 0; i < fits.size(); ++i) {
                std::string cs;
                for (size_t j = 0; j < fits[i].coef.size(); ++j) cs += (j ? ", " : "") + num(fits[i].coef[j], 5);
                mt.add({axis, fits[i].name, meaning[i], std::to_string(fits[i].k), num(fits[i].rss, 7),
                        num(fits[i].aic, 2), num(fits[i].aic - best, 2), cs});
            }
            return fits;
        };
        auto fy = run_models("Y", aY);
        auto ft = run_models("T", aT);
        mt.csv(results / "model_aic.csv");

        // ---- 그림 ----
        auto plot_axis = [&](const std::string& file, const std::string& title, const std::vector<double>& a,
                             const std::vector<ModelFit>& fits) {
            hye::Chart ch(title, "block x [px]", "sensitivity [px per unit]");
            for (int up = 0; up < 2; ++up) {
                std::vector<double> x, y;
                for (int b = 0; b < kLines; ++b) if (blocks[b].upper == (up == 1)) { x.push_back(blocks[b].x); y.push_back(a[b]); }
                ch.scatter(x, y, hye::color(up), up ? "upper" : "lower", 5);
            }
            // 가장 좋은 모델의 적합값
            size_t best = 0;
            for (size_t i = 1; i < fits.size(); ++i) if (fits[i].aic < fits[best].aic) best = i;
            for (int up = 0; up < 2; ++up) {
                std::vector<std::pair<double, double>> pts;
                for (int b = 0; b < kLines; ++b) if (blocks[b].upper == (up == 1)) pts.push_back({blocks[b].x, fits[best].pred[b]});
                std::sort(pts.begin(), pts.end());
                std::vector<double> x, y;
                for (auto& p : pts) { x.push_back(p.first); y.push_back(p.second); }
                ch.line(x, y, hye::color(up + 2), up ? ("fit " + fits[best].name + " upper") : ("fit lower"));
            }
            ch.hline(noise_med * 3, hye::kGray);
            ch.hline(-noise_med * 3, hye::kGray);
            ch.save(file);
        };
        plot_axis("41_블록감도_Y.png", "A4-41 line block sensitivity to Y [px per mm]", aY, fy);
        plot_axis("42_블록감도_T.png", "A4-42 line block sensitivity to T [px per deg]", aT, ft);
        {
            std::vector<std::string> cats;
            std::vector<double> vals;
            for (auto& r : mt.rows) { cats.push_back(r[0] + " " + r[1]); vals.push_back(std::stod(r[6])); }
            hye::bar_chart("44_AIC.png", "A4-44 AIC minus best (lower is better)", cats, vals, "dAIC", -1, 10);
        }

        // ---- 판정 ----
        // 선이 위·아래 두 줄뿐이라 M4(쌍선형)는 M3 과 같은 4차원 공간을 친다. 실제로 RSS 가 거의 같다.
        // 그래서 판정은 "구조 모델 M3" 대 "단순 모델 중 좋은 쪽(M1, M2)" 으로 한다.
        auto compare = [](const std::vector<ModelFit>& f) {
            double simple = std::min(f[0].aic, f[1].aic);
            return std::make_pair(size_t(2), simple - f[2].aic);  // M3 가 단순 모델보다 얼마나 좋은가
        };
        auto by = compare(fy), bt2 = compare(ft);
        double aY_rms = rms(aY), aT_rms = rms(aT);
        double bup_t = ft[2].coef[2], blo_t = ft[2].coef[3];
        double bup_y = fy[2].coef[2], blo_y = fy[2].coef[3];

        std::ostringstream o;
        o << "# A4 결과 — 가로선 반응의 원인\n\n"
          << "`실험/src/main.cpp` 가 다시 쓴다. 해석은 [보고서](보고서.md). 계획은 [계획서](../계획서.md).\n\n"
          << "1014 단일축 " << n << "장으로 블록별 감도를 냈다. 선 측정 잡음 중앙값 " << num(noise_med, 5) << " px.\n\n"
          << "## 1. 블록별 감도\n\n"
          << "| 축 | 감도 RMS (px per unit) | 잡음 대비 |\n|---|---|---|\n"
          << "| Y | " << num(aY_rms, 5) << " | " << num(aY_rms / noise_med, 1) << "배 |\n"
          << "| T | " << num(aT_rms, 5) << " | " << num(aT_rms / noise_med, 1) << "배 |\n\n"
          << hye::img_md("41_블록감도_Y.png", "alphaY") << "\n\n"
          << hye::img_md("42_블록감도_T.png", "alphaT") << "\n\n"
          << "회색 선은 잡음의 3배다. 블록별 값: [block_sensitivity.csv](results/block_sensitivity.csv)\n\n"
          << "## 2. 모델 비교 (AIC)\n\n" << mt.md() << "\n"
          << hye::img_md("44_AIC.png", "AIC") << "\n\n"
          << "빨간 선이 판정선 10이다.\n\n"
          << "**M4(쌍선형)는 M3 과 같은 모델이다.** 선이 위·아래 두 줄뿐이라 y 가 사실상 두 값만 가지므로 "
          << "{상수, x, y, xy} 와 {위상수, 아래상수, 위기울기, 아래기울기} 가 같은 4차원 공간을 친다. RSS 가 거의 같은 것이 그 증거다. "
          << "그래서 판정은 **M3 대 단순 모델(M1, M2)**로 한다. (판정 절차 보완: 2026-11-02, 결과를 본 뒤)\n\n"
          << "## 3. 판정 기준 대비\n\n"
          << "| 조건 | 실측 | 판정 |\n|---|---|---|\n"
          << "| Y 감도가 잡음의 3배 초과 (H04 깊이 시차) | " << num(aY_rms / noise_med, 1) << "배 | "
          << (aY_rms > 3 * noise_med ? "**지지**" : "지지 안 됨") << " |\n"
          << "| T 축: M3(회전 투영)이 단순 모델 M1/M2 보다 AIC 10 이상 낮음 | 차 " << num(bt2.second, 1) << " | "
          << (bt2.second >= 10 ? "**M3 채택 = 회전 투영(H05)**" : "보류 (B4로)") << " |\n"
          << "| Y 축: 같은 비교 | 차 " << num(by.second, 1) << " | "
          << (by.second >= 10 ? "**M3 채택**" : "보류 (B4로)") << " |\n"
          << "| M3 기울기 부호 (T) | b_up " << num(bup_t, 5) << ", b_lo " << num(blo_t, 5) << " | "
          << (bup_t * blo_t < 0 ? "**반대 부호 → 회전 투영(H05)**" : "같은 부호 → 롤(H06)") << " |\n"
          << "| M3 기울기 부호 (Y) | b_up " << num(bup_y, 5) << ", b_lo " << num(blo_y, 5) << " | "
          << (bup_y * blo_y < 0 ? "반대 부호" : "같은 부호") << " |\n";
        write_text(result / fs::u8path(u8"결과.md"), o.str());
        std::cout << "끝\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "오류: " << e.what() << "\n";
        return 1;
    }
}
