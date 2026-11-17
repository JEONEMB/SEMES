// A9: 실측 영점 기반 데이터 증강 (H12, H18, 규칙 64행).
//  학습 날짜만으로 생성기(J, 영점, 잔차 풀)를 만들고, 특징 수준에서 합성 표본을 만들어 학습에 더한다.
//  빠진 날짜의 시험 사진에서 실측만 학습과 비교한다.
#include "cache.hpp"
#include "noise.hpp"
#include "model.hpp"
#include "hye/plot.hpp"
#include <functional>
#include <iostream>
#include <set>

using namespace a0;

int main() {
    try {
        const fs::path result = fs::u8path(EXPERIMENT_DIR);
        const fs::path results = result / "results";
        fs::create_directories(results);
        hye::begin_results();
        Cache c = load(cache_dir());

        std::vector<std::vector<double>> f66(c.n());
        for (auto& s : c.shots) if (s.measured) f66[s.idx] = features(c, s.idx, true);

        Table gen;
        gen.head = {"빠진 날짜", "학습 사진", "합성 사진", "Y 범위", "T 범위", "YxT 결합항 사용", "잔차 풀 크기"};
        Table st;
        st.head = {"빠진 날짜", "학습", "장수", "Y 평균", "Y 최대", "T 평균", "T 최대", "통과 ±0.045"};
        Table perf;
        perf.head = {"experiment", "method", "split", "n", "mean_y", "max_y", "mean_t", "max_t", "pass_0045", "pass_009"};

        std::vector<double> ax, ay, sx, sy;  // 자극 지도용

        const char* dname[] = {"1014", "1020", "1029"};
        for (int hold = 0; hold < 3; ++hold) {
            // 학습 사진
            std::vector<int> train, test;
            for (auto& s : c.shots) {
                if (!s.measured) continue;
                (s.fold_date == hold ? test : train).push_back(s.idx);
            }
            if (train.empty() || test.empty()) continue;

            // ---- 생성기 ----
            // 1) 특징 감도: 학습 사진에서 f = A*Y + B*T + C*Y*T + f0
            const int n = int(train.size());
            cv::Mat X(n, 4, CV_64F), Yf(n, kFeat, CV_64F);
            for (int i = 0; i < n; ++i) {
                const Shot& s = c.shots[train[i]];
                X.at<double>(i, 0) = s.y;
                X.at<double>(i, 1) = s.t;
                X.at<double>(i, 2) = s.y * s.t;   // 규칙 64행: Y x T 결합항을 0 으로 두지 않는다
                X.at<double>(i, 3) = 1;
                for (int j = 0; j < kFeat; ++j) Yf.at<double>(i, j) = f66[train[i]][j];
            }
            cv::Mat B;
            cv::solve(X, Yf, B, cv::DECOMP_SVD);
            cv::Mat R = Yf - X * B;   // 잔차 풀: 실측 영점 변동 + 측정 잡음 + 모델 오차가 모두 들어 있다

            // 2) 합성할 (Y,T): 학습 날짜의 자극 범위 안에서 격자. 밖으로 외삽하지 않는다.
            double ylo = 1e9, yhi = -1e9, tlo = 1e9, thi = -1e9;
            for (int i : train) {
                ylo = std::min(ylo, c.shots[i].y); yhi = std::max(yhi, c.shots[i].y);
                tlo = std::min(tlo, c.shots[i].t); thi = std::max(thi, c.shots[i].t);
            }
            std::vector<std::vector<double>> Xa;
            std::vector<cv::Point2d> Ya;
            for (int i : train) { Xa.push_back(f66[i]); Ya.push_back({c.shots[i].y, c.shots[i].t}); }
            const int gy = 21, gt = 21;
            int synth = 0;
            for (int a = 0; a < gy; ++a)
                for (int b = 0; b < gt; ++b) {
                    double yy = ylo + (yhi - ylo) * a / (gy - 1.0);
                    double tt = tlo + (thi - tlo) * b / (gt - 1.0);
                    // 잔차 풀에서 한 행을 뽑아 더한다 (독립 가우스를 쓰지 않는다. 상관 구조를 그대로 가져온다)
                    int r = (a * gt + b) % R.rows;
                    std::vector<double> f(kFeat);
                    for (int j = 0; j < kFeat; ++j)
                        f[j] = B.at<double>(0, j) * yy + B.at<double>(1, j) * tt + B.at<double>(2, j) * yy * tt +
                               B.at<double>(3, j) + R.at<double>(r, j);
                    Xa.push_back(std::move(f));
                    Ya.push_back({yy, tt});
                    ++synth;
                    if (hold == 0) { sx.push_back(yy); sy.push_back(tt); }
                }
            gen.add({dname[hold], std::to_string(n), std::to_string(synth), num(ylo, 2) + " ~ " + num(yhi, 2),
                     num(tlo, 2) + " ~ " + num(thi, 2), "예", std::to_string(R.rows)});

            // ---- 두 학습 비교 ----
            auto eval = [&](const std::vector<std::vector<double>>& XX, const std::vector<cv::Point2d>& YY, const std::string& label) {
                LinearModel m;
                m.fit(XX, YY, 0);
                std::vector<double> ey, et;
                int pass = 0, cnt = 0, pass09 = 0;
                for (int i : test) {
                    if (c.shots[i].t > 1.001) continue;   // T+2.00 범위 밖
                    auto p = m.predict(f66[i]);
                    double a = p.x - c.shots[i].y, b = p.y - c.shots[i].t;
                    ey.push_back(a); et.push_back(b);
                    ++cnt;
                    if (std::abs(a) <= kTol && std::abs(b) <= kTol) ++pass;
                    if (std::abs(a) <= 0.09 && std::abs(b) <= 0.09) ++pass09;
                }
                std::vector<double> aay, aat;
                for (double v : ey) aay.push_back(std::abs(v));
                for (double v : et) aat.push_back(std::abs(v));
                st.add({dname[hold], label, std::to_string(cnt), num(mean(aay), 4), num(max_abs(ey), 4),
                        num(mean(aat), 4), num(max_abs(et), 4), num(cnt ? double(pass) / cnt : 0, 4)});
                perf.add({"A9", label, std::string("hold-") + dname[hold], std::to_string(cnt), num(mean(aay), 4),
                          num(max_abs(ey), 4), num(mean(aat), 4), num(max_abs(et), 4),
                          num(cnt ? double(pass) / cnt : 0, 4), num(cnt ? double(pass09) / cnt : 0, 4)});
                return std::make_pair(max_abs(ey), cnt ? double(pass) / cnt : 0);
            };
            std::vector<std::vector<double>> Xr;
            std::vector<cv::Point2d> Yr;
            for (int i : train) { Xr.push_back(f66[i]); Yr.push_back({c.shots[i].y, c.shots[i].t}); }
            eval(Xr, Yr, "실측만");
            eval(Xa, Ya, "실측 + 증강");
        }
        gen.csv(results / "generator.csv");
        st.csv(results / "aug_scores.csv");
        perf.csv(results / "perf_rows.csv");

        for (auto& s : c.shots) if (s.measured) { ax.push_back(s.y); ay.push_back(s.t); }
        {
            hye::Chart ch("A9-92 excitation: measured vs synthesized (hold-out 1014)", "Y [mm]", "T [deg]");
            ch.scatter(sx, sy, hye::color(1), "synthesized grid", 3);
            ch.scatter(ax, ay, hye::color(0), "measured", 4);
            ch.save("92_증강자극지도.png");
        }
        {
            std::vector<std::string> cats;
            std::vector<double> vals;
            for (auto& r : st.rows) { cats.push_back(r[0] + " " + r[1]); vals.push_back(std::stod(r[7])); }
            hye::bar_chart("91_증강효과.png", "A9-91 pass rate at +-0.045 by held-out date", cats, vals, "pass rate");
        }

        // 판정
        bool any_better = false;
        std::ostringstream verdict;
        for (size_t i = 0; i + 1 < st.rows.size(); i += 2) {
            double m0 = std::stod(st.rows[i][4]), m1 = std::stod(st.rows[i + 1][4]);
            double p0 = std::stod(st.rows[i][7]), p1 = std::stod(st.rows[i + 1][7]);
            bool ok = m1 < m0 && p1 >= p0;
            any_better = any_better || ok;
            verdict << "| " << st.rows[i][0] << " | " << num(m0, 4) << " -> " << num(m1, 4) << " | " << num(p0, 4)
                    << " -> " << num(p1, 4) << " | " << (ok ? "**개선**" : "개선 없음") << " |\n";
        }

        std::ostringstream o;
        o << "# A9 결과 — 실측 영점 기반 데이터 증강\n\n"
          << "`실험/src/main.cpp` 가 다시 쓴다. 해석은 [보고서](보고서.md). 계획은 [계획서](../계획서.md).\n\n"
          << "**합성 성적은 실측 추정 성적이 아니다.** 증강은 특징 수준에서 만들며 영상이 아니다.\n\n"
          << "## 1. 생성기\n\n" << gen.md() << "\n"
          << "잔차 풀은 학습 사진의 모델 잔차를 그대로 재표집한다. 독립 가우스를 쓰지 않아 A1 이 잰 상관 구조(N_eff 19)가 유지된다.\n\n"
          << hye::img_md("92_증강자극지도.png", "excitation map") << "\n\n"
          << "## 2. 날짜 fold 성적\n\n" << st.md() << "\n"
          << hye::img_md("91_증강효과.png", "augmentation effect") << "\n\n"
          << "## 3. 판정 기준 대비\n\n"
          << "최대오차가 줄고 통과율이 떨어지지 않아야 채택 후보다.\n\n"
          << "| 빠진 날짜 | Y 최대오차 | 통과율 | 판정 |\n|---|---|---|---|\n" << verdict.str() << "\n"
          << "종합: " << (any_better ? "일부 날짜에서 개선" : "**어느 날짜에서도 개선 없음**") << "\n";
        write_text(result / fs::u8path(u8"결과.md"), o.str());
        std::cout << "끝\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "오류: " << e.what() << "\n";
        return 1;
    }
}
