// A3: 실제 오차의 방향과 날짜 일반화 (H01 실측, H18).
//  기준 모델 = QR별 아핀 18 + 선 48 = 66차원 선형 (model.hpp). 전달 모델이 아니라 진단용 대리 모델이다.
//  두 분할로 OOF 를 내고, 오차 (eY,eT) 의 상관과 주축을 약한 방향과 비교한다.
#include "cache.hpp"
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

        // 특징
        std::vector<std::vector<double>> feat(c.n()), feat_qr(c.n());
        for (auto& s : c.shots) if (s.measured) {
            feat[s.idx] = features(c, s.idx, true);
            feat_qr[s.idx] = features(c, s.idx, false);
        }
        std::cout << "특징 " << kFeat << "차원\n";

        auto mixed = [&](const Shot& s) { return s.fold_mixed; };
        auto bydate = [&](const Shot& s) { return s.fold_date; };

        struct Run { std::string name; Oof o; };
        std::vector<Run> runs;
        runs.push_back({"혼합 4-fold, QR+양쪽 선", run_oof(c, feat, mixed, 4, 0)});
        runs.push_back({"혼합 4-fold, QR만", run_oof(c, feat_qr, mixed, 4, 0)});
        runs.push_back({"날짜 3-fold, QR+양쪽 선", run_oof(c, feat, bydate, 3, 0)});
        runs.push_back({"날짜 3-fold, QR만", run_oof(c, feat_qr, bydate, 3, 0)});

        // T+2.00 은 범위 밖이라 따로 센다
        auto keep_in_range = [&](const Oof& o) {
            std::vector<unsigned char> k(o.shot.size(), 1);
            // truth = (Y, T). T+2.00 은 1014 학습 범위 ±1.00 밖이라 뺀다.
            for (size_t i = 0; i < o.shot.size(); ++i) if (o.truth[i].y > 1.001) k[i] = 0;
            return k;
        };

        Table st;
        st.head = {"평가", "장수", "Y 평균", "Y 최대", "T 평균", "T 최대", "동시 통과 ±0.045", "참고: ±0.09", "오차 상관 r", "주축 각도 deg"};
        Table perf;  // 종합보고서용
        perf.head = {"experiment", "method", "split", "n", "mean_y", "max_y", "mean_t", "max_t", "pass_0045", "pass_009"};
        for (auto& r : runs) {
            auto k = keep_in_range(r.o);
            auto s = score(r.o, &k);
            auto a = error_axis(r.o, &k);
            st.add({r.name, std::to_string(s.n), num(s.mean_y, 4), num(s.max_y, 4), num(s.mean_t, 4), num(s.max_t, 4),
                    num(s.pass, 4), num(s.pass09, 4), num(a.r, 3), num(a.angle_to_null, 2)});
            std::string split = r.name.find("날짜") != std::string::npos ? "date3" : "mixed4";
            std::string meth = r.name.find("QR만") != std::string::npos ? "QR only (18d)" : "QR + both lines (66d)";
            perf.add({"A3", meth, split, std::to_string(s.n), num(s.mean_y, 4), num(s.max_y, 4), num(s.mean_t, 4),
                      num(s.max_t, 4), num(s.pass, 4), num(s.pass09, 4)});
        }
        // T+2.00 따로
        {
            auto& r = runs[0];
            std::vector<unsigned char> k(r.o.shot.size(), 0);
            for (size_t i = 0; i < r.o.shot.size(); ++i) if (r.o.truth[i].y > 1.001) k[i] = 1;
            auto s = score(r.o, &k);
            st.add({"(참고) T+2.00 범위 밖만", std::to_string(s.n), num(s.mean_y, 4), num(s.max_y, 4), num(s.mean_t, 4),
                    num(s.max_t, 4), num(s.pass, 4), num(s.pass09, 4), "", ""});
        }
        st.csv(results / "scores.csv");
        perf.csv(results / "perf_rows.csv");

        // 날짜별 성적 (날짜 3-fold)
        Table dt;
        dt.head = {"빠진 날짜", "장수", "Y 평균", "Y 최대", "T 평균", "T 최대", "동시 통과 ±0.045"};
        for (int f = 0; f < 3; ++f) {
            auto& o = runs[2].o;
            std::vector<unsigned char> k(o.shot.size(), 0);
            for (size_t i = 0; i < o.shot.size(); ++i)
                if (o.fold[i] == f && !(o.truth[i].y > 1.001)) k[i] = 1;
            auto s = score(o, &k);
            const char* nm[] = {"1014", "1020", "1029"};
            dt.add({nm[f], std::to_string(s.n), num(s.mean_y, 4), num(s.max_y, 4), num(s.mean_t, 4), num(s.max_t, 4), num(s.pass, 4)});
        }
        dt.csv(results / "by_date.csv");

        // 사진별 예측
        Table pr;
        pr.head = {"shot", "set", "rel", "Y 정답", "T 정답", "Y 예측", "T 예측", "eY", "eT", "fold"};
        for (size_t i = 0; i < runs[0].o.shot.size(); ++i) {
            int id = runs[0].o.shot[i];
            pr.add({std::to_string(id), c.shots[id].set.substr(6), c.shots[id].rel, num(runs[0].o.truth[i].x, 3),
                    num(runs[0].o.truth[i].y, 3), num(runs[0].o.pred[i].x, 4), num(runs[0].o.pred[i].y, 4),
                    num(runs[0].o.pred[i].x - runs[0].o.truth[i].x, 4), num(runs[0].o.pred[i].y - runs[0].o.truth[i].y, 4),
                    std::to_string(runs[0].o.fold[i])});
        }
        pr.csv(results / "predictions_mixed.csv");

        // ---- 그림 ----
        {  // 31 오차 산점도와 약한 방향
            hye::Chart ch("A3-31 OOF error (mixed 4-fold, QR + both lines)", "eY [mm]", "eT [deg]");
            std::vector<double> x, y, x2, y2;
            auto& o = runs[0].o;
            for (size_t i = 0; i < o.shot.size(); ++i) {
                bool oor = o.truth[i].y > 1.001;
                (oor ? x2 : x).push_back(o.pred[i].x - o.truth[i].x);
                (oor ? y2 : y).push_back(o.pred[i].y - o.truth[i].y);
            }
            ch.scatter(x, y, hye::color(0), "in range", 3);
            if (!x2.empty()) ch.scatter(x2, y2, hye::kRed, "T+2.00 (out of range)", 4);
            double sp = std::max(max_abs(x), 0.05);
            cv::Point2d nd(-kTtoY, 1.0);
            nd *= 1.0 / cv::norm(nd);
            std::vector<double> lx = {-nd.x * sp * 1.2, nd.x * sp * 1.2}, ly = {-nd.y * sp * 1.2, nd.y * sp * 1.2};
            ch.line(lx, ly, hye::kGray, "dY = -5.617 dT");
            ch.hline(kTol, hye::kRed);
            ch.hline(-kTol, hye::kRed);
            ch.vline(kTol, hye::kRed);
            ch.vline(-kTol, hye::kRed);
            ch.save("31_오차방향.png");
        }
        {  // 32 통과율 비교
            std::vector<std::string> cats;
            std::vector<double> vals;
            for (auto& r : runs) {
                auto k = keep_in_range(r.o);
                cats.push_back(r.name);
                vals.push_back(score(r.o, &k).pass);
            }
            hye::bar_chart("32_통과율.png", "A3-32 simultaneous pass rate at +-0.045", cats, vals, "pass rate");
        }
        {  // 33 T 크기별 오차 (1029)
            hye::Chart ch("A3-33 error vs commanded T (1029, Y fixed +0.05)", "T command [deg]", "error");
            auto& o = runs[0].o;
            std::vector<double> t, ey, et;
            for (size_t i = 0; i < o.shot.size(); ++i) {
                if (c.shots[o.shot[i]].set != "clean_20261029") continue;
                t.push_back(o.truth[i].y);
                ey.push_back(o.pred[i].x - o.truth[i].x);
                et.push_back(o.pred[i].y - o.truth[i].y);
            }
            ch.scatter(t, ey, hye::color(0), "eY [mm]", 3);
            ch.scatter(t, et, hye::color(1), "eT [deg]", 3);
            ch.hline(kTol, hye::kRed);
            ch.hline(-kTol, hye::kRed);
            ch.save("33_T크기별오차.png");
        }
        {  // 34 정답 대 예측
            auto& o = runs[0].o;
            std::vector<double> ty, py, tt, pt;
            for (size_t i = 0; i < o.shot.size(); ++i) {
                ty.push_back(o.truth[i].x); py.push_back(o.pred[i].x);
                tt.push_back(o.truth[i].y); pt.push_back(o.pred[i].y);
            }
            hye::Chart a("A3-34a Y: truth vs OOF prediction", "Y truth [mm]", "Y pred [mm]");
            a.scatter(ty, py, hye::color(0), "", 3);
            a.save("34a_Y예측.png");
            hye::Chart b("A3-34b T: truth vs OOF prediction", "T truth [deg]", "T pred [deg]");
            b.scatter(tt, pt, hye::color(1), "", 3);
            b.save("34b_T예측.png");
        }
        {  // 35 날짜별 오차 분포
            hye::Chart ch("A3-35 |eY| by held-out date (date 3-fold)", "shot index within date", "|eY| [mm]");
            auto& o = runs[2].o;
            for (int f = 0; f < 3; ++f) {
                std::vector<double> x, y;
                int k = 0;
                for (size_t i = 0; i < o.shot.size(); ++i) {
                    if (o.fold[i] != f || o.truth[i].y > 1.001) continue;
                    x.push_back(k++);
                    y.push_back(std::abs(o.pred[i].x - o.truth[i].x));
                }
                const char* nm[] = {"1014 held out", "1020 held out", "1029 held out"};
                ch.scatter(x, y, hye::color(f), nm[f], 3);
            }
            ch.hline(kTol, hye::kRed);
            ch.save("35_날짜별오차.png");
        }

        // ---- 결과.md ----
        auto k0 = keep_in_range(runs[0].o), k2 = keep_in_range(runs[2].o);
        auto s_mixed = score(runs[0].o, &k0), s_date = score(runs[2].o, &k2);
        auto a_mixed = error_axis(runs[0].o, &k0);
        bool h01 = std::abs(a_mixed.r) > 0.8 && a_mixed.angle_to_null < 10;
        bool h18 = (s_mixed.pass - s_date.pass) >= 0.10;
        std::ostringstream o;
        o << "# A3 결과 — 실제 오차의 방향과 날짜 일반화\n\n"
          << "`실험/src/main.cpp` 가 다시 쓴다. 해석은 [보고서](보고서.md). 계획은 [계획서](../계획서.md).\n\n"
          << "기준 모델: QR별 아핀 18 + 선 48 = " << kFeat << "차원 선형. **전달한 joint/control 모델이 아니다.** "
          << "fold 마다 다시 학습해야 해서 만든 진단용 대리 모델이다.\n\n"
          << "허용 절대오차 ±" << num(kTol, 3) << ". ±0.09 열은 이전 실험과 견주기 위한 참고다.\n\n"
          << "## 1. 성적\n\n" << st.md() << "\n"
          << hye::img_md("32_통과율.png", "pass rate") << "\n\n"
          << "## 2. 오차가 어느 방향으로 몰리는가 (H01)\n\n"
          << hye::img_md("31_오차방향.png", "error direction") << "\n\n"
          << "상관 r = " << num(a_mixed.r, 3) << ", 주축이 ΔY = −5.617ΔT 와 이루는 각 " << num(a_mixed.angle_to_null, 2) << "°.\n\n"
          << "## 3. 날짜 일반화 (H18)\n\n" << dt.md() << "\n"
          << hye::img_md("35_날짜별오차.png", "by date") << "\n\n"
          << "## 4. T 크기와 예측\n\n"
          << hye::img_md("33_T크기별오차.png", "error vs T") << "\n\n"
          << hye::img_md("34a_Y예측.png", "Y pred") << "\n\n"
          << hye::img_md("34b_T예측.png", "T pred") << "\n\n"
          << "## 5. 판정 기준 대비\n\n"
          << "| 가설 | 조건 | 실측 | 판정 |\n|---|---|---|---|\n"
          << "| H01 | |r| > 0.8 그리고 주축이 약한 방향과 10° 이내 | r " << num(a_mixed.r, 3) << ", "
          << num(a_mixed.angle_to_null, 2) << "° | " << (h01 ? "**성립**" : "기각") << " |\n"
          << "| H18 | 날짜 fold 통과율이 혼합 fold보다 10%p 이상 낮음 | 혼합 " << num(s_mixed.pass * 100, 1)
          << "% 대 날짜 " << num(s_date.pass * 100, 1) << "% (차 " << num((s_mixed.pass - s_date.pass) * 100, 1)
          << "%p) | " << (h18 ? "**성립**" : "기각") << " |\n";
        write_text(result / fs::u8path(u8"결과.md"), o.str());
        std::cout << "끝\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "오류: " << e.what() << "\n";
        return 1;
    }
}
