// A10: 전달 모듈(joint, control)을 575장에 그대로 돌려 기준 모델과 비교한다.
//  joint()   : QR 102관측(54 QR 표본 변위 + 48 선 이동) -> 204차원 -> (Y,T) 동시 추정. **위·아래 바를 쓴다.**
//  control() : QR 536점 P12 -> Y 또는 T 하나만 반환. **위·아래 바를 쓰지 않는다.**
//  두 모듈은 이 575장으로 학습됐으므로 여기 성적은 재대입(개발) 성적이다. OOF 가 아니다.
#include "alignment.hpp"
#include "cache.hpp"
#include "model.hpp"
#include "hye/plot.hpp"
#include <chrono>
#include <functional>
#include <iostream>

using namespace a0;
namespace {
cv::Mat imread_u8(const fs::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    std::vector<char> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return cv::imdecode(buf, cv::IMREAD_GRAYSCALE);
}
struct Acc {
    std::vector<double> ey, et, ms;
    int n = 0, pass = 0, pass09 = 0, fail = 0, axis_ok = 0, axis_n = 0;
};
void add(Acc& a, double ey, double et, double ms) {
    a.ey.push_back(ey); a.et.push_back(et); a.ms.push_back(ms);
    ++a.n;
    if (std::abs(ey) <= kTol && std::abs(et) <= kTol) ++a.pass;
    if (std::abs(ey) <= 0.09 && std::abs(et) <= 0.09) ++a.pass09;
}
std::vector<std::string> row(const std::string& name, const Acc& a) {
    std::vector<double> ay, at;
    for (double v : a.ey) ay.push_back(std::abs(v));
    for (double v : a.et) at.push_back(std::abs(v));
    return {name, std::to_string(a.n), std::to_string(a.fail), num(mean(ay), 4), num(max_abs(a.ey), 4),
            num(mean(at), 4), num(max_abs(a.et), 4), num(a.n ? double(a.pass) / a.n : 0, 4),
            num(a.n ? double(a.pass09) / a.n : 0, 4), num(median(a.ms), 1), num(quantile(a.ms, 0.95), 1)};
}
}  // namespace

int main() {
    try {
        const fs::path repo = fs::u8path(REPO_DIR);
        const fs::path result = fs::u8path(EXPERIMENT_DIR);
        const fs::path results = result / "results";
        fs::create_directories(results);
        hye::begin_results();
        Cache c = load(cache_dir());

        initialize_joint(JOINT_ASSETS);
        initialize_control(CONTROL_ASSETS);
        std::cout << "전달 모듈 초기화 완료\n";

        // 기준 모델 두 가지: OOF(혼합 4-fold) 와 전수 재대입
        std::vector<std::vector<double>> f66(c.n());
        for (auto& s : c.shots) if (s.measured) f66[s.idx] = features(c, s.idx, true);
        auto oof = run_oof(c, f66, [](const Shot& s) { return s.fold_mixed; }, 4, 0);
        std::map<int, cv::Point2d> oof_pred;
        for (size_t i = 0; i < oof.shot.size(); ++i) oof_pred[oof.shot[i]] = oof.pred[i];
        LinearModel full;
        {
            std::vector<std::vector<double>> X;
            std::vector<cv::Point2d> Y;
            for (auto& s : c.shots) if (s.measured) { X.push_back(f66[s.idx]); Y.push_back({s.y, s.t}); }
            full.fit(X, Y, 0);
        }

        // ---- 사진마다 네 추정기 ----
        Table per;
        per.head = {"shot", "set", "rel", "Y", "T", "joint_Y", "joint_T", "control_axis", "control_Y", "control_T",
                    "기준OOF_Y", "기준OOF_T", "기준재대입_Y", "기준재대입_T", "joint_ms", "control_ms"};
        Acc A_joint, A_control, A_oof, A_full;
        Acc A_joint_in, A_control_in, A_oof_in, A_full_in;  // 범위 안(T<=1.00)만
        Table failed;
        failed.head = {"shot", "사진", "모듈", "이유"};
        int axis_ok = 0, axis_n = 0;

        for (auto& s : c.shots) {
            if (!s.measured) continue;
            cv::Mat img = imread_u8(repo / "data/clean" / fs::u8path(s.set) / fs::u8path(s.rel));
            if (img.empty()) throw std::runtime_error("읽기 실패: " + s.rel);
            Alignment_t j{0, 0}, k{0, 0};
            double jms = 0, kms = 0;
            std::string axis = "";
            bool jok = true, kok = true;
            try {
                auto t0 = std::chrono::steady_clock::now();
                j = joint(img);
                jms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            } catch (const std::exception& e) { jok = false; failed.add({std::to_string(s.idx), s.name(), "joint", e.what()}); }
            try {
                auto t0 = std::chrono::steady_clock::now();
                k = control(img);
                kms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                axis = alignment_diagnostics().selected_axis;
            } catch (const std::exception& e) { kok = false; failed.add({std::to_string(s.idx), s.name(), "control", e.what()}); }

            auto po = oof_pred.count(s.idx) ? oof_pred[s.idx] : cv::Point2d(0, 0);
            auto pf = full.predict(f66[s.idx]);
            const bool in_range = s.t <= 1.001;
            if (jok) { add(A_joint, j.y - s.y, j.t - s.t, jms); if (in_range) add(A_joint_in, j.y - s.y, j.t - s.t, jms); }
            else { ++A_joint.fail; if (in_range) ++A_joint_in.fail; }
            if (kok) {
                add(A_control, k.y - s.y, k.t - s.t, kms);
                if (in_range) add(A_control_in, k.y - s.y, k.t - s.t, kms);
                // 축 선택 정확도: 단일축 사진에서만 뜻이 있다
                bool single_y = std::abs(s.t) < 1e-9 && std::abs(s.y) > 1e-9;
                bool single_t = std::abs(s.y) < 1e-9 && std::abs(s.t) > 1e-9;
                if (single_y || single_t) { ++axis_n; if ((single_y && axis == "Y") || (single_t && axis == "T")) ++axis_ok; }
            } else { ++A_control.fail; if (in_range) ++A_control_in.fail; }
            add(A_oof, po.x - s.y, po.y - s.t, 0);
            add(A_full, pf.x - s.y, pf.y - s.t, 0);
            if (in_range) { add(A_oof_in, po.x - s.y, po.y - s.t, 0); add(A_full_in, pf.x - s.y, pf.y - s.t, 0); }

            per.add({std::to_string(s.idx), s.set.substr(6), s.rel, num(s.y, 3), num(s.t, 3),
                     jok ? num(j.y, 4) : "FAIL", jok ? num(j.t, 4) : "FAIL", axis,
                     kok ? num(k.y, 4) : "FAIL", kok ? num(k.t, 4) : "FAIL",
                     num(po.x, 4), num(po.y, 4), num(pf.x, 4), num(pf.y, 4), num(jms, 1), num(kms, 1)});
        }
        per.csv(results / "per_shot.csv");
        failed.csv(results / "failures.csv");

        Table st;
        st.head = {"추정기", "장수", "예외", "Y 평균", "Y 최대", "T 평균", "T 최대", "통과 ±0.045", "참고 ±0.09", "시간 중앙 ms", "시간 p95 ms"};
        st.add(row("joint() 전달 모듈 (QR+위아래 바)", A_joint_in));
        st.add(row("control() 전달 모듈 (QR만, Y or T)", A_control_in));
        st.add(row("기준 모델 66d, 전수 재대입", A_full_in));
        st.add(row("기준 모델 66d, 혼합 4-fold OOF", A_oof_in));
        st.csv(results / "scores_in_range.csv");

        Table st2;
        st2.head = st.head;
        st2.add(row("joint() 전달 모듈", A_joint));
        st2.add(row("control() 전달 모듈", A_control));
        st2.add(row("기준 모델 66d, 전수 재대입", A_full));
        st2.add(row("기준 모델 66d, OOF", A_oof));
        st2.csv(results / "scores_all.csv");

        Table perf;
        perf.head = {"experiment", "method", "split", "n", "mean_y", "max_y", "mean_t", "max_t", "pass_0045", "pass_009"};
        auto prow = [&](const std::string& n, const Acc& a) {
            auto r = row(n, a);
            perf.add({"A10", n, "in-range refit/OOF", r[1], r[3], r[4], r[5], r[6], r[7], r[8]});
        };
        prow("joint() delivered (QR+bars)", A_joint_in);
        prow("control() delivered (QR only, Y-or-T)", A_control_in);
        prow("baseline 66d refit-all", A_full_in);
        prow("baseline 66d OOF mixed4", A_oof_in);
        perf.csv(results / "perf_rows.csv");

        // ---- 그림 ----
        {
            std::vector<std::string> cats = {"joint()", "control()", "기준 재대입", "기준 OOF"};
            std::vector<double> v;
            for (auto* a : {&A_joint_in, &A_control_in, &A_full_in, &A_oof_in})
                v.push_back(a->n ? double(a->pass) / a->n : 0);
            hye::bar_chart("101_통과율비교.png", "A10-101 pass rate at +-0.045 (in-range only)", cats, v, "pass rate");
            std::vector<double> v2;
            for (auto* a : {&A_joint_in, &A_control_in, &A_full_in, &A_oof_in}) v2.push_back(max_abs(a->ey));
            hye::bar_chart("102_Y최대오차.png", "A10-102 max |eY| (in-range only)", cats, v2, "max |eY| [mm]", -1, kTol);
        }
        {
            hye::Chart ch("A10-103 error direction: delivered joint() vs baseline", "eY [mm]", "eT [deg]");
            std::vector<double> x1, y1, x2, y2;
            for (size_t i = 0; i < A_joint_in.ey.size(); ++i) { x1.push_back(A_joint_in.ey[i]); y1.push_back(A_joint_in.et[i]); }
            for (size_t i = 0; i < A_oof_in.ey.size(); ++i) { x2.push_back(A_oof_in.ey[i]); y2.push_back(A_oof_in.et[i]); }
            ch.scatter(x2, y2, hye::color(1), "baseline OOF", 3);
            ch.scatter(x1, y1, hye::color(0), "joint()", 3);
            ch.hline(kTol, hye::kRed); ch.hline(-kTol, hye::kRed);
            ch.vline(kTol, hye::kRed); ch.vline(-kTol, hye::kRed);
            ch.save("103_오차방향비교.png");
        }
        {
            hye::Chart ch("A10-104 |eY| sorted (in-range)", "rank", "|eY| [mm]");
            int k = 0;
            for (auto* a : {&A_joint_in, &A_control_in, &A_full_in, &A_oof_in}) {
                std::vector<double> v;
                for (double e : a->ey) v.push_back(std::abs(e));
                std::sort(v.begin(), v.end());
                std::vector<double> x(v.size());
                for (size_t i = 0; i < v.size(); ++i) x[i] = double(i);
                const char* nm[] = {"joint()", "control()", "baseline refit", "baseline OOF"};
                ch.line(x, v, hye::color(k), nm[k]);
                ++k;
            }
            ch.hline(kTol, hye::kRed);
            ch.save("104_오차분포.png");
        }

        std::ostringstream o;
        o << "# A10 결과 — 전달 모듈 대조\n\n"
          << "`실험/src/main.cpp` 가 다시 쓴다. 해석은 [보고서](보고서.md).\n\n"
          << "**공정성 주의.** `joint()`·`control()` 은 이 575장으로 학습된 모델이다. 여기 성적은 **재대입(개발) 성적**이고 "
          << "교차검증이 아니다. 기준 모델도 같은 조건으로 보려고 '전수 재대입' 줄을 함께 넣었다. "
          << "'혼합 4-fold OOF' 줄만 학습에 안 쓴 사진의 성적이다.\n\n"
          << "## 1. 관측에 무엇을 쓰나\n\n"
          << "| 모듈 | QR | 위·아래 바 | 출력 |\n|---|---|---|---|\n"
          << "| `joint()` | 3 QR × 9 표본점 × 2 = 54 | **48블록 이동량 (쓴다)** | (Y, T) 동시 |\n"
          << "| `control()` | 선택 536점 P12 (12차원) | **쓰지 않는다** | Y 또는 T 하나 |\n"
          << "| 기준 모델 (A3·A6) | QR별 아핀 18 | **48블록 (쓴다)** | (Y, T) 동시 |\n\n"
          << "## 2. 성적 (범위 안, T ≤ 1.00)\n\n" << st.md() << "\n"
          << hye::img_md("101_통과율비교.png", "pass rate") << "\n\n"
          << hye::img_md("102_Y최대오차.png", "max error") << "\n\n"
          << hye::img_md("104_오차분포.png", "error distribution") << "\n\n"
          << "## 3. 성적 (T+2.00 포함 전체 " << c.n() << "장)\n\n" << st2.md() << "\n"
          << "## 4. control() 의 축 선택\n\n"
          << "단일축 사진 " << axis_n << "장 중 축을 맞힌 것 " << axis_ok << "장 ("
          << num(axis_n ? 100.0 * axis_ok / axis_n : 0, 2) << "%).\n\n"
          << "`control()` 은 구조적으로 한 축만 반환한다. 혼합 자세에서는 **다른 축을 반드시 0 으로 내므로** "
          << "그 축의 오차가 명령값 전체가 된다. 위 표의 control 행에서 Y·T 평균이 큰 이유다.\n\n"
          << "## 5. 오차 방향\n\n" << hye::img_md("103_오차방향비교.png", "error direction") << "\n\n"
          << "## 6. 측정 실패\n\n";
        if (failed.rows.empty()) o << "없다.\n";
        else o << failed.md();
        write_text(result / fs::u8path(u8"결과.md"), o.str());
        std::cout << "끝\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "오류: " << e.what() << "\n";
        return 1;
    }
}
