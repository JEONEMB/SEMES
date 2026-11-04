// A15: 뒷벽 가로선을 "영점 기준"으로 쓴다 (ASM US7248931 방식).
//
//   ASM 특허의 발상: 측정을 로봇 절대 위치가 아니라 **같은 영상 속 고정 기준 대비 상대 거리**로 하면
//                    측정 기준 자체의 흔들림(영점 변동)이 상쇄된다.
//   우리 배치에 옮기면: 뒷벽 가로선은 판과 함께 움직이지 않는다(A4: Y 감도가 선 잡음의 2.3배).
//                      그러니 QR 변위를 center.bmp 가 아니라 **같은 사진의 선 위치**를 기준으로 재면
//                      세션·날짜 영점 변동이 빠질 수 있다.
//
//   1부 진단: 날짜가 바뀔 때 QR 과 선이 **같이** 움직이는가, QR 만 움직이는가?
//             같이 움직이면 카메라가 움직인 것 -> 선으로 보정 가능
//             QR 만 움직이면 판이 움직인 것   -> 선으로 보정 불가
//   2부 적용: 선 기준으로 바꾼 특징으로 A3 와 같은 평가를 다시 한다.
#include "cache.hpp"
#include "noise.hpp"
#include "model.hpp"
#include "hye/plot.hpp"
#include <functional>
#include <iostream>
#include <set>

using namespace a0;
namespace {

// 사진의 선 48블록 평균 이동 (세로). 뒷벽의 공통 세로 이동량.
double line_mean(const Cache& c, int shot) {
    double s = 0;
    int n = 0;
    for (int b = 0; b < kLines; ++b)
        if (c.line_ok(shot)[b]) { s += c.line(shot)[b * 2]; ++n; }
    return n ? s / n : 0;
}
// 위 선 / 아래 선을 따로. 둘이 반대로 움직이면 회전, 같이 움직이면 평행이동이다.
std::pair<double, double> line_updown(const Cache& c, int shot) {
    double su = 0, sl = 0;
    int nu = 0, nl = 0;
    for (int b = 0; b < kLines; ++b) {
        if (!c.line_ok(shot)[b]) continue;
        if (b < 24) { su += c.line(shot)[b * 2]; ++nu; }
        else { sl += c.line(shot)[b * 2]; ++nl; }
    }
    return {nu ? su / nu : 0, nl ? sl / nl : 0};
}
// 판 전체 이동 (선택점 평균)
cv::Point2d plate(const Cache& c, int shot) {
    cv::Point2d m(0, 0);
    int n = 0;
    for (int p = 0; p < kPoints; ++p)
        if (c.points[p].selected && c.v(shot)[p]) { m += c.dz(shot, p); ++n; }
    return n ? m * (1.0 / n) : cv::Point2d(0, 0);
}

}  // namespace

int main() {
    try {
        const fs::path result = fs::u8path(EXPERIMENT_DIR);
        const fs::path results = result / "results";
        fs::create_directories(results);
        hye::begin_results();
        Cache c = load(cache_dir());

        auto group_ids = [&](const std::string& set, double y, double t) {
            std::vector<int> v;
            for (auto& s : c.shots)
                if (s.set == set && std::abs(s.y - y) < 1e-9 && std::abs(s.t - t) < 1e-9 && s.measured) v.push_back(s.idx);
            return v;
        };
        auto avg = [&](const std::vector<int>& ids, std::function<double(int)> f) {
            double s = 0;
            for (int i : ids) s += f(i);
            return ids.empty() ? 0 : s / ids.size();
        };

        // ================= 1부: 진단 =================
        // 날짜가 바뀔 때 QR 과 선이 같이 움직이는가?
        Table dg;
        dg.head = {"비교", "QR dx", "QR dy", "선 평균 이동", "위 선", "아래 선", "QR dy − 선 평균", "판정"};
        auto diag = [&](const std::string& name, const std::vector<int>& a, const std::vector<int>& b) {
            if (a.empty() || b.empty()) return;
            double dx = avg(a, [&](int i) { return plate(c, i).x; }) - avg(b, [&](int i) { return plate(c, i).x; });
            double dy = avg(a, [&](int i) { return plate(c, i).y; }) - avg(b, [&](int i) { return plate(c, i).y; });
            double lm = avg(a, [&](int i) { return line_mean(c, i); }) - avg(b, [&](int i) { return line_mean(c, i); });
            double lu = avg(a, [&](int i) { return line_updown(c, i).first; }) - avg(b, [&](int i) { return line_updown(c, i).first; });
            double ll = avg(a, [&](int i) { return line_updown(c, i).second; }) - avg(b, [&](int i) { return line_updown(c, i).second; });
            std::string verdict;
            double ratio = std::abs(lm) > 1e-9 ? dy / lm : 0;
            if (std::abs(lm) < 0.02) verdict = u8"선은 안 움직임 → **판만 움직였다**";
            else if (ratio > 0.5 && ratio < 2.0) verdict = u8"같이 움직임 → **카메라가 움직였다**";
            else verdict = u8"부분적으로 같이";
            dg.add({name, num(dx, 4), num(dy, 4), num(lm, 4), num(lu, 4), num(ll, 4), num(dy - lm, 4), verdict});
        };
        diag(u8"1020 center − 1014 center", group_ids("clean_20261020", 0, 0), {});
        // 1014 center 는 기준 영상 자신이라 변위가 0 이다. 직접 비교가 된다.
        {
            auto a = group_ids("clean_20261020", 0, 0);
            if (!a.empty()) {
                double dx = avg(a, [&](int i) { return plate(c, i).x; });
                double dy = avg(a, [&](int i) { return plate(c, i).y; });
                double lm = avg(a, [&](int i) { return line_mean(c, i); });
                auto ud = std::make_pair(avg(a, [&](int i) { return line_updown(c, i).first; }),
                                         avg(a, [&](int i) { return line_updown(c, i).second; }));
                std::string verdict = std::abs(lm) < 0.02 ? u8"선은 거의 안 움직임 → **판만 움직였다**"
                                                          : u8"선도 움직임 → 카메라 쪽 성분 있음";
                dg.rows.clear();
                dg.add({u8"1020 기준위치 − 1014 center (둘 다 명령 0,0)", num(dx, 4), num(dy, 4), num(lm, 4),
                        num(ud.first, 4), num(ud.second, 4), num(dy - lm, 4), verdict});
            }
        }
        // 1020 과 1029 의 겹치는 명령 (Y+0.05, T 내삽)
        {
            auto a = group_ids("clean_20261020", 0.05, 0.10);
            auto b1 = group_ids("clean_20261029", 0.05, 0.09), b2 = group_ids("clean_20261029", 0.05, 0.11);
            if (!a.empty() && !b1.empty() && !b2.empty()) {
                auto mid = [&](std::function<double(int)> f) { return 0.5 * (avg(b1, f) + avg(b2, f)); };
                double dx = avg(a, [&](int i) { return plate(c, i).x; }) - mid([&](int i) { return plate(c, i).x; });
                double dy = avg(a, [&](int i) { return plate(c, i).y; }) - mid([&](int i) { return plate(c, i).y; });
                double lm = avg(a, [&](int i) { return line_mean(c, i); }) - mid([&](int i) { return line_mean(c, i); });
                double lu = avg(a, [&](int i) { return line_updown(c, i).first; }) - mid([&](int i) { return line_updown(c, i).first; });
                double ll = avg(a, [&](int i) { return line_updown(c, i).second; }) - mid([&](int i) { return line_updown(c, i).second; });
                dg.add({u8"1020 Y+0.05 T+0.10 − 1029 내삽", num(dx, 4), num(dy, 4), num(lm, 4), num(lu, 4), num(ll, 4),
                        num(dy - lm, 4), std::abs(lm) < 0.02 ? u8"선은 안 움직임" : u8"선도 움직임"});
            }
        }
        dg.csv(results / "drift_diagnosis.csv");

        // 세션 내 반복에서도 같은 것을 본다 (묶음 안에서 QR 과 선이 함께 흔들리는가)
        Table rp;
        rp.head = {"세트", "묶음 수", "QR dy 반복 SD", "선 평균 반복 SD", "둘의 상관"};
        for (auto set : {"clean_20261020", "clean_20261029"}) {
            std::map<int, std::vector<int>> g;
            for (auto& s : c.shots) if (s.measured && s.set == set) g[s.group].push_back(s.idx);
            std::vector<double> q, l, allq, alll;
            int ng = 0;
            for (auto& k : g) {
                if (k.second.size() < 3) continue;
                ++ng;
                std::vector<double> qy, lm;
                for (int i : k.second) { qy.push_back(plate(c, i).y); lm.push_back(line_mean(c, i)); }
                q.push_back(sd(qy));
                l.push_back(sd(lm));
                double mq = mean(qy), ml = mean(lm);
                for (size_t j = 0; j < qy.size(); ++j) { allq.push_back(qy[j] - mq); alll.push_back(lm[j] - ml); }
            }
            double mq = mean(allq), ml = mean(alll), sq = 0, sl = 0, sc = 0;
            for (size_t i = 0; i < allq.size(); ++i) {
                sq += (allq[i] - mq) * (allq[i] - mq);
                sl += (alll[i] - ml) * (alll[i] - ml);
                sc += (allq[i] - mq) * (alll[i] - ml);
            }
            rp.add({std::string(set).substr(6), std::to_string(ng), num(median(q), 5), num(median(l), 5),
                    num((sq > 0 && sl > 0) ? sc / std::sqrt(sq * sl) : 0, 3)});
        }
        rp.csv(results / "repeat_correlation.csv");

        // ================= 2부: 선 기준 특징으로 다시 평가 =================
        // V0 기준 (A3 와 같음)
        // V1 QR 의 세로 상수항에서 선 평균 이동을 뺀다 (ASM: 같은 영상 속 고정 기준 대비)
        // V2 V1 에 더해 위·아래 선의 평균도 각각 빼서 QR 세로를 두 기준에 묶는다
        // V3 QR만 18차원 (선을 전혀 안 쓴다 = control() 과 같은 처지)
        // V4 QR만 18차원 + 선 평균으로 세로 보정  <- 여기서만 ASM 보정이 "새 정보"가 된다
        // V5 QR만 18차원 + 선이 말하는 T 로 가로까지 보정
        std::vector<std::vector<double>> V0(c.n()), V1(c.n()), V2(c.n()), V3(c.n()), V4(c.n()), V5(c.n());
        for (auto& s : c.shots) {
            if (!s.measured) continue;
            V0[s.idx] = features(c, s.idx, true);
            double lm = line_mean(c, s.idx);
            auto ud = line_updown(c, s.idx);
            V1[s.idx] = V0[s.idx];
            V2[s.idx] = V0[s.idx];
            for (int q = 0; q < 3; ++q) {
                V1[s.idx][q * 6 + 3] -= lm;                       // dy 상수항
                V2[s.idx][q * 6 + 3] -= 0.5 * (ud.first + ud.second);
            }
            V3[s.idx].assign(V0[s.idx].begin(), V0[s.idx].begin() + kFeatQr);
            V4[s.idx] = V3[s.idx];
            V5[s.idx] = V3[s.idx];
            // 선이 말하는 회전: 위·아래 선의 기울기 차가 T 에 비례한다 (A4·A13)
            double tilt = ud.second - ud.first;
            for (int q = 0; q < 3; ++q) {
                V4[s.idx][q * 6 + 3] -= lm;                       // 세로만 보정 (ASM 그대로)
                V5[s.idx][q * 6 + 3] -= lm;
                V5[s.idx][q * 6 + 0] -= tilt * (80.06 / 2.528);   // 가로도 선이 말하는 T 만큼
            }
        }

        std::function<int(const Shot&)> mixed = [](const Shot& s) { return s.fold_mixed; };
        std::function<int(const Shot&)> bydate = [](const Shot& s) { return s.fold_date; };
        Table st;
        st.head = {"특징", "분할", "장수", "Y 평균", "Y 최대", "T 평균", "T 최대", "통과 ±0.045"};
        Table perf;
        perf.head = {"experiment", "method", "split", "n", "mean_y", "max_y", "mean_t", "max_t", "pass_0045", "pass_009"};
        struct V { std::string name; std::vector<std::vector<double>>* f; };
        std::vector<V> vs = {{u8"V0 기준 QR+선 66d (A3 와 같음)", &V0},
                             {u8"V1 선 평균을 뺀 QR 세로 (66d)", &V1},
                             {u8"V2 위·아래 선 평균을 뺀 QR 세로 (66d)", &V2},
                             {u8"V3 QR만 18d (선 안 씀)", &V3},
                             {u8"V4 QR만 18d + 선 평균 세로 보정", &V4},
                             {u8"V5 QR만 18d + 선 세로·가로 보정", &V5}};
        for (auto& v : vs) {
            for (int d = 0; d < 2; ++d) {
                auto o = run_oof(c, *v.f, d ? bydate : mixed, d ? 3 : 4, 0);
                std::vector<unsigned char> keep(o.shot.size(), 1);
                for (size_t i = 0; i < o.shot.size(); ++i) if (o.truth[i].y > 1.001) keep[i] = 0;
                auto sc2 = score(o, &keep);
                st.add({v.name, d ? u8"날짜 3-fold" : u8"혼합 4-fold", std::to_string(sc2.n), num(sc2.mean_y, 4),
                        num(sc2.max_y, 4), num(sc2.mean_t, 4), num(sc2.max_t, 4), num(sc2.pass, 4)});
                perf.add({"A15", v.name, d ? "date3" : "mixed4", std::to_string(sc2.n), num(sc2.mean_y, 4),
                          num(sc2.max_y, 4), num(sc2.mean_t, 4), num(sc2.max_t, 4), num(sc2.pass, 4), num(sc2.pass09, 4)});
            }
        }
        st.csv(results / "scores.csv");
        perf.csv(results / "perf_rows.csv");

        // 날짜별로도 본다 (A3 에서 1014 를 빼면 0% 였다)
        Table bd;
        bd.head = {"특징", "빠진 날짜", "장수", "Y 평균", "통과 ±0.045"};
        for (auto& v : vs) {
            auto o = run_oof(c, *v.f, bydate, 3, 0);
            for (int f = 0; f < 3; ++f) {
                std::vector<unsigned char> keep(o.shot.size(), 0);
                for (size_t i = 0; i < o.shot.size(); ++i)
                    if (o.fold[i] == f && !(o.truth[i].y > 1.001)) keep[i] = 1;
                auto sc2 = score(o, &keep);
                const char* nm[] = {"1014", "1020", "1029"};
                bd.add({v.name, nm[f], std::to_string(sc2.n), num(sc2.mean_y, 4), num(sc2.pass, 4)});
            }
        }
        bd.csv(results / "by_date.csv");

        // ================= 그림 =================
        {
            hye::Chart ch("A15-151 plate dy vs line mean shift (per shot)", "line mean shift [px]", "QR plate dy [px]");
            for (int k = 0; k < 3; ++k) {
                const char* sets[] = {"clean_20261014", "clean_20261020", "clean_20261029"};
                std::vector<double> x, y;
                for (auto& s : c.shots)
                    if (s.measured && s.set == sets[k] && s.t <= 1.001) { x.push_back(line_mean(c, s.idx)); y.push_back(plate(c, s.idx).y); }
                ch.scatter(x, y, hye::color(k), std::string(sets[k]).substr(6), 3);
            }
            ch.save("151_판세로_대_선이동.png");
        }
        {
            std::vector<std::string> cats;
            std::vector<double> v;
            for (auto& r : st.rows) { cats.push_back(r[0].substr(0, 2) + " " + (r[1][0] == u8"날"[0] ? "date" : "mixed")); v.push_back(std::stod(r[7])); }
            hye::bar_chart("152_통과율.png", "A15-152 pass rate at +-0.045", cats, v, "pass rate");
        }

        std::ostringstream o;
        o << u8"# A15 결과 — 뒷벽 가로선을 영점 기준으로 (ASM 방식)\n\n"
          << u8"`실험/src/main.cpp` 가 다시 쓴다. 해석은 [보고서](보고서.md).\n\n"
          << u8"근거: `01_주제_관련/원인분석/특허/06_US7248931_ASM_고정기준점_드리프트측정`\n\n"
          << u8"> 측정을 로봇 절대 위치가 아니라 **같은 영상 속 고정 기준 대비 상대 거리**로 하면 "
          << u8"측정 기준 자체의 흔들림이 상쇄된다.\n\n"
          << u8"## 1. 진단 — 날짜가 바뀔 때 QR 과 선이 같이 움직이는가\n\n" << dg.md() << "\n"
          << u8"**선이 안 움직였는데 QR 만 움직였다면 판이 움직인 것이다.** 그 경우 선으로는 보정할 수 없다.\n\n"
          << u8"## 2. 세션 내 반복에서도 같은가\n\n" << rp.md() << "\n"
          << hye::img_md("151_판세로_대_선이동.png", "plate dy vs line") << "\n\n"
          << u8"## 3. 선 기준 특징으로 다시 평가\n\n" << st.md() << "\n"
          << hye::img_md("152_통과율.png", "pass rate") << "\n\n"
          << u8"## 4. 날짜별\n\n" << bd.md() << "\n";
        write_text(result / fs::u8path(u8"결과.md"), o.str());
        std::cout << u8"끝\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << u8"오류: " << e.what() << "\n";
        return 1;
    }
}
