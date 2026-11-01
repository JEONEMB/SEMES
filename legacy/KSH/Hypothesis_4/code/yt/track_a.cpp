// =====================================================================================
// track_a.cpp — 트랙 A : QR 3개만으로 (Y, Θ) 추정
//
//  근거 : Sutton et al. 2008 — 카메라가 돌면(팬) 영상 안에 위치에 따라 다른 배율(가짜 변형률)이 생김
//         Y 이동은 거의 균일 이동 → 좌우 QR 의 가로 변형률 차 ε_R − ε_L 이 Θ 를 알려 주는 신호
//  특징 : s   = QR 3개 평균 이동 (M1)
//         dEx = ε_R − ε_L (M2 IC-GN, B-spline + 가우시안 σ 1.0 — 채택)
//         dS  = Gabor 위상 기울기 차 s_R − s_L (M4, px / 100 px)
//  주 조합 : {s, dEx} (M2 채택안, 사전 지정). 나머지 조합은 비교용
//  추정 : PoseModel (트랙 B 와 같은 모델·평가 절차)
// =====================================================================================
#include "yt_common.hpp"
#include "yt_model.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <map>

namespace yt {
namespace trackA {

namespace {
// 조합 : 이름과 특징 추출 함수
struct Combo {
    std::string name;
    std::function<std::vector<double>(const std::string&)> f;
};

std::vector<Obs> build(const std::vector<Sample>& ds, const Combo& c) {
    std::vector<Obs> out;
    for (const auto& s : ds) {
        if (s.is_ref) continue;
        Obs o;
        o.stem = s.stem; o.y = s.y_mm; o.t = s.t_deg; o.set = s.set_id; o.shot = s.shot;
        o.f = c.f(s.stem);
        out.push_back(o);
    }
    return out;
}

void split(const std::vector<Obs>& all, std::vector<Obs>& s1, std::vector<Obs>& s2) {
    for (const auto& o : all) (o.set == 1 ? s1 : s2).push_back(o);
}

// 같은 라벨의 반복 촬영 쌍 : 정답 차는 0 → 추정 Θ 차 = 순수 재현성 오차. |Δs| 로 '정지 쌍 / 이동 쌍' 구분
struct PairBin { std::string name; double lo, hi; std::vector<double> dt, dy; };
} // namespace

bool verify(const std::vector<Sample>& ds, const std::string& outDir, const std::string& m1Csv, const std::string& m2Csv,
            const std::string& m4Csv) {
    Report rep(outDir, "트랙 A (QR 3개만) — 검증 결과");
    const CsvTable m1 = readCsv(m1Csv), m2 = readCsv(m2Csv), m4 = readCsv(m4Csv);
    if (m1.empty() || m2.empty() || m4.empty()) {
        rep.check("선행 결과", false, "m1_shifts / m2_affine / m4_slopes 중 없는 것이 있음 — M1·M2·M4 먼저 실행");
        return false;
    }
    auto sOf = [&](const std::string& st) {
        double sum = 0; int n = 0;
        for (const char* q : {"QR_L_dx", "QR_C_dx", "QR_R_dx"}) { const double v = num(m1, st, q); if (std::isfinite(v)) { sum += v; ++n; } }
        return n == 3 ? sum / 3 : std::nan("");
    };
    const std::vector<Combo> combos = {
        {"{s, dEx} M2 채택 (주)", [&](const std::string& st) { return std::vector<double>{sOf(st), num(m2, st, "dEx")}; }},
        {"{s, dEx, dS} M2+M4", [&](const std::string& st) { return std::vector<double>{sOf(st), num(m2, st, "dEx"), num(m4, st, "dS")}; }},
        {"{s, dS} M4 Gabor", [&](const std::string& st) { return std::vector<double>{sOf(st), num(m4, st, "dS")}; }},
        {"{s, dEx_bs0} B-spline 만", [&](const std::string& st) { return std::vector<double>{sOf(st), num(m2, st, "dEx_bs0")}; }},
        {"{s, dEx_lin} 선형 보간", [&](const std::string& st) { return std::vector<double>{sOf(st), num(m2, st, "dEx_lin")}; }},
        {"{s, dEx, dEy}", [&](const std::string& st) { return std::vector<double>{sOf(st), num(m2, st, "dEx"), num(m2, st, "dEy")}; }},
    };

    // ---------------- A-1 감도 : 두 독립 알고리즘(M2, M4)의 Θ 신호가 같은 값을 주는가 ----------------
    rep.h("A-1 Θ 신호의 감도 (1차 세트 학습) — 독립 알고리즘 일치 확인");
    std::vector<Obs> s1, s2;
    split(build(ds, combos[1]), s1, s2);
    PoseModel full;
    full.fit(s1);
    const auto& c = full.coef();
    const double sensM2 = c[1][2], sensM4 = c[2][2] / 100.0;   // dS 는 px / 100 px → 변형률로 환산
    rep.table({"특징", "Y 감도 (/mm)", "Θ 감도 (/°)", "잔차 σ", "Θ 환산 잡음 σ/감도 (°)"},
              {{"s (px)", fmt("%.4f", c[0][1]), fmt("%.4f", c[0][2]), fmt("%.4f", full.sigma()[0]), "—"},
               {"dEx = ε_R − ε_L", fmt("%.3e", c[1][1]), fmt("%.4e", c[1][2]), fmt("%.3e", full.sigma()[1]),
                fmt("%.4f", full.sigma()[1] / std::abs(c[1][2]))},
               {"dS (px/100px)", fmt("%.4f", c[2][1]), fmt("%.4f", c[2][2]), fmt("%.4f", full.sigma()[2]),
                fmt("%.4f", full.sigma()[2] / std::abs(c[2][2]))}});
    const double agree = std::abs(sensM2 - sensM4) / std::abs(sensM2);
    rep.line(fmt("M2 (IC-GN) Θ 감도 %.4e /°, M4 (Gabor) %.4e /° → 상대 차 %.2f %%.", sensM2, sensM4, 100 * agree));
    rep.line(fmt("한 장에서 Θ 를 0.01° 로 읽으려면 특징 잡음이 감도×0.01 = %.2e 보다 작아야 함. 현재 dEx 잡음 %.2e (%.1f 배 큼).",
                 std::abs(c[1][2]) * 0.01, full.sigma()[1], full.sigma()[1] / (std::abs(c[1][2]) * 0.01)));

    // ---------------- A-2 조합별 LOO / 계단 / 다른 날 ----------------
    rep.h("A-2 조합별 결과 (1차 LOO · 0.01° 계단 · 2차 목표 근처)");
    std::vector<std::vector<std::string>> rows;
    Summary mainLoo, mainNear, mainTeach;
    Stat mainStep;
    std::vector<Est> mainLooE, main2E;
    for (size_t k = 0; k < combos.size(); ++k) {
        std::vector<Obs> a1, a2;
        split(build(ds, combos[k]), a1, a2);
        const auto loo = leaveOneOut(a1);
        PoseModel m;
        m.fit(a1);
        const auto e2 = estimateAll(m, a2);
        const Summary L = summarize(loo, [](const Est&) { return true; });
        const Summary N = summarize(e2, [](const Est& e) { return e.t <= 0.2 + 1e-9; });
        const Stat st = stepTest(loo, 0.01, true);
        rows.push_back({combos[k].name, fmt("%.4f", L.t.rms), fmt("%d/%d", L.t.pass, L.n), fmt("%.1f", L.y.rms), fmt("%.5f", st.mean),
                        fmt("%.4f", st.sd), fmt("%.4f", N.t.rms), fmt("%d/%d", N.t.pass, N.n)});
        if (k == 0) { mainLoo = L; mainNear = N; mainStep = st; mainLooE = loo; main2E = e2; }
    }
    rep.table({"조합", "1차 LOO Θ RMS (°)", "±0.01° 통과", "Y RMS (µm)", "0.01° 계단 평균 (°)", "계단 σ (°)", "2차 Θ≤0.2° RMS (°)",
               "±0.01° 통과"},
              rows);
    writeEst(outDir + "/trackA_set1_loo.csv", mainLooE);
    writeEst(outDir + "/trackA_set2_raw.csv", main2E);

    // ---------------- A-3 주 조합 세부 ----------------
    rep.h("A-3 주 조합 {s, dEx} 세부");
    rep.line("1차 LOO 전체 : " + summaryRow(mainLoo));
    rep.line("1차 LOO Θ 절댓값 ≤ 0.5° : " + summaryRow(summarize(mainLooE, [](const Est& e) { return std::abs(e.t) <= 0.5; })));
    rep.line("1차 LOO Y 스윕 : " + summaryRow(summarize(mainLooE, [](const Est& e) { return e.t == 0; })));
    rep.line("2차 Θ ≤ 0.2° : " + summaryRow(mainNear));
    rep.line("2차 Θ = 1° : " + summaryRow(summarize(main2E, [](const Est& e) { return std::abs(e.t - 1.0) < 1e-6; })));
    rep.line("2차 Θ = 2° : " + summaryRow(summarize(main2E, [](const Est& e) { return std::abs(e.t - 2.0) < 1e-6; })));
    {
        std::vector<Obs> a1, a2, known, rest;
        split(build(ds, combos[0]), a1, a2);
        const std::vector<std::string> teach = {"Y+1.00 T+0.10 (1)", "Y+0.05 T+1.00 (1)", "Y+0.50 T+0.10 (1)"};
        for (const auto& o : a2) (std::find(teach.begin(), teach.end(), o.stem) != teach.end() ? known : rest).push_back(o);
        PoseModel m;
        m.fit(a1);
        m.calibrate(known);
        const auto e3 = estimateAll(m, rest);
        writeEst(outDir + "/trackA_set2_teach.csv", e3);
        mainTeach = summarize(e3, [](const Est& e) { return e.t <= 0.2 + 1e-9; });
        rep.line("teaching 보정 3회 후 2차 Θ ≤ 0.2° : " + summaryRow(mainTeach));
    }

    // ---------------- A-4 반복 촬영 쌍 : 정지 vs 이동 ----------------
    rep.h("A-4 같은 라벨 반복 촬영 쌍 (2차) — 정지 쌍 vs 플레이트가 움직인 쌍");
    {
        std::map<std::pair<long, long>, std::vector<const Est*>> g;
        for (const auto& e : main2E)
            if (e.ok) g[{std::lround(e.y * 1000), std::lround(e.t * 1000)}].push_back(&e);
        std::vector<PairBin> bins = {{"|Δs| < 0.03 px (정지)", 0, 0.03, {}, {}},
                                     {"0.03 ~ 0.3 px", 0.03, 0.3, {}, {}},
                                     {"≥ 0.3 px (이동)", 0.3, 1e9, {}, {}}};
        Csv csv(outDir + "/trackA_repeat_pairs.csv");
        csv << "stem_a" << "stem_b" << "abs_ds_px" << "dTheta_est" << "dY_est_um";
        csv.endl();
        for (const auto& kv : g)
            for (size_t i = 0; i < kv.second.size(); ++i)
                for (size_t j = i + 1; j < kv.second.size(); ++j) {
                    const Est& a = *kv.second[i];
                    const Est& b = *kv.second[j];
                    const double das = std::abs(sOf(a.stem) - sOf(b.stem));
                    const double dt = b.te - a.te, dy = (b.ye - a.ye) * 1000;
                    csv << a.stem << b.stem << das << dt << dy;
                    csv.endl();
                    for (auto& bn : bins)
                        if (das >= bn.lo && das < bn.hi) { bn.dt.push_back(dt); bn.dy.push_back(dy); }
                }
        std::vector<std::vector<std::string>> r;
        for (const auto& bn : bins) {
            const Stat a = stat(bn.dt, 0.01), b = stat(bn.dy, 90);
            r.push_back({bn.name, fmt("%d", a.n), fmt("%.4f", a.rms), fmt("%.4f", a.maxAbs), fmt("%.1f", b.rms)});
        }
        rep.table({"쌍 종류", "쌍 수", "추정 Θ 차 RMS (°)", "최대 (°)", "추정 Y 차 RMS (µm)"}, r);
        rep.line("정답 차가 0 인 쌍. 정지 쌍은 촬영 잡음만, 이동 쌍은 'QR 이 화면의 다른 화소 위에 놓일 때 생기는 오차'까지 포함.");
    }

    // ---------------- A-5 기준 영상과의 거리별 오차 ----------------
    rep.h("A-5 기준(center)에서 얼마나 떨어진 사진인가 |s| 별 Θ 오차 (1차 LOO + 2차)");
    {
        struct Bin { const char* name; double lo, hi; std::vector<double> e; };
        std::vector<Bin> bins = {{"|s| < 1 px", 0, 1, {}}, {"1 ~ 5 px", 1, 5, {}}, {"5 ~ 20 px", 5, 20, {}}, {"≥ 20 px", 20, 1e9, {}}};
        auto add = [&](const std::vector<Est>& v) {
            for (const auto& e : v) {
                if (!e.ok) continue;
                const double as = std::abs(sOf(e.stem));
                for (auto& b : bins) if (as >= b.lo && as < b.hi) b.e.push_back(e.te - e.t);
            }
        };
        add(mainLooE);
        add(main2E);
        std::vector<std::vector<std::string>> r;
        for (const auto& b : bins) {
            const Stat st = stat(b.e, 0.01);
            r.push_back({b.name, fmt("%d", st.n), fmt("%.4f", st.rms), fmt("%d/%d", st.pass, st.n)});
        }
        rep.table({"|s| (기준 대비 플레이트 이동)", "장 수", "Θ 오차 RMS (°)", "±0.01° 통과"}, r);
        rep.line("반복 보정의 마지막 단계는 목표(기준) 근처 = |s| 가 작은 구간. 이 구간의 오차가 트랙 A 의 실제 판정 능력.");
    }

    // ---------------- 판정 ----------------
    rep.h("판정 — 구현 검증 (알고리즘이 설계대로 동작하는가)");
    rep.check("M2·M4 두 독립 알고리즘의 Θ 감도 일치 (상대 차 ≤ 5 %)", agree <= 0.05, fmt("%.2f %%", 100 * agree));
    rep.check("0.01° 계단 추정이 치우치지 않음 (평균 0.0100 ± 0.0020°)", std::abs(mainStep.mean - 0.01) <= 0.002,
              fmt("평균 %.5f° (%d 쌍)", mainStep.mean, mainStep.n));
    // Y 오차가 Θ 오차의 전파로 설명되는가 : s 고정이면 ΔY = −(b_s / a_s)·ΔΘ  (Y 를 따로 망가뜨리는 버그가 없음을 확인)
    const double kYperT_um = 1000.0 * c[0][2] / c[0][1];
    const double yPred = kYperT_um * mainLoo.t.rms;
    rep.check("Y 오차 = Θ 오차 전파로 설명됨 (예측 대비 ±15 %)", std::abs(mainLoo.y.rms - yPred) <= 0.15 * yPred,
              fmt("실측 Y RMS %.1f µm, 예측 %.2f µm/° × %.4f° = %.1f µm", mainLoo.y.rms, kYperT_um, mainLoo.t.rms, yPred));
    rep.h("판정 — 사양 달성 (한 장 Θ ±0.01°, Y ±90 µm)");
    rep.spec("1차 LOO Y ±90 µm 95 % 이상", mainLoo.y.pass >= int(std::ceil(0.95 * mainLoo.n)), fmt("%d / %d", mainLoo.y.pass, mainLoo.n));
    rep.spec("0.01° 계단 σ ≤ 0.003°", mainStep.sd <= 0.003, fmt("σ %.4f°", mainStep.sd));
    rep.spec("1차 LOO ±0.01° 95 % 이상", mainLoo.t.pass >= int(std::ceil(0.95 * mainLoo.n)), fmt("%d / %d", mainLoo.t.pass, mainLoo.n));
    rep.spec("다른 날 Θ ≤ 0.2° ±0.01° 95 % 이상", mainNear.t.pass >= int(std::ceil(0.95 * mainNear.n)), fmt("%d / %d", mainNear.t.pass, mainNear.n));
    rep.spec("teaching 후 Θ ≤ 0.2° ±0.01° 95 % 이상", mainTeach.t.pass >= int(std::ceil(0.95 * mainTeach.n)),
             fmt("%d / %d", mainTeach.t.pass, mainTeach.n));
    return rep.allOk();
}

} // namespace trackA
} // namespace yt
