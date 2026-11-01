// =====================================================================================
// track_b.cpp — 트랙 B : QR + 배경 시차로 (Y, Θ) 추정
//
//  근거 : Sutton et al. 2008 — 2D 한 장으로는 면외/회전 성분을 분리 못 하고 깊이 정보가 있어야 함
//         → 깊이가 다른 플레이트(가까움)와 벽(멂)의 이동을 함께 씀
//  특징 : s  = QR 3개 평균 이동 (M1)                 — Y·Θ 모두 크게 반응
//         d  = s − 아래쪽 벽 4곳 평균 이동 (시차)      — Y 에 크게, Θ 에 작게 반응
//         (비교) d2 = s − 위쪽 벽 2곳 평균 이동
//  추정 : PoseModel (특징 = c + a·Y + b·Θ + q·Θ², 1차 세트 학습, 가중 Gauss–Newton 역산)
// =====================================================================================
#include "yt_common.hpp"
#include "yt_model.hpp"

#include <cmath>
#include <functional>
#include <iostream>

namespace yt {
namespace trackB {

namespace {
// m1 csv → 특징 (variant 0 : s, d / 1 : s, d, d2 / 2 : s 만 — 분리 불가 대조군)
std::vector<Obs> features(const std::vector<Sample>& ds, const CsvTable& m1, int variant) {
    std::vector<Obs> out;
    for (const auto& s : ds) {
        if (s.is_ref) continue;
        auto mean = [&](std::initializer_list<const char*> names) {
            double sum = 0; int n = 0;
            for (const char* nm : names) { const double v = num(m1, s.stem, std::string(nm) + "_dx"); if (std::isfinite(v)) { sum += v; ++n; } }
            return n ? sum / n : std::nan("");
        };
        const double sp = mean({"QR_L", "QR_C", "QR_R"});
        const double wl = mean({"wall_W1", "wall_W2", "wall_W3", "wall_W4"});
        const double wu = mean({"wall_U1", "wall_U2"});
        Obs o;
        o.stem = s.stem; o.y = s.y_mm; o.t = s.t_deg; o.set = s.set_id; o.shot = s.shot;
        if (variant == 0) o.f = {sp, sp - wl};
        else if (variant == 1) o.f = {sp, sp - wl, sp - wu};
        else o.f = {sp};
        out.push_back(o);
    }
    return out;
}
} // namespace

bool verify(const std::vector<Sample>& ds, const std::string& outDir, const std::string& m1Csv) {
    Report rep(outDir, "트랙 B (QR + 배경 시차) — 검증 결과");
    const CsvTable m1 = readCsv(m1Csv);
    if (m1.empty()) { rep.check("M1 결과", false, "m1_shifts.csv 없음 — M1 먼저 실행"); return false; }

    const std::vector<Obs> all = features(ds, m1, 0);
    std::vector<Obs> s1, s2;
    for (const auto& o : all) (o.set == 1 ? s1 : s2).push_back(o);

    PoseModel model;
    model.fit(s1);
    rep.h("B-1 학습된 감도 (1차 세트 전체)");
    {
        const auto& c = model.coef();
        rep.table({"특징", "Y 감도 (px/mm)", "Θ 감도 (px/°)", "Θ² (px/°²)", "학습 잔차 σ (px)"},
                  {{"s = QR 평균 이동", fmt("%.3f", c[0][1]), fmt("%.3f", c[0][2]), fmt("%.3f", c[0][3]), fmt("%.3f", model.sigma()[0])},
                   {"d = s − 아래 벽", fmt("%.3f", c[1][1]), fmt("%.3f", c[1][2]), fmt("%.3f", c[1][3]), fmt("%.3f", model.sigma()[1])}});
        const double tEq = c[0][1] / c[0][2];                 // 플레이트 이동이 같은 Y 1 mm ↔ Θ
        const double dd = c[1][1] * (-0.01 / tEq) + c[1][2] * 0.01;
        rep.line(fmt("같은 s 에서 Θ 만 0.01° 바꾸면 d 는 %.3f px 변함 (시차 신호). d 의 잔차 σ %.3f px 대비 %.1f 배.", std::abs(dd), model.sigma()[1],
                     std::abs(dd) / model.sigma()[1]));
    }

    rep.h("B-2 1차 세트 — 한 장 빼고 학습(LOO)");
    const auto loo = leaveOneOut(s1);
    writeEst(outDir + "/trackB_set1_loo.csv", loo);
    const Summary a1 = summarize(loo, [](const Est&) { return true; });
    const Summary a2 = summarize(loo, [](const Est& e) { return std::abs(e.t) <= 0.5; });
    const Summary a3 = summarize(loo, [](const Est& e) { return e.t == 0; });
    const Summary a4 = summarize(loo, [](const Est& e) { return e.y == 0; });
    rep.line("전체 : " + summaryRow(a1));
    rep.line("Θ 절댓값 ≤ 0.5° : " + summaryRow(a2));
    rep.line("Y 스윕 : " + summaryRow(a3));
    rep.line("Θ 스윕 : " + summaryRow(a4));

    rep.h("B-3 0.01° 계단 시험 (Θ 스윕 이웃 쌍) / 0.05 mm 계단 (Y 스윕)");
    std::vector<double> steps;
    const Stat st = stepTest(loo, 0.01, true, &steps);
    const Stat sy = stepTest(loo, 0.05, false);
    {
        Csv c(outDir + "/trackB_step_theta.csv");
        c << "dTheta_est";
        c.endl();
        for (double v : steps) { c << v; c.endl(); }
    }
    rep.table({"시험", "쌍 수", "추정 차 평균", "표준편차", "정답"},
              {{"Θ 0.01° 계단", fmt("%d", st.n), fmt("%.5f °", st.mean), fmt("%.5f °", st.sd), "0.01000 °"},
               {"Y 0.05 mm 계단", fmt("%d", sy.n), fmt("%.4f mm", sy.mean), fmt("%.4f mm", sy.sd), "0.0500 mm"}});

    rep.h("B-4 다른 날 (2차 세트, Y·Θ 동시 이동) — 1차 모델 그대로");
    const auto e2 = estimateAll(model, s2);
    writeEst(outDir + "/trackB_set2_raw.csv", e2);
    const Summary b1 = summarize(e2, [](const Est& e) { return e.t <= 0.2 + 1e-9; });
    const Summary b2 = summarize(e2, [](const Est& e) { return std::abs(e.t - 1.0) < 1e-6; });
    const Summary b3 = summarize(e2, [](const Est& e) { return std::abs(e.t - 2.0) < 1e-6; });
    rep.line("Θ ≤ 0.2° (목표 근처) : " + summaryRow(b1));
    rep.line("Θ = 1° : " + summaryRow(b2));
    rep.line("Θ = 2° (학습 범위 밖) : " + summaryRow(b3));

    rep.h("B-5 teaching 보정 이동 3회 후 (2차 세트)");
    const std::vector<std::string> teach = {"Y+1.00 T+0.10 (1)", "Y+0.05 T+1.00 (1)", "Y+0.50 T+0.10 (1)"};
    PoseModel cal = model;
    {
        std::vector<Obs> known, rest;
        for (const auto& o : s2) {
            bool k = false;
            for (const auto& t : teach) if (o.stem == t) k = true;
            (k ? known : rest).push_back(o);
        }
        cal.calibrate(known);
        const auto e3 = estimateAll(cal, rest);
        writeEst(outDir + "/trackB_set2_teach.csv", e3);
        rep.line("보정 이동 : " + teach[0] + ", " + teach[1] + ", " + teach[2] + " (평가에서 제외)");
        const Summary c1 = summarize(e3, [](const Est& e) { return e.t <= 0.2 + 1e-9; });
        const Summary c2 = summarize(e3, [](const Est& e) { return std::abs(e.t - 1.0) < 1e-6; });
        const Summary c3 = summarize(e3, [](const Est& e) { return std::abs(e.t - 2.0) < 1e-6; });
        rep.line("Θ ≤ 0.2° : " + summaryRow(c1));
        rep.line("Θ = 1° : " + summaryRow(c2));
        rep.line("Θ = 2° : " + summaryRow(c3));
    }

    rep.h("B-6 대조군 — 배경 없이 s 하나만 쓰면");
    {
        const auto only = features(ds, m1, 2);
        std::vector<Obs> o1;
        for (const auto& o : only) if (o.set == 1) o1.push_back(o);
        rep.line("특징이 1개(s)뿐이면 Y·Θ 두 미지수를 풀 수 없음 → 감도 행렬의 계수가 1 (분리 불가). 트랙 B 의 분리 능력은 배경 시차 d 에서 나옴.");
        // 3 특징 (위쪽 벽 추가) 비교
        const auto three = features(ds, m1, 1);
        std::vector<Obs> t1;
        for (const auto& o : three) if (o.set == 1) t1.push_back(o);
        const auto l3 = leaveOneOut(t1);
        const Stat s3 = stepTest(l3, 0.01, true);
        rep.line("(비교) 위쪽 벽까지 3 특징 : " + summaryRow(summarize(l3, [](const Est&) { return true; })) +
                 fmt(" | 0.01° 계단 평균 %.5f σ %.5f", s3.mean, s3.sd));
    }

    // 판정 (구현 + 사양)
    rep.h("판정");
    rep.check("0.01° 계단 구분 (평균 0.0100 ± 0.0010°, σ ≤ 0.003°)", std::abs(st.mean - 0.01) <= 0.001 && st.sd <= 0.003,
              fmt("평균 %.5f°, σ %.5f° (%d 쌍)", st.mean, st.sd, st.n));
    rep.check("1차 LOO Y ±90 µm", a1.y.pass >= int(0.95 * a1.n), fmt("%d / %d", a1.y.pass, a1.n));
    rep.check("1차 LOO |Θ| ≤ 0.5° 에서 ±0.01° 95 % 이상", a2.t.pass >= int(std::ceil(0.95 * a2.n)), fmt("%d / %d", a2.t.pass, a2.n));
    rep.check("다른 날 목표 근처(Θ ≤ 0.2°) ±0.01° 95 % 이상", b1.t.pass >= int(std::ceil(0.95 * b1.n)), fmt("%d / %d", b1.t.pass, b1.n));
    rep.line("Θ 1~2° 구간 오차는 반복 보정의 초반 이동에 해당 (판정 대상 아님). 반복 수렴 여부는 시뮬레이터에서 확인 예정.");
    return rep.allOk();
}

} // namespace trackB
} // namespace yt
