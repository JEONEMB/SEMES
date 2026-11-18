// =====================================================================================
// slides_grid.cpp  -  grid 묶음 (2차: Y, Θ 조합을 여러 번 촬영) 평가 슬라이드
//   기준 묶음(--ref, 1차) 의 summary.yml 에서 기울기를 가져와 2차 이동을 예측하고 측정과 비교한다.
//
// + notes.md -> 글 카드 슬라이드 (sweep, grid 공통)
// =====================================================================================
#include "slides_common.hpp"

#include <numeric>
#include <set>

namespace hw {

namespace {

struct Condition {
    std::string name;          // "Y+0.05 T+0.01"
    double y = 0, t = 0;
    std::vector<size_t> idx;   // 반복 사진
};

std::vector<Condition> conditions(const DataSet &ds) {
    std::map<std::pair<double, double>, Condition> by;
    for (size_t i = 0; i < ds.shots.size(); ++i) {
        const Shot &s = ds.shots[i];
        if (s.center) continue;
        Condition &c = by[{s.y, s.t}];
        c.name = s.group; c.y = s.y; c.t = s.t;
        c.idx.push_back(i);
    }
    std::vector<Condition> out;
    for (auto &[k, c] : by) out.push_back(c);
    return out;
}

double meanOf(const Context &ctx, const std::vector<size_t> &idx, Method m, bool dy) {
    std::vector<double> v;
    for (size_t i : idx) v.push_back(dy ? shotDy(ctx.m.shots[i], m) : shotDx(ctx.m.shots[i], m));
    return mean(v);
}
double sdOf(const Context &ctx, const std::vector<size_t> &idx, Method m, bool dy) {
    std::vector<double> v;
    for (size_t i : idx) v.push_back(dy ? shotDy(ctx.m.shots[i], m) : shotDx(ctx.m.shots[i], m));
    return stdev(v);
}

}  // namespace

void buildGridDeck(Context &ctx) {
    const DataSet &ds = ctx.ds;
    const auto conds = conditions(ds);
    std::set<double> ys, tsv;
    for (const auto &c : conds) { ys.insert(c.y); tsv.insert(c.t); }
    const double totalMin = ds.shots.back().minutes;

    // ---------------------------------------------------------------- 1. 표지
    coverSlide(ctx, "자동 생성 (hw_eda) · 촬영 " + ds.shots.front().takenAt.substr(0, 10) + (ctx.ref ? " · 기준 " + ctx.refName : ""));

    // ---------------------------------------------------------------- 2. 기초 정보
    {
        Slide &s = contentSlide(ctx, "① 데이터", "기초 정보");
        const fs::path thumb = centerThumbnail(ctx);
        s.picture(thumb, 0.8, 1.95, 5.78, 5.78 * 9 / 16);
        caption(s, 0.8, 1.95 + 5.78 * 9 / 16 + 0.05, 5.78, "기준 사진 " + ds.shots[ds.centers[0]].rel + ", 파랑 = QR 라벨 3개");
        // 조합표: 행 = Y, 열 = Θ, 칸 = 장수
        std::vector<std::string> header{"Y \\ Θ"};
        for (double t : tsv) header.push_back(signedNum(t, 2));
        std::vector<std::vector<std::string>> rows;
        for (double y : ys) {
            std::vector<std::string> r{signedNum(y, 2)};
            for (double t : tsv) {
                auto it = std::find_if(conds.begin(), conds.end(), [&](const Condition &c) { return c.y == y && c.t == t; });
                r.push_back(it == conds.end() ? "" : std::to_string(it->idx.size()));
            }
            rows.push_back(r);
        }
        std::vector<double> cx, cw;
        const double colW = std::min(0.75, 4.7 / tsv.size());
        cx.push_back(0.8); cw.push_back(1.0);
        for (size_t k = 0; k < tsv.size(); ++k) { cx.push_back(1.8 + k * colW); cw.push_back(colW); }
        table(s, header, rows, cx, cw, 5.5, 0.2, 8);
        int reps = 0;
        for (const auto &c : conds) reps = std::max(reps, static_cast<int>(c.idx.size()));
        kvTable(s, {{"장수", {std::to_string(ds.shots.size()) + "장 (center " + std::to_string(ds.centers.size()) + ", 조합 " + std::to_string(conds.size()) + "개 × 최대 " + std::to_string(reps) + "장)"}},
                    {"사진 형식", {std::to_string(ctx.m.imageSize.width) + " × " + std::to_string(ctx.m.imageSize.height) + "px, 8bit",
                                 ctx.m.grayAsColor ? "3채널 값이 모두 같음 = 흑백을 컬러로 저장" : std::to_string(ctx.m.channels) + "채널"}},
                    {"구성", {"Y 와 Θ 를 같이 움직인 조합을 반복 촬영", "Y " + std::to_string(ys.size()) + "가지 × Θ " + std::to_string(tsv.size()) + "가지 중 일부"}},
                    {"명령값 범위", {"Y " + signedNum(*ys.begin(), 2) + " ~ " + signedNum(*ys.rbegin(), 2) + "mm, Θ " + signedNum(*tsv.begin(), 2) + " ~ " + signedNum(*tsv.rbegin(), 2)}},
                    {"촬영 시간", {"전체 " + num(totalMin, 1) + "분, 첫 사진 " + ds.shots.front().takenAt}}});
        ctx.sum.set("dataset.shots", static_cast<double>(ds.shots.size()), 0);
        ctx.sum.set("dataset.conditions", static_cast<double>(conds.size()), 0);
        ctx.sum.set("dataset.minutes_total", totalMin, 2);
    }

    // ---------------------------------------------------------------- 3~5. 기준 위치
    referenceSlides(ctx);

    // ---------------------------------------------------------------- 6. center 반복
    {
        Slide &s = contentSlide(ctx, "③ 반복성", "기준 사진 반복 – center " + std::to_string(ds.centers.size()) + "장");
        std::vector<std::vector<std::string>> rows;
        std::vector<double> spread;
        for (size_t i : ds.centers) {
            const auto &mm = ctx.m.shots[i];
            rows.push_back({ds.shots[i].rel, ds.shots[i].takenAt.substr(11), num(mm.cornerDx, 3), num(mm.cornerDy, 3), num(mm.eccDx, 3), num(mm.eccDy, 3)});
            spread.push_back(mm.cornerDx);
        }
        table(s, {"사진", "촬영 시각", "특징점 가로", "특징점 세로", "ECC 가로", "ECC 세로"}, rows, {0.8, 3.6, 5.3, 7.0, 8.7, 10.4}, {2.8, 1.7, 1.7, 1.7, 1.7, 1.7}, 2.0, 0.4, 11);
        const double sp = *std::max_element(spread.begin(), spread.end()) - *std::min_element(spread.begin(), spread.end());
        std::vector<Para> ps{para("· 첫 center 사진을 기준(0)으로, 나머지 center 사진이 얼마나 움직였는지 (px)", 11),
                             para("· 같은 '기준 위치' 명령인데 가로로 최대 " + num(sp, 3) + "px 차이" +
                                      (ctx.ref ? " = " + num(sp / ctx.ref->get("corner.slope_y_mean") * 1000, 1) + "µm (1차 Y 기울기로 환산)" : ""), 11),
                             para("· 정지 상태 반복 정밀도와 원점 복귀 오차가 섞인 값. 사진 사이 시간 간격이 길수록 커질 수 있음", 11)};
        s.text(0.8, 2.6 + 0.4 * ds.centers.size(), 11.7, 1.5, ps);
        ctx.sum.set("repeat.center_spread_dx_px", sp, 4);
    }

    // ---------------------------------------------------------------- 7. 같은 조합 반복 흔들림
    std::vector<double> sdC, sdE, sdCy;
    for (const auto &c : conds)
        if (c.idx.size() >= 2) { sdC.push_back(sdOf(ctx, c.idx, Method::Corner, false)); sdE.push_back(sdOf(ctx, c.idx, Method::Ecc, false)); sdCy.push_back(sdOf(ctx, c.idx, Method::Corner, true)); }
    {
        Slide &s = contentSlide(ctx, "③ 반복성", "같은 조합 반복 촬영 – 흔들림");
        ChartSpec cs;
        for (const auto &c : conds) cs.categories.push_back(c.name);
        Series a{"특징점 가로", color::accent}, b{"ECC 가로", color::orange}, d{"특징점 세로", color::teal};
        for (const auto &c : conds) {
            a.y.push_back(sdOf(ctx, c.idx, Method::Corner, false));
            b.y.push_back(sdOf(ctx, c.idx, Method::Ecc, false));
            d.y.push_back(sdOf(ctx, c.idx, Method::Corner, true));
        }
        cs.series = {a, b, d};
        cs.horizontal = true; cs.fontSize = 7; cs.yTitle = "같은 조합 " + std::to_string(conds.front().idx.size()) + "장의 σ (px)"; cs.numFmtY = "0.000";
        s.bars(cs, 0.8, 1.9, 6.0, 4.6);
        std::vector<KvRow> rows{{"방법", {"조합마다 반복 사진의 이동 σ", "같은 명령을 다시 찍었을 때 흔들림"}},
                                {"특징점 가로", {"중앙값 " + num(median(sdC), 4) + "px, 최대 " + num(maxAbs(sdC), 4) + "px"}},
                                {"ECC 가로", {"중앙값 " + num(median(sdE), 4) + "px, 최대 " + num(maxAbs(sdE), 4) + "px"}},
                                {"특징점 세로", {"중앙값 " + num(median(sdCy), 4) + "px"}}};
        if (ctx.ref)
            rows.push_back({"1차와 비교", {"1차 사진 공통 흔들림 " + num(ctx.ref->get("noise.common_sd_min_px"), 3) + " ~ " + num(ctx.ref->get("noise.common_sd_max_px"), 3) + "px",
                                          "(1차는 연속 촬영에서 곡선을 뺀 값)"}});
        kvTable(s, rows);
        ctx.sum.set("repeat.condition_sd_corner_median_px", median(sdC), 5);
        ctx.sum.set("repeat.condition_sd_corner_max_px", maxAbs(sdC), 5);
        ctx.sum.set("repeat.condition_sd_ecc_median_px", median(sdE), 5);
    }

    // ---------------------------------------------------------------- 8~9. 기준 모델로 예측 vs 측정
    if (ctx.ref) {
        const double kY = ctx.ref->get("corner.slope_y_mean"), kT = ctx.ref->get("corner.slope_t_plus");
        const double kTm = ctx.ref->get("corner.slope_t_minus"), vert = ctx.ref->get("corner.dy_per_dx");
        std::vector<double> pred, meas, resid;
        Series pts{"조합 (특징점, 반복 평균)", color::accent}, line{"예측 = 측정", color::ink, {}, {}, true, false, 0, true};
        double lo = 1e9, hi = -1e9;
        for (const auto &c : conds) {
            const double p = kY * c.y + (c.t >= 0 ? kT : kTm) * c.t;
            const double mv = meanOf(ctx, c.idx, Method::Corner, false);
            pred.push_back(p); meas.push_back(mv); resid.push_back(mv - p);
            pts.x.push_back(p); pts.y.push_back(mv);
            lo = std::min({lo, p, mv}); hi = std::max({hi, p, mv});
        }
        line.x = {lo, hi}; line.y = {lo, hi};
        const double rms = std::sqrt(std::inner_product(resid.begin(), resid.end(), resid.begin(), 0.0) / resid.size());
        const double tMaxRef = 1.0;
        {
            Slide &s = contentSlide(ctx, "④ 1차 모델로 평가", "1차 기울기로 예측한 이동 vs 측정");
            ChartSpec cs;
            cs.series = {pts, line};
            cs.xTitle = "예측 가로 이동 (px) = " + num(kY, 2) + "×Y + " + num(kT, 2) + "×Θ";
            cs.yTitle = "측정 가로 이동 (px)"; cs.fontSize = 9;
            s.scatter(cs, 0.8, 1.9, 6.0, 4.5);
            caption(s, 0.8, 6.45, 6.0, "center1 기준. 1차(" + ctx.refName + ") 특징점 기울기를 그대로 씀. 0점 보정 없음");
            const double fitSlope = fitLine(pred, meas).slope;
            kvTable(s, {{"모델", {"가로 이동 = kY × Y + kΘ × Θ", "kY " + num(kY, 3) + "px/mm, kΘ " + num(kT, 3) + "px (1차)"}},
                        {"차이 (측정 - 예측)", {"RMS " + num(rms, 3) + "px, 최대 " + num(maxAbs(resid), 3) + "px",
                                              "Y 로 환산 RMS " + num(rms / kY * 1000, 0) + "µm"}},
                        {"기울기 맞음", {"측정 / 예측 기울기 " + num(fitSlope, 4) + " (1 이면 같음)"}},
                        {"범위 주의", {"2차 Θ 최대 " + num(*tsv.rbegin(), 2) + ", 1차는 ±" + num(tMaxRef, 1) + " 까지만 측정", "범위 밖은 1차 직선을 늘려 쓴 예측"}},
                        {"Y·Θ 구분", {"가로 이동 하나로는 Y 와 Θ 를 나눌 수 없음", "예측이 맞아도 두 값의 합만 맞는 것"}}});
            ctx.sum.set("model.k_y", kY, 4);
            ctx.sum.set("model.k_t", kT, 4);
            ctx.sum.set("model.resid_rms_px", rms, 4);
            ctx.sum.set("model.resid_max_px", maxAbs(resid), 4);
            ctx.sum.set("model.meas_over_pred", fitSlope, 5);
        }
        {
            Slide &s = contentSlide(ctx, "④ 1차 모델로 평가", "세로 이동 – 1차 비율과 비교");
            ChartSpec cs;
            Series a{"조합 (특징점, 반복 평균)", color::accent};
            double xm = 0;
            for (const auto &c : conds) { a.x.push_back(meanOf(ctx, c.idx, Method::Corner, false)); a.y.push_back(meanOf(ctx, c.idx, Method::Corner, true)); xm = std::max(xm, std::fabs(a.x.back())); }
            Series b{"1차 비율 " + num(100 * vert, 2) + "%", color::ink, {0, xm}, {0, vert * xm}, true, false, 0, true};
            cs.series = {a, b};
            cs.xTitle = "가로 이동 (px)"; cs.yTitle = "세로 이동 (px)"; cs.fontSize = 9; cs.numFmtY = "0.00";
            s.scatter(cs, 0.8, 1.9, 6.0, 4.5);
            std::vector<double> dx, dy;
            for (const auto &c : conds) { dx.push_back(meanOf(ctx, c.idx, Method::Corner, false)); dy.push_back(meanOf(ctx, c.idx, Method::Corner, true)); }
            const LineFit lf = fitLine(dx, dy);
            kvTable(s, {{"2차 비율", {"세로 / 가로 = " + num(100 * lf.slope, 3) + "% (" + num(std::atan(lf.slope) * 180 / CV_PI, 2) + "°)"}},
                        {"1차 비율", {num(100 * vert, 3) + "%"}},
                        {"직선 잔차", {num(lf.residSd, 4) + "px (σ)"}},
                        {"뜻", {"세로 이동이 가로 이동에 비례하면", "Y·Θ 를 나누는 단서가 되지 못함"}}});
            ctx.sum.set("vertical.dy_per_dx", lf.slope, 6);
            ctx.sum.set("vertical.resid_sd_px", lf.residSd, 5);
        }
        // ---------------------------------------------------------------- 10. 조합별 결과표
        {
            Slide &s = contentSlide(ctx, "④ 1차 모델로 평가", "조합별 결과");
            std::vector<std::vector<std::string>> rows;
            for (size_t k = 0; k < conds.size(); ++k) {
                const auto &c = conds[k];
                rows.push_back({c.name, std::to_string(c.idx.size()), num(pred[k], 2), num(meas[k], 2), signedNum(resid[k], 3),
                                num(sdOf(ctx, c.idx, Method::Corner, false), 4), num(meanOf(ctx, c.idx, Method::Ecc, false), 2)});
            }
            const double rowH = std::min(0.24, 4.9 / rows.size());
            table(s, {"조합", "장수", "예측 가로", "측정 가로", "차이", "반복 σ", "ECC 가로"}, rows, {0.8, 2.9, 3.8, 5.4, 7.0, 8.6, 10.2}, {2.1, 0.9, 1.6, 1.6, 1.6, 1.6, 1.6}, 1.85, rowH, 9);
        }
        Csv f(ctx.out / "csv" / "conditions.csv", {"condition", "y", "t", "shots", "pred_dx", "corner_dx", "corner_dy", "corner_sd", "ecc_dx", "ecc_sd"});
        for (size_t k = 0; k < conds.size(); ++k) {
            const auto &c = conds[k];
            f.row({c.name, num(c.y, 2), num(c.t, 2), std::to_string(c.idx.size()), num(pred[k], 4), num(meas[k], 4), num(meanOf(ctx, c.idx, Method::Corner, true), 4),
                   num(sdOf(ctx, c.idx, Method::Corner, false), 5), num(meanOf(ctx, c.idx, Method::Ecc, false), 4), num(sdOf(ctx, c.idx, Method::Ecc, false), 5)});
        }
    } else {
        Slide &s = contentSlide(ctx, "④ 1차 모델로 평가", "기준 묶음 없음");
        s.text(0.8, 2.0, 11.7, 1.0, {para("--ref <기준 묶음> 을 주면 1차 기울기로 예측한 이동과 측정을 비교합니다.", 14)});
    }
}

// -------------------------------------------------------------------------------------
// notes.md -> 글 카드 슬라이드
//   "## 제목" 이 나오면 새 슬라이드, "- 내용" 은 항목, 빈 줄은 무시
// -------------------------------------------------------------------------------------

void addNotesSlides(Context &ctx) {
    const fs::path notes = docsRoot() / "sets" / u8path(ctx.ds.name) / "notes.md";
    if (!fs::exists(notes)) {
        Slide &s = contentSlide(ctx, "⑤ 해석", "해석 (작성 필요)");
        s.text(0.8, 2.0, 11.7, 1.5, {para("이 슬라이드는 직접 쓰는 해석 자리입니다.", 14, color::ink),
                                      para("docs/hw/eda/sets/" + ctx.ds.name + "/notes.md 에 '## 제목' 과 '- 내용' 으로 쓰면 다음 실행 때 여기에 들어갑니다.", 12)});
        return;
    }
    std::ifstream in(notes, std::ios::binary);
    std::string line, title;
    std::vector<std::string> items;
    auto flush = [&]() {
        if (title.empty()) return;
        Slide &s = contentSlide(ctx, "⑤ 해석", title);
        std::vector<Para> ps;
        for (const auto &it : items) ps.push_back(para("· " + it, 14, color::sub));
        s.text(0.8, 2.0, 11.7, 4.8, ps);
        items.clear();
    };
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.rfind("\xEF\xBB\xBF", 0) == 0) line = line.substr(3);
        if (line.rfind("## ", 0) == 0) { flush(); title = line.substr(3); }
        else if (line.rfind("- ", 0) == 0) items.push_back(line.substr(2));
        else if (!line.empty() && line[0] != '#') items.push_back(line);
    }
    flush();
}

}  // namespace hw
