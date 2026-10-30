// =====================================================================================
// slides_sweep.cpp  -  sweep 묶음 (1차: 한 축씩 연속 촬영) 분석과 슬라이드
// =====================================================================================
#include "slides_common.hpp"

#include <set>

namespace hw {

namespace {

const std::vector<std::string> kTrialOrder = {"Y 0.1 간격 +", "Y 0.1 간격 -", "Y 0.05 홀수 +", "Y 0.05 홀수 -", "Θ +", "Θ -"};

struct Trial {
    std::string name;
    char axis;                 // 'Y' 또는 'T'
    std::vector<size_t> idx;   // 사진 번호, 명령값 크기 순 (0 에서 가까운 것부터)
};

double value(const Shot &s) { return s.t != 0 ? s.t : s.y; }

// 한 칸
struct Step {
    std::string trial;
    char axis;
    double from, to;           // 명령값
    double px;                 // 이동량 (바깥 방향 +)
    bool first;                // center -> 첫 사진
};

struct MethodStats {
    std::map<std::string, LineFit> fit;     // trial -> 직선 (명령값 vs 가로 이동)
    std::vector<Step> steps;
    LineFit vertical;                        // 가로 이동 vs 세로 이동
};

std::vector<Trial> trials(const DataSet &ds) {
    std::vector<Trial> out;
    for (const auto &name : kTrialOrder) {
        Trial t{name, name[0] == 'Y' ? 'Y' : 'T', {}};
        for (size_t i = 0; i < ds.shots.size(); ++i)
            if (ds.shots[i].group == name) t.idx.push_back(i);
        if (t.idx.empty()) continue;
        std::sort(t.idx.begin(), t.idx.end(), [&](size_t a, size_t b) { return std::fabs(value(ds.shots[a])) < std::fabs(value(ds.shots[b])); });
        out.push_back(t);
    }
    return out;
}

MethodStats methodStats(const Context &ctx, const std::vector<Trial> &ts, Method meth) {
    MethodStats st;
    std::vector<double> allDx, allDy;
    for (const auto &t : ts) {
        std::vector<double> x, y;
        double prevDx = 0, prevV = 0;   // center 에서 시작
        for (size_t k = 0; k < t.idx.size(); ++k) {
            const Shot &s = ctx.ds.shots[t.idx[k]];
            const double dx = shotDx(ctx.m.shots[t.idx[k]], meth), v = value(s);
            x.push_back(v);
            y.push_back(dx);
            allDx.push_back(dx);
            allDy.push_back(shotDy(ctx.m.shots[t.idx[k]], meth));
            const double sign = v > 0 ? 1 : -1;
            st.steps.push_back({t.name, t.axis, prevV, v, sign * (dx - prevDx), k == 0});
            prevDx = dx;
            prevV = v;
        }
        st.fit[t.name] = fitLine(x, y);
    }
    st.vertical = fitLine(allDx, allDy);
    return st;
}

// 한 칸 통계 (첫 칸 제외)
struct StepSummary { double mean = kNaN, sd = kNaN; int n = 0; std::vector<Step> outliers; };
StepSummary stepSummary(const MethodStats &st, char axis) {
    std::vector<double> v;
    for (const auto &s : st.steps)
        if (s.axis == axis && !s.first) v.push_back(s.px);
    StepSummary r;
    r.mean = mean(v);
    r.sd = stdev(v);
    r.n = static_cast<int>(v.size());
    for (const auto &s : st.steps)
        if (s.axis == axis && !s.first && std::fabs(s.px - r.mean) > 3 * r.sd) r.outliers.push_back(s);
    return r;
}

std::string shortTrial(const std::string &t) { return t; }

// 노이즈 (특징점 점마다)
struct Noise {
    std::vector<double> common;              // 사진마다 사진 공통 흔들림 (가로)
    std::map<std::string, double> commonSd;  // trial -> σ
    double pointSdX = kNaN, pointSdY = kNaN;   // 점 하나 흔들림 (중앙값)
    double diagSdX = kNaN;                    // 대각선 모서리 흔들림 (중앙값)
    std::vector<double> pointSdXs, pointSdYs;   // 점마다 σ (히스토그램)
};

Noise noise(const Context &ctx, const std::vector<Trial> &ts) {
    Noise n;
    n.common.assign(ctx.ds.shots.size(), kNaN);
    const auto &C = ctx.m.corners;
    std::vector<double> diagSds;
    for (const auto &t : ts) {
        if (t.idx.size() < 5) continue;
        std::vector<double> x;
        for (size_t i : t.idx) x.push_back(value(ctx.ds.shots[i]));
        // 점마다 2차 곡선을 빼고 남은 값
        std::vector<std::vector<double>> rx(C.size()), ry(C.size());
        for (size_t k = 0; k < C.size(); ++k) {
            // 보통 모서리: 모든 사진에서 추적된 점만. 대각선: 이 trial 안에서만 모두 추적되면 비교에 씀
            if (!C[k].diagonal && !ctx.m.cornerUsable[k]) continue;
            std::vector<double> px, py;
            bool all = true;
            for (size_t i : t.idx) {
                all = all && std::isfinite(ctx.m.shots[i].pos[k].x);
                px.push_back(ctx.m.shots[i].pos[k].x - C[k].p.x);
                py.push_back(ctx.m.shots[i].pos[k].y - C[k].p.y);
            }
            if (!all) continue;
            rx[k] = quadResiduals(x, px);
            ry[k] = quadResiduals(x, py);
        }
        // 사진 공통 흔들림 = 보통 모서리 잔차의 평균
        std::vector<double> comX(t.idx.size(), 0), comY(t.idx.size(), 0);
        int used = 0;
        for (size_t k = 0; k < C.size(); ++k) {
            if (rx[k].empty() || C[k].diagonal) continue;
            for (size_t j = 0; j < t.idx.size(); ++j) { comX[j] += rx[k][j]; comY[j] += ry[k][j]; }
            ++used;
        }
        for (size_t j = 0; j < t.idx.size(); ++j) { comX[j] /= used; comY[j] /= used; n.common[t.idx[j]] = comX[j]; }
        n.commonSd[t.name] = stdev(comX);
        // 점 하나 흔들림 = 공통을 뺀 나머지의 σ
        for (size_t k = 0; k < C.size(); ++k) {
            if (rx[k].empty()) continue;
            std::vector<double> ex, ey;
            for (size_t j = 0; j < t.idx.size(); ++j) { ex.push_back(rx[k][j] - comX[j]); ey.push_back(ry[k][j] - comY[j]); }
            if (C[k].diagonal) diagSds.push_back(stdev(ex));
            else { n.pointSdXs.push_back(stdev(ex)); n.pointSdYs.push_back(stdev(ey)); }
        }
    }
    n.pointSdX = median(n.pointSdXs);
    n.pointSdY = median(n.pointSdYs);
    n.diagSdX = median(diagSds);
    return n;
}

// 축 범위를 보기 좋게 (양쪽 대칭, 반올림)
double niceMax(double v) {
    const double p = std::pow(10, std::floor(std::log10(v)));
    for (double m : {1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0})
        if (m * p >= v) return m * p;
    return 10 * p;
}

}  // namespace

void buildSweepDeck(Context &ctx) {
    const DataSet &ds = ctx.ds;
    const auto ts = trials(ds);
    const MethodStats stC = methodStats(ctx, ts, Method::Corner), stE = methodStats(ctx, ts, Method::Ecc);
    const Noise nz = noise(ctx, ts);

    // 촬영 시간, 밝기
    const double totalMin = ds.shots.back().minutes;
    std::vector<std::pair<size_t, size_t>> gaps;
    for (size_t i = 1; i < ds.shots.size(); ++i)
        if (ds.shots[i].minutes - ds.shots[i - 1].minutes > 5.0) gaps.emplace_back(i - 1, i);
    const size_t c0 = ds.centers[0];
    std::vector<double> plateOthers, wholeOthers;
    for (size_t i = 0; i < ds.shots.size(); ++i)
        if (!ds.shots[i].center) { plateOthers.push_back(ctx.m.shots[i].plateMean); wholeOthers.push_back(ctx.m.shots[i].wholeMean); }
    const double platePct = 100 * (ctx.m.shots[c0].plateMean / mean(plateOthers) - 1);
    const double wholePct = 100 * (ctx.m.shots[c0].wholeMean / mean(wholeOthers) - 1);

    // 축별 Y, Θ 개수
    int nY = 0, nT = 0;
    for (const auto &s : ds.shots) { if (!s.center && s.t == 0) ++nY; if (!s.center && s.t != 0) ++nT; }

    // ---------------------------------------------------------------- 1. 표지
    coverSlide(ctx, "자동 생성 (hw_eda) · 촬영 " + ds.shots.front().takenAt.substr(0, 10));

    // ---------------------------------------------------------------- 2. 기초 정보
    {
        Slide &s = contentSlide(ctx, "① 데이터", "기초 정보");
        const fs::path thumb = centerThumbnail(ctx);
        s.picture(thumb, 0.8, 1.95, 5.78, 5.78 * 9 / 16);
        caption(s, 0.8, 1.95 + 5.78 * 9 / 16 + 0.05, 5.78, "center 사진, 파랑 = QR 라벨 3개");
        std::vector<std::vector<std::string>> rows;
        for (const auto &t : ts) {
            const auto &a = ds.shots[t.idx.front()], &b = ds.shots[t.idx.back()];
            double st = 1e9, en = -1e9;
            for (size_t i : t.idx) { st = std::min(st, ds.shots[i].minutes); en = std::max(en, ds.shots[i].minutes); }
            const bool outward = ds.shots[t.idx.front()].minutes < ds.shots[t.idx.back()].minutes;
            rows.push_back({t.name, std::to_string(t.idx.size()), signedNum(value(a), 2) + " ~ " + signedNum(value(b), 2),
                            num(st, 1) + " ~ " + num(en, 1), outward ? "0 → 바깥" : "바깥 → 0"});
        }
        table(s, {"trial", "장수", "명령값", "촬영(분)", "찍은 방향"}, rows, {0.8, 2.1, 2.7, 4.1, 5.3}, {1.3, 0.6, 1.4, 1.2, 1.1}, 5.55, 0.2, 8);
        std::string gapText = gaps.empty() ? "5분 넘는 공백 없음" : "";
        for (const auto &g : gaps)
            gapText += num(ds.shots[g.first].minutes, 1) + " → " + num(ds.shots[g.second].minutes, 1) + "분 (" +
                       num(ds.shots[g.second].minutes - ds.shots[g.first].minutes, 1) + "분) ";
        kvTable(s, {{"장수", {std::to_string(ds.shots.size()) + "장 (center " + std::to_string(ds.centers.size()) + ", Y " + std::to_string(nY) + ", Θ " + std::to_string(nT) + ")"}},
                    {"사진 형식", {std::to_string(ctx.m.imageSize.width) + " × " + std::to_string(ctx.m.imageSize.height) + "px, 8bit",
                                 ctx.m.grayAsColor ? "3채널 값이 모두 같음 = 흑백을 컬러로 저장" : std::to_string(ctx.m.channels) + "채널"}},
                    {"trial", {std::to_string(ts.size()) + "개, 명령값을 한 방향으로 이어서 촬영"}},
                    {"촬영 시간", {"전체 " + num(totalMin, 1) + "분, 첫 사진 " + ds.shots.front().takenAt, "(파일 수정 시각 = 촬영 시각)"}},
                    {"촬영 공백", {gapText}},
                    {"밝기", {"center 만 QR 판 기준 " + signedNum(platePct, 2) + "%", "(사진 전체 기준 " + signedNum(wholePct, 2) + "%)"}}});
        ctx.sum.set("dataset.shots", static_cast<double>(ds.shots.size()), 0);
        ctx.sum.set("dataset.minutes_total", totalMin, 2);
        ctx.sum.set("dataset.center_plate_pct", platePct, 3);
        ctx.sum.set("dataset.center_whole_pct", wholePct, 3);
    }

    // ---------------------------------------------------------------- 3~5. 기준 위치
    referenceSlides(ctx);

    // ---------------------------------------------------------------- 6. 0점 위치
    auto slopeOf = [&](const MethodStats &st, char axis) {
        std::vector<double> v;
        for (const auto &t : ts) if (t.axis == axis) v.push_back(st.fit.at(t.name).slope);
        return v;
    };
    const double kY = mean(slopeOf(stC, 'Y'));   // px / mm (µm 환산용)
    {
        Slide &s = contentSlide(ctx, "② 분석에 필요한 정보", "이동 관계 – 0점 위치");
        ChartSpec cs;
        for (const auto &t : ts) cs.categories.push_back(t.name);
        Series a{"특징점", color::accent}, b{"ECC", color::orange};
        for (const auto &t : ts) { a.y.push_back(stC.fit.at(t.name).intercept); b.y.push_back(stE.fit.at(t.name).intercept); }
        cs.series = {a, b};
        cs.dataLabels = true;
        cs.yTitle = "0점 (px, center = 0)";
        cs.fontSize = 9;
        s.bars(cs, 0.8, 1.95, 6.0, 4.4);
        caption(s, 0.8, 6.4, 6.0, "trial 마다 명령값과 가로 이동을 직선으로 맞추고, 명령값 0 에서의 값(절편)을 그 trial 의 0점으로 봄");
        std::vector<double> zy;
        for (const auto &t : ts) if (t.axis == 'Y') zy.push_back(stC.fit.at(t.name).intercept);
        const double spread = *std::max_element(zy.begin(), zy.end()) - *std::min_element(zy.begin(), zy.end());
        std::vector<KvRow> rows{{"방법", {"trial 마다 직선 맞춤", "명령값 0 에서의 가로 이동 = 0점"}},
                                {"Y trial 끼리", {"최대 " + num(spread, 3) + "px 차이 (" + num(spread / kY * 1000, 0) + "µm)"}}};
        for (const auto &t : ts)
            if (t.axis == 'T') rows.push_back({t.name + " 0점", {num(stC.fit.at(t.name).intercept, 3) + "px (ECC " + num(stE.fit.at(t.name).intercept, 3) + ")"}});
        rows.push_back({"영향", {"정답(명령값) 자체에 오차가 섞임", "center → 첫 사진 한 칸이 크게 틀어짐 (9쪽)"}});
        kvTable(s, rows);
        ctx.sum.set("corner.zero_spread_y_px", spread, 4);
        ctx.sum.set("corner.zero_spread_y_um", spread / kY * 1000, 1);
        for (const auto &t : ts) {
            ctx.sum.set("corner.zero." + t.name, stC.fit.at(t.name).intercept, 4);
            ctx.sum.set("ecc.zero." + t.name, stE.fit.at(t.name).intercept, 4);
        }
    }

    // ---------------------------------------------------------------- 7~8. 이동 경로
    double dxMax = 0, dyMin = 1e9, dyMax = -1e9;
    for (size_t i = 0; i < ds.shots.size(); ++i)
        for (Method mm : {Method::Corner, Method::Ecc}) {
            dxMax = std::max(dxMax, std::fabs(shotDx(ctx.m.shots[i], mm)));
            dyMin = std::min(dyMin, shotDy(ctx.m.shots[i], mm));
            dyMax = std::max(dyMax, shotDy(ctx.m.shots[i], mm));
        }
    const double xr = niceMax(dxMax * 1.02), ypad = 0.05 * (dyMax - dyMin);
    std::vector<double> diffX, diffY;
    for (size_t i = 0; i < ds.shots.size(); ++i) {
        diffX.push_back(ctx.m.shots[i].cornerDx - ctx.m.shots[i].eccDx);
        diffY.push_back(ctx.m.shots[i].cornerDy - ctx.m.shots[i].eccDy);
    }
    for (Method meth : {Method::Corner, Method::Ecc}) {
        const MethodStats &st = meth == Method::Corner ? stC : stE;
        const std::string key = methodKey(meth);
        Slide &s = contentSlide(ctx, "② 분석에 필요한 정보", std::string("이동 관계 – 이동 경로 (") + methodName(meth) + ")");
        for (int panel = 0; panel < 2; ++panel) {
            const char axis = panel == 0 ? 'Y' : 'T';
            ChartSpec cs;
            cs.xTitle = panel == 1 ? "가로 이동 (px)" : "";
            cs.yTitle = std::string(axis == 'Y' ? "Y" : "Θ") + " trial 세로 이동 (px)";
            cs.xMin = -xr; cs.xMax = xr; cs.yMin = dyMin - ypad; cs.yMax = dyMax + ypad;
            cs.numFmtY = "0.00";
            size_t ci = 0;
            for (const auto &t : ts) {
                if (t.axis != axis) continue;
                Series se{t.name, seriesColor(ci++)};
                se.x.push_back(0); se.y.push_back(0);   // center 에서 출발
                for (size_t i : t.idx) { se.x.push_back(shotDx(ctx.m.shots[i], meth)); se.y.push_back(shotDy(ctx.m.shots[i], meth)); }
                se.line = true; se.markerSize = 2;
                cs.series.push_back(se);
            }
            s.scatter(cs, 0.8, 1.9 + panel * 2.25, 6.0, 2.25);
        }
        caption(s, 0.8, 6.45, 6.0, std::string(methodName(meth)) + " 사진별 평균 이동. 두 방법 같은 축. 선은 명령값 크기 순 연결 (촬영 시간 순 아님)");
        const auto sy = slopeOf(st, 'Y');
        std::vector<double> ry, rt;
        for (const auto &t : ts) (t.axis == 'Y' ? ry : rt).push_back(st.fit.at(t.name).residSd);
        std::vector<KvRow> rows{{"Y 기울기", {"trial별 " + num(*std::min_element(sy.begin(), sy.end()), 2) + " ~ " + num(*std::max_element(sy.begin(), sy.end()), 2) + "px / 1mm"}}};
        double tp = kNaN, tm = kNaN;
        for (const auto &t : ts)
            if (t.axis == 'T') { (t.name.back() == '+' ? tp : tm) = st.fit.at(t.name).slope; }
        if (std::isfinite(tp) && std::isfinite(tm))
            rows.push_back({"Θ 기울기", {"+쪽 " + num(tp, 2) + "px, -쪽 " + num(tm, 2) + "px / Θ 1", "(+쪽과 -쪽이 " + num(100 * std::fabs(tm / tp - 1), 1) + "% 다름)"}});
        rows.push_back({"직선 잔차", {"Y " + num(*std::min_element(ry.begin(), ry.end()), 3) + " ~ " + num(*std::max_element(ry.begin(), ry.end()), 3) + "px,  Θ " +
                                     (rt.empty() ? "-" : num(*std::min_element(rt.begin(), rt.end()), 3) + " ~ " + num(*std::max_element(rt.begin(), rt.end()), 3)) + "px (σ)"}});
        rows.push_back({"세로 이동", {"가로 이동의 " + num(100 * st.vertical.slope, 2) + "% (이동 방향 " + num(std::atan(st.vertical.slope) * 180 / CV_PI, 2) + "° 기울어짐)"}});
        rows.push_back({meth == Method::Corner ? "ECC와 차이" : "특징점과 차이", {"사진마다 가로 최대 " + num(maxAbs(diffX), 3) + "px, 세로 최대 " + num(maxAbs(diffY), 3) + "px"}});
        kvTable(s, rows);
        ctx.sum.set(key + ".slope_y_mean", mean(sy), 4);
        ctx.sum.set(key + ".slope_y_min", *std::min_element(sy.begin(), sy.end()), 4);
        ctx.sum.set(key + ".slope_y_max", *std::max_element(sy.begin(), sy.end()), 4);
        if (std::isfinite(tp)) ctx.sum.set(key + ".slope_t_plus", tp, 4);
        if (std::isfinite(tm)) ctx.sum.set(key + ".slope_t_minus", tm, 4);
        ctx.sum.set(key + ".dy_per_dx", st.vertical.slope, 6);
    }
    ctx.sum.set("method_diff.dx_max_px", maxAbs(diffX), 4);
    ctx.sum.set("method_diff.dy_max_px", maxAbs(diffY), 4);

    // ---------------------------------------------------------------- 9. 한 칸 이동량
    const StepSummary yC = stepSummary(stC, 'Y'), yE = stepSummary(stE, 'Y'), tC = stepSummary(stC, 'T'), tE = stepSummary(stE, 'T');
    {
        Slide &s = contentSlide(ctx, "② 분석에 필요한 정보", "이동 관계 – 한 칸 이동량");
        for (int panel = 0; panel < 2; ++panel) {
            const char axis = panel == 0 ? 'Y' : 'T';
            const StepSummary &sm = axis == 'Y' ? yC : tC;
            ChartSpec cs;
            cs.yTitle = axis == 'Y' ? "Y 한 칸 (px / 0.1mm)" : "Θ 한 칸 (px / 0.01)";
            cs.xTitle = panel == 1 ? "명령값 크기 (한 칸이 끝난 위치)" : "";
            cs.numFmtY = "0.00";
            Series plus{"+쪽 trial", color::accent}, minus{"-쪽 trial", color::orange}, first{"첫 칸 (center → 첫 사진)", color::gray};
            double xmax = 0;
            for (const auto &st : stC.steps) {
                if (st.axis != axis) continue;
                Series &se = st.first ? first : (st.to > 0 ? plus : minus);
                se.x.push_back(std::fabs(st.to)); se.y.push_back(st.px);
                xmax = std::max(xmax, std::fabs(st.to));
            }
            Series avg{"평균", color::ink, {0, xmax}, {sm.mean, sm.mean}, true, false, 0, true};
            first.markerSize = 5;
            cs.series = {plus, minus, first, avg};
            s.scatter(cs, 0.8, 1.9 + panel * 2.25, 6.0, 2.25);
        }
        caption(s, 0.8, 6.45, 6.0, "같은 trial 안 이웃 사진의 가로 이동 차이 (특징점)");
        auto outs = [](const StepSummary &sm) {
            std::string r;
            for (const auto &o : sm.outliers) r += (r.empty() ? "" : ", ") + signedNum(o.from, 2) + "→" + signedNum(o.to, 2) + " " + num(o.px, 3);
            return r.empty() ? std::string("없음") : r;
        };
        std::vector<std::vector<std::string>> rows{
            {"Y 한 칸", num(yC.mean, 3) + "px (σ " + num(yC.sd, 3) + ", " + std::to_string(yC.n) + "칸)", num(yE.mean, 3) + "px (σ " + num(yE.sd, 3) + ")"},
            {"Θ 한 칸", num(tC.mean, 3) + "px (σ " + num(tC.sd, 3) + ", " + std::to_string(tC.n) + "칸)", num(tE.mean, 3) + "px (σ " + num(tE.sd, 3) + ")"},
            {"±3σ 밖 Y", outs(yC), outs(yE)},
            {"±3σ 밖 Θ", outs(tC), outs(tE)}};
        std::vector<double> d;
        for (size_t k = 0; k < stC.steps.size(); ++k)
            if (!stC.steps[k].first) d.push_back(stE.steps[k].px - stC.steps[k].px);
        rows.push_back({"두 방법 차이", "같은 칸끼리 σ " + num(stdev(d), 4) + "px", "최대 " + num(maxAbs(d), 4) + "px"});
        table(s, {"항목", "특징점 (모서리)", "ECC (QR 패치)"}, rows, {7.21, 8.5, 10.55}, {1.25, 2.0, 2.0}, 2.0, 0.75, 10);
        caption(s, 7.21, 6.05, 5.33, "첫 칸 = center → 첫 사진. 0점 어긋남이 얹혀 평균에서 뺌");
        ctx.sum.set("corner.step_y_px", yC.mean, 4);
        ctx.sum.set("corner.step_y_sd", yC.sd, 4);
        ctx.sum.set("corner.step_t_px", tC.mean, 4);
        ctx.sum.set("corner.step_t_sd", tC.sd, 4);
        ctx.sum.set("ecc.step_y_px", yE.mean, 4);
        ctx.sum.set("ecc.step_t_px", tE.mean, 4);
        ctx.sum.set("method_diff.step_sd_px", stdev(d), 5);
    }

    // ---------------------------------------------------------------- 10. 노이즈 표
    double commonMin = 1e9, commonMax = 0;
    for (const auto &[k, v] : nz.commonSd) { commonMin = std::min(commonMin, v); commonMax = std::max(commonMax, v); }
    double dirFirst = kNaN;    // 거꾸로 찍은 trial 의 방향 바뀐 직후 칸
    std::string dirTrial;
    for (const auto &t : ts) {
        const bool inward = ds.shots[t.idx.front()].minutes > ds.shots[t.idx.back()].minutes;
        if (!inward || t.axis != 'Y') continue;
        for (const auto &st : stC.steps)
            if (st.trial == t.name && std::fabs(st.to) == std::fabs(value(ds.shots[t.idx.back()]))) { dirFirst = st.px; dirTrial = t.name; }
    }
    double vertMax = 0;
    for (size_t i = 0; i < ds.shots.size(); ++i) if (ds.shots[i].t != 0) vertMax = std::max(vertMax, std::fabs(ctx.m.shots[i].cornerDy));
    {
        Slide &s = contentSlide(ctx, "③ 노이즈", "노이즈 – Trial별 노이즈 분석");
        std::vector<double> zy;
        for (const auto &t : ts) if (t.axis == 'Y') zy.push_back(stC.fit.at(t.name).intercept);
        const double spread = *std::max_element(zy.begin(), zy.end()) - *std::min_element(zy.begin(), zy.end());
        std::vector<std::vector<std::string>> rows{
            {"0점 어긋남", "trial끼리 최대 " + num(spread, 2) + "px (" + num(spread / kY * 1000, 0) + "µm)", "정답 자체에 오차. center → 첫 사진 한 칸이 크게 틀어짐", "6쪽"},
            {"방향 바뀐 직후", std::isfinite(dirFirst) ? dirTrial + " 첫 칸 " + num(dirFirst, 3) + "px" : "해당 없음",
             "거꾸로 찍은 trial 의 첫 칸. 평균 " + num(yC.mean, 3) + "px 와 비교 (백래시 추정)", "9쪽"},
            {"사진 공통 흔들림", "가로 " + num(commonMin, 3) + " ~ " + num(commonMax, 3) + "px", "사진 전체가 같이 흔들림. 점을 많이 평균해도 안 줄어듦", "12쪽"},
            {"점 하나 흔들림", "가로 " + num(nz.pointSdX, 3) + "px, 세로 " + num(nz.pointSdY, 3) + "px", "점 수백 개를 평균하면 거의 사라짐", "12쪽"},
            {"대각선 모서리", num(nz.diagSdX, 3) + "px (보통 점의 " + num(nz.diagSdX / nz.pointSdX, 1) + "배)", "흔들림이 커서 측정에서 뺌", "12쪽"},
            {"밝기 변화", "center 만 " + signedNum(platePct, 2) + "% (QR 판)", "템플릿 맞춤에 영향 가능", "11쪽"},
            {"세로 밀림", "Θ trial 에서 최대 " + num(vertMax, 3) + "px", "가로 이동과 함께 판정에 섞일 수 있음", "12쪽"}};
        table(s, {"항목", "크기", "영향", "자세히"}, rows, {0.8, 2.8, 6.3, 11.5}, {1.9, 3.4, 5.1, 0.9}, 1.9, 0.55, 11);
        ctx.sum.set("noise.common_sd_min_px", commonMin, 4);
        ctx.sum.set("noise.common_sd_max_px", commonMax, 4);
        ctx.sum.set("noise.point_sd_x_px", nz.pointSdX, 4);
        ctx.sum.set("noise.point_sd_y_px", nz.pointSdY, 4);
        ctx.sum.set("noise.diag_sd_x_px", nz.diagSdX, 4);
        ctx.sum.set("noise.vertical_max_t_px", vertMax, 4);
        if (std::isfinite(dirFirst)) ctx.sum.set("noise.direction_change_step_px", dirFirst, 4);
    }

    // ---------------------------------------------------------------- 11. 노이즈 그림 1
    {
        Slide &s = bareSlide(ctx);
        Cell c = gridCell(s, 0, "촬영 순서", std::to_string(ts.size()) + " trial 을 " + num(totalMin, 0) + "분에 걸쳐 찍음. 공백 " + std::to_string(gaps.size()) + "곳");
        ChartSpec a;
        a.xTitle = "촬영 시작 후 분"; a.yTitle = "명령값 (Y 는 ÷2)"; a.fontSize = 7;
        for (size_t k = 0; k < ts.size(); ++k) {
            Series se{ts[k].name, seriesColor(k)};
            for (size_t i : ts[k].idx) { se.x.push_back(ds.shots[i].minutes); se.y.push_back(ts[k].axis == 'Y' ? value(ds.shots[i]) / 2 : value(ds.shots[i])); }
            a.series.push_back(se);
        }
        s.scatter(a, c.x, c.y, c.w, c.h);

        c = gridCell(s, 1, "0점 어긋남", "trial 마다 0점이 다름 (특징점, px). 6쪽 참고");
        ChartSpec b;
        for (const auto &t : ts) b.categories.push_back(t.name);
        Series z{"0점", color::accent};
        for (const auto &t : ts) z.y.push_back(stC.fit.at(t.name).intercept);
        b.series = {z}; b.dataLabels = true; b.horizontal = true; b.legend = false; b.fontSize = 7; b.yTitle = "0점 (px)";
        s.bars(b, c.x, c.y, c.w, c.h);

        c = gridCell(s, 2, "방향 바뀐 직후", std::isfinite(dirFirst) ? dirTrial + " 첫 칸 " + num(dirFirst, 3) + "px, 평균 " + num(yC.mean, 3) + "px" : "거꾸로 찍은 trial 없음");
        ChartSpec d;
        d.xTitle = "한 칸이 끝난 위치 (촬영 순서대로)"; d.yTitle = "한 칸 (px)"; d.fontSize = 7; d.numFmtY = "0.00";
        Series ln{dirTrial.empty() ? "trial" : dirTrial, color::orange};
        for (const auto &st : stC.steps) if (st.trial == dirTrial && !st.first) { ln.x.push_back(st.to); ln.y.push_back(st.px); }
        ln.line = true;
        Series avg{"Y 평균", color::ink, {}, {}, true, false, 0, true};
        if (!ln.x.empty()) {
            const auto [mn, mx] = std::minmax_element(ln.x.begin(), ln.x.end());
            avg.x = {*mn, *mx}; avg.y = {yC.mean, yC.mean};
        }
        d.series = {ln, avg};
        s.scatter(d, c.x, c.y, c.w, c.h);

        c = gridCell(s, 3, "밝기 변화", "center 만 QR 판 기준 " + signedNum(platePct, 2) + "%, trial 안에서도 서서히 변함");
        ChartSpec e;
        e.xTitle = "촬영 시작 후 분"; e.yTitle = "QR 판 평균 밝기"; e.fontSize = 7; e.numFmtY = "0.0";
        Series cen{"center", color::ink, {ds.shots[c0].minutes}, {ctx.m.shots[c0].plateMean}};
        cen.markerSize = 6;
        e.series.push_back(cen);
        for (size_t k = 0; k < ts.size(); ++k) {
            Series se{ts[k].name, seriesColor(k)};
            for (size_t i : ts[k].idx) { se.x.push_back(ds.shots[i].minutes); se.y.push_back(ctx.m.shots[i].plateMean); }
            e.series.push_back(se);
        }
        s.scatter(e, c.x, c.y, c.w, c.h);
    }

    // ---------------------------------------------------------------- 12. 노이즈 그림 2
    {
        Slide &s = bareSlide(ctx);
        Cell c = gridCell(s, 0, "세로 이동", "세로 이동은 가로 이동에 비례 (" + num(100 * stC.vertical.slope, 2) + "%, " + num(std::atan(stC.vertical.slope) * 180 / CV_PI, 2) + "° 기울어짐)");
        ChartSpec a;
        a.xTitle = "가로 이동 (px)"; a.yTitle = "세로 이동 (px)"; a.fontSize = 7; a.numFmtY = "0.00";
        for (size_t k = 0; k < ts.size(); ++k) {
            Series se{ts[k].name, seriesColor(k)};
            for (size_t i : ts[k].idx) { se.x.push_back(ctx.m.shots[i].cornerDx); se.y.push_back(ctx.m.shots[i].cornerDy); }
            se.markerSize = 2;
            a.series.push_back(se);
        }
        Series fit{"직선", color::ink, {-dxMax, dxMax}, {stC.vertical.intercept - stC.vertical.slope * dxMax, stC.vertical.intercept + stC.vertical.slope * dxMax}, true, false, 0, true};
        a.series.push_back(fit);
        s.scatter(a, c.x, c.y, c.w, c.h);

        c = gridCell(s, 1, "사진 공통 흔들림", "사진 전체가 같이 흔들린 양 (σ " + num(commonMin, 3) + " ~ " + num(commonMax, 3) + "px). 판정 한계의 주원인");
        ChartSpec b;
        b.xTitle = "촬영 시작 후 분"; b.yTitle = "공통 흔들림 (px)"; b.fontSize = 7; b.numFmtY = "0.00";
        for (size_t k = 0; k < ts.size(); ++k) {
            Series se{ts[k].name + " σ" + num(nz.commonSd.count(ts[k].name) ? nz.commonSd.at(ts[k].name) : kNaN, 3), seriesColor(k)};
            for (size_t i : ts[k].idx) if (std::isfinite(nz.common[i])) { se.x.push_back(ds.shots[i].minutes); se.y.push_back(nz.common[i]); }
            se.markerSize = 2;
            b.series.push_back(se);
        }
        s.scatter(b, c.x, c.y, c.w, c.h);

        c = gridCell(s, 2, "점 하나 흔들림", "점 하나는 가로 " + num(nz.pointSdX, 3) + "px, 세로 " + num(nz.pointSdY, 3) + "px. 점을 평균하면 거의 사라짐");
        ChartSpec d;
        d.xTitle = "점 하나의 흔들림 σ (px)"; d.yTitle = "점 수"; d.fontSize = 7; d.numFmtX = "0.00";
        auto hist = [](const std::vector<double> &v, const std::string &name, const std::string &col) {
            Series se{name, col};
            const double w = 0.004;
            std::map<int, int> cnt;
            for (double x : v) if (std::isfinite(x)) ++cnt[static_cast<int>(x / w)];
            for (const auto &kv : cnt) { se.x.push_back((kv.first + 0.5) * w); se.y.push_back(kv.second); }
            se.line = true; se.markerSize = 2;
            return se;
        };
        d.series = {hist(nz.pointSdXs, "가로 (중앙값 " + num(nz.pointSdX, 3) + ")", color::accent), hist(nz.pointSdYs, "세로 (중앙값 " + num(nz.pointSdY, 3) + ")", color::orange)};
        s.scatter(d, c.x, c.y, c.w, c.h);

        c = gridCell(s, 3, "대각선 모서리", "검정 칸 두 개가 꼭짓점만 맞닿은 점은 " + num(nz.diagSdX, 3) + "px 로 " + num(nz.diagSdX / nz.pointSdX, 1) + "배 흔들려 측정에서 뺌");
        ChartSpec e;
        e.categories = {"보통 모서리", "대각선 모서리"};
        e.series = {Series{"가로 흔들림 σ (중앙값, px)", color::accent, {}, {nz.pointSdX, nz.diagSdX}}};
        e.dataLabels = true; e.labelFmt = "0.000"; e.horizontal = true; e.legend = false; e.fontSize = 8; e.yTitle = "점 하나의 가로 흔들림 σ (중앙값, px)";
        s.bars(e, c.x, c.y, c.w, c.h);
    }

    // CSV: trial 별 직선, 한 칸
    {
        Csv f(ctx.out / "csv" / "trials.csv", {"trial", "method", "slope", "zero", "resid_sd", "common_sd"});
        for (const auto &t : ts)
            for (Method mm : {Method::Corner, Method::Ecc}) {
                const LineFit &lf = (mm == Method::Corner ? stC : stE).fit.at(t.name);
                f.row({t.name, methodKey(mm), num(lf.slope, 5), num(lf.intercept, 5), num(lf.residSd, 5),
                       mm == Method::Corner && nz.commonSd.count(t.name) ? num(nz.commonSd.at(t.name), 5) : ""});
            }
        Csv g(ctx.out / "csv" / "steps.csv", {"trial", "from", "to", "first", "corner_px", "ecc_px"});
        for (size_t k = 0; k < stC.steps.size(); ++k)
            g.row({stC.steps[k].trial, num(stC.steps[k].from, 2), num(stC.steps[k].to, 2), stC.steps[k].first ? "1" : "0", num(stC.steps[k].px, 5), num(stE.steps[k].px, 5)});
    }
}

}  // namespace hw
