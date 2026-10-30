#include "slides_common.hpp"

namespace hw {

const std::string &seriesColor(size_t i) {
    static const std::vector<std::string> c = {color::accent, color::orange, color::teal, color::brown, "7C3AED", "DB2777", color::green, "0891B2"};
    return c[i % c.size()];
}

fs::path saveFig(Context &ctx, const std::string &name, const cv::Mat &img) {
    const fs::path p = ctx.out / "fig" / (name + ".png");
    writeImage(p, img);
    return p;
}

// -------------------------------------------------------------------------------------
// 슬라이드 틀
// -------------------------------------------------------------------------------------

static void pageNumber(Context &ctx, Slide &s) {
    s.text(11.93, 7.0, 0.6, 0.3, {para(std::to_string(++ctx.page), 10, color::faint, false, "r")});
}

void coverSlide(Context &ctx, const std::string &subtitle) {
    Slide &s = ctx.deck.add("DIVIDER");
    ++ctx.page;
    s.text(0.8, 3.55, 8.0, 1.1, {Para{{Run{"데이터 EDA", 52, color::ink, true}}}}, "ctr");
    s.line(0.8, 4.8, 12.53, 4.8, color::ink, 0.75);
    s.text(8.33, 5.0, 4.2, 0.5, {para("묶음 " + ctx.ds.name, 15, color::accent, true, "r")}, "ctr");
    s.text(8.33, 5.5, 4.2, 0.5, {para(subtitle, 11, color::sub, false, "r")}, "ctr");
}

Slide &contentSlide(Context &ctx, const std::string &section, const std::string &title) {
    Slide &s = ctx.deck.add("CONTENT_GRAY");
    s.text(0.8, 0.6, 8.0, 0.3, {Para{{Run{"1-1  ", 11, color::faint}, Run{ctx.ds.name + "  |  " + section, 11, color::ink}}}});
    s.text(0.8, 0.95, 11.73, 0.8, {Para{{Run{title, 28, color::ink, true}}}}, "ctr");
    pageNumber(ctx, s);
    return s;
}

Slide &bareSlide(Context &ctx) {
    Slide &s = ctx.deck.add("CONTENT_GRAY");
    pageNumber(ctx, s);
    return s;
}

void kvTable(Slide &s, const std::vector<KvRow> &rows, double top, double bottom) {
    const double KX = 7.21, KW = 1.3, VX = 8.51, VW = 4.03, LW = 5.33, LH = 0.2;
    size_t lines = 0;
    for (const auto &r : rows) lines += std::max<size_t>(1, r.values.size());
    const double pad = (bottom - top - lines * LH) / rows.size();
    double y = top;
    for (const auto &r : rows) {
        const size_t n = std::max<size_t>(1, r.values.size());
        s.text(KX, y + pad / 2 - 0.02, KW, 0.3, {para(r.key, 12, color::sub)});
        std::vector<Para> ps;
        for (const auto &v : r.values) ps.push_back(para(v, 11, color::sub));
        s.text(VX, y + pad / 2, VW, n * LH + 0.05, ps);
        y += n * LH + pad;
        s.line(KX, y, KX + LW, y);
    }
}

void table(Slide &s, const std::vector<std::string> &header, const std::vector<std::vector<std::string>> &rows,
           const std::vector<double> &colX, const std::vector<double> &colW, double top, double rowH, double fontSize) {
    const double x0 = colX.front(), x1 = colX.back() + colW.back();
    for (size_t c = 0; c < header.size(); ++c) s.text(colX[c], top, colW[c], 0.3, {para(header[c], fontSize, color::ink, true)}, "ctr");
    double y = top + 0.36;
    s.line(x0, y, x1, y, color::ink, 1.0);
    for (const auto &r : rows) {
        for (size_t c = 0; c < r.size() && c < colX.size(); ++c) s.text(colX[c], y, colW[c], rowH, {para(r[c], fontSize, color::sub)}, "ctr");
        y += rowH;
        s.line(x0, y, x1, y);
    }
}

void caption(Slide &s, double x, double y, double w, const std::string &text) {
    s.text(x, y, w, 0.45, {para(text, 9, color::faint)});
}

Cell gridCell(Slide &s, int index, const std::string &title, const std::string &desc) {
    // v4 노이즈 4분할: 여백 0.35, 칸 사이 0.30, 위 0.30
    const double MX = 0.35, GAP = 0.30, Y0 = 0.30, RGAP = 0.25, W = 13.333, H = 7.5;
    const double CW = (W - 2 * MX - GAP) / 2, CH = (H - Y0 - 0.5 - RGAP) / 2;
    const int r = index / 2, c = index % 2;
    const double x = MX + c * (CW + GAP), y = Y0 + r * (CH + RGAP);
    s.text(x, y, CW, 0.28, {para(title, 13, color::ink, true)});
    s.text(x, y + 0.30, CW, 0.30, {para(desc, 9.5, color::sub)});
    return Cell{x, y + 0.62, CW, CH - 0.62};
}

// -------------------------------------------------------------------------------------
// 사진 그림
// -------------------------------------------------------------------------------------

// QR 3개를 둘러싼 영역을 잘라 3배 확대한 컬러 그림
static cv::Mat qrCrop(const Context &ctx, cv::Rect &area, int scale) {
    const auto &q = ctx.m.qrs;
    area = q[0].patch | q[1].patch | q[2].patch;
    area.x -= 6; area.y -= 6; area.width += 12; area.height += 12;
    area &= cv::Rect(0, 0, ctx.m.imageSize.width, ctx.m.imageSize.height);
    const cv::Mat gray = readGray(ctx.ds.shots[ctx.ds.centers[0]].file);
    cv::Mat crop, big;
    cv::cvtColor(gray(area), crop, cv::COLOR_GRAY2BGR);
    cv::resize(crop, big, cv::Size(), scale, scale, cv::INTER_CUBIC);
    return big;
}

fs::path centerThumbnail(Context &ctx) {
    const cv::Mat gray = readGray(ctx.ds.shots[ctx.ds.centers[0]].file);
    cv::Mat img;
    cv::cvtColor(gray, img, cv::COLOR_GRAY2BGR);
    for (const auto &q : ctx.m.qrs) cv::rectangle(img, q.label, cv::Scalar(227, 113, 0), 8);
    cv::Mat small;
    cv::resize(img, small, cv::Size(1280, 720), 0, 0, cv::INTER_AREA);
    return saveFig(ctx, "02-center", small);
}

void referenceSlides(Context &ctx) {
    const auto &q = ctx.m.qrs;
    const int K = 3;
    cv::Rect area;
    const cv::Scalar blue(227, 113, 0), green(74, 163, 22), orange(12, 88, 234), red(38, 38, 220);
    auto P = [&](cv::Point2d p) { return cv::Point(cvRound((p.x - area.x) * K), cvRound((p.y - area.y) * K)); };
    auto R = [&](const cv::Rect &r) { return cv::Rect((r.x - area.x) * K, (r.y - area.y) * K, r.width * K, r.height * K); };
    const double imgW = 5.78;

    // ---- 전체 QR 기준
    {
        cv::Mat img = qrCrop(ctx, area, K);
        for (const auto &r : q) {
            cv::rectangle(img, R(r.label), blue, 5);
            cv::circle(img, P(r.labelCenter), 10, blue, -1);
        }
        const fs::path fig = saveFig(ctx, "03-qr", img);
        Slide &s = contentSlide(ctx, "② 분석에 필요한 정보", "기준 위치 – 전체 QR 기준");
        const double h = imgW * img.rows / img.cols;
        s.picture(fig, 0.8, 4.0 - h / 2, imgW, h);
        caption(s, 0.8, 4.0 + h / 2 + 0.1, imgW, "center 사진의 QR 라벨 3개. 파랑 네모 = 밝은 라벨 경계, 점 = 라벨 가운데");
        std::vector<std::string> sizes, centers;
        for (int i = 0; i < kQrCount; ++i) {
            centers.push_back(std::string(kQrNames[i]) + " (" + num(q[i].labelCenter.x, 1) + ", " + num(q[i].labelCenter.y, 1) + ")");
            sizes.push_back(std::string(kQrNames[i]) + " " + std::to_string(q[i].label.width) + " × " + std::to_string(q[i].label.height) + "px");
        }
        kvTable(s, {{"사진 전체 크기", {std::to_string(ctx.m.imageSize.width) + " × " + std::to_string(ctx.m.imageSize.height) + "px"}},
                    {"QR 찾는 법", {"가운데 띠를 오츠 이진화", "정사각형에 가깝고 나란한 밝은 덩어리 3개"}},
                    {"라벨 중심", {centers[0] + "  " + centers[1], centers[2]}},
                    {"라벨 크기", {sizes[0] + ", " + sizes[1], sizes[2]}},
                    {"라벨 간격", {num(q[1].labelCenter.x - q[0].labelCenter.x, 1) + "px (L와 C), " + num(q[2].labelCenter.x - q[1].labelCenter.x, 1) + "px (C와 R)"}},
                    {"라벨 밝기", {"L " + num(q[0].labelMean, 1) + ", C " + num(q[1].labelMean, 1) + ", R " + num(q[2].labelMean, 1) + " (0~255)"}}});
        ctx.sum.set("qr.label_gap_lc_px", q[1].labelCenter.x - q[0].labelCenter.x, 2);
        ctx.sum.set("qr.label_gap_cr_px", q[2].labelCenter.x - q[1].labelCenter.x, 2);
    }

    // ---- 특징점
    {
        cv::Mat img = qrCrop(ctx, area, K);
        int counts[kQrCount] = {0}, used[kQrCount] = {0}, diag = 0, lost = 0;
        for (size_t k = 0; k < ctx.m.corners.size(); ++k) {
            const Corner &c = ctx.m.corners[k];
            ++counts[c.qr];
            const bool ok = c.diagonal || ctx.m.cornerUsable[k];   // 대각선은 평균에 안 쓰므로 추적 실패와 따로 셈
            if (c.diagonal) ++diag;
            else if (!ok) ++lost;
            else ++used[c.qr];
            const cv::Scalar col = !ok ? red : (c.diagonal ? orange : green);
            cv::circle(img, P(c.p), 5, col, -1);
        }
        const fs::path fig = saveFig(ctx, "04-corners", img);
        const int total = static_cast<int>(ctx.m.corners.size()), usedAll = used[0] + used[1] + used[2];
        Slide &s = contentSlide(ctx, "② 분석에 필요한 정보", "이동 관계 – 특징점 분석");
        const double h = imgW * img.rows / img.cols;
        s.picture(fig, 0.8, 4.0 - h / 2, imgW, h);
        caption(s, 0.8, 4.0 + h / 2 + 0.1, imgW, "center 사진의 QR 격자 모서리. 초록 = 사용, 주황 = 대각선 모서리 (평균에서 제외, 흔들림 비교용), 빨강 = 일부 사진에서 맞춤 실패 (제외)");
        kvTable(s, {{"모서리 수", {std::to_string(total) + "점 (L " + std::to_string(counts[0]) + ", C " + std::to_string(counts[1]) + ", R " + std::to_string(counts[2]) + ")",
                                 "라벨 안에서 코너 검출 후 서브픽셀"}},
                    {"모서리 종류", {"주변 4칸 중 검정 1칸·3칸 = 보통 모서리", "검정 2칸이 대각선 = 대각선 모서리"}},
                    {"사용 점", {std::to_string(usedAll) + "점 (L " + std::to_string(used[0]) + ", C " + std::to_string(used[1]) + ", R " + std::to_string(used[2]) + ")",
                                "대각선 " + std::to_string(diag) + "점, 추적 실패 " + std::to_string(lost) + "점 제외"}},
                    {"위치 재는 법", {"흐린 2x2 모듈 밝기 모델을 9x9 창에 맞춤", "ECC 로 자리 예측 → 맞춤, 기준 못 넘으면 실패"}},
                    {"사진 이동", {"사용 점의 이동 평균 (모든 사진 같은 점)"}}});
        ctx.sum.set("corner.points_total", total, 0);
        ctx.sum.set("corner.points_used", usedAll, 0);
        ctx.sum.set("corner.points_diagonal", diag, 0);
        ctx.sum.set("corner.points_lost", lost, 0);
    }

    // ---- ECC
    {
        cv::Mat img = qrCrop(ctx, area, K);
        for (const auto &r : q) {
            cv::rectangle(img, R(r.patch), blue, 5);
            cv::circle(img, P(cv::Point2d(r.patch.x + (r.patch.width - 1) / 2.0, r.patch.y + (r.patch.height - 1) / 2.0)), 10, blue, -1);
        }
        const fs::path fig = saveFig(ctx, "05-ecc", img);
        std::vector<double> cc;
        for (size_t i = 0; i < ctx.ds.shots.size(); ++i) {
            if (ctx.ds.shots[i].center && i == ctx.ds.centers[0]) continue;
            for (const auto &e : ctx.m.shots[i].ecc) cc.push_back(e.cc);
        }
        const double ccMin = *std::min_element(cc.begin(), cc.end());
        Slide &s = contentSlide(ctx, "② 분석에 필요한 정보", "이동 관계 – ECC 분석");
        const double h = imgW * img.rows / img.cols;
        s.picture(fig, 0.8, 4.0 - h / 2, imgW, h);
        caption(s, 0.8, 4.0 + h / 2 + 0.1, imgW, "center 사진의 QR 3개. 파랑 네모 = ECC 기준 패치 (QR 라벨 + 바깥 " + std::to_string(kPatchMargin) + "px), 점 = 패치 가운데");
        kvTable(s, {{"기준 패치", {"QR 라벨 전체 + 바깥 " + std::to_string(kPatchMargin) + "px (세 QR 같은 규칙)",
                                 "L " + std::to_string(q[0].patch.width) + "×" + std::to_string(q[0].patch.height) + ", C " + std::to_string(q[1].patch.width) + "×" +
                                     std::to_string(q[1].patch.height) + ", R " + std::to_string(q[2].patch.width) + "×" + std::to_string(q[2].patch.height) + "px"}},
                    {"사용 픽셀", {"패치 전체 (점을 고르지 않음)", "라벨 테두리와 QR 모듈을 모두 씀"}},
                    {"맞추는 방법", {"① 템플릿 매칭으로 대략 위치 (가로 ±320px)", "② ECC 아핀으로 서브픽셀 정렬, 가운데 점 이동"}},
                    {"맞춤 품질", {"ECC 상관계수 최저 " + num(ccMin, 4) + ", 중앙값 " + num(median(cc), 4), "(" + std::to_string(cc.size() / 3) + "장 × QR 3개, 1 = center 와 똑같음)"}}});
        ctx.sum.set("ecc.cc_min", ccMin, 5);
        ctx.sum.set("ecc.cc_median", median(cc), 5);
    }
}

}  // namespace hw
