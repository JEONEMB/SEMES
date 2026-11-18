// 281장 형식·밝기·촬영 순서와 이동 구간별 직선 맞춤(기울기·0점)으로 정답값(파일명 명령값)의 신뢰도를 본다.
// 원본 코드는 남아 있지 않다. 근거 문서: 00000.legacies/00_데이터분석.md (16, 18, 19번 그림 설명)
#include <hye/dataset.hpp>
#include <hye/plot.hpp>
#include <hye/report.hpp>
#include <omp.h>
#include <array>
#include <iostream>
#include <map>
#include <set>
#include <sstream>

using namespace hye;

namespace {

// 촬영 순서. 원래 분석은 zip 안 파일 수정 시각으로 구했다. data/clean 에는 시각이 없어 00_데이터분석.md 5절의 순서를 규칙으로 옮긴다.
//  center → Y +0.10..+2.00 (0.1) → Y -0.10..-2.00 → T +0.01..+1.00 → T -0.01..-1.00 → (15분 공백) → Y +0.05..+1.95 (0.05 홀수) → Y -1.95..-0.05
const char* kSeg[] = {"center", "Y1+", "Y1-", "T+", "T-", "Y2+", "Y2-"};
int segment(const Shot& s) {
    if (s.condition == "center") return 0;
    if (s.condition == "T") return s.t > 0 ? 3 : 4;
    bool odd = std::lround(std::abs(s.y) * 20) % 2 == 1;
    return odd ? (s.y > 0 ? 5 : 6) : (s.y > 0 ? 1 : 2);
}
// 파인더 패턴 한 점 템플릿 추적 (track_all.py 의 0번 점과 같은 설정: 48x48, ±120 / ±12 px, NCC, 포물선 보간)
const cv::Point2d kFinder{1846.7, 1010.3};
const int HP = 24, SXR = 120, SYR = 12;
double parab(float am, float a0, float ap) {
    float d = am - 2.0f * a0 + ap;
    return d != 0 ? (0.5f * (am - ap)) / d : 0.0f;
}
cv::Point2d track(const cv::Mat& tpl, const cv::Mat& g) {
    int xi = (int)std::nearbyint(kFinder.x), yi = (int)std::nearbyint(kFinder.y);
    cv::Mat s, r;
    g(cv::Rect(xi - HP - SXR, yi - HP - SYR, 2 * (HP + SXR), 2 * (HP + SYR))).convertTo(s, CV_32F);
    cv::matchTemplate(s, tpl, r, cv::TM_CCOEFF_NORMED);
    cv::Point b;
    cv::minMaxLoc(r, nullptr, nullptr, nullptr, &b);
    double sx = b.x, sy = b.y;
    if (b.x > 0 && b.x < r.cols - 1) sx += parab(r.at<float>(b.y, b.x - 1), r.at<float>(b.y, b.x), r.at<float>(b.y, b.x + 1));
    if (b.y > 0 && b.y < r.rows - 1) sy += parab(r.at<float>(b.y - 1, b.x), r.at<float>(b.y, b.x), r.at<float>(b.y + 1, b.x));
    return {sx - SXR + (xi - kFinder.x), sy - SYR + (yi - kFinder.y)};
}

struct Stat {
    double mean = 0, sat = 0, noise = 0;
    int levels = 0, zeros = 0;
    bool gray = true;
    cv::Point2d d;
};

std::pair<double, double> polyfit1(const std::vector<double>& x, const std::vector<double>& y) {
    double mx = mean(x), my = mean(y), sxy = 0, sxx = 0;
    for (size_t i = 0; i < x.size(); ++i) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) * (x[i] - mx); }
    double k = sxy / sxx;
    return {k, my - k * mx};
}
double sd0(const std::vector<double>& v) {
    double m = mean(v), s = 0;
    for (double x : v) s += (x - m) * (x - m);
    return std::sqrt(s / v.size());
}

}  // namespace

static int run();
int main() {
    try {
        return run();
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}

static int run() {
    omp_set_num_threads(4);
    begin_results();
    auto shots = load_1014();
    const int n = (int)shots.size();
    cv::Mat C = read_gray(center_1014(shots)), tpl;
    {
        int xi = (int)std::nearbyint(kFinder.x), yi = (int)std::nearbyint(kFinder.y);
        C(cv::Rect(xi - HP, yi - HP, 2 * HP, 2 * HP)).convertTo(tpl, CV_32F);
    }
    // ---------- 사진별 형식·밝기·잡음·파인더 이동 ----------
    std::vector<Stat> st(n);
    std::vector<int> bmp_w(n), bmp_h(n), bmp_c(n);
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; ++i) {
        cv::Mat im = read_color(shots[i]);
        bmp_w[i] = im.cols; bmp_h[i] = im.rows; bmp_c[i] = im.channels();
        std::vector<cv::Mat> ch;
        cv::split(im, ch);
        Stat s;
        s.gray = cv::countNonZero(ch[0] != ch[1]) == 0 && cv::countNonZero(ch[1] != ch[2]) == 0;
        cv::Mat g = ch[0];
        s.mean = cv::mean(g)[0];
        int hist[256] = {0};
        for (int y = 0; y < g.rows; ++y) { const uchar* p = g.ptr<uchar>(y); for (int x = 0; x < g.cols; ++x) ++hist[p[x]]; }
        for (int v = 0; v < 256; ++v) s.levels += hist[v] > 0;
        s.sat = 100.0 * hist[255] / g.total();
        s.zeros = hist[0];
        // 잡음: 평평한 벽 (100..500, 100..400) 의 가로 이웃 차 / sqrt(2) 의 표준편차
        cv::Mat w;
        g(cv::Rect(100, 100, 400, 300)).convertTo(w, CV_64F);
        cv::Mat dd = (w.colRange(1, w.cols) - w.colRange(0, w.cols - 1)) / std::sqrt(2.0);
        cv::Scalar mu, sd;
        cv::meanStdDev(dd, mu, sd);
        s.noise = sd[0];
        s.d = track(tpl, g);
        st[i] = s;
    }
    std::cout << "measured " << n << "\n";
    // 템플릿 중심이 정수 화소(1847, 1010)라 center 자신도 dx = +0.3 으로 나온다. center 대비 이동으로 바꾼다
    int ci = -1;
    for (int i = 0; i < n; ++i) if (shots[i].condition == "center") ci = i;
    const cv::Point2d d0 = st[ci].d;
    for (auto& s : st) s.d -= d0;
    std::set<std::string> hashes;
    {
        std::ifstream in(fs::u8path(clean_dir() + "/clean_20261014/_metadata/manifest.csv"));
        std::string line;
        std::getline(in, line);
        while (std::getline(in, line)) if (!line.empty()) hashes.insert(split_csv(line)[3]);
    }
    const int dup = n - (int)hashes.size();
    long long fbytes = (long long)fs::file_size(fs::u8path(shots[0].path));
    bool same_size = true;
    for (auto& s : shots) same_size &= (long long)fs::file_size(fs::u8path(s.path)) == fbytes;

    // ---------- 촬영 순서 ----------
    std::vector<int> ord(n);
    std::iota(ord.begin(), ord.end(), 0);
    std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) {
        int ga = segment(shots[a]), gb = segment(shots[b]);
        if (ga != gb) return ga < gb;
        double va = shots[a].condition == "T" ? shots[a].t : shots[a].y, vb = shots[b].condition == "T" ? shots[b].t : shots[b].y;
        if (ga == 6) return va < vb;  // -1.95 → -0.05
        return std::abs(va) < std::abs(vb);
    });
    std::vector<double> others;
    for (int i = 0; i < n; ++i) if (i != ci) others.push_back(st[i].mean);
    double all_mean = 0;
    for (auto& s : st) all_mean += s.mean / n;
    const double omin = *std::min_element(others.begin(), others.end()), omax = *std::max_element(others.begin(), others.end());
    const double center_rel = (st[ci].mean / mean(others) - 1) * 100;
    int lv_min = 999, lv_max = 0, n_gray = 0, zero_px = 0;
    double sat_min = 1e9, sat_max = 0;
    std::vector<double> noise;
    for (auto& s : st) {
        lv_min = std::min(lv_min, s.levels); lv_max = std::max(lv_max, s.levels);
        sat_min = std::min(sat_min, s.sat); sat_max = std::max(sat_max, s.sat);
        n_gray += s.gray; zero_px += s.zeros; noise.push_back(s.noise);
    }
    const double noise_med = percentile(noise, 0.5);
    // 첫 3분(약 40장) 밝기 추세
    std::vector<double> first_x, first_y;
    for (int k = 1; k <= 40; ++k) { first_x.push_back(k); first_y.push_back(st[ord[k]].mean); }

    // ---------- 구간별 직선 맞춤 (파인더 가로 이동 vs 명령값) ----------
    struct Seg { int g; double k, b, res, kdy; std::vector<int> idx; };
    std::vector<Seg> segs;
    std::map<int, std::pair<double, int>> worst_res;  // 축(0 T, 1 Y) → (잔차, 사진)
    std::vector<double> seg_res(n, kNaN);
    for (int g = 1; g <= 6; ++g) {
        Seg s{g};
        std::vector<double> x, y, yv;
        for (int i : ord)
            if (segment(shots[i]) == g) {
                s.idx.push_back(i);
                x.push_back(shots[i].condition == "T" ? shots[i].t : shots[i].y);
                y.push_back(st[i].d.x);
                yv.push_back(st[i].d.y);
            }
        std::tie(s.k, s.b) = polyfit1(x, y);
        std::vector<double> r;
        for (size_t j = 0; j < x.size(); ++j) {
            r.push_back(y[j] - (s.k * x[j] + s.b));
            seg_res[s.idx[j]] = r.back();
            int ax = shots[s.idx[j]].condition == "T" ? 0 : 1;
            if (!worst_res.count(ax) || std::abs(r.back()) > std::abs(worst_res[ax].first)) worst_res[ax] = {r.back(), s.idx[j]};
        }
        s.res = sd0(r);
        s.kdy = polyfit1(x, yv).first;
        segs.push_back(s);
    }
    double bmin = 1e9, bmax = -1e9, rmin = 1e9, rmax = 0;
    for (auto& s : segs) { bmin = std::min(bmin, s.b); bmax = std::max(bmax, s.b); rmin = std::min(rmin, s.res); rmax = std::max(rmax, s.res); }
    // 같은 축 안 구간끼리의 0점 차 최대 (T: T+ vs T-, Y: 네 구간)
    double zero_spread = 0;
    for (auto& a : segs)
        for (auto& b : segs)
            if (((a.g == 3 || a.g == 4) == (b.g == 3 || b.g == 4))) zero_spread = std::max(zero_spread, std::abs(a.b - b.b));
    const double all_spread = bmax - bmin;
    double kY = 0;
    for (auto& s : segs) if (s.g == 1 || s.g == 2 || s.g == 5 || s.g == 6) kY += s.k / 4;
    const double kT = (segs[2].k + segs[3].k) / 2;
    // 세로 이동: T 전체, Y 전체 기울기
    std::vector<double> tx, tdy, yx, ydy;
    for (int i = 0; i < n; ++i) {
        if (shots[i].condition == "T") { tx.push_back(shots[i].t); tdy.push_back(st[i].d.y); }
        if (shots[i].condition == "Y") { yx.push_back(shots[i].y); ydy.push_back(st[i].d.y); }
    }
    const double kTdy = polyfit1(tx, tdy).first, kYdy = polyfit1(yx, ydy).first;
    double cmin = 1e9, cmax = 0;
    for (auto& s : segs) { cmin = std::min(cmin, std::abs(s.b)); cmax = std::max(cmax, std::abs(s.b)); }

    // ---------- 그림 ----------
    {
        Chart c("Mean brightness in capture order", "capture order (documented sequence)", "mean gray level");
        for (int g = 0; g <= 6; ++g) {
            std::vector<double> x, y;
            for (int k = 0; k < n; ++k) if (segment(shots[ord[k]]) == g) { x.push_back(k); y.push_back(st[ord[k]].mean); }
            c.scatter(x, y, g == 0 ? kRed : color(g - 1), kSeg[g], g == 0 ? 6 : 3);
        }
        c.note(0, st[ci].mean, "center " + fmt(st[ci].mean, 1));
        c.save("01_촬영순서_밝기.png");
    }
    auto seg_chart = [&](bool isT, const std::string& file) {
        Chart c(std::string("Zero offset by segment: dx - common slope x command (") + (isT ? "T" : "Y") + ")", isT ? "command T (deg)" : "command Y (mm)",
                "finder dx minus common line (px)");
        double kc = isT ? kT : kY;
        for (auto& s : segs) {
            bool sT = s.g == 3 || s.g == 4;
            if (sT != isT) continue;
            std::vector<double> x, y;
            for (int i : s.idx) { double v = isT ? shots[i].t : shots[i].y; x.push_back(v); y.push_back(st[i].d.x - kc * v); }
            c.scatter(x, y, color(s.g - 1), std::string(kSeg[s.g]) + " zero " + fmts(s.b, 2) + " px");
        }
        c.scatter({0}, {0}, kRed, "center", 6).hline(0);
        c.save(file);
    };
    seg_chart(true, "02_구간별0점_T.png");
    seg_chart(false, "03_구간별0점_Y.png");
    {
        Chart c("Linearity: within-segment residual from own line", "capture order", "residual (px)");
        for (auto& s : segs) {
            std::vector<double> x, y;
            for (int i : s.idx) { x.push_back((double)(std::find(ord.begin(), ord.end(), i) - ord.begin())); y.push_back(seg_res[i]); }
            c.scatter(x, y, color(s.g - 1), std::string(kSeg[s.g]) + " sd " + fmt(s.res, 3));
        }
        c.hline(0);
        for (auto& w : worst_res) {
            int i = w.second.second;
            c.note((double)(std::find(ord.begin(), ord.end(), i) - ord.begin()), w.second.first, shots[i].rel.substr(shots[i].rel.find('/') + 1));
        }
        c.save("04_구간안_직선잔차.png");
    }
    {
        Chart c("Vertical finder shift vs command", "command (T deg or Y mm)", "finder dy (px)");
        c.scatter(tx, tdy, color(1), "T scan, " + fmt(kTdy, 3) + " px/deg").scatter(yx, ydy, color(2), "Y scan, " + fmt(kYdy, 3) + " px/mm").hline(0);
        c.save("05_세로이동.png");
    }
    // 튀는 값: 축별 구간 직선에서 가장 벗어난 사진의 파인더 확대
    auto nm = [&](int i) { auto r = shots[i].rel.substr(shots[i].rel.find('/') + 1); return r.substr(0, r.rfind('.')); };
    auto outlier_fig = [&](int i, const std::string& file) {
        cv::Mat im = read_color(shots[i]);
        cv::Point2d ref = kFinder, now = kFinder + st[i].d;
        cv::Rect2d rb(ref.x - HP, ref.y - HP, 2 * HP, 2 * HP), nb(now.x - HP, now.y - HP, 2 * HP, 2 * HP);
        cv::Rect roi = cv::Rect(rb | nb) + cv::Size(80, 80) - cv::Point(40, 40);
        auto a = zoom(im, roi, 520, {{now, {0, 0}, "", kRed}, {ref, {0, 0}, "", kGreen}}, 0, nm(i) + " residual " + fmts(seg_res[i], 3) + " px",
                      {{rb, kGreen, "center"}, {nb, kRed, "dx " + fmt(st[i].d.x, 2)}});
        save_png(file, a);
    };
    const int wT = worst_res[0].second, wY = worst_res[1].second;
    const std::string fT = "07_튀는값_T_" + nm(wT) + ".png", fY = "06_튀는값_Y_" + nm(wY) + ".png";
    outlier_fig(wY, fY);
    outlier_fig(wT, fT);

    // ---------- 대조 ----------
    // 출처: 00000.legacies/00_데이터분석.md 2절·6절 표
    Parity par;
    par.add("해상도 가로 (px)", 3840, bmp_w[0], 0, 0);
    par.add("해상도 세로 (px)", 2160, bmp_h[0], 0, 0);
    par.add("파일 크기 (byte, 전 사진 동일)", 24883254, same_size ? (double)fbytes : -1, 0, 0);
    par.add("R=G=B 사진 수", 281, n_gray, 0, 0);
    par.add("전체 평균 밝기", 171.6, all_mean, 0.05, 1);
    par.add("center 밝기", 173.7, st[ci].mean, 0.05, 1);
    par.add("나머지 밝기 최소", 171.4, omin, 0.05, 1);
    par.add("나머지 밝기 최대", 172.3, omax, 0.05, 1);
    par.add("center 밝기 차이 (%)", 1.2, center_rel, 0.1, 1);
    par.add("밝기 단계 최소", 247, lv_min, 0, 0);
    par.add("밝기 단계 최대", 252, lv_max, 0, 0);
    par.add("포화(255) 비율 최소 (%)", 0.18, sat_min, 0.005, 2);
    par.add("포화(255) 비율 최대 (%)", 0.25, sat_max, 0.005, 2);
    par.add("0 픽셀 수", 0, zero_px, 0, 0);
    par.add("벽 잡음 (중앙값)", 0.73, noise_med, 0.05, 2);
    par.add("완전히 같은 파일 수", 0, dup, 0, 0);
    // 구간 표: 기울기, 0점, 잔차
    const double K[6] = {14.33, 14.35, 80.73, 81.55, 14.36, 14.33}, Bz[6] = {-0.56, 0.06, -0.84, -0.22, -0.49, -0.49}, Rz[6] = {0.041, 0.043, 0.058, 0.050, 0.036, 0.043};
    for (auto& s : segs) {
        int j = s.g - 1;
        par.add(std::string(kSeg[s.g]) + " 기울기 (px/단위)", K[j], s.k, (s.g == 3 || s.g == 4) ? 0.2 : 0.05, 2);
        par.add(std::string(kSeg[s.g]) + " 0점 (px)", Bz[j], s.b, 0.05, 2);
        par.add(std::string(kSeg[s.g]) + " 직선 잔차 (px)", Rz[j], s.res, 0.01, 3);
    }
    par.add("같은 축 구간 간 0점 차 최대 (px)", 0.6, zero_spread, 0.1, 2);
    par.add("T 세로 이동 (px/deg)", 0.43, kTdy, 0.05, 2);
    par.add("Y 세로 이동 (px/mm)", 0.02, kYdy, 0.02, 2);
    std::cout << "parity " << par.ok << "/" << par.total << "\n";

    // ---------- 결과.md ----------
    Table segt{{"구간", "장수", "기울기 (px/단위)", "0에서의 위치 (px)", "직선 잔차 sd (px)", "세로 기울기 (px/단위)", "0점 차를 Y로 (mm)"}};
    for (auto& s : segs)
        segt.add({kSeg[s.g], std::to_string(s.idx.size()), fmt(s.k, 3), fmts(s.b, 3), fmt(s.res, 4), fmts(s.kdy, 3), fmts(s.b / kY, 4)});
    Table fmt_t{{"항목", "값"}};
    fmt_t.add({"해상도", std::to_string(bmp_w[0]) + " x " + std::to_string(bmp_h[0]) + ", " + std::to_string(bmp_c[0]) + "채널"});
    fmt_t.add({"파일 크기", std::to_string(fbytes) + " byte" + (same_size ? ", 전 사진 동일" : ", 사진마다 다름")});
    fmt_t.add({"R=G=B", std::to_string(n_gray) + " / " + std::to_string(n)});
    fmt_t.add({"평균 밝기", fmt(all_mean, 2) + " (center " + fmt(st[ci].mean, 2) + ", 나머지 " + fmt(omin, 2) + " ~ " + fmt(omax, 2) + ")"});
    fmt_t.add({"밝기 단계 수", std::to_string(lv_min) + " ~ " + std::to_string(lv_max)});
    fmt_t.add({"포화(255) 비율", fmt(sat_min, 3) + " ~ " + fmt(sat_max, 3) + " %"});
    fmt_t.add({"0 픽셀", std::to_string(zero_px)});
    fmt_t.add({"벽 잡음 (이웃 차/√2 sd)", "중앙값 " + fmt(noise_med, 3) + ", 범위 " + fmt(*std::min_element(noise.begin(), noise.end()), 3) + " ~ " +
                                         fmt(*std::max_element(noise.begin(), noise.end()), 3)});
    fmt_t.add({"같은 파일 (sha256)", std::to_string(dup)});
    Table perimg{{"촬영 순서", "사진", "구간", "명령값", "평균 밝기", "밝기 단계", "포화 %", "벽 잡음", "파인더 dx (px)", "파인더 dy (px)", "구간 직선 잔차 (px)"}};
    for (int k = 0; k < n; ++k) {
        int i = ord[k];
        auto& s = st[i];
        perimg.add({std::to_string(k), shots[i].rel, kSeg[segment(shots[i])], fmt(shots[i].condition == "T" ? shots[i].t : shots[i].y, 2), fmt(s.mean, 3), std::to_string(s.levels),
                    fmt(s.sat, 4), fmt(s.noise, 4), fmt(s.d.x, 4), fmt(s.d.y, 4), fmt(seg_res[i], 4)});
    }
    perimg.csv(out("사진별_측정.csv"));
    std::ostringstream R;
    R << "# 결과: 정답값 신뢰도, 구간별 0점 분석\n\n"
      << "입력: `data/clean/clean_20261014` 281장. 측정량은 파인더 패턴 한 점(1846.7, 1010.3)의 48x48 템플릿 매칭 가로·세로 이동(NCC + 포물선 보간)에서 center 자신의 값(템플릿 정수 위치 때문에 생기는 +0.3 px)을 뺀 center 대비 이동이다.\n\n"
      << "촬영 순서는 원래 zip 파일 수정 시각에서 구했다. `data/clean`에는 시각이 없어 `00_데이터분석.md` 5절의 순서(center → Y 1차 + → Y 1차 − → T + → T − → Y 2차 + → Y 2차 − 반대 방향)를 규칙으로 썼다.\n\n"
      << "## 이미지 형식\n\n" << fmt_t.md() << "\n## 구간별 직선 맞춤 (파인더 가로 이동 = 기울기 x 명령값 + 0점)\n\n" << segt.md()
      << "\nY 평균 기울기 " << fmt(kY, 3) << " px/mm, T 평균 기울기 " << fmt(kT, 3) << " px/deg. 구간 간 0점 차 최대 " << fmt(zero_spread, 3) << " px = Y " << fmt(zero_spread / kY, 4)
      << " mm = T " << fmt(zero_spread / kT, 4) << " (같은 축 안). 축을 섞으면 " << fmt(all_spread, 3) << " px. center(0점 = 0)와 각 구간 0점의 차 " << fmt(cmin, 2) << " ~ " << fmt(cmax, 2) << " px.\n\n"
      << "## 세로 이동\n\nT 스캔 " << fmts(kTdy, 3) << " px/deg (T ±1에서 ±" << fmt(std::abs(kTdy), 2) << " px), Y 스캔 " << fmts(kYdy, 3) << " px/mm.\n\n"
      << "## 옮기지 않은 부분\n\n- 원래 분석 코드가 남아 있지 않다. 구간 표의 측정량이 무엇이었는지(어느 영역의 이동인지) 문서에 없어, track_all.py의 파인더 점(0번)과 같은 측정으로 다시 만들었다.\n"
      << "- 촬영 시각(58분, 장당 4.5초, 113초·15분 공백)은 `data/clean`에 없어 다시 재지 않았다.\n\n"
      << "## 원래 결과와 대조 (" << par.ok << "/" << par.total << " 일치)\n\n" << par.t.md()
      << "\n## 파일\n\n- `사진별_측정.csv`: 촬영 순서별 밝기·잡음·파인더 이동·구간 잔차\n- `이미지/`: 그림 7장\n";
    write_text(out("결과.md"), R.str());

    // ---------- 보고서.md ----------
    std::ostringstream P;
    P << "# 보고서: 정답값 신뢰도, 구간별 0점 분석 (10-12 16:34)\n\n"
      << "## 문제\n\n받은 281장과 정답값(파일 이름)이 얼마나 믿을 만한지 몰랐다.\n\n"
      << "## 해결 방법\n\n이미지 형식과 촬영 순서별 밝기를 재고, 파인더 패턴 이동을 이동 구간 6개로 나눠 구간마다 명령값에 직선(기울기·0점)을 맞췄다.\n\n"
      << "## 결과\n\n구간 안 직선 잔차는 " << fmt(rmin, 3) << "~" << fmt(rmax, 3) << " px로 작지만 구간마다 0점이 최대 " << fmt(zero_spread, 2) << " px(Y " << fmt(zero_spread / kY, 3)
      << " mm) 다르고, center만 밝기가 " << fmts(center_rel, 1) << " % 높았다.\n\n"
      << "| 지표 | 값 |\n|---|---|\n"
      << "| 사진 | " << n << "장, " << bmp_w[0] << "x" << bmp_h[0] << ", R=G=B " << n_gray << "장 |\n"
      << "| 평균 밝기 (center / 나머지) | " << fmt(st[ci].mean, 1) << " / " << fmt(omin, 1) << " ~ " << fmt(omax, 1) << " |\n"
      << "| 구간 안 직선 잔차 sd | " << fmt(rmin, 3) << " ~ " << fmt(rmax, 3) << " px |\n"
      << "| 구간 간 0점 차 최대 | " << fmt(zero_spread, 3) << " px (Y " << fmt(zero_spread / kY, 4) << " mm, T " << fmt(zero_spread / kT, 4) << ") |\n"
      << "| T 기울기 + / − | " << fmt(segs[2].k, 2) << " / " << fmt(segs[3].k, 2) << " px/deg |\n"
      << "| 세로 이동 T / Y | " << fmts(kTdy, 3) << " px/deg / " << fmts(kYdy, 3) << " px/mm |\n\n"
      << segt.md() << "\n" << img_md("01_촬영순서_밝기.png", "밝기") << "\n\n" << img_md("02_구간별0점_T.png", "T 0점") << "\n\n" << img_md("03_구간별0점_Y.png", "Y 0점") << "\n\n"
      << img_md("04_구간안_직선잔차.png", "직선 잔차") << "\n\n" << img_md("05_세로이동.png", "세로 이동") << "\n\n"
      << "## 튀는 값\n\n자기 구간 직선에서 가장 벗어난 사진은 Y는 " << nm(wY) << "(" << fmts(seg_res[wY], 3) << " px), T는 " << nm(wT) << "(" << fmts(seg_res[wT], 3)
      << " px)이고, 둘 다 구간 간 0점 차(" << fmt(zero_spread, 2) << " px)보다 작다.\n\n"
      << "| 구분 | 사진 | 구간 | 명령값 | 파인더 dx (px) | 파인더 dy (px) | 구간 직선 잔차 (px) | 밝기 |\n|---|---|---|---|---|---|---|---|\n";
    for (auto [lab, i] : std::vector<std::pair<std::string, int>>{{"Y 최대 잔차", wY}, {"T 최대 잔차", wT}, {"center", ci}})
        P << "| " << lab << " | " << nm(i) << " | " << kSeg[segment(shots[i])] << " | " << fmt(shots[i].condition == "T" ? shots[i].t : shots[i].y, 2) << " | " << fmt(st[i].d.x, 3)
          << " | " << fmt(st[i].d.y, 3) << " | " << (std::isnan(seg_res[i]) ? std::string("-") : fmts(seg_res[i], 3)) << " | " << fmt(st[i].mean, 2) << " |\n";
    P << "\n" << img_md(fY, "Y 튀는 값") << "\n\n" << img_md(fT, "T 튀는 값") << "\n\n초록 상자는 center의 파인더 템플릿 위치, 빨간 상자는 이 사진에서 잰 위치다.\n\n"
      << "## 한줄평\n\n이미지 측정은 구간 안에서 " << fmt(rmax, 3) << " px 안으로 직선을 따르지만 스테이지가 이동 방향·시점마다 다른 0점(최대 " << fmt(zero_spread, 2)
      << " px)으로 가서, 파일 이름 정답에 Y 약 " << fmt(zero_spread / kY / 2, 3) << "~" << fmt(zero_spread / kY, 3) << " mm의 오차가 섞여 있다.\n\n"
      << "## 원래 결과와 대조\n\n원래 값은 `00000.legacies/00_데이터분석.md`에서 가져왔다. 원래 코드가 남지 않아 구간 표의 측정량은 파인더 점 템플릿 매칭으로 다시 만들었다. " << par.ok << "/"
      << par.total << " 일치.\n\n" << par.t.md()
      << "\n불일치 원인:\n\n"
         "- T 구간 기울기·잔차, T 세로 이동: 원래 분석 코드가 없고 문서에 측정 영역이 적혀 있지 않다. C++은 track_all.py 0번 점(파인더 한 점)으로 다시 쟀다. 같은 점의 원래 측정(15_데이터_파인더_볼트이동량.csv finder_dx)도 T 기울기 80.1 / 80.6 px/deg, 0점 -0.85 / -0.21 px로 C++과 같고 문서 표(80.73 / 81.55)와 다르다. 문서 표는 다른 측정량에서 나왔다.\n"
         "- Y1−·Y2− 0점: 차이 0.05 px로 허용치 경계다. 위 측정량 차이와 같은 원인이다(finder_dx 원래 값 +0.10 / -0.45 px).\n"
         "- 0 픽셀 수: 문서는 \"검은색(0)으로 뭉개진 픽셀은 없음\"이라고 적었다. C++ 전수 세기로는 281장 전체에 0 값 픽셀이 3개 있다. 뭉개진 영역은 없지만 개수 0은 아니다. 문서가 어떻게 셌는지는 남아 있지 않다.\n";
    write_text(out("보고서.md"), P.str());
    return 0;
}
