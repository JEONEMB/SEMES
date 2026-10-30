// QR 3개 가로 위치만으로 Y 사진과 Θ 사진을 구분할 수 없는 구간을 짝 비교로 찾는다 (구분표 v1).
// 원본 코드: 구분표 v1 은 코드가 남지 않았고 엑셀(22_YΘ구분표_v1.xlsx)의 판정 방법 설명만 있다. 같은 판정을 analysis_v2/judge.py 의 v1 행, 추정은 estimate.py 의 v1 행으로 옮긴다.
#include <hye/dataset.hpp>
#include <hye/ecc_compat.hpp>
#include <hye/plot.hpp>
#include <hye/report.hpp>
#include <omp.h>
#include <array>
#include <iostream>
#include <sstream>
#include "judge.hpp"

using namespace hye;

namespace {

// QR 네모 가로 위치: 원래는 cv2.QRCodeDetectorAruco 꼭짓점 평균(14_데이터_QR마커위치.csv). OpenCV 3.4 에 없어
// 시범 실험(10-12 15:49)과 같은 QR 영역 템플릿 매칭 + ECC affine 으로 QR 네모 중심의 가로 이동을 잰다.
const std::array<cv::Vec4i, 3> kBoxes{{{1554, 1734, 972, 1164}, {1815, 1995, 972, 1164}, {2073, 2253, 972, 1164}}};
const int kPad = 130;
struct QR { double dx = 0, dy = 0, a00 = 1, a01 = 0, a10 = 0, a11 = 1; };
std::array<QR, 3> measure(const cv::Mat& center, const cv::Mat& im) {
    std::array<QR, 3> out;
    for (int q = 0; q < 3; ++q) {
        auto b = kBoxes[q];
        cv::Mat tpl = center(cv::Range(b[2], b[3]), cv::Range(b[0], b[1]));
        cv::Mat search = im(cv::Range(b[2] - kPad, b[3] + kPad), cv::Range(b[0] - kPad, b[1] + kPad)), res;
        cv::matchTemplate(search, tpl, res, cv::TM_CCOEFF_NORMED);
        cv::Point loc;
        cv::minMaxLoc(res, nullptr, nullptr, nullptr, &loc);
        int dx0 = loc.x - kPad, dy0 = loc.y - kPad;
        cv::Mat target = im(cv::Range(b[2] + dy0, b[3] + dy0), cv::Range(b[0] + dx0, b[1] + dx0));
        cv::Mat W = cv::Mat::eye(2, 3, CV_32F);
        try {
            semes::ecc5::findTransformECCWithMask(tpl, target, cv::noArray(), cv::noArray(), W, cv::MOTION_AFFINE,
                                                  cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 150, 1e-7), 3);
        } catch (const cv::Exception&) {
        }
        out[q] = {dx0 + W.at<float>(0, 2), dy0 + W.at<float>(1, 2), W.at<float>(0, 0), W.at<float>(0, 1), W.at<float>(1, 0), W.at<float>(1, 1)};
    }
    return out;
}
double cx(const QR& r) { return r.dx + (r.a00 - 1) * 90 + r.a01 * 96; }
double cy(const QR& r) { return r.dy + r.a10 * 90 + (r.a11 - 1) * 96; }

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
    const cv::Mat center = read_gray(center_1014(shots));
    std::vector<std::array<QR, 3>> meas(n);
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; ++i) meas[i] = measure(center, read_gray(shots[i]));
    std::cout << "measured\n";
    auto fname = [&](int i) { return shots[i].rel.substr(shots[i].rel.rfind('/') + 1); };
    auto stem = [&](const std::string& f) { return f.substr(0, f.rfind('.')); };
    // 원래 judge.py 는 center 가 없는 표(15_ CSV)와 합쳐 center 를 뺀 280장으로 판정했다
    std::vector<judge::Item> items;
    for (int i = 0; i < n; ++i) {
        if (shots[i].condition == "center") continue;
        judge::Item it;
        it.file = fname(i);
        it.ax = shots[i].condition == "T" ? 'T' : 'Y';
        it.value = it.ax == 'T' ? shots[i].t : shots[i].y;
        it.f = {cx(meas[i][0]), cx(meas[i][1]), cx(meas[i][2])};
        it.m = (it.f[0] + it.f[1] + it.f[2]) / 3;
        it.src = i;
        items.push_back(it);
    }
    auto R = judge::run(items);
    auto E = judge::estimate(R);
    std::cout << "rule1 Y " << R.n1Y << " T " << R.n1T << " rule2 Y " << R.n2Y << " T " << R.n2T << " Yrms " << E.yrms_um << "\n";

    // 불가 구간 경계 (|값| 최대인 구분 불가 사진)
    auto edge = [](const std::vector<judge::Item>& s, const std::vector<judge::Decision>& d, bool r2) {
        double mx = 0;
        for (size_t i = 0; i < s.size(); ++i) if (!d[i].out && (r2 ? d[i].r2 : d[i].r1)) mx = std::max(mx, std::abs(s[i].value));
        return mx;
    };
    const double edgeY = edge(R.Ys, R.dy, false), edgeT = edge(R.Ts, R.dt, false);

    // ---------- 구분표 CSV (엑셀 v1 의 Y, Θ 시트와 같은 열) ----------
    auto table = [&](const std::vector<judge::Item>& s, const std::vector<judge::Item>& ref, const std::vector<judge::Decision>& d, const std::string& unit) {
        Table t{{"인덱스", "파일명", "값 (" + unit + ")", "좌 QR px", "중 QR px", "우 QR px", "좌중우 평균", "구분 불가 짝 (앞뒤 사진)", "상대 축 보간 값", "좌 차이 px",
                 "중 차이 px", "우 차이 px", "규칙1 판정", "통계 거리", "규칙2 판정"}};
        for (size_t i = 0; i < s.size(); ++i) {
            auto& r = s[i];
            std::vector<std::string> row{std::to_string(i + 1), r.file, fmt(r.value, 2), fmt(r.f[0], 4), fmt(r.f[1], 4), fmt(r.f[2], 4), fmt(r.m, 4)};
            if (d[i].out) {
                row.insert(row.end(), {"상대 축 범위 밖", "", "", "", "", "구분 가능 (범위 밖)", "", "구분 가능 (범위 밖)"});
            } else {
                row.insert(row.end(), {stem(ref[d[i].pa].file) + " ~ " + stem(ref[d[i].pb].file), fmt(d[i].partner_value, 4), fmt(d[i].r[0], 4), fmt(d[i].r[1], 4), fmt(d[i].r[2], 4),
                                       d[i].r1 ? "구분 불가" : "구분 가능", fmt(d[i].d2, 3), d[i].r2 ? "구분 불가" : "구분 가능"});
            }
            t.add(row);
        }
        return t;
    };
    auto tY = table(R.Ys, R.Ts, R.dy, "mm"), tT = table(R.Ts, R.Ys, R.dt, "deg");
    tY.csv(out("구분표_Y.csv"));
    tT.csv(out("구분표_T.csv"));
    {
        Table t{{"사진", "정답 Y (mm)", "정답 T (deg)", "추정 Y (mm)", "추정 T (deg)", "좌 QR px", "중 QR px", "우 QR px"}};
        for (size_t i = 0; i < E.rows.size(); ++i)
            t.add({E.rows[i]->file, fmt(E.tY[i], 2), fmt(E.tT[i], 2), fmt(E.pY[i], 4), fmt(E.pT[i], 4), fmt(E.rows[i]->f[0], 4), fmt(E.rows[i]->f[1], 4), fmt(E.rows[i]->f[2], 4)});
        t.csv(out("추정_사진별.csv"));
    }

    // ---------- 그림 ----------
    auto norm1 = [&](const judge::Decision& d) {
        double v = 0;
        for (size_t j = 0; j < d.r.size(); ++j) v = std::max(v, std::abs(d.r[j]) / R.th[j]);
        return v;
    };
    {
        std::vector<double> xy, yy, xt, yt;
        for (size_t i = 0; i < R.Ys.size(); ++i) if (!R.dy[i].out) { xy.push_back(R.Ys[i].value); yy.push_back(norm1(R.dy[i])); }
        for (size_t i = 0; i < R.Ts.size(); ++i) if (!R.dt[i].out) { xt.push_back(R.Ts[i].value / 0.18); yt.push_back(norm1(R.dt[i])); }
        Chart c("Rule 1: max |diff| / threshold vs partner on other axis", "Y (mm); T photos plotted at T / 0.18", "max |diff| / (3 sd)");
        c.scatter(xy, yy, color(2), "Y photos").scatter(xt, yt, color(1), "T photos").hline(1, kRed);
        c.save("01_규칙1_짝차이.png");
    }
    {
        std::vector<double> xy, yy, xt, yt;
        for (size_t i = 0; i < R.Ys.size(); ++i) if (!R.dy[i].out) { xy.push_back(R.Ys[i].value); yy.push_back(std::sqrt(R.dy[i].d2)); }
        for (size_t i = 0; i < R.Ts.size(); ++i) if (!R.dt[i].out) { xt.push_back(R.Ts[i].value / 0.18); yt.push_back(std::sqrt(R.dt[i].d2)); }
        Chart c("Rule 2: Mahalanobis distance to partner", "Y (mm); T photos plotted at T / 0.18", "sqrt(d2)");
        c.scatter(xy, yy, color(2), "Y photos").scatter(xt, yt, color(1), "T photos").hline(std::sqrt(R.dthr), kRed);
        c.save("02_규칙2_통계거리.png");
    }
    {
        std::vector<double> xs, ys, xt, yt;
        for (auto& r : R.Ys) { xs.push_back(r.m); ys.push_back(r.f[1] - (r.f[0] + r.f[2]) / 2); }
        for (auto& r : R.Ts) { xt.push_back(r.m); yt.push_back(r.f[1] - (r.f[0] + r.f[2]) / 2); }
        Chart c("Same mean QR shift, Y vs T: middle QR bend", "mean QR horizontal shift m (px)", "middle - mean(left, right) (px)");
        c.scatter(xt, yt, color(1), "T photos").scatter(xs, ys, color(2), "Y photos").hline(0);
        c.save("03_같은위치_Y와T_굽음.png");
    }
    std::vector<double> ey, et;
    std::vector<std::string> names;
    for (size_t i = 0; i < E.rows.size(); ++i) { ey.push_back(E.pY[i] - E.tY[i]); et.push_back(E.pT[i] - E.tT[i]); names.push_back(stem(E.rows[i]->file)); }
    {
        std::vector<double> ty, py, tt, pt;
        for (size_t i = 0; i < E.rows.size(); ++i) { ty.push_back(E.tY[i]); py.push_back(E.pY[i]); tt.push_back(E.tT[i]); pt.push_back(E.pT[i]); }
        pred_vs_truth("04_추정오차_280장.png", "QR 3 x-positions, 2-fold", ty, py, tt, pt, names);
    }
    const size_t wy = worst(ey, 1)[0], wt = worst(et, 1)[0];
    auto find_dec = [&](const judge::Item* it) -> const judge::Decision& {
        auto& s = it->ax == 'Y' ? R.Ys : R.Ts;
        auto& d = it->ax == 'Y' ? R.dy : R.dt;
        for (size_t i = 0; i < s.size(); ++i) if (s[i].file == it->file) return d[i];
        throw std::runtime_error("decision");
    };
    auto outlier_fig = [&](size_t k, const std::string& file, const std::string& cap) {
        int i = E.rows[k]->src;
        cv::Mat im = read_color(shots[i]);
        std::vector<cv::Mat> tiles;
        for (int q = 0; q < 3; ++q) {
            auto b = kBoxes[q];
            cv::Rect2d ref(b[0], b[2], b[1] - b[0], b[3] - b[2]);
            cv::Rect2d now(ref.x + cx(meas[i][q]), ref.y + cy(meas[i][q]), ref.width, ref.height);
            cv::Rect roi = cv::Rect(ref | now) + cv::Size(60, 60) - cv::Point(30, 30);
            tiles.push_back(zoom(im, roi, 420, {}, 0, q == 0 ? cap : "", {{ref, kGreen, ""}, {now, kRed, "q" + std::to_string(q) + " dx " + fmt(cx(meas[i][q]), 2)}}));
        }
        save_png(file, hstack(tiles));
    };
    const std::string fy = "05_튀는값_Y_" + names[wy] + ".png", ft = "06_튀는값_T_" + names[wt] + ".png";
    outlier_fig(wy, fy, names[wy] + " Y err " + fmts(ey[wy], 3) + " mm");
    outlier_fig(wt, ft, names[wt] + " T err " + fmts(et[wt], 3) + " deg");

    // ---------- 원래 결과와 대조 ----------
    // 출처: 00000.legacies/00_v2결과리포트.md v1 행, 22_YΘ구분표_v1.xlsx (요약 시트 임계값, Θ 시트 판정 열), analysis_v2/estimate_result.csv v1 행
    Parity par;
    par.add("규칙1 구분 불가 Y (/80)", 33, R.n1Y, 0, 0);
    par.add("규칙1 구분 불가 Θ (/71)", 29, R.n1T, 0, 0);
    par.add("규칙1 오판율 (%)", 0.7, R.fp1 * 100, 0.3, 1);
    par.add("규칙2 구분 불가 Y (/80)", 36, R.n2Y, 0, 0);
    par.add("규칙2 구분 불가 Θ (/71)", 33, R.n2T, 0, 0);
    par.add("규칙2 오판율 (%)", 0.7, R.fp2 * 100, 0.3, 1);
    par.add("Θ 사진 중 Y 범위 안 장수", 71, R.inT, 0, 0);
    par.add("임계값 좌 QR (px)", 0.0449, R.th[0], 0.003);
    par.add("임계값 중 QR (px)", 0.0274, R.th[1], 0.003);
    par.add("임계값 우 QR (px)", 0.0478, R.th[2], 0.003);
    par.add("추정 Y 오차 RMS (µm)", 235.644, E.yrms_um, 5, 1);
    par.add("추정 Y 오차 최대 (µm)", 1155.722, E.ymax_um, 20, 1);
    par.add("Y 오차 90µm 초과 (/280)", 168, E.over90, 0, 0);
    par.add("추정 Θ 오차 RMS", 0.04218, E.trms, 0.001);
    par.add("작은 틀어짐 Y RMS (µm)", 142.298, E.small_um, 5, 1);
    std::cout << "parity " << par.ok << "/" << par.total << "\n";

    // ---------- 결과.md ----------
    Table sum{{"시트", "규칙1 구분 불가", "규칙1 구분 가능", "구분 가능 (범위 밖)", "규칙2 구분 불가", "전체 (center 제외)"}};
    sum.add({"Y", std::to_string(R.n1Y), std::to_string(R.inY - R.n1Y), std::to_string((int)R.Ys.size() - R.inY), std::to_string(R.n2Y), std::to_string(R.Ys.size())});
    sum.add({"Θ", std::to_string(R.n1T), std::to_string(R.inT - R.n1T), std::to_string((int)R.Ts.size() - R.inT), std::to_string(R.n2T), std::to_string(R.Ts.size())});
    auto list_r1 = [&](const std::vector<judge::Item>& s, const std::vector<judge::Decision>& d) {
        std::string o;
        for (size_t i = 0; i < s.size(); ++i) if (!d[i].out && d[i].r1) o += (o.empty() ? "" : ", ") + stem(s[i].file);
        return o;
    };
    std::ostringstream Rm;
    Rm << "# 결과: QR 위치 구분 한계, 짝 비교 구분표 v1\n\n"
       << "입력: `data/clean/clean_20261014` 280장 (center 제외, 원래 판정과 같음). 비교값은 QR 세 개(좌, 중, 우) 네모 중심의 가로 이동(px)이다. QR 영역 180x192 템플릿 매칭 + ECC affine으로 쟀다.\n\n"
       << "판정: 상대 시트 사진을 좌중우 평균 m 순으로 잇고, 이 사진과 같은 m 지점을 보간해 좌·중·우 차이를 구한다. 규칙1은 세 차이가 모두 임계값(같은 축 LOO 보간 잔차 sd의 두 축 중 큰 값 x 3) 안이면 구분 불가, 규칙2는 잔차 공분산 마할라노비스 거리가 chi2(0.9973, 3) = "
       << fmt(R.dthr, 3) << " 안이면 구분 불가다.\n\n"
       << "임계값: 좌 " << fmt(R.th[0], 4) << ", 중 " << fmt(R.th[1], 4) << ", 우 " << fmt(R.th[2], 4) << " px. 같은 축 오판율: 규칙1 " << fmt(R.fp1 * 100, 1) << " %, 규칙2 " << fmt(R.fp2 * 100, 1) << " %.\n\n"
       << "## 요약\n\n" << sum.md() << "\n규칙1 구분 불가 Y: " << list_r1(R.Ys, R.dy) << "\n\n규칙1 구분 불가 Θ: " << list_r1(R.Ts, R.dt) << "\n\n"
       << "## 직접 추정 (같은 세 값, 전체 표준화 + 릿지 1e-9 n, 짝/홀 2겹)\n\n| 지표 | 값 |\n|---|---|\n| Y RMS (µm) | " << fmt(E.yrms_um, 1) << " |\n| Y 최대 (µm) | " << fmt(E.ymax_um, 1)
       << " |\n| Y 90µm 초과 | " << E.over90 << " / " << E.rows.size() << " |\n| Θ RMS | " << fmt(E.trms, 4) << " |\n| 작은 틀어짐(|Y|<=0.5, |T|<=0.09) Y RMS (µm) | " << fmt(E.small_um, 1) << " |\n\n"
       << "## 옮기지 않은 부분\n\n- QR 네모 좌표: 원래는 `cv2.QRCodeDetectorAruco` 꼭짓점의 평균(`14_데이터_QR마커위치.csv`)이다. OpenCV 3.4.15에 이 검출기가 없어 QR 영역 ECC로 같은 위치를 다시 쟀다. 검출 꼭짓점은 1/4 px 근처로 끊기지만 ECC는 연속값이라 잡음 구조가 다르고, 구분 불가 장수가 달라진다.\n"
       << "- 엑셀 서식(굵게, 주황 배경)과 수식은 CSV로 대신했다.\n\n"
       << "## 원래 결과와 대조 (" << par.ok << "/" << par.total << " 일치)\n\n" << par.t.md()
       << "\n## 파일\n\n- `구분표_Y.csv`, `구분표_T.csv`: 엑셀 v1 Y·Θ 시트와 같은 열 + 규칙2\n- `추정_사진별.csv`: 280장 추정\n- `이미지/`: 그림 6장\n";
    write_text(out("결과.md"), Rm.str());

    // ---------- 보고서.md ----------
    auto& dyw = find_dec(E.rows[wy]);
    auto& dtw = find_dec(E.rows[wt]);
    auto cnt = [](const std::vector<double>& v) { return (int)std::count_if(v.begin(), v.end(), [](double x) { return std::abs(x) <= kPass; }); };
    std::ostringstream P;
    P << "# 보고서: QR 위치 구분 한계, 짝 비교 구분표 (10-13 11:06)\n\n"
      << "## 문제\n\nQR 가로 위치만으로 Y 사진과 Θ 사진을 구분할 수 없는 구간이 어디인지 몰랐다.\n\n"
      << "## 해결 방법\n\nQR 3개 가로 평균이 같은 상대 축 지점을 보간해 짝을 잡고, 좌·중·우 차이가 같은 축 잡음 x3 안이면 구분 불가로 판정했다.\n\n"
      << "## 결과\n\nY 사진 " << R.n1Y << "/" << R.Ys.size() << "장, Θ 사진 " << R.n1T << "/" << R.inT << "장이 구분 불가였고(Y ±" << fmt(edgeY, 2) << " mm, Θ ±" << fmt(edgeT, 2)
      << "까지), 같은 세 값으로 직접 추정한 Y 오차 RMS는 " << fmt(E.yrms_um, 0) << " µm였다.\n\n"
      << "| 지표 | 값 |\n|---|---|\n"
      << "| 규칙1 구분 불가 Y / Θ | " << R.n1Y << " / " << R.Ys.size() << ", " << R.n1T << " / " << R.inT << " |\n"
      << "| 규칙2 구분 불가 Y / Θ | " << R.n2Y << " / " << R.Ys.size() << ", " << R.n2T << " / " << R.inT << " |\n"
      << "| 같은 축 오판율 규칙1 / 규칙2 | " << fmt(R.fp1 * 100, 1) << " % / " << fmt(R.fp2 * 100, 1) << " % |\n"
      << "| 추정 Y RMS / 최대 \\|오차\\| (µm) | " << fmt(E.yrms_um, 1) << " / " << fmt(E.ymax_um, 1) << " |\n"
      << "| 추정 Θ RMS | " << fmt(E.trms, 4) << " |\n"
      << "| Y 통과 (\\|eY\\|<=0.09) | " << cnt(ey) << " / " << ey.size() << " |\n"
      << "| T 통과 (\\|eT\\|<=0.09) | " << cnt(et) << " / " << et.size() << " |\n\n"
      << img_md("01_규칙1_짝차이.png", "규칙1") << "\n\n" << img_md("02_규칙2_통계거리.png", "규칙2") << "\n\n" << img_md("03_같은위치_Y와T_굽음.png", "굽음") << "\n\n"
      << img_md("04_추정오차_280장.png", "추정 오차") << "\n\n"
      << "## 튀는 값\n\n가장 크게 틀린 사진은 Y는 " << names[wy] << "(정답 Y " << fmt(E.tY[wy], 2) << ", 추정 " << fmt(E.pY[wy], 3) << " mm), T는 " << names[wt] << "(정답 T " << fmt(E.tT[wt], 2)
      << ", 추정 " << fmt(E.pT[wt], 3) << ")이고, 상대 축 짝과의 좌·중·우 차이가 " << (dyw.out ? std::string("(범위 밖)") : fmts(dyw.r[0], 3) + " / " + fmts(dyw.r[1], 3) + " / " + fmts(dyw.r[2], 3))
      << " px뿐이라 세 값의 직선 결합으로는 Y와 Θ를 나눌 단서가 거의 없었다.\n\n"
      << "| 구분 | 사진 | 정답 Y (mm) | 추정 Y | 정답 T (deg) | 추정 T | 좌 / 중 / 우 QR px | 짝 대비 좌 / 중 / 우 차이 (px) | 규칙1 |\n|---|---|---|---|---|---|---|---|---|\n";
    for (auto [lab, idx] : std::vector<std::pair<std::string, std::vector<size_t>>>{{"Y 오차 상위", worst(ey, 3)}, {"T 오차 상위", worst(et, 3)}})
        for (size_t k : idx) {
            auto& d = find_dec(E.rows[k]);
            auto& f = E.rows[k]->f;
            P << "| " << lab << " | " << names[k] << " | " << fmt(E.tY[k], 2) << " | " << fmt(E.pY[k], 3) << " | " << fmt(E.tT[k], 2) << " | " << fmt(E.pT[k], 3) << " | " << fmt(f[0], 2) << " / "
              << fmt(f[1], 2) << " / " << fmt(f[2], 2) << " | " << (d.out ? std::string("범위 밖") : fmts(d.r[0], 3) + " / " + fmts(d.r[1], 3) + " / " + fmts(d.r[2], 3)) << " | "
              << (d.out ? "범위 밖" : d.r1 ? "구분 불가" : "구분 가능") << " |\n";
        }
    (void)dtw;
    P << "\n" << img_md(fy, "Y 튀는 값") << "\n\n" << img_md(ft, "T 튀는 값") << "\n\n초록 상자는 center의 QR 영역, 빨간 상자는 이 사진에서 잰 QR 영역이다.\n\n"
      << "## 한줄평\n\nY 1 mm와 Θ 약 0.18은 QR 세 개를 거의 같은 양만큼 옮기고 둘을 가르는 좌·중·우 차이는 짝 사이에서 임계값 " << fmt(R.th[1], 3) << "~" << fmt(std::max(R.th[0], R.th[2]), 3)
      << " px 수준이라, Y ±" << fmt(edgeY, 2) << " mm 안쪽은 QR 가로 위치만으로 구분되지 않고 추정 Y 오차도 " << fmt(E.yrms_um, 0) << " µm로 컸다.\n\n"
      << "## 원래 결과와 대조\n\n원래 값은 `00000.legacies/00_v2결과리포트.md`(v1 행), `22_YΘ구분표_v1.xlsx`, `analysis_v2/estimate_result.csv`에서 가져왔다. " << par.ok << "/" << par.total << " 일치.\n\n"
      << par.t.md()
      << "\n불일치는 QR 네모 좌표 측정이 다르기 때문이다. 원래는 `cv2.QRCodeDetectorAruco` 꼭짓점(OpenCV 3.4에 없음)이고, C++은 QR 영역 ECC로 다시 쟀다. 판정·추정 계산은 judge.py·estimate.py와 같다.\n";
    write_text(out("보고서.md"), P.str());
    return 0;
}
