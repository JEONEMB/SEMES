// 센터 사진 4장(1014 center.bmp, 1020 기준 위치/center1~3.bmp)을 원본에서 다시 측정해
// 명령값이 같은 Y=0, T=0 에서 영점이 얼마나 흔들리는지 본다.
//  측정은 joint 관측 102개(QR 27점 x 가로·세로 54 + 위·아래 가로선 48)를 쓰고,
//  기준틀은 assets/center.bmp = 1014 center.bmp 이므로 1014 는 정의상 0 이 된다.
//  측정 코드는 A19(20261104_093558)의 실험/src 를 그대로 링크한다. 그 파일은 고치지 않는다.
#include "observer_internal.hpp"
#include "hye/plot.hpp"
#include "hye/report.hpp"
#include <fstream>
#include <iostream>
#include <iterator>

using namespace alignment_detail;
using hye::Table;
using hye::fmt;
using hye::fmts;

// OpenCV 3.4 의 imread 는 한글 경로(기준 위치/)를 못 연다. 바이트를 읽어 imdecode 한다.
static cv::Mat read_gray_u8(const hye::fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("사진 없음: " + p.u8string());
    std::vector<char> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    cv::Mat m = cv::imdecode(cv::Mat(1, int(b.size()), CV_8U, b.data()), cv::IMREAD_GRAYSCALE);
    if (m.empty()) throw std::runtime_error("디코드 실패: " + p.u8string());
    return m;
}

// 고정 joint 모델(both_model.json)로 관측 102 -> (Y mm, T deg)
struct Frozen {
    cv::Mat origin, projector, initial;
    double scale[2];
    explicit Frozen(const hye::fs::path& p) {
        cv::FileStorage f(p.u8string(), cv::FileStorage::READ);
        auto m = f["model"];
        origin = node_matrix(m["origin"]);
        projector = node_matrix(m["projector"]);
        initial = node_matrix(m["initial"]);
        scale[0] = double(m["scale"][0]);
        scale[1] = double(m["scale"][1]);
    }
    cv::Point2d predict(const cv::Mat& x) const {
        cv::Mat q = (feature_map(x) - origin) * projector * initial;
        return {q.at<double>(0, 0) * scale[0], q.at<double>(0, 1) * scale[1]};
    }
};

// 표본 표준편차
static double sd(const std::vector<double>& v) {
    if (v.size() < 2) return hye::kNaN;
    double m = hye::mean(v), s = 0;
    for (double x : v) s += (x - m) * (x - m);
    return std::sqrt(s / (v.size() - 1));
}
static double span(const std::vector<double>& v) {
    return *std::max_element(v.begin(), v.end()) - *std::min_element(v.begin(), v.end());
}

struct Shot {
    std::string name, rel;
    std::vector<double> dx, dy;   // QR 27점 가로·세로 이동 (px)
    std::vector<double> line;     // 위·아래 가로선 48 블록 이동 (px)
    double sx = 0, sy = 0, resid = 0, sline = 0;
    cv::Point2d yt;
};

int main() {
    hye::begin_results();
    const hye::fs::path repo = hye::fs::u8path(REPO_DIR);
    const hye::fs::path assets = repo / hye::fs::u8path(u8"experiments/hye/20261028_1258_final_Y_T_joint/assets");
    Observer observer(assets);
    Frozen model(assets / "both_model.json");

    std::vector<Shot> shots = {
        {"1014 center", u8"clean_20261014/center.bmp"},
        {"1020 center1", u8"clean_20261020/기준 위치/center1.bmp"},
        {"1020 center2", u8"clean_20261020/기준 위치/center2.bmp"},
        {"1020 center3", u8"clean_20261020/기준 위치/center3.bmp"},
    };
    for (auto& s : shots) {
        cv::Mat g = read_gray_u8(repo / "data" / "clean" / hye::fs::u8path(s.rel));
        Observation o = observer.measure(g);
        for (int i = 0; i < 54; i += 2) { s.dx.push_back(o.values.at<double>(i)); s.dy.push_back(o.values.at<double>(i + 1)); }
        for (int i = 54; i < 102; ++i) s.line.push_back(o.values.at<double>(i));
        s.sx = hye::mean(s.dx);
        s.sy = hye::mean(s.dy);
        s.sline = hye::mean(s.line);
        std::vector<double> r;
        for (size_t k = 0; k < s.dx.size(); ++k) { r.push_back(s.dx[k] - s.sx); r.push_back(s.dy[k] - s.sy); }
        s.resid = hye::rms(r);
        s.yt = model.predict(o.values);
    }

    // 1. 사진별 측정값
    Table per;
    per.head = {"사진", "QR 평균 가로 (px)", "QR 평균 세로 (px)", "평균 뺀 점별 변형 RMS (px)", "가로선 평균 이동 (px)",
                "고정 joint Y (mm)", "고정 joint T (deg)"};
    for (auto& s : shots)
        per.add({s.name, fmts(s.sx, 4), fmts(s.sy, 4), fmt(s.resid, 4), fmts(s.sline, 4), fmts(s.yt.x, 4), fmts(s.yt.y, 4)});
    per.csv(hye::out("center_per_shot.csv"));

    // 2. 점별 원자료
    Table pts;
    pts.head = {"사진", "점", "가로 이동 (px)", "세로 이동 (px)"};
    for (auto& s : shots)
        for (size_t k = 0; k < s.dx.size(); ++k)
            pts.add({s.name, std::to_string(k + 1), fmts(s.dx[k], 4), fmts(s.dy[k], 4)});
    pts.csv(hye::out("center_points.csv"));

    // 3. 흔들림: 1020 같은 세션 3장 반복 / 1014 대비 세션 차이
    auto col = [&](int a, int b, double Shot::*f) { std::vector<double> v; for (int i = a; i <= b; ++i) v.push_back(shots[i].*f); return v; };
    auto ytcol = [&](int a, int b, bool y) { std::vector<double> v; for (int i = a; i <= b; ++i) v.push_back(y ? shots[i].yt.x : shots[i].yt.y); return v; };

    const double kYtick = 0.7124, kTtick = 0.8006;  // 한 틱당 가로 이동 (px): Y 0.05 mm, T 0.01 deg
    Table rep;
    rep.head = {"관측량", "1020 반복 3장 평균", "표준편차(표본)", "최댓값 - 최솟값", "Y 한 틱 대비", "T 한 틱 대비"};
    struct Row { const char* name; std::vector<double> v; bool px; };
    std::vector<Row> rows = {
        {"QR 평균 가로 (px)", col(1, 3, &Shot::sx), true},
        {"QR 평균 세로 (px)", col(1, 3, &Shot::sy), true},
        {"가로선 평균 이동 (px)", col(1, 3, &Shot::sline), true},
        {"고정 joint Y (mm)", ytcol(1, 3, true), false},
        {"고정 joint T (deg)", ytcol(1, 3, false), false},
    };
    for (auto& r : rows)
        rep.add({r.name, fmts(hye::mean(r.v), 4), fmt(sd(r.v), 4), fmt(span(r.v), 4),
                 r.px ? fmt(span(r.v) / kYtick, 3) : "", r.px ? fmt(span(r.v) / kTtick, 3) : ""});
    rep.csv(hye::out("center_repeat.csv"));

    Table ses;
    ses.head = {"관측량", "1014 center", "1020 평균(3장)", "차이 (1020 - 1014)", "Y 한 틱 대비", "T 한 틱 대비"};
    auto one = [&](const char* name, double v0, const std::vector<double>& v, bool px) {
        double d = hye::mean(v) - v0;
        ses.add({name, fmts(v0, 4), fmts(hye::mean(v), 4), fmts(d, 4),
                 px ? fmt(std::abs(d) / kYtick, 3) : "", px ? fmt(std::abs(d) / kTtick, 3) : ""});
    };
    one("QR 평균 가로 (px)", shots[0].sx, col(1, 3, &Shot::sx), true);
    one("QR 평균 세로 (px)", shots[0].sy, col(1, 3, &Shot::sy), true);
    one("가로선 평균 이동 (px)", shots[0].sline, col(1, 3, &Shot::sline), true);
    one("고정 joint Y (mm)", shots[0].yt.x, ytcol(1, 3, true), false);
    one("고정 joint T (deg)", shots[0].yt.y, ytcol(1, 3, false), false);
    ses.csv(hye::out("center_session.csv"));

    // 4. 그림: 점별 가로 이동, 센터별 평균 가로 이동
    {
        hye::Chart c("Center shots: per-point horizontal shift", "QR point index (1-27)", "shift x (px)");
        std::vector<double> idx;
        for (int i = 1; i <= 27; ++i) idx.push_back(i);
        for (size_t i = 0; i < shots.size(); ++i) c.line(idx, shots[i].dx, hye::color(int(i)), shots[i].name, true);
        c.hline(0);
        c.save("center_point_dx.png");
    }
    {
        hye::Chart c("Center shots: per-point vertical shift", "QR point index (1-27)", "shift y (px)");
        std::vector<double> idx;
        for (int i = 1; i <= 27; ++i) idx.push_back(i);
        for (size_t i = 0; i < shots.size(); ++i) c.line(idx, shots[i].dy, hye::color(int(i)), shots[i].name, true);
        c.hline(0);
        c.save("center_point_dy.png");
    }
    {
        std::vector<std::string> cats;
        std::vector<double> vals;
        for (auto& s : shots) { cats.push_back(s.name); vals.push_back(s.sx); }
        hye::bar_chart("center_mean_dx.png", "Center shots: mean horizontal shift (px)", cats, vals, "mean dx (px)");
    }
    {
        std::vector<std::string> cats;
        std::vector<double> vals;
        for (auto& s : shots) { cats.push_back(s.name); vals.push_back(s.yt.x); }
        hye::bar_chart("center_joint_y.png", "Center shots: frozen joint Y (mm)", cats, vals, "Y (mm)");
    }

    // 5. 결과.md / 보고서.md
    std::string md;
    md += "# 결과 - 센터 사진 4장의 영점 흔들림\n\n";
    md += "입력: `data/clean/clean_20261014/center.bmp`, `data/clean/clean_20261020/기준 위치/center1~3.bmp` (명령값 Y=0, T=0).\n";
    md += "관측: joint 102개 중 QR 27점의 가로·세로 이동과 위·아래 가로선 48블록 이동. 기준틀은 `assets/center.bmp`(= 1014 center.bmp)이므로 1014 는 정의상 0 이다.\n";
    md += "한 틱 환산은 Y 0.05 mm = 0.7124 px, T 0.01 deg = 0.8006 px 를 썼다.\n\n";
    md += "## 1. 사진별 측정값\n\n" + per.md() + "\n";
    md += "## 2. 1020 같은 세션 반복 3장\n\n" + rep.md() + "\n";
    md += "## 3. 세션 사이 (1014 대비 1020)\n\n" + ses.md() + "\n";
    md += "## 4. 점별 원자료\n\n`center_points.csv` (4장 x 27점).\n\n";
    md += "## 5. 그림\n\n" + hye::img_md("center_point_dx.png", "per-point dx") + "\n\n" +
          hye::img_md("center_point_dy.png", "per-point dy") + "\n\n" +
          hye::img_md("center_mean_dx.png", "mean dx") + "\n\n" +
          hye::img_md("center_joint_y.png", "joint Y") + "\n";
    hye::write_text(hye::out("결과.md"), md);

    const double rsx = span(col(1, 3, &Shot::sx)), dsx = hye::mean(col(1, 3, &Shot::sx)) - shots[0].sx;
    const double rY = span(ytcol(1, 3, true)), rT = span(ytcol(1, 3, false));
    std::string rp;
    rp += "# 보고서 - 센터 영점 흔들림\n\n";
    rp += "- 문제: 명령값이 같은 센터 사진끼리도 영점이 흔들리는지, 흔들린다면 한 틱에 견줘 얼마나 큰지 몰랐다.\n";
    rp += "- 해결: 센터 4장을 원본에서 다시 측정해 QR 27점 평균 이동과 고정 joint (Y, T) 를 뽑고, 같은 세션 반복과 세션 사이를 나눠 봤다.\n";
    rp += "- 결과: 같은 세션 반복 3장의 가로 평균은 " + fmt(rsx, 4) + " px 폭(Y 한 틱의 " + fmt(rsx / kYtick, 2) +
          "배) 안에 들어오고, 1014 대비 1020 세션은 " + fmts(dsx, 4) + " px 어긋난다(Y 한 틱의 " + fmt(std::abs(dsx) / kYtick, 2) + "배).\n\n";
    rp += "## 대표값\n\n" + rep.md() + "\n" + ses.md() + "\n";
    rp += "## 사진별\n\n" + per.md() + "\n";
    rp += "## 한줄평\n\n";
    rp += "QR 27점을 ECC 로 맞춰 평균을 내는 측정이라 사진 한 장 안의 잡음은 평균으로 지워지고, 남는 것은 사진끼리의 실제 자세 차이다. ";
    rp += "그래서 같은 세션 반복에서 남는 폭은 구동계 복귀와 촬영 조건의 흔들림으로, 세션 사이 차이는 그날의 영점 자체가 옮겨간 것으로 읽는다. ";
    rp += "고정 joint 로 환산하면 반복 3장의 Y 폭은 " + fmt(rY, 4) + " mm, T 폭은 " + fmt(rT, 4) + " deg 다.\n\n";
    rp += "## 그림\n\n" + hye::img_md("center_mean_dx.png", "mean dx") + "\n\n" + hye::img_md("center_point_dx.png", "per-point dx") + "\n";
    hye::write_text(hye::out("보고서.md"), rp);

    std::cout << per.md() << "\n" << rep.md() << "\n" << ses.md() << std::endl;
    return 0;
}
