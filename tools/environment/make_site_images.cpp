// 환경 A~J의 합성 영상과 측정 반응
// 사용: make_site_images <model 폴더> <출력 폴더>
// 출력: site_<A~J>.png, site_overview.png, site_results.tsv
// 고정 자세의 합성 실험. 환경 물체는 측정 창 밖에 배치
#include "alignment.hpp"
#include "scene_effects.hpp"
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <vector>

namespace fs = std::filesystem;
using namespace scene;

namespace {

const double kFocalPx = 4492;
const cv::Rect kBody(1360, 1260, 1110, 590);   // 판 아래 몸체 (측정 창 밖)

struct Site {
    std::string id, title, changes;     // changes: 그림 아래에 쓰는 화면 변화 요약 (영어)
    std::function<cv::Mat(const cv::Mat&, const Layout&)> make;
    bool yellow = false;                // 그림만 노란색으로 보이게 (측정은 회색)
};

// 환경 물체 (판 아래 몸체, 위쪽 빈 곳)

// draw_rail: 천장 운반 장치의 레일과 벨트 그리기
//   입력: g (회색 사진)
//   출력: 없음 (g에 환경 물체 추가)
void draw_rail(cv::Mat& g) {
    cv::rectangle(g, cv::Rect(700, 800, 2440, 55), cv::Scalar(70), -1);
    cv::line(g, {1700, 855}, {1700, 940}, cv::Scalar(40), 10);
    cv::line(g, {2150, 855}, {2150, 940}, cv::Scalar(40), 10);
}

// draw_foup: 웨이퍼 운반 상자의 테두리 그리기
//   입력: g (회색 사진)
//   출력: 없음 (g에 환경 물체 추가)
void draw_foup(cv::Mat& g) {
    cv::rectangle(g, cv::Rect(kBody.x + 60, kBody.y + 120, kBody.width - 120, kBody.height - 160), cv::Scalar(90), 8, cv::LINE_AA);
    cv::rectangle(g, cv::Rect(kBody.x + 380, kBody.y + 80, kBody.width - 760, 50), cv::Scalar(70), -1);
}

// draw_shelf: 보관 선반 받침 그리기
//   입력: g (회색 사진)
//   출력: 없음 (g에 환경 물체 추가)
void draw_shelf(cv::Mat& g) {
    cv::rectangle(g, cv::Rect(kBody.x, kBody.y + 20, kBody.width, 28), cv::Scalar(60), -1);
    cv::rectangle(g, cv::Rect(kBody.x, kBody.br().y - 40, kBody.width, 28), cv::Scalar(60), -1);
    cv::putText(g, "SHELF 12-07", cv::Point(kBody.x + 380, kBody.y + 300), cv::FONT_HERSHEY_SIMPLEX, 2.0, cv::Scalar(40), 5, cv::LINE_AA);
}

// draw_person: 지나가는 사람의 다리 그리기
//   입력: g (회색 사진)
//   출력: 없음 (g에 환경 물체 추가)
void draw_person(cv::Mat& g) {
    cv::ellipse(g, cv::Point(kBody.x + 820, kBody.br().y), cv::Size(90, 330), 0, 0, 360, cv::Scalar(45), -1, cv::LINE_AA);
    cv::ellipse(g, cv::Point(kBody.x + 1000, kBody.br().y), cv::Size(85, 300), 0, 0, 360, cv::Scalar(50), -1, cv::LINE_AA);
}

// draw_robot_hand: 웨이퍼를 잡는 로봇 손 그리기
//   입력: g (회색 사진)
//   출력: 없음 (g에 환경 물체 추가)
void draw_robot_hand(cv::Mat& g) {
    cv::rectangle(g, cv::Rect(kBody.x + 250, kBody.y + 260, 610, 60), cv::Scalar(170), -1);
    cv::rectangle(g, cv::Rect(kBody.x + 250, kBody.y + 260, 80, 330), cv::Scalar(170), -1);
    cv::rectangle(g, cv::Rect(kBody.x + 780, kBody.y + 260, 80, 330), cv::Scalar(170), -1);
}

// draw_wafer: 웨이퍼와 칩 격자 그리기
//   입력: g (회색 사진), dies (칩 격자 표시 여부)
//   출력: 없음 (g에 환경 물체 추가)
void draw_wafer(cv::Mat& g, bool dies) {
    const cv::Point c(kBody.x + kBody.width / 2, kBody.y + 330);
    cv::ellipse(g, c, cv::Size(380, 230), 0, 0, 360, cv::Scalar(120), -1, cv::LINE_AA);
    cv::circle(g, c + cv::Point(0, 225), 12, cv::Scalar(200), -1);
    if (dies)
        for (int x = -340; x <= 340; x += 60)
            for (int y = -200; y <= 200; y += 40)
                if (std::pow(x / 380.0, 2) + std::pow(y / 230.0, 2) < 0.9) cv::rectangle(g, cv::Rect(c.x + x - 25, c.y + y - 16, 50, 32), cv::Scalar(140), 2);
}

// draw_bowl: 세정 장치의 원형 테두리 그리기
//   입력: g (회색 사진)
//   출력: 없음 (g에 환경 물체 추가)
void draw_bowl(cv::Mat& g) {
    cv::ellipse(g, cv::Point(kBody.x + kBody.width / 2, kBody.y + 330), cv::Size(520, 260), 0, 0, 360, cv::Scalar(200), 18, cv::LINE_AA);
    cv::ellipse(g, cv::Point(kBody.x + kBody.width / 2, kBody.y + 330), cv::Size(420, 200), 0, 0, 360, cv::Scalar(80), 6, cv::LINE_AA);
}

// draw_window_bolts: 진공 창 주변 볼트 그리기
//   입력: g (회색 사진)
//   출력: 없음 (g에 환경 물체 추가)
void draw_window_bolts(cv::Mat& g) {
    for (int k = 0; k < 12; ++k) {
        const double a = k * CV_PI / 6;
        const cv::Point p(int(1920 + 1230 * std::cos(a)), int(1300 + 1230 * std::sin(a)));
        if (p.y > 800 && p.y < 1850) cv::circle(g, p, 30, cv::Scalar(30), -1, cv::LINE_AA);
    }
}

// draw_gantry: 칩 부착 장치의 지지대 그리기
//   입력: g (회색 사진)
//   출력: 없음 (g에 환경 물체 추가)
void draw_gantry(cv::Mat& g) {
    cv::rectangle(g, cv::Rect(kBody.x + 480, kBody.y, 150, kBody.height), cv::Scalar(55), -1);
    for (int i = 0; i < 4; ++i) cv::rectangle(g, cv::Rect(kBody.x + 60 + i * 95, kBody.y + 380, 70, 70), cv::Scalar(190), -1);
}

// draw_glass_edges: 유리판 모서리와 반사 줄 그리기
//   입력: g (회색 사진)
//   출력: 없음 (g에 환경 물체 추가)
void draw_glass_edges(cv::Mat& g) {
    cv::line(g, {kBody.x, kBody.y + 60}, {kBody.br().x, kBody.y + 60}, cv::Scalar(230), 6, cv::LINE_AA);
    cv::line(g, {kBody.x + 100, kBody.br().y}, {kBody.x + 500, kBody.y + 80}, cv::Scalar(225), 14, cv::LINE_AA);
    cv::line(g, {kBody.x + 300, kBody.br().y}, {kBody.x + 700, kBody.y + 80}, cv::Scalar(215), 6, cv::LINE_AA);
}

// yellow_tint: 노란 조명처럼 색 변경
//   입력: bgr (컬러 사진)
//   출력: 파란색을 줄인 사진
cv::Mat yellow_tint(const cv::Mat& bgr) {
    cv::Mat out = bgr.clone();
    std::vector<cv::Mat> channels;
    cv::split(out, channels);
    channels[0] *= 0.35;   // 파랑을 줄여 노랗게
    cv::merge(channels, out);
    return out;
}

// list_sites: 설치 환경 A~J 구성
//   입력: 없음
//   출력: 환경별 이름, 변환 함수, 조명 조건
std::vector<Site> list_sites() {
    return {
        {"A", "A. OHT port teaching", "port height -0.5%, vertical shake 7 px, ceiling light, dust",
         [](const cv::Mat& g0, const Layout&) {
             cv::Mat g = spots(side_light(motion_blur(scale_rotate(g0, 0.995, 0), 7, false), 0.85, 1.15), 8, 15, 35, false, kSeed);
             draw_rail(g); draw_foup(g); return g; }},
        {"B", "B. Stocker / tower lift shelf", "shelf distance +0.5%, dim x0.7, other shelf marks",
         [](const cv::Mat& g0, const Layout&) {
             cv::Mat g = brightness(scale_rotate(g0, 1.005, 0), 0.7, 0);
             draw_shelf(g); return g; }},
        {"C", "C. AMR docking", "floor tilt: roll 0.3 deg + tilt 0.2 deg, side light, noise, person",
         [](const cv::Mat& g0, const Layout&) {
             cv::Mat g = noise(side_light(tilt(scale_rotate(g0, 1, 0.3), 0.2, kFocalPx), 1.2, 0.8), 6, kSeed);
             draw_person(g); return g; }},
        {"D", "D. EFEM / load port robot", "distance +0.5%, glare on screw 2 and plate, robot hand",
         [](const cv::Mat& g0, const Layout& l) {
             cv::Mat g = glare(glare(scale_rotate(g0, 1.005, 0), l.screw[1] + cv::Point(10, -10), 25, 230), cv::Point(2330, 1080), 40, 200);
             draw_robot_hand(g); return g; }},
        {"E", "E. Photo track (coat / bake)", "yellow light, heat shimmer 0.8 px, flicker 8%, wafer on hot plate",
         [](const cv::Mat& g0, const Layout&) {
             cv::Mat g = banding(heat_shimmer(gamma(brightness(g0, 0.55, 70), 1.3), 0.8, 120, kSeed), 0.08, 90);
             draw_wafer(g, false); return g; }, true},
        {"F", "F. Clean (single wafer wet)", "chemical droplets, light fog, bowl rim",
         [](const cv::Mat& g0, const Layout&) {
             cv::Mat g = fog(spots(g0, 20, 12, 40, true, kSeed + 3), 0.2, 2);
             draw_bowl(g); return g; }},
        {"G", "G. Etch chamber viewport", "round window, plasma glow, window ghost, defocus 1 px, distortion",
         [](const cv::Mat& g0, const Layout&) {
             cv::Mat g = vignette(defocus(distort(glow_from_top(ghost(g0, 40, 25, 0.15), 60, 600, 500), 0.01), 1.0), {1920, 1300}, 1150);
             draw_window_bolts(g); return g; }},
        {"H", "H. Test prober (cold / hot chuck)", "condensation fog, stage shake 7 px, wafer with dies",
         [](const cv::Mat& g0, const Layout&) {
             cv::Mat g = motion_blur(fog(g0, 0.3, 2.5), 7, true);
             draw_wafer(g, true); return g; }},
        {"I", "I. HB bonding (TC bonder)", "distance +0.5%, mirror bright x1.25, hot head shimmer 0.8 px, gantry",
         [](const cv::Mat& g0, const Layout&) {
             cv::Mat g = heat_shimmer(brightness(scale_rotate(g0, 1.005, 0), 1.25, 0), 0.8, 120, kSeed + 5);
             draw_gantry(g); return g; }},
        {"J", "J. Display glass stage", "glass ghost reflection, wide side light 0.6 -> 1.3, glass edges",
         [](const cv::Mat& g0, const Layout&) {
             cv::Mat g = side_light(ghost(g0, 60, 0, 0.2), 0.6, 1.3);
             draw_glass_edges(g); return g; }},
    };
}

// make_overview: 사례 그림 2열 배치
//   입력: thumbnails (작게 줄인 사례 그림 목록)
//   출력: 절반 크기의 요약 영상
cv::Mat make_overview(const std::vector<cv::Mat>& thumbnails) {
    const int columns = 2, th = thumbnails[0].rows;
    cv::Mat overview(int((thumbnails.size() + columns - 1) / columns) * th, columns * thumbnails[0].cols, CV_8UC3, cv::Scalar(255, 255, 255));
    for (size_t i = 0; i < thumbnails.size(); ++i)
        thumbnails[i].copyTo(overview(cv::Rect(int(i % columns) * thumbnails[0].cols, int(i / columns) * th, thumbnails[0].cols, th)));
    cv::Mat small;
    cv::resize(overview, small, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
    return small;
}

}   // 파일 내부

// main: 환경 A~J의 합성 영상과 측정 반응
//   입력: 실행할 때 지정한 경로와 옵션(파일 상단 사용법)
//   출력: 실행 결과와 성공 여부를 나타내는 종료 코드
int main(int argc, char** argv) {
    if (argc != 3) { std::cerr << "사용: make_site_images <model 폴더> <출력 폴더>\n"; return 2; }
    const fs::path data = fs::u8path(argv[1]), out = fs::u8path(argv[2]);
    fs::create_directories(out);
    if (alignment::initialize(data.u8string()).status == alignment::Status::Failed) return 1;
    const cv::Mat center = alignment::read_photo((data / "center.bmp").u8string());
    const Layout L = read_layout(data);

    const std::vector<Site> sites = list_sites();

    alignment::clear_zero();
    const alignment::Result base = alignment::measure_alignment(center);
    std::ofstream table(out / "site_results.tsv", std::ios::binary);
    table << std::setprecision(6) << "id\ttitle\tchanges\tstatus\tcode\tdY_mm\tdT_deg\tpass\tsilent_error\n";
    std::vector<cv::Mat> thumbnails;
    for (const Site& site : sites) {
        const cv::Mat image = site.make(center, L);
        const alignment::Result r = alignment::measure_alignment(image);
        const bool failed = r.status == alignment::Status::Failed;
        const double dy = r.y_mm - base.y_mm, dt = r.t_deg - base.t_deg;
        const bool pass = !failed && std::abs(dy) <= kToleranceYmm && std::abs(dt) <= kToleranceTdeg;
        const bool silent = !failed && r.status == alignment::Status::Ok && !pass;
        const char* status = r.status == alignment::Status::Ok ? "Ok" : r.status == alignment::Status::Warning ? "Warning" : "Failed";
        table << site.id << '\t' << site.title << '\t' << site.changes << '\t' << status << '\t' << r.code << '\t'
              << (failed ? std::string("nan") : std::to_string(dy)) << '\t' << (failed ? std::string("nan") : std::to_string(dt)) << '\t' << pass << '\t' << silent << '\n';

        cv::Mat view = labeled_view(image, site.title);
        if (site.yellow) view = yellow_tint(view);
        cv::Mat caption(60, view.cols, CV_8UC3, cv::Scalar(255, 255, 255));
        cv::putText(caption, site.changes, cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(30, 30, 30), 2, cv::LINE_AA);
        cv::Mat framed;
        cv::vconcat(view, caption, framed);
        save_png(out / ("site_" + site.id + ".png"), framed);
        thumbnails.push_back(framed);
        std::cout << site.id << "  " << status << " " << r.code << "  dY " << dy << "  dT " << dt << "\n";
    }
    save_png(out / "site_overview.png", make_overview(thumbnails));
    return 0;
}
