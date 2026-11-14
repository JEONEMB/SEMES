// 화면 변화 한 가지씩: 카메라가 볼 화면을 기준 사진으로 만들고, 그 화면에서 지금 알고리즘이 어떻게 반응하는지 잰다.
//
// 사용: make_environment_images <model 폴더> <출력 폴더>
//   출력: 화면마다 env_<번호>.png (판 둘레를 잘라 절반 크기), 한눈에 보기 overview.png, 알고리즘 반응 표 results.tsv
//
// 주의
//   - 실제 촬영이 아니라 기준 사진(판이 움직이지 않은 화면)을 바꾼 합성 화면이다.
//   - 판은 그대로이므로 이상적인 출력 변화는 0 이다. 출력이 변했다면 그만큼 "화면 변화 때문에 틀린 값"이다.
//   - 그림 안의 글자는 영어다 (OpenCV 가 한글을 그리지 못한다).
#include "alignment.hpp"
#include "scene_effects.hpp"
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace fs = std::filesystem;
using namespace scene;

namespace {

const double kFocalPx = 4492;   // 카메라 초점거리 추정값 (사진으로 추정, 실험 20261102_141140)

struct Change {
    std::string id, group, title;     // title 은 그림에 쓰는 영어 이름
    std::function<cv::Mat(const cv::Mat&)> make;
};

// 로봇 팔(사각형)과 케이블(선)이 화면을 가림
cv::Mat occlude_arm(const cv::Mat& gray) {
    cv::Mat out = gray.clone();
    const std::vector<cv::Point> arm = {{700, 1500}, {1600, 1150}, {1700, 1330}, {800, 1750}};
    cv::fillConvexPoly(out, arm, cv::Scalar(55), cv::LINE_AA);
    cv::line(out, {2500, 800}, {3300, 1900}, cv::Scalar(35), 28, cv::LINE_AA);
    return out;
}

// 판 둘레 배경에 다른 물체, 표가 많음 (QR 줄과 나사 둘레는 그대로)
cv::Mat busy_background(const cv::Mat& gray, const Layout& layout) {
    cv::Mat out = gray.clone();
    std::mt19937 random(kSeed);
    std::uniform_int_distribution<int> pick_x(kView.x, kView.br().x - 200), pick_y(kView.y, kView.br().y - 120), pick_v(40, 220);
    const cv::Rect keep(layout.qr[0].x - 260, layout.qr[0].y - 120, layout.qr[2].br().x - layout.qr[0].x + 520, 520);
    for (int i = 0; i < 40; ++i) {
        const cv::Rect r(pick_x(random), pick_y(random), 60 + pick_x(random) % 160, 20 + pick_y(random) % 90);
        if ((r & keep).area() > 0) continue;
        bool near_screw = false;
        for (const cv::Point& s : layout.screw) near_screw |= std::abs(r.x + r.width / 2 - s.x) < 180 && std::abs(r.y + r.height / 2 - s.y) < 180;
        if (!near_screw) cv::rectangle(out, r, cv::Scalar(pick_v(random)), -1);
    }
    return out;
}

// 화면 변화 목록 (번호, 묶음, 그림에 쓰는 이름, 기준 사진을 바꾸는 함수)
std::vector<Change> list_changes(const Layout& L) {
    return {
        {"01", "geometry", "Distance +0.5% (scale)", [=](const cv::Mat& g) { return scale_rotate(g, 1.005, 0); }},
        {"02", "geometry", "Distance -2% (scale)", [=](const cv::Mat& g) { return scale_rotate(g, 0.98, 0); }},
        {"03", "geometry", "Other install +5% (scale)", [=](const cv::Mat& g) { return scale_rotate(g, 1.05, 0); }},
        {"04", "geometry", "Camera roll 0.05 deg", [=](const cv::Mat& g) { return scale_rotate(g, 1, 0.05); }},
        {"05", "geometry", "Camera roll 2 deg", [=](const cv::Mat& g) { return scale_rotate(g, 1, 2); }},
        {"06", "geometry", "Camera tilt 2 deg (perspective)", [=](const cv::Mat& g) { return tilt(g, 2, kFocalPx); }},
        {"07", "geometry", "Camera sag 15 px down", [=](const cv::Mat& g) { return shift(g, 0, 15); }},
        {"08", "geometry", "Camera sag 40 px down", [=](const cv::Mat& g) { return shift(g, 0, 40); }},
        {"10", "geometry", "Lens distortion (barrel)", [=](const cv::Mat& g) { return distort(g, 0.02); }},
        {"11", "target", "QR content changed (1 of 3)", [=](const cv::Mat& g) { return replace_qr(g, L.qr[1], kSeed); }},
        {"12", "target", "QR content changed (all 3)", [=](const cv::Mat& g) { cv::Mat m = g; for (int k = 0; k < 3; ++k) m = replace_qr(m, L.qr[k], kSeed + k); return m; }},
        {"13", "target", "Screw 2 missing", [=](const cv::Mat& g) { return remove_screw(g, L.screw[1]); }},
        {"14", "target", "Screw 2 different head", [=](const cv::Mat& g) { return different_screw(g, L.screw[1]); }},
        {"15", "target", "Busy background", [=](const cv::Mat& g) { return busy_background(g, L); }},
        {"16", "light", "Dark x0.4", [=](const cv::Mat& g) { return brightness(g, 0.4, 0); }},
        {"17", "light", "Bright x1.8 (saturated)", [=](const cv::Mat& g) { return brightness(g, 1.8, 0); }},
        {"18", "light", "Yellow room (low contrast)", [=](const cv::Mat& g) { return gamma(brightness(g, 0.55, 70), 1.3); }},
        {"19", "light", "Side light 0.5 -> 1.4", [=](const cv::Mat& g) { return side_light(g, 0.5, 1.4); }},
        {"20", "light", "Glare on QR 2", [=](const cv::Mat& g) { return glare(g, L.qr[1].tl() + cv::Point(90, 100), 45, 230); }},
        {"21", "light", "Glare on screw 2", [=](const cv::Mat& g) { return glare(g, L.screw[1] + cv::Point(10, -10), 25, 230); }},
        {"22", "light", "Flicker banding 15%", [=](const cv::Mat& g) { return banding(g, 0.15, 90); }},
        {"23", "environment", "Defocus sigma 2", [=](const cv::Mat& g) { return defocus(g, 2); }},
        {"24", "environment", "Defocus sigma 4", [=](const cv::Mat& g) { return defocus(g, 4); }},
        {"25", "environment", "Motion blur 15 px (horizontal)", [=](const cv::Mat& g) { return motion_blur(g, 15, true); }},
        {"26", "environment", "Motion blur 15 px (vertical)", [=](const cv::Mat& g) { return motion_blur(g, 15, false); }},
        {"27", "environment", "Heat shimmer 1.5 px", [=](const cv::Mat& g) { return heat_shimmer(g, 1.5, 120, kSeed); }},
        {"28", "environment", "Fog / condensation", [=](const cv::Mat& g) { return fog(g, 0.45, 3); }},
        {"29", "environment", "Chemical droplets", [=](const cv::Mat& g) { return spots(g, 25, 12, 40, true, kSeed); }},
        {"30", "environment", "Dust on lens", [=](const cv::Mat& g) { return spots(g, 15, 15, 45, false, kSeed + 7); }},
        {"31", "environment", "Plasma glow + window ghost", [=](const cv::Mat& g) { return glow_from_top(ghost(g, 40, 25, 0.18), 70, 600, 500); }},
        {"32", "environment", "Sensor noise sigma 12", [=](const cv::Mat& g) { return noise(g, 12, kSeed); }},
        {"33", "environment", "JPEG quality 20", [=](const cv::Mat& g) { return jpeg(g, 20); }},
        {"34", "occlusion", "Robot arm + cable", [=](const cv::Mat& g) { return occlude_arm(g); }},
    };
}

// 한눈에 보기: 작은 그림들을 5열로 붙인다
cv::Mat make_overview(const std::vector<cv::Mat>& thumbnails) {
    const int columns = 5, tw = 488, th = 210;
    const int rows = int((thumbnails.size() + columns - 1) / columns);
    cv::Mat overview(rows * th, columns * tw, CV_8UC3, cv::Scalar(255, 255, 255));
    for (size_t i = 0; i < thumbnails.size(); ++i) {
        cv::Mat small;
        cv::resize(thumbnails[i], small, cv::Size(tw - 6, th - 6), 0, 0, cv::INTER_AREA);
        small.copyTo(overview(cv::Rect(int(i % columns) * tw + 3, int(i / columns) * th + 3, tw - 6, th - 6)));
    }
    return overview;
}

}   // namespace

int main(int argc, char** argv) {
    if (argc != 3) { std::cerr << "사용: make_environment_images <model 폴더> <출력 폴더>\n"; return 2; }
    const fs::path data = fs::u8path(argv[1]), out = fs::u8path(argv[2]);
    fs::create_directories(out);
    if (alignment::initialize(data.u8string()).status == alignment::Status::Failed) return 1;
    const cv::Mat center = alignment::read_photo((data / "center.bmp").u8string());
    const Layout L = read_layout(data);

    const std::vector<Change> changes = list_changes(L);

    alignment::clear_zero();
    const alignment::Result base = alignment::measure_alignment(center);
    std::ofstream table(out / "results.tsv", std::ios::binary);
    table << std::setprecision(6) << "id\tgroup\ttitle\tstatus\tcode\tdY_mm\tdT_deg\tpass\tsilent_error\n";
    std::vector<cv::Mat> thumbnails;
    save_png(out / "env_00_reference.png", labeled_view(center, "00 Reference (no change)"));
    thumbnails.push_back(labeled_view(center, "00 Reference"));
    for (const Change& change : changes) {
        const cv::Mat image = change.make(center);
        const alignment::Result r = alignment::measure_alignment(image);
        const bool failed = r.status == alignment::Status::Failed;
        const double dy = r.y_mm - base.y_mm, dt = r.t_deg - base.t_deg;
        const bool pass = !failed && std::abs(dy) <= kToleranceYmm && std::abs(dt) <= kToleranceTdeg;
        const bool silent = !failed && r.status == alignment::Status::Ok && !pass;   // 정상이라고 했는데 틀림
        const char* status = r.status == alignment::Status::Ok ? "Ok" : r.status == alignment::Status::Warning ? "Warning" : "Failed";
        table << change.id << '\t' << change.group << '\t' << change.title << '\t' << status << '\t' << r.code << '\t'
              << (failed ? std::string("nan") : std::to_string(dy)) << '\t' << (failed ? std::string("nan") : std::to_string(dt)) << '\t' << pass << '\t' << silent << '\n';
        const cv::Mat labeled = labeled_view(image, change.id + " " + change.title);
        save_png(out / ("env_" + change.id + ".png"), labeled);
        thumbnails.push_back(labeled);
        std::cout << change.id << " " << change.title << "  " << status << " " << r.code << "  dY " << dy << "  dT " << dt << "\n";
    }
    save_png(out / "overview.png", make_overview(thumbnails));
    return 0;
}
