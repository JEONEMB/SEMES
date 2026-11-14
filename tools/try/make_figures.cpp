// 문서(docs/)에 넣는 그림을 실제 사진으로 만든다.
// 사용: make_figures <model 폴더> <Y 가 큰 사진> <T 가 큰 사진> <출력 폴더>
//   예: make_figures data/model data/sample/sample_3.png data/sample/sample_5.png docs/images
// 그림 안의 글자는 영어다 (OpenCV 가 한글을 그리지 못한다). 설명은 문서 본문에 한글로 적는다.
#include "common/reference.hpp"
#include "steps/01_find_qr.hpp"
#include "steps/02_find_screws.hpp"
#include "alignment.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace alignment;

namespace {

const cv::Scalar kGreen(60, 200, 60), kRed(50, 50, 230), kBlue(230, 140, 40), kYellow(40, 220, 240), kWhite(255, 255, 255), kBlack(0, 0, 0);

void save(const std::string& folder, const std::string& name, const cv::Mat& picture) {
    std::vector<unsigned char> bytes;
    cv::imencode(".png", picture, bytes);
    std::ofstream(folder + "/" + name, std::ios::binary).write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
    std::cout << name << " " << picture.cols << "x" << picture.rows << "\n";
}

cv::Mat to_color(const cv::Mat& gray) {
    cv::Mat color;
    cv::cvtColor(gray, color, cv::COLOR_GRAY2BGR);
    return color;
}

// 글자 뒤에 어두운 바탕을 깔아 어디서나 읽히게 한다.
void label(cv::Mat& picture, const std::string& text, cv::Point at, const cv::Scalar& color, double size = 1.0) {
    int baseline = 0;
    const cv::Size box = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, size, 2, &baseline);
    cv::rectangle(picture, cv::Rect(at.x - 4, at.y - box.height - 6, box.width + 8, box.height + baseline + 8), kBlack, -1);
    cv::putText(picture, text, at, cv::FONT_HERSHEY_SIMPLEX, size, color, 2, cv::LINE_AA);
}

std::string number(double value, int digits = 2) {
    std::ostringstream text;
    text << std::fixed << std::setprecision(digits) << value;
    return text.str();
}

// 그림 1: 사진에서 보는 곳 (QR 3개, 나사 4개, QR 탐색 창)
cv::Mat figure_overview(const Reference& reference) {
    const cv::Rect view(700, 800, 2440, 1050);
    cv::Mat picture = to_color(reference.photo);
    for (int k = 0; k < 3; ++k) {
        const cv::Rect box = reference.qr[k];
        cv::rectangle(picture, cv::Rect(box.x - 110, box.y - 22, box.width + 220, box.height + 44), kYellow, 2);
        cv::rectangle(picture, box, kGreen, 4);
        label(picture, "QR " + std::to_string(k + 1), cv::Point(box.x, box.y - 40), kGreen, 1.3);
    }
    for (int k = 0; k < 4; ++k) {
        cv::circle(picture, reference.screw[k], 45, kRed, 4);
        label(picture, "screw " + std::to_string(k + 1), reference.screw[k] + cv::Point(-90, -65), kRed, 1.3);
    }
    label(picture, "yellow: QR search window (+-110 px)", cv::Point(view.x + 30, view.y + 60), kYellow, 1.3);
    return picture(view).clone();
}

// 그림 2: 판이 움직이면 QR 과 나사가 어떻게 움직이나 (화살표는 20배 확대)
cv::Mat figure_motion(const Reference& reference, const cv::Mat& photo, const std::string& title) {
    const QrFinder qr_finder(reference);
    const ScrewFinder screw_finder(reference);
    const QrShift qr = qr_finder.find_qr_shift(photo);
    const auto screws = screw_finder.find_screw_shifts(photo, qr.dx, qr.dy, {true, true, true, true});
    const cv::Rect view(700, 800, 2440, 1050);
    cv::Mat picture = to_color(photo);
    // 화살표가 그림 안에 들어오도록 확대 배율을 정한다 (가장 긴 화살표가 약 600 px)
    const double magnify = std::max(1.0, std::floor(std::min(20.0, 600.0 / std::max(1.0, std::abs(qr.dx)))));
    for (int k = 0; k < 3; ++k) {
        const cv::Point center = (reference.qr[k].tl() + reference.qr[k].br()) / 2;
        cv::arrowedLine(picture, center, center + cv::Point(int(qr.dx * magnify), 0), kGreen, 8, cv::LINE_AA, 0, 0.08);
    }
    label(picture, "QR moved " + number(qr.dx) + " px", cv::Point(1640, 1230), kGreen, 1.4);
    for (int k = 0; k < 4; ++k) {
        if (!screws[k].found) continue;
        cv::arrowedLine(picture, reference.screw[k], reference.screw[k] + cv::Point(int(screws[k].dx * magnify), 0), kRed, 8, cv::LINE_AA, 0, 0.08);
        label(picture, number(screws[k].dx) + " px", reference.screw[k] + cv::Point(-90, 110), kRed, 1.3);
    }
    double screw_mean = 0;
    for (const ScrewShift& screw : screws) screw_mean += screw.dx / 4;
    label(picture, title, cv::Point(view.x + 30, view.y + 60), kWhite, 1.5);
    label(picture, "QR - screws = " + number(qr.dx - screw_mean) + " px   (arrows x" + number(magnify, 0) + ")", cv::Point(view.x + 30, view.y + 130), kYellow, 1.4);
    return picture(view).clone();
}

// 그림 3: 1단계. 기준 QR 조각, 새 사진의 탐색 창, 닮은 정도 지도와 가장 닮은 자리
cv::Mat figure_qr_step(const Reference& reference, const cv::Mat& photo) {
    const cv::Rect box = reference.qr[0];
    const cv::Rect window(box.x - 110, box.y - 22, box.width + 220, box.height + 44);
    cv::Mat blurred_reference, blurred_photo, similarity;
    cv::GaussianBlur(reference.photo, blurred_reference, cv::Size(0, 0), 2.0);
    cv::GaussianBlur(photo, blurred_photo, cv::Size(0, 0), 2.0);
    cv::matchTemplate(blurred_photo(window), blurred_reference(box), similarity, cv::TM_CCOEFF_NORMED);
    double best = 0;
    cv::Point peak;
    cv::minMaxLoc(similarity, nullptr, &best, nullptr, &peak);

    cv::Mat template_picture = to_color(blurred_reference(box));
    cv::Mat window_picture = to_color(blurred_photo(window));
    cv::rectangle(window_picture, cv::Rect(peak.x, peak.y, box.width, box.height), kGreen, 2);
    cv::rectangle(window_picture, cv::Rect(110, 22, box.width, box.height), kYellow, 1);

    cv::Mat map8, map_color;
    cv::normalize(similarity, map8, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::applyColorMap(map8, map_color, cv::COLORMAP_JET);
    cv::resize(map_color, map_color, cv::Size(), 2, 4, cv::INTER_NEAREST);
    cv::circle(map_color, cv::Point(peak.x * 2, peak.y * 4), 9, kWhite, 2);

    const int gap = 30, top = 70;
    cv::Mat canvas(top + std::max({template_picture.rows, window_picture.rows, map_color.rows}) + 20, template_picture.cols + window_picture.cols + map_color.cols + 4 * gap, CV_8UC3, cv::Scalar(30, 30, 30));
    int x = gap;
    template_picture.copyTo(canvas(cv::Rect(x, top, template_picture.cols, template_picture.rows)));
    label(canvas, "A. reference", cv::Point(x, 40), kWhite, 0.7);
    x += template_picture.cols + gap;
    window_picture.copyTo(canvas(cv::Rect(x, top, window_picture.cols, window_picture.rows)));
    label(canvas, "B. new photo (green = found)", cv::Point(x, 40), kWhite, 0.7);
    x += window_picture.cols + gap;
    map_color.copyTo(canvas(cv::Rect(x, top, map_color.cols, map_color.rows)));
    label(canvas, "C. similarity map (peak " + number(best, 3) + ")", cv::Point(x, 40), kWhite, 0.7);
    return canvas;
}

// 그림 4: 2단계. 나사 조각, 테두리 그림, 맞추는 원
cv::Mat figure_screw_step(const Reference& reference) {
    const cv::Point center = reference.screw[0];
    const cv::Rect area(center.x - 70, center.y - 70, 141, 141);
    cv::Mat as_float, blurred, slope_x, slope_y, strength, strength8;
    reference.photo(area).convertTo(as_float, CV_32F);
    cv::GaussianBlur(as_float, blurred, cv::Size(0, 0), 1.5);
    cv::Sobel(blurred, slope_x, CV_32F, 1, 0, 3);
    cv::Sobel(blurred, slope_y, CV_32F, 0, 1, 3);
    cv::magnitude(slope_x, slope_y, strength);
    cv::normalize(strength, strength8, 0, 255, cv::NORM_MINMAX, CV_8U);

    cv::Mat patch = to_color(reference.photo(area)), edges = to_color(strength8), masked = to_color(strength8);
    cv::rectangle(patch, cv::Rect(70 - 45, 70 - 45, 91, 91), kYellow, 1);
    cv::circle(masked, cv::Point(70, 70), 45, kRed, 1);
    const int zoom = 3, gap = 30, top = 70;
    cv::Mat canvas(top + 141 * zoom + 20, 3 * 141 * zoom + 4 * gap, CV_8UC3, cv::Scalar(30, 30, 30));
    const char* titles[3] = {"A. screw 1 (yellow = 91 px patch)", "B. edge picture", "C. circle used for matching"};
    cv::Mat pictures[3] = {patch, edges, masked};
    for (int i = 0; i < 3; ++i) {
        cv::Mat big;
        cv::resize(pictures[i], big, cv::Size(), zoom, zoom, cv::INTER_NEAREST);
        const int x = gap + i * (141 * zoom + gap);
        big.copyTo(canvas(cv::Rect(x, top, big.cols, big.rows)));
        label(canvas, titles[i], cv::Point(x, 40), kWhite, 0.7);
    }
    return canvas;
}

// 그림 5: 탐색 창 세 단계 (좁게, 넓게, 사진 전체 폭)
cv::Mat figure_search_windows(const Reference& reference) {
    cv::Mat picture = to_color(reference.photo);
    const cv::Rect box = reference.qr[1];
    cv::rectangle(picture, cv::Rect(0, box.y - 22, kPhotoWidth, box.height + 44), kRed, 6);
    cv::rectangle(picture, cv::Rect(box.x - 250, box.y - 22, box.width + 500, box.height + 44), kBlue, 6);
    cv::rectangle(picture, cv::Rect(box.x - 110, box.y - 22, box.width + 220, box.height + 44), kYellow, 6);
    cv::rectangle(picture, box, kGreen, 5);
    label(picture, "1. +-110 px", cv::Point(box.x - 110, box.y - 60), kYellow, 1.6);
    label(picture, "2. +-250 px", cv::Point(box.x - 250, box.y + box.height + 110), kBlue, 1.6);
    label(picture, "3. full width", cv::Point(60, box.y - 60), kRed, 1.6);
    cv::Mat small;
    cv::resize(picture(cv::Rect(0, 760, kPhotoWidth, 700)), small, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
    return small;
}

// 그림 7: 나사 하나를 가렸을 때. 가린 사진과 그때의 결과를 함께 보인다.
cv::Mat figure_covered_screw(const std::string& data, const cv::Mat& photo) {
    alignment::initialize(data);
    const alignment::Result before = alignment::measure_alignment(photo);
    cv::Mat covered = photo.clone();
    const cv::Rect cover(2864 - 200, 1358 - 80, 400, 160);   // 나사 2 주변
    covered(cover).setTo(128);
    const alignment::Result after = alignment::measure_alignment(covered);

    const cv::Rect view(700, 800, 2440, 1050);
    cv::Mat picture = to_color(covered);
    cv::rectangle(picture, cover, kRed, 4);
    label(picture, "screw 2 covered", cv::Point(cover.x - 60, cover.y - 30), kRed, 1.3);
    label(picture, "before: Y " + number(before.y_mm, 4) + " mm, T " + number(before.t_deg, 4) + " deg (screws " + before.screws_used + ")", cv::Point(view.x + 30, view.y + 60), kWhite, 1.3);
    label(picture, "after : Y " + number(after.y_mm, 4) + " mm, T " + number(after.t_deg, 4) + " deg (screws " + after.screws_used + ", " + after.code + ")", cv::Point(view.x + 30, view.y + 130), kYellow, 1.3);
    cv::Mat small;
    cv::resize(picture(view), small, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
    return small;
}

}   // namespace

int main(int argc, char** argv) {
    if (argc != 5) { std::cerr << "사용: make_figures <model 폴더> <Y 가 큰 사진> <T 가 큰 사진> <출력 폴더>\n"; return 2; }
    Reference reference;
    std::string error;
    if (!load_reference(argv[1], reference, error)) { std::cerr << error << "\n"; return 1; }
    const cv::Mat y_photo = alignment::read_photo(argv[2]), t_photo = alignment::read_photo(argv[3]);
    if (y_photo.empty() || t_photo.empty()) { std::cerr << "사진을 읽지 못했습니다\n"; return 1; }
    const std::string out = argv[4];

    cv::Mat overview;
    cv::resize(figure_overview(reference), overview, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
    save(out, "01_what_we_look_at.png", overview);
    cv::Mat moved_y, moved_t;
    cv::resize(figure_motion(reference, y_photo, "Y moved (plate slides)"), moved_y, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
    cv::resize(figure_motion(reference, t_photo, "T moved (plate rotates)"), moved_t, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
    save(out, "02_y_motion.png", moved_y);
    save(out, "03_t_motion.png", moved_t);
    save(out, "04_step1_qr.png", figure_qr_step(reference, y_photo));
    save(out, "05_step2_screw.png", figure_screw_step(reference));
    save(out, "06_search_windows.png", figure_search_windows(reference));
    save(out, "07_covered_screw.png", figure_covered_screw(argv[1], y_photo));
    return 0;
}
