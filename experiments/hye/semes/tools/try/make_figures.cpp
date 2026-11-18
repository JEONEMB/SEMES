#include "common/console.hpp"
// 보고서 그림 생성: 실촬영 사례와 합성 사례
// 사용: make_figures [모델_폴더 실촬영_폴더 출력_폴더]
// 기본 경로: data/model, data/new_photos, docs/부록_A_시각화
#include "figure_text.hpp"
#include "alignment.hpp"
#include "common/photo_files.hpp"
#include "common/reference.hpp"
#include "scene_effects.hpp"
#include "steps/01_find_qr.hpp"
#include "steps/02_find_screws.hpp"
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>

namespace fs = std::filesystem;
using namespace alignment;

namespace {

const cv::Scalar kWhite(248, 249, 250), kRed(104, 111, 255), kYellow(90, 222, 255), kGreen(144, 230, 103);
const cv::Size kFigureSize(1440, 810);

// number: 부호를 포함한 측정값 표시
//   입력: value (측정값), digits (소수 자릿수)
//   출력: 부호와 소수 자릿수를 맞춘 문자열
std::string number(double value, int digits = 4) {
    if (!std::isfinite(value)) return "-";
    std::ostringstream out;
    out << std::fixed << std::showpos << std::setprecision(digits) << value;
    return out.str();
}

struct Parts {
    QrShift qr;
    std::array<ScrewShift, 4> screws{};
    std::string search;
};

// inspect: 사진의 QR과 나사 상태 확인
//   입력: reference (기준 사진과 위치), photo (확인할 사진)
//   출력: QR 위치, 나사 위치, 측정 결과
Parts inspect(const Reference& reference, const cv::Mat& photo) {
    Parts parts;
    if (photo.size() != kPhotoRect.size()) return parts;
    const QrFinder finder(reference);
    parts.qr = finder.find_qr_shift(photo);
    parts.search = parts.qr.found ? "QR 띠 봉우리 3개, 간격 맞음" : parts.qr.problem;
    if (parts.qr.found)
        parts.screws = ScrewFinder(reference).find_screw_shifts(photo, parts.qr.dx, parts.qr.dy, {true, true, true, true});
    return parts;
}

// dashed_line: 두 점 사이 점선 표시
//   입력: image (그림), a와 b (양 끝 위치)
//   출력: 없음 (image에 점선 추가)
void dashed_line(cv::Mat& image, cv::Point a, cv::Point b) {
    const double length = cv::norm(b - a);
    for (double pos = 0; pos < length; pos += 12)
        cv::line(image, a + (b - a) * (pos / length), a + (b - a) * (std::min(pos + 7, length) / length), kRed, 1, cv::LINE_AA);
}

struct Example {
    tools::PhotoFile source;
    cv::Mat photo;
    Result result;
    std::string description;
    std::string call = "measure_alignment";
    bool edited = false;
    bool zero_photo = false;
    double zero_y = 0;
    double zero_t = 0;
    bool day_zero = false;
};

// draw_parts: 검출한 QR과 나사 위치 표시
//   입력: image (그림), reference (기준), parts (검출 결과), ink (색), 표시 설정
//   출력: 없음 (image에 위치와 이름 추가)
void draw_parts(cv::Mat& image, const Reference& reference, const Parts& parts, const cv::Scalar& ink,
                bool show_screws, bool missing_qr_only) {
    const double scale = double(image.cols) / kPhotoWidth;
    for (int k = 0; k < 3; ++k) {
        const auto& match = parts.qr.qr[k];
        if (missing_qr_only && match.found) continue;
        const auto& original = reference.qr[k];
        const cv::Point top(cvRound((original.x + (match.found ? match.dx : parts.qr.dx)) * scale),
                            cvRound((original.y + (match.found ? match.dy : parts.qr.dy)) * scale));
        const cv::Rect box(top, cv::Size(cvRound(original.width * scale), cvRound(original.height * scale)));
        const auto& color = match.found ? ink : kRed;
        if (match.found) {
            for (int x : {box.x - 3, box.br().x + 3}) {
                for (int y : {box.y - 3, box.br().y + 3}) {
                    const cv::Point corner(x, y);
                    cv::line(image, corner, corner + cv::Point(x < box.x ? 12 : -12, 0), color, 1, cv::LINE_AA);
                    cv::line(image, corner, corner + cv::Point(0, y < box.y ? 12 : -12), color, 1, cv::LINE_AA);
                }
            }
        } else {
            const cv::Point corners[] = {box.tl(), {box.br().x, box.y}, box.br(), {box.x, box.br().y}};
            for (int i = 0; i < 4; ++i) dashed_line(image, corners[i], corners[(i + 1) % 4]);
        }
        write_text(image, "QR " + std::to_string(k + 1), top - cv::Point(0, 22), 18, color);
    }
    if (!show_screws || missing_qr_only || !parts.qr.found) return;
    for (int k = 0; k < 4; ++k) {
        const auto& screw = parts.screws[k];
        const cv::Point at(cvRound((reference.screw[k].x + (screw.found ? screw.dx : parts.qr.dx)) * scale),
                           cvRound((reference.screw[k].y + (screw.found ? screw.dy : parts.qr.dy)) * scale));
        const int radius = cvRound(45 * scale);
        const auto& color = screw.found ? ink : kRed;
        if (screw.found) {
            cv::circle(image, at, radius, color, 1, cv::LINE_AA);
        } else {
            for (int angle = 0; angle < 360; angle += 45)
                cv::ellipse(image, at, {radius, radius}, 0, angle, angle + 26, color, 1, cv::LINE_AA);
        }
        write_text(image, "나사 " + std::to_string(k + 1), at - cv::Point(radius, radius + 23), 18, color);
    }
}

// draw: 단계별 측정 설명 그림 생성
//   입력: example (사례), reference (기준), parts (측정 결과), 표시 단계와 설정
//   출력: 설명과 수치를 넣은 그림
cv::Mat draw(const Example& example, const Reference& reference, const Parts& parts, int step = 8,
             bool show_parts = true) {
    cv::Mat small, image;
    cv::resize(example.photo, small, kFigureSize, 0, 0, cv::INTER_AREA);
    if (small.channels() == 1) cv::cvtColor(small, image, cv::COLOR_GRAY2BGR);
    else image = small;
    const auto& result = example.result;
    const bool failed = result.status == Status::Failed;
    const auto& color = failed ? kRed : example.zero_photo ? kGreen : kYellow;
    // 글자 뒤쪽만 어둡게 처리. 장비와 검출 위치 밝기 유지
    for (int y = 0; y < image.rows; ++y) {
        const double top = std::clamp((170.0 - y) / 65.0, 0.0, 1.0);
        const double bottom = std::clamp((y - image.rows + 126.0) / 55.0, 0.0, 1.0);
        if (top == 0 && bottom == 0) continue;
        auto* row = image.ptr<cv::Vec3b>(y);
        for (int x = 0; x < image.cols; ++x) {
            const double left = std::clamp((760.0 - x) / 210.0, 0.0, 1.0);
            const double right = std::clamp((x - 900.0) / 140.0, 0.0, 1.0);
            const double shade = 0.57 * std::max(top * std::max(left, right), bottom * left);
            row[x] *= 1.0 - shade;
        }
    }
    const bool detection_step = step == 3 || step == 4 || step == 8;
    if (show_parts && detection_step && !parts.search.empty())
        draw_parts(image, reference, parts, color, step >= 4, result.code == "E_QR_NOT_FOUND");

    write_text(image, example.description, {28, 21}, 27, kWhite);
    if (example.edited) write_text(image, "재현용 이미지", {28, 56}, 17, kWhite);
    if (example.call != "measure_alignment")
        write_text(image, example.call, {28, 81}, 17, kWhite);
    const int right = image.cols - 26, label = right - 170;
    write_text(image, example.edited ? "원본 명령값" : "라벨", {label, 18}, 20, kWhite, true);
    const char* heading = example.call == "set_reference_from_photos" ? "갱신 결과" :
                          example.day_zero && example.call == "measure_alignment" ? "영점 적용 측정값" : "측정";
    write_text(image, heading, {right, 18}, 20, kWhite, true);
    for (int axis = 0; axis < 2; ++axis) {
        const int y = 47 + axis * 30;
        write_text(image, axis == 0 ? "Y (mm)" : "T (°)", {label - 175, y}, 20, kWhite);
        const double command = axis == 0 ? example.source.command_y_mm : example.source.command_t_deg;
        write_text(image, example.source.has_command ? number(command) : "-", {label, y}, 23, kWhite, true);
        const double value = axis == 0 ? result.y_mm : result.t_deg;
        write_text(image, step >= 5 && std::isfinite(value) ? number(value) : "-", {right, y}, 23, kWhite, true);
    }
    if (step >= 5 && !example.day_zero && example.call == "measure_alignment")
        write_text(image, "촬영일 영점 미적용", {32, 101}, 17, kWhite);
    if (step >= 7) {
        const std::string status = failed ? "실패" : result.status == Status::Warning ? "경고" : "정상";
        const auto& status_color = failed ? kRed : result.status == Status::Warning ? kYellow : kWhite;
        write_text(image, "상태: " + status, {right, 110}, 19, status_color, true);
    }
    const int caption = image.rows - 68;
    if (show_parts && step == 3 && !parts.search.empty())
        write_text(image, "QR 탐색   " + parts.search, {28, caption}, 20, kWhite);
    if (show_parts && step == 4 && parts.qr.found) {
        std::string shifts;
        for (int k = 0; k < 4; ++k)
            shifts += "    " + std::to_string(k + 1) + "   " + number(parts.screws[k].found ? parts.screws[k].dx : NAN, 2);
        write_text(image, "나사 가로 이동 (px)" + shifts, {32, caption}, 20, kWhite);
    }
    if (step == 5 && !result.screws_used.empty())
        write_text(image, "계산에 사용한 나사   " + result.screws_used, {32, caption}, 20, kWhite);
    if (step == 6 && example.day_zero)
        write_text(image, "적용한 촬영일 영점   Y " + number(example.zero_y) + " mm   T " +
                   number(example.zero_t) + "°", {32, caption}, 20, kWhite);
    if (step == 7 && example.call == "measure_alignment" && std::isfinite(result.check_limit))
        write_text(image, "일치 점검값 " + number(result.check_value) + "   한계 " + number(result.check_limit), {32, caption}, 20, kWhite);
    if (step >= 7 && !result.code.empty()) {
        std::string message = result.message.substr(0, result.message.find(" / "));
        if (result.code == "W_OUT_OF_RANGE") message = "영점 차감 전 판정값이 학습 범위를 초과";
        write_text(image, result.code, {28, caption}, 22, color);
        write_text(image, message, {28, caption + 29}, 18, kWhite);
    }
    return image;
}

// frame: 여러 사례 그림을 한 장으로 배치
//   입력: output (저장 경로), pictures (사례 그림)
//   출력: 없음 (요약 그림 저장)
void frame(const fs::path& output, const std::vector<cv::Mat>& pictures) {
    if (pictures.empty()) return;
    const int w = kFigureSize.width, h = kFigureSize.height, gap = 20;
    const int cols = pictures.size() > 1 ? 2 : 1;
    const int rows = (int(pictures.size()) + cols - 1) / cols;
    cv::Mat sheet(rows * h + (rows + 1) * gap, cols * w + (cols + 1) * gap, CV_8UC3, cv::Scalar(24, 24, 24));
    for (int k = 0; k < int(pictures.size()); ++k)
        pictures[k].copyTo(sheet(cv::Rect(gap + k % cols * (w + gap), gap + k / cols * (h + gap), w, h)));
    scene::save_png(output, sheet);
}

// QR 띠 탐색 그림

const int kBandCropX = 1150, kBandCropWidth = 1500;   // px. 사진에서 QR 3개와 둘레가 다 들어오는 가로 구간
const int kBandTop = 104, kBandListHeight = 150;      // 그림 px. 제목 영역, 아래 봉우리 목록 영역
const double kBandScale = double(kFigureSize.width) / kBandCropWidth;

// dashed_box: 사각 영역을 점선으로 표시
//   입력: image (그림), box (영역), color (색)
//   출력: 없음 (image에 점선 테두리 추가)
void dashed_box(cv::Mat& image, const cv::Rect& box, const cv::Scalar& color) {
    const cv::Point corners[] = {box.tl(), {box.br().x, box.y}, box.br(), {box.x, box.br().y}};
    for (int i = 0; i < 4; ++i) {
        const cv::Point a = corners[i], b = corners[(i + 1) % 4];
        const double length = cv::norm(b - a);
        for (double pos = 0; pos < length; pos += 14)
            cv::line(image, a + (b - a) * (pos / length), a + (b - a) * (std::min(pos + 8, length) / length), color, 2, cv::LINE_AA);
    }
}

// band_figure: QR 띠 봉우리 탐색 그림 생성
//   입력: reference (기준), gray (사진), title과 subtitle (제목)
//   출력: QR 띠 조각에 봉우리 자리와 판정을 표시한 영상
cv::Mat band_figure(const Reference& reference, const cv::Mat& gray, const std::string& title, const std::string& subtitle) {
    const QrFinder finder(reference);
    const cv::Rect band = finder.qr_band();
    const int strip_height = cvRound(band.height * kBandScale);
    cv::Mat sheet(kBandTop + strip_height + kBandListHeight, kFigureSize.width, CV_8UC3, cv::Scalar(24, 24, 24));
    write_text(sheet, title, {28, 16}, 32, kWhite);
    write_text(sheet, subtitle, {28, 62}, 20, kWhite);
    const int right = sheet.cols - 28;
    write_text(sheet, "초록 = QR 봉우리 (문턱 이상)", {right, 14}, 19, kGreen, true);
    write_text(sheet, "흰 점선 = 기준 사진에서의 자리", {right, 40}, 19, kWhite, true);
    write_text(sheet, "노랑 = QR 띠 (가로는 사진 전체 폭)", {right, 66}, 19, kYellow, true);

    // 사진 좌표에서 그림 좌표로 변환
    const auto on_strip = [&](const cv::Rect& r) {
        return cv::Rect(cvRound((r.x - kBandCropX) * kBandScale), cvRound((r.y - band.y) * kBandScale), cvRound(r.width * kBandScale), cvRound(r.height * kBandScale));
    };
    cv::Mat small, strip;
    cv::resize(gray(cv::Rect(kBandCropX, band.y, kBandCropWidth, band.height)), small, cv::Size(sheet.cols, strip_height), 0, 0, cv::INTER_AREA);
    cv::cvtColor(small, strip, cv::COLOR_GRAY2BGR);
    cv::rectangle(strip, cv::Rect(0, 0, strip.cols, strip.rows), kYellow, 2, cv::LINE_AA);
    for (int k = 0; k < 3; ++k) dashed_box(strip, on_strip(reference.qr[k]), kWhite);

    // 봉우리: 문턱 이상은 초록 상자. 3개이면 왼쪽부터 QR 번호, 아니면 봉우리 번호 (번호를 정하지 않음)
    std::vector<QrFinder::Peak> peaks = finder.find_peaks(gray), qr_peaks;
    for (const auto& peak : peaks) if (peak.similarity >= kPeakThreshold) qr_peaks.push_back(peak);
    std::sort(qr_peaks.begin(), qr_peaks.end(), [](const QrFinder::Peak& a, const QrFinder::Peak& b) { return a.x < b.x; });
    for (size_t i = 0; i < qr_peaks.size(); ++i) {
        const cv::Rect box = on_strip(cv::Rect(qr_peaks[i].x, qr_peaks[i].y, reference.qr[0].width, reference.qr[0].height));
        cv::rectangle(strip, box, kGreen, 3, cv::LINE_AA);
        const std::string name = qr_peaks.size() == 3 ? "QR " : "봉우리 ";
        write_text(strip, name + std::to_string(i + 1), {box.x + 4, box.y - 28}, 20, kGreen);
    }
    strip.copyTo(sheet(cv::Rect(0, kBandTop, sheet.cols, strip_height)));

    // 아래: 봉우리 높이 목록과 판정
    int y = kBandTop + strip_height + 14;
    std::string heights = "봉우리 높이 (높은 순):";
    for (const auto& peak : peaks) {
        std::ostringstream text;
        text << std::fixed << std::setprecision(2) << peak.similarity;
        heights += "   " + text.str();
    }
    write_text(sheet, heights + "    (문턱 " + number(kPeakThreshold, 1).substr(1) + ")", {28, y}, 20, kWhite);
    const QrShift shift = finder.find_qr_shift(gray);
    y += 40;
    if (shift.found)
        write_text(sheet, "판정: QR 3개, 간격 맞음. 왼쪽부터 QR 1, 2, 3 으로 정하고 정밀 위치 계산 (평균 가로 이동 " + number(shift.dx, 2) + " px)", {28, y}, 20, kGreen);
    else
        write_text(sheet, "판정: 실패 (E_QR_NOT_FOUND). " + shift.problem, {28, y}, 20, kRed);
    return sheet;
}

} // 파일 내부

namespace {

struct DayZero {
    double y = 0, t = 0;
    std::vector<std::string> sources;
    bool command_zero = false;
};

// find_day_zero: 같은 날 촬영한 사진으로 영점 설정
//   입력: photos (사진 목록), example (설명용 사진 경로)
//   출력: 영점값과 사용 사진 목록. 확보하지 못하면 빈 결과
DayZero find_day_zero(const std::vector<tools::PhotoFile>& photos, const fs::path& example) {
    DayZero zero;
    std::vector<cv::Mat> actual_zero;
    for (const auto& photo : photos) {
        if (!tools::is_zero_command(photo)) continue;
        cv::Mat image = read_photo(photo.path.u8string());
        if (image.empty()) continue;
        actual_zero.push_back(image);
        zero.sources.push_back(photo.name);
        if (actual_zero.size() == 10) break;
    }
    if (actual_zero.size() >= 5) {
        const auto result = set_zero_from_photos(actual_zero);
        if (result.status == Status::Ok) {
            zero.y = result.y_mm;
            zero.t = result.t_deg;
            zero.command_zero = true;
            return zero;
        }
    }
    zero.sources.clear();
    // 명령 0 사진 부재 시 같은 날 근접 명령 5장으로 영점 추정
    for (const auto& photo : photos) {
        if (photo.path == example || !photo.has_command || std::abs(photo.command_y_mm) > 0.25 ||
            std::abs(photo.command_t_deg) > 0.05) continue;
        const auto result = measure_diagnostic(read_photo(photo.path.u8string()));
        if (result.status != Status::Ok || result.screws_used != "1234") continue;
        zero.y += result.y_mm - photo.command_y_mm;
        zero.t += result.t_deg - photo.command_t_deg;
        zero.sources.push_back(photo.path.lexically_relative(example.parent_path().parent_path()).generic_u8string());
        if (zero.sources.size() == 5) {
            zero.y /= 5;
            zero.t /= 5;
            set_zero_value(zero.y, zero.t);
            return zero;
        }
    }
    return {};
}

// local_shift: 나사 주변만 가로 이동
//   입력: photo (원본), center (나사 중심), dx (가로 이동 px)
//   출력: 주변 경계를 원본과 섞은 이동 사진
cv::Mat local_shift(const cv::Mat& photo, cv::Point center, double dx) {
    const cv::Rect region(center.x - 90, center.y - 90, 181, 181);
    const cv::Mat shifted = scene::shift(photo(region), dx, 0);
    cv::Mat result = photo.clone();
    // 이동 영역과 원본의 경계 22 px 혼합
    for (int y = 0; y < region.height; ++y) {
        for (int x = 0; x < region.width; ++x) {
            const double weight = std::min({x, y, region.width - 1 - x, region.height - 1 - y, 22}) / 22.0;
            auto& pixel = result.at<uchar>(region.y + y, region.x + x);
            pixel = cv::saturate_cast<uchar>((1 - weight) * pixel + weight * shifted.at<uchar>(y, x));
        }
    }
    return result;
}

// generate: 보고서용 측정 사례 그림 생성
//   입력: model (모델 폴더), input (실촬영 폴더), output (저장 폴더)
//   출력: 없음 (사례 그림과 측정 근거 표 저장)
void generate(const std::string& model, const fs::path& input, const fs::path& output) {
    Reference reference;
    std::string error;
    if (!load_reference(model, reference, error)) throw std::runtime_error(error);
    const auto initialize_model = [&] {
        const Result result = initialize(model);
        if (result.status == Status::Failed) throw std::runtime_error(result.message);
    };
    for (const char* folder : {"step", "ok", "warning", "fail"}) fs::create_directories(output / folder);
    std::ofstream evidence(output / "cases.tsv", std::ios::binary);
    evidence << "figure\tcall\tsource\tinput\tcommand_Y_mm\tcommand_T_deg\tstatus\tcode\tcondition\tzero_Y_mm\tzero_T_deg\treturned_Y_mm\treturned_T_deg\tQR_search\tscrews\tcheck_value\tcheck_limit\tmessage\n";
    evidence << std::setprecision(12);
    std::ofstream zeros(output / "zeros.tsv", std::ios::binary);
    zeros << "day\tmethod\tzero_Y_mm\tzero_T_deg\tsource\n" << std::setprecision(12);

    std::vector<cv::Mat> warnings, failures;
    const auto save = [&](const std::string& name, const Example& example, const Parts& parts, bool show_parts = true) {
        const bool failed = example.result.status == Status::Failed;
        const bool normal = example.result.code.empty();
        const std::string folder = failed ? "fail" : normal ? "ok" : "warning";
        const std::string file = folder + "/" + name + ".png";
        const cv::Mat picture = draw(example, reference, parts, 8, show_parts);
        scene::save_png(output / fs::u8path(file), picture);
        if (!normal) (failed ? failures : warnings).push_back(picture);
        const auto& result = example.result;
        evidence << file << '\t' << example.call << '\t' << example.source.path.lexically_relative(input).generic_u8string()
                 << '\t' << (example.edited ? "augmented" : "real") << '\t' << example.source.command_y_mm
                 << '\t' << example.source.command_t_deg << '\t' << int(result.status) << '\t' << result.code
                 << '\t' << example.description << '\t' << example.zero_y << '\t' << example.zero_t
                 << '\t' << result.y_mm << '\t' << result.t_deg << '\t' << parts.search << '\t' << result.screws_used
                 << '\t';
        if (example.call == "measure_alignment") evidence << result.check_value << '\t' << result.check_limit;
        else evidence << '\t';
        evidence << '\t' << result.message << '\n';
    };

    // 실촬영 정상 및 범위 초과 사례. 날짜 변경 시 모델 초기화
    const auto load_example = [&](const char* day, const char* relative, const char* description) {
        initialize_model();
        const auto folder = input / day;
        const auto path = folder / fs::u8path(relative);
        const auto photos = tools::find_photos(folder);
        const auto source = std::find_if(photos.begin(), photos.end(), [&](const auto& photo) { return photo.path == path; });
        if (source == photos.end()) throw std::runtime_error("실촬영 예제 없음: " + path.u8string());
        const DayZero zero = find_day_zero(photos, path);
        Example example{*source, read_photo(path.u8string()), {}, description};
        if (example.photo.empty()) throw std::runtime_error("예제 사진 읽기 실패: " + path.u8string());
        example.day_zero = !zero.sources.empty();
        example.zero_y = zero.y;
        example.zero_t = zero.t;
        example.zero_photo = zero.command_zero && tools::is_zero_command(*source) &&
            std::find(zero.sources.begin(), zero.sources.end(), source->name) != zero.sources.end();
        example.result = measure_diagnostic(example.photo);
        std::cout << day << " zero " << zero.y << " " << zero.t << " using";
        for (const auto& name : zero.sources) {
            std::cout << " | " << name;
            zeros << day << '\t' << (zero.command_zero ? "command_zero" : "mean_measurement_minus_command")
                  << '\t' << zero.y << '\t' << zero.t << '\t' << name << '\n';
        }
        std::cout << '\n';
        return example;
    };
    const auto range = load_example("20261020", "Y+0.05 T+2.00/Y+0.05 T+2.00 (1).bmp", "학습 범위를 벗어난 실촬영");
    if (range.result.code != "W_OUT_OF_RANGE") throw std::runtime_error("실촬영 범위 초과 예제의 반환 코드가 달라졌습니다");
    const auto range_parts = inspect(reference, range.photo);
    save(range.result.code, range, range_parts);

    const auto base = load_example("20261029", "Y+0.05 T+0.05/Y+0.05 T+0.05 (1).bmp", "실촬영 측정 결과");
    const auto parts = inspect(reference, base.photo);
    if (base.result.status != Status::Ok || !base.result.code.empty()) throw std::runtime_error("정상 예제의 반환 코드가 달라졌습니다");
    save("OK", base, parts);
    const char* step_titles[] = {"입력 사진", "회색 영상", "1단계  QR 탐색", "2단계  나사 탐색",
                                 "3단계  Y, T 계산", "촬영일 영점 적용",
                                 "4단계  결과 점검", "출력  측정값과 상태"};
    for (int step = 1; step <= 8; ++step) {
        auto example = base;
        example.description = step_titles[step - 1];
        if (step == 1) {
            std::ifstream file(example.source.path, std::ios::binary);
            const std::vector<uchar> bytes((std::istreambuf_iterator<char>(file)), {});
            example.photo = cv::imdecode(bytes, cv::IMREAD_COLOR);
        }
        scene::save_png(output / "step" / ("0" + std::to_string(step) + ".png"), draw(example, reference, parts, step));
    }

    // QR 띠 탐색: 보통 사진, 크게 옮긴 사진, QR 2 반사광 사진
    const int band_shift_px = 300;
    const cv::Rect qr2_now = reference.qr[1] + cv::Point(cvRound(parts.qr.dx), cvRound(parts.qr.dy));
    scene::save_png(output / "step" / "03_band.png", band_figure(reference, base.photo,
        "QR 띠 탐색: 봉우리 3개를 찾고 왼쪽부터 번호",
        "사진 전체 폭의 QR 띠를 1/" + std::to_string(kCoarseScale) + " 로 줄여 QR 모양 봉우리를 찾는다. 정확히 3개이고 간격이 맞아야 통과"));
    scene::save_png(output / "step" / "03_band_shift.png", band_figure(reference, scene::shift(base.photo, band_shift_px, 0),
        "큰 이동: 창을 넓혀 다시 찾지 않는다",
        "재현용 이미지 (사진을 오른쪽으로 " + std::to_string(band_shift_px) + " px 옮김). 띠가 사진 전체 폭이라 한 번에 찾는다"));
    scene::save_png(output / "step" / "03_band_glare.png", band_figure(reference, scene::glare(base.photo, (qr2_now.tl() + qr2_now.br()) / 2, 150, 255),
        "QR 하나가 망가지면 실패",
        "재현용 이미지 (QR 2 위치에 반사광). 이웃 QR 을 대신 잡지 않고 E_QR_NOT_FOUND 를 낸다"));

    std::map<std::string, bool> generated;
    const auto augmented = [&](const cv::Mat& photo, const std::string& description) {
        auto example = base;
        example.edited = true;
        example.zero_photo = false;
        example.description = description;
        example.photo = photo;
        return example;
    };
    const auto candidate = [&](const std::string& code, const cv::Mat& photo, const std::string& description) {
        if (generated[code]) return true;
        auto example = augmented(photo, description);
        example.result = measure_diagnostic(photo);
        const auto checked = inspect(reference, photo);
        std::cout << "candidate " << code << " => " << example.result.code << " | " << checked.search << '\n';
        if (example.result.code != code) return false;
        save(code, example, checked);
        return generated[code] = true;
    };

    const cv::Point qr_offset(cvRound(parts.qr.dx), cvRound(parts.qr.dy));
    const auto screw_center = [&](int k) {
        return reference.screw[k] + cv::Point(cvRound(parts.screws[k].dx), cvRound(parts.screws[k].dy));
    };
    const auto remove_two = scene::remove_screw(base.photo, screw_center(1));
    candidate("W_SCREW_MISSING", remove_two, "나사 2 제거");
    const auto coarse = scene::remove_screw(remove_two, screw_center(3));
    candidate("W_COARSE", coarse, "오른쪽 나사 2, 4 제거");

    cv::Mat disagreement;
    for (int dx : {3, -3, 5, -5, 8}) {
        const cv::Mat shifted = local_shift(base.photo, screw_center(3), dx);
        if (candidate("W_VALUES_DISAGREE", shifted, "나사 4 주변 가로 이동 " + std::to_string(dx) + " px")) {
            disagreement = shifted;
            break;
        }
    }
    // QR 하나가 반사로 망가지면 봉우리가 줄거나 정밀 위치를 얻지 못해 실패
    const cv::Point middle = (reference.qr[1].tl() + reference.qr[1].br()) / 2 + qr_offset;
    candidate("E_QR_NOT_FOUND", scene::glare(base.photo, middle, 150, 255), "QR 2 위치에 반사광");
    const cv::Mat glare = scene::glare(base.photo, middle, 440, 255);   // QR 영역 전체 반사광 (기준 갱신 실패 예제)

    auto bad_size = augmented({}, "입력 해상도 1920×1080");
    cv::resize(base.photo, bad_size.photo, {1920, 1080}, 0, 0, cv::INTER_AREA);
    bad_size.result = measure_diagnostic(bad_size.photo);
    if (bad_size.result.code == "E_PHOTO") save("E_PHOTO", bad_size, {}, false);

    auto zero = augmented(coarse, "영점 설정 입력에서 오른쪽 나사 2, 4 제거");
    zero.call = "set_zero_from_photos";
    zero.result = set_zero_from_photos({zero.photo});
    if (zero.result.code == "E_ZERO") save("E_ZERO", zero, inspect(reference, zero.photo));

    auto new_reference = augmented(glare, "기준 갱신 입력의 QR 영역 반사광");
    new_reference.call = "set_reference_from_photos";
    new_reference.result = set_reference_from_photos({new_reference.photo});
    if (new_reference.result.code == "E_REFERENCE") save("E_REFERENCE", new_reference, {}, false);

    auto reset = augmented(scene::shift(base.photo, 0, 60), "카메라 위치 변화 재현, 사진을 아래로 60 px 이동");
    reset.call = "set_reference_from_photos";
    reset.result = set_reference_from_photos({reset.photo});
    if (reset.result.code == "W_ZERO_RESET") save("W_ZERO_RESET", reset, inspect(reference, reset.photo));

    if (!disagreement.empty()) {
        const auto settings = output / "_fail_on_disagreement.json";
        std::ofstream(settings) << R"({"screws":[1,2,3,4],"fail_on_disagreement":1})";
        const Result initialized = initialize(model, settings.u8string());
        fs::remove(settings);
        if (initialized.status == Status::Failed) throw std::runtime_error(initialized.message);
        set_zero_value(base.zero_y, base.zero_t);
        auto strict = augmented(disagreement, "값 불일치 시 실패를 반환하도록 설정");
        strict.result = measure_diagnostic(strict.photo);
        if (strict.result.status == Status::Failed && strict.result.code == "W_VALUES_DISAGREE")
            save("W_VALUES_DISAGREE", strict, inspect(reference, strict.photo));
    }
    frame(output / "all_warning.png", warnings);
    frame(output / "all_fail.png", failures);
    if (!evidence || !zeros) throw std::runtime_error("사례 표를 쓰지 못했습니다");
}

} // 파일 내부

// wmain: 보고서 그림 생성: 실촬영 사례와 합성 사례
//   입력: 실행할 때 지정한 경로와 옵션(파일 상단 사용법)
//   출력: 실행 결과와 성공 여부를 나타내는 종료 코드
int wmain(int argc, wchar_t** argv) {
    tools::use_korean_console();
    if (argc != 1 && argc != 4) {
        std::cerr << "사용: make_figures [모델_폴더 실촬영_폴더 출력_폴더]\n";
        return 2;
    }
    try {
        const auto model = argc == 4 ? fs::path(argv[1]) : fs::path("data/model");
        const auto input = argc == 4 ? fs::path(argv[2]) : fs::path("data/new_photos");
        const auto output = argc == 4 ? fs::path(argv[3]) : fs::u8path("docs/부록_A_시각화");
        generate(model.u8string(), input, output);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
