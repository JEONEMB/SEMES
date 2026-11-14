// 새 설치 시험: QR 내용, QR 인쇄 크기, 카메라와 판 사이 거리가 바뀐 곳에서
//   (가) 지금 기준과 계수를 그대로 쓰면 어떻게 되는지
//   (나) 바뀐 화면으로 기준 사진과 QR, 나사 위치를 새로 등록하면 움직인 사진까지 맞게 재는지
// 를 잰다. 계수(px → mm)는 바꾸지 않는다.
//
// 사용: check_new_installation <model 폴더> <예제 사진 목록 list.tsv> <기대값 expected.tsv> <작업 폴더> <출력 폴더>
//   작업 폴더에는 경우마다 새 model 폴더(기준 사진, reference.json, 계수 복사)를 만든다. 영문 경로를 쓴다.
//   출력: installation_results.tsv, installation_<경우>.png (몇 가지 경우의 화면)
//
// 바꾸는 순서 (같은 변화를 기준 사진과 움직인 사진에 똑같이)
//   1. QR 내용: 바꿀 QR 의 안쪽 칸을 무작위 무늬로 (움직인 사진에서는 QR 이 움직인 자리에)
//   2. QR 인쇄 크기: QR 3개를 둘러싼 영역을 그 가운데 둘레로 키우거나 줄임
//   3. 거리: 사진 전체를 사진 가운데 둘레로 키우거나 줄임 (카메라가 다가가거나 멀어짐)
#include "alignment.hpp"
#include "scene_effects.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

namespace fs = std::filesystem;
using namespace scene;

namespace {

const double kToleranceYmm = 0.045, kToleranceTdeg = 0.01;
const int kContentChanges[] = {0, 1, 3};          // 내용이 바뀐 QR 수
const double kQrSizes[] = {0.8, 1.0, 1.2};        // QR 인쇄 크기 배율
const double kDistanceScales[] = {0.98, 1.0, 1.02};   // 화면 배율 (0.98 = 약 2% 멀어짐)

struct Case {
    int content = 0;
    double qr_size = 1, distance = 1;
    std::string name() const {
        std::ostringstream text;
        text << "qr" << content << "_size" << std::fixed << std::setprecision(1) << qr_size << "_dist" << std::setprecision(2) << distance;
        return text.str();
    }
};

struct SamplePhoto { cv::Mat gray; double expected_y = 0, expected_t = 0, qr_shift_px = 0; };

// 기준 사진에 QR 내용, 크기 변화를 준다 (거리 변화 전)
cv::Mat change_qrs(const cv::Mat& reference, const Layout& l, const Case& c) {
    cv::Mat g = reference;
    const int order[3] = {1, 0, 2};   // 1개만 바꿀 때는 가운데 QR
    for (int k = 0; k < c.content; ++k) g = replace_qr(g, l.qr[order[k]], kSeed + order[k]);
    if (c.qr_size != 1)
        for (int k = 0; k < 3; ++k) g = scale_qr(g, l.qr[k], c.qr_size);
    return g;
}

// 사진 한 장에 같은 변화를 준다. qr_dx = 그 사진에서 QR 이 기준보다 가로로 움직인 양 (소수 px)
//   QR 내용, 크기가 바뀌면: 바뀐 기준 화면을 qr_dx 만큼 옮겨 QR 자리에 붙인다 (판이 그만큼 평행이동한 것과 같음)
//   거리가 바뀌면: 사진 전체를 사진 가운데 둘레로 키우거나 줄인다
cv::Mat apply_case(const cv::Mat& gray, const cv::Mat& changed_reference, const Layout& l, const Case& c, double qr_dx) {
    cv::Mat g = gray.clone();
    if (c.content > 0 || c.qr_size != 1) {
        const cv::Mat moved_reference = shift(changed_reference, qr_dx, 0);
        const cv::Rect photo(0, 0, g.cols, g.rows);
        const int dx = int(std::floor(qr_dx));
        for (int k = 0; k < 3; ++k) {
            const cv::Rect area = (l.qr[k] | scaled_box(l.qr[k], c.qr_size));
            const cv::Rect pasted = cv::Rect(area.x - 6 + dx, area.y - 6, area.width + 13, area.height + 12) & photo;
            moved_reference(pasted).copyTo(g(pasted));
        }
    }
    if (c.distance != 1) g = scale_rotate(g, c.distance, 0);
    return g;
}

// 기준 사진에서 QR, 나사 위치가 어디로 가는지
Layout transform_layout(const Layout& l, const Case& c, cv::Size photo) {
    const cv::Point2d center(photo.width / 2.0, photo.height / 2.0);
    auto by_distance = [&](cv::Point2d q) { return center + c.distance * (q - center); };
    Layout out;
    for (int k = 0; k < 3; ++k) {
        const cv::Rect sized = scaled_box(l.qr[k], c.qr_size);
        const cv::Point2d moved_tl = by_distance(cv::Point2d(sized.tl()));
        out.qr[k] = cv::Rect(int(std::lround(moved_tl.x)), int(std::lround(moved_tl.y)), int(std::lround(sized.width * c.distance)), int(std::lround(sized.height * c.distance)));
    }
    for (int k = 0; k < 4; ++k) {
        const cv::Point2d s = by_distance(cv::Point2d(l.screw[k]));
        out.screw[k] = cv::Point(int(std::lround(s.x)), int(std::lround(s.y)));
    }
    return out;
}

// 새 model 폴더: 바뀐 기준 사진, 새 위치, 계수와 점검식은 그대로 복사
void write_registered_data(const fs::path& from, const fs::path& to, const cv::Mat& center, const Layout& l) {
    fs::create_directories(to);
    std::vector<unsigned char> bytes;
    cv::imencode(".bmp", center, bytes);
    std::ofstream(to / "center.bmp", std::ios::binary).write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
    cv::FileStorage old_file(from.u8string() + "/reference.json", cv::FileStorage::READ);
    std::ofstream json(to / "reference.json", std::ios::binary);
    json << std::setprecision(6) << "{\n \"qr\": [";
    for (int k = 0; k < 3; ++k) json << (k ? ", " : "") << "[" << l.qr[k].x << ", " << l.qr[k].y << ", " << l.qr[k].width << ", " << l.qr[k].height << "]";
    json << "],\n \"screws\": [";
    for (int k = 0; k < 4; ++k) json << (k ? ", " : "") << "[" << l.screw[k].x << ", " << l.screw[k].y << "]";
    json << "],\n \"trained_y_mm\": [" << double(old_file["trained_y_mm"][0]) << ", " << double(old_file["trained_y_mm"][1]) << "],\n"
         << " \"trained_t_deg\": [" << double(old_file["trained_t_deg"][0]) << ", " << double(old_file["trained_t_deg"][1]) << "]\n}\n";
    for (const char* folder : {"coefficients", "checks"}) {
        fs::create_directories(to / folder);
        for (const auto& entry : fs::directory_iterator(from / folder)) fs::copy_file(entry.path(), to / folder / entry.path().filename(), fs::copy_options::overwrite_existing);
    }
}

std::vector<std::vector<std::string>> read_tsv(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    std::vector<std::vector<std::string>> rows;
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        std::vector<std::string> cells;
        std::stringstream stream(line);
        for (std::string cell; std::getline(stream, cell, '\t');) cells.push_back(cell);
        rows.push_back(cells);
    }
    return rows;
}

const char* status_name(alignment::Status s) { return s == alignment::Status::Ok ? "Ok" : s == alignment::Status::Warning ? "Warning" : "Failed"; }

}   // namespace

int main(int argc, char** argv) {
    if (argc != 6) { std::cerr << "사용: check_new_installation <model 폴더> <list.tsv> <expected.tsv> <작업 폴더> <출력 폴더>\n"; return 2; }
    const fs::path data = fs::u8path(argv[1]), work = fs::u8path(argv[4]), out = fs::u8path(argv[5]);
    fs::create_directories(work);
    fs::create_directories(out);
    if (alignment::initialize(data.u8string()).status == alignment::Status::Failed) return 1;
    const cv::Mat center = alignment::read_photo((data / "center.bmp").u8string());
    const Layout layout = read_layout(data);

    // 예제 사진: 기대값과, 지금 기준으로 잰 QR 이동 (변화를 같은 자리에 주려고)
    std::vector<SamplePhoto> samples;
    const auto expected = read_tsv(fs::u8path(argv[3]));
    for (const auto& row : read_tsv(fs::u8path(argv[2]))) {
        SamplePhoto s;
        s.gray = alignment::read_photo(row.at(1));
        for (const auto& e : expected) if (e.at(0) == row.at(0)) { s.expected_y = std::stod(e.at(1)); s.expected_t = std::stod(e.at(2)); }
        alignment::clear_zero();
        s.qr_shift_px = alignment::measure_alignment(s.gray).qr_shift_px;
        samples.push_back(s);
    }
    alignment::clear_zero();
    const alignment::Result base = alignment::measure_alignment(center);

    std::ofstream table(out / "installation_results.tsv", std::ios::binary);
    table << std::setprecision(5) << "case\tqr_changed\tqr_size\tdistance\tas_is_status\tas_is_code\tas_is_dY\tas_is_dT\tregistered\tsample_failed\tsample_pass\tsample_total\tmax_eY_mm\tmax_eT_deg\n";
    for (int content : kContentChanges)
        for (double size : kQrSizes)
            for (double distance : kDistanceScales) {
                const Case c{content, size, distance};
                // (가) 그대로 쓰면
                alignment::initialize(data.u8string());
                const cv::Mat changed_qrs = change_qrs(center, layout, c);
                const cv::Mat changed_center = c.distance != 1 ? scale_rotate(changed_qrs, c.distance, 0) : changed_qrs;
                const alignment::Result as_is = alignment::measure_alignment(changed_center);
                const bool as_is_failed = as_is.status == alignment::Status::Failed;
                // (나) 새로 등록하면
                const fs::path registered = work / c.name();
                write_registered_data(data, registered, changed_center, transform_layout(layout, c, center.size()));
                const bool ready = alignment::initialize(registered.u8string()).status != alignment::Status::Failed;
                int failed = 0, passed = 0;
                double worst_y = 0, worst_t = 0;
                if (ready)
                    for (const SamplePhoto& s : samples) {
                        const alignment::Result r = alignment::measure_alignment(apply_case(s.gray, changed_qrs, layout, c, s.qr_shift_px));
                        if (r.status == alignment::Status::Failed) { ++failed; continue; }
                        const double ey = r.y_mm - s.expected_y, et = r.t_deg - s.expected_t;
                        worst_y = std::max(worst_y, std::abs(ey));
                        worst_t = std::max(worst_t, std::abs(et));
                        passed += std::abs(ey) <= kToleranceYmm && std::abs(et) <= kToleranceTdeg;
                    }
                table << c.name() << '\t' << content << '\t' << size << '\t' << distance << '\t' << status_name(as_is.status) << '\t' << as_is.code << '\t'
                      << (as_is_failed ? std::string("nan") : std::to_string(as_is.y_mm - base.y_mm)) << '\t' << (as_is_failed ? std::string("nan") : std::to_string(as_is.t_deg - base.t_deg)) << '\t'
                      << ready << '\t' << failed << '\t' << passed << '\t' << samples.size() << '\t' << worst_y << '\t' << worst_t << '\n';
                std::cout << c.name() << "  그대로: " << status_name(as_is.status) << " " << as_is.code << "  |  등록 후: 통과 " << passed << "/" << samples.size()
                          << ", 실패 " << failed << ", 최대 오차 Y " << worst_y << " T " << worst_t << "\n";
                if ((size != 1 && distance == 1 && content == 0) || (content == 3 && size == 1.2 && distance == 0.98) || (content == 1 && size == 1 && distance == 1))
                    save_png(out / ("installation_" + c.name() + ".png"), labeled_view(changed_center, "QR changed " + std::to_string(content) + ", QR size x" + std::to_string(size).substr(0, 3) + ", distance x" + std::to_string(distance).substr(0, 4)));
            }
    return 0;
}
