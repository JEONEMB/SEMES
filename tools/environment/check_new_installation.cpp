#include "common/console.hpp"
// QR 내용, 인쇄 크기, 촬영 거리 변경 후 기준 재등록 효과 비교
// 사용: check_new_installation <model 폴더> <list.tsv> <expected.tsv> <작업 폴더> <출력 폴더>
// 변환 순서: QR 내용 -> QR 크기 -> 화면 배율. 이동량 계산식 유지
// 출력: installation_results.tsv, installation_<경우>.png
// 작업 폴더: 경우별 모델 생성용 영문 경로
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

const int kContentChanges[] = {0, 1, 3};          // 내용이 바뀐 QR 수
const double kQrSizes[] = {0.8, 1.0, 1.2};        // QR 인쇄 크기 배율
const double kDistanceScales[] = {0.98, 1.0, 1.02};   // 화면 배율 (0.98 = 약 2% 멀어짐)

struct Case {
    int content = 0;
    double qr_size = 1, distance = 1;
    // name: 실험 조건을 파일명으로 표시
    //   입력: 없음 (QR 변경 수, 크기, 거리 조건 사용)
    //   출력: 조건을 구분할 이름 문자열
    std::string name() const {
        std::ostringstream text;
        text << "qr" << content << "_size" << std::fixed << std::setprecision(1) << qr_size << "_dist" << std::setprecision(2) << distance;
        return text.str();
    }
};

struct SamplePhoto { cv::Mat gray; double expected_y = 0, expected_t = 0, qr_shift_px = 0; };

// change_qrs: 기준 사진의 QR 내용과 크기 변경
//   입력: reference (기준 사진), l (QR과 나사 위치), c (변경 조건)
//   출력: 변환한 기준 영상
cv::Mat change_qrs(const cv::Mat& reference, const Layout& l, const Case& c) {
    cv::Mat g = reference;
    const int order[3] = {1, 0, 2};   // 1개만 바꿀 때는 가운데 QR
    for (int k = 0; k < c.content; ++k) g = replace_qr(g, l.qr[order[k]], kSeed + order[k]);
    if (c.qr_size != 1)
        for (int k = 0; k < 3; ++k) g = scale_qr(g, l.qr[k], c.qr_size);
    return g;
}

// apply_case: 사진에 동일 환경 변화 적용
//   입력: gray (사진), changed_reference (바꾼 기준 사진), l (위치), c (조건), qr_dx (QR 이동 px)
//   출력: QR 변환 후 화면 배율을 적용한 영상
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

// transform_layout: 변환 후 QR과 나사 좌표 계산
//   입력: l (원래 위치), c (변경 조건), photo (영상 크기)
//   출력: Layout (변환 후 QR과 나사 위치)
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

// write_registered_data: 실험용 모델 저장
//   입력: from과 to (원본과 저장 폴더), center (새 기준 사진), l (새 위치)
//   출력: 없음 (기준 교체, 계산식과 점검식 복사)
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

// read_tsv: 탭으로 구분한 표 읽기
//   입력: path (파일 경로)
//   출력: 행별 문자열 목록
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

// status_name: 측정 상태를 글자로 표시
//   입력: status (정상, 경고, 실패 상태)
//   출력: 상태 이름 문자열
const char* status_name(alignment::Status s) { return s == alignment::Status::Ok ? "Ok" : s == alignment::Status::Warning ? "Warning" : "Failed"; }

// read_samples: 예제와 예상값 읽기
//   입력: list_file (사진 목록 파일), expected_file (정답 파일)
//   출력: 사진별 예상값과 기존 QR 이동량
std::vector<SamplePhoto> read_samples(const fs::path& list_file, const fs::path& expected_file) {
    std::vector<SamplePhoto> samples;
    const auto expected = read_tsv(expected_file);
    for (const auto& row : read_tsv(list_file)) {
        SamplePhoto s;
        s.gray = alignment::read_photo(row.at(1));
        for (const auto& e : expected)
            if (e.at(0) == row.at(0)) { s.expected_y = std::stod(e.at(1)); s.expected_t = std::stod(e.at(2)); }
        alignment::clear_zero();
        s.qr_shift_px = alignment::measure_diagnostic(s.gray).qr_shift_px;
        samples.push_back(s);
    }
    return samples;
}

// 새로 등록한 뒤 예제 사진을 잰 결과
struct SampleScore {
    int failed = 0, passed = 0;
    double worst_y = 0, worst_t = 0;   // 예상값과의 차이 최대
};

// score_samples: 환경을 바꾼 사진의 측정 오차 확인
//   입력: samples (예제), changed_qrs (바꾼 기준), layout (위치), c (변경 조건)
//   출력: 실패 수와 기대값 대비 최대 차이
SampleScore score_samples(const std::vector<SamplePhoto>& samples, const cv::Mat& changed_qrs, const Layout& layout, const Case& c) {
    SampleScore score;
    for (const SamplePhoto& s : samples) {
        const alignment::Result r = alignment::measure_diagnostic(apply_case(s.gray, changed_qrs, layout, c, s.qr_shift_px));
        if (r.status == alignment::Status::Failed) { ++score.failed; continue; }
        const double ey = r.y_mm - s.expected_y, et = r.t_deg - s.expected_t;
        score.worst_y = std::max(score.worst_y, std::abs(ey));
        score.worst_t = std::max(score.worst_t, std::abs(et));
        score.passed += std::abs(ey) <= kToleranceYmm && std::abs(et) <= kToleranceTdeg;
    }
    return score;
}

// is_shown_in_document: 문서 그림 대상 확인
//   입력: c (실험 조건)
//   출력: 그림 대상이면 true
bool is_shown_in_document(const Case& c) {
    return (c.qr_size != 1 && c.distance == 1 && c.content == 0) || (c.content == 3 && c.qr_size == 1.2 && c.distance == 0.98) ||
           (c.content == 1 && c.qr_size == 1 && c.distance == 1);
}

struct Folders { fs::path model, work, out; };

// check_case: 기준 유지와 재등록 결과 비교
//   입력: c (조건), center (기준 사진), layout (위치), samples (예제), base (기준 결과), 저장 경로와 표
//   출력: 없음 (결과 표와 사례 그림 저장)
void check_case(const Case& c, const cv::Mat& center, const Layout& layout, const std::vector<SamplePhoto>& samples, const alignment::Result& base,
                const Folders& folders, std::ostream& table) {
    // (가) 그대로 쓰면
    alignment::initialize(folders.model.u8string());
    const cv::Mat changed_qrs = change_qrs(center, layout, c);
    const cv::Mat changed_center = c.distance != 1 ? scale_rotate(changed_qrs, c.distance, 0) : changed_qrs;
    const alignment::Result as_is = alignment::measure_diagnostic(changed_center);
    const bool as_is_failed = as_is.status == alignment::Status::Failed;
    // (나) 새로 등록하면
    const fs::path registered = folders.work / c.name();
    write_registered_data(folders.model, registered, changed_center, transform_layout(layout, c, center.size()));
    const bool ready = alignment::initialize(registered.u8string()).status != alignment::Status::Failed;
    const SampleScore score = ready ? score_samples(samples, changed_qrs, layout, c) : SampleScore();

    table << c.name() << '\t' << c.content << '\t' << c.qr_size << '\t' << c.distance << '\t' << status_name(as_is.status) << '\t' << as_is.code << '\t'
          << (as_is_failed ? std::string("nan") : std::to_string(as_is.y_mm - base.y_mm)) << '\t' << (as_is_failed ? std::string("nan") : std::to_string(as_is.t_deg - base.t_deg)) << '\t'
          << ready << '\t' << score.failed << '\t' << score.passed << '\t' << samples.size() << '\t' << score.worst_y << '\t' << score.worst_t << '\n';
    std::cout << c.name() << "  그대로: " << status_name(as_is.status) << " " << as_is.code << "  |  등록 후: 통과 " << score.passed << "/" << samples.size()
              << ", 실패 " << score.failed << ", 최대 오차 Y " << score.worst_y << " T " << score.worst_t << "\n";
    if (is_shown_in_document(c))
        save_png(folders.out / ("installation_" + c.name() + ".png"),
                 labeled_view(changed_center, "QR changed " + std::to_string(c.content) + ", QR size x" + std::to_string(c.qr_size).substr(0, 3) + ", distance x" + std::to_string(c.distance).substr(0, 4)));
}

}   // 파일 내부

// main: QR 내용, 인쇄 크기, 촬영 거리 변경 후 기준 재등록 효과 비교
//   입력: 실행할 때 지정한 경로와 옵션(파일 상단 사용법)
//   출력: 실행 결과와 성공 여부를 나타내는 종료 코드
int main(int argc, char** argv) {
    tools::use_korean_console();
    if (argc != 6) { std::cerr << "사용: check_new_installation <model 폴더> <list.tsv> <expected.tsv> <작업 폴더> <출력 폴더>\n"; return 2; }
    const Folders folders{fs::u8path(argv[1]), fs::u8path(argv[4]), fs::u8path(argv[5])};
    fs::create_directories(folders.work);
    fs::create_directories(folders.out);
    if (alignment::initialize(folders.model.u8string()).status == alignment::Status::Failed) return 1;
    const cv::Mat center = alignment::read_photo((folders.model / "center.bmp").u8string());
    const Layout layout = read_layout(folders.model);

    const std::vector<SamplePhoto> samples = read_samples(fs::u8path(argv[2]), fs::u8path(argv[3]));
    alignment::clear_zero();
    const alignment::Result base = alignment::measure_diagnostic(center);

    std::ofstream table(folders.out / "installation_results.tsv", std::ios::binary);
    table << std::setprecision(5) << "case\tqr_changed\tqr_size\tdistance\tas_is_status\tas_is_code\tas_is_dY\tas_is_dT\tregistered\tsample_failed\tsample_pass\tsample_total\tmax_eY_mm\tmax_eT_deg\n";
    for (int content : kContentChanges)
        for (double size : kQrSizes)
            for (double distance : kDistanceScales) check_case(Case{content, size, distance}, center, layout, samples, base, folders, table);
    return 0;
}
