// 기준 자료 읽기와 입력 영상 변환
#include "common/reference.hpp"
#include "alignment.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace alignment {

// read_photo: 한글 경로 사진 읽기
//   입력: path (UTF-8 우선, 실패 시 원래 경로 사용)
//   출력: 회색 Mat. 읽기 실패 시 빈 Mat, 해석 예외 가능
cv::Mat read_photo(const std::string& path) {
    std::ifstream file;
    const auto mode = std::ios::binary | std::ios::ate;
    try { file.open(std::filesystem::u8path(path), mode); } catch (const std::exception&) {}
    if (!file.is_open()) { file.clear(); file.open(path, mode); }
    if (!file.is_open()) return cv::Mat();

    const std::streamoff length = file.tellg();
    if (length <= 0) return cv::Mat();
    const auto size = static_cast<std::uintmax_t>(length);
    std::vector<unsigned char> bytes;
    // imdecode의 파일 바이트 배열은 int 범위의 Mat 크기로 표현된다.
    if (size > static_cast<std::uintmax_t>((std::numeric_limits<int>::max)()) ||
        size > static_cast<std::uintmax_t>((std::numeric_limits<std::streamsize>::max)()) ||
        size > static_cast<std::uintmax_t>(bytes.max_size())) return cv::Mat();

    bytes.resize(static_cast<std::size_t>(size));
    file.seekg(0, std::ios::beg);
    if (!file) return cv::Mat();
    // 확인한 길이를 한 번에 읽고, 일부만 읽힌 파일은 해석하지 않는다.
    const auto count = static_cast<std::streamsize>(size);
    file.read(reinterpret_cast<char*>(bytes.data()), count);
    if (!file || file.gcount() != count) return cv::Mat();
    return cv::imdecode(bytes, cv::IMREAD_GRAYSCALE);
}

// load_reference: 기준 자료 읽기
//   입력: data_folder (center.bmp, reference.json)
//   출력: 성공 여부, reference (기준 사진과 위치), error (실패 사유)
bool load_reference(const std::string& data_folder, Reference& reference, std::string& error) {
    reference.photo = read_photo(data_folder + "/center.bmp");
    if (reference.photo.empty()) { error = "center.bmp 를 읽지 못했습니다: " + data_folder; return false; }
    if (reference.photo.cols != kPhotoWidth || reference.photo.rows != kPhotoHeight) { error = "center.bmp 가 3840×2160 이 아닙니다"; return false; }

    std::ifstream input(std::filesystem::u8path(data_folder + "/reference.json"), std::ios::binary);
    if (!input) { error = "reference.json 을 읽지 못했습니다: " + data_folder; return false; }
    std::string json((std::istreambuf_iterator<char>(input)), {});
    cv::FileStorage file(json, cv::FileStorage::READ | cv::FileStorage::MEMORY);
    if (!file.isOpened()) { error = "reference.json 을 읽지 못했습니다: " + data_folder; return false; }
    const cv::FileNode qr = file["qr"], screws = file["screws"];
    if (!qr.isSeq() || qr.size() != 3 || !screws.isSeq() || screws.size() != 4) { error = "reference.json 에 qr 3개와 screws 4개가 있어야 합니다"; return false; }
    for (int k = 0; k < 3; ++k) {
        if (!qr[k].isSeq() || qr[k].size() != 4) { error = "QR 좌표 형식 오류"; return false; }
        for (int j=0; j<4; ++j) { double v=double(qr[k][j]); if (!qr[k][j].isInt() || !std::isfinite(v) || v<0 || v>kPhotoWidth) { error="QR 좌표 숫자 오류"; return false; } }
        if (int(qr[k][2])<=0 || int(qr[k][3])<=0) { error="QR 크기는 양수여야 합니다"; return false; }
        reference.qr[k] = cv::Rect(int(qr[k][0]), int(qr[k][1]), int(qr[k][2]), int(qr[k][3]));
        if (!inside_photo(reference.qr[k])) { error = "reference.json 의 QR 상자가 사진 밖입니다"; return false; }
    }
    for (int k = 0; k < 4; ++k) {
        if (!screws[k].isSeq() || screws[k].size() != 2) { error = "나사 좌표 형식 오류"; return false; }
        for (int j=0;j<2;++j) if (!screws[k][j].isInt()) { error="나사 좌표 숫자 오류"; return false; }
        reference.screw[k] = cv::Point(int(screws[k][0]), int(screws[k][1]));
        if (reference.screw[k].x < 0 || reference.screw[k].x >= kPhotoWidth || reference.screw[k].y < 0 || reference.screw[k].y >= kPhotoHeight) { error = "나사 좌표가 사진 밖입니다"; return false; }
    }
    for (const char* key : {"trained_y_mm", "trained_t_deg"}) {
        auto range = file[key];
        if (!range.isSeq() || range.size() != 2 || (!range[0].isInt() && !range[0].isReal()) || (!range[1].isInt() && !range[1].isReal()) || !std::isfinite(double(range[0])) || !std::isfinite(double(range[1])) || double(range[0]) > double(range[1])) { error = "학습 범위 형식 오류"; return false; }
    }
    for (int i = 0; i < 2; ++i) {
        reference.trained_y_mm[i] = double(file["trained_y_mm"][i]);
        reference.trained_t_deg[i] = double(file["trained_t_deg"][i]);
    }
    return true;
}

// to_gray: 회색 영상 변환
//   입력: photo (3840x2160, 밝기 0~255의 회색 또는 컬러 BGR, BGRA)
//   출력: 성공 여부, gray (회색 사진), error (실패 사유)
bool to_gray(const cv::Mat& photo, cv::Mat& gray, std::string& error) {
    if (photo.empty()) { error = "사진이 비어 있습니다"; return false; }
    if (photo.cols != kPhotoWidth || photo.rows != kPhotoHeight) {
        error = "입력 해상도 " + std::to_string(photo.cols) + "×" + std::to_string(photo.rows) + " (요구 3840×2160)";
        return false;
    }
    if (photo.depth() != CV_8U) {
        error = "입력 화소 형식 불일치 (요구 8비트 부호 없는 정수)";
        return false;
    }
    if (photo.channels() == 1) gray = photo;
    else if (photo.channels() == 3) cv::cvtColor(photo, gray, cv::COLOR_BGR2GRAY);
    else if (photo.channels() == 4) cv::cvtColor(photo, gray, cv::COLOR_BGRA2GRAY);
    else { error = "입력 채널 수 " + std::to_string(photo.channels()) + " (허용 1, 3, 4)"; return false; }
    return true;
}

// average_gray: 여러 사진의 같은 위치 밝기 평균
//   입력: photos (같은 크기의 입력 사진 목록)
//   출력: 성공 여부, gray (평균 회색 사진), error (실패 사유)
bool average_gray(const std::vector<cv::Mat>& photos, cv::Mat& gray, std::string& error) {
    if (photos.empty()) { error = "사진이 한 장도 없습니다"; return false; }
    cv::Mat sum = cv::Mat::zeros(kPhotoHeight, kPhotoWidth, CV_64F);
    for (const cv::Mat& photo : photos) {
        cv::Mat one, as_double;
        if (!to_gray(photo, one, error)) return false;
        one.convertTo(as_double, CV_64F);
        sum += as_double;
    }
    sum.convertTo(gray, CV_8U, 1.0 / double(photos.size()));
    return true;
}

}   // alignment
