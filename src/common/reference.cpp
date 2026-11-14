#include "common/reference.hpp"
#include "alignment.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <fstream>
#include <vector>

namespace alignment {

// 파일을 바이트로 읽어 풀어낸다 (cv::imread 는 한글 경로를 못 읽는다).
// 경로를 UTF-8 로 열어 보고, 안 열리면 받은 글자 그대로(Windows 기본 문자 집합) 다시 연다.
cv::Mat read_photo(const std::string& path) {
    std::ifstream file;
    try { file.open(std::filesystem::u8path(path), std::ios::binary); } catch (const std::exception&) {}
    if (!file.is_open()) { file.clear(); file.open(path, std::ios::binary); }
    const std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    return bytes.empty() ? cv::Mat() : cv::imdecode(bytes, cv::IMREAD_GRAYSCALE);
}

bool load_reference(const std::string& data_folder, Reference& reference, std::string& error) {
    reference.photo = read_photo(data_folder + "/center.bmp");
    if (reference.photo.empty()) { error = "center.bmp 를 읽지 못했습니다: " + data_folder; return false; }
    if (reference.photo.cols != kPhotoWidth || reference.photo.rows != kPhotoHeight) { error = "center.bmp 가 3840×2160 이 아닙니다"; return false; }

    cv::FileStorage file(data_folder + "/reference.json", cv::FileStorage::READ);
    if (!file.isOpened()) { error = "reference.json 을 읽지 못했습니다: " + data_folder; return false; }
    const cv::FileNode qr = file["qr"], screws = file["screws"];
    if (!qr.isSeq() || qr.size() != 3 || !screws.isSeq() || screws.size() != 4) { error = "reference.json 에 qr 3개와 screws 4개가 있어야 합니다"; return false; }
    for (int k = 0; k < 3; ++k) {
        reference.qr[k] = cv::Rect(int(qr[k][0]), int(qr[k][1]), int(qr[k][2]), int(qr[k][3]));
        if (!inside_photo(reference.qr[k])) { error = "reference.json 의 QR 상자가 사진 밖입니다"; return false; }
    }
    for (int k = 0; k < 4; ++k) reference.screw[k] = cv::Point(int(screws[k][0]), int(screws[k][1]));
    for (int i = 0; i < 2; ++i) {
        reference.trained_y_mm[i] = double(file["trained_y_mm"][i]);
        reference.trained_t_deg[i] = double(file["trained_t_deg"][i]);
    }
    return true;
}

bool to_gray(const cv::Mat& photo, cv::Mat& gray, std::string& error) {
    if (photo.empty()) { error = "사진이 비어 있습니다"; return false; }
    if (photo.cols != kPhotoWidth || photo.rows != kPhotoHeight || photo.depth() != CV_8U) {
        error = "사진 형식이 다릅니다 (받은 크기 " + std::to_string(photo.cols) + "×" + std::to_string(photo.rows) + "). 3840×2160 8비트 사진을 넣어 주세요";
        return false;
    }
    if (photo.channels() == 1) gray = photo;
    else if (photo.channels() == 3) cv::cvtColor(photo, gray, cv::COLOR_BGR2GRAY);
    else if (photo.channels() == 4) cv::cvtColor(photo, gray, cv::COLOR_BGRA2GRAY);
    else { error = "사진의 채널 수가 1, 3, 4 가 아닙니다"; return false; }
    return true;
}

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

}   // namespace alignment
