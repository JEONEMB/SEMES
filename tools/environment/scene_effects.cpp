// 카메라, 조명, 대상물 변화의 합성 영상. 현장 성능 보증용 아님
#include "scene_effects.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <fstream>
#include <random>
#include <vector>

namespace fs = std::filesystem;

namespace scene {

namespace {

// as_float: 소수 밝기값 계산 준비
//   입력: gray (회색 사진)
//   출력: 소수 밝기값을 저장할 사진
cv::Mat as_float(const cv::Mat& gray) { cv::Mat f; gray.convertTo(f, CV_32F); return f; }
// to_gray8: 계산한 밝기를 사진으로 변환
//   입력: f (소수 밝기값 사진)
//   출력: 밝기를 0~255로 제한한 회색 사진
cv::Mat to_gray8(const cv::Mat& f) { cv::Mat g; f.convertTo(g, CV_8U); return g; }

// warp_affine: 지정한 이동과 회전 적용
//   입력: gray (회색 사진), matrix (위치 변환값)
//   출력: 같은 크기의 변환 사진
cv::Mat warp_affine(const cv::Mat& gray, const cv::Mat& matrix) {
    cv::Mat out;
    cv::warpAffine(gray, out, matrix, gray.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    return out;
}

}   // 파일 내부

// 읽고 쓰기

// save_png: PNG 저장
//   입력: path (파일 경로), picture (사진)
//   출력: 없음 (파일 저장)
void save_png(const fs::path& path, const cv::Mat& picture) {
    std::vector<unsigned char> bytes;
    cv::imencode(".png", picture, bytes);
    std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
}

// read_layout: 기준 좌표 읽기
//   입력: data_folder (reference.json 포함)
//   출력: QR 영역과 나사 중심
Layout read_layout(const fs::path& data_folder) {
    cv::FileStorage file(data_folder.u8string() + "/reference.json", cv::FileStorage::READ);
    Layout layout;
    for (int k = 0; k < 3; ++k) layout.qr[k] = cv::Rect(int(file["qr"][k][0]), int(file["qr"][k][1]), int(file["qr"][k][2]), int(file["qr"][k][3]));
    for (int k = 0; k < 4; ++k) layout.screw[k] = cv::Point(int(file["screws"][k][0]), int(file["screws"][k][1]));
    return layout;
}

// labeled_view: 제목 포함 사례 그림 생성
//   입력: picture (사진), title (제목)
//   출력: 판 주변을 절반 크기로 줄인 컬러 사진(BGR)
cv::Mat labeled_view(const cv::Mat& picture, const std::string& title) {
    cv::Mat view, color;
    cv::resize(picture(kView), view, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
    if (view.channels() == 1) cv::cvtColor(view, color, cv::COLOR_GRAY2BGR);
    else color = view;
    int baseline = 0;
    const cv::Size box = cv::getTextSize(title, cv::FONT_HERSHEY_SIMPLEX, 0.9, 2, &baseline);
    cv::rectangle(color, cv::Rect(0, 0, box.width + 20, box.height + 20), cv::Scalar(0, 0, 0), -1);
    cv::putText(color, title, cv::Point(10, box.height + 9), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    return color;
}

// 카메라 위치, 거리, 렌즈

// scale_rotate: 영상 중심 기준 배율과 회전
//   입력: gray (회색 사진), scale (크기 배율), angle_deg (회전 각도)
//   출력: 변환 영상
cv::Mat scale_rotate(const cv::Mat& gray, double scale, double angle_deg) {
    return warp_affine(gray, cv::getRotationMatrix2D(cv::Point2f(gray.cols / 2.0f, gray.rows / 2.0f), angle_deg, scale));
}

// shift: 영상 평행이동
//   입력: gray (회색 사진), right_px와 down_px (오른쪽, 아래쪽 이동 px)
//   출력: 이동 영상
cv::Mat shift(const cv::Mat& gray, double right_px, double down_px) {
    return warp_affine(gray, (cv::Mat_<double>(2, 3) << 1, 0, right_px, 0, 1, down_px));
}

// tilt: 판을 앞뒤로 기울여 본 모습 만들기
//   입력: gray (회색 사진), tilt_deg (기울기 각도), focal_px (카메라 초점거리 px)
//   출력: 기울어진 영상
cv::Mat tilt(const cv::Mat& gray, double tilt_deg, double focal_px) {
    const double w = gray.cols, h = gray.rows, a = tilt_deg * CV_PI / 180;
    const cv::Mat camera = (cv::Mat_<double>(3, 3) << focal_px, 0, w / 2, 0, focal_px, h / 2, 0, 0, 1);
    const cv::Mat rotation = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, std::cos(a), -std::sin(a), 0, std::sin(a), std::cos(a));
    cv::Mat out;
    cv::warpPerspective(gray, out, camera * rotation * camera.inv(), gray.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    return out;
}

// distort: 렌즈 왜곡 합성
//   입력: gray (회색 사진), k1 (휘어짐 정도, 양수이면 바깥으로 볼록)
//   출력: 왜곡 영상
cv::Mat distort(const cv::Mat& gray, double k1) {
    cv::Mat map_x(gray.size(), CV_32F), map_y(gray.size(), CV_32F);
    const double cx = gray.cols / 2.0, cy = gray.rows / 2.0, norm = gray.cols / 2.0;
    for (int y = 0; y < gray.rows; ++y)
        for (int x = 0; x < gray.cols; ++x) {
            const double u = (x - cx) / norm, v = (y - cy) / norm, f = 1 + k1 * (u * u + v * v);
            map_x.at<float>(y, x) = float(cx + u * f * norm);
            map_y.at<float>(y, x) = float(cy + v * f * norm);
        }
    cv::Mat out;
    cv::remap(gray, out, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    return out;
}

// 대상물

// replace_qr: QR 내부 무늬 변경
//   입력: gray (회색 사진), box (QR 영역), seed (무작위 시작값)
//   출력: QR 모서리의 위치 표시 사각형 3개를 유지한 사진
cv::Mat replace_qr(const cv::Mat& gray, const cv::Rect& box, unsigned seed) {
    cv::Mat out = gray.clone();
    cv::Mat patch = out(box & cv::Rect(0, 0, gray.cols, gray.rows));
    const int modules = 25;   // QR 버전 2 (25×25 칸)
    const cv::Rect inner(int(box.width * 0.06), int(box.height * 0.06), int(box.width * 0.88), int(box.height * 0.88));
    const double cell = inner.width / double(modules);
    std::mt19937 random(seed);
    std::bernoulli_distribution dark(0.5);
    double black = 255, white = 0;
    cv::minMaxLoc(patch, &black, &white);
    auto in_finder = [&](int r, int c) { return (r < 8 && c < 8) || (r < 8 && c >= modules - 8) || (r >= modules - 8 && c < 8); };
    for (int r = 0; r < modules; ++r)
        for (int c = 0; c < modules; ++c) {
            const bool is_dark = dark(random);
            if (in_finder(r, c)) continue;
            const cv::Rect m(inner.x + int(c * cell), inner.y + int(r * cell), int(cell + 1), int(cell + 1));
            patch(m & cv::Rect(0, 0, patch.cols, patch.rows)).setTo(is_dark ? black + 25 : white - 25);
        }
    cv::GaussianBlur(patch, patch, cv::Size(0, 0), 1.2);
    return out;
}

// scaled_box: 배율 적용 QR 영역 계산
//   입력: box (QR 영역), factor (크기 배율)
//   출력: 변경 영역
cv::Rect scaled_box(const cv::Rect& box, double factor) {
    const double cx = box.x + box.width / 2.0, cy = box.y + box.height / 2.0;
    return cv::Rect(int(std::lround(cx - box.width * factor / 2)), int(std::lround(cy - box.height * factor / 2)),
                    int(std::lround(box.width * factor)), int(std::lround(box.height * factor)));
}

// scale_qr: QR 인쇄 크기 변경
//   입력: gray (회색 사진), box (QR 영역), factor (배율)
//   출력: 원래 자리를 판 색으로 덮은 변환 영상
cv::Mat scale_qr(const cv::Mat& gray, const cv::Rect& box, double factor) {
    const cv::Rect photo(0, 0, gray.cols, gray.rows);
    cv::Mat out = gray.clone();
    // 원래 QR 영역을 위쪽 띠의 평균 밝기로 덮기
    const double plate = cv::mean(gray(cv::Rect(box.x, box.y - 20, box.width, 8) & photo))[0];
    out(cv::Rect(box.x - 3, box.y - 3, box.width + 6, box.height + 6) & photo).setTo(plate);
    // 중심 기준 배율을 적용한 QR 배치
    const cv::Point2f center(box.x + box.width / 2.0f, box.y + box.height / 2.0f);
    const cv::Mat scaled = warp_affine(gray, cv::getRotationMatrix2D(center, 0, factor));
    const cv::Rect target = scaled_box(box, factor) & photo;
    scaled(target).copyTo(out(target));
    return out;
}

// remove_screw: 나사 제거 합성
//   입력: gray (회색 사진), center (중심 위치)
//   출력: 나사 영역을 덮은 영상
cv::Mat remove_screw(const cv::Mat& gray, cv::Point center) {
    cv::Mat out = gray.clone();
    const double plate = cv::mean(gray(cv::Rect(center.x - 70, center.y - 120, 140, 30)))[0];
    cv::circle(out, center, 34, cv::Scalar(plate), -1, cv::LINE_AA);
    cv::Mat area = out(cv::Rect(center.x - 70, center.y - 70, 140, 140));
    cv::GaussianBlur(area, area, cv::Size(0, 0), 2.0);
    return out;
}

// different_screw: 나사 모양 변경 합성
//   입력: gray (회색 사진), center (중심 위치)
//   출력: 변경 영상
cv::Mat different_screw(const cv::Mat& gray, cv::Point center) {
    cv::Mat out = remove_screw(gray, center);
    cv::circle(out, center, 30, cv::Scalar(150), -1, cv::LINE_AA);
    cv::circle(out, center, 30, cv::Scalar(90), 3, cv::LINE_AA);
    cv::line(out, center + cv::Point(-16, 0), center + cv::Point(16, 0), cv::Scalar(60), 6, cv::LINE_AA);
    cv::line(out, center + cv::Point(0, -16), center + cv::Point(0, 16), cv::Scalar(60), 6, cv::LINE_AA);
    return out;
}

// 조명

// brightness: 밝기를 곱하고 더하여 조정
//   입력: gray (회색 사진), gain (밝기 배율), offset (더할 밝기)
//   출력: 밝기 조정 사진. 밝기값은 0~255로 제한
cv::Mat brightness(const cv::Mat& gray, double gain, double offset) {
    cv::Mat out;
    gray.convertTo(out, CV_8U, gain, offset);
    return out;
}

// gamma: 어두운 부분과 밝은 부분의 밝기 비율 변경
//   입력: gray (회색 사진), g (밝기 비율 조정값)
//   출력: 밝기 변환 영상
cv::Mat gamma(const cv::Mat& gray, double g) {
    cv::Mat table(1, 256, CV_8U);
    for (int i = 0; i < 256; ++i) table.at<uchar>(i) = cv::saturate_cast<uchar>(255.0 * std::pow(i / 255.0, g));
    cv::Mat out;
    cv::LUT(gray, table, out);
    return out;
}

// side_light: 좌우 밝기 차이 합성
//   입력: gray (회색 사진), left (왼쪽 밝기 배율), right (오른쪽 밝기 배율)
//   출력: 조명 변화 영상
cv::Mat side_light(const cv::Mat& gray, double left, double right) {
    cv::Mat f = as_float(gray);
    for (int x = 0; x < gray.cols; ++x) f.col(x) *= float(left + (right - left) * x / (gray.cols - 1.0));
    return to_gray8(f);
}

// glare: 일부 위치의 빛 반사 만들기
//   입력: gray (회색 사진), center (중심 위치), radius_px (반지름 px), strength (빛 강도)
//   출력: 반사광 영상
cv::Mat glare(const cv::Mat& gray, cv::Point center, double radius_px, double strength) {
    cv::Mat f = as_float(gray);
    const cv::Rect area = cv::Rect(int(center.x - 3 * radius_px), int(center.y - 3 * radius_px), int(6 * radius_px), int(6 * radius_px)) & cv::Rect(0, 0, gray.cols, gray.rows);
    for (int y = area.y; y < area.br().y; ++y)
        for (int x = area.x; x < area.br().x; ++x)
            f.at<float>(y, x) += float(strength * std::exp(-(std::pow(x - center.x, 2) + std::pow(y - center.y, 2)) / (radius_px * radius_px)));
    return to_gray8(f);
}

// banding: 조명 줄무늬 합성
//   입력: gray (회색 사진), depth (줄무늬 강도), period_px (줄 간격 px)
//   출력: 줄무늬 영상
cv::Mat banding(const cv::Mat& gray, double depth, double period_px) {
    cv::Mat f = as_float(gray);
    for (int y = 0; y < gray.rows; ++y) f.row(y) *= float(1 + depth * std::sin(2 * CV_PI * y / period_px));
    return to_gray8(f);
}

// 공정 환경

// defocus: 초점 흐림 합성
//   입력: gray (회색 사진), sigma_px (흐림 정도 px)
//   출력: 주변 밝기를 섞어 흐리게 만든 사진
cv::Mat defocus(const cv::Mat& gray, double sigma_px) {
    cv::Mat out;
    cv::GaussianBlur(gray, out, cv::Size(0, 0), sigma_px);
    return out;
}

// motion_blur: 이동 흐림 합성
//   입력: gray (회색 사진), length_px (흐려질 길이 px), horizontal (가로 방향 여부)
//   출력: 이동 방향으로 흐린 영상
cv::Mat motion_blur(const cv::Mat& gray, int length_px, bool horizontal) {
    cv::Mat kernel = cv::Mat::zeros(length_px, length_px, CV_32F);
    if (horizontal) kernel.row(length_px / 2).setTo(1.0f / length_px);
    else kernel.col(length_px / 2).setTo(1.0f / length_px);
    cv::Mat out;
    cv::filter2D(gray, out, -1, kernel);
    return out;
}

// heat_shimmer: 열에 의한 영상 흔들림 합성
//   입력: gray (회색 사진), amplitude_px (흔들림 크기 px), cell_px (흔들림 간격 px), seed (무작위 시작값)
//   출력: 변형 영상
cv::Mat heat_shimmer(const cv::Mat& gray, double amplitude_px, double cell_px, unsigned seed) {
    std::mt19937 random(seed);
    std::normal_distribution<float> normal(0, 1);
    const cv::Size small(int(gray.cols / cell_px) + 2, int(gray.rows / cell_px) + 2);
    cv::Mat dx(small, CV_32F), dy(small, CV_32F);
    for (int y = 0; y < small.height; ++y)
        for (int x = 0; x < small.width; ++x) { dx.at<float>(y, x) = normal(random); dy.at<float>(y, x) = normal(random); }
    cv::resize(dx, dx, gray.size(), 0, 0, cv::INTER_CUBIC);
    cv::resize(dy, dy, gray.size(), 0, 0, cv::INTER_CUBIC);
    cv::Mat map_x(gray.size(), CV_32F), map_y(gray.size(), CV_32F);
    for (int y = 0; y < gray.rows; ++y)
        for (int x = 0; x < gray.cols; ++x) {
            map_x.at<float>(y, x) = float(x + amplitude_px * dx.at<float>(y, x));
            map_y.at<float>(y, x) = float(y + amplitude_px * dy.at<float>(y, x));
        }
    cv::Mat out;
    cv::remap(gray, out, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    return out;
}

// fog: 안개 합성
//   입력: gray (회색 사진), amount (안개 농도), sigma_px (흐림 정도 px)
//   출력: 흐림과 밝기 변화 영상
cv::Mat fog(const cv::Mat& gray, double amount, double sigma_px) {
    cv::Mat blurred, out;
    cv::GaussianBlur(gray, blurred, cv::Size(0, 0), sigma_px);
    cv::addWeighted(blurred, 1 - amount, cv::Mat(gray.size(), CV_8U, cv::Scalar(225)), amount, 0, out);
    return out;
}

// spots: 오염 또는 물방울 합성
//   입력: gray (회색 사진), count (개수), min_radius (최소 반지름), max_radius (최대 반지름), droplets (물방울 여부), seed (무작위 시작값)
//   출력: 얼룩 영상
cv::Mat spots(const cv::Mat& gray, int count, double min_radius, double max_radius, bool droplets, unsigned seed) {
    std::mt19937 random(seed);
    std::uniform_real_distribution<double> pick_x(kView.x, kView.br().x), pick_y(kView.y, kView.br().y), pick_r(min_radius, max_radius);
    cv::Mat layer = cv::Mat::zeros(gray.size(), CV_32F);
    for (int i = 0; i < count; ++i) {
        const cv::Point c(int(pick_x(random)), int(pick_y(random)));
        const int r = int(pick_r(random));
        if (droplets) {
            cv::circle(layer, c, r, cv::Scalar(-35), -1, cv::LINE_AA);
            cv::circle(layer, c, r, cv::Scalar(60), std::max(2, r / 6), cv::LINE_AA);
        } else {
            cv::circle(layer, c, r, cv::Scalar(-70), -1, cv::LINE_AA);
        }
    }
    cv::GaussianBlur(layer, layer, cv::Size(0, 0), droplets ? 3 : max_radius * 0.6);
    const cv::Mat sum = as_float(gray) + layer;
    return to_gray8(sum);
}

// ghost: 사진이 두 겹으로 보이는 모습 만들기
//   입력: gray, right_px와 down_px (오른쪽, 아래쪽 이동 px), weight (겹칠 비중)
//   출력: 이동한 상을 겹친 영상
cv::Mat ghost(const cv::Mat& gray, double right_px, double down_px, double weight) {
    const cv::Mat mixed = (1 - weight) * as_float(gray) + weight * as_float(shift(gray, right_px, down_px));
    return to_gray8(mixed);
}

// glow_from_top: 위쪽 빛 번짐 합성
//   입력: gray (회색 사진), strength (빛 강도), center_y (세로 중심), width (빛의 폭)
//   출력: 빛 번짐 영상
cv::Mat glow_from_top(const cv::Mat& gray, double strength, double center_y, double width) {
    cv::Mat f = as_float(gray);
    for (int y = 0; y < gray.rows; ++y) f.row(y) += float(strength * std::exp(-std::pow((y - center_y) / width, 2)));
    return to_gray8(f);
}

// vignette: 원형 창 밖 어둡게 처리
//   입력: gray (회색 사진), center (중심 위치), radius_px (반지름 px)
//   출력: 원 밖이 어두운 사진
cv::Mat vignette(const cv::Mat& gray, cv::Point center, double radius_px) {
    cv::Mat f = as_float(gray);
    for (int y = 0; y < gray.rows; ++y)
        for (int x = 0; x < gray.cols; ++x) {
            const double r = std::hypot(x - center.x, y - center.y);
            if (r > radius_px) f.at<float>(y, x) *= float(std::max(0.15, 1 - (r - radius_px) / 120.0));
        }
    return to_gray8(f);
}

// noise: 밝기 잡음 합성
//   입력: gray (회색 사진), sigma (잡음 크기), seed (무작위 시작값)
//   출력: 잡음 영상
cv::Mat noise(const cv::Mat& gray, double sigma, unsigned seed) {
    cv::Mat n(gray.size(), CV_32F);
    cv::theRNG().state = seed;
    cv::randn(n, 0, sigma);
    const cv::Mat sum = as_float(gray) + n;
    return to_gray8(sum);
}

// jpeg: JPEG 압축 손실 합성
//   입력: gray (회색 사진), quality (압축 품질)
//   출력: 압축 후 복원 영상
cv::Mat jpeg(const cv::Mat& gray, int quality) {
    std::vector<unsigned char> bytes;
    cv::imencode(".jpg", gray, bytes, {cv::IMWRITE_JPEG_QUALITY, quality});
    return cv::imdecode(bytes, cv::IMREAD_GRAYSCALE);
}

}   // scene
