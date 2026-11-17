#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

namespace {
const std::array<cv::Rect, 3> qrBoxes{{
    {1552, 969, 184, 198}, {1809, 971, 184, 198}, {2067, 973, 185, 198}}};

struct Match {
    cv::Point shift;
    double score;
};

Match ncc(const cv::Mat& image, const cv::Mat& reference, const cv::Rect& roi) {
    cv::Mat coefficients;
    cv::matchTemplate(image(roi), reference, coefficients, cv::TM_CCOEFF_NORMED);
    cv::Point peak;
    double score;
    cv::minMaxLoc(coefficients, nullptr, &score, nullptr, &peak);
    return {roi.tl() + peak, score};
}

cv::Rect window(const cv::Rect& qr, cv::Point around, int xPad, int yPad) {
    return {qr.x + around.x - xPad, qr.y + around.y - yPad,
            qr.width + 2*xPad, qr.height + 2*yPad};
}

Match locate(const cv::Mat& image, const cv::Mat& reference, const cv::Mat& smallReference,
             const cv::Rect& qr, int downsample) {
    if (downsample == 1) {
        const auto result = ncc(image, reference, window(qr, {}, 240, 88));
        return {result.shift - qr.tl(), result.score};
    }
    const cv::Rect roi = window(qr, {}, 240, 88);
    cv::Mat small;
    cv::resize(image(roi), small, {}, 1.0/downsample, 1.0/downsample, cv::INTER_AREA);
    const auto coarse = ncc(small, smallReference, {0, 0, small.cols, small.rows});
    const cv::Point initial = coarse.shift * downsample + roi.tl() - qr.tl();
    const auto precise = ncc(image, reference, window(qr, initial, downsample*3, downsample*3));
    return {precise.shift - qr.tl(), precise.score};
}
} // namespace

int main(int argc, char** argv) try {
    if (argc != 3) throw std::runtime_error("Usage: semes_coarse_probe CENTER_BMP OUTPUT_TSV");
    cv::setNumThreads(1);
    std::ifstream input(std::filesystem::u8path(argv[1]), std::ios::binary);
    if (!input) throw std::runtime_error("Cannot read center reference");
    const std::vector<unsigned char> bytes{std::istreambuf_iterator<char>(input), {}};
    const cv::Mat center = cv::imdecode(bytes, cv::IMREAD_GRAYSCALE);
    if (center.size() != cv::Size(3840, 2160)) throw std::runtime_error("Expected 4K reference");
    std::ofstream output(std::filesystem::u8path(argv[2]));
    if (!output) throw std::runtime_error("Cannot write output");
    output << "dx\tdy\tcondition\tdownsample\tmax_coordinate_error_px\tmin_ncc\t"
              "max_qr_disagreement_px\taccepted\tcorrect\tms\n";

    const std::array<int, 4> scales{1, 2, 4, 8};
    std::array<std::array<cv::Mat, 3>, scales.size()> templates;
    for (size_t s = 0; s < scales.size(); ++s)
        for (size_t i = 0; i < qrBoxes.size(); ++i)
            cv::resize(center(qrBoxes[i]), templates[s][i], {}, 1.0/scales[s], 1.0/scales[s], cv::INTER_AREA);
    for (int dx : {-230, -165, -100, 0, 100, 165, 230}) for (int dy : {-64, 0, 64}) {
        const cv::Mat transform = (cv::Mat_<double>(2, 3) << 1., 0., dx, 0., 1., dy);
        cv::Mat shifted;
        cv::warpAffine(center, shifted, transform, center.size(), cv::INTER_NEAREST, cv::BORDER_REFLECT_101);
        for (const std::string condition : {"clean", "blur1.2_dim0.8", "partial_mask"}) {
            cv::Mat test = shifted.clone();
            if (condition == "blur1.2_dim0.8") {
                cv::GaussianBlur(test, test, {}, 1.2);
                test.convertTo(test, CV_8U, 0.8);
            } else if (condition == "partial_mask") {
                const auto& region = qrBoxes[1];
                test(cv::Rect(region.x + dx, region.y + dy, region.width/2, region.height)).setTo(128);
            }
            for (size_t s = 0; s < scales.size(); ++s) {
                std::array<Match, 3> matches;
                const auto begin = std::chrono::steady_clock::now();
                for (size_t i = 0; i < matches.size(); ++i)
                    matches[i] = locate(test, templates[0][i], templates[s][i], qrBoxes[i], scales[s]);
                const auto ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - begin).count();
                int error = 0, disagreement = 0;
                double minScore = 1;
                for (const auto& a : matches) {
                    error = std::max({error, std::abs(a.shift.x - dx), std::abs(a.shift.y - dy)});
                    minScore = std::min(minScore, a.score);
                    for (const auto& b : matches)
                        disagreement = std::max({disagreement, std::abs(a.shift.x - b.shift.x), std::abs(a.shift.y - b.shift.y)});
                }
                output << dx << '\t' << dy << '\t' << condition << '\t' << scales[s] << '\t' << error << '\t'
                       << minScore << '\t' << disagreement << '\t' << (minScore >= 0.6 && disagreement <= 4)
                       << '\t' << (error <= 1) << '\t' << ms << '\n';
            }
        }
    }
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
