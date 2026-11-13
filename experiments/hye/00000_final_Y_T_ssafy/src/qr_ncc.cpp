#include "qr_ncc.hpp"
#include "site.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include <array>
#include <exception>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace final00 {
QrNcc::QrNcc(const cv::Mat& center) {
    if (center.type() != CV_8UC1 || center.size() != cv::Size(3840, 2160)) throw std::invalid_argument("QrNcc: 센터 사진은 3840x2160 8비트 회색이어야 한다");
    for (int k = 0; k < 3; ++k) templates_[k] = center(site::get().qr[k]).clone();
}

QrShift QrNcc::measure(const cv::Mat& gray) const {
    if (gray.type() != CV_8UC1 || gray.size() != cv::Size(3840, 2160)) throw std::invalid_argument("QrNcc: 사진은 3840x2160 8비트 회색이어야 한다");
    const auto& g = site::get();
    QrShift out{0, 0, 1};
    // QR 3개는 서로 독립이라 동시에 계산하고, 평균은 QR 순서대로 더한다(결과는 한 개씩 할 때와 같다).
    std::array<double, 3> dx{}, dy{}, ncc{}; std::array<std::exception_ptr, 3> errors;
    cv::parallel_for_(cv::Range(0, 3), [&](const cv::Range& range) { for (int k = range.start; k < range.end; ++k) try {
        const cv::Rect b = g.qr[k], search(b.x - g.search_x, b.y - g.search_y, b.width + 2 * g.search_x, b.height + 2 * g.search_y);
        if ((search & cv::Rect(0, 0, gray.cols, gray.rows)) != search) throw std::runtime_error("QR 탐색 창이 화면 밖이다");
        cv::Mat c; cv::matchTemplate(gray(search), templates_[k], c, cv::TM_CCOEFF_NORMED);
        double best; cv::Point p; cv::minMaxLoc(c, nullptr, &best, nullptr, &p);
        if (best < .8) throw std::runtime_error("QR " + std::to_string(k + 1) + " NCC 가 낮다 (" + std::to_string(best) + ")");
        if (p.x <= 0 || p.y <= 0 || p.x >= c.cols - 1 || p.y >= c.rows - 1) throw std::runtime_error("QR " + std::to_string(k + 1) + " 이 탐색 창 끝에 걸렸다(움직임이 너무 크다)");
        // 3점 포물선: 봉우리 양옆 상관으로 소수 이동
        double sx = 0, sy = 0;
        { const double a = c.at<float>(p.y, p.x - 1), m = c.at<float>(p.y, p.x), d = c.at<float>(p.y, p.x + 1), den = a - 2 * m + d; if (den < 0) sx = .5 * (a - d) / den; }
        { const double a = c.at<float>(p.y - 1, p.x), m = c.at<float>(p.y, p.x), d = c.at<float>(p.y + 1, p.x), den = a - 2 * m + d; if (den < 0) sy = .5 * (a - d) / den; }
        dx[k] = p.x - g.search_x + sx; dy[k] = p.y - g.search_y + sy; ncc[k] = best;
    } catch (...) { errors[k] = std::current_exception(); } });
    for (auto& e : errors) if (e) std::rethrow_exception(e);
    for (int k = 0; k < 3; ++k) { out.dx += dx[k] / 3.0; out.dy += dy[k] / 3.0; out.min_ncc = std::min(out.min_ncc, ncc[k]); }
    return out;
}
}
