// =====================================================================================
// report_plot.hpp — H3 · H6 · H7 요약 그림 (OpenCV 로 가로 막대 3칸)
//  cv::putText 는 한글을 그리지 못하므로 그림 안 글자는 영어로 쓴다.
// =====================================================================================
#ifndef __H3_REPORT_PLOT_HPP__
#define __H3_REPORT_PLOT_HPP__

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

namespace h3 {

struct Bar {
    std::string label;
    double value = 0;          // 막대 길이
    double value2 = -1;        // 두 번째 막대 (없으면 < 0)
    std::string text;          // 막대 끝 글자
    std::string text2;
    cv::Scalar color = cv::Scalar(214, 120, 42);   // BGR
};

inline void drawBarPanel(cv::Mat& img, cv::Rect r, const std::string& title, const std::string& xlabel,
                         const std::vector<Bar>& bars, double xmax, const std::string& legend1 = "", const std::string& legend2 = "")
{
    const cv::Scalar ink(42, 43, 43), mute(133, 138, 138), grid(224, 229, 230), c2(200, 206, 207);
    const int labelW = 250, top = r.y + 50, bottom = r.y + r.height - 50;
    const int x0 = r.x + labelW, x1 = r.x + r.width - 20;
    cv::putText(img, title, cv::Point(r.x + 10, r.y + 26), cv::FONT_HERSHEY_SIMPLEX, 0.62, ink, 1, cv::LINE_AA);
    // 눈금
    for (int k = 0; k <= 5; ++k) {
        double v = xmax * k / 5; int x = x0 + (int)((x1 - x0 - 120) * v / xmax);
        cv::line(img, cv::Point(x, top - 5), cv::Point(x, bottom), grid, 1);
        char b[32]; std::snprintf(b, sizeof(b), xmax >= 10 ? "%.0f" : "%.2f", v);
        cv::putText(img, b, cv::Point(x - 10, bottom + 20), cv::FONT_HERSHEY_SIMPLEX, 0.45, mute, 1, cv::LINE_AA);
    }
    cv::putText(img, xlabel, cv::Point(x0, bottom + 42), cv::FONT_HERSHEY_SIMPLEX, 0.5, ink, 1, cv::LINE_AA);
    const int n = (int)bars.size(), slot = (bottom - top) / std::max(1, n);
    for (int i = 0; i < n; ++i) {
        const Bar& b = bars[i];
        int yc = top + slot * i + slot / 2;
        cv::putText(img, b.label, cv::Point(r.x + 10, yc + 5), cv::FONT_HERSHEY_SIMPLEX, 0.47, ink, 1, cv::LINE_AA);
        auto bar = [&](double v, int y, int h, cv::Scalar col, const std::string& t, cv::Scalar tc) {
            int w = (int)((x1 - x0 - 120) * std::max(0.0, v) / xmax);
            cv::rectangle(img, cv::Rect(x0, y - h / 2, std::max(1, w), h), col, cv::FILLED);
            cv::putText(img, t, cv::Point(x0 + w + 6, y + 5), cv::FONT_HERSHEY_SIMPLEX, 0.45, tc, 1, cv::LINE_AA);
        };
        if (b.value2 >= 0) {
            bar(b.value, yc - slot / 5, slot / 3, c2, b.text, mute);
            bar(b.value2, yc + slot / 5, slot / 3, b.color, b.text2, ink);
        } else bar(b.value, yc, slot / 2, b.color, b.text, ink);
    }
    cv::line(img, cv::Point(x0, top - 5), cv::Point(x0, bottom), mute, 1);
    if (!legend1.empty()) {
        cv::rectangle(img, cv::Rect(x1 - 250, bottom - 46, 14, 10), c2, cv::FILLED);
        cv::putText(img, legend1, cv::Point(x1 - 230, bottom - 36), cv::FONT_HERSHEY_SIMPLEX, 0.45, ink, 1, cv::LINE_AA);
        cv::rectangle(img, cv::Rect(x1 - 250, bottom - 26, 14, 10), cv::Scalar(214, 120, 42), cv::FILLED);
        cv::putText(img, legend2, cv::Point(x1 - 230, bottom - 16), cv::FONT_HERSHEY_SIMPLEX, 0.45, ink, 1, cv::LINE_AA);
    }
}

} // namespace h3

#endif
