// scipy.ndimage 3차 B-spline (spline_filter, map_coordinates order=3) 와 np.gradient 를 그대로 옮긴 것.
//   spline_filter(mode='mirror'): 극점 z=sqrt(3)-2, 인과 초기값은 거울 경계의 전체 합.
//   map_coordinates(prefilter=False, mode='nearest'): 좌표는 자르지 않고 계수 색인만 가장자리로 자른다.
//   map_coordinates(prefilter=True, mode='nearest'): 가장자리 값으로 12칸 덧댄 뒤 거울 필터 (scipy 1.6 이후 방식).
// Python scipy 1.18 과 같은 값인지 무작위 배열로 확인했다 (차이 1e-12 이하).
#pragma once
#include <cmath>
#include <vector>
#include <opencv2/core.hpp>

namespace g09 {

inline void spline_filter1d(double* c, int n, int stride) {
    const double z = std::sqrt(3.0) - 2.0;
    std::vector<double> v(n);
    for (int i = 0; i < n; ++i) v[i] = c[(size_t)i * stride] * 6.0;  // gain (1-z)(1-1/z) = 6
    if (n == 1) return;
    const double zn1 = std::pow(z, n - 1);
    double c0 = v[0] + zn1 * v[n - 1], zi = z;
    for (int i = 1; i < n - 1; ++i) { c0 += zi * (v[i] + zn1 * v[n - 1 - i]); zi *= z; }
    v[0] = c0 / (1 - zn1 * zn1);
    for (int i = 1; i < n; ++i) v[i] += z * v[i - 1];
    v[n - 1] = (z * v[n - 2] + v[n - 1]) * z / (z * z - 1);
    for (int i = n - 2; i >= 0; --i) v[i] = z * (v[i + 1] - v[i]);
    for (int i = 0; i < n; ++i) c[(size_t)i * stride] = v[i];
}

// 2차원 거울 경계 spline_filter. 축 0(세로) 먼저, 축 1(가로) 다음 (scipy 순서)
inline cv::Mat spline_filter(const cv::Mat& src64) {
    cv::Mat c = src64.clone();
    CV_Assert(c.type() == CV_64F && c.isContinuous());
    double* p = c.ptr<double>();
    for (int x = 0; x < c.cols; ++x) spline_filter1d(p + x, c.rows, c.cols);
    for (int y = 0; y < c.rows; ++y) spline_filter1d(p + (size_t)y * c.cols, c.cols, 1);
    return c;
}

// 계수 배열 + 좌표 오프셋. sample(y, x) 는 원래 배열 좌표
struct Spline {
    cv::Mat c;      // CV_64F 계수
    double off = 0; // 덧댄 칸 수
    // prefilter=False, mode='nearest' 로 쓸 계수 (spline_filter 결과 그대로)
    static Spline from_coeff(const cv::Mat& coeff) { return Spline{coeff, 0}; }
    // map_coordinates(prefilter=True, mode='nearest') 와 같은 표본기
    static Spline prefiltered_nearest(const cv::Mat& src64) {
        cv::Mat pad;
        cv::copyMakeBorder(src64, pad, 12, 12, 12, 12, cv::BORDER_REPLICATE);
        return Spline{spline_filter(pad), 12};
    }
    inline double sample(double y, double x) const {
        y += off; x += off;
        const int H = c.rows, W = c.cols;
        double wy[4], wx[4];
        weights(y, wy);
        weights(x, wx);
        const int y0 = (int)std::floor(y) - 1, x0 = (int)std::floor(x) - 1;
        int iy[4], ix[4];
        for (int k = 0; k < 4; ++k) {
            iy[k] = std::min(std::max(y0 + k, 0), H - 1);
            ix[k] = std::min(std::max(x0 + k, 0), W - 1);
        }
        double s = 0;
        for (int a = 0; a < 4; ++a) {
            const double* row = c.ptr<double>(iy[a]);
            double t = 0;
            for (int b = 0; b < 4; ++b) t += wx[b] * row[ix[b]];
            s += wy[a] * t;
        }
        return s;
    }
    static inline void weights(double x, double* w) {
        double t = x - std::floor(x);
        double y = t, z = 1.0 - t;
        w[1] = (y * y * (y - 2.0) * 3.0 + 4.0) / 6.0;
        w[2] = (z * z * (z - 2.0) * 3.0 + 4.0) / 6.0;
        w[0] = z * z * z / 6.0;
        w[3] = 1.0 - w[0] - w[1] - w[2];
    }
};

// np.gradient (edge_order=1): 안쪽은 중앙차분, 가장자리는 한쪽 차분. gy = 축 0, gx = 축 1
inline void np_gradient(const cv::Mat& a, cv::Mat& gy, cv::Mat& gx) {
    gy.create(a.size(), CV_64F);
    gx.create(a.size(), CV_64F);
    const int H = a.rows, W = a.cols;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            double vy, vx;
            if (y == 0) vy = a.at<double>(1, x) - a.at<double>(0, x);
            else if (y == H - 1) vy = a.at<double>(H - 1, x) - a.at<double>(H - 2, x);
            else vy = (a.at<double>(y + 1, x) - a.at<double>(y - 1, x)) / 2.0;
            if (x == 0) vx = a.at<double>(y, 1) - a.at<double>(y, 0);
            else if (x == W - 1) vx = a.at<double>(y, W - 1) - a.at<double>(y, W - 2);
            else vx = (a.at<double>(y, x + 1) - a.at<double>(y, x - 1)) / 2.0;
            gy.at<double>(y, x) = vy;
            gx.at<double>(y, x) = vx;
        }
}

}  // namespace g09
