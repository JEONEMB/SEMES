// interp_kernel.cpp — 보간 커널 구현 (Keys 1981, Blu·Thévenaz·Unser 2001, Unser 1993)
#include "ksh/interp_kernel.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace ksh {

namespace {
double keys_weight(double t, double a) {
    t = std::abs(t);
    if (t < 1.0) return ((a + 2.0) * t - (a + 3.0)) * t * t + 1.0;
    if (t < 2.0) return ((a * t - 5.0 * a) * t + 8.0 * a) * t - 4.0 * a;
    return 0.0;
}
double sinc(double x) { return std::abs(x) < 1e-12 ? 1.0 : std::sin(CV_PI * x) / (CV_PI * x); }
} // namespace

// 커널 이름과 맞춤 타입으로 설정
bool parse_kernel(const std::string &name, Kernel &kernel) {
    if (name == "linear") kernel = Kernel::linear;
    else if (name == "keys05") kernel = Kernel::keys05;
    else if (name == "keys075") kernel = Kernel::keys075;
    else if (name == "lanczos3") kernel = Kernel::lanczos3;
    else if (name == "bspline3") kernel = Kernel::bspline3;
    else if (name == "omoms3") kernel = Kernel::omoms3;
    // 주어진 커널 이름 외 다른 이름이면 없다고 false 반환
    else return false;

    // 정상적으로 커널을 등록하면 true 반환
    return true;
}

const char *kernel_name(Kernel kernel) {
    switch (kernel) {
    case Kernel::linear: return "linear";
    case Kernel::keys05: return "keys05";
    case Kernel::keys075: return "keys075";
    case Kernel::lanczos3: return "lanczos3";
    case Kernel::bspline3: return "bspline3";
    case Kernel::omoms3: return "omoms3";
    }
    return "?";
}

int kernel_radius(Kernel kernel) {
    return kernel == Kernel::linear ? 1 : (kernel == Kernel::lanczos3 ? 3 : 2);
}

double kernel_weight(Kernel kernel, double t) {
    const double a = std::abs(t);
    switch (kernel) {
    case Kernel::linear: return a < 1.0 ? 1.0 - a : 0.0;
    case Kernel::keys05: return keys_weight(t, -0.5);
    case Kernel::keys075: return keys_weight(t, -0.75);
    case Kernel::lanczos3: return a < 3.0 ? sinc(t) * sinc(t / 3.0) : 0.0;
    case Kernel::bspline3:
        if (a < 1.0) return (4.0 - 6.0 * a * a + 3.0 * a * a * a) / 6.0;
        if (a < 2.0) return (2.0 - a) * (2.0 - a) * (2.0 - a) / 6.0;
        return 0.0;
    case Kernel::omoms3:   // 같은 지지 폭(4탭)의 3차 커널 중 근사 오차가 가장 작음
        if (a < 1.0) return 0.5 * a * a * a - a * a + a / 14.0 + 13.0 / 21.0;
        if (a < 2.0) return -a * a * a / 6.0 + a * a - 85.0 / 42.0 * a + 29.0 / 21.0;
        return 0.0;
    }
    return 0.0;
}

double kernel_pole(Kernel kernel) {
    if (kernel == Kernel::bspline3) return std::sqrt(3.0) - 2.0;
    if (kernel == Kernel::omoms3) return (std::sqrt(105.0) - 13.0) / 8.0;
    return 0.0;
}

void prefilter_1d(double *c, int n, double pole) {
    const double z = pole;
    const double gain = (1.0 - z) * (1.0 - 1.0 / z);
    for (int k = 0; k < n; ++k) c[k] *= gain;
    const int horizon = std::min(n, 40);
    double zn = z, sum = c[0];
    for (int k = 1; k < horizon; ++k) { sum += zn * c[k]; zn *= z; }
    c[0] = sum;
    for (int k = 1; k < n; ++k) c[k] += z * c[k - 1];   // 인과 방향
    c[n - 1] = (z / (z * z - 1.0)) * (z * c[n - 2] + c[n - 1]);
    for (int k = n - 2; k >= 0; --k) c[k] = z * (c[k + 1] - c[k]);   // 반인과 방향
}

void to_kernel_coeffs(cv::Mat &image64, Kernel kernel) {
    const double z = kernel_pole(kernel);
    if (z == 0.0) return;
    for (int y = 0; y < image64.rows; ++y) prefilter_1d(image64.ptr<double>(y), image64.cols, z);
    std::vector<double> col(image64.rows);
    for (int x = 0; x < image64.cols; ++x) {
        for (int y = 0; y < image64.rows; ++y) col[y] = image64.at<double>(y, x);
        prefilter_1d(col.data(), image64.rows, z);
        for (int y = 0; y < image64.rows; ++y) image64.at<double>(y, x) = col[y];
    }
}

} // namespace ksh
