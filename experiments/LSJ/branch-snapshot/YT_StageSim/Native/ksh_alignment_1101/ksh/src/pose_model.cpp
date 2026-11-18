// pose_model.cpp — ④ 특징 ↔ 자세 모델 구현
#include "ksh/pose_model.hpp"

#include <cmath>
#include <ctime>
#include <filesystem>

namespace ksh {

namespace {

// 기저 [1, Y, Θ, Θ², ...]
std::vector<double> basis(double y_mm, double t_deg, int theta_order) {
    std::vector<double> b = {1.0, y_mm};
    double tp = 1.0;
    for (int k = 1; k <= theta_order; ++k) { tp *= t_deg; b.push_back(tp); }
    return b;
}

bool fit_one(const std::vector<FeatureSample> &samples, bool use_s, int theta_order, FeatureModel &m) {
    const int n = int(samples.size()), k = theta_order + 2;
    if (theta_order < 1 || n < k) return false;
    cv::Mat A(n, k, CV_64F), B(n, 1, CV_64F), x;
    for (int i = 0; i < n; ++i) {
        const std::vector<double> b = basis(samples[i].y_mm, samples[i].t_deg, theta_order);
        for (int j = 0; j < k; ++j) A.at<double>(i, j) = b[j];
        B.at<double>(i) = use_s ? samples[i].s_px : samples[i].d_px;
    }
    if (!cv::solve(A, B, x, cv::DECOMP_SVD)) return false;
    m.c_px = x.at<double>(0);
    m.a_px_per_mm = x.at<double>(1);
    m.b_px_per_deg_k.assign(x.begin<double>() + 2, x.end<double>());
    const cv::Mat r = A * x - B;
    m.sigma_px = std::max(1e-6, std::sqrt(r.dot(r) / n));
    return true;
}

void write_feature(cv::FileStorage &fs, const char *name, const FeatureModel &m) {
    fs << name << "{";
    fs << "c_px" << m.c_px << "a_px_per_mm" << m.a_px_per_mm;
    fs << "theta_px_per_deg_k" << "[:";
    for (double b : m.b_px_per_deg_k) fs << b;
    fs << "]";
    fs << "sigma_px" << m.sigma_px;
    fs << "}";
}

bool read_feature(const cv::FileNode &n, FeatureModel &m) {
    if (n.empty()) return false;
    m = FeatureModel();
    m.c_px = double(n["c_px"]);
    m.a_px_per_mm = double(n["a_px_per_mm"]);
    const cv::FileNode k = n["theta_px_per_deg_k"];
    if (k.isSeq()) for (auto it = k.begin(); it != k.end(); ++it) m.b_px_per_deg_k.push_back(double(*it));
    else if (!n["b_px_per_deg"].empty()) m.b_px_per_deg_k.push_back(double(n["b_px_per_deg"]));   // 이전 형식 (1차식)
    if (!n["sigma_px"].empty()) m.sigma_px = double(n["sigma_px"]);
    return !m.b_px_per_deg_k.empty() && m.sigma_px > 0;
}

} // namespace

double eval_feature(const FeatureModel &m, double y_mm, double t_deg, double *dv_dy, double *dv_dt) {
    double v = m.c_px + m.a_px_per_mm * y_mm, dt = 0.0, tp = 1.0;   // tp = Θ^(k−1)
    for (size_t k = 0; k < m.b_px_per_deg_k.size(); ++k) {
        dt += double(k + 1) * m.b_px_per_deg_k[k] * tp;
        tp *= t_deg;
        v += m.b_px_per_deg_k[k] * tp;
    }
    if (dv_dy) *dv_dy = m.a_px_per_mm;
    if (dv_dt) *dv_dt = dt;
    return v;
}

bool invert_pose_model(const PoseModel &model, double s_px, double d_px, double &y_mm, double &t_deg) {
    const FeatureModel &S = model.s, &D = model.d;
    if (S.b_px_per_deg_k.empty() || D.b_px_per_deg_k.empty() || !std::isfinite(s_px) || !std::isfinite(d_px)) return false;
    // 출발점 : 1차 항만 쓴 두 직선의 교점
    const double det = S.a_px_per_mm * D.b_px_per_deg_k[0] - D.a_px_per_mm * S.b_px_per_deg_k[0];
    if (std::abs(det) < 1e-12) return false;
    const double rs = s_px - S.c_px, rd = d_px - D.c_px;
    double y = (rs * D.b_px_per_deg_k[0] - rd * S.b_px_per_deg_k[0]) / det;
    double t = (S.a_px_per_mm * rd - D.a_px_per_mm * rs) / det;
    // 가중 가우스–뉴턴 (가중 = 1/σ²). 1차식이면 첫 걸음에서 바로 멈춤
    const double ws = 1.0 / (S.sigma_px * S.sigma_px), wd = 1.0 / (D.sigma_px * D.sigma_px);
    for (int it = 0; it < 30; ++it) {
        double sy, st, dy, dt;
        const double es = s_px - eval_feature(S, y, t, &sy, &st), ed = d_px - eval_feature(D, y, t, &dy, &dt);
        const double a00 = ws * sy * sy + wd * dy * dy, a01 = ws * sy * st + wd * dy * dt, a11 = ws * st * st + wd * dt * dt;
        const double b0 = ws * sy * es + wd * dy * ed, b1 = ws * st * es + wd * dt * ed;
        const double dd = a00 * a11 - a01 * a01;
        if (std::abs(dd) < 1e-18) return false;
        const double ddy = (a11 * b0 - a01 * b1) / dd, ddt = (a00 * b1 - a01 * b0) / dd;
        y += ddy; t += ddt;
        if (std::abs(ddy) < 1e-10 && std::abs(ddt) < 1e-11) break;
    }
    if (!std::isfinite(y) || !std::isfinite(t)) return false;
    y_mm = y; t_deg = t;
    return true;
}

bool fit_pose_model(const std::vector<FeatureSample> &samples, int theta_order_s, int theta_order_d, PoseModel &model) {
    PoseModel m;
    if (!fit_one(samples, true, theta_order_s, m.s) || !fit_one(samples, false, theta_order_d, m.d)) return false;
    model = m;
    return true;
}

bool read_pose_model(const cv::FileNode &node, PoseModel &model) {
    PoseModel m;
    if (!read_feature(node["s"], m.s) || !read_feature(node["d"], m.d)) return false;
    model = m;
    return true;
}

bool save_pose_model(const std::string &path, const PoseModel &model, const std::string &note, int n_samples) {
    cv::FileStorage fs(path, cv::FileStorage::WRITE);
    if (!fs.isOpened()) return false;
    char date[32];
    const std::time_t now = std::time(nullptr);
    std::tm local_tm{};
#ifdef _WIN32
    localtime_s(&local_tm, &now);   // MSVC 는 std::localtime 을 안전하지 않은 함수(C4996)로 막음
#else
    localtime_r(&now, &local_tm);
#endif
    std::strftime(date, sizeof(date), "%Y-%m-%d %H:%M", &local_tm);
    fs << "trained_at" << date;
    fs << "note" << note;
    fs << "n_samples" << n_samples;
    fs << "theta_order_s" << int(model.s.b_px_per_deg_k.size());
    fs << "theta_order_d" << int(model.d.b_px_per_deg_k.size());
    fs << "model" << "{";
    write_feature(fs, "s", model.s);
    write_feature(fs, "d", model.d);
    fs << "}";
    return true;
}

bool load_pose_model(const std::string &path, PoseModel &model) {
    std::error_code ec;
    if (!std::filesystem::exists(std::filesystem::u8path(path), ec)) return false;
    cv::FileStorage fs;
    try {
        if (!fs.open(path, cv::FileStorage::READ)) return false;
    } catch (const cv::Exception &) {
        return false;
    }
    return read_pose_model(fs["model"], model);
}

} // namespace ksh
