// pose_model.cpp — ④ 특징 ↔ 자세 모델 구현
#include "ksh/pose_model.hpp"

#include <cmath>
#include <ctime>
#include <filesystem>

namespace ksh {

namespace {

// 보정식의 기저 [1, Y, Θ, Θ², ...] — 특징 = 기저 · 계수 (Y 는 1차, Θ 는 theta_order 차까지)
std::vector<double> basis(double y_mm, double t_deg, int theta_order) {
    std::vector<double> b = {1.0, y_mm};
    double tp = 1.0;
    for (int k = 1; k <= theta_order; ++k) { tp *= t_deg; b.push_back(tp); }
    return b;
}

// 특징 하나(s 또는 d)의 계수 학습 : 최소제곱 A·x ≈ B 를 SVD 로 풂 (A 행 = 사진 한 장의 기저, B = 측정 특징)
//  σ = 학습 잔차 RMS (역산에서 그 식을 얼마나 믿을지에 씀)
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

// 모델 파일 쓰기·읽기 (특징 하나 = c_px, a_px_per_mm, theta_px_per_deg_k[], sigma_px)
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

// 특징값 v = c + a·Y + Σ b_k·Θ^k 와 미분 ∂v/∂Y = a, ∂v/∂Θ = Σ k·b_k·Θ^(k−1) (뉴턴 반복의 야코비안)
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

// ④ 역산 : 측정 (s, d) 를 동시에 만족하는 (Y, Θ) — (Y, Θ) 평면에서 두 곡선(1차식이면 직선)의 교점
//  초기값 : 1차 항만 쓴 두 직선의 교점 (크라메르 공식, det = a_s·b_d − a_d·b_s)
//  반복   : 가우스–뉴턴 (JᵀWJ)Δ = JᵀW e,  e = 측정 − 모델 예측,  W = diag(1/σ_s², 1/σ_d²)
//           1차식이면 첫 걸음에서 수렴. 식 2개·미지수 2개라 가중치는 해를 바꾸지 않음 (식이 더 많아질 때를 위한 형태)
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

// s, d 두 식을 각각 학습 (학습 도구용)
bool fit_pose_model(const std::vector<FeatureSample> &samples, int theta_order_s, int theta_order_d, PoseModel &model) {
    PoseModel m;
    if (!fit_one(samples, true, theta_order_s, m.s) || !fit_one(samples, false, theta_order_d, m.d)) return false;
    model = m;
    return true;
}

namespace {
// 세트별 상수 + 공통 기울기 : 열 = [세트 원핫 k개, Y, Θ, Θ², ...], 행 가중 √w
bool fit_one_sets(const std::vector<std::vector<FeatureSample>> &sets, bool use_s, int theta_order, bool balance, FeatureModel &m,
                  std::vector<double> *cs) {
    const int k = int(sets.size()), ncol = k + 1 + theta_order;
    int n = 0;
    for (const auto &s : sets) n += int(s.size());
    if (theta_order < 1 || k < 1 || n < ncol) return false;
    for (const auto &s : sets) if (s.empty()) return false;
    cv::Mat A = cv::Mat::zeros(n, ncol, CV_64F), B(n, 1, CV_64F), Aw, Bw, x;
    std::vector<double> w(n);
    int r = 0;
    for (int j = 0; j < k; ++j)
        for (const auto &smp : sets[j]) {
            A.at<double>(r, j) = 1.0;
            A.at<double>(r, k) = smp.y_mm;
            double tp = 1.0;
            for (int q = 1; q <= theta_order; ++q) { tp *= smp.t_deg; A.at<double>(r, k + q) = tp; }
            B.at<double>(r) = use_s ? smp.s_px : smp.d_px;
            w[r] = balance ? 1.0 / double(sets[j].size()) : 1.0;
            ++r;
        }
    Aw = A.clone(); Bw = B.clone();
    for (int i = 0; i < n; ++i) { const double sw = std::sqrt(w[i]); Aw.row(i) *= sw; Bw.at<double>(i) *= sw; }
    if (!cv::solve(Aw, Bw, x, cv::DECOMP_SVD)) return false;
    m = FeatureModel();
    m.c_px = x.at<double>(0);
    m.a_px_per_mm = x.at<double>(k);
    for (int q = 1; q <= theta_order; ++q) m.b_px_per_deg_k.push_back(x.at<double>(k + q));
    const cv::Mat res = A * x - B;
    m.sigma_px = std::max(1e-6, std::sqrt(res.dot(res) / n));
    if (cs) { cs->clear(); for (int j = 0; j < k; ++j) cs->push_back(x.at<double>(j)); }
    return true;
}
} // namespace

bool fit_pose_model_sets(const std::vector<std::vector<FeatureSample>> &sets, int theta_order_s, int theta_order_d, bool balance,
                         PoseModel &model, std::vector<double> *set_c_s, std::vector<double> *set_c_d) {
    PoseModel m;
    if (!fit_one_sets(sets, true, theta_order_s, balance, m.s, set_c_s) || !fit_one_sets(sets, false, theta_order_d, balance, m.d, set_c_d)) return false;
    model = m;
    return true;
}

// 설정 노드 model: {s, d} 읽기
bool read_pose_model(const cv::FileNode &node, PoseModel &model) {
    PoseModel m;
    if (!read_feature(node["s"], m.s) || !read_feature(node["d"], m.d)) return false;
    model = m;
    return true;
}

// 모델 파일 저장 (학습 시각 · 메모 · 표본 수 · 차수 · 계수)
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

// 모델 파일 읽기 (파일이 없으면 OpenCV 가 오류 문구를 찍으므로 먼저 존재 확인)
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
