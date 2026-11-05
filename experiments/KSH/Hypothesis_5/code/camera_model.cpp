// =====================================================================================
// camera_model.cpp — 운동 모델 구현 (식은 camera_model.hpp 머리말 참고)
// =====================================================================================
#include "camera_model.hpp"

#include <opencv2/core/persistence.hpp>
#include <cmath>
#include <regex>

namespace mm {

namespace {
constexpr double kDeg = CV_PI / 180.0;

cv::Matx33d rotY(double ang) {
    const double c = std::cos(ang), s = std::sin(ang);
    return cv::Matx33d(c, 0, s,
                       0, 1, 0,
                       -s, 0, c);
}
} // namespace

bool CameraModel::load(const std::string& path) {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) return false;
    auto rd = [&](const char* key, double& v) { if (!fs[key].empty()) fs[key] >> v; };
    rd("fx", fx); rd("fy", fy); rd("cx", cx); rd("cy", cy); rd("k1", k1);
    rd("z_plate", z_plate); rd("z_bg", z_bg); rd("z_wall_upper", z_wall_upper); rd("axis_a", axis_a);
    rd("qr_pitch_mm", qr_pitch_mm); rd("qr_size_mm", qr_size_mm);
    if (!fs["plate_center_x"].empty()) fs["plate_center_x"] >> plate_center.x;
    if (!fs["plate_center_y"].empty()) fs["plate_center_y"] >> plate_center.y;
    return true;
}

bool CameraModel::save(const std::string& path) const {
    cv::FileStorage fs(path, cv::FileStorage::WRITE);
    if (!fs.isOpened()) return false;
    fs << "fx" << fx << "fy" << fy << "cx" << cx << "cy" << cy << "k1" << k1;
    fs << "z_plate" << z_plate << "z_bg" << z_bg << "z_wall_upper" << z_wall_upper << "axis_a" << axis_a;
    fs << "plate_center_x" << plate_center.x << "plate_center_y" << plate_center.y;
    fs << "qr_pitch_mm" << qr_pitch_mm << "qr_size_mm" << qr_size_mm;
    return true;
}

cv::Matx33d CameraModel::K() const {
    return cv::Matx33d(fx, 0, cx,
                       0, fy, cy,
                       0, 0, 1);
}

void CameraModel::motion(const Pose& pose, cv::Matx33d& R, cv::Vec3d& C) const {
    // Θ+ 이면 장면이 +x 로 움직이도록 카메라는 R_y(−Θ) 로 회전
    R = rotY(-pose.t_deg * kDeg);
    const cv::Vec3d A(0.0, 0.0, axis_a);
    C = A - R * A;               // 축 A 둘레 회전에 따른 광학 중심 이동
    C[0] -= pose.y_mm;           // Y+ : 카메라 −x 이동 → 장면 +x 이동
}

cv::Point2d CameraModel::toPixel(double xn, double yn) const {
    const double f = 1.0 + k1 * (xn * xn + yn * yn);
    return cv::Point2d(fx * xn * f + cx, fy * yn * f + cy);
}

cv::Point2d CameraModel::toNormalized(const cv::Point2d& p) const {
    const double xd = (p.x - cx) / fx, yd = (p.y - cy) / fy;
    double xn = xd, yn = yd;
    for (int i = 0; i < 8; ++i) {          // 왜곡 역변환 (고정점 반복)
        const double f = 1.0 + k1 * (xn * xn + yn * yn);
        xn = xd / f; yn = yd / f;
    }
    return cv::Point2d(xn, yn);
}

cv::Vec3d CameraModel::backProject(const cv::Point2d& p, double Z) const {
    const cv::Point2d n = toNormalized(p);
    return cv::Vec3d(n.x * Z, n.y * Z, Z);
}

cv::Point2d CameraModel::project(const cv::Vec3d& Xw, const Pose& pose) const {
    cv::Matx33d R; cv::Vec3d C;
    motion(pose, R, C);
    const cv::Vec3d Xc = R.t() * (Xw - C);
    return toPixel(Xc[0] / Xc[2], Xc[1] / Xc[2]);
}

cv::Point2d CameraModel::mapToReference(const cv::Point2d& q, double Z, const Pose& pose) const {
    cv::Matx33d R; cv::Vec3d C;
    motion(pose, R, C);
    const cv::Point2d n = toNormalized(q);
    const cv::Vec3d d = R * cv::Vec3d(n.x, n.y, 1.0);   // 현재 카메라 광선 (세계 좌표)
    const double s = (Z - C[2]) / d[2];
    return project(C + s * d, Pose{});
}

cv::Point2d CameraModel::displacement(const cv::Point2d& p, double Z, const Pose& pose) const {
    return project(backProject(p, Z), pose) - p;
}

cv::Point2d CameraModel::displacementLinear(const cv::Point2d& p, double Z, const Pose& pose) const {
    const double th = pose.t_deg * kDeg;
    const double u = p.x - cx, v = p.y - cy;
    const double xn = u / fx, yn = v / fy;
    const double g = 1.0 + 3.0 * k1 * xn * xn + k1 * yn * yn;   // 왜곡에 의한 가로 배율 (1차 근사)
    const double du = g * (fx * pose.y_mm / Z                   // Y : 깊이에 반비례
                         + fx * th * (1.0 + xn * xn)            // Θ : 순수 회전 (U 자 2차 항 포함)
                         - fx * axis_a * th / Z);               // Θ : 축 어긋남에 의한 평행 이동
    const double dv = th * u * v / fx;                          // Θ : 세로 u·v 항 (왜곡 보정 생략)
    return cv::Point2d(du, dv);
}

cv::Matx33d CameraModel::planeHomography(double Z, const Pose& pose) const {
    cv::Matx33d R; cv::Vec3d C;
    motion(pose, R, C);
    const cv::Matx33d M(Z - 0.0, 0.0, -C[0],
                        0.0, Z, -C[1],
                        0.0, 0.0, Z - C[2]);   // Z·I − C·e3ᵀ
    cv::Matx33d H = K() * R.t() * M * K().inv();
    return H * (1.0 / H(2, 2));
}

cv::Point2d CameraModel::qrCenterRef(int k) const {
    const double pitch_px = qr_pitch_mm * fx / z_plate;
    return cv::Point2d(plate_center.x + (k - 1) * pitch_px, plate_center.y);
}

std::array<std::array<cv::Point2d, 4>, 3> CameraModel::qrCorners(const Pose& pose) const {
    std::array<std::array<cv::Point2d, 4>, 3> out;
    const cv::Vec3d X0 = backProject(plate_center, z_plate);
    const double s = qr_size_mm / 2.0;
    const cv::Vec3d off[4] = {cv::Vec3d(-s, -s, 0), cv::Vec3d(s, -s, 0), cv::Vec3d(s, s, 0), cv::Vec3d(-s, s, 0)};
    for (int k = 0; k < 3; ++k) {
        const cv::Vec3d Xc = X0 + cv::Vec3d((k - 1) * qr_pitch_mm, 0, 0);
        for (int i = 0; i < 4; ++i) out[k][i] = project(Xc + off[i], pose);
    }
    return out;
}

QrPrediction predictQr(const CameraModel& cam, const Pose& pose) {
    QrPrediction q;
    const auto qc = cam.qrCorners(pose);
    for (int k = 0; k < 3; ++k) {
        const auto& c = qc[k];   // TL, TR, BR, BL
        q.w[k] = 0.5 * ((c[1].x - c[0].x) + (c[2].x - c[3].x));
        q.h[k] = 0.5 * ((c[3].y - c[0].y) + (c[2].y - c[1].y));
        // 중심 = 대각선 교점 (EDA C-1 정의)
        const cv::Point2d d1 = c[2] - c[0], d2 = c[3] - c[1], r = c[1] - c[0];
        const double den = d1.x * d2.y - d1.y * d2.x;
        const double t = (r.x * d2.y - r.y * d2.x) / den;
        const cv::Point2d ctr = c[0] + t * d1;
        q.plate_cx += ctr.x / 3.0;
        q.plate_cy += ctr.y / 3.0;
    }
    q.w_diff = q.w[2] - q.w[0];
    q.h_diff = q.h[2] - q.h[0];
    return q;
}

Sensitivity sensitivities(const CameraModel& cam) {
    Sensitivity s;
    const double dy = 0.5, dt = 0.05;  // 중앙 차분 간격 (mm, °)
    auto P = [&](double y, double t) { return predictQr(cam, Pose{y, t}); };
    const QrPrediction yp = P(dy, 0), ym = P(-dy, 0), tp = P(0, dt), tm = P(0, -dt);
    s.plate_per_mm = (yp.plate_cx - ym.plate_cx) / (2 * dy);
    s.plate_per_deg = (tp.plate_cx - tm.plate_cx) / (2 * dt);
    s.wdiff_per_mm = (yp.w_diff - ym.w_diff) / (2 * dy);
    s.wdiff_per_deg = (tp.w_diff - tm.w_diff) / (2 * dt);
    s.hdiff_per_mm = (yp.h_diff - ym.h_diff) / (2 * dy);
    s.hdiff_per_deg = (tp.h_diff - tm.h_diff) / (2 * dt);
    const cv::Point2d p = cam.plate_center;
    s.bg_per_mm = (cam.displacement(p, cam.z_bg, Pose{dy, 0}).x - cam.displacement(p, cam.z_bg, Pose{-dy, 0}).x) / (2 * dy);
    s.bg_per_deg = (cam.displacement(p, cam.z_bg, Pose{0, dt}).x - cam.displacement(p, cam.z_bg, Pose{0, -dt}).x) / (2 * dt);
    s.equal_shift_deg_per_mm = s.plate_per_mm / s.plate_per_deg;
    return s;
}

bool parseLabel(const std::string& stem, Pose& pose, int& set_id) {
    static const std::regex reCenter(R"(^center$)");
    static const std::regex reCenterN(R"(^center(\d+)$)");
    static const std::regex reMixed(R"(^Y([+-]?\d+(?:\.\d+)?)\s+T([+-]?\d+(?:\.\d+)?)\s*\((\d+)\)$)");
    static const std::regex reY(R"(^Y([+-]?\d+(?:\.\d+)?)$)");
    static const std::regex reT(R"(^T([+-]?\d+(?:\.\d+)?)$)");
    std::smatch m;
    pose = Pose{};
    if (std::regex_match(stem, m, reCenter)) { set_id = 1; return true; }
    if (std::regex_match(stem, m, reCenterN)) { set_id = 2; return true; }
    if (std::regex_match(stem, m, reMixed)) {
        pose.y_mm = std::stod(m[1].str()); pose.t_deg = std::stod(m[2].str()); set_id = 2; return true;
    }
    if (std::regex_match(stem, m, reY)) { pose.y_mm = std::stod(m[1].str()); set_id = 1; return true; }
    if (std::regex_match(stem, m, reT)) { pose.t_deg = std::stod(m[1].str()); set_id = 1; return true; }
    return false;
}

} // namespace mm
