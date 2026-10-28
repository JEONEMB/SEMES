// =====================================================================================
// alignment.cpp — Y·Θ 추정 구현
//
//  ① 거친 이동   : 플레이트 주변 창의 위상 상관 (해닝 창). 정수~소수 px, 밝기 변화에 강함
//  ② 정밀 이동   : ROI 마다 ECC 평행이동 (가우스–뉴턴 반복). 배경 ROI 는 2배 창 위상 상관으로 한 번 더 맞춘 뒤 ECC
//  ③ 특징        : s = QR 3개 평균 dx,  d = s − 배경 ROI 평균 dx  (가까운 플레이트와 먼 하단 문의 시차)
//  ④ 추정        : s = c_s + a_s·Y + b_s·Θ,  d = c_d + a_d·Y + b_d·Θ  두 직선의 교점 (2×2 연립방정식)
//                  식 2개·미지수 2개이므로 가중 최소제곱의 해 = 연립방정식의 해 (가중치는 결과에 영향 없음)
//  계수 학습     : QR_Image 연구 프로젝트의 H5 단계 (1차 첫 번째 측정 사진만, 블록 교차검증) → alignment.yml
// =====================================================================================
#include "alignment.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>

namespace {

// ---------------------------------------------------------------- 설정 (alignment.yml)
struct RoiCfg {
    std::string name;
    int cx_px = 0, cy_px = 0, hw_px = 0;   // 기준 사진 좌표의 중심, 반폭
};
struct LineCfg {                            // 특징 = c + a·Y + b·Θ
    double c_px = 0.0, a_px_per_mm = 0.0, b_px_per_deg = 0.0;
};
struct Config {
    bool loaded = false;
    std::string reference_path;
    cv::Size image_size_px;
    cv::Point coarse_center_px;
    cv::Size coarse_window_px;
    int wide_window_scale = 2;              // 배경 ROI 위상 상관 창 = ROI 의 이 배수
    double qr_max_dev_px = 15.0;            // QR 이 거친 이동에서 이만큼 벗어나면 이웃 QR 에 붙은 것으로 보고 실패
    int ecc_max_iter = 100;
    double ecc_eps = 1e-7;
    std::vector<RoiCfg> qr;
    std::vector<RoiCfg> bg;
    LineCfg s, d;
};

Config g_cfg;
cv::Mat g_ref32;            // 기준 사진 (CV_32F 회색)
AlignmentInfo g_info;

double nan_value() { return std::numeric_limits<double>::quiet_NaN(); }

bool inside(const cv::Rect &r, const cv::Size &s) {
    return r.x >= 0 && r.y >= 0 && r.x + r.width <= s.width && r.y + r.height <= s.height;
}
cv::Point round_pt(const cv::Point2d &p) { return cv::Point(int(std::lround(p.x)), int(std::lround(p.y))); }

cv::Mat to_gray32(const cv::Mat &src) {
    cv::Mat gray, out;
    if (src.channels() == 3) cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    else if (src.channels() == 4) cv::cvtColor(src, gray, cv::COLOR_BGRA2GRAY);
    else gray = src;
    gray.convertTo(out, CV_32F);
    return out;
}

std::vector<RoiCfg> read_rois(const cv::FileNode &n) {
    std::vector<RoiCfg> v;
    for (auto it = n.begin(); it != n.end(); ++it) {
        RoiCfg r;
        (*it)["name"] >> r.name;
        r.cx_px = int((*it)["cx_px"]);
        r.cy_px = int((*it)["cy_px"]);
        r.hw_px = int((*it)["hw_px"]);
        v.push_back(r);
    }
    return v;
}
LineCfg read_line(const cv::FileNode &n) {
    LineCfg l;
    l.c_px = double(n["c_px"]);
    l.a_px_per_mm = double(n["a_px_per_mm"]);
    l.b_px_per_deg = double(n["b_px_per_deg"]);
    return l;
}

// ---------------------------------------------------------------- ① 위상 상관
// b 가 a 대비 이동한 양 (px). cv::phaseCorrelate 는 창 함수를 입력에 제자리로 곱하므로 복사본을 넘김
cv::Point2d phase_shift(const cv::Mat &a, const cv::Mat &b) {
    cv::Mat win;
    cv::createHanningWindow(win, a.size(), CV_32F);
    cv::Mat a2 = a.clone(), b2 = b.clone();
    return cv::phaseCorrelate(a2, b2, win);
}

// ---------------------------------------------------------------- ② ECC
// OpenCV 디버그 라이브러리는 예외 전에 오류 문구를 콘솔에 찍음 → ECC 동안만 문구 출력을 막음 (미수렴 = 정상적인 측정 실패 경로)
int quiet_error(int, const char *, const char *, const char *, int, void *) { return 0; }

bool ecc_translation(const cv::Mat &tmpl, const cv::Mat &img, const cv::Point2d &init_px, cv::Point2d &out_px, double &cc) {
    cv::Mat warp = cv::Mat::eye(2, 3, CV_32F);
    warp.at<float>(0, 2) = float(init_px.x);
    warp.at<float>(1, 2) = float(init_px.y);
    void *prev_data = nullptr;
    const cv::ErrorCallback prev = cv::redirectError(quiet_error, nullptr, &prev_data);
    try {
        cc = cv::findTransformECC(tmpl, img, warp, cv::MOTION_TRANSLATION,
                                  cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, g_cfg.ecc_max_iter, g_cfg.ecc_eps),
                                  cv::noArray());
    } catch (const cv::Exception &) {
        cv::redirectError(prev, prev_data);
        return false;
    }
    cv::redirectError(prev, prev_data);
    out_px = cv::Point2d(warp.at<float>(0, 2), warp.at<float>(1, 2));
    return true;
}

// ROI 하나의 이동. wide = 넓은 창 위상 상관을 먼저 (배경). QR 은 반복 무늬라 넓은 창을 쓰면 이웃 QR 에 붙을 수 있어 바로 ECC
AlignmentRoi measure_roi(const cv::Mat &ref, const cv::Mat &cur, const RoiCfg &r, const cv::Point2d &guess_px, bool wide) {
    AlignmentRoi m;
    m.name = r.name;
    m.ref_rect_px = cv::Rect(r.cx_px - r.hw_px, r.cy_px - r.hw_px, 2 * r.hw_px, 2 * r.hw_px);
    cv::Point2d g = guess_px;
    if (wide) {
        const int half_px = g_cfg.wide_window_scale * r.hw_px;
        const cv::Rect A(r.cx_px - half_px, r.cy_px - half_px, 2 * half_px, 2 * half_px);
        const cv::Rect B = A + round_pt(guess_px);
        if (!inside(A, ref.size()) || !inside(B, cur.size())) return m;
        g = cv::Point2d(round_pt(guess_px)) + phase_shift(ref(A), cur(B));
    }
    const cv::Point gi = round_pt(g);
    const cv::Rect C = m.ref_rect_px + gi;
    if (!inside(m.ref_rect_px, ref.size()) || !inside(C, cur.size())) return m;
    cv::Point2d frac_px;
    if (!ecc_translation(ref(m.ref_rect_px), cur(C), g - cv::Point2d(gi), frac_px, m.ecc_cc)) return m;
    m.shift_px = frac_px + cv::Point2d(gi);
    m.ok = true;
    return m;
}

bool ensure_ready() {
    if (g_cfg.loaded && !g_ref32.empty()) return true;
    if (!g_cfg.loaded && !alignment_init()) return false;
    return !g_ref32.empty();
}

} // namespace

// =====================================================================================
bool alignment_init(const std::string &config_path) {
    g_cfg = Config();
    g_ref32.release();
    cv::FileStorage fs;
    try {
        if (!fs.open(config_path, cv::FileStorage::READ)) {
            std::cerr << "[alignment] 설정 파일을 열 수 없음 : " << config_path << "\n";
            return false;
        }
    } catch (const cv::Exception &) {
        std::cerr << "[alignment] 설정 파일 형식 오류 : " << config_path << "\n";
        return false;
    }
    Config c;
    fs["reference_path"] >> c.reference_path;
    c.image_size_px = cv::Size(int(fs["image_width_px"]), int(fs["image_height_px"]));
    const cv::FileNode co = fs["coarse"];
    c.coarse_center_px = cv::Point(int(co["center_x_px"]), int(co["center_y_px"]));
    c.coarse_window_px = cv::Size(int(co["width_px"]), int(co["height_px"]));
    c.wide_window_scale = int(fs["wide_window_scale"]);
    c.qr_max_dev_px = double(fs["qr_max_dev_px"]);
    c.ecc_max_iter = int(fs["ecc"]["max_iter"]);
    c.ecc_eps = double(fs["ecc"]["eps"]);
    c.qr = read_rois(fs["qr_rois"]);
    c.bg = read_rois(fs["background_rois"]);
    c.s = read_line(fs["model"]["s"]);
    c.d = read_line(fs["model"]["d"]);
    if (c.qr.empty() || c.bg.empty() || c.coarse_window_px.area() == 0 || c.wide_window_scale <= 0 || c.ecc_max_iter <= 0) {
        std::cerr << "[alignment] 설정 항목 누락 : " << config_path << "\n";
        return false;
    }
    g_cfg = c;
    g_cfg.loaded = true;
    // 기준 사진 경로는 설정 파일 위치 기준 상대 경로도 허용
    std::string ref_path = c.reference_path;
    cv::Mat ref = cv::imread(ref_path, cv::IMREAD_GRAYSCALE);
    if (ref.empty()) {
        const size_t p = config_path.find_last_of("/\\");
        if (p != std::string::npos) ref = cv::imread(config_path.substr(0, p + 1) + ref_path, cv::IMREAD_GRAYSCALE);
    }
    if (ref.empty()) {
        std::cerr << "[alignment] 기준 사진을 읽을 수 없음 : " << ref_path << "\n";
        return true;   // 설정은 읽음. alignment_set_reference 로 기준을 따로 줄 수 있음
    }
    return alignment_set_reference(ref);
}

bool alignment_set_reference(const cv::Mat &reference) {
    if (reference.empty()) return false;
    if (g_cfg.image_size_px.area() > 0 && reference.size() != g_cfg.image_size_px) {
        std::cerr << "[alignment] 기준 사진 크기 " << reference.cols << "×" << reference.rows << " ≠ 설정 "
                  << g_cfg.image_size_px.width << "×" << g_cfg.image_size_px.height << "\n";
        return false;
    }
    g_ref32 = to_gray32(reference);
    return true;
}

const AlignmentInfo &alignment_last_info() { return g_info; }

// =====================================================================================
Alignment_t alignment(cv::Mat &src)
{
    Alignment_t result;
    result.y = nan_value();
    result.t = nan_value();

    const auto t0 = std::chrono::steady_clock::now();
    g_info = AlignmentInfo();
    auto finish = [&](const std::string &msg) {
        g_info.message = msg;
        g_info.elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        return result;
    };

    if (!ensure_ready()) return finish("설정 또는 기준 사진 없음");
    if (src.empty()) return finish("입력 사진이 비어 있음");
    if (src.size() != g_ref32.size()) return finish("입력 사진 크기가 기준 사진과 다름");
    const cv::Mat cur = to_gray32(src);

    // ① 거친 이동 : 플레이트 주변 창
    const cv::Rect W(g_cfg.coarse_center_px.x - g_cfg.coarse_window_px.width / 2,
                     g_cfg.coarse_center_px.y - g_cfg.coarse_window_px.height / 2,
                     g_cfg.coarse_window_px.width, g_cfg.coarse_window_px.height);
    if (!inside(W, cur.size())) return finish("거친 이동 창이 사진 밖");
    g_info.coarse_px = phase_shift(g_ref32(W), cur(W));

    // ② 정밀 이동 : QR (바로 ECC) + 배경 ROI (넓은 창 위상 상관 → ECC)
    double qr_sum_px = 0.0, bg_sum_px = 0.0;
    bool all_ok = true;
    for (const auto &r : g_cfg.qr) {
        AlignmentRoi m = measure_roi(g_ref32, cur, r, g_info.coarse_px, false);
        if (m.ok && cv::norm(m.shift_px - g_info.coarse_px) > g_cfg.qr_max_dev_px) m.ok = false;
        all_ok = all_ok && m.ok;
        qr_sum_px += m.shift_px.x;
        g_info.rois.push_back(m);
    }
    for (const auto &r : g_cfg.bg) {
        const AlignmentRoi m = measure_roi(g_ref32, cur, r, g_info.coarse_px, true);
        all_ok = all_ok && m.ok;
        bg_sum_px += m.shift_px.x;
        g_info.rois.push_back(m);
    }
    if (!all_ok) return finish("ROI 측정 실패 (ECC 미수렴 또는 사진 밖)");

    // ③ 특징
    g_info.s_px = qr_sum_px / double(g_cfg.qr.size());
    g_info.d_px = g_info.s_px - bg_sum_px / double(g_cfg.bg.size());

    // ④ 두 직선의 교점
    const LineCfg &S = g_cfg.s, &D = g_cfg.d;
    const double det = S.a_px_per_mm * D.b_px_per_deg - D.a_px_per_mm * S.b_px_per_deg;
    if (std::abs(det) < 1e-12) return finish("모델 계수가 Y·Θ 를 구별하지 못함 (행렬식 0)");
    const double rs_px = g_info.s_px - S.c_px, rd_px = g_info.d_px - D.c_px;
    result.y = (rs_px * D.b_px_per_deg - rd_px * S.b_px_per_deg) / det;
    result.t = (S.a_px_per_mm * rd_px - D.a_px_per_mm * rs_px) / det;
    g_info.ok = true;
    return finish("");
}
