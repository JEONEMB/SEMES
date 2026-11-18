// =====================================================================================
// sim_config.cpp — 설정 읽기·쓰기
// =====================================================================================
#include "sim_config.hpp"
#include "sim_paths.hpp"

#include <opencv2/core.hpp>

#include <filesystem>

namespace sim {

namespace {
template <typename T> void rd(const cv::FileNode& n, const char* key, T& v) {
    const cv::FileNode x = n[key];
    if (!x.empty()) x >> v;
}
void rdb(const cv::FileNode& n, const char* key, bool& v) {
    const cv::FileNode x = n[key];
    if (!x.empty()) v = int(x) != 0;
}
void rdu(const cv::FileNode& n, const char* key, unsigned int& v) {
    const cv::FileNode x = n[key];
    if (!x.empty()) v = unsigned(int(x));
}
// 문자열 하나 또는 문자열 목록
void rdl(const cv::FileNode& n, const char* key, std::vector<std::string>& v) {
    const cv::FileNode x = n[key];
    if (x.empty()) return;
    v.clear();
    if (x.isSeq()) { for (const auto& e : x) v.push_back(std::string(e)); }
    else v.push_back(std::string(x));
}
void wrl(cv::FileStorage& fs, const char* key, const std::vector<std::string>& v) {
    fs << key << "[";
    for (const auto& s : v) fs << s;
    fs << "]";
}
} // namespace

bool SimConfig::load(const std::string& path) {
    std::error_code ec;
    if (!std::filesystem::exists(u8p(path), ec)) return false;   // 없는 파일을 열면 OpenCV 가 오류 문구를 찍으므로 먼저 확인
    cv::FileStorage fs;
    try {
        if (!fs.open(path, cv::FileStorage::READ)) return false;
    } catch (const cv::Exception&) {
        return false;
    }
    const cv::FileNode p = fs["paths"], e = fs["estimator"], f = fs["field"], c = fs["canvas"], r = fs["render"], ro = fs["robot"],
                       ct = fs["control"], t = fs["trial"], v = fs["validation"], d = fs["display"];
    rdl(p, "image_dir", data_dirs);                 // 이전 이름 (문자열 하나)
    rdl(p, "data_dirs", data_dirs);
    rdl(p, "alignment_config", alignment_configs);  // 이전 이름
    rdl(p, "alignment_configs", alignment_configs);
    rd(p, "world_dir", world_dir); rd(p, "result_dir", result_dir); rd(p, "reference_photo", reference_photo);
    rd(e, "type", estimator_type); rd(e, "plugin", estimator_plugin); rd(e, "plugin_config", estimator_plugin_config); rd(e, "zero_calibration", zero_calibration);
    rd(f, "step_px", field_step_px); rd(f, "window_px", field_window_px); rd(f, "theta_step_deg", field_theta_step_deg);
    rd(f, "min_ok_ratio", field_min_ok_ratio); rd(f, "max_resid_px", field_max_resid_px); rd(f, "min_cc", field_min_cc); rd(f, "min_texture", field_min_texture);
    rd(f, "edge_min_energy", field_edge_min_energy); rd(f, "edge_min_ratio", field_edge_min_ratio); rd(f, "edge_search_px", field_edge_search_px);
    rd(c, "margin_x_px", canvas_margin_x_px); rd(c, "margin_y_px", canvas_margin_y_px);
    rd(r, "noise_sigma", render_noise_sigma); rd(r, "gain", render_gain); rdb(r, "reference_offset", render_reference_offset); rdb(r, "teach_center_photo", teach_center_photo);
    rd(ro, "max_step_y_mm", robot_max_step_y_mm); rd(ro, "max_step_t_deg", robot_max_step_t_deg); rd(ro, "gain_error", robot_gain_error);
    rd(ro, "repeat_y_um", robot_repeat_y_um); rd(ro, "repeat_t_deg", robot_repeat_t_deg);
    rd(ct, "max_moves", control_max_moves); rd(ct, "tol_y_um", tol_y_um); rd(ct, "tol_t_deg", tol_t_deg); rd(ct, "stop_ratio", control_stop_ratio);
    rd(t, "start_range_y_mm", start_range_y_mm); rd(t, "start_range_t_deg", start_range_t_deg);
    rd(t, "monte_carlo_trials", monte_carlo_trials); rdu(t, "seed", seed);
    rdb(v, "include_set3", validate_include_set3);
    rd(d, "plot_font_scale", plot_font_scale); rd(d, "ui_font_pt", ui_font_pt);
    return true;
}

bool SimConfig::save(const std::string& path) const {
    cv::FileStorage fs;
    try {
        if (!fs.open(path, cv::FileStorage::WRITE)) return false;
    } catch (const cv::Exception&) {
        return false;
    }
    fs << "paths" << "{";
    wrl(fs, "data_dirs", data_dirs);
    wrl(fs, "alignment_configs", alignment_configs);
    fs << "reference_photo" << reference_photo << "world_dir" << world_dir << "result_dir" << result_dir << "}";
    fs << "estimator" << "{" << "type" << estimator_type << "plugin" << estimator_plugin << "plugin_config" << estimator_plugin_config << "zero_calibration" << zero_calibration << "}";
    fs << "field" << "{" << "step_px" << field_step_px << "window_px" << field_window_px << "theta_step_deg" << field_theta_step_deg
       << "min_ok_ratio" << field_min_ok_ratio << "max_resid_px" << field_max_resid_px << "min_cc" << field_min_cc << "min_texture" << field_min_texture
       << "edge_min_energy" << field_edge_min_energy << "edge_min_ratio" << field_edge_min_ratio << "edge_search_px" << field_edge_search_px << "}";
    fs << "canvas" << "{" << "margin_x_px" << canvas_margin_x_px << "margin_y_px" << canvas_margin_y_px << "}";
    fs << "render" << "{" << "noise_sigma" << render_noise_sigma << "gain" << render_gain << "reference_offset" << int(render_reference_offset) << "teach_center_photo" << int(teach_center_photo) << "}";
    fs << "robot" << "{" << "max_step_y_mm" << robot_max_step_y_mm << "max_step_t_deg" << robot_max_step_t_deg << "gain_error" << robot_gain_error
       << "repeat_y_um" << robot_repeat_y_um << "repeat_t_deg" << robot_repeat_t_deg << "}";
    fs << "control" << "{" << "max_moves" << control_max_moves << "tol_y_um" << tol_y_um << "tol_t_deg" << tol_t_deg
       << "stop_ratio" << control_stop_ratio << "}";
    fs << "trial" << "{" << "start_range_y_mm" << start_range_y_mm << "start_range_t_deg" << start_range_t_deg
       << "monte_carlo_trials" << monte_carlo_trials << "seed" << int(seed) << "}";
    fs << "validation" << "{" << "include_set3" << int(validate_include_set3) << "}";
    fs << "display" << "{" << "plot_font_scale" << plot_font_scale << "ui_font_pt" << ui_font_pt << "}";
    return true;
}

void SimConfig::resolvePaths() {
    image_dir = firstExisting(data_dirs);
    alignment_config = firstExisting(alignment_configs);
}

std::string SimConfig::check() const {
    if (field_step_px <= 0 || 3840 % field_step_px || 2160 % field_step_px) return "field.step_px 는 3840 과 2160 의 공약수여야 함 (예 : 40, 80, 120)";
    if (field_window_px < 32) return "field.window_px 가 너무 작음";
    if (canvas_margin_x_px % field_step_px || canvas_margin_y_px % field_step_px) return "canvas.margin 은 field.step_px 의 배수여야 함";
    if (control_max_moves <= 0 || tol_y_um <= 0 || tol_t_deg <= 0) return "control 값이 올바르지 않음";
    if (robot_max_step_y_mm <= 0 || robot_max_step_t_deg <= 0) return "robot.max_step 값이 올바르지 않음";
    if (robot_repeat_y_um < 0 || robot_repeat_t_deg < 0 || robot_gain_error < 0) return "robot 오차 값은 0 이상이어야 함";
    if (plot_font_scale <= 0 || ui_font_pt < 6) return "display 값이 올바르지 않음 (plot_font_scale > 0, ui_font_pt ≥ 6)";
    if (estimator_type != "ksh" && estimator_type != "plugin") return "estimator.type 은 ksh 또는 plugin";
    if (zero_calibration < 0 || zero_calibration > 2) return "estimator.zero_calibration 은 0 (안 함), 1 (항상), 2 (자동)";
    return "";
}

} // namespace sim
