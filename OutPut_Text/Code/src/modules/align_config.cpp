// align_config.cpp — 설정 파일 읽기·쓰기 구현
#include "ksh/align_config.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <regex>
#include <sstream>

namespace ksh {

namespace fs = std::filesystem;

namespace {

std::string str(const cv::FileNode &n, const std::string &fallback = "") {
    if (n.empty()) return fallback;
    std::string s;
    n >> s;
    return s;
}
double num(const cv::FileNode &n, double fallback) { return n.empty() ? fallback : double(n); }
int integer(const cv::FileNode &n, int fallback) { return n.empty() ? fallback : int(n); }

Tolerance read_tol(const cv::FileNode &n, Tolerance fallback) {
    if (n.empty()) return fallback;
    return Tolerance{num(n["y_um"], fallback.y_um), num(n["t_deg"], fallback.t_deg)};
}

// ROI 목록 : - { name, cx_px, cy_px, hw_px }
bool read_boxes(const cv::FileNode &n, std::vector<BoxSpec> &out, std::string &error, const char *what) {
    out.clear();
    for (auto it = n.begin(); it != n.end(); ++it) {
        BoxSpec b;
        b.name = str((*it)["name"]);
        b.cx_px = integer((*it)["cx_px"], -1);
        b.cy_px = integer((*it)["cy_px"], -1);
        b.hw_px = integer((*it)["hw_px"], -1);
        if (b.name.empty() || b.cx_px < 0 || b.cy_px < 0 || b.hw_px <= 0) { error = std::string(what) + " 항목 형식 오류 (name, cx_px, cy_px, hw_px)"; return false; }
        out.push_back(b);
    }
    return true;
}

std::vector<TrainingSet> read_sets(const cv::FileNode &n, const std::string &cfg_path) {
    std::vector<TrainingSet> v;
    for (auto it = n.begin(); it != n.end(); ++it) {
        TrainingSet s;
        s.name = str((*it)["name"]);
        s.image_dir = resolve_path(cfg_path, str((*it)["image_dir"]));
        s.feature_csv = resolve_path(cfg_path, str((*it)["feature_csv"]));
        s.zero_ref = str((*it)["zero_ref"]);
        const cv::FileNode ex = (*it)["exclude"];
        for (auto e = ex.begin(); e != ex.end(); ++e) s.exclude.push_back(str(*e));
        v.push_back(s);
    }
    return v;
}

// 파일을 줄 단위로 바꿔 쓰기 (원본 .bak)
bool rewrite_lines(const std::string &path, const std::vector<std::string> &lines, std::string &error, bool backup) {
    std::error_code ec;
    if (backup) fs::copy_file(fs::u8path(path), fs::u8path(path + ".bak"), fs::copy_options::overwrite_existing, ec);
    std::ofstream o(fs::u8path(path));
    if (!o) { error = "설정 파일에 쓸 수 없음 : " + path; return false; }
    for (const auto &x : lines) o << x << "\n";
    return true;
}
bool read_lines(const std::string &path, std::vector<std::string> &lines, std::string &error) {
    std::ifstream in(fs::u8path(path));
    if (!in) { error = "설정 파일을 열 수 없음 : " + path; return false; }
    std::string l;
    while (std::getline(in, l)) { if (!l.empty() && l.back() == '\r') l.pop_back(); lines.push_back(l); }
    return true;
}

} // namespace

std::string resolve_path(const std::string &config_path, const std::string &path) {
    if (path.empty()) return path;
    const fs::path p = fs::u8path(path);
    std::error_code ec;
    if (p.is_absolute() || fs::exists(p, ec)) return path;
    const fs::path base = fs::u8path(config_path).parent_path();
    if (base.empty()) return path;
    const fs::path q = base / p;
    return fs::exists(q, ec) ? q.u8string() : path;
}

bool load_align_config(const std::string &path, AlignConfig &cfg, std::string &error, bool require_model) {
    cv::FileStorage f;
    try {
        if (!f.open(path, cv::FileStorage::READ)) { error = "설정 파일을 열 수 없음 : " + path; return false; }
    } catch (const cv::Exception &) {
        error = "설정 파일 형식 오류 : " + path;
        return false;
    }
    AlignConfig c;
    c.path = path;
    c.image_size_px = cv::Size(integer(f["image_width_px"], 0), integer(f["image_height_px"], 0));

    // ① QR
    QrParams &qp = c.features.qr;
    if (!read_boxes(f["qr_rois"], qp.boxes, error, "qr_rois")) return false;
    if (qp.boxes.size() < 2) { error = "qr_rois 는 2개 이상 (교보재 3개) : " + path; return false; }
    const cv::FileNode q = f["qr"];
    qp.search_px = integer(q["search_px"], qp.search_px);
    qp.coarse_range_px = integer(q["coarse_range_px"], qp.coarse_range_px);
    qp.coarse_max_spread_px = num(q["coarse_max_spread_px"], qp.coarse_max_spread_px);
    qp.close_kernel_px = integer(q["close_kernel_px"], qp.close_kernel_px);
    qp.gap_dark_frac = num(q["gap_dark_frac"], qp.gap_dark_frac);
    qp.gap_min_px = integer(q["gap_min_px"], qp.gap_min_px);
    qp.gap_max_px = integer(q["gap_max_px"], qp.gap_max_px);
    qp.edge_inset_px = integer(q["edge_inset_px"], qp.edge_inset_px);
    qp.edge_half_window_px = integer(q["edge_half_window_px"], qp.edge_half_window_px);
    qp.blur_sigma_px = num(q["blur_sigma_px"], qp.blur_sigma_px);
    qp.edge_gauss_peak = str(q["edge_subpixel"], qp.edge_gauss_peak ? "gauss" : "parabola") != "parabola";

    // ② 배경 홀
    HoleParams &hp = c.features.holes;
    if (!read_boxes(f["background_rois"], hp.boxes, error, "background_rois")) return false;
    if (hp.boxes.empty()) { error = "background_rois 가 비어 있음 : " + path; return false; }
    const cv::FileNode b = f["background"];
    hp.search_px = integer(b["search_px"], hp.search_px);
    hp.window_half_px = integer(b["window_half_px"], hp.window_half_px);
    hp.blur_sigma_px = num(b["blur_sigma_px"], hp.blur_sigma_px);
    hp.locate_sigma_px = num(b["locate_sigma_px"], hp.locate_sigma_px);
    hp.locate_radius_px = integer(b["locate_radius_px"], hp.locate_radius_px);
    hp.min_contrast = num(b["min_contrast"], hp.min_contrast);
    hp.max_center_gap_px = num(b["max_center_gap_px"], hp.max_center_gap_px);
    hp.robust_iter = integer(b["robust_iter"], hp.robust_iter);
    hp.max_line_dist_px = num(b["max_line_dist_px"], hp.max_line_dist_px);
    hp.max_spread_px = num(b["max_spread_px"], hp.max_spread_px);

    // ④ 보정식
    const std::string model_file = str(f["model_file"]);
    if (!model_file.empty()) {
        c.model_path = resolve_path(path, model_file);
        if (!load_pose_model(c.model_path, c.model) && require_model) { error = "보정식 파일을 읽을 수 없음 : " + c.model_path; return false; }
    } else if (require_model) {
        error = "model_file 이 없음 : " + path;
        return false;
    }
    const cv::FileNode z = f["zero"];
    c.zero.y_mm = num(z["y_mm"], 0.0);
    c.zero.t_deg = num(z["t_deg"], 0.0);
    const cv::FileNode ce = f["center_to_encoder"];
    c.center_to_encoder.y_mm = num(ce["y_mm"], 0.0);
    c.center_to_encoder.t_deg = num(ce["t_deg"], 0.0);
    c.tolerance = read_tol(f["tolerance"], Tolerance{45.0, 0.005});
    c.spec_tolerance = read_tol(f["spec_tolerance"], Tolerance{90.0, 0.01});

    const cv::FileNode tr = f["training"];
    c.training.train = read_sets(tr["train"], path);
    c.training.test = read_sets(tr["test"], path);
    c.training.theta_order_s = integer(tr["theta_order_s"], c.training.theta_order_s);
    c.training.theta_order_d = integer(tr["theta_order_d"], c.training.theta_order_d);
    c.training.balance_sets = integer(tr["balance_sets"], 1) != 0;
    c.training.threads = integer(tr["threads"], c.training.threads);
    c.training.output = str(tr["output"]);
    c.training.result_dir = str(tr["result_dir"], c.training.result_dir);
    const cv::FileNode de = f["demo"];
    c.demo.image = str(de["image"]);
    c.demo.display_width_px = integer(de["display_width_px"], c.demo.display_width_px);

    cfg = c;
    return true;
}

bool save_zero_offset(const std::string &path, const ZeroOffset &zero, std::string &error, bool backup) {
    std::vector<std::string> lines, out;
    if (!read_lines(path, lines, error)) return false;
    std::ostringstream blk;
    blk << std::setprecision(10) << "zero:\n   y_mm: " << zero.y_mm << "\n   t_deg: " << zero.t_deg;
    bool done = false;
    for (size_t i = 0; i < lines.size(); ++i) {
        if (!done && lines[i].rfind("zero:", 0) == 0) {
            out.push_back(blk.str());
            while (i + 1 < lines.size() && !lines[i + 1].empty() && (lines[i + 1][0] == ' ' || lines[i + 1][0] == '\t')) ++i;
            done = true;
            continue;
        }
        out.push_back(lines[i]);
    }
    if (!done) out.push_back(blk.str());
    return rewrite_lines(path, out, error, backup);
}

bool save_roi_centers(const std::string &path, const std::vector<BoxSpec> &boxes, std::string &error, bool backup) {
    std::vector<std::string> lines;
    if (!read_lines(path, lines, error)) return false;
    int changed = 0;
    for (const auto &b : boxes) {
        const std::regex re_name("name:\\s*" + b.name + "\\s*,");
        for (auto &l : lines) {
            if (!std::regex_search(l, re_name)) continue;
            l = std::regex_replace(l, std::regex(R"(cx_px:\s*-?\d+)"), "cx_px: " + std::to_string(b.cx_px));
            l = std::regex_replace(l, std::regex(R"(cy_px:\s*-?\d+)"), "cy_px: " + std::to_string(b.cy_px));
            ++changed;
            break;
        }
    }
    if (changed != int(boxes.size())) { error = "설정 파일에서 ROI 줄을 모두 찾지 못함 (" + std::to_string(changed) + "/" + std::to_string(boxes.size()) + ")"; return false; }
    return rewrite_lines(path, lines, error, backup);
}

} // namespace ksh
