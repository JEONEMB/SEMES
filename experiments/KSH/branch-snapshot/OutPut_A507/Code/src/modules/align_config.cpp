// align_config.cpp — 설정 파일 읽기 구현
#include "ksh/align_config.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace ksh {

namespace fs = std::filesystem;

namespace {

// 항목 읽기 도우미 : 항목이 없으면 fallback (기본값)
std::string str(const cv::FileNode &n, const std::string &fallback = "") {
    if (n.empty()) return fallback;
    std::string s;
    n >> s;
    return s;
}
double num(const cv::FileNode &n, double fallback) { return n.empty() ? fallback : double(n); }
int integer(const cv::FileNode &n, int fallback) { return n.empty() ? fallback : int(n); }

// 허용 범위 { y_um, t_deg } 읽기
Tolerance read_tol(const cv::FileNode &n, Tolerance fallback) {
    if (n.empty()) return fallback;
    return Tolerance{num(n["y_um"], fallback.y_um), num(n["t_deg"], fallback.t_deg)};
}

// 학습 세트 목록 : - { name, image_dir [, feature_csv] }
std::vector<TrainingSet> read_sets(const cv::FileNode &n, const std::string &cfg_path) {
    std::vector<TrainingSet> v;
    for (auto it = n.begin(); it != n.end(); ++it) {
        TrainingSet s;
        s.name = str((*it)["name"]);
        s.image_dir = resolve_path(cfg_path, str((*it)["image_dir"]));
        s.feature_csv = resolve_path(cfg_path, str((*it)["feature_csv"]));
        v.push_back(s);
    }
    return v;
}

} // namespace

// 상대 경로 해석 : ① 절대 경로이거나 실행 폴더 기준으로 있으면 그대로 ② 아니면 설정 파일 폴더 기준 ③ 둘 다 없으면 원래 문자열
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

// 설정 파일(config/alignment.yml)과 보정식(model_file) 읽기
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
    const cv::FileNode q = f["qr"];
    if (q.empty()) { error = "설정 항목 누락 (qr) : " + path; return false; }
    QrParams &qp = c.features.qr;
    const cv::FileNode sr = q["search_roi"];
    if (!sr.empty())
        qp.search_roi = cv::Rect(integer(sr["x_px"], 0), integer(sr["y_px"], 0), integer(sr["width_px"], 0), integer(sr["height_px"], 0));
    qp.strip_dark_thr = num(q["strip_dark_thr"], qp.strip_dark_thr);
    qp.strip_min_w_px = num(q["strip_min_width_px"], qp.strip_min_w_px);
    qp.strip_aspect_min = num(q["strip_aspect_min"], qp.strip_aspect_min);
    qp.strip_aspect_max = num(q["strip_aspect_max"], qp.strip_aspect_max);
    qp.thr_frac = num(q["square_thr_frac"], qp.thr_frac);
    qp.proj_smooth_px = integer(q["projection_smooth_px"], qp.proj_smooth_px);
    qp.gap_merge_px = integer(q["gap_merge_px"], qp.gap_merge_px);
    qp.size_min_px = num(q["square_size_min_px"], qp.size_min_px);
    qp.size_max_px = num(q["square_size_max_px"], qp.size_max_px);
    qp.edge_margin = num(q["edge_margin"], qp.edge_margin);
    qp.edge_search_px = num(q["edge_search_px"], qp.edge_search_px);
    qp.edge_plateau_px = num(q["edge_plateau_px"], qp.edge_plateau_px);
    qp.edge_segments = integer(q["edge_segments"], qp.edge_segments);

    // ② 나사
    const cv::FileNode s = f["screws"];
    if (s.empty()) { error = "설정 항목 누락 (screws) : " + path; return false; }
    ScrewParams &sp = c.features.screws;
    const cv::FileNode rois = s["rois"];
    for (auto it = rois.begin(); it != rois.end(); ++it) {
        ScrewSpec spec;
        spec.name = str((*it)["name"]);
        spec.ref_center_px = cv::Point2d(num((*it)["cx_px"], 0), num((*it)["cy_px"], 0));
        sp.screws.push_back(spec);
    }
    if (sp.screws.empty()) { error = "screws.rois 가 비어 있음 : " + path; return false; }
    sp.qr_ref_px = cv::Point2d(num(s["qr_ref_x_px"], sp.qr_ref_px.x), num(s["qr_ref_y_px"], sp.qr_ref_px.y));
    sp.follow_x = num(s["follow_x"], sp.follow_x);
    sp.follow_y = num(s["follow_y"], sp.follow_y);
    sp.roi_half_width_px = integer(s["roi_half_width_px"], sp.roi_half_width_px);
    sp.roi_half_height_px = integer(s["roi_half_height_px"], sp.roi_half_height_px);
    sp.blur_sigma_px = num(s["blur_sigma_px"], sp.blur_sigma_px);
    sp.thr_frac = num(s["rim_thr_frac"], sp.thr_frac);
    sp.rmax_px = num(s["radius_max_px"], sp.rmax_px);
    sp.rays = integer(s["rays"], sp.rays);
    sp.min_contrast = num(s["min_contrast"], sp.min_contrast);
    if (sp.roi_half_width_px <= 40 || sp.roi_half_height_px <= 40) { error = "screws.roi_half_*_px 는 40 보다 커야 함 : " + path; return false; }

    // ④ 모델 : model_file 이 있으면 그 파일, 없으면 이 파일의 model:
    const std::string model_file = str(f["model_file"]);
    if (!model_file.empty()) {
        c.model_path = resolve_path(path, model_file);
        if (!load_pose_model(c.model_path, c.model) && require_model) { error = "모델 파일을 읽을 수 없음 : " + c.model_path; return false; }
    } else if (!read_pose_model(f["model"], c.model) && require_model) {
        error = "model_file 도 model: 도 없음 : " + path;
        return false;
    }
    const cv::FileNode z = f["zero"];
    c.zero.y_mm = num(z["y_mm"], 0.0);
    c.zero.t_deg = num(z["t_deg"], 0.0);

    c.tolerance = read_tol(f["tolerance"], Tolerance{45.0, 0.045});
    c.spec_tolerance = read_tol(f["spec_tolerance"], Tolerance{90.0, 0.09});

    // 학습·예제
    const cv::FileNode tr = f["training"];
    c.training.train = read_sets(tr["train"], path);
    c.training.test = read_sets(tr["test"], path);
    c.training.theta_order_s = integer(tr["theta_order_s"], c.training.theta_order_s);
    c.training.theta_order_d = integer(tr["theta_order_d"], c.training.theta_order_d);
    c.training.cv_blocks = integer(tr["cv_blocks"], c.training.cv_blocks);
    c.training.threads = integer(tr["threads"], c.training.threads);
    c.training.output = str(tr["output"]);
    c.training.result_dir = str(tr["result_dir"], c.training.result_dir);
    const cv::FileNode de = f["demo"];
    c.demo.image = str(de["image"]);
    c.demo.display_width_px = integer(de["display_width_px"], c.demo.display_width_px);

    cfg = c;
    return true;
}

// zero: 블록 다시 쓰기 — 'zero:' 줄과 그 아래 들여쓴 줄을 새 값으로 바꿈 (없으면 끝에 추가)
bool save_zero_offset(const std::string &path, const ZeroOffset &zero, std::string &error) {
    std::ifstream in(fs::u8path(path));
    if (!in) { error = "설정 파일을 열 수 없음 : " + path; return false; }
    std::vector<std::string> lines;
    std::string l;
    while (std::getline(in, l)) lines.push_back(l);
    in.close();
    std::ostringstream blk;
    blk << std::setprecision(8) << "zero:\n   y_mm: " << zero.y_mm << "\n   t_deg: " << zero.t_deg;
    std::vector<std::string> out;
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
    std::error_code ec;
    fs::copy_file(fs::u8path(path), fs::u8path(path + ".bak"), fs::copy_options::overwrite_existing, ec);
    std::ofstream o(fs::u8path(path));
    if (!o) { error = "설정 파일에 쓸 수 없음 : " + path; return false; }
    for (const auto &x : out) o << x << "\n";
    return true;
}

} // namespace ksh
