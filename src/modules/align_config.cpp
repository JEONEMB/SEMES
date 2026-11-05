// align_config.cpp — 설정 파일 읽기 구현
#include "ksh/align_config.hpp"

#include <filesystem>

namespace ksh {

namespace fs = std::filesystem;

namespace {

// ROI 목록 읽기 : - { name, cx_px, cy_px, hw_px [, hh_px] }
std::vector<RoiSpec> read_rois(const cv::FileNode &n) {
    std::vector<RoiSpec> v;
    for (auto it = n.begin(); it != n.end(); ++it) {
        RoiSpec r;
        (*it)["name"] >> r.name;
        r.center_px = cv::Point(int((*it)["cx_px"]), int((*it)["cy_px"]));
        r.half_width_px = int((*it)["hw_px"]);
        if (!(*it)["hh_px"].empty()) r.half_height_px = int((*it)["hh_px"]);   // 선택 : 직사각형 ROI 세로 반폭
        v.push_back(r);
    }
    return v;
}

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
// 인자 : 설정 파일 경로, 결과를 담을 설정 구조체, 실패 이유를 담을 문자열, 보정식이 꼭 있어야 하는지 (학습 도구는 false)
// 반환 : 성공 true. 실패하면 false 와 error 에 이유 (cfg 는 바꾸지 않음)
bool load_align_config(const std::string &path, AlignConfig &cfg, std::string &error, bool require_model) {

    cv::FileStorage f;      // openCV의 파일 저장 시스템
    
    // 시도
    try {
        // 만약 정상적으로 파일을 열 수 없다면, 에러메시지를 "설정 파일을 열 수 없음 : {경로}"로 설정하고,
        // 이미지를 못불러왔다고 false로 반환
        if (!f.open(path, cv::FileStorage::READ)) { error = "설정 파일을 열 수 없음 : " + path; return false; }
    }
    // openCV의 예외사항이 발생하면 error 메시지를 "설정 파일 형식 오류 : {경로}"로 설정하고
    // 이미지를 못 불러왔다고 false로 반환
    catch (const cv::Exception &) {
        error = "설정 파일 형식 오류 : " + path;
        return false;
    }

    // ========== 정상적으로 설정 파일(yml)을 잘 불러왔다면 이미지 분석 파라미터 설정 ==========
    AlignConfig c;      // 이미지 분석 구조체 변수 설정

    // yml 파일에서 경로, 이미지 크기등을 가져 옮.
    c.path = path;
    c.reference_path = str(f["reference_path"]);    // 기준 사진 경로. 없으면 빈 문자열
    c.image_size_px = cv::Size(integer(f["image_width_px"], 0), integer(f["image_height_px"], 0));  // 이미지 가로x세로, 지정된 값이 없으면 0;

    // ①②③ 측정
    FeatureParams &fp = c.features;
    // 대략적인 중심 맞추기
    const cv::FileNode co = f["coarse"];
    // 창 중심
    fp.coarse.center_px = cv::Point(integer(co["center_x_px"], 0), integer(co["center_y_px"], 0));
    // 윈도우 사이즈
    fp.coarse.window_px = cv::Size(integer(co["width_px"], 0), integer(co["height_px"], 0));
    // 봉우리 세기 하한 (없으면 0 = 검사 안 함)
    fp.coarse.min_response = num(co["min_response"], 0.0);
    // 배경 RoI 찾을 때 사용하는 비율
    fp.wide_window_scale = integer(f["wide_window_scale"], 0);
    // QR 한계점
    fp.qr_max_dev_px = num(f["qr_max_dev_px"], 0.0);
    // 배경 RoI 넓은 창 위상 상관 봉우리 세기 하한 (없으면 0 = 검사 안 함)
    fp.wide_min_response = num(f["wide_min_response"], 0.0);


    // 상세한 부분 맞추기
    const cv::FileNode fi = f["fine"];
    // 상세 부분이 있다면 값을 채워 넣음
    if (!fi.empty()) {
        // 방법과 커널을 설정
        const std::string method = str(fi["method"], "icgn"), kernel = str(fi["kernel"], "omoms3");
        
        // 정합 방법은 IC-GN 하나 (이전 비교용 opencv_ecc 는 제거)
        if (method != "icgn") { error = "fine.method 은 icgn 만 지원 : " + method; return false; }
        
        // 커널을 제대로 파악할 수 없다면 에러메시지 설정 및 false 반환
        if (!parse_kernel(kernel, fp.fine.kernel)) { error = "fine.kernel 을 알 수 없음 : " + kernel; return false; }
        
        // 커널 부분에 값을 대입
        fp.fine.gauss_sigma_px = num(fi["gauss_sigma_px"], fp.fine.gauss_sigma_px);
        fp.fine.max_iter = integer(fi["max_iter"], fp.fine.max_iter);
        fp.fine.eps_px = num(fi["eps_px"], fp.fine.eps_px);
        fp.fine.search_margin_px = integer(fi["search_margin_px"], fp.fine.search_margin_px);
        fp.fine.min_zncc = num(fi["min_zncc"], fp.fine.min_zncc);   // 정합 상관 하한 (없으면 0 = 검사 안 함)
    }   // fine 항목이 없으면 FineParams 기본값 (IC-GN, O-MOMS3)
    fp.qr_rois = read_rois(f["qr_rois"]);
    fp.background_rois = read_rois(f["background_rois"]);
    if (fp.qr_rois.empty() || fp.background_rois.empty() || fp.coarse.window_px.area() == 0 || fp.wide_window_scale <= 0) {
        error = "설정 항목 누락 (coarse, qr_rois, background_rois, wide_window_scale) : " + path;
        return false;
    }
    // ROI 반폭이 0 이하이면 영역이 비어 정합할 수 없음 → 설정 오류로 바로 알림
    for (const auto *list : {&fp.qr_rois, &fp.background_rois})
        for (const auto &r : *list)
            if (r.half_width_px <= 0 || r.half_height_px < 0) { error = "ROI 반폭(hw_px)은 1 이상이어야 함 : " + r.name; return false; }

    // 기준 사진 교체 시 ROI 재배치 (relocation 항목이 없으면 재배치하지 않음)
    const cv::FileNode rl = f["relocation"];
    c.relocation.enable = !rl.empty() && integer(rl["enable"], 1) != 0;
    c.relocation.template_path = resolve_path(path, str(rl["template_path"]));
    c.relocation.downscale = integer(rl["downscale"], c.relocation.downscale);
    c.relocation.min_response = num(rl["min_response"], c.relocation.min_response);
    c.relocation.max_shift_px = num(rl["max_shift_px"], c.relocation.max_shift_px);
    c.relocation.max_spread_px = num(rl["max_spread_px"], c.relocation.max_spread_px);
    if (c.relocation.enable && c.relocation.template_path.empty()) { error = "relocation.template_path 가 비어 있음 : " + path; return false; }

    // ④ 모델 : model_file 이 있으면 그 파일, 없으면 이 파일의 model:
    const std::string model_file = str(f["model_file"]);
    if (!model_file.empty()) {
        c.model_path = resolve_path(path, model_file);
        if (!load_pose_model(c.model_path, c.model) && require_model) { error = "모델 파일을 읽을 수 없음 : " + c.model_path; return false; }
    } else if (!read_pose_model(f["model"], c.model) && require_model) {
        error = "model_file 도 model: 도 없음 : " + path;
        return false;
    }

    // 허용 범위 (이전 형식 tolerance_y_um / tolerance_t_deg 도 허용)
    Tolerance old_tol{num(f["tolerance_y_um"], 45.0), num(f["tolerance_t_deg"], 0.005)};
    c.tolerance = read_tol(f["tolerance"], old_tol);
    c.spec_tolerance = read_tol(f["spec_tolerance"], Tolerance{90.0, 0.01});

    // 학습·평가·예제
    const cv::FileNode tr = f["training"];
    c.training.image_dir = str(tr["image_dir"]);
    c.training.reference = str(tr["reference"], "center");
    c.training.set = integer(tr["set"], c.training.set);
    c.training.y_grid_mm = num(tr["y_grid_mm"], c.training.y_grid_mm);
    c.training.theta_order_s = integer(tr["theta_order_s"], c.training.theta_order_s);
    c.training.theta_order_d = integer(tr["theta_order_d"], c.training.theta_order_d);
    c.training.output = str(tr["output"]);
    const cv::FileNode ev = f["evaluation"];
    c.evaluation.image_dir = str(ev["image_dir"], str(f["eval_dir"]));
    c.evaluation.reference_set1 = str(ev["reference_set1"], str(f["eval_reference_set1"], "center"));
    c.evaluation.reference_set2 = str(ev["reference_set2"], str(f["eval_reference_set2"], "center1"));
    c.evaluation.result_dir = str(ev["result_dir"], c.evaluation.result_dir);
    c.evaluation.near_target_t_deg = num(ev["near_target_t_deg"], c.evaluation.near_target_t_deg);
    c.evaluation.cv_near_y_mm = num(ev["cv_near_y_mm"], c.evaluation.cv_near_y_mm);
    c.evaluation.cv_blocks_y = integer(ev["cv_blocks_y"], c.evaluation.cv_blocks_y);
    c.evaluation.cv_blocks_t = integer(ev["cv_blocks_t"], c.evaluation.cv_blocks_t);
    c.evaluation.score_cap = num(ev["score_cap"], c.evaluation.score_cap);
    const cv::FileNode de = f["demo"];
    c.demo.image = str(de["image"], str(f["demo_image"]));
    c.demo.reference = str(de["reference"], str(f["demo_reference"]));
    c.demo.display_width_px = integer(de["display_width_px"], integer(f["display_width_px"], c.demo.display_width_px));


    // 이미지 변수 설정을 마쳤으면 이를 cfg 원래 인자에 넘겨주고 true로 반환
    cfg = c;
    return true;
}

} // namespace ksh
