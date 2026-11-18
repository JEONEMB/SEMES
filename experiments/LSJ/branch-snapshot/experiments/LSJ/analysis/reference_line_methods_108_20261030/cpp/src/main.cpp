// line_methods: runs the registered line-extraction methods on the manifest's reference images and
// writes line measurements, intermediates, the 108 E x Q combinations and the method status.
//
// usage: line_methods.exe [--root DIR] [--out DIR] [--refs R01,R02] [--methods E01,Q12]
//   --root  analysis folder that contains configs/ (default: current directory)
//   --out   results folder (default: <root>/results)
#include "common.hpp"
#include "effective_settings.hpp"
#include "io_utils.hpp"

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>

#include <chrono>
#include <cstdio>
#include <iostream>
#include <set>
#include <sstream>

#ifdef _WIN32
#include <windows.h>
#endif

using namespace lm;

namespace {

std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> v;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) if (!item.empty()) v.push_back(item);
    return v;
}

// Config keys are a single id ("Q10") or a shared family ("E01/Q01").
const json& method_config(const json& methods, const std::string& id) {
    for (auto it = methods.begin(); it != methods.end(); ++it) {
        std::string key = it.key();
        size_t slash = key.find('/');
        if (key == id || (slash != std::string::npos && (key.substr(0, slash) == id || key.substr(slash + 1) == id))) return it.value();
    }
    throw std::runtime_error("no config for method " + id);
}

std::string q12_definition(const std::string& qr) {
    return u8"QR " + qr + u8" 위쪽 두 파인더(A_top_left, B_top_right) 외곽 사각형의 대각선 교점(중심)을 연결한 방향선. "
           u8"바깥 밝은 테두리 상단 경계(Q01~Q11)와 다른 특징이다.";
}

TargetSpec make_target(const json& j, const json& common) {
    TargetSpec t;
    t.id = j.at("id").get<std::string>();
    t.kind = j.at("kind").get<std::string>();
    t.feature_definition = j.at("feature_definition").get<std::string>();
    t.x0 = j.at("x_range")[0].get<int>(); t.x1 = j.at("x_range")[1].get<int>();
    t.y0 = j.at("band_y")[0].get<int>(); t.y1 = j.at("band_y")[1].get<int>();
    for (auto& e : j.at("exclude_x")) t.exclude_x.push_back({e[0].get<int>(), e[1].get<int>()});
    t.polarity = j.at("polarity").get<int>();
    t.margin = common.at("detector_margin_px").at(t.kind).get<int>();
    t.column_step = common.at("column_step_px").at(t.kind).get<int>();
    if (j.contains("roi_xyxy")) {
        auto r = j.at("roi_xyxy");
        t.roi = cv::Rect(r[0].get<int>(), r[1].get<int>(), r[2].get<int>() - r[0].get<int>(), r[3].get<int>() - r[1].get<int>());
    } else {
        t.roi = cv::Rect(t.x0, t.y0, t.x1 - t.x0 + 1, t.y1 - t.y0 + 1);
    }
    t.raw = j;
    return t;
}

json roi_json(const TargetSpec& t) {
    json j;
    j["x_range_inclusive"] = {t.x0, t.x1};
    j["band_y_inclusive"] = {t.y0, t.y1};
    json ex = json::array();
    for (auto& iv : t.exclude_x) ex.push_back({iv.a, iv.b});
    j["exclude_x_inclusive"] = ex;
    if (t.kind == "qr") j["roi_xyxy_exclusive_end"] = {t.roi.x, t.roi.y, t.roi.x + t.roi.width, t.roi.y + t.roi.height};
    j["detector_margin_px"] = t.margin;
    j["column_step_px"] = t.column_step;
    j["polarity"] = t.polarity;
    return j;
}

std::string csv_escape(const std::string& s) {
    std::string o = "\"";
    for (char c : s) { if (c == '"') o += "\"\""; else o += c; }
    return o + "\"";
}

std::string num_or_empty(const json& v) {
    if (v.is_null()) return "";
    std::ostringstream os;
    os.precision(17);
    os << v.get<double>();
    return os.str();
}

}  // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t** wargv) {
    SetConsoleOutputCP(CP_UTF8);
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) args.push_back(wide_to_utf8(wargv[i]));
#else
int main(int argc, char** argv) {
    std::vector<std::string> args(argv + 1, argv + argc);
#endif
    std::string root = ".", out;
    bool check_config = false;
    std::set<std::string> want_refs, want_methods;
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--check-config") check_config = true;
        else if (args[i] == "--root" && i + 1 < args.size()) root = args[++i];
        else if (args[i] == "--out" && i + 1 < args.size()) out = args[++i];
        else if (args[i] == "--refs" && i + 1 < args.size()) for (auto& s : split(args[++i])) want_refs.insert(s);
        else if (args[i] == "--methods" && i + 1 < args.size()) for (auto& s : split(args[++i])) want_methods.insert(s);
        else { std::cerr << "unknown argument: " << args[i] << "\n"; return 2; }
    }
    if (out.empty()) out = join_path(root, "results");
    try {
        cv::setNumThreads(1);
        auto t_start = std::chrono::steady_clock::now();
        const std::string cfg_dir = join_path(root, "configs");
        std::map<std::string, std::string> cfg_hash;
        auto load = [&](const std::string& name) {
            std::string path = join_path(cfg_dir, name);
            auto bytes = read_bytes(path);
            cfg_hash[name] = sha256_hex(bytes);
            return json::parse(std::string(bytes.begin(), bytes.end()));
        };
        json manifest = load("inputs_manifest.json");
        json targets_cfg = load("targets.json");
        json methods_cfg = load("methods.json");
        json sources_cfg = file_exists(join_path(cfg_dir, "method_sources.json")) ? load("method_sources.json") : json::object();
        const json& common = methods_cfg.at("common");

        std::vector<TargetSpec> targets;
        for (auto& j : targets_cfg.at("targets")) targets.push_back(make_target(j, common));

        std::vector<const MethodInfo*> methods;
        for (auto& m : registry())
            if (want_methods.empty() || want_methods.count(m.id)) methods.push_back(&m);
        if(methods.empty())throw std::invalid_argument("no selected method is registered");
        for(const auto& wanted:want_methods) {
            bool found=false;for(const auto* m:methods)if(m->id==wanted)found=true;
            if(!found)throw std::invalid_argument("unknown requested method: "+wanted);
        }
        // Resolve/validate before image reads or output directory creation.
        json settings_by_slot=json::object();
        cv::Size frame(manifest.at("expected_width").get<int>(),manifest.at("expected_height").get<int>());
        for(const auto* m:methods)for(const auto& t:targets)if(t.kind==(m->family=="E"?"external":"qr"))
            settings_by_slot[m->id+"_"+t.id]=effective_settings(m->id,method_config(methods_cfg.at("methods"),m->id).at(t.kind),common,t,frame);
        if(check_config){std::cout<<settings_by_slot.dump(2)<<std::endl;return 0;}
        make_dirs(out);

        json records = json::array();
        json inputs = json::array();
        std::map<std::string, std::string> status_of;  // measurement_id -> status
        std::map<std::string, std::map<std::string, int>> method_counts;  // method -> status -> count

        for (auto& ref : manifest.at("references")) {
            const std::string rid = ref.at("reference_id").get<std::string>();
            if (!want_refs.empty() && !want_refs.count(rid)) continue;
            const std::string path = ref.at("path").get<std::string>();
            json in = {{"reference_id", rid}, {"path", path}, {"expected_sha256", ref.at("sha256")}};
            std::string problem;
            cv::Mat gray;
            std::string digest;
            try {
                auto bytes = read_bytes(path);
                digest = sha256_hex(bytes);
                BmpHeader h = parse_bmp_header(bytes);
                in["sha256_before"] = digest;
                in["bmp_header"] = {{"valid", h.valid}, {"width", h.width}, {"height", h.height}, {"bits_per_pixel", h.bits_per_pixel}};
                if (digest != ref.at("sha256").get<std::string>()) problem = "sha256 mismatch";
                else if (!h.valid || h.width != manifest.at("expected_width").get<int>() || h.height != manifest.at("expected_height").get<int>() ||
                         h.bits_per_pixel != manifest.at("expected_bits_per_pixel").get<int>()) problem = "bmp header mismatch";
                else {
                    gray = decode_gray(bytes);
                    if (gray.empty() || gray.cols != h.width || gray.rows != h.height) problem = "decode failed or size mismatch";
                }
            } catch (const std::exception& e) {
                problem = std::string("read failed: ") + e.what();
            }
            in["status"] = problem.empty() ? "ok" : "mismatch";
            if (!problem.empty()) in["problem"] = problem;
            std::cout << rid << " input " << (problem.empty() ? "ok" : problem) << std::endl;

            for (const MethodInfo* m : methods) {
                const std::string kind = m->family == "E" ? "external" : "qr";
                const json& mcfg = method_config(methods_cfg.at("methods"), m->id);
                for (const TargetSpec& t : targets) {
                    if (t.kind != kind) continue;
                    const std::string mid = rid + "_" + m->id + "_" + t.id;
                    json params = mcfg.at(kind);
                    json hashed = {{"method", params}, {"common_fit", common.at("fit")}, {"segment_selection", common.at("segment_selection").at(kind)},
                                   {"roi", roi_json(t)}};
                    const json& settings=settings_by_slot.at(m->id+"_"+t.id);
                    std::string configured_dump=hashed.dump();
                    std::string configured_hash=sha256_hex(std::vector<uint8_t>(configured_dump.begin(),configured_dump.end()));
                    std::string dumped = settings.at("applied").dump();
                    std::string config_hash = sha256_hex(std::vector<uint8_t>(dumped.begin(), dumped.end()));
                    std::string rel_dir = "intermediates/" + rid + "/" + m->id;
                    Sink sink{join_path(out, rel_dir), rel_dir, t.id};
                    make_dirs(sink.abs_dir);
                    MethodResult res;
                    auto t0 = std::chrono::steady_clock::now();
                    if (!problem.empty()) {
                        res.fail("input_mismatch", problem);
                    } else {
                        MethodContext ctx{gray, t, params, common, uses_common_fit(m->id)?fit_params_for(common, kind):FitParams{}, sink};
                        try {
                            res = m->fn(ctx);
                        } catch (const std::exception& e) {
                            res = MethodResult();
                            res.fail("exception", e.what());
                        }
                    }
                    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                    json rec;
                    rec["measurement_id"] = mid;
                    rec["reference_id"] = rid;
                    rec["dataset"] = ref.at("dataset");
                    rec["input_path"] = path;
                    rec["input_sha256"] = digest;
                    rec["method_id"] = m->id;
                    rec["method_name"] = m->name_ko;
                    rec["implementation"] = m->implementation;
                    rec["target_id"] = t.id;
                    rec["target_kind"] = kind;
                    rec["feature_definition"] = m->id == "Q12" ? q12_definition(t.id.substr(3)) : t.feature_definition;
                    rec["feature_class"] = m->id == "Q12" ? "qr_finder_center_baseline" : (kind == "qr" ? "qr_outer_border_top_edge" : "external_structure_edge");
                    rec["status"] = res.status;
                    rec["failure_category"] = res.status == "ok" ? json(nullptr) : json(res.failure_category);
                    rec["failure_reason"] = res.status == "ok" ? json(nullptr) : json(res.failure_reason);
                    rec["roi"] = roi_json(t);
                    rec["parameters"] = params;
                    rec["parameters_note"] = "configured values; see settings.applied, settings.inactive and settings.reporting_only for actual scope";
                    rec["settings"] = settings;
                    rec["config_hash"] = config_hash;
                    rec["config_hash_scope"] = "settings.applied only; inactive, display and explanatory values excluded";
                    rec["configured_config_hash"] = configured_hash;
                    if (res.has_line) {
                        json lj = line_json(res.line);
                        rec["line"] = lj;
                        rec["slope_dy_dx"] = lj["slope_dy_dx"];
                        rec["intercept_y_px"] = lj["intercept_y_px"];
                        rec["image_angle_deg"] = lj["image_angle_deg"];
                        rec["line_endpoints_xy"] = res.has_endpoints ? json::array({json::array({res.ex1, res.ey1}), json::array({res.ex2, res.ey2})}) : json(nullptr);
                        rec["draw_endpoints_at_x_range"] = line_draw_endpoints(res.line, t.x0, t.x1);
                    } else {
                        rec["line"] = nullptr;
                        rec["slope_dy_dx"] = nullptr;
                        rec["intercept_y_px"] = nullptr;
                        rec["image_angle_deg"] = nullptr;
                        rec["line_endpoints_xy"] = nullptr;
                        rec["draw_endpoints_at_x_range"] = nullptr;
                    }
                    rec["n_candidates"] = res.n_candidates;
                    rec["n_selected"] = res.n_selected;
                    rec["support_type"] = res.support_type.empty() ? json(nullptr) : json(res.support_type);
                    if (res.has_rms) { rec["fit_rms_px"] = res.fit_rms; rec["fit_rms_null_reason"] = nullptr; }
                    else {
                        rec["fit_rms_px"] = nullptr;
                        rec["fit_rms_null_reason"] = !res.rms_null_reason.empty() ? res.rms_null_reason : std::string("no line");
                    }
                    rec["fit_rms_point_set"] = res.has_rms && res.diagnostics.contains("fit_rms_point_set")
                        ? res.diagnostics["fit_rms_point_set"] : json(nullptr);
                    rec["n_fit_used"] = res.diagnostics.contains("fit_used_points") ? res.diagnostics["fit_used_points"] : json(nullptr);
                    rec["fit_rms_note"] = "common fit RMS uses actual last-solve input points; final-distance-inlier RMS is separate in diagnostics; neither is measurement accuracy";
                    rec["diagnostics"] = res.diagnostics;
                    std::string inter_rel = rel_dir + "/" + t.id + ".json";
                    rec["intermediate_json"] = inter_rel;
                    rec["intermediate_images"] = res.images;
                    rec["runtime_ms"] = ms;
                    json inter = {{"measurement_id", mid}, {"intermediates", res.intermediates}, {"images", res.images}};
                    write_text(join_path(out, inter_rel), inter.dump(-1, ' ', false, json::error_handler_t::replace));
                    records.push_back(rec);
                    status_of[mid] = res.status;
                    method_counts[m->id][res.status]++;
                    std::printf("%s %s %.1f ms%s%s\n", mid.c_str(), res.status.c_str(), ms,
                                res.status == "ok" ? "" : " : ", res.status == "ok" ? "" : res.failure_reason.c_str());
                    std::fflush(stdout);
                }
            }
            if (!digest.empty()) {
                std::string after = sha256_hex(read_bytes(path));
                in["sha256_after"] = after;
                in["unchanged_after_run"] = after == digest;
            }
            inputs.push_back(in);
        }

        json meta;
        meta["tool"] = "line_methods (reference_line_methods_108_20261030)";
        meta["opencv_version"] = CV_VERSION;
#ifdef _MSC_VER
        meta["msvc_version"] = _MSC_VER;
        meta["msvc_full_version"] = _MSC_FULL_VER;
#endif
        meta["cplusplus"] = (long)__cplusplus;
        meta["opencv_threads"] = cv::getNumThreads();
        meta["config_sha256"] = cfg_hash;
        meta["settings_schema_version"] = 2;
        meta["point_record_schema_version"] = 2;
        meta["point_record_precision"] = "fit_points/line_input_points preserve finite calculation coordinates without decimal rounding; viewer coordinates are display-only; upstream precision is unchanged";
        meta["coordinate_convention"] = targets_cfg.at("coordinate_convention");
        meta["angle_convention"] = "image_angle_deg = atan2(dy, dx) with dx > 0 (x-increasing direction), range [-90, 90). Image line angle only; NOT robot theta.";
        meta["inputs"] = inputs;
        meta["refs_filter"] = std::vector<std::string>(want_refs.begin(), want_refs.end());
        meta["methods_filter"] = std::vector<std::string>(want_methods.begin(), want_methods.end());
        meta["slot_count"] = (int)records.size();
        meta["total_runtime_ms"] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();
        json lm_json = {{"meta", meta}, {"records", records}};
        write_text(join_path(out, "line_measurements.json"), lm_json.dump(1, ' ', false, json::error_handler_t::replace));

        // CSV (one row per main-line slot).
        std::ostringstream csv;
        csv << "measurement_id,reference_id,method_id,target_id,feature_class,status,failure_category,failure_reason,slope_dy_dx,intercept_y_px,image_angle_deg,"
               "point_x,point_y,dir_x,dir_y,gen_a,gen_b,gen_c,end1_x,end1_y,end2_x,end2_y,n_candidates,n_selected,support_type,fit_rms_px,fit_rms_null_reason,runtime_ms,config_hash,input_sha256,fit_rms_point_set,n_fit_used\n";
        for (auto& r : records) {
            const json& l = r["line"];
            auto g = [&](const json& v) { return num_or_empty(v); };
            csv << r["measurement_id"].get<std::string>() << "," << r["reference_id"].get<std::string>() << "," << r["method_id"].get<std::string>() << ","
                << r["target_id"].get<std::string>() << "," << r["feature_class"].get<std::string>() << "," << r["status"].get<std::string>() << ","
                << (r["failure_category"].is_null() ? "" : r["failure_category"].get<std::string>()) << ","
                << csv_escape(r["failure_reason"].is_null() ? "" : r["failure_reason"].get<std::string>()) << ","
                << g(r["slope_dy_dx"]) << "," << g(r["intercept_y_px"]) << "," << g(r["image_angle_deg"]) << ",";
            if (l.is_null()) csv << ",,,,,,,";
            else csv << g(l["point_xy"][0]) << "," << g(l["point_xy"][1]) << "," << g(l["direction_xy"][0]) << "," << g(l["direction_xy"][1]) << ","
                     << g(l["general_form_abc"][0]) << "," << g(l["general_form_abc"][1]) << "," << g(l["general_form_abc"][2]) << ",";
            const json& e = r["line_endpoints_xy"];
            if (e.is_null()) csv << ",,,,";
            else csv << g(e[0][0]) << "," << g(e[0][1]) << "," << g(e[1][0]) << "," << g(e[1][1]) << ",";
            csv << r["n_candidates"].get<int>() << "," << r["n_selected"].get<int>() << ","
                << csv_escape(r["support_type"].is_null() ? "" : r["support_type"].get<std::string>()) << "," << g(r["fit_rms_px"]) << ","
                << csv_escape(r["fit_rms_null_reason"].is_null() ? "" : r["fit_rms_null_reason"].get<std::string>()) << ","
                << g(r["runtime_ms"]) << "," << r["config_hash"].get<std::string>() << "," << r["input_sha256"].get<std::string>() << ","
                << (r["fit_rms_point_set"].is_null() ? "" : r["fit_rms_point_set"].get<std::string>()) << "," << g(r["n_fit_used"]) << "\n";
        }
        write_text(join_path(out, "line_measurements.csv"), "\xEF\xBB\xBF" + csv.str());

        // Combinations: every E x Q pair per reference, referencing the single-method measurements.
        json combos = json::array();
        std::vector<std::string> ext_t = {"upper_structure", "lower_structure"}, qr_t = {"qr_L", "qr_C", "qr_R"};
        int n_complete = 0, n_partial = 0, n_failed = 0;
        for (auto& in : inputs) {
            std::string rid = in["reference_id"].get<std::string>();
            for (const MethodInfo* e : methods) {
                if (e->family != "E") continue;
                for (const MethodInfo* q : methods) {
                    if (q->family != "Q") continue;
                    json refs = json::object(), failed = json::array();
                    int ok = 0, total = 0;
                    for (auto& tt : ext_t) {
                        std::string id = rid + "_" + e->id + "_" + tt;
                        refs[tt] = id; ++total;
                        if (status_of[id] == "ok") ++ok; else failed.push_back(id);
                    }
                    for (auto& tt : qr_t) {
                        std::string id = rid + "_" + q->id + "_" + tt;
                        refs[tt] = id; ++total;
                        if (status_of[id] == "ok") ++ok; else failed.push_back(id);
                    }
                    std::string st = ok == total ? "complete" : ok == 0 ? "failed" : "partial";
                    (st == "complete" ? n_complete : st == "partial" ? n_partial : n_failed)++;
                    combos.push_back({{"combination_id", rid + "_" + e->id + "_" + q->id}, {"reference_id", rid}, {"external_method", e->id},
                                      {"qr_method", q->id}, {"status", st}, {"lines_ok", ok}, {"lines_total", total},
                                      {"measurement_refs", refs}, {"failed_measurements", failed}});
                }
            }
        }
        json cj = {{"meta", {{"meaning", u8"조합 = 같은 이미지에서 외부 방법 1개의 선 2개와 QR 방법 1개의 선 3개를 함께 열람하는 항목. 새 측정이나 각도 평균이 아님."},
                             {"count", (int)combos.size()}, {"complete", n_complete}, {"partial", n_partial}, {"failed", n_failed}}},
                   {"combinations", combos}};
        write_text(join_path(out, "combinations.json"), cj.dump(1));

        // Method status.
        json ms = json::array();
        for (auto& m : registry()) {
            json s;
            s["method_id"] = m.id;
            s["name"] = m.name_ko;
            s["family"] = m.family == "E" ? "external" : "qr";
            s["implementation"] = m.implementation;
            s["compiled_in_this_binary"] = true;
            s["ran_in_this_run"] = (bool)method_counts.count(m.id);
            json counts = json::object();
            if (method_counts.count(m.id)) for (auto& kv : method_counts[m.id]) counts[kv.first] = kv.second;
            s["slot_status_counts"] = counts;
            if (sources_cfg.contains("methods") && sources_cfg["methods"].contains(m.id)) s["source"] = sources_cfg["methods"][m.id];
            ms.push_back(s);
        }
        write_text(join_path(out, "method_status.json"), json({{"meta", meta}, {"methods", ms}}).dump(1));
        std::printf("slots=%d combinations=%d complete=%d partial=%d failed=%d\n", (int)records.size(), (int)combos.size(), n_complete, n_partial, n_failed);
        bool all_unchanged = true;
        for (auto& in : inputs) if (!in.value("unchanged_after_run", false)) all_unchanged = false;
        std::printf("inputs_unchanged=%s\n", all_unchanged ? "true" : "false");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "fatal: " << e.what() << std::endl;
        return 1;
    }
}
