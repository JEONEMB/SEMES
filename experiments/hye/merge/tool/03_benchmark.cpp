// Scores joint/control on every photo under one folder and writes a NEW result folder.
// Labels: nearest _metadata/manifest.csv (from 01_prepare_clean), else Y+-n / T+-n / center in the file name.
// Output: predictions.csv (one row per photo and module) and summary.json. Read by apps/web (평가 탭).
#include "alignment.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <regex>
#include <sstream>
namespace fs = std::filesystem;

namespace {
const double TOLERANCE_Y = 0.045;
const double TOLERANCE_T = 0.01;
struct Label { double y, t; };
struct Row { std::string path, condition, module; std::optional<Label> label; bool measured = false, outside = false; double y = 0, t = 0, ms = 0; std::string message; };

cv::Mat read_image(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    std::vector<unsigned char> b{std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
    return cv::imdecode(b, cv::IMREAD_GRAYSCALE);
}
std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> out; std::string cur; bool quoted = false;
    for (char c : line) {
        if (c == '"') quoted = !quoted;
        else if (c == ',' && !quoted) { out.push_back(cur); cur.clear(); }
        else cur += c;
    }
    out.push_back(cur);
    return out;
}
// manifest.csv: relative_path, ..., filename_Y_command (col 7), filename_T_command (col 8)
std::map<std::string, Label> read_manifest(const fs::path& csv) {
    std::map<std::string, Label> labels; std::ifstream f(csv); std::string line;
    std::getline(f, line);
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        auto c = split_csv(line);
        if (c.size() < 9) continue;
        auto num = [](const std::string& s) { return s.empty() ? 0.0 : std::stod(s); };
        labels[c[0]] = {num(c[7]), num(c[8])};
    }
    return labels;
}
std::optional<Label> label_from_name(const std::string& stem) {
    static const std::regex yr("Y([+-][0-9]+(?:\\.[0-9]+)?)"), tr("T([+-][0-9]+(?:\\.[0-9]+)?)");
    std::smatch ym, tm;
    bool hy = std::regex_search(stem, ym, yr), ht = std::regex_search(stem, tm, tr);
    std::string lower = stem; std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    if (!hy && !ht && lower.find("center") == std::string::npos) return std::nullopt;
    return Label{hy ? std::stod(ym[1]) : 0.0, ht ? std::stod(tm[1]) : 0.0};
}
std::string condition_of(const std::optional<Label>& l) {
    if (!l) return "unlabeled";
    if (l->y == 0 && l->t == 0) return "center";
    if (l->t == 0) return "Y";
    if (l->y == 0) return "T";
    return "mixed";
}
std::string json_escape(const std::string& s) {
    std::string o;
    for (char c : s) { if (c == '"' || c == '\\') o += '\\'; o += c; }
    return o;
}
std::string csv_quote(const std::string& s) {
    std::string o = "\"";
    for (char c : s) { if (c == '"') o += '"'; o += c; }
    return o + "\"";
}
double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    size_t k = (size_t)std::ceil(p * v.size()); return v[std::max<size_t>(1, k) - 1];
}
std::string stats_json(const std::vector<double>& v) {
    double sum = 0, mx = 0; for (double x : v) { sum += x; mx = std::max(mx, x); }
    std::ostringstream o; o << std::setprecision(10)
        << "{\"mean\": " << (v.empty() ? 0 : sum / v.size()) << ", \"max\": " << mx << ", \"p95\": " << percentile(v, 0.95) << "}";
    return o.str();
}
std::string now_text(const char* format) {
    auto t = std::time(nullptr); std::tm tm{}; localtime_s(&tm, &t);
    std::ostringstream o; o << std::put_time(&tm, format); return o.str();
}
}

int wmain(int argc, wchar_t** argv) {
    try {
        cv::setNumThreads(1);
        if (argc < 3) throw std::runtime_error("Usage: benchmark.exe PHOTO_FOLDER NEW_RESULT_FOLDER [joint|control|both] [--joint-assets DIR] [--control-assets DIR]");
        fs::path dataset = fs::absolute(argv[1]), output = fs::absolute(argv[2]);
        std::string which = "both", joint_assets = SEMES_JOINT_ASSETS_ABS, control_assets = SEMES_CONTROL_ASSETS_ABS;
        for (int i = 3; i < argc; ++i) {
            std::string a = fs::path(argv[i]).u8string();
            if (a == "--joint-assets" && i + 1 < argc) joint_assets = fs::path(argv[++i]).u8string();
            else if (a == "--control-assets" && i + 1 < argc) control_assets = fs::path(argv[++i]).u8string();
            else if (a == "joint" || a == "control" || a == "both") which = a;
            else throw std::invalid_argument("Unknown option: " + a);
        }
        if (!fs::is_directory(dataset)) throw std::runtime_error("Photo folder not found: " + dataset.u8string());
        if (fs::exists(output)) throw std::runtime_error("Result folder already exists (results are never overwritten): " + output.u8string());
        std::vector<std::string> modules;
        if (which != "control") { initialize_joint(joint_assets); modules.push_back("joint"); }
        if (which != "joint") { initialize_control(control_assets); modules.push_back("control"); }

        std::vector<fs::path> photos;
        for (const auto& e : fs::recursive_directory_iterator(dataset)) {
            auto ext = e.path().extension().u8string(); std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (e.is_regular_file() && (ext == ".bmp" || ext == ".png") && e.path().u8string().find("_metadata") == std::string::npos) photos.push_back(e.path());
        }
        std::sort(photos.begin(), photos.end());
        if (photos.empty()) throw std::runtime_error("No .bmp/.png photos under " + dataset.u8string());

        std::map<fs::path, std::map<std::string, Label>> manifests;
        auto find_label = [&](const fs::path& photo) -> std::optional<Label> {
            for (fs::path dir = photo.parent_path();; dir = dir.parent_path()) {
                auto csv = dir / "_metadata" / "manifest.csv";
                if (fs::exists(csv)) {
                    if (!manifests.count(dir)) manifests[dir] = read_manifest(csv);
                    auto rel = fs::relative(photo, dir).generic_u8string();
                    auto it = manifests[dir].find(rel);
                    if (it != manifests[dir].end()) return it->second;
                    break;
                }
                if (dir == dataset || dir == dir.parent_path()) break;
            }
            return label_from_name(photo.stem().u8string());
        };

        std::vector<Row> rows;
        for (size_t i = 0; i < photos.size(); ++i) {
            auto label = find_label(photos[i]);
            cv::Mat image = read_image(photos[i]);
            for (const auto& m : modules) {
                Row r{fs::relative(photos[i], dataset).generic_u8string(), condition_of(label), m, label};
                auto t0 = std::chrono::steady_clock::now();
                try {
                    if (image.empty()) throw std::runtime_error("Cannot decode image");
                    auto q = m == "joint" ? joint(image) : control(image);
                    auto d = alignment_diagnostics();
                    r.measured = true; r.y = q.y; r.t = q.t; r.outside = d.outside_training_range;
                } catch (const std::exception& e) { r.message = e.what(); }
                r.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                rows.push_back(r);
            }
            if ((i + 1) % 20 == 0 || i + 1 == photos.size()) std::cout << "scored " << (i + 1) << " / " << photos.size() << std::endl;
        }

        fs::create_directories(output);
        std::ofstream csv(output / "predictions.csv", std::ios::binary);
        csv << "\xEF\xBB\xBF" << "path,condition,label_y,label_t,module,est_y,est_t,measured,outside_training_range,error_y,error_t,pass,ms,message\n" << std::setprecision(12);
        for (const auto& r : rows) {
            csv << csv_quote(r.path) << ',' << r.condition << ',';
            if (r.label) csv << r.label->y << ',' << r.label->t; else csv << ',';
            csv << ',' << r.module << ',';
            if (r.measured) csv << r.y << ',' << r.t; else csv << ',';
            csv << ',' << r.measured << ',' << r.outside << ',';
            if (r.measured && r.label) {
                double ey = std::abs(r.y - r.label->y), et = std::abs(r.t - r.label->t);
                csv << ey << ',' << et << ',' << (ey <= TOLERANCE_Y && et <= TOLERANCE_T);
            } else csv << ",,";
            csv << ',' << std::setprecision(4) << r.ms << std::setprecision(12) << ',' << csv_quote(r.message) << '\n';
        }

        std::ofstream js(output / "summary.json", std::ios::binary);
        auto module_name = [](const std::string& assets) { return fs::u8path(assets).parent_path().filename().u8string(); };
        js << std::setprecision(10) << "{\n  \"tool\": \"03_benchmark\",\n  \"created\": \"" << now_text("%Y-%m-%d %H:%M") << "\",\n"
           << "  \"dataset\": \"" << json_escape(dataset.generic_u8string()) << "\",\n  \"photos\": " << photos.size() << ",\n  \"tolerance_y\": " << TOLERANCE_Y << ",\n  \"tolerance_t\": " << TOLERANCE_T << ",\n"
           << "  \"note\": \"Same-photo scoring. Photos already used in development give a resubstitution score, not an independent one.\",\n  \"modules\": {";
        for (size_t k = 0; k < modules.size(); ++k) {
            auto assets = modules[k] == "joint" ? joint_assets : control_assets;
            js << (k ? ", " : "") << "\"" << modules[k] << "\": {\"name\": \"" << json_escape(module_name(assets)) << "\", \"assets\": \"" << json_escape(fs::u8path(assets).generic_u8string()) << "\"}";
        }
        js << "},\n  \"results\": {\n";
        for (size_t k = 0; k < modules.size(); ++k) {
            std::vector<double> ey, et, ms; int n = 0, measured = 0, labeled = 0, pass = 0;
            std::map<std::string, std::pair<int, int>> by;
            for (const auto& r : rows) {
                if (r.module != modules[k]) continue;
                ++n; measured += r.measured; ms.push_back(r.ms);
                if (!r.label) continue;
                ++labeled; auto& b = by[r.condition]; ++b.first;
                if (!r.measured) continue;
                double dy = std::abs(r.y - r.label->y), dt = std::abs(r.t - r.label->t);
                ey.push_back(dy); et.push_back(dt);
                bool ok = dy <= TOLERANCE_Y && dt <= TOLERANCE_T; pass += ok; b.second += ok;
            }
            js << "    \"" << modules[k] << "\": {\"photos\": " << n << ", \"measured\": " << measured << ", \"labeled\": " << labeled << ", \"pass\": " << pass
               << ", \"error_y\": " << stats_json(ey) << ", \"error_t\": " << stats_json(et) << ", \"ms_median\": " << percentile(ms, 0.5) << ", \"by_condition\": {";
            size_t j = 0;
            for (const auto& [c, v] : by) js << (j++ ? ", " : "") << "\"" << c << "\": {\"photos\": " << v.first << ", \"pass\": " << v.second << "}";
            js << "}}" << (k + 1 < modules.size() ? "," : "") << "\n";
        }
        js << "  }\n}\n";
        std::cout << "wrote " << output.u8string() << std::endl;
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
