// dataset.cpp — 라벨 사진 목록 · 특징 CSV 구현
#include "ksh/dataset.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <regex>
#include <sstream>

namespace ksh {

namespace fs = std::filesystem;

bool parse_label(const std::string &stem, LabeledPhoto &photo) {
    static const std::regex re_center(R"(^center\d*$)");
    static const std::regex re_mixed(R"(^Y([+-]?\d+(?:\.\d+)?)\s+T([+-]?\d+(?:\.\d+)?)(?:\s*\(\d+\))?$)");
    static const std::regex re_y(R"(^Y([+-]?\d+(?:\.\d+)?)$)"), re_t(R"(^T([+-]?\d+(?:\.\d+)?)$)");
    std::smatch m;
    LabeledPhoto p;
    p.stem = stem;
    p.path = photo.path;
    if (std::regex_match(stem, re_center)) p.is_reference = true;
    else if (std::regex_match(stem, m, re_mixed)) { p.y_mm = std::stod(m[1]); p.t_deg = std::stod(m[2]); }
    else if (std::regex_match(stem, m, re_y)) p.y_mm = std::stod(m[1]);
    else if (std::regex_match(stem, m, re_t)) p.t_deg = std::stod(m[1]);
    else return false;
    photo = p;
    return true;
}

std::vector<LabeledPhoto> list_labeled_photos(const std::string &dir) {
    std::map<std::string, std::string> files;   // 이름 → 경로 (bmp 우선)
    std::error_code ec;
    for (const auto &e : fs::recursive_directory_iterator(fs::u8path(dir), ec)) {
        if (!e.is_regular_file()) continue;
        std::string ext = e.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char ch) { return char(std::tolower(ch)); });
        if (ext != ".bmp" && ext != ".png") continue;
        std::string stem = e.path().stem().u8string();
        const std::string folder = e.path().parent_path().filename().u8string();
        LabeledPhoto by_folder, by_file;
        const size_t k = stem.rfind(" (");
        if (k != std::string::npos && parse_label(folder + stem.substr(k), by_folder) && parse_label(stem, by_file) &&
            (std::abs(by_folder.y_mm - by_file.y_mm) > 1e-9 || std::abs(by_folder.t_deg - by_file.t_deg) > 1e-9))
            stem = folder + stem.substr(k);
        if (!files.count(stem) || ext == ".bmp") files[stem] = e.path().u8string();
    }
    std::vector<LabeledPhoto> out;
    for (const auto &f : files) {
        LabeledPhoto p;
        p.path = f.second;
        if (parse_label(f.first, p)) out.push_back(p);
    }
    return out;
}

bool save_feature_csv(const std::string &path, const std::vector<FeatureRow> &rows) {
    std::ofstream f(fs::u8path(path));
    if (!f) return false;
    f << "\xEF\xBB\xBF" << "stem,y_mm,t_deg,s_px,d_px,is_ref\n" << std::setprecision(17);
    for (const auto &r : rows)
        f << '"' << r.sample.stem << "\"," << r.sample.y_mm << ',' << r.sample.t_deg << ',' << r.sample.s_px << ',' << r.sample.d_px << ','
          << int(r.is_reference) << '\n';
    return true;
}

bool load_feature_csv(const std::string &path, std::vector<FeatureRow> &rows) {
    std::ifstream f(fs::u8path(path));
    if (!f) return false;
    std::string line;
    std::getline(f, line);   // 머리줄
    rows.clear();
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        FeatureRow r;
        const size_t q = line.find('"', 1);
        if (line[0] != '"' || q == std::string::npos) return false;
        r.sample.stem = line.substr(1, q - 1);
        std::stringstream ss(line.substr(q + 2));
        char comma;
        int ref = 0;
        if (!(ss >> r.sample.y_mm >> comma >> r.sample.t_deg >> comma >> r.sample.s_px >> comma >> r.sample.d_px)) return false;
        if (ss >> comma >> ref) r.is_reference = ref != 0;
        rows.push_back(r);
    }
    return true;
}

} // namespace ksh
