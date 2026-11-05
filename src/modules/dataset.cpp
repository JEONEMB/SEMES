// dataset.cpp — 라벨 사진 목록 구현
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

// 파일 이름 → 라벨
//  "center"              : 1차 기준 사진          "center1" ~ : 2차 기준 사진
//  "Y+0.50 T+0.10 (2)"   : 동시 이동 (2차·3차)     "Y+0.50" / "T+0.10" : 1차 단독 이동
bool parse_label(const std::string &stem, LabeledPhoto &photo) {
    static const std::regex re_center(R"(^center$)"), re_center_n(R"(^center\d+$)");
    static const std::regex re_mixed(R"(^Y([+-]?\d+(?:\.\d+)?)\s+T([+-]?\d+(?:\.\d+)?)\s*\(\d+\)$)");
    static const std::regex re_y(R"(^Y([+-]?\d+(?:\.\d+)?)$)"), re_t(R"(^T([+-]?\d+(?:\.\d+)?)$)");
    std::smatch m;
    LabeledPhoto p;
    p.stem = stem;
    if (std::regex_match(stem, re_center)) p.is_reference = true;
    else if (std::regex_match(stem, re_center_n)) { p.set = 2; p.is_reference = true; }
    else if (std::regex_match(stem, m, re_mixed)) { p.y_mm = std::stod(m[1]); p.t_deg = std::stod(m[2]); p.set = 2; }
    else if (std::regex_match(stem, m, re_y)) p.y_mm = std::stod(m[1]);
    else if (std::regex_match(stem, m, re_t)) p.t_deg = std::stod(m[1]);
    else return false;
    p.path = photo.path;
    photo = p;
    return true;
}

// 폴더(하위 폴더 포함)의 라벨 사진 목록 — 이름 순, 같은 이름이면 bmp 우선
std::vector<LabeledPhoto> list_labeled_photos(const std::string &dir) {
    std::map<std::string, std::string> files;   // 이름 → 경로 (bmp 우선)
    std::error_code ec;
    // 하위 폴더까지 찾음 (원본 배치 data/img/1st/Y, 2nd/Y+a T+b/ … 를 그대로 쓸 수 있게)
    for (const auto &e : fs::recursive_directory_iterator(fs::u8path(dir), ec)) {
        if (!e.is_regular_file()) continue;
        std::string ext = e.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char ch) { return char(std::tolower(ch)); });
        if (ext != ".bmp" && ext != ".png") continue;
        std::string stem = e.path().stem().u8string();
        // 폴더 이름이 라벨(Y+a T+b)인데 파일 라벨과 다르면 폴더를 따름
        // (3rd 세트 'Y+0.05 T+0.32' 폴더 안의 'Y+0.05 T+0.33 (3)' — 촬영 시각·모터 값으로 T+0.32 (3) 확인)
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

// 이름으로 사진 찾기
const LabeledPhoto *find_photo(const std::vector<LabeledPhoto> &photos, const std::string &stem) {
    for (const auto &p : photos)
        if (p.stem == stem) return &p;
    return nullptr;
}

// 학습 사진 : 지정 세트의 Θ 단독 사진 전부 + Y 단독 사진 중 y_grid_mm 배수 (기준 사진 · 동시 이동 사진 제외)
bool is_training_photo(const LabeledPhoto &photo, int set, double y_grid_mm) {
    if (photo.is_reference || photo.set != set) return false;
    const bool y_only = photo.t_deg == 0.0 && photo.y_mm != 0.0, t_only = photo.y_mm == 0.0 && photo.t_deg != 0.0;
    if (t_only) return true;
    if (!y_only) return false;
    if (y_grid_mm <= 0) return true;
    const double k = photo.y_mm / y_grid_mm;
    return std::abs(k - std::round(k)) < 1e-6;
}

// 특징 표본 CSV 저장 (UTF-8 BOM, 엑셀에서 한글이 깨지지 않게)
bool save_feature_csv(const std::string &path, const std::vector<FeatureSample> &samples) {
    std::ofstream f(fs::u8path(path));
    if (!f) return false;
    f << "\xEF\xBB\xBF" << "stem,y_mm,t_deg,s_px,d_px\n" << std::setprecision(17);
    for (const auto &s : samples) f << '"' << s.stem << "\"," << s.y_mm << ',' << s.t_deg << ',' << s.s_px << ',' << s.d_px << '\n';
    return true;
}

// 특징 표본 CSV 읽기 (save_feature_csv 형식)
bool load_feature_csv(const std::string &path, std::vector<FeatureSample> &samples) {
    std::ifstream f(fs::u8path(path));
    if (!f) return false;
    std::string line;
    std::getline(f, line);   // 머리줄
    samples.clear();
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        FeatureSample s;
        size_t q = line.find('"', 1);
        if (line[0] != '"' || q == std::string::npos) return false;
        s.stem = line.substr(1, q - 1);
        std::stringstream ss(line.substr(q + 2));
        char comma;
        if (!(ss >> s.y_mm >> comma >> s.t_deg >> comma >> s.s_px >> comma >> s.d_px)) return false;
        samples.push_back(s);
    }
    return true;
}

} // namespace ksh
