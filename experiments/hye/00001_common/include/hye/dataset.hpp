// data/clean 의 사진 목록(manifest.csv)을 읽는다. 파일명 명령값(Y mm, T deg)을 정답으로 쓴다.
#pragma once
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace hye {

#ifndef REPO_DIR
#define REPO_DIR "."
#endif

inline std::string repo_dir() { return REPO_DIR; }
inline std::string clean_dir() { return repo_dir() + "/data/clean"; }

struct Shot {
    std::string set;        // clean_20261014 / clean_20261020 / clean_20261029
    std::string rel;        // set 안 상대 경로
    std::string path;       // 절대 경로
    double y = 0, t = 0;    // 파일명 명령값
    int repeat = 1;         // 같은 명령의 반복 번호
    std::string condition;  // center / Y / T / mixed
    std::string name() const { return set.substr(6) + "/" + rel; }
};

inline std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> out;
    std::string cur;
    bool q = false;
    for (char c : line) {
        if (c == '"') q = !q;
        else if (c == ',' && !q) { out.push_back(cur); cur.clear(); }
        else if (c != '\r') cur += c;
    }
    out.push_back(cur);
    return out;
}

// set: "clean_20261014" 등. manifest 순서를 그대로 유지한다.
inline std::vector<Shot> load_set(const std::string& set) {
    const std::string base = clean_dir() + "/" + set;
    std::ifstream in(base + "/_metadata/manifest.csv");
    if (!in) throw std::runtime_error("manifest 없음: " + base);
    std::string line;
    std::getline(in, line);
    if (line.size() >= 3 && (unsigned char)line[0] == 0xEF) line = line.substr(3);
    auto head = split_csv(line);
    auto col = [&](const char* n) {
        for (size_t i = 0; i < head.size(); ++i) if (head[i] == n) return (int)i;
        throw std::runtime_error(std::string("열 없음: ") + n);
    };
    const int cp = col("relative_path"), cy = col("filename_Y_command"), ct = col("filename_T_command"),
              cr = col("capture_repeat"), cc = col("condition");
    std::vector<Shot> out;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        auto f = split_csv(line);
        Shot s;
        s.set = set;
        s.rel = f[cp];
        s.path = base + "/" + s.rel;
        s.y = std::stod(f[cy]);
        s.t = std::stod(f[ct]);
        s.repeat = std::stoi(f[cr]);
        s.condition = f[cc];
        out.push_back(s);
    }
    return out;
}

// 10-14 세트: center.bmp 1장 + Y 80장 + T 200장
inline std::vector<Shot> load_1014() { return load_set("clean_20261014"); }

// 10-14(281) + 10-20(60) = 341장. dev 평가에서 쓰던 묶음.
inline std::vector<Shot> load_341() {
    auto a = load_set("clean_20261014");
    auto b = load_set("clean_20261020");
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

inline const Shot& center_1014(const std::vector<Shot>& shots) {
    for (auto& s : shots) if (s.set == "clean_20261014" && s.condition == "center") return s;
    throw std::runtime_error("center.bmp 없음");
}

inline cv::Mat read_gray(const Shot& s) {
    cv::Mat m = cv::imread(s.path, cv::IMREAD_GRAYSCALE);
    if (m.empty()) throw std::runtime_error("읽기 실패: " + s.path);
    return m;
}

inline cv::Mat read_color(const Shot& s) {
    cv::Mat m = cv::imread(s.path, cv::IMREAD_COLOR);
    if (m.empty()) throw std::runtime_error("읽기 실패: " + s.path);
    return m;
}

// 판정 통과 기준: |eY| <= 0.09 mm, |eT| <= 0.09 deg
constexpr double kPass = 0.09;

}  // namespace hye
