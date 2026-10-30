// 결과/ 폴더 쓰기: CSV, 마크다운 표, 숫자 서식, 기본 통계. 출력에 시각·난수를 넣지 않아 다시 돌려도 같은 파일이 나온다.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace hye {

#ifndef EXPERIMENT_DIR
#define EXPERIMENT_DIR "."
#endif

namespace fs = std::filesystem;
const double kNaN = std::numeric_limits<double>::quiet_NaN();

// 결과/ 폴더. 이미지/ 하위 폴더도 만든다. 한글 경로는 모두 fs::u8path 로 만든다(소스는 /utf-8).
inline fs::path result_dir() {
    fs::path d = fs::u8path(EXPERIMENT_DIR);
    fs::create_directories(d / fs::u8path("이미지"));
    return d;
}
inline fs::path image_dir() { return result_dir() / fs::u8path("이미지"); }
// 실행 시작 때 한 번: 이미지/ 를 비워서 이전 실행의 그림이 남지 않게 한다
inline void begin_results() {
    fs::remove_all(result_dir() / fs::u8path("이미지"));
    result_dir();
}
inline fs::path out(const std::string& name) { return result_dir() / fs::u8path(name); }
inline std::string u8(const fs::path& p) { return p.u8string(); }

inline std::string fmt(double v, int prec = 4) {
    if (std::isnan(v)) return "nan";
    char b[64];
    std::snprintf(b, sizeof b, "%.*f", prec, v);
    std::string s = b;
    if (s == "-0" || s.find_first_not_of("-0.") == std::string::npos) s.erase(0, s[0] == '-' ? 1 : 0);
    return s;
}
inline std::string fmts(double v, int prec = 3) {  // 부호 붙임
    std::string s = fmt(v, prec);
    return (v >= 0 && s[0] != '-') ? "+" + s : s;
}

struct Table {
    std::vector<std::string> head;
    std::vector<std::vector<std::string>> rows;
    void add(std::vector<std::string> r) { rows.push_back(std::move(r)); }
    std::string md() const {
        std::ostringstream o;
        auto line = [&](const std::vector<std::string>& r) {
            o << "|";
            for (auto& c : r) {
                std::string x = c;
                for (size_t p = 0; (p = x.find('|', p)) != std::string::npos; p += 2) x.replace(p, 1, "\\|");
                o << " " << x << " |";
            }
            o << "\n";
        };
        line(head);
        o << "|";
        for (size_t i = 0; i < head.size(); ++i) o << "---|";
        o << "\n";
        for (auto& r : rows) line(r);
        return o.str();
    }
    void csv(const fs::path& p) const {
        std::ofstream f(p, std::ios::binary);
        auto line = [&](const std::vector<std::string>& r) {
            for (size_t i = 0; i < r.size(); ++i) {
                bool q = r[i].find_first_of(",\"") != std::string::npos;
                f << (i ? "," : "") << (q ? "\"" + r[i] + "\"" : r[i]);
            }
            f << "\n";
        };
        line(head);
        for (auto& r : rows) line(r);
    }
};

inline void write_text(const fs::path& p, const std::string& s) {
    std::ofstream f(p, std::ios::binary);
    f << s;
}

// 그림 링크: 보고서.md 기준 상대 경로
inline std::string img_md(const std::string& file, const std::string& alt) {
    return "![" + alt + "](" + std::string("이미지/") + file + ")";
}

// 통계
inline double mean(const std::vector<double>& v) {
    return v.empty() ? kNaN : std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}
inline double rms(const std::vector<double>& v) {
    if (v.empty()) return kNaN;
    double s = 0;
    for (double x : v) s += x * x;
    return std::sqrt(s / v.size());
}
inline double max_abs(const std::vector<double>& v) {
    double m = 0;
    for (double x : v) m = std::max(m, std::abs(x));
    return m;
}
inline double percentile(std::vector<double> v, double q) {
    if (v.empty()) return kNaN;
    std::sort(v.begin(), v.end());
    double k = (v.size() - 1) * q;
    size_t i = (size_t)std::floor(k);
    double f = k - i;
    return i + 1 < v.size() ? v[i] * (1 - f) + v[i + 1] * f : v[i];
}
inline std::vector<double> absv(std::vector<double> v) {
    for (auto& x : v) x = std::abs(x);
    return v;
}
// |오차| 큰 순서의 인덱스
inline std::vector<size_t> worst(const std::vector<double>& err, size_t k) {
    std::vector<size_t> idx(err.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return std::abs(err[a]) > std::abs(err[b]); });
    if (idx.size() > k) idx.resize(k);
    return idx;
}

// 원래 결과와 C++ 결과 대조 한 줄
struct Parity {
    Table t{{"지표", "원래 결과", "C++ 결과", "차이", "판정"}};
    int ok = 0, total = 0;
    void add(const std::string& name, double orig, double now, double tol, int prec = 4) {
        bool pass = std::abs(orig - now) <= tol;
        ok += pass;
        ++total;
        t.add({name, fmt(orig, prec), fmt(now, prec), fmt(now - orig, prec), pass ? "일치" : "불일치"});
    }
    void add_text(const std::string& name, const std::string& orig, const std::string& now, bool pass) {
        ok += pass;
        ++total;
        t.add({name, orig, now, "-", pass ? "일치" : "불일치"});
    }
};

}  // namespace hye
