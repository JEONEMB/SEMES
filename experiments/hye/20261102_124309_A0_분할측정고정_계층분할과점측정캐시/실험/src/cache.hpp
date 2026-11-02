// A0 가 만든 분할·점 측정 캐시의 형식과 읽기 함수.
// A1~A9 는 이 헤더를 include 해서 사진을 다시 측정하지 않고 캐시만 읽는다.
#pragma once
#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace a0 {
namespace fs = std::filesystem;

constexpr int kPoints = 630;   // 고정 점 수
constexpr int kLines = 48;     // 허용된 위·아래 가로선 블록 수
constexpr int kQuality = 5;    // 점별 품질 열 수

// 2026-11-02 사용자 재정의: 0.09 는 5회 반복 정밀도(전폭)이고 한 장의 허용 절대오차는 그 절반이다.
constexpr double kTol = 0.045;      // |eY|, |eT| 통과 기준
constexpr double kTight = 0.015;    // 판정용 "작다" 경계 = kTol/3. 독립 기준기 요구 정밀도와 같다.
// 팀 측정 감도 (docs/hye/문헌연구 01_주제_관련/원인분석/논문/01_원인의_수식정리.md 58행)
constexpr double kPxPerMmY = 14.25;  // 판 가로 이동 px per Y mm
constexpr double kPxPerT = 80.04;    // 판 가로 이동 px per T 1단위
constexpr double kTtoY = kPxPerT / kPxPerMmY;  // T 1단위 = Y 5.62 mm 상당

struct Shot {
    int idx = 0;
    std::string set, rel, condition;
    double y = 0, t = 0;
    int repeat = 1;
    int group = -1;        // 같은 (세트, Y, T) 묶음. 반복 3장은 같은 번호.
    int fold_mixed = -1;   // 혼합 4-fold. -1 = center.bmp (어느 시험에도 넣지 않음)
    int fold_date = -1;    // 날짜 3-fold. 1014=0, 1020=1, 1029=2
    double ncc[3] = {0, 0, 0};
    int valid_count = 0, valid_selected = 0;
    bool measured = false;
    std::string name() const { return set.substr(6) + "/" + rel; }
};

struct Point {
    double x = 0, y = 0;
    int region = 0, selected = 0;
};

struct Cache {
    std::vector<Shot> shots;
    std::vector<Point> points;
    std::vector<float> disp;            // n*630*2, center.bmp 기준 변위 px
    std::vector<unsigned char> valid;   // n*630
    std::vector<float> quality;         // n*630*5
    std::vector<float> zero;            // 630*2, center 자기 측정
    std::vector<float> lines;           // n*48*2 (이동량 px, 상관)
    std::vector<unsigned char> lines_valid;  // n*48

    int n() const { return int(shots.size()); }
    const float* d(int i) const { return &disp[size_t(i) * kPoints * 2]; }
    const unsigned char* v(int i) const { return &valid[size_t(i) * kPoints]; }
    const float* q(int i) const { return &quality[size_t(i) * kPoints * kQuality]; }
    const float* line(int i) const { return &lines[size_t(i) * kLines * 2]; }
    const unsigned char* line_ok(int i) const { return &lines_valid[size_t(i) * kLines]; }
    // center 영점을 뺀 점 변위 (control 의 features() 와 같은 정의)
    cv::Point2d dz(int i, int p) const {
        return {double(d(i)[p * 2]) - double(zero[p * 2]),
                double(d(i)[p * 2 + 1]) - double(zero[p * 2 + 1])};
    }
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

template <class T>
void read_binary(const fs::path& p, std::vector<T>& out, size_t count) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("캐시 없음: " + p.u8string());
    out.resize(count);
    f.read(reinterpret_cast<char*>(out.data()), std::streamsize(count * sizeof(T)));
    if (size_t(f.gcount()) != count * sizeof(T)) throw std::runtime_error("캐시 크기 불일치: " + p.u8string());
}

template <class T>
void write_binary(const fs::path& p, const std::vector<T>& v) {
    std::ofstream f(p, std::ios::binary);
    f.write(reinterpret_cast<const char*>(v.data()), std::streamsize(v.size() * sizeof(T)));
}

// A0 의 결과/cache 폴더를 읽는다.
inline Cache load(const fs::path& dir) {
    Cache c;
    {
        std::ifstream f(dir / "shots.csv");
        if (!f) throw std::runtime_error("shots.csv 없음: " + (dir / "shots.csv").u8string());
        std::string line;
        std::getline(f, line);
        auto head = split_csv(line);
        auto col = [&](const char* n) {
            for (size_t i = 0; i < head.size(); ++i) if (head[i] == n) return int(i);
            throw std::runtime_error(std::string("shots.csv 열 없음: ") + n);
        };
        const int ci = col("idx"), cs = col("set"), cr = col("rel"), cy = col("y"), ct = col("t"),
                  cp = col("repeat"), cc = col("condition"), cg = col("group"), cm = col("fold_mixed"),
                  cd = col("fold_date"), c0 = col("ncc0"), c1 = col("ncc1"), c2 = col("ncc2"),
                  cv = col("valid_count"), cvs = col("valid_selected"), cme = col("measured");
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            auto r = split_csv(line);
            Shot s;
            s.idx = std::stoi(r[ci]); s.set = r[cs]; s.rel = r[cr];
            s.y = std::stod(r[cy]); s.t = std::stod(r[ct]); s.repeat = std::stoi(r[cp]);
            s.condition = r[cc]; s.group = std::stoi(r[cg]);
            s.fold_mixed = std::stoi(r[cm]); s.fold_date = std::stoi(r[cd]);
            s.ncc[0] = std::stod(r[c0]); s.ncc[1] = std::stod(r[c1]); s.ncc[2] = std::stod(r[c2]);
            s.valid_count = std::stoi(r[cv]); s.valid_selected = std::stoi(r[cvs]);
            s.measured = r[cme] == "1";
            c.shots.push_back(s);
        }
    }
    {
        std::ifstream f(dir / "points.csv");
        if (!f) throw std::runtime_error("points.csv 없음");
        std::string line;
        std::getline(f, line);
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            auto r = split_csv(line);
            Point p;
            p.x = std::stod(r[1]); p.y = std::stod(r[2]);
            p.region = std::stoi(r[3]); p.selected = std::stoi(r[4]);
            c.points.push_back(p);
        }
        if (c.points.size() != kPoints) throw std::runtime_error("points.csv 행 수가 630이 아니다");
    }
    const size_t n = c.shots.size();
    read_binary(dir / "disp.f32", c.disp, n * kPoints * 2);
    read_binary(dir / "valid.u8", c.valid, n * kPoints);
    read_binary(dir / "quality.f32", c.quality, n * kPoints * kQuality);
    read_binary(dir / "zero.f32", c.zero, size_t(kPoints) * 2);
    read_binary(dir / "lines.f32", c.lines, n * kLines * 2);
    read_binary(dir / "lines_valid.u8", c.lines_valid, n * kLines);
    return c;
}

// A0 결과/cache 의 기본 위치. 실험마다 CMake 가 A0_CACHE 를 정의해 준다.
inline fs::path cache_dir() {
#ifdef A0_CACHE
    return fs::u8path(A0_CACHE);
#else
    throw std::runtime_error("A0_CACHE 가 정의되지 않았다");
#endif
}

// ---- 공용 통계 ----
inline double median(std::vector<double> v) {
    if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::sort(v.begin(), v.end());
    size_t m = v.size() / 2;
    return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}
inline double quantile(std::vector<double> v, double q) {
    if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::sort(v.begin(), v.end());
    double pos = q * (v.size() - 1);
    size_t lo = size_t(pos);
    double f = pos - lo;
    return lo + 1 < v.size() ? v[lo] * (1 - f) + v[lo + 1] * f : v[lo];
}
inline double mean(const std::vector<double>& v) {
    return v.empty() ? std::numeric_limits<double>::quiet_NaN()
                     : std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}
inline double sd(const std::vector<double>& v) {
    if (v.size() < 2) return std::numeric_limits<double>::quiet_NaN();
    double m = mean(v), s = 0;
    for (double x : v) s += (x - m) * (x - m);
    return std::sqrt(s / (v.size() - 1));
}
inline double rms(const std::vector<double>& v) {
    if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
    double s = 0;
    for (double x : v) s += x * x;
    return std::sqrt(s / v.size());
}
inline double max_abs(const std::vector<double>& v) {
    double m = 0;
    for (double x : v) m = std::max(m, std::abs(x));
    return v.empty() ? std::numeric_limits<double>::quiet_NaN() : m;
}
inline std::string num(double v, int p = 4) {
    if (!std::isfinite(v)) return "";
    std::ostringstream o;
    o.setf(std::ios::fixed);
    o.precision(p);
    o << v;
    return o.str();
}

// ---- 간단한 마크다운/CSV 표 ----
struct Table {
    std::vector<std::string> head;
    std::vector<std::vector<std::string>> rows;
    void add(std::vector<std::string> r) { rows.push_back(std::move(r)); }
    std::string md() const {
        std::ostringstream o;
        auto line = [&](const std::vector<std::string>& r) {
            o << "|";
            for (auto& c : r) o << " " << c << " |";
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

}  // namespace a0
