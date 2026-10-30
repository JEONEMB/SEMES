#include "core.hpp"

#include <chrono>
#include <ctime>
#include <iostream>
#include <numeric>
#include <regex>
#include <sstream>

namespace hw {

// -------------------------------------------------------------------------------------
// 경로
// -------------------------------------------------------------------------------------

fs::path dataRoot() { return fs::current_path() / "data" / "hw" / "bmp"; }
fs::path outRoot() { return fs::current_path() / "docs" / "validation-runs" / "hw" / "eda"; }
fs::path docsRoot() { return fs::current_path() / "docs" / "hw" / "eda"; }

std::vector<std::string> listSets() {
    std::vector<std::string> sets;
    if (!fs::exists(dataRoot())) return sets;
    for (const auto &e : fs::directory_iterator(dataRoot()))
        if (e.is_directory()) sets.push_back(u8(e.path().filename()));
    std::sort(sets.begin(), sets.end());
    return sets;
}

// -------------------------------------------------------------------------------------
// 로그
// -------------------------------------------------------------------------------------

static std::ofstream g_log;

void openLog(const fs::path &file) { g_log.open(file, std::ios::binary); }

void log(const std::string &msg) {
    std::cout << msg << std::endl;
    if (g_log) g_log << msg << "\n";
}

void fail(const std::string &msg) {
    log("[오류] " + msg);
    std::exit(1);
}

// -------------------------------------------------------------------------------------
// 묶음 읽기
// -------------------------------------------------------------------------------------

std::string trialOf(char axis, double value) {
    const std::string sign = value > 0 ? "+" : "-";
    if (axis == 'T') return "Θ " + sign;
    const long hundredths = std::lround(std::fabs(value) * 100);   // 0.05 단위 판별
    return (hundredths % 10 == 0 ? "Y 0.1 간격 " : "Y 0.05 홀수 ") + sign;
}

// 파일 수정 시각 -> 초, 문자열
static double fileSeconds(const fs::path &p) {
    return std::chrono::duration<double>(fs::last_write_time(p).time_since_epoch()).count();
}
static std::string fileTimeText(const fs::path &p) {
    const auto ft = fs::last_write_time(p);
    const auto sys = std::chrono::system_clock::now() +
                     std::chrono::duration_cast<std::chrono::system_clock::duration>(ft - fs::file_time_type::clock::now());
    const std::time_t t = std::chrono::system_clock::to_time_t(sys);
    std::tm tm{};
    localtime_s(&tm, &t);
    char s[32];
    std::strftime(s, sizeof(s), "%Y-%m-%d %H:%M:%S", &tm);
    return s;
}

// 파일명 해석. 아는 형식이 아니면 false
static bool parseName(const std::string &name, Shot &s) {
    static const std::regex center(R"(^center(\d*)\.bmp$)", std::regex::icase);
    static const std::regex sweepY(R"(^Y([+-]\d+(?:\.\d+)?)\.bmp$)");
    static const std::regex sweepT(R"(^T([+-]\d+(?:\.\d+)?)\.bmp$)");
    static const std::regex grid(R"(^Y([+-]\d+(?:\.\d+)?) T([+-]\d+(?:\.\d+)?) \((\d+)\)\.bmp$)");
    std::smatch m;
    if (std::regex_match(name, m, center)) {
        s.center = true;
        s.rep = m[1].length() ? std::stoi(m[1]) : 1;
        s.group = "center";
    } else if (std::regex_match(name, m, sweepY)) {
        s.y = std::stod(m[1]);
        s.group = trialOf('Y', s.y);
    } else if (std::regex_match(name, m, sweepT)) {
        s.t = std::stod(m[1]);
        s.group = trialOf('T', s.t);
    } else if (std::regex_match(name, m, grid)) {
        s.y = std::stod(m[1]);
        s.t = std::stod(m[2]);
        s.rep = std::stoi(m[3]);
        // 파일마다 자릿수가 달라서(Y+0.1, Y+0.10) 조합 이름은 숫자로 다시 만든다
        s.group = "Y" + signedNum(s.y, 2) + " T" + signedNum(s.t, 2);
    } else {
        return false;
    }
    return true;
}

DataSet loadSet(const std::string &name) {
    DataSet ds;
    ds.name = name;
    ds.dir = dataRoot() / u8path(name);
    if (!fs::exists(ds.dir)) fail("묶음이 없습니다: " + u8(ds.dir));

    bool anyGrid = false;
    for (const auto &e : fs::recursive_directory_iterator(ds.dir)) {
        if (!e.is_regular_file()) continue;
        std::string ext = u8(e.path().extension());
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext != ".bmp") continue;
        Shot s;
        s.file = e.path();
        s.name = u8(e.path().filename());
        s.rel = u8(fs::relative(e.path(), ds.dir));
        if (!parseName(s.name, s)) { log("  건너뜀 (파일명 형식 모름): " + s.rel); continue; }
        if (!s.center && s.y != 0 && s.t != 0) anyGrid = true;
        s.minutes = fileSeconds(s.file);
        s.takenAt = fileTimeText(s.file);
        ds.shots.push_back(s);
    }
    if (ds.shots.empty()) fail("BMP 가 없습니다: " + u8(ds.dir));
    ds.kind = anyGrid ? SetKind::Grid : SetKind::Sweep;

    std::sort(ds.shots.begin(), ds.shots.end(), [](const Shot &a, const Shot &b) { return a.minutes < b.minutes; });
    const double t0 = ds.shots.front().minutes;
    for (size_t i = 0; i < ds.shots.size(); ++i) {
        ds.shots[i].minutes = (ds.shots[i].minutes - t0) / 60.0;
        if (ds.shots[i].center) ds.centers.push_back(i);
    }
    if (ds.centers.empty()) fail("center 사진이 없습니다: " + u8(ds.dir));
    // center 가 여러 장이면 번호 순 (center1 이 기준)
    std::sort(ds.centers.begin(), ds.centers.end(), [&](size_t a, size_t b) { return ds.shots[a].rep < ds.shots[b].rep; });
    return ds;
}

// -------------------------------------------------------------------------------------
// 이미지
// -------------------------------------------------------------------------------------

cv::Mat readImage(const fs::path &p, int flags) {
    std::ifstream in(p, std::ios::binary);
    std::vector<uchar> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    cv::Mat img = cv::imdecode(bytes, flags);
    if (img.empty()) fail("이미지를 읽지 못했습니다: " + u8(p));
    return img;
}

cv::Mat readGray(const fs::path &p) {
    cv::Mat img = readImage(p, cv::IMREAD_UNCHANGED);
    if (img.channels() == 1) return img;
    cv::Mat ch;
    cv::extractChannel(img, ch, 0);
    return ch;
}

void writeImage(const fs::path &p, const cv::Mat &img) {
    fs::create_directories(p.parent_path());
    std::vector<uchar> bytes;
    if (!cv::imencode(u8(p.extension()), img, bytes)) fail("그림 저장 실패: " + u8(p));
    std::ofstream out(p, std::ios::binary);
    out.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

// -------------------------------------------------------------------------------------
// 숫자, CSV, 요약값
// -------------------------------------------------------------------------------------

std::string num(double v, int prec) {
    if (!std::isfinite(v)) return "nan";
    std::ostringstream s;
    s.setf(std::ios::fixed);
    s.precision(prec);
    s << v;
    std::string r = s.str();
    if (r.size() > 1 && r[0] == '-' && r.find_first_not_of("-0.") == std::string::npos) r = r.substr(1);  // -0.000 -> 0.000
    return r;
}

std::string signedNum(double v, int prec) {
    const std::string r = num(v, prec);
    return (r[0] == '-' || r == "nan") ? r : "+" + r;
}

Csv::Csv(const fs::path &file, const std::vector<std::string> &header) {
    fs::create_directories(file.parent_path());
    out_.open(file, std::ios::binary);
    out_ << "\xEF\xBB\xBF";   // 엑셀에서 한글이 깨지지 않게 UTF-8 BOM
    row(header);
}

void Csv::row(const std::vector<std::string> &cells) {
    for (size_t i = 0; i < cells.size(); ++i) {
        const bool quote = cells[i].find_first_of(",\"") != std::string::npos;
        if (i) out_ << ",";
        if (quote) {
            std::string c = cells[i];
            for (size_t k = 0; (k = c.find('"', k)) != std::string::npos; k += 2) c.insert(k, "\"");
            out_ << "\"" << c << "\"";
        } else {
            out_ << cells[i];
        }
    }
    out_ << "\n";
}

void Summary::put(const std::string &key, const std::string &v) {
    for (auto &p : kv_)
        if (p.first == key) { p.second = v; return; }
    kv_.emplace_back(key, v);
}

bool Summary::has(const std::string &key) const {
    return std::any_of(kv_.begin(), kv_.end(), [&](const auto &p) { return p.first == key; });
}

double Summary::get(const std::string &key) const {
    for (const auto &p : kv_)
        if (p.first == key) {
            try { return std::stod(p.second); } catch (...) { return kNaN; }
        }
    return kNaN;
}

void Summary::write(const fs::path &file, const std::string &header) const {
    fs::create_directories(file.parent_path());
    std::ofstream out(file, std::ios::binary);
    out << "# " << header << "\n";
    for (const auto &p : kv_) out << p.first << ": " << p.second << "\n";
}

Summary Summary::read(const fs::path &file) {
    Summary s;
    std::ifstream in(file, std::ios::binary);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        const size_t c = line.find(": ");
        if (c == std::string::npos) continue;
        std::string v = line.substr(c + 2);
        if (!v.empty() && v.back() == '\r') v.pop_back();
        s.kv_.emplace_back(line.substr(0, c), v);
    }
    return s;
}

// -------------------------------------------------------------------------------------
// 통계
// -------------------------------------------------------------------------------------

double mean(const std::vector<double> &v) {
    return v.empty() ? kNaN : std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

double stdev(const std::vector<double> &v, int ddof) {
    if (static_cast<int>(v.size()) <= ddof) return kNaN;
    const double m = mean(v);
    double s = 0;
    for (double x : v) s += (x - m) * (x - m);
    return std::sqrt(s / (v.size() - ddof));
}

double median(std::vector<double> v) {
    v.erase(std::remove_if(v.begin(), v.end(), [](double x) { return !std::isfinite(x); }), v.end());
    if (v.empty()) return kNaN;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

double maxAbs(const std::vector<double> &v) {
    double m = 0;
    for (double x : v)
        if (std::isfinite(x)) m = std::max(m, std::fabs(x));
    return m;
}

LineFit fitLine(const std::vector<double> &x, const std::vector<double> &y) {
    LineFit f;
    const size_t n = x.size();
    if (n < 2) return f;
    const double mx = mean(x), my = mean(y);
    double sxx = 0, sxy = 0;
    for (size_t i = 0; i < n; ++i) { sxx += (x[i] - mx) * (x[i] - mx); sxy += (x[i] - mx) * (y[i] - my); }
    f.slope = sxy / sxx;
    f.intercept = my - f.slope * mx;
    if (n > 2) {
        double r = 0;
        for (size_t i = 0; i < n; ++i) { const double e = y[i] - (f.slope * x[i] + f.intercept); r += e * e; }
        f.residSd = std::sqrt(r / (n - 2));
    }
    return f;
}

std::vector<double> quadResiduals(const std::vector<double> &x, const std::vector<double> &y) {
    const int n = static_cast<int>(x.size());
    cv::Mat A(n, 3, CV_64F), b(n, 1, CV_64F), c;
    for (int i = 0; i < n; ++i) {
        A.at<double>(i, 0) = 1; A.at<double>(i, 1) = x[i]; A.at<double>(i, 2) = x[i] * x[i];
        b.at<double>(i) = y[i];
    }
    cv::solve(A, b, c, cv::DECOMP_SVD);
    std::vector<double> r(n);
    for (int i = 0; i < n; ++i) r[i] = y[i] - (c.at<double>(0) + c.at<double>(1) * x[i] + c.at<double>(2) * x[i] * x[i]);
    return r;
}

}  // namespace hw
