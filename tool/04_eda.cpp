// 04_eda: 새 자료(--target, 기본 20261020) EDA 측정과 분석. C++17 / OpenCV 3.4 / Windows tar.exe.
//
// 기준 영상은 ALGORITHM_RULES.md 대로 clean_20261014/center.bmp 하나로 고정한다.
// 1차 281장도 같은 방법으로 다시 재서, 대상 자료를 1차 결과(기울기, 노이즈)와 비교한다.
// 대상이 20261020 이 아니면 같은 세션의 20261020 도 함께 재서 center 와 시간 비교에 쓴다.
//
//   1. QR 라벨      center 에서 밝은 라벨 3개(L, C, R)를 찾는다
//   2. ECC 패치     라벨 + 바깥 8px 를 사진마다 템플릿 매칭 -> ECC 아핀으로 맞춘다
//   3. 구조 모서리  QR 을 디코딩한 모듈 격자에서 검정/흰 칸이 만나는 꼭짓점에 ID 를 고정하고,
//                  사진마다 ECC 로 예측한 자리에서 흐린 2x2 경계 모델을 서브픽셀로 맞춘다.
//                  이동 = 대각선 모서리를 뺀 점들의 평균 (통과 여부와 무관하게 같은 점)
//   4. 촬영 시각    clean 은 파일 시각을 보존하지 않아, 원본 ZIP 마지막 분할 조각의 중앙 디렉터리
//                  (NTFS 수정 시각, 100ns)에서 읽는다. 압축 데이터는 풀지 않는다.
//   5. 분석         1차: trial 직선/2차 곡선, 0점, 사진 공통 흔들림, 점 흔들림
//                  2차: 1차 곡선으로 만든 예측과 비교(더해지는지, 0점), 반복 3장 흔들림,
//                       center 3장 0점과 시간 변화, 밝기, Y/Θ 구분 단서
//                  대상이 한 Y 에서 Θ 만 훑은 자료면: 기울기, 0점, 한 칸 이동량, 반복별 경로 (sweep.csv)
//
// 출력(새 폴더): shots.csv, corners_ref.csv, qr.csv, set1_trials.csv, set2_conditions.csv,
//               model.csv, sweep.csv(해당 시), summary.yml, crop/*.png
#include <windows.h>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
namespace fs = std::filesystem;

namespace {
const double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr int kQr = 3;
const char *const kQrName[kQr] = {"L", "C", "R"};
constexpr int kPatchMargin = 8;

// -------------------------------------------------------------------------------------
// 1. 공통: 문자열, 파일, CSV, 통계
// -------------------------------------------------------------------------------------

void require(bool ok, const std::string &msg) {
    if (!ok) throw std::runtime_error(msg);
}
std::string utf8(const std::wstring &s) {
    if (s.empty()) return {};
    int n = WideCharToMultiByte(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0, nullptr, nullptr);
    std::string out(n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), &out[0], n, nullptr, nullptr);
    return out;
}
std::wstring wide(const std::string &s, UINT cp = CP_UTF8) {
    if (s.empty()) return {};
    int n = MultiByteToWideChar(cp, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
    std::wstring out(n, L'\0');
    MultiByteToWideChar(cp, 0, s.data(), static_cast<int>(s.size()), &out[0], n);
    return out;
}
std::vector<unsigned char> readBytes(const fs::path &p) {
    std::ifstream f(p, std::ios::binary);
    require(f.good(), "파일을 열 수 없음: " + p.u8string());
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}
void writePng(const fs::path &p, const cv::Mat &img) {
    std::vector<unsigned char> buf;
    require(cv::imencode(".png", img, buf), "PNG 인코딩 실패");
    std::ofstream(p, std::ios::binary).write(reinterpret_cast<const char *>(buf.data()), buf.size());
}
std::vector<std::string> splitCsv(const std::string &line) {
    std::vector<std::string> out;
    std::string cur;
    bool quoted = false;
    for (char c : line) {
        if (c == '"') quoted = !quoted;
        else if (c == ',' && !quoted) { out.push_back(cur); cur.clear(); }
        else cur += c;
    }
    out.push_back(cur);
    return out;
}
std::string num(double v, int prec) {
    if (!std::isfinite(v)) return "";
    std::ostringstream o;
    o << std::fixed << std::setprecision(prec) << v;
    std::string s = o.str();
    if (s.find_first_not_of("-0.") == std::string::npos && s[0] == '-') s.erase(0, 1);
    return s;
}
std::string quote(const std::string &s) { return "\"" + s + "\""; }

class Csv {
public:
    Csv(const fs::path &p, const std::vector<std::string> &header) : out_(p, std::ios::binary) {
        out_ << "\xEF\xBB\xBF";
        row(header);
    }
    void row(const std::vector<std::string> &cells) {
        for (size_t i = 0; i < cells.size(); ++i) out_ << (i ? "," : "") << cells[i];
        out_ << "\n";
    }
private:
    std::ofstream out_;
};

// 슬라이드 문장에 쓰는 핵심 숫자 ("key: value" 한 줄씩)
struct Summary {
    std::vector<std::pair<std::string, std::string>> kv;
    void set(const std::string &k, double v, int prec = 4) { kv.emplace_back(k, num(v, prec)); }
    void text(const std::string &k, const std::string &v) { kv.emplace_back(k, quote(v)); }
    void write(const fs::path &p) const {
        std::ofstream o(p, std::ios::binary);
        o << "# 04_eda 핵심 숫자 (자동 생성)\n";
        for (const auto &[k, v] : kv) o << k << ": " << v << "\n";
    }
};

double mean(const std::vector<double> &v) {
    double s = 0;
    int n = 0;
    for (double x : v) if (std::isfinite(x)) { s += x; ++n; }
    return n ? s / n : kNaN;
}
double stdev(const std::vector<double> &v, int ddof = 1) {
    const double m = mean(v);
    double s = 0;
    int n = 0;
    for (double x : v) if (std::isfinite(x)) { s += (x - m) * (x - m); ++n; }
    return n > ddof ? std::sqrt(s / (n - ddof)) : kNaN;
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
    for (double x : v) if (std::isfinite(x)) m = std::max(m, std::fabs(x));
    return m;
}
double rms(const std::vector<double> &v) {
    double s = 0;
    int n = 0;
    for (double x : v) if (std::isfinite(x)) { s += x * x; ++n; }
    return n ? std::sqrt(s / n) : kNaN;
}

// 최소제곱: 설계 행렬 열(cols[j][i])로 y 를 맞춘다. 잔차 σ 는 자유도(n - 열 수)로 나눈다.
struct LsFit { std::vector<double> coef; double residSd = kNaN; std::vector<double> resid; };
LsFit leastSquares(const std::vector<std::vector<double>> &cols, const std::vector<double> &y) {
    const int n = static_cast<int>(y.size()), p = static_cast<int>(cols.size());
    cv::Mat A(n, p, CV_64F), b(n, 1, CV_64F), x;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < p; ++j) A.at<double>(i, j) = cols[j][i];
        b.at<double>(i) = y[i];
    }
    cv::solve(A, b, x, cv::DECOMP_SVD);
    LsFit f;
    for (int j = 0; j < p; ++j) f.coef.push_back(x.at<double>(j));
    double ss = 0;
    for (int i = 0; i < n; ++i) {
        double pred = 0;
        for (int j = 0; j < p; ++j) pred += f.coef[j] * cols[j][i];
        f.resid.push_back(y[i] - pred);
        ss += (y[i] - pred) * (y[i] - pred);
    }
    if (n > p) f.residSd = std::sqrt(ss / (n - p));
    return f;
}
// y = a + b x
LsFit fitLine(const std::vector<double> &x, const std::vector<double> &y) {
    return leastSquares({std::vector<double>(x.size(), 1.0), x}, y);
}
// y = a + b x + c x^2
LsFit fitQuad(const std::vector<double> &x, const std::vector<double> &y) {
    std::vector<double> x2;
    for (double v : x) x2.push_back(v * v);
    return leastSquares({std::vector<double>(x.size(), 1.0), x, x2}, y);
}

// -------------------------------------------------------------------------------------
// 2. 자료 목록 (clean 의 _metadata/manifest.csv) 과 촬영 시각 (원본 ZIP 중앙 디렉터리)
// -------------------------------------------------------------------------------------

struct Shot {
    std::string set;        // "20261014" / "20261020" / "20261029" ...
    fs::path file;
    std::string rel;        // clean 폴더 기준 경로 (UTF-8)
    std::string member;     // 원본 ZIP 멤버 이름 (촬영 시각 대조용)
    bool center = false;
    double y = 0, t = 0;    // 파일명 명령값
    int rep = 1;
    std::string group;      // 1차: trial 이름, 2차: 조합 이름 / "center"
    std::string labelFix;   // 파일명 명령이 폴더 이름과 달라 폴더 기준으로 고친 경우 원래 파일명 명령
    std::string takenAt;    // 촬영 시각 (현지)
    double minutes = kNaN;  // 그 자료의 첫 사진부터 지난 분
    // 측정값
    std::array<double, kQr> eccDx{}, eccDy{}, eccCc{};
    double eccDxMean = kNaN, eccDyMean = kNaN;
    std::array<double, kQr> qDx{}, qDy{};   // QR 별 모서리 평균 (대각선 제외)
    double dx = kNaN, dy = kNaN;            // 모서리 평균 (대각선 제외)
    int cornersOk = 0, cornersFound = 0;
    double plateMean = kNaN, wholeMean = kNaN, channelDiffMax = kNaN;
    std::vector<cv::Point2f> pos;           // 모서리마다 위치 (NaN = 맞춤 실패)
};

std::string trialOf(char axis, double v) {
    const std::string side = v > 0 ? "+쪽" : "-쪽";
    if (axis == 'T') return "Θ " + side;
    const long hundredths = std::lround(std::fabs(v) * 100);
    return (hundredths % 10 == 0 ? "Y 0.1 간격 " : "Y 0.05 홀수 ") + side;
}

std::vector<Shot> loadSet(const fs::path &cleanRoot, const std::string &date) {
    const fs::path dir = cleanRoot / ("clean_" + date);
    std::ifstream f(dir / "_metadata" / "manifest.csv", std::ios::binary);
    require(f.good(), "manifest.csv 없음: " + dir.u8string());
    std::string line;
    std::getline(f, line);
    if (line.rfind("\xEF\xBB\xBF", 0) == 0) line.erase(0, 3);
    const auto header = splitCsv(line);
    auto col = [&](const std::string &name) {
        auto it = std::find(header.begin(), header.end(), name);
        require(it != header.end(), "manifest 열 없음: " + name);
        return static_cast<size_t>(it - header.begin());
    };
    const size_t cRel = col("relative_path"), cMem = col("source_member"), cY = col("filename_Y_command"),
                 cT = col("filename_T_command"), cRep = col("capture_repeat"), cCond = col("condition");
    std::vector<Shot> shots;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        const auto c = splitCsv(line);
        Shot s;
        s.set = date;
        s.rel = c[cRel];
        s.member = c[cMem];
        s.file = dir / fs::u8path(s.rel);
        s.y = c[cY].empty() ? 0 : std::stod(c[cY]);
        s.t = c[cT].empty() ? 0 : std::stod(c[cT]);
        s.rep = c[cRep].empty() ? 1 : std::stoi(c[cRep]);
        // 파일명 명령과 촬영 폴더 이름("Y+0.05 T+0.32")이 다르면 폴더를 따른다 (20261029 의 T+0.32 폴더 안
        // "T+0.33 (3).bmp" 처럼 이름만 잘못 붙은 사진). 고친 사실은 label_fix 열과 summary 에 남긴다.
        {
            static const std::regex folderRe("^Y([+-][0-9.]+) T([+-][0-9.]+)$");
            const std::string folder = fs::u8path(s.rel).parent_path().filename().u8string();
            std::smatch m;
            if (std::regex_match(folder, m, folderRe)) {
                const double fy = std::stod(m[1].str()), ft = std::stod(m[2].str());
                if (std::fabs(fy - s.y) > 1e-9 || std::fabs(ft - s.t) > 1e-9) {
                    s.labelFix = "Y" + c[cY] + " T" + c[cT];
                    s.y = fy;
                    s.t = ft;
                }
            }
        }
        s.center = c[cCond] == "center" || (s.y == 0 && s.t == 0);
        if (s.center) s.group = "center";
        else if (date == "20261014") s.group = s.t != 0 ? trialOf('T', s.t) : trialOf('Y', s.y);
        else s.group = "Y" + num(s.y, 2).insert(0, s.y >= 0 ? "+" : "") + " T" + (s.t >= 0 ? "+" : "") + num(s.t, 2);
        shots.push_back(s);
    }
    require(!shots.empty(), "사진 없음: " + dir.u8string());
    return shots;
}

// 원본 ZIP(바깥) 안의 마지막 분할 조각(.zip)만 tar.exe 로 꺼내 중앙 디렉터리를 읽는다.
std::map<std::string, std::pair<std::string, double>> captureTimes(const fs::path &rawZip, const fs::path &scratch) {
    fs::remove_all(scratch);
    fs::create_directories(scratch);
    wchar_t sysDir[MAX_PATH];
    GetSystemDirectoryW(sysDir, MAX_PATH);
    std::wstring cmd = L"\"" + std::wstring(sysDir) + L"\\tar.exe\" -xf \"" + rawZip.wstring() + L"\" -C \"" +
                       scratch.wstring() + L"\" --include=*.zip";
    STARTUPINFOW si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    require(CreateProcessW(nullptr, &cmd[0], nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi),
            "tar.exe 실행 실패");
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    require(code == 0, "tar.exe 로 마지막 분할 조각을 꺼내지 못함: " + rawZip.u8string());
    fs::path last;
    for (const auto &e : fs::recursive_directory_iterator(scratch))
        if (e.is_regular_file() && e.path().extension() == L".zip") last = e.path();
    require(!last.empty(), "원본 ZIP 안에 마지막 분할 조각(.zip)이 없음");

    const auto d = readBytes(last);
    auto u16 = [&](size_t p) { return static_cast<uint32_t>(d[p] | (d[p + 1] << 8)); };
    auto u32 = [&](size_t p) { return u16(p) | (u16(p + 2) << 16); };
    auto u64 = [&](size_t p) { return static_cast<uint64_t>(u32(p)) | (static_cast<uint64_t>(u32(p + 4)) << 32); };
    // 끝에서 EOCD 를 찾는다
    size_t eocd = std::string::npos;
    for (size_t p = d.size() - 22; p + 1 > 0 && d.size() - p < 70000; --p)
        if (u32(p) == 0x06054b50) { eocd = p; break; }
    require(eocd != std::string::npos, "중앙 디렉터리 끝(EOCD)을 찾지 못함");
    uint64_t cdSize = u32(eocd + 12);
    size_t cdEnd = eocd;
    if (eocd >= 20 && u32(eocd - 20) == 0x07064b50) {   // ZIP64: 바로 앞 ZIP64 EOCD 레코드
        const size_t rec = eocd - 20 - 56;
        require(u32(rec) == 0x06064b50, "ZIP64 EOCD 레코드 위치가 예상과 다름");
        cdSize = u64(rec + 40);
        cdEnd = rec;
    }
    require(cdSize <= cdEnd, "중앙 디렉터리가 마지막 분할 조각에 다 들어 있지 않음");
    std::map<std::string, std::pair<std::string, double>> out;   // 멤버 이름 -> (시각 문자열, FILETIME 분)
    for (size_t p = cdEnd - static_cast<size_t>(cdSize); p < cdEnd;) {
        require(u32(p) == 0x02014b50, "중앙 디렉터리 항목 서명이 다름");
        const uint32_t flags = u16(p + 8), dosTime = u16(p + 12), dosDate = u16(p + 14);
        const size_t nameLen = u16(p + 28), extraLen = u16(p + 30), commentLen = u16(p + 32);
        const std::string raw(reinterpret_cast<const char *>(&d[p + 46]), nameLen);
        const std::string name = (flags & 0x800) ? raw : utf8(wide(raw, 949));
        FILETIME ft{};
        bool haveNtfs = false;
        for (size_t q = p + 46 + nameLen; q + 4 <= p + 46 + nameLen + extraLen;) {
            const uint32_t id = u16(q), len = u16(q + 2);
            if (id == 0x000a && len >= 32 && u16(q + 8) == 1 && u16(q + 10) >= 8) {
                const uint64_t m = u64(q + 12);
                ft.dwLowDateTime = static_cast<DWORD>(m);
                ft.dwHighDateTime = static_cast<DWORD>(m >> 32);
                haveNtfs = true;
            }
            q += 4 + len;
        }
        SYSTEMTIME local{};
        if (haveNtfs) {
            SYSTEMTIME utc{};
            FileTimeToSystemTime(&ft, &utc);
            SystemTimeToTzSpecificLocalTime(nullptr, &utc, &local);
        } else {   // DOS 시각 (2초 단위, 이미 현지 시각)
            local.wYear = static_cast<WORD>(1980 + (dosDate >> 9));
            local.wMonth = static_cast<WORD>((dosDate >> 5) & 15);
            local.wDay = static_cast<WORD>(dosDate & 31);
            local.wHour = static_cast<WORD>(dosTime >> 11);
            local.wMinute = static_cast<WORD>((dosTime >> 5) & 63);
            local.wSecond = static_cast<WORD>((dosTime & 31) * 2);
        }
        FILETIME lft{};
        SystemTimeToFileTime(&local, &lft);
        const double minutes = ((static_cast<uint64_t>(lft.dwHighDateTime) << 32) | lft.dwLowDateTime) / 6.0e8;
        std::ostringstream ts;
        ts << std::setfill('0') << local.wYear << '-' << std::setw(2) << local.wMonth << '-' << std::setw(2) << local.wDay << ' '
           << std::setw(2) << local.wHour << ':' << std::setw(2) << local.wMinute << ':' << std::setw(2) << local.wSecond;
        out[name] = {ts.str(), minutes};
        p += 46 + nameLen + extraLen + commentLen;
    }
    fs::remove_all(scratch);
    return out;
}

// -------------------------------------------------------------------------------------
// 3. 이미지 측정 (QR 라벨, ECC, 구조 모서리)
// -------------------------------------------------------------------------------------

struct QrRef {
    cv::Rect label, patch;
    std::string text;     // 디코딩 문자열
    int modules = 0;      // 한 변 모듈 수
};

cv::Mat readGray(const fs::path &p, double *channelDiffMax = nullptr) {
    const auto bytes = readBytes(p);
    cv::Mat color = cv::imdecode(bytes, cv::IMREAD_COLOR);
    require(!color.empty(), "이미지를 읽을 수 없음: " + p.u8string());
    std::vector<cv::Mat> ch;
    cv::split(color, ch);
    if (channelDiffMax) {
        double a = 0, b = 0;
        cv::Mat diff;
        cv::absdiff(ch[0], ch[1], diff);
        cv::minMaxLoc(diff, nullptr, &a);
        cv::absdiff(ch[0], ch[2], diff);
        cv::minMaxLoc(diff, nullptr, &b);
        *channelDiffMax = std::max(a, b);
    }
    return ch[0];   // R=G=B 흑백
}

std::array<QrRef, kQr> findQrLabels(const cv::Mat &gray) {
    const cv::Rect band(gray.cols / 4, gray.rows / 4, gray.cols / 2, gray.rows / 2);
    cv::Mat roi, bin, labels, stats, centroids;
    cv::GaussianBlur(gray(band), roi, cv::Size(0, 0), 1.0);
    cv::threshold(roi, bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    const int n = cv::connectedComponentsWithStats(bin, labels, stats, centroids, 8, CV_32S);
    std::vector<cv::Rect> cand;
    for (int i = 1; i < n; ++i) {
        const int w = stats.at<int>(i, cv::CC_STAT_WIDTH), h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        const double aspect = static_cast<double>(w) / h;
        if (w < 100 || h < 100 || w > 400 || h > 400 || aspect < 0.75 || aspect > 1.33) continue;
        if (stats.at<int>(i, cv::CC_STAT_AREA) < 0.5 * w * h) continue;
        cand.emplace_back(stats.at<int>(i, cv::CC_STAT_LEFT) + band.x, stats.at<int>(i, cv::CC_STAT_TOP) + band.y, w, h);
    }
    std::sort(cand.begin(), cand.end(), [](const cv::Rect &a, const cv::Rect &b) { return a.x < b.x; });
    std::array<QrRef, kQr> qrs{};
    double best = 1e9;
    for (size_t a = 0; a < cand.size(); ++a)
        for (size_t b = a + 1; b < cand.size(); ++b)
            for (size_t c = b + 1; c < cand.size(); ++c) {
                const double ya = cand[a].y + cand[a].height / 2.0, yb = cand[b].y + cand[b].height / 2.0, yc = cand[c].y + cand[c].height / 2.0;
                const double spread = std::max({ya, yb, yc}) - std::min({ya, yb, yc});
                const double sizeSpread = std::max({cand[a].area(), cand[b].area(), cand[c].area()}) /
                                          static_cast<double>(std::min({cand[a].area(), cand[b].area(), cand[c].area()}));
                const double score = spread + 100 * (sizeSpread - 1);
                if (score < best) { best = score; qrs[0].label = cand[a]; qrs[1].label = cand[b]; qrs[2].label = cand[c]; }
            }
    require(best <= 60, "center 에서 나란한 QR 라벨 3개를 찾지 못함");
    for (auto &q : qrs)
        q.patch = cv::Rect(q.label.x - kPatchMargin, q.label.y - kPatchMargin, q.label.width + 2 * kPatchMargin, q.label.height + 2 * kPatchMargin);
    return qrs;
}

struct EccMotion { double dx = kNaN, dy = kNaN, cc = kNaN; cv::Matx23d warp; bool ok = false; };

class EccMeasurer {
public:
    EccMeasurer(const cv::Mat &center, const std::array<QrRef, kQr> &qrs) : qrs_(qrs) {
        for (int q = 0; q < kQr; ++q) center(qrs[q].patch).convertTo(tmpl_[q], CV_32F);
    }
    EccMotion measure(const cv::Mat &gray, int q) const {
        EccMotion m;
        const cv::Rect patch = qrs_[q].patch;
        // 가로로 크게(±320px = Y 1mm + Θ 2 정도), 세로로 작게 찾는다
        const int padX = 320, padY = 60;
        cv::Rect search(patch.x - padX, patch.y - padY, patch.width + 2 * padX, patch.height + 2 * padY);
        search &= cv::Rect(0, 0, gray.cols, gray.rows);
        cv::Mat searchF, score;
        gray(search).convertTo(searchF, CV_32F);
        cv::matchTemplate(searchF, tmpl_[q], score, cv::TM_CCOEFF_NORMED);
        cv::Point loc;
        cv::minMaxLoc(score, nullptr, nullptr, nullptr, &loc);
        const int ox = search.x + loc.x - patch.x, oy = search.y + loc.y - patch.y;
        const cv::Rect target(patch.x + ox, patch.y + oy, patch.width, patch.height);
        if ((target & cv::Rect(0, 0, gray.cols, gray.rows)) != target) return m;
        cv::Mat targetF;
        gray(target).convertTo(targetF, CV_32F);
        cv::Mat W = cv::Mat::eye(2, 3, CV_32F);
        try {
            m.cc = cv::findTransformECC(tmpl_[q], targetF, W, cv::MOTION_AFFINE,
                                        cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 200, 1e-8), cv::noArray(), 5);
        } catch (const cv::Exception &) {
            return m;
        }
        const double cu = (patch.width - 1) / 2.0, cv_ = (patch.height - 1) / 2.0;
        m.dx = ox + W.at<float>(0, 0) * cu + W.at<float>(0, 1) * cv_ + W.at<float>(0, 2) - cu;
        m.dy = oy + W.at<float>(1, 0) * cu + W.at<float>(1, 1) * cv_ + W.at<float>(1, 2) - cv_;
        m.warp = cv::Matx23d(W.at<float>(0, 0), W.at<float>(0, 1), W.at<float>(0, 2) + patch.x + ox,
                             W.at<float>(1, 0), W.at<float>(1, 1), W.at<float>(1, 2) + patch.y + oy);
        m.ok = true;
        return m;
    }
private:
    std::array<cv::Mat, kQr> tmpl_;
    std::array<QrRef, kQr> qrs_;
};

struct Corner {
    cv::Point2f p;              // 기준 center 에서의 위치
    int qr = 0, gi = 0, gj = 0; // QR, 격자 꼭짓점 (열, 행)
    bool diagonal = false;      // 검정 두 칸이 꼭짓점만 맞닿음
    bool centerOk = false;      // 기준 center 에서 통과 기준을 만족
    std::array<int, 4> black{}; // 왼위, 오른위, 왼아래, 오른아래
};

struct CornerFit { cv::Point2d p; double rmse = 0, contrast = 0, blur = 0; bool converged = false, ok = false; std::string reason; };

// 흐린 2x2 경계 모델 (x, y, σx, σy, low, contrast) 을 창 안 픽셀에 soft-L1 최소제곱으로 맞춘다.
struct FitResult { double p[6]; double rmse; bool converged; };
FitResult fitWindow(const cv::Mat &gray, const Corner &c, cv::Point2d center, cv::Point2d start, int radius) {
    const int x0 = cvRound(center.x), y0 = cvRound(center.y);
    std::vector<double> X, Y, I;
    for (int y = y0 - radius; y <= y0 + radius; ++y)
        for (int x = x0 - radius; x <= x0 + radius; ++x)
            if (x >= 0 && y >= 0 && x < gray.cols && y < gray.rows) { X.push_back(x); Y.push_back(y); I.push_back(gray.at<uchar>(y, x)); }
    const int n = static_cast<int>(I.size());
    std::vector<double> sorted = I;
    std::sort(sorted.begin(), sorted.end());
    const double low0 = sorted[n / 10], high0 = sorted[n * 9 / 10];
    double p[6] = {start.x, start.y, 1.0, 1.0, low0, std::max(high0 - low0, 20.0)};
    const double lo[6] = {start.x - 3.8, start.y - 3.8, 0.35, 0.35, 0, 10};
    const double hi[6] = {start.x + 3.8, start.y + 3.8, 2.7, 2.7, 240, 255};
    const double tl = c.black[0], tr = c.black[1], bl = c.black[2], br = c.black[3];
    const double fScale = 6.0, kInvSqrt2 = 0.70710678118654752, kInvSqrt2Pi = 0.39894228040143268;
    auto residuals = [&](const double *q, std::vector<double> &r, cv::Mat *J) {
        for (int k = 0; k < n; ++k) {
            const double a = X[k] - q[0], b = Y[k] - q[1];
            const double U = 0.5 * (1 + std::erf(a / q[2] * kInvSqrt2)), V = 0.5 * (1 + std::erf(b / q[3] * kInvSqrt2));
            const double B = tl * (1 - U) * (1 - V) + tr * U * (1 - V) + bl * (1 - U) * V + br * U * V;
            r[k] = q[4] + q[5] * (1 - B) - I[k];
            if (!J) continue;
            const double gU = kInvSqrt2Pi / q[2] * std::exp(-0.5 * a * a / (q[2] * q[2]));
            const double gV = kInvSqrt2Pi / q[3] * std::exp(-0.5 * b * b / (q[3] * q[3]));
            const double dBdU = (tr - tl) * (1 - V) + (br - bl) * V, dBdV = (bl - tl) * (1 - U) + (br - tr) * U;
            double *row = J->ptr<double>(k);
            row[0] = q[5] * dBdU * gU;
            row[1] = q[5] * dBdV * gV;
            row[2] = q[5] * dBdU * gU * a / q[2];
            row[3] = q[5] * dBdV * gV * b / q[3];
            row[4] = 1;
            row[5] = 1 - B;
        }
    };
    auto cost = [&](const std::vector<double> &res) {
        double s = 0;
        for (double v : res) s += 2 * fScale * fScale * (std::sqrt(1 + (v / fScale) * (v / fScale)) - 1);
        return s;
    };
    std::vector<double> r(n), rNew(n);
    cv::Mat J(n, 6, CV_64F);
    residuals(p, r, &J);
    double lambda = 1e-3, cur = cost(r), lastChange = 1e9;
    bool converged = false;
    for (int it = 0; it < 80 && !converged; ++it) {
        cv::Matx66d A = cv::Matx66d::zeros();
        cv::Vec6d g = cv::Vec6d::all(0);
        for (int k = 0; k < n; ++k) {
            const double w = 1.0 / std::sqrt(1 + (r[k] / fScale) * (r[k] / fScale));
            const double *row = J.ptr<double>(k);
            for (int a = 0; a < 6; ++a) {
                g[a] += w * row[a] * r[k];
                for (int b = 0; b < 6; ++b) A(a, b) += w * row[a] * row[b];
            }
        }
        for (int a = 0; a < 6; ++a) {   // 범위 끝에서 바깥으로 가려는 매개변수는 고정
            const bool atLo = p[a] <= lo[a] + 1e-12 && g[a] > 0, atHi = p[a] >= hi[a] - 1e-12 && g[a] < 0;
            if (!atLo && !atHi) continue;
            for (int b = 0; b < 6; ++b) { A(a, b) = 0; A(b, a) = 0; }
            A(a, a) = 1;
            g[a] = 0;
        }
        bool improved = false;
        for (int tries = 0; tries < 8 && !improved; ++tries) {
            cv::Matx66d Ad = A;
            for (int a = 0; a < 6; ++a) Ad(a, a) *= (1 + lambda);
            cv::Vec6d step;
            cv::solve(Ad, -g, step, cv::DECOMP_SVD);
            double q[6], change2 = 0, norm2 = 0;
            for (int a = 0; a < 6; ++a) {
                q[a] = std::min(hi[a], std::max(lo[a], p[a] + step[a]));
                change2 += (q[a] - p[a]) * (q[a] - p[a]);
                norm2 += p[a] * p[a];
            }
            residuals(q, rNew, nullptr);
            const double nc = cost(rNew);
            if (nc <= cur) {
                improved = true;
                std::copy(q, q + 6, p);
                const double rel = (cur - nc) / std::max(cur, 1e-12);
                cur = nc;
                lambda = std::max(lambda * 0.3, 1e-7);
                residuals(p, r, &J);
                lastChange = std::sqrt(change2);
                if (lastChange < 1e-7 * (1e-7 + std::sqrt(norm2)) || rel < 1e-7) converged = true;
            } else {
                lambda *= 10;
            }
        }
        if (!improved) converged = true;
    }
    if (!converged && lastChange < 1e-3) converged = true;
    FitResult fr;
    std::copy(p, p + 6, fr.p);
    double ss = 0;
    for (double v : r) ss += v * v;
    fr.rmse = std::sqrt(ss / n);
    fr.converged = converged;
    return fr;
}

// 9x9 창으로 맞추고 맞춘 위치로 창을 옮겨 다시 맞춘다(최대 3번). 11x11 창은 안정성 확인용.
CornerFit fitCorner(const cv::Mat &gray, const Corner &c, cv::Point2d start) {
    cv::Point2d est = start;
    FitResult f1{};
    for (int k = 0; k < 3; ++k) {
        f1 = fitWindow(gray, c, est, start, 4);
        const cv::Point2d got(f1.p[0], f1.p[1]);
        const bool moved = cvRound(got.x) != cvRound(est.x) || cvRound(got.y) != cvRound(est.y);
        est = got;
        if (!moved) break;
    }
    const FitResult f2 = fitWindow(gray, c, est, start, 5);
    CornerFit out;
    out.p = cv::Point2d(f1.p[0], f1.p[1]);
    out.rmse = f1.rmse;
    out.contrast = f1.p[5];
    out.blur = std::max(f1.p[2], f1.p[3]);
    out.converged = f1.converged;
    const double stability = cv::norm(out.p - cv::Point2d(f2.p[0], f2.p[1]));
    if (!f1.converged || !f2.converged) out.reason = "최적화 미수렴";
    else if (out.rmse > 15) out.reason = "밝기 모델 오차 큼";
    else if (out.contrast < 45) out.reason = "대비 낮음";
    else if (stability > 0.65) out.reason = "창 크기에 따라 위치가 달라짐";
    else if (std::fabs(out.p.x - start.x) > 3.6 || std::fabs(out.p.y - start.y) > 3.6) out.reason = "시작점에서 멂";
    else if (out.blur > 2.6) out.reason = "흐림 큼";
    out.ok = out.reason.empty();
    return out;
}

// QR 을 디코딩해 모듈 격자를 얻고, 검정/흰 칸이 만나는 꼭짓점을 구조 ID 로 고정한다.
std::vector<Corner> findStructuralCorners(const cv::Mat &gray, std::array<QrRef, kQr> &qrs, std::map<std::string, int> &dropped) {
    std::vector<Corner> out;
    cv::QRCodeDetector det;
    for (int q = 0; q < kQr; ++q) {
        // 검출기는 라벨 바깥 여백이 넉넉해야 잘 잡는다. 여백과 배율을 바꿔 가며 처음 성공한 것을 쓴다.
        cv::Mat straight;
        std::vector<cv::Point2f> pts;
        for (int pad : {60, 100, 140, 180}) {
            for (int scale : {1, 2}) {
                const cv::Rect roi = cv::Rect(qrs[q].label.x - pad, qrs[q].label.y - pad, qrs[q].label.width + 2 * pad, qrs[q].label.height + 2 * pad) &
                                     cv::Rect(0, 0, gray.cols, gray.rows);
                cv::Mat crop;
                cv::resize(gray(roi), crop, cv::Size(), scale, scale, cv::INTER_CUBIC);
                std::vector<cv::Point2f> p;
                if (!det.detect(crop, p) || p.size() != 4) continue;
                cv::Mat s;
                const std::string text = det.decode(crop, p, s);
                if (text.empty() || s.empty()) continue;
                qrs[q].text = text;
                straight = s;
                for (auto &v : p) pts.push_back(v * (1.0f / scale) + cv::Point2f(static_cast<float>(roi.x), static_cast<float>(roi.y)));
                break;
            }
            if (!straight.empty()) break;
        }
        require(!straight.empty(), std::string("QR 검출/디코딩 실패: ") + kQrName[q]);
        const int N = straight.cols;
        require(straight.rows == N && N >= 21 && (N - 17) % 4 == 0, "QR 모듈 격자 크기가 예상과 다름: " + std::to_string(N));
        qrs[q].modules = N;

        // 모듈 (열 c, 행 r) -> 사진. 검출 꼭짓점 순서가 격자 방향과 다를 수 있어 8가지 대칭 중
        // 모듈 가운데 밝기와 가장 잘 맞는 것을 고른다.
        cv::Mat bin;
        const double thr = cv::threshold(gray(qrs[q].label), bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        const std::vector<cv::Point2f> unit = {{0.f, 0.f}, {static_cast<float>(N), 0.f}, {static_cast<float>(N), static_cast<float>(N)}, {0.f, static_cast<float>(N)}};
        cv::Mat bestH;
        cv::Mat M;   // 1 = 검정
        int bestMatch = -1;
        for (int rot = 0; rot < 4; ++rot)
            for (int flip = 0; flip < 2; ++flip) {
                cv::Mat s = straight.clone();
                for (int k = 0; k < rot; ++k) cv::rotate(s, s, cv::ROTATE_90_CLOCKWISE);
                if (flip) cv::flip(s, s, 1);
                cv::Mat H = cv::getPerspectiveTransform(unit, pts);
                for (int inv = 0; inv < 2; ++inv) {
                    int match = 0;
                    std::vector<cv::Point2f> src, dst;
                    for (int r = 0; r < N; ++r)
                        for (int c = 0; c < N; ++c) src.emplace_back(c + 0.5f, r + 0.5f);
                    cv::perspectiveTransform(src, dst, H);
                    for (int k = 0; k < N * N; ++k) {
                        const bool dark = gray.at<uchar>(cvRound(dst[k].y), cvRound(dst[k].x)) < thr;
                        const bool black = (s.at<uchar>(k / N, k % N) < 128) != (inv == 1);
                        match += dark == black;
                    }
                    if (match > bestMatch) {
                        bestMatch = match;
                        bestH = H;
                        M = cv::Mat(N, N, CV_8U);
                        for (int k = 0; k < N * N; ++k) M.at<uchar>(k / N, k % N) = (s.at<uchar>(k / N, k % N) < 128) != (inv == 1);
                    }
                }
            }
        require(bestMatch > 0.97 * N * N, std::string("QR 격자와 사진 밝기가 맞지 않음: ") + kQrName[q]);

        auto blackAt = [&](int r, int c) { return (r < 0 || c < 0 || r >= N || c >= N) ? 0 : static_cast<int>(M.at<uchar>(r, c)); };
        std::vector<Corner> qc;
        for (int j = 0; j <= N; ++j)
            for (int i = 0; i <= N; ++i) {
                Corner c;
                c.qr = q;
                c.gi = i;
                c.gj = j;
                c.black = {blackAt(j - 1, i - 1), blackAt(j - 1, i), blackAt(j, i - 1), blackAt(j, i)};
                const int count = c.black[0] + c.black[1] + c.black[2] + c.black[3];
                if (count == 1 || count == 3) c.diagonal = false;
                else if (count == 2 && c.black[0] == c.black[3]) c.diagonal = true;
                else continue;
                qc.push_back(c);
            }
        // 투영 위치에서 맞추고, 통과한 점으로 격자 투영을 다시 잡아 한 번 더 맞춘다
        auto project = [&](const cv::Mat &H) {
            std::vector<cv::Point2f> src, dst;
            for (const auto &c : qc) src.emplace_back(static_cast<float>(c.gi), static_cast<float>(c.gj));
            cv::perspectiveTransform(src, dst, H);
            return dst;
        };
        cv::Mat H = bestH;
        for (int pass = 0; pass < 2; ++pass) {
            const auto start = project(H);
            std::vector<cv::Point2f> gridOk, imgOk;
            for (size_t k = 0; k < qc.size(); ++k) {
                const CornerFit f = fitCorner(gray, qc[k], start[k]);
                qc[k].p = f.converged ? cv::Point2f(static_cast<float>(f.p.x), static_cast<float>(f.p.y)) : start[k];
                qc[k].centerOk = f.ok;
                if (pass == 1 && !f.ok) ++dropped[f.reason];
                if (f.ok && !qc[k].diagonal) { gridOk.emplace_back(static_cast<float>(qc[k].gi), static_cast<float>(qc[k].gj)); imgOk.push_back(qc[k].p); }
            }
            if (pass == 0) H = cv::findHomography(gridOk, imgOk, 0);
        }
        out.insert(out.end(), qc.begin(), qc.end());
    }
    return out;
}

// -------------------------------------------------------------------------------------
// 4. 사진 한 장 측정
// -------------------------------------------------------------------------------------

struct Reference {
    cv::Mat gray;
    std::array<QrRef, kQr> qrs;
    std::vector<Corner> corners;
    std::unique_ptr<EccMeasurer> ecc;
};

void measureShot(const Reference &ref, Shot &s) {
    const cv::Mat gray = readGray(s.file, &s.channelDiffMax);
    require(gray.cols == 3840 && gray.rows == 2160, "크기가 3840x2160 이 아님: " + s.rel);
    s.wholeMean = cv::mean(gray)[0];
    std::array<EccMotion, kQr> em;
    double plate = 0;
    for (int q = 0; q < kQr; ++q) {
        em[q] = ref.ecc->measure(gray, q);
        s.eccDx[q] = em[q].dx;
        s.eccDy[q] = em[q].dy;
        s.eccCc[q] = em[q].cc;
        // 라벨 밝기: 정수 이동만큼 옮긴 라벨 안 평균
        cv::Rect lab = ref.qrs[q].label + cv::Point(cvRound(std::isfinite(em[q].dx) ? em[q].dx : 0), cvRound(std::isfinite(em[q].dy) ? em[q].dy : 0));
        lab &= cv::Rect(0, 0, gray.cols, gray.rows);
        plate += cv::mean(gray(lab))[0] / kQr;
    }
    s.plateMean = plate;
    s.eccDxMean = (s.eccDx[0] + s.eccDx[1] + s.eccDx[2]) / 3;
    s.eccDyMean = (s.eccDy[0] + s.eccDy[1] + s.eccDy[2]) / 3;

    const float nanf = std::numeric_limits<float>::quiet_NaN();
    s.pos.assign(ref.corners.size(), cv::Point2f(nanf, nanf));
    std::array<double, kQr> sx{}, sy{};
    std::array<int, kQr> sn{};
    for (size_t k = 0; k < ref.corners.size(); ++k) {
        const Corner &c = ref.corners[k];
        if (!em[c.qr].ok) continue;
        const cv::Point2d local(c.p.x - ref.qrs[c.qr].patch.x, c.p.y - ref.qrs[c.qr].patch.y);
        const cv::Matx23d &W = em[c.qr].warp;
        const cv::Point2d pred(W(0, 0) * local.x + W(0, 1) * local.y + W(0, 2), W(1, 0) * local.x + W(1, 1) * local.y + W(1, 2));
        const CornerFit f = fitCorner(gray, c, pred);
        if (!f.converged) continue;
        ++s.cornersFound;
        s.cornersOk += f.ok;
        s.pos[k] = cv::Point2f(static_cast<float>(f.p.x), static_cast<float>(f.p.y));
        if (c.diagonal) continue;
        sx[c.qr] += f.p.x - c.p.x;
        sy[c.qr] += f.p.y - c.p.y;
        ++sn[c.qr];
    }
    double tx = 0, ty = 0;
    int tn = 0;
    for (int q = 0; q < kQr; ++q) {
        s.qDx[q] = sn[q] ? sx[q] / sn[q] : kNaN;
        s.qDy[q] = sn[q] ? sy[q] / sn[q] : kNaN;
        tx += sx[q];
        ty += sy[q];
        tn += sn[q];
    }
    s.dx = tn ? tx / tn : kNaN;
    s.dy = tn ? ty / tn : kNaN;
}

// -------------------------------------------------------------------------------------
// 5. 분석
// -------------------------------------------------------------------------------------

// Y 와 Θ 를 가를 단서로 볼 특징: 평균 가로/세로 이동, QR 줄 늘어남(R-L 가로), QR 줄 기울어짐(R-L 세로)
constexpr int kFeat = 4;
const char *const kFeatName[kFeat] = {"dx", "dy", "stretch", "tilt"};
std::array<double, kFeat> features(const Shot &s) {
    return {s.dx, s.dy, s.qDx[2] - s.qDx[0], s.qDy[2] - s.qDy[0]};
}

}  // namespace

int wmain(int argc, wchar_t **argv) {
    try {
        SetConsoleOutputCP(CP_UTF8);
        cv::setNumThreads(1);
        fs::path repo = fs::current_path(), out;
        std::string target = "20261020";
        int threads = std::max(1u, std::thread::hardware_concurrency());
        for (int i = 1; i < argc; ++i) {
            const std::wstring a = argv[i];
            if (a == L"--repo-root" && i + 1 < argc) repo = argv[++i];
            else if (a == L"--out" && i + 1 < argc) out = argv[++i];
            else if (a == L"--threads" && i + 1 < argc) threads = std::stoi(argv[++i]);
            else if (a == L"--target" && i + 1 < argc) target = utf8(argv[++i]);
            else {
                std::cout << "사용법: 04_eda [--target 20261020|20261029] [--repo-root 저장소] [--out 새 폴더] [--threads N]\n"
                             "  기본 출력: build/eda_<target> (이미 있으면 오류, 덮어쓰지 않음)\n";
                return a == L"--help" ? 0 : 1;
            }
        }
        const fs::path cleanRoot = repo / "data" / "clean";
        require(target != "20261014", "--target 은 1차(20261014)가 아닌 날짜");
        require(fs::exists(cleanRoot / "clean_20261014") && fs::exists(cleanRoot / ("clean_" + target)),
                "저장소 루트에서 실행하거나 --repo-root 를 주세요 (data/clean/clean_20261014, clean_" + target + " 필요)");
        std::vector<std::string> dates = {"20261014", target};
        if (target != "20261020" && fs::exists(cleanRoot / "clean_20261020")) dates.push_back("20261020");   // 같은 세션 비교용
        if (out.empty()) out = repo / "build" / ("eda_" + target);
        require(!fs::exists(out), "출력 폴더가 이미 있음 (새 폴더를 주세요): " + out.u8string());
        fs::create_directories(out / "crop");
        Summary sum;

        // ---- 자료와 촬영 시각
        std::vector<Shot> shots;
        for (const auto &date : dates) {
            auto sd = loadSet(cleanRoot, date);
            shots.insert(shots.end(), sd.begin(), sd.end());
        }
        // 분 = 1차는 1차 첫 사진부터, 나머지(같은 세션으로 보는 2차 이후)는 그중 가장 이른 사진부터
        double first1 = 1e300, firstN = 1e300;
        for (const auto &date : dates) {
            const auto times = captureTimes(repo / "data" / "raw" / ("zip_" + date + ".zip"), out / ("_zip_" + date));
            for (auto &s : shots) {
                if (s.set != date) continue;
                auto it = times.find(s.member);
                require(it != times.end(), "원본 ZIP 에 촬영 시각이 없음: " + s.member);
                s.takenAt = it->second.first;
                s.minutes = it->second.second;
                (date == "20261014" ? first1 : firstN) = std::min(date == "20261014" ? first1 : firstN, s.minutes);
            }
        }
        for (auto &s : shots) s.minutes -= s.set == "20261014" ? first1 : firstN;
        std::stable_sort(shots.begin(), shots.end(), [](const Shot &a, const Shot &b) {
            return a.set != b.set ? a.set < b.set : a.minutes < b.minutes;
        });
        std::cout << "사진 " << shots.size() << "장 (촬영 시각: 원본 ZIP 중앙 디렉터리)\n";

        // ---- 기준 center (1차 center.bmp)
        Reference ref;
        const auto refIt = std::find_if(shots.begin(), shots.end(), [](const Shot &s) { return s.set == "20261014" && s.center; });
        require(refIt != shots.end(), "clean_20261014/center.bmp 없음");
        ref.gray = readGray(refIt->file);
        ref.qrs = findQrLabels(ref.gray);
        std::map<std::string, int> dropped;
        ref.corners = findStructuralCorners(ref.gray, ref.qrs, dropped);
        ref.ecc = std::make_unique<EccMeasurer>(ref.gray, ref.qrs);
        std::array<int, kQr> perQr{}, perQrUsed{};
        int diag = 0, centerOk = 0;
        for (const auto &c : ref.corners) {
            ++perQr[c.qr];
            if (c.diagonal) ++diag;
            else ++perQrUsed[c.qr];
            centerOk += c.centerOk;
        }
        std::cout << "구조 모서리 " << ref.corners.size() << "점 (대각선 " << diag << ", center 통과 " << centerOk << ")\n";
        sum.set("corner.total", static_cast<double>(ref.corners.size()), 0);
        sum.set("corner.diagonal", diag, 0);
        sum.set("corner.used", static_cast<double>(ref.corners.size()) - diag, 0);
        sum.set("corner.center_ok", centerOk, 0);
        for (const auto &[reason, n] : dropped) sum.set("corner.center_fail." + reason, n, 0);
        {
            Csv qc(out / "qr.csv", {"qr", "text", "modules", "label_x", "label_y", "label_w", "label_h", "corners", "corners_used"});
            for (int q = 0; q < kQr; ++q) {
                const auto &l = ref.qrs[q].label;
                qc.row({kQrName[q], quote(ref.qrs[q].text), std::to_string(ref.qrs[q].modules), std::to_string(l.x), std::to_string(l.y),
                        std::to_string(l.width), std::to_string(l.height), std::to_string(perQr[q]), std::to_string(perQrUsed[q])});
                sum.text(std::string("qr.") + kQrName[q] + ".text", ref.qrs[q].text);
                sum.set(std::string("qr.") + kQrName[q] + ".corners", perQr[q], 0);
                sum.set(std::string("qr.") + kQrName[q] + ".corners_used", perQrUsed[q], 0);
            }
            Csv cc(out / "corners_ref.csv", {"id", "qr", "gi", "gj", "x", "y", "diagonal", "center_ok"});
            for (size_t k = 0; k < ref.corners.size(); ++k) {
                const auto &c = ref.corners[k];
                cc.row({std::to_string(k), kQrName[c.qr], std::to_string(c.gi), std::to_string(c.gj), num(c.p.x, 4), num(c.p.y, 4),
                        std::to_string(c.diagonal), std::to_string(c.centerOk)});
            }
        }

        // ---- 사진 전부 측정 (여러 스레드)
        {
            std::atomic<size_t> next{0}, done{0};
            std::mutex errMu;
            std::string err;
            std::vector<std::thread> pool;
            for (int w = 0; w < threads; ++w)
                pool.emplace_back([&] {
                    for (size_t i; (i = next++) < shots.size();) {
                        try {
                            measureShot(ref, shots[i]);
                        } catch (const std::exception &e) {
                            std::lock_guard<std::mutex> lk(errMu);
                            err = shots[i].rel + ": " + e.what();
                        }
                        const size_t d = ++done;
                        if (d % 20 == 0) { std::lock_guard<std::mutex> lk(errMu); std::cout << "측정 " << d << " / " << shots.size() << std::endl; }
                    }
                });
            for (auto &t : pool) t.join();
            require(err.empty(), err);
        }

        // ---- 사진별 결과
        {
            int fixes = 0;
            std::string fixList;
            for (const auto &s : shots)
                if (!s.labelFix.empty()) { ++fixes; fixList += (fixList.empty() ? "" : "; ") + s.rel + " (파일명 " + s.labelFix + ")"; }
            sum.set("label.fixed", fixes, 0);
            if (fixes) sum.text("label.fixed_files", fixList);
            Csv sc(out / "shots.csv", {"set", "file", "group", "center", "y", "t", "rep", "taken_at", "minutes", "dx", "dy",
                                       "dx_L", "dx_C", "dx_R", "dy_L", "dy_C", "dy_R", "ecc_dx", "ecc_dy", "ecc_dx_L", "ecc_dx_C", "ecc_dx_R",
                                       "ecc_cc_L", "ecc_cc_C", "ecc_cc_R", "corners_found", "corners_ok", "plate_mean", "whole_mean", "channel_diff_max", "label_fix"});
            for (const auto &s : shots)
                sc.row({s.set, quote(s.rel), quote(s.group), std::to_string(s.center), num(s.y, 2), num(s.t, 2), std::to_string(s.rep), s.takenAt,
                        num(s.minutes, 4), num(s.dx, 4), num(s.dy, 4), num(s.qDx[0], 4), num(s.qDx[1], 4), num(s.qDx[2], 4),
                        num(s.qDy[0], 4), num(s.qDy[1], 4), num(s.qDy[2], 4), num(s.eccDxMean, 4), num(s.eccDyMean, 4),
                        num(s.eccDx[0], 4), num(s.eccDx[1], 4), num(s.eccDx[2], 4), num(s.eccCc[0], 5), num(s.eccCc[1], 5), num(s.eccCc[2], 5),
                        std::to_string(s.cornersFound), std::to_string(s.cornersOk), num(s.plateMean, 3), num(s.wholeMean, 3), num(s.channelDiffMax, 0), quote(s.labelFix)});
        }
        std::vector<size_t> set1, set2, centers2;
        for (size_t i = 0; i < shots.size(); ++i) {
            if (shots[i].set == "20261014") { if (!shots[i].center) set1.push_back(i); }
            else if (shots[i].center) centers2.push_back(i);   // 세션 center (대상에 없으면 같은 세션 20261020 의 것)
            else if (shots[i].set == target) set2.push_back(i);
        }
        require(!set2.empty(), "대상 자료에 center 가 아닌 사진이 없음");
        sum.text("target", target);
        sum.set("target.shots", static_cast<double>(set2.size()), 0);
        const size_t refIdx = static_cast<size_t>(std::find_if(shots.begin(), shots.end(), [](const Shot &s) { return s.set == "20261014" && s.center; }) - shots.begin());

        // ---- 형식, 시간, 품질
        {
            double chMax = 0, cc2Min = 1, okMin2 = 1e9, okMax2 = 0, foundMin = 1e9;
            std::vector<double> cc2;
            for (size_t i : set2) {
                chMax = std::max(chMax, shots[i].channelDiffMax);
                for (int q = 0; q < kQr; ++q) { cc2Min = std::min(cc2Min, shots[i].eccCc[q]); cc2.push_back(shots[i].eccCc[q]); }
                okMin2 = std::min(okMin2, static_cast<double>(shots[i].cornersOk));
                okMax2 = std::max(okMax2, static_cast<double>(shots[i].cornersOk));
                foundMin = std::min(foundMin, static_cast<double>(shots[i].cornersFound));
            }
            for (size_t i : centers2) chMax = std::max(chMax, shots[i].channelDiffMax);
            sum.set("set2.channel_diff_max", chMax, 0);
            sum.set("set2.ecc_cc_min", cc2Min, 5);
            sum.set("set2.ecc_cc_median", median(cc2), 5);
            sum.set("set2.corners_ok_min", okMin2, 0);
            sum.set("set2.corners_ok_max", okMax2, 0);
            sum.set("set2.corners_found_min", foundMin, 0);
            double end2 = 0;
            for (size_t i : set2) end2 = std::max(end2, shots[i].minutes);
            for (size_t i : centers2) end2 = std::max(end2, shots[i].minutes);
            sum.set("set2.minutes_total", end2, 2);
            double end1 = 0;
            for (size_t i : set1) end1 = std::max(end1, shots[i].minutes);
            sum.set("set1.minutes_total", end1, 2);
            sum.text("set2.first_taken", shots[centers2.empty() ? set2.front() : std::min(set2.front(), centers2.front())].takenAt);
            // 반복 3장 사이 간격
            std::vector<double> gaps;
            std::map<std::string, std::vector<size_t>> byGroup;
            for (size_t i : set2) byGroup[shots[i].group].push_back(i);
            for (auto &[g, idx] : byGroup)
                for (size_t k = 1; k < idx.size(); ++k) gaps.push_back((shots[idx[k]].minutes - shots[idx[k - 1]].minutes) * 60);
            sum.set("set2.rep_gap_sec_min", *std::min_element(gaps.begin(), gaps.end()), 1);
            sum.set("set2.rep_gap_sec_median", median(gaps), 1);
            sum.set("set2.rep_gap_sec_max", *std::max_element(gaps.begin(), gaps.end()), 1);
        }

        // ---- 1차: trial 별 직선 (기울기, 0점), 사진 공통 흔들림, 점 흔들림
        std::map<std::string, std::vector<size_t>> trials;
        for (size_t i : set1) trials[shots[i].group].push_back(i);
        const size_t nC = ref.corners.size();
        std::vector<double> commonAll, pointSdX, pointSdY;
        {
            Csv tc(out / "set1_trials.csv", {"trial", "shots", "slope", "zero", "resid_sd", "slope_ecc", "zero_ecc", "resid_sd_ecc", "common_sd", "dy_per_dx"});
            for (auto &[name, idx] : trials) {
                std::vector<double> x, y, ye, dxs, dys;
                for (size_t i : idx) {
                    x.push_back(shots[i].t != 0 ? shots[i].t : shots[i].y);
                    y.push_back(shots[i].dx);
                    ye.push_back(shots[i].eccDxMean);
                    dxs.push_back(shots[i].dx);
                    dys.push_back(shots[i].dy);
                }
                const LsFit f = fitLine(x, y), fe = fitLine(x, ye), fv = fitLine(dxs, dys);
                // 점마다 2차 곡선을 뺀 잔차 -> 사진 공통(평균), 점 고유(공통을 뺀 σ)
                std::vector<double> com(idx.size(), 0.0), comY(idx.size(), 0.0);
                std::vector<std::vector<double>> rx(nC), ry(nC);
                int used = 0;
                for (size_t k = 0; k < nC; ++k) {
                    if (ref.corners[k].diagonal) continue;
                    std::vector<double> px, py;
                    bool all = true;
                    for (size_t i : idx) {
                        all = all && std::isfinite(shots[i].pos[k].x);
                        px.push_back(shots[i].pos[k].x - ref.corners[k].p.x);
                        py.push_back(shots[i].pos[k].y - ref.corners[k].p.y);
                    }
                    if (!all) continue;
                    rx[k] = fitQuad(x, px).resid;
                    ry[k] = fitQuad(x, py).resid;
                    for (size_t j = 0; j < idx.size(); ++j) { com[j] += rx[k][j]; comY[j] += ry[k][j]; }
                    ++used;
                }
                for (size_t j = 0; j < idx.size(); ++j) { com[j] /= used; comY[j] /= used; }
                for (size_t k = 0; k < nC; ++k) {
                    if (rx[k].empty()) continue;
                    std::vector<double> ex, ey;
                    for (size_t j = 0; j < idx.size(); ++j) { ex.push_back(rx[k][j] - com[j]); ey.push_back(ry[k][j] - comY[j]); }
                    pointSdX.push_back(stdev(ex));
                    pointSdY.push_back(stdev(ey));
                }
                const double comSd = stdev(com);
                commonAll.push_back(comSd);
                tc.row({quote(name), std::to_string(idx.size()), num(f.coef[1], 4), num(f.coef[0], 4), num(f.residSd, 4),
                        num(fe.coef[1], 4), num(fe.coef[0], 4), num(fe.residSd, 4), num(comSd, 4), num(fv.coef[1], 5)});
            }
            sum.set("set1.common_sd_min", *std::min_element(commonAll.begin(), commonAll.end()), 4);
            sum.set("set1.common_sd_max", *std::max_element(commonAll.begin(), commonAll.end()), 4);
            sum.set("set1.point_sd_x", median(pointSdX), 4);
            sum.set("set1.point_sd_y", median(pointSdY), 4);
        }

        // ---- 1차 + 쪽 2차 곡선 (2차 자료 예측용): dx = a + b v + c v^2
        auto plusFit = [&](char axis, bool ecc) {
            std::vector<double> x, y;
            for (size_t i : set1) {
                const double v = axis == 'T' ? shots[i].t : shots[i].y;
                const double other = axis == 'T' ? shots[i].y : shots[i].t;
                if (v <= 0 || other != 0) continue;
                x.push_back(v);
                y.push_back(ecc ? shots[i].eccDxMean : shots[i].dx);
            }
            return fitQuad(x, y);
        };
        const LsFit fy = plusFit('Y', false), ft = plusFit('T', false), fyE = plusFit('Y', true), ftE = plusFit('T', true);
        auto predict = [&](double y, double t, bool ecc) {
            const LsFit &a = ecc ? fyE : fy, &b = ecc ? ftE : ft;
            return a.coef[1] * y + a.coef[2] * y * y + b.coef[1] * t + b.coef[2] * t * t;   // 0점(절편)은 빼고 이동분만
        };
        sum.set("model.y_plus.a", fy.coef[0], 4); sum.set("model.y_plus.b", fy.coef[1], 4); sum.set("model.y_plus.c", fy.coef[2], 4); sum.set("model.y_plus.resid_sd", fy.residSd, 4);
        sum.set("model.t_plus.a", ft.coef[0], 4); sum.set("model.t_plus.b", ft.coef[1], 4); sum.set("model.t_plus.c", ft.coef[2], 4); sum.set("model.t_plus.resid_sd", ft.residSd, 4);
        {
            const LsFit ftLin = fitLine([&] { std::vector<double> x; for (size_t i : set1) if (shots[i].t > 0) x.push_back(shots[i].t); return x; }(),
                                        [&] { std::vector<double> y; for (size_t i : set1) if (shots[i].t > 0) y.push_back(shots[i].dx); return y; }());
            sum.set("model.t_plus.line_slope", ftLin.coef[1], 4);
            sum.set("model.t_plus.line_resid_sd", ftLin.residSd, 4);
            // Θ +2.00 에서 직선과 2차 곡선의 차이 (범위 밖으로 늘린 차이)
            sum.set("model.t_plus.quad_minus_line_at2", (ft.coef[0] + 2 * ft.coef[1] + 4 * ft.coef[2]) - (ftLin.coef[0] + 2 * ftLin.coef[1]), 4);
        }

        // ---- 2차: center 3장 (기준 center 대비 = 이번 세션 0점)
        std::vector<double> cdx, cdy;
        for (size_t i : centers2) {
            cdx.push_back(shots[i].dx);
            cdy.push_back(shots[i].dy);
            sum.set("set2.center" + std::to_string(shots[i].rep) + ".dx", shots[i].dx, 4);
            sum.set("set2.center" + std::to_string(shots[i].rep) + ".dy", shots[i].dy, 4);
            sum.set("set2.center" + std::to_string(shots[i].rep) + ".minutes", shots[i].minutes, 2);
            sum.set("set2.center" + std::to_string(shots[i].rep) + ".ecc_dx", shots[i].eccDxMean, 4);
            sum.text("set2.center" + std::to_string(shots[i].rep) + ".taken", shots[i].takenAt);
        }
        const double zeroCenters = mean(cdx);
        require(!centers2.empty(), "세션 center 사진이 없음 (clean_20261020 필요)");
        sum.set("set2.center_dx_mean", zeroCenters, 4);
        sum.set("set2.center_dx_spread", maxAbs({cdx.empty() ? 0 : *std::max_element(cdx.begin(), cdx.end()) - *std::min_element(cdx.begin(), cdx.end())}), 4);
        sum.set("set2.center_dy_mean", mean(cdy), 4);

        // ---- 2차: 조합별 (예측과 비교, 반복 흔들림)
        std::map<std::string, std::vector<size_t>> conds;
        for (size_t i : set2) conds[shots[i].group].push_back(i);
        std::vector<double> resid, residEcc, condSd, condSdEcc, condSdDy, ys, ts, yts;
        std::vector<double> repCommon, repPointX, repPointY;
        {
            Csv cc(out / "set2_conditions.csv", {"condition", "y", "t", "shots", "minutes_first", "span_sec", "pred_dx", "dx", "dx_sd", "dy", "dy_sd",
                                                "ecc_dx", "ecc_sd", "resid", "resid_ecc", "dx_rep1", "dx_rep2", "dx_rep3", "corners_ok_mean"});
            for (auto &[name, idx] : conds) {
                std::vector<double> d, de, dyv, okv;
                double first = 1e300, last = -1e300;
                for (size_t i : idx) {
                    d.push_back(shots[i].dx);
                    de.push_back(shots[i].eccDxMean);
                    dyv.push_back(shots[i].dy);
                    okv.push_back(shots[i].cornersOk);
                    first = std::min(first, shots[i].minutes);
                    last = std::max(last, shots[i].minutes);
                }
                const double y = shots[idx[0]].y, t = shots[idx[0]].t;
                const double pred = predict(y, t, false), predE = predict(y, t, true);
                const double r = mean(d) - pred, rE = mean(de) - predE;
                resid.push_back(r);
                residEcc.push_back(rE);
                ys.push_back(y);
                ts.push_back(t);
                yts.push_back(y * t);
                condSd.push_back(stdev(d));
                condSdEcc.push_back(stdev(de));
                condSdDy.push_back(stdev(dyv));
                // 반복 3장의 점별 편차 -> 사진 공통(평균)과 점 고유
                std::vector<double> com(idx.size(), 0.0), comY(idx.size(), 0.0);
                std::vector<std::vector<double>> ex(nC), ey(nC);
                int used = 0;
                for (size_t k = 0; k < nC; ++k) {
                    if (ref.corners[k].diagonal) continue;
                    bool all = true;
                    std::vector<double> px, py;
                    for (size_t i : idx) {
                        all = all && std::isfinite(shots[i].pos[k].x);
                        px.push_back(shots[i].pos[k].x);
                        py.push_back(shots[i].pos[k].y);
                    }
                    if (!all) continue;
                    const double mx = mean(px), my = mean(py);
                    for (size_t j = 0; j < idx.size(); ++j) { ex[k].push_back(px[j] - mx); ey[k].push_back(py[j] - my); com[j] += px[j] - mx; comY[j] += py[j] - my; }
                    ++used;
                }
                for (size_t j = 0; j < idx.size(); ++j) { com[j] /= used; comY[j] /= used; }
                repCommon.push_back(stdev(com));
                for (size_t k = 0; k < nC; ++k) {
                    if (ex[k].empty()) continue;
                    std::vector<double> rx, ry;
                    for (size_t j = 0; j < idx.size(); ++j) { rx.push_back(ex[k][j] - com[j]); ry.push_back(ey[k][j] - comY[j]); }
                    repPointX.push_back(stdev(rx));
                    repPointY.push_back(stdev(ry));
                }
                cc.row({quote(name), num(y, 2), num(t, 2), std::to_string(idx.size()), num(first, 3), num((last - first) * 60, 1), num(pred, 4),
                        num(mean(d), 4), num(stdev(d), 4), num(mean(dyv), 4), num(stdev(dyv), 4), num(mean(de), 4), num(stdev(de), 4),
                        num(r, 4), num(rE, 4), num(d.size() > 0 ? d[0] : kNaN, 4), num(d.size() > 1 ? d[1] : kNaN, 4), num(d.size() > 2 ? d[2] : kNaN, 4),
                        num(mean(okv), 1)});
            }
        }
        sum.set("set2.conditions", static_cast<double>(conds.size()), 0);
        sum.set("set2.resid_mean", mean(resid), 4);
        sum.set("set2.resid_sd", stdev(resid), 4);
        sum.set("set2.resid_min", *std::min_element(resid.begin(), resid.end()), 4);
        sum.set("set2.resid_max", *std::max_element(resid.begin(), resid.end()), 4);
        sum.set("set2.resid_rms", rms(resid), 4);
        sum.set("set2.resid_ecc_mean", mean(residEcc), 4);
        {
            // 남는 차이 = 0점 + 교차항 (y t) 으로 설명되는지
            const LsFit fz = leastSquares({std::vector<double>(resid.size(), 1.0)}, resid);
            const LsFit fi = leastSquares({std::vector<double>(resid.size(), 1.0), yts}, resid);
            const LsFit fl = leastSquares({std::vector<double>(resid.size(), 1.0), ys, ts}, resid);
            sum.set("set2.resid_after_zero_sd", fz.residSd, 4);
            sum.set("set2.interaction_coef", fi.coef[1], 4);
            sum.set("set2.interaction_resid_sd", fi.residSd, 4);
            sum.set("set2.resid_vs_y_slope", fl.coef[1], 4);
            sum.set("set2.resid_vs_t_slope", fl.coef[2], 4);
            sum.set("set2.resid_lin_resid_sd", fl.residSd, 4);
            // 남는 차이를 기준 기울기로 환산
            sum.set("set2.resid_mean_um", mean(resid) / fy.coef[1] * 1000, 1);
            sum.set("set2.resid_mean_t", mean(resid) / ft.coef[1], 4);
            Csv mc(out / "model.csv", {"name", "a", "b", "c", "resid_sd"});
            mc.row({"y_plus", num(fy.coef[0], 5), num(fy.coef[1], 5), num(fy.coef[2], 6), num(fy.residSd, 5)});
            mc.row({"t_plus", num(ft.coef[0], 5), num(ft.coef[1], 5), num(ft.coef[2], 6), num(ft.residSd, 5)});
            mc.row({"y_plus_ecc", num(fyE.coef[0], 5), num(fyE.coef[1], 5), num(fyE.coef[2], 6), num(fyE.residSd, 5)});
            mc.row({"t_plus_ecc", num(ftE.coef[0], 5), num(ftE.coef[1], 5), num(ftE.coef[2], 6), num(ftE.residSd, 5)});
        }
        sum.set("set2.rep_sd_median", median(condSd), 4);
        sum.set("set2.rep_sd_max", *std::max_element(condSd.begin(), condSd.end()), 4);
        sum.set("set2.rep_sd_min", *std::min_element(condSd.begin(), condSd.end()), 4);
        sum.set("set2.rep_sd_ecc_median", median(condSdEcc), 4);
        sum.set("set2.rep_sd_dy_median", median(condSdDy), 4);
        sum.set("set2.rep_common_sd_median", median(repCommon), 4);
        sum.set("set2.rep_point_sd_x", median(repPointX), 4);
        sum.set("set2.rep_point_sd_y", median(repPointY), 4);
        {
            // 반복 흔들림이 Θ 크기와 관계있는지: |t| <= 0.03 과 >= 1.0 비교
            std::vector<double> small, large;
            size_t k = 0;
            for (auto &[name, idx] : conds) {
                (void)name;
                if (ts[k] <= 0.03) small.push_back(condSd[k]);
                if (ts[k] >= 1.0) large.push_back(condSd[k]);
                ++k;
            }
            sum.set("set2.rep_sd_t_small_median", median(small), 4);
            sum.set("set2.rep_sd_t_large_median", median(large), 4);
        }

        // ---- 두 방법 차이 (모서리 - ECC), 세로 이동 비율
        {
            std::vector<double> diffX, dxs, dys;
            for (size_t i : set2) { diffX.push_back(shots[i].dx - shots[i].eccDxMean); dxs.push_back(shots[i].dx); dys.push_back(shots[i].dy); }
            sum.set("set2.corner_minus_ecc_max", maxAbs(diffX), 4);
            sum.set("set2.corner_minus_ecc_sd", stdev(diffX), 4);
            const LsFit fv = fitLine(dxs, dys);
            sum.set("set2.dy_per_dx", fv.coef[1], 5);
            sum.set("set2.dy_resid_sd", fv.residSd, 4);
            std::vector<double> dx1, dy1;
            for (size_t i : set1) { dx1.push_back(shots[i].dx); dy1.push_back(shots[i].dy); }
            sum.set("set1.dy_per_dx", fitLine(dx1, dy1).coef[1], 5);
        }

        // ---- 밝기
        {
            std::vector<double> p1, p2;
            for (size_t i : set1) p1.push_back(shots[i].plateMean);
            for (size_t i : set2) p2.push_back(shots[i].plateMean);
            sum.set("bright.ref_center", shots[refIdx].plateMean, 2);
            sum.set("bright.set1_mean", mean(p1), 2);
            sum.set("bright.set2_mean", mean(p2), 2);
            sum.set("bright.set2_min", *std::min_element(p2.begin(), p2.end()), 2);
            sum.set("bright.set2_max", *std::max_element(p2.begin(), p2.end()), 2);
            std::vector<double> pc;
            for (size_t i : centers2) pc.push_back(shots[i].plateMean);
            sum.set("bright.set2_centers_mean", mean(pc), 2);
            sum.set("bright.set2_vs_ref_pct", (mean(p2) / shots[refIdx].plateMean - 1) * 100, 2);
        }

        // ---- Y/Θ 구분 단서: 1차 281장에서 특징마다 F = a0 + bY y + cY y^2 + bT t + cT t^2 를 맞추고,
        //      Y 1mm / Θ 1 당 반응과 남는 흔들림, 2차 center 3장의 특징 0점(기준 center 대비)을 비교한다
        {
            std::vector<size_t> train = set1;
            train.push_back(refIdx);
            std::vector<double> c1(train.size(), 1.0), cy, cy2, ct, ct2;
            for (size_t i : train) { cy.push_back(shots[i].y); cy2.push_back(shots[i].y * shots[i].y); ct.push_back(shots[i].t); ct2.push_back(shots[i].t * shots[i].t); }
            for (int f = 0; f < kFeat; ++f) {
                std::vector<double> v, z;
                for (size_t i : train) v.push_back(features(shots[i])[f]);
                const LsFit fit = leastSquares({c1, cy, cy2, ct, ct2}, v);
                for (size_t i : centers2) z.push_back(features(shots[i])[f]);
                sum.set(std::string("cue.") + kFeatName[f] + ".per_y", fit.coef[1], 5);
                sum.set(std::string("cue.") + kFeatName[f] + ".per_t", fit.coef[3], 5);
                sum.set(std::string("cue.") + kFeatName[f] + ".sd", fit.residSd, 5);
                sum.set(std::string("cue.") + kFeatName[f] + ".set2_center", mean(z), 5);
            }
        }

        // ---- 반복 3장의 순서: 2번째 사진 - (1, 3번째 평균)
        {
            std::vector<double> d2, d2small, d2large;
            int positive = 0, largest = 0;
            for (auto &[name, idx] : conds) {
                (void)name;
                if (idx.size() != 3) continue;
                std::array<double, 3> v{};
                for (size_t i : idx) v[shots[i].rep - 1] = shots[i].dx;
                const double d = v[1] - 0.5 * (v[0] + v[2]);
                d2.push_back(d);
                (shots[idx[0]].t <= 0.03 ? d2small : d2large).push_back(d);
                positive += d > 0;
                largest += v[1] >= v[0] && v[1] >= v[2];
            }
            sum.set("set2.rep2_minus_others_median", median(d2), 4);
            sum.set("set2.rep2_minus_others_t_large_median", median(d2large), 4);
            sum.set("set2.rep2_minus_others_t_small_median", median(d2small), 4);
            sum.set("set2.rep2_positive", positive, 0);
            sum.set("set2.rep2_largest", largest, 0);
            sum.set("set2.rep2_conditions", static_cast<double>(d2.size()), 0);
        }

        // ---- 한 Y 에서 Θ 만 훑은 자료: 1차 trial 처럼 기울기, 0점, 한 칸 이동량, 반복별 경로
        {
            std::set<long> yset;
            for (size_t i : set2) yset.insert(std::lround(shots[i].y * 100));
            if (yset.size() == 1 && conds.size() >= 10) {
                struct Row { double t, first, mean; std::array<double, 3> rep; };
                std::vector<Row> rows;
                for (auto &[name, idx] : conds) {
                    (void)name;
                    Row r{shots[idx[0]].t, 1e300, 0, {kNaN, kNaN, kNaN}};
                    std::vector<double> d;
                    for (size_t i : idx) {
                        d.push_back(shots[i].dx);
                        if (shots[i].rep >= 1 && shots[i].rep <= 3) r.rep[shots[i].rep - 1] = shots[i].dx;
                        r.first = std::min(r.first, shots[i].minutes);
                    }
                    r.mean = mean(d);
                    rows.push_back(r);
                }
                std::sort(rows.begin(), rows.end(), [](const Row &a, const Row &b) { return a.t < b.t; });
                std::vector<double> ts_, ms_;
                for (const auto &r : rows) { ts_.push_back(r.t); ms_.push_back(r.mean); }
                const LsFit fl = fitLine(ts_, ms_), fq = fitQuad(ts_, ms_);
                sum.set("sweep.y", rows[0].t * 0 + shots[set2[0]].y, 2);
                sum.set("sweep.t_min", ts_.front(), 2);
                sum.set("sweep.t_max", ts_.back(), 2);
                sum.set("sweep.conditions", static_cast<double>(rows.size()), 0);
                sum.set("sweep.slope", fl.coef[1], 4);
                sum.set("sweep.zero", fl.coef[0] - fy.coef[1] * shots[set2[0]].y - fy.coef[2] * shots[set2[0]].y * shots[set2[0]].y, 4);
                sum.set("sweep.intercept", fl.coef[0], 4);
                sum.set("sweep.line_resid_sd", fl.residSd, 4);
                sum.set("sweep.quad_resid_sd", fq.residSd, 4);
                sum.set("sweep.quad_c", fq.coef[2], 4);
                // 촬영 순서가 Θ 순서인지
                int inversions = 0;
                for (size_t k = 1; k < rows.size(); ++k) inversions += rows[k].first < rows[k - 1].first;
                sum.set("sweep.order_inversions", inversions, 0);
                sum.set("sweep.minutes_first", rows.front().first, 2);
                double lastMin = 0;
                for (size_t i : set2) lastMin = std::max(lastMin, shots[i].minutes);
                sum.set("sweep.minutes_last", lastMin, 2);
                // 한 칸(Θ 0.01) 이동량: 3장 평균, 반복별
                std::vector<double> step, stepRep[3];
                std::vector<std::pair<double, double>> stepAt;
                for (size_t k = 1; k < rows.size(); ++k) {
                    if (std::lround((rows[k].t - rows[k - 1].t) * 100) != 1) continue;
                    step.push_back(rows[k].mean - rows[k - 1].mean);
                    stepAt.emplace_back(rows[k].t, step.back());
                    for (int r = 0; r < 3; ++r) stepRep[r].push_back(rows[k].rep[r] - rows[k - 1].rep[r]);
                }
                const double sm = mean(step), ss = stdev(step);
                sum.set("sweep.step_mean", sm, 4);
                sum.set("sweep.step_sd", ss, 4);
                sum.set("sweep.step_n", static_cast<double>(step.size()), 0);
                for (int r = 0; r < 3; ++r) sum.set("sweep.step_sd_rep" + std::to_string(r + 1), stdev(stepRep[r]), 4);
                int out3 = 0;
                for (const auto &[t, v] : stepAt) out3 += std::fabs(v - sm) > 3 * ss;
                sum.set("sweep.step_out3", out3, 0);
                // 반복별 경로: 2차 곡선을 뺀 잔차 σ (= 그 반복 줄의 사진 공통 흔들림), 반복끼리 평균 차이
                for (int r = 0; r < 3; ++r) {
                    std::vector<double> x, v, dv;
                    for (const auto &row : rows) {
                        if (!std::isfinite(row.rep[r])) continue;
                        x.push_back(row.t);
                        v.push_back(row.rep[r]);
                        dv.push_back(row.rep[r] - row.mean);
                    }
                    sum.set("sweep.rep" + std::to_string(r + 1) + ".quad_resid_sd", fitQuad(x, v).residSd, 4);
                    sum.set("sweep.rep" + std::to_string(r + 1) + ".minus_mean", mean(dv), 4);
                    sum.set("sweep.rep" + std::to_string(r + 1) + ".minus_mean_sd", stdev(dv), 4);
                }
                Csv wc(out / "sweep.csv", {"t", "minutes_first", "dx_mean", "dx_rep1", "dx_rep2", "dx_rep3", "pred_dx", "resid", "line_resid"});
                for (size_t k = 0; k < rows.size(); ++k) {
                    const double y0 = shots[set2[0]].y;
                    wc.row({num(rows[k].t, 2), num(rows[k].first, 3), num(rows[k].mean, 4), num(rows[k].rep[0], 4), num(rows[k].rep[1], 4),
                            num(rows[k].rep[2], 4), num(predict(y0, rows[k].t, false), 4), num(rows[k].mean - predict(y0, rows[k].t, false), 4),
                            num(fl.resid[k], 4)});
                }
            }
        }

        // ---- 사진 조각 (QR 줄 주변, 고정 위치) : 기준 center, 세션 center, 조합별 첫 사진
        {
            const auto &L = ref.qrs[0].label, &R = ref.qrs[2].label;
            const cv::Rect band(L.x - 300, L.y - 40, R.x + R.width + 300 - (L.x - 300), L.height + 80);
            Csv cr(out / "crop" / "crops.csv", {"png", "file", "group", "x", "y", "w", "h", "dx", "dy"});
            auto save = [&](size_t i, const std::string &name) {
                const cv::Mat g = readGray(shots[i].file);
                writePng(out / "crop" / (name + ".png"), g(band));
                cr.row({name + ".png", quote(shots[i].rel), quote(shots[i].group), std::to_string(band.x), std::to_string(band.y),
                        std::to_string(band.width), std::to_string(band.height), num(shots[i].dx, 4), num(shots[i].dy, 4)});
            };
            save(refIdx, "ref_center");
            for (size_t i : centers2) save(i, "center" + std::to_string(shots[i].rep));
            int k = 0;
            for (auto &[name, idx] : conds) save(idx[0], "cond" + std::to_string(k++));
        }

        sum.write(out / "summary.yml");
        std::cout << "완료: " << out.u8string() << "\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "오류: " << e.what() << "\n";
        return 1;
    }
}
