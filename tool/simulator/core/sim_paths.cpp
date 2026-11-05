// =====================================================================================
// sim_paths.cpp — 루트 찾기, UTF-8 경로, 사진 읽기
// =====================================================================================
#include "sim_paths.hpp"

#include <opencv2/imgcodecs.hpp>

#include <fstream>
#include <iterator>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#include <climits>
#endif

namespace fs = std::filesystem;

namespace sim {

fs::path u8p(const std::string& utf8) { return fs::u8path(utf8); }

std::string u8s(const fs::path& p) {
    const auto s = p.u8string();
    return std::string(s.begin(), s.end());
}

fs::path exeDir() {
#ifdef _WIN32
    std::wstring buf(32768, L'\0');
    const DWORD n = GetModuleFileNameW(nullptr, &buf[0], DWORD(buf.size()));
    buf.resize(n);
    return fs::path(buf).parent_path();
#else
    char buf[PATH_MAX] = {0};
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) return fs::current_path();
    return fs::path(std::string(buf, size_t(n))).parent_path();
#endif
}

namespace {
bool hasConfig(const fs::path& dir) {
    std::error_code ec;
    return fs::is_regular_file(dir / kConfigRelPath, ec);
}
// dir 또는 dir/Simulator 에 설정이 있으면 그 폴더
bool probe(const fs::path& dir, fs::path& found) {
    if (dir.empty()) return false;
    if (hasConfig(dir)) { found = dir; return true; }
    if (hasConfig(dir / "Simulator")) { found = dir / "Simulator"; return true; }
    return false;
}
} // namespace

bool enterRoot(const std::string& rootArg, fs::path& root, std::string& message) {
    std::error_code ec;
    fs::path found;
    if (!rootArg.empty()) {
        if (!probe(fs::absolute(u8p(rootArg), ec), found)) {
            message = "지정한 루트에 " + std::string(kConfigRelPath) + " 가 없음 : " + rootArg;
            return false;
        }
    } else {
        bool ok = probe(fs::current_path(ec), found);
        fs::path d = exeDir();
        for (int i = 0; !ok && i <= kSearchUpLevels && !d.empty(); ++i) {
            ok = probe(d, found);
            if (d == d.parent_path()) break;
            d = d.parent_path();
        }
        if (!ok) {
            message = "시뮬레이터 루트를 찾지 못함 : 작업 폴더나 exe 폴더(및 상위 " + std::to_string(kSearchUpLevels) + "단계)에 " +
                      kConfigRelPath + " 가 없음";
            return false;
        }
    }
    root = fs::weakly_canonical(found, ec);
    if (ec) root = fs::absolute(found);
    fs::current_path(root, ec);
    if (ec) { message = "작업 폴더를 바꾸지 못함 : " + u8s(root); return false; }
    message = "시뮬레이터 루트 : " + u8s(root);
    return true;
}

std::string firstExisting(const std::vector<std::string>& candidates) {
    std::error_code ec;
    for (const auto& c : candidates)
        if (!c.empty() && fs::exists(u8p(c), ec)) return c;
    return std::string();
}

cv::Mat readImage(const std::string& utf8Path, int flags) {
    std::ifstream f(u8p(utf8Path), std::ios::binary);
    if (!f) return cv::Mat();
    std::vector<unsigned char> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (buf.empty()) return cv::Mat();
    return cv::imdecode(buf, flags);
}

} // namespace sim
