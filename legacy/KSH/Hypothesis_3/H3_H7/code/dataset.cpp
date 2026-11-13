// =====================================================================================
// dataset.cpp — 영상 목록 · 이름 해석 · 학습/검증 분할
// =====================================================================================
#include "dataset.hpp"
#include <algorithm>
#include <fstream>
#include <regex>
#include <cmath>

namespace h3 {

std::string u8(const fs::path& p) { auto s = p.u8string(); return std::string(s.begin(), s.end()); }

cv::Mat readGray(const fs::path& p)
{
    std::ifstream f(p, std::ios::binary);
    if (!f) return cv::Mat();
    std::vector<uchar> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return buf.empty() ? cv::Mat() : cv::imdecode(buf, cv::IMREAD_GRAYSCALE);
}

bool parseName(const std::string& stem, Item& it)
{
    std::string s = stem;
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    static const std::regex reC("^center(\\d*)$");
    static const std::regex reY("(?:^|[^a-z])y\\s*([+-])\\s*(\\d+(?:\\.\\d+)?)");
    static const std::regex reT("(?:^|[^a-z])(?:theta|t)\\s*([+-])\\s*(\\d+(?:\\.\\d+)?)");
    static const std::regex reS("\\((\\d+)\\)");
    std::smatch m;
    if (std::regex_match(s, m, reC)) {
        it.isCenter = true;
        it.shot = m[1].str().empty() ? 0 : std::stoi(m[1].str());
        it.set = it.shot == 0 ? 1 : 2;
        return true;
    }
    bool hy = false, ht = false;
    if (std::regex_search(s, m, reY)) { it.y_mm = std::stod(m[2].str()) * (m[1].str() == "-" ? -1 : 1); hy = true; }
    if (std::regex_search(s, m, reT)) { it.t_deg = std::stod(m[2].str()) * (m[1].str() == "-" ? -1 : 1); ht = true; }
    if (std::regex_search(s, m, reS)) { it.set = 2; it.shot = std::stoi(m[1].str()); }
    return hy || ht;
}

std::vector<Item> listImages(const fs::path& imgDir)
{
    std::vector<Item> v;
    if (!fs::exists(imgDir)) return v;
    for (const auto& de : fs::directory_iterator(imgDir)) {
        if (!de.is_regular_file()) continue;
        std::string ext = u8(de.path().extension());
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return (char)std::tolower(c); });
        if (ext != ".bmp" && ext != ".png") continue;
        Item it; it.path = de.path(); it.name = u8(de.path().filename());
        if (parseName(u8(de.path().stem()), it)) v.push_back(it);
    }
    std::sort(v.begin(), v.end(), [](const Item& a, const Item& b) {
        if (a.set != b.set) return a.set < b.set;
        if (a.isCenter != b.isCenter) return a.isCenter;
        if (a.y_mm != b.y_mm) return a.y_mm < b.y_mm;
        if (a.t_deg != b.t_deg) return a.t_deg < b.t_deg;
        return a.shot < b.shot; });
    // 1차 세트 : 스윕별 값 순서로 짝수 번째 학습, 홀수 번째 검증 (center.bmp 는 기준 영상이라 제외)
    std::vector<int> ys, ts;
    for (int i = 0; i < (int)v.size(); ++i) {
        if (v[i].set != 1 || v[i].isCenter) continue;
        if (v[i].t_deg == 0) ys.push_back(i); else if (v[i].y_mm == 0) ts.push_back(i);
    }
    std::sort(ys.begin(), ys.end(), [&](int a, int b) { return v[a].y_mm < v[b].y_mm; });
    std::sort(ts.begin(), ts.end(), [&](int a, int b) { return v[a].t_deg < v[b].t_deg; });
    for (auto* s : { &ys, &ts })
        for (size_t k = 0; k < s->size(); ++k) { if (k % 2 == 0) v[(*s)[k]].train = true; else v[(*s)[k]].test = true; }
    return v;
}

int findItem(const std::vector<Item>& v, const std::string& name)
{
    for (int i = 0; i < (int)v.size(); ++i) if (v[i].name == name) return i;
    return -1;
}

int findPose(const std::vector<Item>& v, int set, double y_mm, double t_deg, int shot, bool isCenter)
{
    for (int i = 0; i < (int)v.size(); ++i)
        if (v[i].set == set && v[i].isCenter == isCenter && v[i].shot == shot &&
            std::fabs(v[i].y_mm - y_mm) < 1e-6 && std::fabs(v[i].t_deg - t_deg) < 1e-6) return i;
    return -1;
}

} // namespace h3
