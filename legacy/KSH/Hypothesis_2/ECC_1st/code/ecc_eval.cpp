// =====================================================================================
// ecc_eval.cpp — 가설 H2 검증용 데이터 목록 · 평가 · 저장
// =====================================================================================
#include "ecc_eval.hpp"
#include "result_plot.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <regex>
#include <map>
#include <algorithm>
#include <cmath>
#include <cstdio>

namespace ev {

std::string u8(const fs::path& p) {
    auto s = p.u8string();                       // C++17 : std::string, C++20 : std::u8string
    return std::string(s.begin(), s.end());
}

cv::Mat readGray(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return cv::Mat();
    std::vector<uchar> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (buf.empty()) return cv::Mat();
    return cv::imdecode(buf, cv::IMREAD_GRAYSCALE);
}

bool parseTruth(const std::string& stem, double& y, double& t) {
    y = 0; t = 0;
    std::string s = stem;
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    if (s.rfind("center", 0) == 0) return true;
    static const std::regex reY("(?:^|[^a-z])y\\s*([+-])\\s*(\\d+(?:\\.\\d+)?)");
    static const std::regex reT("(?:^|[^a-z])(?:theta|t)\\s*([+-])\\s*(\\d+(?:\\.\\d+)?)");
    std::smatch m;
    bool hy = false, ht = false;
    if (std::regex_search(s, m, reY)) { y = std::stod(m[2].str()) * (m[1].str() == "-" ? -1.0 : 1.0); hy = true; }
    if (std::regex_search(s, m, reT)) { t = std::stod(m[2].str()) * (m[1].str() == "-" ? -1.0 : 1.0); ht = true; }
    return hy || ht;
}

static bool isImage(const fs::path& p) {
    std::string e = u8(p.extension());
    std::transform(e.begin(), e.end(), e.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    return e == ".bmp" || e == ".png" || e == ".jpg" || e == ".tif" || e == ".tiff";
}

std::vector<Item> listSet1(const fs::path& imgDir) {
    std::vector<Item> out;
    const char* sub[2] = { "Y", "Theta" };
    for (int k = 0; k < 2; ++k) {
        fs::path d = imgDir / sub[k];
        if (!fs::exists(d)) continue;
        std::vector<Item> v;
        for (const auto& de : fs::directory_iterator(d)) {
            if (!de.is_regular_file() || !isImage(de.path())) continue;
            Item it;
            it.path = de.path();
            it.name = u8(de.path().filename());
            it.set = sub[k];
            it.hasTruth = parseTruth(u8(de.path().stem()), it.trueY, it.trueT);
            if (it.hasTruth) v.push_back(it);
        }
        // 값 순서로 정렬 → 짝수 번째 = 학습, 홀수 번째 = 검증 (1차 실험과 같은 분할)
        std::sort(v.begin(), v.end(), [&](const Item& a, const Item& b) {
            return k == 0 ? a.trueY < b.trueY : a.trueT < b.trueT; });
        for (size_t i = 0; i < v.size(); ++i) v[i].train = (i % 2 == 0);
        out.insert(out.end(), v.begin(), v.end());
    }
    return out;
}

std::vector<Item> listSet2(const fs::path& set2Dir) {
    std::vector<Item> out;
    if (!fs::exists(set2Dir)) return out;
    static const std::regex reShot("\\((\\d+)\\)");
    static const std::regex reCenter("center(\\d+)");
    for (const auto& dd : fs::directory_iterator(set2Dir)) {
        if (!dd.is_directory()) continue;
        const std::string setName = u8(dd.path().filename());
        double fy = 0, ft = 0;
        bool folderTruth = parseTruth(setName, fy, ft);   // "Y+0.10 T+0.01" (기준 위치 폴더는 false)
        for (const auto& de : fs::directory_iterator(dd.path())) {
            if (!de.is_regular_file() || !isImage(de.path())) continue;
            Item it;
            it.path = de.path();
            it.name = u8(de.path().filename());
            const std::string stem = u8(de.path().stem());
            std::smatch m;
            if (std::regex_search(stem, m, reShot)) it.shot = std::stoi(m[1].str());
            else if (std::regex_search(stem, m, reCenter)) it.shot = std::stoi(m[1].str());
            if (folderTruth) { it.set = setName; it.trueY = fy; it.trueT = ft; it.hasTruth = true; }
            else { it.set = "center"; it.hasTruth = parseTruth(stem, it.trueY, it.trueT); }
            out.push_back(it);
        }
    }
    std::sort(out.begin(), out.end(), [](const Item& a, const Item& b) {
        if (a.trueY != b.trueY) return a.trueY < b.trueY;
        if (a.trueT != b.trueT) return a.trueT < b.trueT;
        return a.shot < b.shot; });
    return out;
}

void measureAll(const ea::EccAligner& al, std::vector<Item>& items, const char* label) {
    double sumMs = 0;
    int ok = 0;
    for (size_t i = 0; i < items.size(); ++i) {
        cv::Mat g = readGray(items[i].path);
        if (g.empty()) { std::cout << "  읽기 실패: " << items[i].name << "\n"; continue; }
        items[i].m = al.measure(g);
        if (items[i].m.ok) { ok++; sumMs += items[i].m.ms; }
        if (i % 40 == 0 || i + 1 == items.size())
            std::cout << "  [" << label << " " << std::setw(3) << i + 1 << "/" << items.size() << "] " << items[i].name
                      << (items[i].m.ok ? "" : "  (ECC 실패)") << "\n";
    }
    std::cout << "  " << label << " : " << ok << "/" << items.size() << " 장 성공, 평균 "
              << std::fixed << std::setprecision(0) << (ok ? sumMs / ok : 0) << " ms/장 (측정만)\n";
}

bool writeCsv(const fs::path& path, const std::vector<Item>& items, double tolY_mm, double tolT_deg) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f << "\xEF\xBB\xBF";   // UTF-8 BOM (엑셀)
    f << "file,set,shot,train,trueY_mm,estY_mm,errY_um,trueT_deg,estT_deg,errT_deg,judge,"
         "ecc_cc,tL,sL,tC,sC,tR,sR,vC,coarseDx,coarseDy,ms\r\n";
    char b[512];
    for (const Item& it : items) {
        f << "\"" << it.name << "\",\"" << it.set << "\"," << it.shot << "," << (it.train ? 1 : 0) << ",";
        const bool both = it.hasTruth && it.e.valid;
        if (it.hasTruth) { std::snprintf(b, sizeof(b), "%.4f,", it.trueY); f << b; } else f << ",";
        if (it.e.valid)  { std::snprintf(b, sizeof(b), "%.6f,", it.e.y); f << b; } else f << ",";
        if (both)        { std::snprintf(b, sizeof(b), "%.2f,", (it.e.y - it.trueY) * 1000); f << b; } else f << ",";
        if (it.hasTruth) { std::snprintf(b, sizeof(b), "%.4f,", it.trueT); f << b; } else f << ",";
        if (it.e.valid)  { std::snprintf(b, sizeof(b), "%.6f,", it.e.t); f << b; } else f << ",";
        if (both) {
            bool py = std::fabs(it.e.y - it.trueY) <= tolY_mm, pt = std::fabs(it.e.t - it.trueT) <= tolT_deg;
            std::snprintf(b, sizeof(b), "%.6f,%s,", it.e.t - it.trueT, (py && pt) ? "OK" : (!py && !pt) ? "Y.T NG" : (!py ? "Y NG" : "T NG"));
            f << b;
        } else f << ",,";
        std::snprintf(b, sizeof(b), "%.6f,%.5f,%.6f,%.5f,%.6f,%.5f,%.6f,%.4f,%.3f,%.3f,%.1f\r\n", it.m.cc,
                      it.m.f[0], it.m.f[1], it.m.f[2], it.m.f[3], it.m.f[4], it.m.f[5], it.m.vC,
                      it.m.coarseDx, it.m.coarseDy, it.m.ms);
        f << b;
    }
    return true;
}

// 같은 자세 3장 : x = 자세(조합) 번호, y = 오차.  위 = Θ 오차, 아래 = Y 오차
cv::Mat drawRepeatability(const std::vector<Item>& set2, double tolT_deg) {
    std::vector<std::string> sets;
    for (const Item& it : set2) if (it.set != "center" && std::find(sets.begin(), sets.end(), it.set) == sets.end()) sets.push_back(it.set);
    const int W = 1820, H = 980;
    cv::Mat img(H, W, CV_8UC3, cv::Scalar::all(255));
    cv::putText(img, "Repeatability: 3 shots at the same pose (set 2)", cv::Point(24, 40), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                cv::Scalar(10, 10, 10), 2, cv::LINE_AA);
    if (sets.empty()) return img;
    const cv::Scalar col[3] = { cv::Scalar(200, 120, 30), cv::Scalar(30, 130, 240), cv::Scalar(60, 170, 60) };
    for (int s = 0; s < 3; ++s) {
        cv::circle(img, cv::Point(1200 + s * 140, 34), 6, col[s], cv::FILLED, cv::LINE_AA);
        cv::putText(img, "shot " + std::to_string(s + 1), cv::Point(1212 + s * 140, 40), cv::FONT_HERSHEY_SIMPLEX, 0.55, col[s], 1, cv::LINE_AA);
    }
    for (int row = 0; row < 2; ++row) {
        std::vector<double> ev;
        for (const Item& it : set2) if (it.set != "center" && it.e.valid)
            ev.push_back(row == 0 ? it.e.t - it.trueT : (it.e.y - it.trueY) * 1000);
        double lo, hi;
        rp::symRange(ev, row == 0 ? tolT_deg * 2.5 : 135.0, lo, hi);
        rp::Panel P(img, cv::Rect(110, 100 + row * 440, 1660, 330), -0.5, sets.size() - 0.5, lo, hi);
        if (row == 0) P.band(-tolT_deg, tolT_deg, cv::Scalar(215, 240, 215));
        else          P.band(-90, 90, cv::Scalar(215, 240, 215));
        P.frame(row == 0 ? "Theta error per pose (deg)" : "Y error per pose (um)", "pose index (sorted by Y, then Theta)", "");
        P.axes();
        for (size_t k = 0; k < sets.size(); ++k) {
            for (const Item& it : set2) {
                if (it.set != sets[k] || !it.e.valid || it.shot < 1 || it.shot > 3) continue;
                double v = row == 0 ? it.e.t - it.trueT : (it.e.y - it.trueY) * 1000;
                P.point(k + (it.shot - 2) * 0.12, v, col[it.shot - 1], 4);
            }
            if (row == 1) {
                cv::Point p = P.px((double)k, lo);
                cv::putText(img, sets[k], cv::Point(p.x - 30, p.y + 52 + (int)(k % 2) * 16), cv::FONT_HERSHEY_PLAIN, 0.85,
                            cv::Scalar(80, 80, 80), 1, cv::LINE_AA);
            }
        }
    }
    return img;
}

} // namespace ev
