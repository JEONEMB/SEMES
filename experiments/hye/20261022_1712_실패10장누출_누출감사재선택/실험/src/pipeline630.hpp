// 341장 목록(원래 평가 순서) + 630점 측정 공통 절차 (20261022_1608 실험과 같은 절차)
#pragma once
#include <hye/dataset.hpp>
#include <hye/report.hpp>
#include <hye/plot.hpp>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "extract630.hpp"
#include "protocol341.hpp"

namespace pipe630 {

using hye::Shot;
namespace fs = std::filesystem;

struct Img {
    Shot shot;
    int fold = 0;
    std::string group, scope, name;
    bool isnew = false;
    e630::Measure m;
    std::vector<double> D;  // 1260 (dx, dy 교차)
    double h = 0, hy = 0;
    int nvalid = 0;
};

// hye::read_gray 는 cv::imread 라 한글 경로(clean_20261020/기준 위치)를 못 읽는다. 바이트로 읽어 imdecode 한다.
inline cv::Mat imread_u8(const Shot& s, int flag) {
    std::ifstream f(fs::u8path(s.path), std::ios::binary);
    std::vector<uchar> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    cv::Mat m = cv::imdecode(buf, flag);
    if (m.empty()) throw std::runtime_error("읽기 실패: " + s.path);
    return m;
}

inline std::string short_name(const Img& im) {
    std::string r = im.shot.rel;
    r = r.substr(r.rfind('/') + 1);
    r = r.substr(0, r.rfind('.'));
    std::string out;
    for (char c : r) {
        if (c == ' ') out += '_';
        else if (c == '(') out += 'r';
        else if (c != ')') out += c;
    }
    if (im.shot.set == "clean_20261020" && im.scope == "center") out = "1020_" + out;
    return out;
}

// 개발용: HYE_DEV_CACHE 가 있으면 측정을 저장·재사용한다(최종 실행에서는 쓰지 않는다)
inline bool load_cache(const std::string& p, std::vector<Img>& v, std::vector<cv::Point2d>& ref) {
    std::ifstream f(fs::u8path(p), std::ios::binary);
    if (!f) return false;
    ref.resize(630);
    f.read((char*)ref.data(), sizeof(cv::Point2d) * 630);
    for (auto& im : v) {
        auto& m = im.m;
        f.read((char*)&m.ncc_min, sizeof m.ncc_min);
        f.read((char*)&m.pad, sizeof m.pad);
        f.read((char*)m.shift, sizeof m.shift);
        m.ok = true;
        m.cur.resize(630); m.q.resize(630); m.valid.resize(630); m.why.resize(630);
        f.read((char*)m.cur.data(), sizeof(cv::Point2d) * 630);
        f.read((char*)m.q.data(), sizeof(double) * 5 * 630);
        f.read((char*)m.valid.data(), 630);
        f.read((char*)m.why.data(), sizeof(int) * 630);
    }
    return (bool)f;
}
inline void save_cache(const std::string& p, const std::vector<Img>& v, const std::vector<cv::Point2d>& ref) {
    std::ofstream f(fs::u8path(p), std::ios::binary);
    f.write((const char*)ref.data(), sizeof(cv::Point2d) * 630);
    for (auto& im : v) {
        auto& m = im.m;
        f.write((const char*)&m.ncc_min, sizeof m.ncc_min);
        f.write((const char*)&m.pad, sizeof m.pad);
        f.write((const char*)m.shift, sizeof m.shift);
        f.write((const char*)m.cur.data(), sizeof(cv::Point2d) * 630);
        f.write((const char*)m.q.data(), sizeof(double) * 5 * 630);
        f.write((const char*)m.valid.data(), 630);
        f.write((const char*)m.why.data(), sizeof(int) * 630);
    }
}

// 원래 순서로 341장을 맞추고 630점을 잰다. ref = center 사진 자신의 scipy-TRF 측정.
// compiled = true 면 각 사진은 Numba LM 변형으로 잰다(기준은 그대로 TRF).
inline std::vector<Img> load_and_measure(std::vector<cv::Point2d>& ref, bool compiled = false, const char* cache_env = "HYE_DEV_CACHE") {
    auto shots = hye::load_341();
    std::map<std::string, Shot> by;
    for (auto& s : shots) by[s.set + "/" + s.rel] = s;
    std::vector<Img> imgs(341);
    for (int i = 0; i < 341; ++i) {
        auto& r = p341::kRows[i];
        auto it = by.find(std::string(r.set) + "/" + r.rel);
        if (it == by.end()) throw std::runtime_error(std::string("사진 없음: ") + r.rel);
        imgs[i].shot = it->second;
        imgs[i].fold = r.fold;
        imgs[i].group = r.cv_group;
        imgs[i].scope = r.scope;
        imgs[i].isnew = std::string(r.set) == "clean_20261020";
    }
    for (auto& im : imgs) im.name = short_name(im);
    const cv::Mat center = imread_u8(hye::center_1014(shots), cv::IMREAD_GRAYSCALE);
    const char* cache = std::getenv(cache_env);
    bool cached = cache && load_cache(cache, imgs, ref);
    if (!cached) {
        ref = e630::measure(center, center).cur;
        std::cout << "measuring 341..." << std::endl;
        std::vector<std::string> errs(341);
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < 341; ++i) {
            try {
                imgs[i].m = e630::measure(imread_u8(imgs[i].shot, cv::IMREAD_GRAYSCALE), center, compiled);
            } catch (const std::exception& e) {
                errs[i] = e.what();
            }
        }
        for (int i = 0; i < 341; ++i)
            if (!errs[i].empty()) throw std::runtime_error("측정 오류 " + imgs[i].shot.rel + ": " + errs[i]);
        if (cache) save_cache(cache, imgs, ref);
    }
    for (auto& im : imgs) {
        if (!im.m.ok) throw std::runtime_error("NCC 실패: " + im.shot.rel);
        im.D.resize(1260);
        double sx = 0, sy = 0;
        for (int k = 0; k < 630; ++k) {
            im.D[2 * k] = im.m.cur[k].x - ref[k].x;
            im.D[2 * k + 1] = im.m.cur[k].y - ref[k].y;
            sx += im.D[2 * k];
            sy += im.D[2 * k + 1];
            im.nvalid += im.m.valid[k];
        }
        im.h = sx / 630;
        im.hy = sy / 630;
    }
    std::cout << "measured" << std::endl;
    return imgs;
}

// 튀는 값 그림: 세 QR 확대. 초록 = 기준 QR 상자, 빨강 = NCC 로 옮긴 상자, 빨간 십자 = 무효 판정 모서리
inline void qr_outlier_fig(const Img& im, const std::string& file, const std::string& cap) {
    using namespace hye;
    cv::Mat col = imread_u8(im.shot, cv::IMREAD_COLOR);
    std::vector<cv::Mat> tiles;
    for (int q = 0; q < 3; ++q) {
        const int* b = e630::kBoxes[q];
        cv::Rect2d refb(b[0], b[1], b[2] - b[0], b[3] - b[1]);
        cv::Rect2d now(refb.x + im.m.shift[q].x, refb.y + im.m.shift[q].y, refb.width, refb.height);
        cv::Rect roi = cv::Rect(refb | now) + cv::Size(40, 40) - cv::Point(20, 20);
        std::vector<Mark> mk;
        for (int k = 0; k < 630; ++k)
            if (c630::kCorners[k].qr == q + 1 && !im.m.valid[k]) mk.push_back({im.m.cur[k], {0, 0}, "", kRed});
        tiles.push_back(zoom(col, roi, 420, mk, 0, q == 0 ? cap : "",
                             {{refb, kGreen, ""}, {now, kRed, "dx " + std::to_string(im.m.shift[q].x) + " dy " + std::to_string(im.m.shift[q].y)}}));
    }
    save_png(file, hstack(tiles));
}

}  // namespace pipe630
