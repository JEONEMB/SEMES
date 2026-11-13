#include "site.hpp"
#include <stdexcept>

namespace site {
namespace {
// 원본 모듈의 상수(SEMES 센터 사진 기준)
const Geometry original{
    "01a69df6771b840d2fcf9cf727760592b5fb199c12277bfc173af514e26fd685",
    cv::Rect(1340, 800, 1130, 530),
    {cv::Rect(1552, 969, 184, 198), cv::Rect(1809, 971, 184, 198), cv::Rect(2067, 973, 185, 198)},
    {cv::Point(974, 1352), cv::Point(2864, 1358), cv::Point(1010, 1677), cv::Point(2817, 1681)},
    110, 22};
Geometry current = original;

cv::Rect rect(const cv::FileNode& n) {
    if (!n.isSeq() || n.size() != 4) throw std::runtime_error("site.json: 상자는 [x, y, 너비, 높이] 여야 한다");
    return cv::Rect(int(n[0]), int(n[1]), int(n[2]), int(n[3]));
}
}

Geometry read(const std::filesystem::path& assets) {
    cv::FileStorage f((assets / "site.json").u8string(), cv::FileStorage::READ);
    if (!f.isOpened()) return original;   // site.json 이 없으면 원본 상수
    Geometry g;
    f["center_sha256"] >> g.center_sha256;
    g.full_crop = rect(f["full_crop"]);
    const auto qr = f["qr"], screws = f["screws"];
    if (!qr.isSeq() || qr.size() != 3 || !screws.isSeq() || screws.size() != 4) throw std::runtime_error("site.json: qr 3개와 screws 4개가 필요하다");
    for (int k = 0; k < 3; ++k) g.qr[k] = rect(qr[k]);
    for (int k = 0; k < 4; ++k) {
        if (!screws[k].isSeq() || screws[k].size() != 2) throw std::runtime_error("site.json: 나사는 [x, y] 여야 한다");
        g.screws[k] = cv::Point(int(screws[k][0]), int(screws[k][1]));
    }
    g.search_x = int(f["search_x"]); g.search_y = int(f["search_y"]);
    if (!f["qr_method"].empty()) f["qr_method"] >> g.qr_method;
    if (g.qr_method != "track536" && g.qr_method != "ncc") throw std::runtime_error("site.json: qr_method 는 track536 또는 ncc 여야 한다");
    const cv::Rect image(0, 0, 3840, 2160);
    if (g.center_sha256.size() != 64 || g.search_x < 1 || g.search_y < 1 || (g.full_crop & image) != g.full_crop)
        throw std::runtime_error("site.json: 값이 잘못됐다");
    for (int k = 0; k < 3; ++k) {
        // 원본과 같은 상자 크기만 받는다(추적 조각과 관측 마스크가 이 크기에 맞춰져 있다).
        const cv::Rect search(g.qr[k].x - g.search_x, g.qr[k].y - g.search_y, g.qr[k].width + 2 * g.search_x, g.qr[k].height + 2 * g.search_y);
        if (g.qr[k].size() != original.qr[k].size() || (search & g.full_crop) != search) throw std::runtime_error("site.json: QR 상자 또는 탐색 범위가 full_crop 을 벗어난다");
    }
    return g;
}

void load(const std::filesystem::path& assets) { current = read(assets); }

const Geometry& get() { return current; }
}
