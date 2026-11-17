#include "measurements.hpp"

#include <iostream>

namespace hw {

Measurements measureSet(const DataSet &ds) {
    Measurements m;
    const Shot &ref = ds.shots[ds.centers[0]];
    log("기준 사진: " + ref.rel);

    // 기준 사진: 형식, QR 라벨, 모서리
    const cv::Mat refColor = readImage(ref.file);
    m.imageSize = refColor.size();
    m.channels = refColor.channels();
    if (m.channels == 3) {
        std::vector<cv::Mat> ch;
        cv::split(refColor, ch);
        m.grayAsColor = cv::countNonZero(ch[0] != ch[1]) == 0 && cv::countNonZero(ch[0] != ch[2]) == 0;
    }
    cv::Mat refGray;
    cv::extractChannel(refColor, refGray, 0);
    m.qrs = findQrLabels(refGray);
    m.corners = findCorners(refGray, m.qrs);
    log("QR 라벨 3개, 모서리 " + std::to_string(m.corners.size()) + "개");

    const EccMeasurer ecc(refGray, m.qrs);
    m.shots.resize(ds.shots.size());
    for (size_t i = 0; i < ds.shots.size(); ++i) {
        const cv::Mat gray = (i == ds.centers[0]) ? refGray : readGray(ds.shots[i].file);
        ShotMeasure &s = m.shots[i];
        double sx = 0, sy = 0, plate = 0;
        for (int q = 0; q < kQrCount; ++q) {
            s.ecc[q] = ecc.measure(gray, q);
            sx += s.ecc[q].dx;
            sy += s.ecc[q].dy;
            const cv::Rect moved = m.qrs[q].label + cv::Point(cvRound(s.ecc[q].dx), cvRound(s.ecc[q].dy));
            plate += cv::mean(gray(moved & cv::Rect(0, 0, gray.cols, gray.rows)))[0];
        }
        s.eccDx = sx / kQrCount;
        s.eccDy = sy / kQrCount;
        s.plateMean = plate / kQrCount;
        s.wholeMean = cv::mean(gray)[0];
        s.pos = trackCorners(gray, m.corners, m.qrs, s.ecc);
        std::cout << "  측정 " << i + 1 << " / " << ds.shots.size() << "\r" << std::flush;
    }
    std::cout << "\n";

    // 모든 사진에서 추적된 모서리만 평균에 쓴다 (사진마다 점 집합이 같아야 비교가 된다)
    m.cornerUsable.assign(m.corners.size(), true);
    for (const auto &s : m.shots)
        for (size_t k = 0; k < m.corners.size(); ++k)
            if (!std::isfinite(s.pos[k].x)) m.cornerUsable[k] = false;
    for (auto &s : m.shots) {
        double sx = 0, sy = 0;
        int n = 0;
        for (size_t k = 0; k < m.corners.size(); ++k) {
            if (!m.cornerUsable[k] || m.corners[k].diagonal) continue;
            sx += s.pos[k].x - m.corners[k].p.x;
            sy += s.pos[k].y - m.corners[k].p.y;
            ++n;
        }
        if (n) { s.cornerDx = sx / n; s.cornerDy = sy / n; }
    }
    return m;
}

void writeMeasureCsv(const DataSet &ds, const Measurements &m, const fs::path &dir) {
    Csv shots(dir / "shots.csv", {"file", "group", "center", "y", "t", "rep", "taken_at", "minutes", "ecc_dx", "ecc_dy",
                                  "corner_dx", "corner_dy", "ecc_cc_L", "ecc_cc_C", "ecc_cc_R", "plate_mean", "whole_mean"});
    for (size_t i = 0; i < ds.shots.size(); ++i) {
        const Shot &s = ds.shots[i];
        const ShotMeasure &r = m.shots[i];
        shots.row({s.rel, s.group, s.center ? "1" : "0", num(s.y, 2), num(s.t, 2), std::to_string(s.rep), s.takenAt,
                   num(s.minutes, 3), num(r.eccDx, 4), num(r.eccDy, 4), num(r.cornerDx, 4), num(r.cornerDy, 4),
                   num(r.ecc[0].cc, 5), num(r.ecc[1].cc, 5), num(r.ecc[2].cc, 5), num(r.plateMean, 3), num(r.wholeMean, 3)});
    }
    Csv corners(dir / "corners.csv", {"id", "qr", "x", "y", "diagonal", "usable"});
    for (size_t k = 0; k < m.corners.size(); ++k)
        corners.row({std::to_string(k), kQrNames[m.corners[k].qr], num(m.corners[k].p.x, 3), num(m.corners[k].p.y, 3),
                     m.corners[k].diagonal ? "1" : "0", m.cornerUsable[k] ? "1" : "0"});
}

}  // namespace hw
