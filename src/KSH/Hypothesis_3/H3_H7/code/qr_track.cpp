// =====================================================================================
// qr_track.cpp — QR 모서리 대응점 (기준 영상 → 각 영상)
// =====================================================================================
#include "qr_track.hpp"
#include <chrono>
#include <cmath>
#include <fstream>

namespace h3 {

bool QrTracker::setReference(const cv::Mat& gray8)
{
    if (gray8.empty()) return false;
    const cv::Rect img(0, 0, gray8.cols, gray8.rows);
    if ((p_.qrRoi & img) != p_.qrRoi) return false;
    ref_ = gray8.clone();
    std::vector<cv::Point2f> c;
    cv::goodFeaturesToTrack(ref_(p_.qrRoi), c, p_.maxCorners, p_.quality, p_.minDist);
    if (c.empty()) return false;
    cv::cornerSubPix(ref_(p_.qrRoi), c, cv::Size(5, 5), cv::Size(-1, -1),
                     cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 50, 1e-4));
    P0_.clear();
    for (const auto& pt : c) P0_.push_back(cv::Point2d(pt.x + p_.qrRoi.x, pt.y + p_.qrRoi.y));
    return true;
}

Track QrTracker::track(const cv::Mat& gray8) const
{
    auto t0 = std::chrono::steady_clock::now();
    Track t;
    if (ref_.empty() || gray8.size() != ref_.size()) return t;
    // 거친 이동 : QR 띠 (좌우 여유 포함) 위상 상관
    cv::Rect band(p_.qrRoi.x - p_.coarsePadX, p_.qrRoi.y, p_.qrRoi.width + 2 * p_.coarsePadX, p_.qrRoi.height);
    band &= cv::Rect(0, 0, ref_.cols, ref_.rows);
    cv::Mat a, b, win;
    ref_(band).convertTo(a, CV_32F); gray8(band).convertTo(b, CV_32F);
    cv::createHanningWindow(win, a.size(), CV_32F);
    cv::Point2d cs = cv::phaseCorrelate(a, b, win);
    t.coarseDx = cs.x; t.coarseDy = cs.y;

    std::vector<cv::Point2f> p0, p1, back;
    for (const auto& p : P0_) { p0.push_back(cv::Point2f((float)p.x, (float)p.y)); p1.push_back(cv::Point2f((float)(p.x + cs.x), (float)(p.y + cs.y))); }
    std::vector<uchar> st, st2; std::vector<float> err;
    const cv::TermCriteria crit(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 100, 1e-5);
    const cv::Size ws(p_.lkWin, p_.lkWin);
    cv::calcOpticalFlowPyrLK(ref_, gray8, p0, p1, st, err, ws, 0, crit, cv::OPTFLOW_USE_INITIAL_FLOW);
    back = p0;
    cv::calcOpticalFlowPyrLK(gray8, ref_, p1, back, st2, err, ws, 0, crit, cv::OPTFLOW_USE_INITIAL_FLOW);
    t.q.resize(P0_.size()); t.ok.resize(P0_.size());
    for (size_t k = 0; k < P0_.size(); ++k) {
        double fb = std::hypot(back[k].x - P0_[k].x, back[k].y - P0_[k].y);
        t.q[k] = cv::Point2d(p1[k].x, p1[k].y);
        t.ok[k] = (st[k] && st2[k] && fb < p_.fbMax_px) ? 1 : 0;
    }
    t.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return t;
}

// ------------------------------------------------------------------ 캐시
static const char MAGIC[8] = { 'H', '3', 'T', 'R', 'K', '0', '0', '1' };

bool saveTracks(const std::string& path, const std::vector<cv::Point2d>& P0,
                const std::vector<std::string>& names, const std::vector<Track>& tr)
{
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write(MAGIC, 8);
    int n = (int)P0.size(), m = (int)names.size();
    f.write((const char*)&n, 4); f.write((const char*)&m, 4);
    f.write((const char*)P0.data(), sizeof(cv::Point2d) * n);
    for (int i = 0; i < m; ++i) {
        int L = (int)names[i].size(); f.write((const char*)&L, 4); f.write(names[i].data(), L);
        f.write((const char*)&tr[i].coarseDx, 8); f.write((const char*)&tr[i].coarseDy, 8); f.write((const char*)&tr[i].ms, 8);
        f.write((const char*)tr[i].q.data(), sizeof(cv::Point2d) * n);
        f.write((const char*)tr[i].ok.data(), n);
    }
    return (bool)f;
}

bool loadTracks(const std::string& path, std::vector<cv::Point2d>& P0,
                std::vector<std::string>& names, std::vector<Track>& tr)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char mg[8]; f.read(mg, 8);
    if (!f || std::string(mg, 8) != std::string(MAGIC, 8)) return false;
    int n = 0, m = 0; f.read((char*)&n, 4); f.read((char*)&m, 4);
    if (n <= 0 || m <= 0) return false;
    P0.resize(n); f.read((char*)P0.data(), sizeof(cv::Point2d) * n);
    names.resize(m); tr.resize(m);
    for (int i = 0; i < m; ++i) {
        int L = 0; f.read((char*)&L, 4); names[i].resize(L); f.read(&names[i][0], L);
        f.read((char*)&tr[i].coarseDx, 8); f.read((char*)&tr[i].coarseDy, 8); f.read((char*)&tr[i].ms, 8);
        tr[i].q.resize(n); tr[i].ok.resize(n);
        f.read((char*)tr[i].q.data(), sizeof(cv::Point2d) * n);
        f.read((char*)tr[i].ok.data(), n);
    }
    return (bool)f;
}

} // namespace h3
