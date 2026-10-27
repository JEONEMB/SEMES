// =====================================================================================
// qr_track.hpp — QR 모서리 대응점 (기준 영상 → 각 영상)
//
//  1) 기준 영상(center.bmp)의 QR 3개 영역에서 모서리 1,251개 (goodFeaturesToTrack + cornerSubPix)
//  2) 플레이트 띠 위상 상관(phaseCorrelate)으로 거친 이동 → LK 광류 초기값
//  3) LK 광류 (창 15×15, 피라미드 없음) 순방향 + 역방향 → 왕복 오차 0.1 px 이내만 유효
//  좌표는 모두 전체 화면(3840×2160) 기준. 결과는 캐시 파일로 저장해 다시 쓸 수 있음.
// =====================================================================================
#ifndef __H3_QR_TRACK_HPP__
#define __H3_QR_TRACK_HPP__

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace h3 {

struct TrackParams {
    cv::Rect qrRoi   = cv::Rect(1530, 955, 750, 235);    // QR 3개 영역 (모서리 추출)
    int      coarsePadX = 260;                          // 거친 이동용 띠 : qrRoi 좌우 여유
    int      maxCorners = 3000;
    double   quality = 0.02, minDist = 6.0;
    int      lkWin = 15;
    double   fbMax_px = 0.1;                            // 순방향-역방향 왕복 오차 허용
};

// 한 영상의 대응점
struct Track {
    std::vector<cv::Point2d> q;     // 현재 영상 위치 (기준점 순서)
    std::vector<uchar> ok;          // 유효 여부
    double coarseDx = 0, coarseDy = 0;
    double ms = 0;
    int nValid() const { int n = 0; for (uchar o : ok) n += o; return n; }
};

class QrTracker {
public:
    explicit QrTracker(const TrackParams& p = TrackParams()) : p_(p) {}
    bool setReference(const cv::Mat& gray8);
    Track track(const cv::Mat& gray8) const;
    const std::vector<cv::Point2d>& refPoints() const { return P0_; }
    void setRefPoints(const std::vector<cv::Point2d>& p) { P0_ = p; }
private:
    TrackParams p_;
    cv::Mat ref_;
    std::vector<cv::Point2d> P0_;
};

// 캐시 (이진 파일) : 기준점 + 영상별 대응점
bool saveTracks(const std::string& path, const std::vector<cv::Point2d>& P0,
                const std::vector<std::string>& names, const std::vector<Track>& tr);
bool loadTracks(const std::string& path, std::vector<cv::Point2d>& P0,
                std::vector<std::string>& names, std::vector<Track>& tr);

} // namespace h3

#endif
