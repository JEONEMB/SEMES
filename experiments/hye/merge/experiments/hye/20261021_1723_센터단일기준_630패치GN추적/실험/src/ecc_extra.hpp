// multiref_4fold_20261021/measure_extra.py : QR 상자별 ECC affine / homography, Hanning 위상상관을 630점에 투영
// ECC 는 Python cv2 와 같은 semes::ecc5h (ecc_compat_h.cpp: 공통 ecc_compat 사본, homography 역변환도 정확한 bilinear)
#pragma once
#include "ecc_compat_h.hpp"
#include "yt630.hpp"

namespace yt {

struct ExtraTracks {
    // 0 ecc_affine, 1 ecc_homography, 2 phase : [img][pt] (중심 뺀 값)
    std::vector<std::vector<double>> Dx[3], Dy[3];
    std::vector<std::string> fallbacks;
};

inline ExtraTracks measure_extra(const Extractor& ex, const std::vector<cv::Mat>& crops, const std::vector<Img>& imgs) {
    size_t M = crops.size(), N = ex.gx.size();
    ExtraTracks E;
    std::vector<std::vector<double>> rx[3], ry[3];
    for (int t = 0; t < 3; ++t) { rx[t].assign(M, std::vector<double>(N)); ry[t].assign(M, std::vector<double>(N)); }
    std::vector<cv::Mat> tmpl(3);
    std::vector<std::vector<int>> mem = ex.members;
    for (int k = 0; k < 3; ++k) ex.ref(cv::Range(ex.box[k][1], ex.box[k][3]), cv::Range(ex.box[k][0], ex.box[k][2])).convertTo(tmpl[k], CV_32F, 1.0 / 255);
    std::vector<std::string> fb(M);
#pragma omp parallel for schedule(dynamic) num_threads(4)
    for (int i = 0; i < (int)M; ++i) {
        auto sh = ex.coarse(crops[i]);
        for (int k = 0; k < 3; ++k) {
            int dx = sh[k][0], dy = sh[k][1];
            cv::Mat target;
            crops[i](cv::Range(ex.box[k][1] + dy, ex.box[k][3] + dy), cv::Range(ex.box[k][0] + dx, ex.box[k][2] + dx)).convertTo(target, CV_32F, 1.0 / 255);
            for (int t = 0; t < 2; ++t) {
                int motion = t == 0 ? cv::MOTION_AFFINE : cv::MOTION_HOMOGRAPHY;
                cv::Mat warp = t == 0 ? cv::Mat::eye(2, 3, CV_32F) : cv::Mat::eye(3, 3, CV_32F);
                bool ok = true;
                try {
                    semes::ecc5h::findTransformECCWithMask(tmpl[k], target, cv::noArray(), cv::noArray(), warp, motion,
                                                          cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 100, 1e-7), 3);
                } catch (const cv::Exception&) { ok = false; }
                for (int p : mem[k]) {
                    // 점 = 전체 좌표 - 상자 원점 (float64)
                    double px = ex.gx[p] - BOXES[k][0], py = ex.gy[p] - BOXES[k][1];
                    double qx = px, qy = py;
                    if (ok) {
                        // q = [p,1] @ warp.T  (warp float32 -> float64 곱)
                        double w00 = warp.at<float>(0, 0), w01 = warp.at<float>(0, 1), w02 = warp.at<float>(0, 2);
                        double w10 = warp.at<float>(1, 0), w11 = warp.at<float>(1, 1), w12 = warp.at<float>(1, 2);
                        qx = px * w00 + py * w01 + w02;
                        qy = px * w10 + py * w11 + w12;
                        if (t == 1) {
                            double z = px * warp.at<float>(2, 0) + py * warp.at<float>(2, 1) + warp.at<float>(2, 2);
                            qx /= z; qy /= z;
                        }
                        rx[t][i][p] = qx - px + dx; ry[t][i][p] = qy - py + dy;
                    } else {
#pragma omp critical
                        fb[i] += (fb[i].empty() ? "" : ";") + std::string(t == 0 ? "ecc_affine" : "ecc_homography") + " QR" + std::to_string(k + 1);
                        rx[t][i][p] = dx; ry[t][i][p] = dy;
                    }
                }
            }
            cv::Mat win;
            cv::createHanningWindow(win, tmpl[k].size(), CV_32F);
            cv::Point2d d = cv::phaseCorrelate(tmpl[k], target, win);
            for (int p : mem[k]) { rx[2][i][p] = dx + d.x; ry[2][i][p] = dy + d.y; }
        }
    }
    for (int t = 0; t < 3; ++t) {
        E.Dx[t] = rx[t]; E.Dy[t] = ry[t];
        for (size_t i = 0; i < M; ++i)
            for (size_t p = 0; p < N; ++p) { E.Dx[t][i][p] -= rx[t][0][p]; E.Dy[t][i][p] -= ry[t][0][p]; }
    }
    for (size_t i = 0; i < M; ++i) if (!fb[i].empty()) E.fallbacks.push_back(imgs[i].file + ": " + fb[i]);
    return E;
}

}  // namespace yt
