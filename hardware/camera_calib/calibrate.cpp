// 체커보드 사진으로 카메라 내부 파라미터(fx, fy, cx, cy)와 렌즈 왜곡 계수를 구한다.
//
// 사용:
//   camera_calib <사진 폴더> [--cols 11] [--rows 8] [--square 15] [--out camera_calib.yml]
//                [--undistort <사진>]
//
// --cols/--rows 는 체커보드 "내부 코너" 수 (12x9 칸 보드 -> 11x8).
// --undistort 를 주면 결과로 그 사진의 왜곡을 펴고, SEMES 와 같은 화각의 3840x2160 사진도 만든다.

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace
{
// Raspberry Pi HQ Camera (IMX477) 화소 크기
constexpr double kPixelMm = 1.55e-3;

// SEMES 카메라: 12 mm / 2.8 um = 4286 px, 4096 -> 3840 가로만 축소 (x 15/16)
constexpr double kSemesFx = 12.0 / 2.8e-3 * 3840.0 / 4096.0;
constexpr double kSemesFy = 12.0 / 2.8e-3;
const cv::Size kSemesSize(3840, 2160);

// 12 MP 원본에서 바로 찾으면 느려서 축소본에서 찾은 뒤 원본에서 정밀화한다
constexpr int kDetectWidth = 1400;

struct Options
{
    std::string dir;
    cv::Size pattern{11, 8};
    double square = 15.0;
    std::string out = "camera_calib.yml";
    std::string undistort;
};

bool parseArgs(int argc, char **argv, Options &opt)
{
    if (argc < 2)
        return false;
    opt.dir = argv[1];
    for (int i = 2; i + 1 < argc; i += 2)
    {
        std::string key = argv[i];
        std::string val = argv[i + 1];
        if (key == "--cols")
            opt.pattern.width = std::stoi(val);
        else if (key == "--rows")
            opt.pattern.height = std::stoi(val);
        else if (key == "--square")
            opt.square = std::stod(val);
        else if (key == "--out")
            opt.out = val;
        else if (key == "--undistort")
            opt.undistort = val;
        else
            return false;
    }
    return true;
}

bool findCorners(const cv::Mat &gray, cv::Size pattern, std::vector<cv::Point2f> &corners)
{
    double scale = std::min(1.0, static_cast<double>(kDetectWidth) / gray.cols);
    cv::Mat small;
    cv::resize(gray, small, cv::Size(), scale, scale, cv::INTER_AREA);
    // FAST_CHECK: 보드가 없거나 잘린 사진을 빨리 건너뛴다
    int flags = cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK;
    if (!cv::findChessboardCorners(small, pattern, corners, flags))
        return false;

    for (auto &p : corners)
        p *= static_cast<float>(1.0 / scale);
    cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                     cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 50, 0.01));
    return true;
}

// 왜곡을 편 사진에서 SEMES 와 같은 화각이 되도록 잘라낼 크기
cv::Size semesCrop(const cv::Mat &K)
{
    return cv::Size(cvRound(kSemesSize.width * K.at<double>(0, 0) / kSemesFx),
                    cvRound(kSemesSize.height * K.at<double>(1, 1) / kSemesFy));
}

void writeUndistorted(const std::string &path, const cv::Mat &K, const cv::Mat &D)
{
    cv::Mat src = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (src.empty())
    {
        std::fprintf(stderr, "사진을 열 수 없음: %s\n", path.c_str());
        return;
    }
    cv::Mat und;
    cv::undistort(src, und, K, D);
    std::string base = path.substr(0, path.find_last_of('.'));
    cv::imwrite(base + "_undistort.png", und);

    // 주점(cx, cy)을 중심으로 잘라 3840x2160 으로 늘린다 -> SEMES 와 같은 화각과 픽셀 비율
    cv::Size crop = semesCrop(K);
    cv::Rect roi(cvRound(K.at<double>(0, 2) - crop.width / 2.0),
                 cvRound(K.at<double>(1, 2) - crop.height / 2.0), crop.width, crop.height);
    roi &= cv::Rect(0, 0, und.cols, und.rows);
    cv::Mat semes;
    cv::resize(und(roi), semes, kSemesSize, 0, 0, cv::INTER_CUBIC);
    cv::imwrite(base + "_semes3840.png", semes);
    std::printf("저장: %s_undistort.png, %s_semes3840.png (크롭 %dx%d)\n", base.c_str(), base.c_str(),
                roi.width, roi.height);
}
} // namespace

int main(int argc, char **argv)
{
    Options opt;
    if (!parseArgs(argc, argv, opt))
    {
        std::fprintf(stderr,
                     "사용: %s <사진 폴더> [--cols 11] [--rows 8] [--square 15] [--out camera_calib.yml] "
                     "[--undistort <사진>]\n",
                     argv[0]);
        return 1;
    }

    std::vector<cv::String> files;
    cv::glob(opt.dir + "/*.png", files);
    if (files.empty())
    {
        std::fprintf(stderr, "PNG 사진이 없음: %s\n", opt.dir.c_str());
        return 1;
    }

    std::vector<cv::Point3f> board;
    for (int r = 0; r < opt.pattern.height; ++r)
        for (int c = 0; c < opt.pattern.width; ++c)
            board.emplace_back(static_cast<float>(c * opt.square), static_cast<float>(r * opt.square), 0.f);

    std::vector<std::vector<cv::Point3f>> objPts;
    std::vector<std::vector<cv::Point2f>> imgPts;
    std::vector<std::string> used;
    cv::Size imageSize;
    for (const auto &f : files)
    {
        if (f.find("_undistort") != std::string::npos || f.find("_semes3840") != std::string::npos)
            continue;
        cv::Mat gray = cv::imread(f, cv::IMREAD_GRAYSCALE);
        if (gray.empty())
            continue;
        if (imageSize.area() == 0)
            imageSize = gray.size();
        else if (gray.size() != imageSize)
        {
            std::printf("  건너뜀 (크기 다름) %s\n", f.c_str());
            continue;
        }
        std::vector<cv::Point2f> corners;
        bool ok = findCorners(gray, opt.pattern, corners);
        std::printf("  %s %s\n", ok ? "OK  " : "실패", f.c_str());
        if (ok)
        {
            objPts.push_back(board);
            imgPts.push_back(corners);
            used.push_back(f);
        }
    }
    if (imgPts.size() < 5)
    {
        std::fprintf(stderr, "코너를 찾은 사진이 %zu장뿐. 최소 5장, 권장 15장 이상\n", imgPts.size());
        return 1;
    }

    cv::Mat K, D;
    std::vector<cv::Mat> rvecs, tvecs;
    double rms = cv::calibrateCamera(objPts, imgPts, imageSize, K, D, rvecs, tvecs);

    // 사진별 재투영 오차와 보드까지 거리
    std::printf("\n사진별 결과 (재투영 오차 1px 이상이면 그 사진을 빼고 다시 실행 권장)\n");
    std::vector<double> dists;
    for (size_t i = 0; i < used.size(); ++i)
    {
        std::vector<cv::Point2f> proj;
        cv::projectPoints(objPts[i], rvecs[i], tvecs[i], K, D, proj);
        double err = cv::norm(imgPts[i], proj, cv::NORM_L2) / std::sqrt(static_cast<double>(proj.size()));
        double z = tvecs[i].at<double>(2);
        dists.push_back(z);
        std::printf("  %-40s 오차 %.3f px  거리 %.1f mm%s\n", used[i].c_str(), err, z, err > 1.0 ? "  <- 확인" : "");
    }
    std::sort(dists.begin(), dists.end());
    double zMed = dists[dists.size() / 2];

    double fovx, fovy, focalMm, aspect;
    cv::Point2d pp;
    cv::calibrationMatrixValues(K, imageSize, imageSize.width * kPixelMm, imageSize.height * kPixelMm, fovx,
                                fovy, focalMm, pp, aspect);
    cv::Size crop = semesCrop(K);
    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);

    std::printf("\n== 결과 (%zu장 사용, 사진 %dx%d) ==\n", used.size(), imageSize.width, imageSize.height);
    std::printf("RMS 재투영 오차 : %.3f px  (0.5 px 이하면 좋음)\n", rms);
    std::printf("fx, fy         : %.1f, %.1f px  (초점거리 %.2f mm)\n", fx, fy, focalMm);
    std::printf("cx, cy         : %.1f, %.1f px\n", K.at<double>(0, 2), K.at<double>(1, 2));
    std::printf("왜곡 k1 k2 p1 p2 k3 : %.4f %.4f %.5f %.5f %.4f\n", D.at<double>(0), D.at<double>(1),
                D.at<double>(2), D.at<double>(3), D.at<double>(4));
    std::printf("화각           : 가로 %.1f° 세로 %.1f°\n", fovx, fovy);
    std::printf("보드 거리 중앙값 : %.1f mm -> 그 거리에서 %.2f px/mm (%.4f mm/px)\n", zMed, fx / zMed, zMed / fx);
    std::printf("SEMES 같은 화각 : 왜곡 편 사진에서 %dx%d 를 잘라 3840x2160 으로\n", crop.width, crop.height);
    // 위 크롭을 쓰면 3840 이미지의 fx 가 SEMES 와 같아지므로, 감도 14.25 px/mm 가 되는 거리도 SEMES 와 같다
    std::printf("크롭 후 SEMES 감도 14.25 px/mm 가 되는 거리 : %.1f mm (광학 중심 ~ 벽)\n", kSemesFx / 14.25);

    cv::FileStorage fs(opt.out, cv::FileStorage::WRITE);
    fs << "image_width" << imageSize.width << "image_height" << imageSize.height;
    fs << "pattern_cols" << opt.pattern.width << "pattern_rows" << opt.pattern.height;
    fs << "square_mm" << opt.square << "images_used" << static_cast<int>(used.size());
    fs << "rms_px" << rms << "camera_matrix" << K << "dist_coeffs" << D;
    fs << "fov_x_deg" << fovx << "fov_y_deg" << fovy << "focal_mm" << focalMm;
    fs << "board_distance_median_mm" << zMed;
    fs << "semes_crop_width" << crop.width << "semes_crop_height" << crop.height;
    fs.release();
    std::printf("\n저장: %s\n", opt.out.c_str());

    if (!opt.undistort.empty())
        writeUndistorted(opt.undistort, K, D);
    return 0;
}
