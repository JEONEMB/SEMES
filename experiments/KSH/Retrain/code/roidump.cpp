// ROI 단위 상세 : 사진마다 (기준 k) coarse x/y, ROI 5곳 shift x/y, zncc
#include "ksh/align_config.hpp"
#include "ksh/features.hpp"
#include "ksh/image_util.hpp"
#include <opencv2/imgcodecs.hpp>
#include <cstdio>
#include <string>
int main(int argc, char **argv) {
    ksh::AlignConfig cfg; std::string err;
    if (!ksh::load_align_config("config/alignment.yml", cfg, err)) return 1;
    cv::Mat ref = ksh::to_gray32(cv::imread(argv[1], cv::IMREAD_GRAYSCALE));
    std::printf("file,ok,cx,cy,resp,s,d");
    const char *nm[5] = {"QL", "QC", "QR", "W2", "W3"};
    for (auto n : nm) std::printf(",%s_x,%s_y,%s_z,%s_r", n, n, n, n);
    std::printf("\n");
    for (int i = 2; i < argc; ++i) {
        cv::Mat cur = ksh::to_gray32(cv::imread(argv[i], cv::IMREAD_GRAYSCALE));
        ksh::FeatureMeasurement m = ksh::measure_features(ref, cur, cfg.features);
        std::string f = argv[i]; f = f.substr(f.rfind('/') + 1);
        std::printf("\"%s\",%d,%.4f,%.4f,%.4f,%.5f,%.5f", f.c_str(), (int)m.ok, m.coarse_px.x, m.coarse_px.y, m.coarse_response, m.s_px, m.d_px);
        for (size_t k = 0; k < 5; ++k) {
            if (k < m.rois.size()) std::printf(",%.5f,%.5f,%.5f,%.4f", m.rois[k].shift_px.x, m.rois[k].shift_px.y, m.rois[k].zncc, m.rois[k].response);
            else std::printf(",,,,");
        }
        std::printf("\n"); std::fflush(stdout);
    }
}
