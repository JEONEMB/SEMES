// 재학습용 특징 추출 : 사진마다 기준 사진 4장(center, center1, center2, center3) 각각에 대한 s, d
#include "ksh/align_config.hpp"
#include "ksh/features.hpp"
#include "ksh/image_util.hpp"
#include <opencv2/imgcodecs.hpp>
#include <cstdio>
#include <string>
int main(int argc, char **argv) {
    // argv[1..4] = 기준 사진 4장, argv[5..] = 측정 사진
    ksh::AlignConfig cfg; std::string err;
    if (!ksh::load_align_config("config/alignment.yml", cfg, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    cv::Mat ref[4];
    for (int k = 0; k < 4; ++k) ref[k] = ksh::to_gray32(cv::imread(argv[1 + k], cv::IMREAD_GRAYSCALE));
    std::printf("file");
    for (int k = 0; k < 4; ++k) std::printf(",ok%d,s%d,d%d,resp%d,msg%d", k, k, k, k, k);
    std::printf("\n");
    for (int i = 5; i < argc; ++i) {
        cv::Mat cur = ksh::to_gray32(cv::imread(argv[i], cv::IMREAD_GRAYSCALE));
        std::string f = argv[i]; f = f.substr(f.rfind('/') + 1);
        std::printf("\"%s\"", f.c_str());
        for (int k = 0; k < 4; ++k) {
            ksh::FeatureMeasurement m = ksh::measure_features(ref[k], cur, cfg.features);
            std::printf(",%d,%.6f,%.6f,%.4f,\"%s\"", (int)m.ok, m.s_px, m.d_px, m.coarse_response, m.message.c_str());
        }
        std::printf("\n"); std::fflush(stdout);
    }
}
