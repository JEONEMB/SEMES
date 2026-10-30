// =====================================================================================
// yt_verify.cpp — 방법별 검증 실행기 (yt_verify.hpp 참고)
// =====================================================================================
#include "yt_verify.hpp"
#include "yt_common.hpp"
#include "m1_phase_ecc.hpp"
#include "../roi/roi_opt.hpp"
#include "../report/report_media.hpp"
#include "../doe/doe.hpp"

#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace yt {

namespace m2 { bool verify(const std::vector<Sample>& ds, const std::string& outDir, const std::string& m1Csv); }
namespace m4 { bool verify(const std::vector<Sample>& ds, const std::string& outDir, const std::string& m1Csv); }
namespace m3 { bool verify(const std::vector<Sample>& ds, const std::string& outDir); }
namespace trackB { bool verify(const std::vector<Sample>& ds, const std::string& outDir, const std::string& m1Csv); }
namespace trackA { bool verify(const std::vector<Sample>& ds, const std::string& outDir, const std::string& m1Csv,
                               const std::string& m2Csv, const std::string& m4Csv); }

int verifyMain(int argc, char** argv) {
    std::string what = "all", img = "img", out = "results/verify";
    bool force = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--img" && i + 1 < argc) img = argv[++i];
        else if (a == "--out" && i + 1 < argc) out = argv[++i];
        else if (a == "--force") force = true;
        else what = a;
    }
    const auto ds = scanDataset(img);
    int n1 = 0, n2 = 0;
    for (const auto& s : ds) (s.set_id == 1 ? n1 : n2)++;
    std::cout << "데이터 : " << img << " — 1차 " << n1 << " 장, 2차 " << n2 << " 장\n";
    if (ds.empty()) { std::cerr << "사진이 없습니다\n"; return 1; }

    const std::string m1Csv = out + "/M1_phase_ecc/m1_shifts.csv";
    const std::string m2Csv = out + "/M2_icgn/m2_affine.csv";
    const std::string m4Csv = out + "/M4_gabor/m4_slopes.csv";
    struct Step { std::string key, dir; std::function<bool()> run; };
    const std::vector<Step> steps = {
        {"m1", "M1_phase_ecc", [&] { return m1::verify(ds, out + "/M1_phase_ecc"); }},
        {"m2", "M2_icgn", [&] { return m2::verify(ds, out + "/M2_icgn", m1Csv); }},
        {"m4", "M4_gabor", [&] { return m4::verify(ds, out + "/M4_gabor", m1Csv); }},
        {"m3", "M3_grid", [&] { return m3::verify(ds, out + "/M3_grid"); }},
        {"trackB", "TrackB_parallax", [&] { return trackB::verify(ds, out + "/TrackB_parallax", m1Csv); }},
        {"trackA", "TrackA_pseudo_strain", [&] { return trackA::verify(ds, out + "/TrackA_pseudo_strain", m1Csv, m2Csv, m4Csv); }},
        // H5 : 최적 ROI 학습 (후보 58곳 측정, 약 5분 — 측정 캐시가 있으면 수 초) / 보고서 영상 프레임
        {"roi", "H5_roi", [&] { return roi::verify(ds, out + "/H5_roi", "config/roi_search.yml"); }},
        {"media", "H5_media", [&] { return media::makeAll(ds, out + "/H5_media"); }},
        // DOE : 정렬 순서 · 보간 방법 · s·d 식 차수 비교 (변형 28가지 측정 — 첫 실행 수십 분, 측정 캐시가 있으면 수 초)
        {"doe", "DOE", [&] { return doe::run(ds, out + "/DOE", "all"); }},
    };
    bool started = what == "all";
    int rc = 0;
    for (const auto& st : steps) {
        if (!started && st.key != what) continue;
        Timer t;
        const bool ok = st.run();
        std::cout << "  (" << st.key << " " << t.ms() / 1000.0 << " 초)\n";
        if (!ok) {
            rc = 2;
            if (!force && what == "all") { std::cout << "\n" << st.key << " 구현 검증 미통과 → 다음 방법을 멈춥니다 (--force 로 계속)\n"; break; }
        }
        if (what != "all") break;
    }
    return rc;
}

} // namespace yt
