// integrated_eval.cpp — 통합 정렬기 정확도 평가 (라벨 = 파일 이름)
//  integrated_eval [--config 파일] --dir 폴더 [--ref 기준사진] [--no-verify] [--out 결과.csv]
//  폴더의 라벨 사진(center 류 제외)을 기준 사진 하나로 평가 → 허용 안(±45 µm·±0.005°), 과제 허용(±90 µm·±0.01°),
//  틱(±0.05 mm·±0.01°), J, RMS, 검증 통과·거부 수, 처리 시간
#include "integrated/integrated.hpp"
#include "ksh/dataset.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

int main(int argc, char **argv) {
    std::string cfg = "config/integrated.yml", dir, ref, out = "integrated_eval.csv";
    bool no_verify = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--config" && i + 1 < argc) cfg = argv[++i];
        else if (a == "--dir" && i + 1 < argc) dir = argv[++i];
        else if (a == "--ref" && i + 1 < argc) ref = argv[++i];
        else if (a == "--out" && i + 1 < argc) out = argv[++i];
        else if (a == "--no-verify") no_verify = true;
        else { std::cerr << "알 수 없는 인자 : " << a << "\n"; return 1; }
    }
    if (dir.empty()) { std::cerr << "사용법 : integrated_eval [--config 파일] --dir 폴더 [--ref 기준사진] [--no-verify] [--out 결과.csv]\n"; return 1; }
    integrated::Aligner al;
    std::string err;
    if (!al.load(cfg, err, no_verify ? 0 : -1)) { std::cerr << err << "\n"; return 1; }
    const integrated::Settings &s = al.settings();
    if (!ref.empty()) {
        const cv::Mat r = cv::imread(ref, cv::IMREAD_COLOR);
        if (!al.set_reference(r, err)) { std::cerr << err << "\n"; return 1; }
    }
    const auto photos = ksh::list_labeled_photos(dir);
    std::ofstream csv(out);
    csv << "stem,y_mm,t_deg,y_est_mm,t_est_deg,ok,estimate_ok,verified,guard_ok,plate_found,payload_match,plate_shift_diff_px,corner_s_diff_px,ms_estimate,ms_verify,message\n";
    int n = 0, pass = 0, spec = 0, tick = 0, fail = 0, rejected = 0, unverified = 0;
    double sum_j = 0, sy = 0, st = 0, ms_e = 0, ms_v = 0, worst_plate = 0, worst_corner = 0;
    const double cap = s.ksh.evaluation.score_cap > 0 ? s.ksh.evaluation.score_cap : 100.0;
    for (const auto &p : photos) {
        if (p.is_reference) continue;
        const cv::Mat img = cv::imread(p.path, cv::IMREAD_COLOR);
        const integrated::Result r = al.estimate(img);
        ++n;
        const double nan = std::numeric_limits<double>::quiet_NaN();   // MSVC v141 에서 NAN 매크로가 이 줄에서 컴파일되지 않음
        double ey = nan, et = nan, j = 2 * cap * cap;
        if (r.ok) {
            ey = (r.y_mm - p.y_mm) * 1000.0; et = r.t_deg - p.t_deg;
            const double a = std::min(cap, std::abs(ey) / s.ksh.tolerance.y_um), b = std::min(cap, std::abs(et) / s.ksh.tolerance.t_deg);
            j = a * a + b * b; sy += ey * ey; st += et * et;
            pass += std::abs(ey) <= s.ksh.tolerance.y_um && std::abs(et) <= s.ksh.tolerance.t_deg;
            spec += std::abs(ey) <= s.ksh.spec_tolerance.y_um && std::abs(et) <= s.ksh.spec_tolerance.t_deg;
            tick += std::abs(ey) <= 50.0 && std::abs(et) <= 0.01;
        } else {
            ++fail;
            rejected += r.estimate_ok && !r.guard_ok;
        }
        unverified += r.plate_checked && r.guard_ok && !r.verified;
        if (std::isfinite(r.plate_shift_diff_px)) worst_plate = std::max(worst_plate, std::abs(r.plate_shift_diff_px));
        if (std::isfinite(r.corner_s_diff_px)) worst_corner = std::max(worst_corner, std::abs(r.corner_s_diff_px));
        sum_j += j; ms_e += r.ms_estimate; ms_v += r.ms_verify;
        csv << p.stem << "," << p.y_mm << "," << p.t_deg << "," << r.y_mm << "," << r.t_deg << "," << r.ok << "," << r.estimate_ok << "," << r.verified << "," << r.guard_ok << ","
            << r.plate_found << "," << r.payload_match << "," << r.plate_shift_diff_px << "," << r.corner_s_diff_px << "," << r.ms_estimate << "," << r.ms_verify << ",\"" << r.message << "\"\n";
    }
    const int ok_n = n - fail;
    std::cout << std::fixed << std::setprecision(3)
              << "사진 " << n << "장 | 허용 안(±" << s.ksh.tolerance.y_um << " µm·±" << s.ksh.tolerance.t_deg << "°) " << pass << " | 과제 허용 " << spec << " | 틱(±0.05 mm·±0.01°) " << tick
              << " | J " << std::sqrt(sum_j / std::max(1, n)) << "\n"
              << "Y RMS " << std::sqrt(sy / std::max(1, ok_n)) << " µm | Θ RMS " << std::setprecision(4) << std::sqrt(st / std::max(1, ok_n)) << "° | 실패 " << fail << " (검증 모순 거부 " << rejected << ") | 검증 못 함 " << unverified << "장\n"
              << std::setprecision(2) << "검증 최대 차 : 판 중심 " << worst_plate << " px, 코너 QR " << std::setprecision(4) << worst_corner << " px\n"
              << std::setprecision(1) << "시간 (사진당) : 추정 " << ms_e / std::max(1, n) << " ms, 검증 " << ms_v / std::max(1, n) << " ms\n"
              << "사진별 결과 : " << out << "\n";
    return 0;
}
