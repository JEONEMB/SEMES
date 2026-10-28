// =====================================================================================
// main.cpp — alignment() 사용 예와 정확도 평가
//
//  실행 방법 (작업 폴더 = 이 프로젝트 폴더)
//   1) 인자 없음                  : alignment.yml 의 demo_image 한 장을 추정하고 결과 창 표시
//   2) <사진> [기준 사진]          : 지정한 사진 한 장 추정 (기준을 주지 않으면 reference_path)
//   3) --eval [폴더]               : 라벨이 붙은 사진 전체를 추정해 허용 범위 합격 수·RMS·처리 시간 보고
//                                   1차 = 학습에 쓴 데이터(참고), 2차 = 학습에 쓰지 않은 시험 데이터
// =====================================================================================
#include <iostream>
#include <opencv2/opencv.hpp>

#include "alignment.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <regex>
#include <string>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>   // 콘솔 한글 출력 (UTF-8)
#endif

namespace fs = std::filesystem;

namespace {

const char *kConfigPath = "config/alignment.yml";

struct DemoCfg {
    std::string demo_image, demo_reference, eval_dir, ref_set1, ref_set2;
    double tol_y_um = 0.0, tol_t_deg = 0.0;
    int display_width_px = 0;
};

bool read_demo_cfg(DemoCfg &c) {
    cv::FileStorage f(kConfigPath, cv::FileStorage::READ);
    if (!f.isOpened()) return false;
    f["demo_image"] >> c.demo_image;
    f["demo_reference"] >> c.demo_reference;
    f["eval_dir"] >> c.eval_dir;
    f["eval_reference_set1"] >> c.ref_set1;
    f["eval_reference_set2"] >> c.ref_set2;
    c.tol_y_um = double(f["tolerance_y_um"]);
    c.tol_t_deg = double(f["tolerance_t_deg"]);
    c.display_width_px = int(f["display_width_px"]);
    return true;
}

// 파일 이름 → 라벨 (정답). 'T+0.01' 의 '.01' 을 확장자로 오인하지 않도록 확장자를 떼고 호출
struct Label { double y_mm = 0, t_deg = 0; int set = 1; bool is_ref = false; };
bool parse_label(const std::string &stem, Label &L) {
    static const std::regex re_center(R"(^center$)"), re_center_n(R"(^center\d+$)");
    static const std::regex re_mixed(R"(^Y([+-]?\d+(?:\.\d+)?)\s+T([+-]?\d+(?:\.\d+)?)\s*\(\d+\)$)");
    static const std::regex re_y(R"(^Y([+-]?\d+(?:\.\d+)?)$)"), re_t(R"(^T([+-]?\d+(?:\.\d+)?)$)");
    std::smatch m;
    L = Label();
    if (std::regex_match(stem, re_center)) { L.is_ref = true; return true; }
    if (std::regex_match(stem, re_center_n)) { L.set = 2; L.is_ref = true; return true; }
    if (std::regex_match(stem, m, re_mixed)) { L.y_mm = std::stod(m[1]); L.t_deg = std::stod(m[2]); L.set = 2; return true; }
    if (std::regex_match(stem, m, re_y)) { L.y_mm = std::stod(m[1]); return true; }
    if (std::regex_match(stem, m, re_t)) { L.t_deg = std::stod(m[1]); return true; }
    return false;
}

// 결과 창 : ROI 를 측정 위치에 그리고 추정값 표시
void show_result(const cv::Mat &img, const Alignment_t &r, int display_width_px) {
    cv::Mat view;
    if (img.channels() == 1) cv::cvtColor(img, view, cv::COLOR_GRAY2BGR);
    else view = img.clone();
    const AlignmentInfo &info = alignment_last_info();
    for (const auto &roi : info.rois) {
        const cv::Point off(int(std::lround(roi.shift_px.x)), int(std::lround(roi.shift_px.y)));
        const cv::Scalar col = roi.ok ? cv::Scalar(0, 200, 0) : cv::Scalar(0, 0, 255);
        cv::rectangle(view, roi.ref_rect_px + off, col, 4);
        cv::putText(view, roi.name, roi.ref_rect_px.tl() + off + cv::Point(0, -12), cv::FONT_HERSHEY_SIMPLEX, 1.4, col, 3);
    }
    char buf[160];
    if (info.ok) std::snprintf(buf, sizeof(buf), "Y = %+.4f mm   Theta = %+.4f deg   (%.0f ms)", r.y, r.t, info.elapsed_ms);
    else std::snprintf(buf, sizeof(buf), "FAILED");
    cv::putText(view, buf, cv::Point(60, 120), cv::FONT_HERSHEY_SIMPLEX, 3.0, cv::Scalar(0, 0, 0), 12);
    cv::putText(view, buf, cv::Point(60, 120), cv::FONT_HERSHEY_SIMPLEX, 3.0, cv::Scalar(255, 255, 255), 4);
    if (display_width_px > 0 && view.cols > display_width_px)
        cv::resize(view, view, cv::Size(display_width_px, view.rows * display_width_px / view.cols), 0, 0, cv::INTER_AREA);
    cv::imshow("Alignment", view);
    cv::waitKey(0);
}

// 콘솔 표 정렬 : 한글·기호(2칸)와 영문(1칸)의 화면 폭을 따로 셈
std::string pad(const std::string &s, int width, bool right = false) {
    int w = 0;
    for (size_t i = 0; i < s.size();) {
        const unsigned char ch = static_cast<unsigned char>(s[i]);
        const int len = ch < 0x80 ? 1 : (ch >> 5) == 6 ? 2 : (ch >> 4) == 14 ? 3 : 4;
        w += (len >= 3 && !(ch == 0xE2 || ch == 0xCE)) ? 2 : 1;   // 한글 등 = 2칸, Θ·≤·° 같은 기호 = 1칸
        i += len;
    }
    const std::string sp(std::max(0, width - w), ' ');
    return right ? sp + s : s + sp;
}
std::string num(double v, int prec) {
    char b[32];
    std::snprintf(b, sizeof(b), "%.*f", prec, v);
    return b;
}

// ---------------------------------------------------------------- 평가
struct Row { std::string stem, group; Label L; double y_est = NAN, t_est = NAN, ms = 0; bool ok = false; };

int run_eval(const std::string &dir, const DemoCfg &c) {
    std::map<std::string, std::string> files;   // stem → 경로 (bmp 우선)
    for (const auto &e : fs::directory_iterator(dir)) {
        if (!e.is_regular_file()) continue;
        std::string ext = e.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char ch) { return char(std::tolower(ch)); });
        if (ext != ".bmp" && ext != ".png") continue;
        const std::string stem = e.path().stem().string();
        if (!files.count(stem) || ext == ".bmp") files[stem] = e.path().string();
    }
    if (!files.count(c.ref_set1) || !files.count(c.ref_set2)) {
        std::cerr << "기준 사진(" << c.ref_set1 << ", " << c.ref_set2 << ")이 " << dir << " 에 없음\n";
        return 1;
    }
    std::vector<Row> rows;
    for (int set = 1; set <= 2; ++set) {
        if (!alignment_set_reference(cv::imread(files[set == 1 ? c.ref_set1 : c.ref_set2], cv::IMREAD_GRAYSCALE))) return 1;
        int n = 0;
        for (const auto &f : files) {
            Row r;
            r.stem = f.first;
            if (!parse_label(r.stem, r.L) || r.L.is_ref || r.L.set != set) continue;
            cv::Mat img = cv::imread(f.second, cv::IMREAD_GRAYSCALE);
            const Alignment_t a = alignment(img);
            r.y_est = a.y; r.t_est = a.t; r.ok = alignment_last_info().ok; r.ms = alignment_last_info().elapsed_ms;
            if (set == 1) r.group = "1차 전체 (학습 데이터, 참고)";
            else if (std::abs(r.L.t_deg) <= 0.2 + 1e-9) r.group = "2차 목표 근처 Θ ≤ 0.2°";
            else if (std::abs(std::abs(r.L.t_deg) - 1.0) < 1e-9) r.group = "2차 Θ = 1°";
            else r.group = "2차 Θ = 2° (학습 범위 밖)";
            rows.push_back(r);
            if (++n % 50 == 0) std::cout << "  " << set << "차 " << n << " 장\n";
        }
    }
    // 결과 CSV
    fs::create_directories("results");
    std::ofstream csv("results/alignment_eval.csv");
    csv << "\xEF\xBB\xBF" << "stem,group,y_mm,t_deg,y_est_mm,t_est_deg,err_y_um,err_t_deg,in_tol,ms\n";
    std::map<std::string, std::vector<const Row *>> groups;
    for (const auto &r : rows) {
        const double ey_um = (r.y_est - r.L.y_mm) * 1000.0, et_deg = r.t_est - r.L.t_deg;
        const bool in = r.ok && std::abs(ey_um) <= c.tol_y_um && std::abs(et_deg) <= c.tol_t_deg;
        csv << '"' << r.stem << "\",\"" << r.group << "\"," << r.L.y_mm << ',' << r.L.t_deg << ',' << std::setprecision(10) << r.y_est << ','
            << r.t_est << ',' << ey_um << ',' << et_deg << ',' << int(in) << ',' << r.ms << '\n';
        groups[r.group].push_back(&r);
        if (r.L.set == 2) groups["2차 전체"].push_back(&r);
    }
    // 보고
    std::cout << "\n허용 범위 : |Y 오차| ≤ " << c.tol_y_um << " µm 이고 |Θ 오차| ≤ " << c.tol_t_deg << "°\n";
    std::cout << pad("묶음", 34) << pad("허용 안", 10, true) << pad("Θ RMS(°)", 11, true) << pad("Θ 최대(°)", 11, true)
              << pad("Y RMS(µm)", 11, true) << pad("Y 최대(µm)", 12, true) << pad("시간 평균(ms)", 15, true) << "\n";
    for (const auto &g : groups) {
        int in = 0, fail = 0;
        double st = 0, sy = 0, mt = 0, my = 0, ms = 0;
        for (const Row *r : g.second) {
            ms += r->ms;
            if (!r->ok) { ++fail; continue; }
            const double ey = (r->y_est - r->L.y_mm) * 1000.0, et = r->t_est - r->L.t_deg;
            st += et * et; sy += ey * ey; mt = std::max(mt, std::abs(et)); my = std::max(my, std::abs(ey));
            if (std::abs(ey) <= c.tol_y_um && std::abs(et) <= c.tol_t_deg) ++in;
        }
        const int n = int(g.second.size()), nok = n - fail;
        if (n == 0) continue;
        std::cout << pad(g.first, 34) << pad(std::to_string(in) + " / " + std::to_string(n), 10, true)
                  << pad(nok ? num(std::sqrt(st / nok), 4) : "-", 11, true) << pad(num(mt, 4), 11, true)
                  << pad(nok ? num(std::sqrt(sy / nok), 1) : "-", 11, true) << pad(num(my, 1), 12, true) << pad(num(ms / n, 1), 15, true);
        if (fail) std::cout << "  (측정 실패 " << fail << ")";
        std::cout << "\n";
    }
    std::cout << "\n사진별 결과 : results/alignment_eval.csv\n";
    return 0;
}

} // namespace

int main(int argc, char **argv)
{
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    std::cout << "OpenCV version: "
              << CV_VERSION << std::endl;

    DemoCfg cfg;
    if (!read_demo_cfg(cfg) || !alignment_init(kConfigPath)) {
        std::cerr << "설정 파일 " << kConfigPath << " 을 읽지 못함 (작업 폴더를 프로젝트 폴더로 설정)\n";
        return 1;
    }

    // 3) 평가
    if (argc >= 2 && std::string(argv[1]) == "--eval")
        return run_eval(argc >= 3 ? argv[2] : cfg.eval_dir, cfg);

    // 1) · 2) 한 장 추정
    const std::string image_path = argc >= 2 ? argv[1] : cfg.demo_image;
    const std::string ref_path = argc >= 3 ? argv[2] : (argc >= 2 ? std::string() : cfg.demo_reference);
    if (!ref_path.empty() && !alignment_set_reference(cv::imread(ref_path, cv::IMREAD_GRAYSCALE))) {
        std::cerr << "기준 사진을 읽지 못함 : " << ref_path << "\n";
        return 1;
    }
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "사진을 읽지 못함 : " << image_path << "\n";
        return 1;
    }

    Alignment_t alignment_result = alignment(image);

    const AlignmentInfo &info = alignment_last_info();
    std::cout << "Image: " << image_path << std::endl;
    if (!info.ok) {
        std::cout << "Alignment failed: " << info.message << std::endl;
    } else {
        std::cout << std::fixed << std::setprecision(4)
                  << "Alignment result: y = " << alignment_result.y << " mm"
                  << ", t = " << alignment_result.t << " deg" << std::endl
                  << "Correction move : y = " << -alignment_result.y << " mm"
                  << ", t = " << -alignment_result.t << " deg" << std::endl
                  << std::setprecision(3) << "Features        : s = " << info.s_px << " px, d = " << info.d_px << " px" << std::endl
                  << std::setprecision(1) << "Elapsed         : " << info.elapsed_ms << " ms" << std::endl;
        Label L;
        const std::string stem = fs::path(image_path).stem().string();
        if (parse_label(stem, L) && !L.is_ref)
            std::cout << std::setprecision(4) << "Label (truth)   : y = " << L.y_mm << " mm, t = " << L.t_deg << " deg  → error "
                      << std::setprecision(1) << (alignment_result.y - L.y_mm) * 1000.0 << " um, " << std::setprecision(4)
                      << alignment_result.t - L.t_deg << " deg" << std::endl;
    }
    show_result(image, alignment_result, cfg.display_width_px);

    return 0;
}
