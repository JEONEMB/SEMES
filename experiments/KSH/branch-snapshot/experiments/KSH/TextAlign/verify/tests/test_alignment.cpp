// =====================================================================================
// test_alignment.cpp — 단위 테스트 (사진 데이터 없이 실행)
//
//  합성 장면 : 어두운 QR 띠 + 밝은 사각형 3개(안에 QR 모듈 무늬) + 뒷벽 어두운 홀 4곳
//             면적 비율(앤티에일리어싱)로 서브픽셀 위치에 그려 정답 좌표를 정확히 안다
//  T1 라벨 이름 해석              T2 QR·홀 중심 정확도 (< 0.05 px)
//  T3 서브픽셀 평행이동 → s 변화    T4 대략 위치 : 큰 이동은 찾고, 범위 밖은 실패 메시지
//  T5 홀 가림 → 실패 메시지        T6 보정식 : 세트별 상수 학습이 기울기를 복원, 역산 왕복
//  T7 설정 파일 : ROI·영점 고쳐 쓰기 → 다시 읽기, 다른 줄 보존, .bak
//  T8 alignment() : 빈 사진 NaN, 초기 세팅 후 센터 = (0, 0), 이동 사진 추정
//  실행 : 작업 폴더 = Code (config/alignment.yml 을 복사해 씀). 실패가 하나라도 있으면 종료 코드 1
// =====================================================================================
#include "alignment.hpp"
#include "ksh/align_config.hpp"
#include "ksh/dataset.hpp"
#include "ksh/features.hpp"
#include "ksh/pose_model.hpp"
#include "ksh/setup.hpp"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>   // 콘솔 한글 출력 (UTF-8)
#endif

namespace fs = std::filesystem;

namespace {

int g_fail = 0, g_pass = 0;
void check(bool ok, const std::string &what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    (ok ? g_pass : g_fail)++;
}

// 화소 i 는 [i − 0.5, i + 0.5] 를 덮음 (화소 중심 = 정수 좌표, 측정 코드와 같은 약속)
double overlap(double a0, double a1, int i) { return std::max(0.0, std::min(a1, i + 0.5) - std::max(a0, i - 0.5)); }

// 축 정렬 사각형 [x0,x1]×[y0,y1] 을 면적 비율로 칠함
void fill_rect_aa(cv::Mat &f, double x0, double y0, double x1, double y1, float v) {
    for (int y = int(std::floor(y0)) - 1; y <= int(std::ceil(y1)) + 1; ++y) {
        const double cy = overlap(y0, y1, y);
        if (cy <= 0) continue;
        float *p = f.ptr<float>(y);
        for (int x = int(std::floor(x0)) - 1; x <= int(std::ceil(x1)) + 1; ++x) {
            const double a = cy * overlap(x0, x1, x);
            if (a > 0) p[x] = float(p[x] * (1.0 - a) + v * a);
        }
    }
}

// 원 (8×8 표본으로 면적 비율)
void fill_disc_aa(cv::Mat &f, double cx, double cy, double r, float v) {
    for (int y = int(cy - r) - 2; y <= int(cy + r) + 2; ++y) {
        float *p = f.ptr<float>(y);
        for (int x = int(cx - r) - 2; x <= int(cx + r) + 2; ++x) {
            int in = 0;
            for (int sy = 0; sy < 8; ++sy)
                for (int sx = 0; sx < 8; ++sx) {
                    const double px = x - 0.5 + (sx + 0.5) / 8.0 - cx, py = y - 0.5 + (sy + 0.5) / 8.0 - cy;
                    in += px * px + py * py <= r * r;
                }
            const double a = in / 64.0;
            if (a > 0) p[x] = float(p[x] * (1.0 - a) + v * a);
        }
    }
}

struct Scene {
    std::vector<cv::Point2d> qr = {{1650.30, 1065.20}, {1907.55, 1065.70}, {2165.10, 1066.10}};
    std::vector<cv::Point2d> holes = {{966.40, 1358.25}, {2865.15, 1359.60}, {1009.70, 1684.35}, {2821.20, 1685.05}};
    double qr_side = 216, hole_r = 22;
    bool hide_w2 = false;
    double optics_sigma_px = 1.2;
};

// 장면 전체를 dx (가로) 만큼 옮겨 그림. 홀은 시차만큼 따로 hole_dx 를 더 옮김
cv::Mat render(const Scene &S, double dx, double hole_dx = 0.0) {
    cv::Mat f(2160, 3840, CV_32F, cv::Scalar(150));
    fill_rect_aa(f, 1300 + dx, 880, 2520 + dx, 1250, 185);          // 플레이트 (밝음)
    fill_rect_aa(f, 1400 + dx, 930, 2420 + dx, 1200, 60);           // QR 띠 (어두움, 사각형보다 높게)
    std::mt19937 rng(7);
    for (const auto &q : S.qr) {
        const double x = q.x + dx, y = q.y, h = S.qr_side / 2;
        fill_rect_aa(f, x - h, y - h, x + h, y + h, 230);
        for (int i = 0; i < 19; ++i)                                  // QR 모듈 (가장자리 16 px 여백)
            for (int j = 0; j < 19; ++j)
                if (rng() % 2) fill_rect_aa(f, x - h + 16 + j * 9.7, y - h + 16 + i * 9.7, x - h + 16 + (j + 1) * 9.7, y - h + 16 + (i + 1) * 9.7, 40);
    }
    for (size_t k = 0; k < S.holes.size(); ++k) {
        if (S.hide_w2 && k == 1) continue;
        fill_disc_aa(f, S.holes[k].x + dx + hole_dx, S.holes[k].y, S.hole_r, 35);
    }
    cv::GaussianBlur(f, f, cv::Size(0, 0), S.optics_sigma_px);   // 렌즈 흐림 (실사진의 변 폭과 비슷하게)
    cv::Mat g;
    f.convertTo(g, CV_8U);
    return g;
}

ksh::FeatureParams params_from_config() {
    ksh::AlignConfig c;
    std::string e;
    ksh::load_align_config("config/alignment.yml", c, e, false);
    return c.features;
}

} // namespace

int main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);   // 소스·문자열이 UTF-8 → 콘솔도 UTF-8 로 (안 하면 한글이 CP949 로 깨짐)
#endif
    std::printf("== T1 라벨 이름 해석\n");
    {
        ksh::LabeledPhoto p;
        check(ksh::parse_label("center", p) && p.is_reference, "center = 센터");
        check(ksh::parse_label("center1", p) && p.is_reference, "center1 = 센터");
        check(ksh::parse_label("Y+0.05 T+0.03", p) && std::abs(p.y_mm - 0.05) < 1e-12 && std::abs(p.t_deg - 0.03) < 1e-12, "Y+0.05 T+0.03 (4차 형식)");
        check(ksh::parse_label("Y+0.1 T+0.01 (2)", p) && std::abs(p.y_mm - 0.1) < 1e-12 && std::abs(p.t_deg - 0.01) < 1e-12, "Y+0.1 T+0.01 (2) (2차 형식)");
        check(ksh::parse_label("Y-1.60", p) && std::abs(p.y_mm + 1.6) < 1e-12 && p.t_deg == 0.0, "Y-1.60 (Y 단독)");
        check(ksh::parse_label("T-0.55", p) && p.y_mm == 0.0 && std::abs(p.t_deg + 0.55) < 1e-12, "T-0.55 (Θ 단독)");
        check(!ksh::parse_label("ROI 위치들", p), "라벨 아닌 이름은 거부");
    }

    const ksh::FeatureParams P = params_from_config();
    const Scene S;
    std::printf("== T2 QR·홀 중심 정확도 (합성 장면, 정답과 비교)\n");
    {
        const ksh::FeatureMeasurement f = ksh::measure_features(render(S, 0.0), P);
        check(f.ok, "측정 성공 " + f.message);
        if (f.ok) {
            double eq = 0, eh = 0;
            for (int i = 0; i < 3; ++i) eq = std::max(eq, cv::norm(f.qr.centers_px[i] - S.qr[i]));
            for (int i = 0; i < 4; ++i) eh = std::max(eh, cv::norm(f.holes[i].center_px - S.holes[i]));
            char b[120];
            std::snprintf(b, sizeof(b), "QR 중심 최대 오차 %.4f px < 0.05", eq);
            check(eq < 0.05, b);
            std::snprintf(b, sizeof(b), "홀 중심 최대 오차 %.4f px < 0.05", eh);
            check(eh < 0.05, b);
        }
    }
    std::printf("== T3 서브픽셀 평행이동 → s, d\n");
    {
        const ksh::FeatureMeasurement a = ksh::measure_features(render(S, 0.0), P), b = ksh::measure_features(render(S, 0.37), P),
                                      c = ksh::measure_features(render(S, 0.0, -0.25), P);
        char m[160];
        std::snprintf(m, sizeof(m), "전체 +0.37 px 이동 : Δs = %.4f (≈0.37), Δd = %.4f (≈0)", b.s_px - a.s_px, b.d_px - a.d_px);
        // 허용 0.03 px : 경계가 완전히 날카로운 합성 영상에서 잰 서브픽셀 보간 치우침(사진 위상에 따라 ±0.015 px)의 두 배
        check(a.ok && b.ok && std::abs(b.s_px - a.s_px - 0.37) < 0.03 && std::abs(b.d_px - a.d_px) < 0.03, m);
        std::snprintf(m, sizeof(m), "홀만 −0.25 px (시차) : Δs = %.4f (≈0), Δd = %.4f (≈+0.25)", c.s_px - a.s_px, c.d_px - a.d_px);
        check(c.ok && std::abs(c.s_px - a.s_px) < 0.02 && std::abs(c.d_px - a.d_px - 0.25) < 0.02, m);
    }
    std::printf("== T4 대략 위치 (coarse)\n");
    {
        const ksh::FeatureMeasurement f = ksh::measure_features(render(S, 180.0), P);
        // 대략 위치는 박스 여유(search_px 40) 안이면 충분 — 정밀 위치(s)가 맞는지가 기준
        check(f.ok && std::abs(f.qr.coarse_dx_px - 180.0) < 20.0 && std::abs(f.s_px - (1907.65 + 180.0)) < 0.1,
              "+180 px 이동 (Θ 약 2.2°) 을 찾음, dx = " + std::to_string(f.qr.coarse_dx_px));
        const ksh::FeatureMeasurement g = ksh::measure_features(render(S, -300.0), P);
        check(g.ok && std::abs(g.s_px - (1907.65 - 300.0)) < 0.1, "−300 px 이동을 찾음");
        const ksh::FeatureMeasurement h = ksh::measure_features(render(S, 420.0), P);
        check(!h.ok && !h.message.empty(), "+420 px (탐색 범위 350 밖) → 실패 : " + h.message);
    }
    std::printf("== T5 가림\n");
    {
        Scene S2 = S;
        S2.hide_w2 = true;
        const ksh::FeatureMeasurement f = ksh::measure_features(render(S2, 0.0), P);
        check(!f.ok && f.message.find("W2") != std::string::npos, "홀 W2 가림 → 실패 : " + f.message);
        cv::Mat blank(2160, 3840, CV_8U, cv::Scalar(128));
        const ksh::FeatureMeasurement g = ksh::measure_features(blank, P);
        check(!g.ok, "빈 화면 → 실패 : " + g.message);
    }
    std::printf("== T6 보정식 (세트별 상수 학습 · 역산)\n");
    {
        ksh::PoseModel truth;
        truth.s = {0.0, 14.30, {80.16}, 0.28};
        truth.d = {0.0, 3.128, {-1.268}, 0.05};
        std::mt19937 rng(3);
        std::normal_distribution<double> ns(0, 0.05), nd(0, 0.01);
        std::vector<std::vector<ksh::FeatureSample>> sets(2);
        const double cs[2] = {1900.8, 1900.2}, cd[2] = {-13.9, -14.2};
        for (int k = 0; k < 2; ++k)
            for (int i = 0; i < (k ? 140 : 280); ++i) {
                ksh::FeatureSample s;
                s.y_mm = (i % 20 - 10) * 0.2;
                s.t_deg = (i / 20 % 14 - 7) * 0.13;
                s.s_px = cs[k] + truth.s.a_px_per_mm * s.y_mm + truth.s.b_px_per_deg_k[0] * s.t_deg + ns(rng);
                s.d_px = cd[k] + truth.d.a_px_per_mm * s.y_mm + truth.d.b_px_per_deg_k[0] * s.t_deg + nd(rng);
                sets[k].push_back(s);
            }
        ksh::PoseModel m;
        std::vector<double> ccs, ccd;
        const bool ok = ksh::fit_pose_model_sets(sets, 1, 1, true, m, &ccs, &ccd);
        char b[200];
        std::snprintf(b, sizeof(b), "기울기 복원 a_s %.4f · b_s %.4f · a_d %.4f · b_d %.4f", m.s.a_px_per_mm, m.s.b_px_per_deg_k[0], m.d.a_px_per_mm, m.d.b_px_per_deg_k[0]);
        check(ok && std::abs(m.s.a_px_per_mm - 14.30) < 0.02 && std::abs(m.d.a_px_per_mm - 3.128) < 0.005 && std::abs(m.s.b_px_per_deg_k[0] - 80.16) < 0.05 &&
                  std::abs(m.d.b_px_per_deg_k[0] + 1.268) < 0.01, b);
        std::snprintf(b, sizeof(b), "세트별 상수 복원 (%.3f, %.3f) (%.3f, %.3f)", ccs[0], ccd[0], ccs[1], ccd[1]);
        check(ok && std::abs(ccs[0] - 1900.8) < 0.02 && std::abs(ccs[1] - 1900.2) < 0.02 && std::abs(ccd[0] + 13.9) < 0.005 && std::abs(ccd[1] + 14.2) < 0.005, b);
        double emax = 0;
        for (double y : {-7.0, -0.1, 0.0, 0.05, 2.0, 7.0})
            for (double t : {-2.0, -0.01, 0.0, 0.5, 2.0}) {
                const double s = ksh::eval_feature(m.s, y, t), d = ksh::eval_feature(m.d, y, t);
                double ye, te;
                ksh::invert_pose_model(m, s, d, ye, te);
                emax = std::max(emax, std::max(std::abs(ye - y), std::abs(te - t)));
            }
        check(emax < 1e-9, "역산 왕복 오차 " + std::to_string(emax) + " < 1e-9 (Y ±7 mm, Θ ±2°)");
    }
    std::printf("== T7 설정 파일 고쳐 쓰기\n");
    {
        const std::string tmp = (fs::temp_directory_path() / "ksh_cfg_test.yml").string();
        fs::copy_file("config/alignment.yml", tmp, fs::copy_options::overwrite_existing);
        auto count_lines = [](const std::string &p) { std::ifstream f(p); int n = 0; std::string l; while (std::getline(f, l)) ++n; return n; };
        const int n0 = count_lines(tmp);
        ksh::AlignConfig c;
        std::string e;
        ksh::load_align_config(tmp, c, e, false);
        std::vector<ksh::BoxSpec> boxes = c.features.qr.boxes;
        boxes.insert(boxes.end(), c.features.holes.boxes.begin(), c.features.holes.boxes.end());
        for (auto &b : boxes) { b.cx_px += 3; b.cy_px -= 2; }
        ksh::ZeroOffset z;
        z.y_mm = 0.0123456;
        z.t_deg = -0.0045678;
        const bool w = ksh::save_roi_centers(tmp, boxes, e, true) && ksh::save_zero_offset(tmp, z, e, false);
        ksh::AlignConfig c2;
        ksh::load_align_config(tmp, c2, e, false);
        bool same = w && c2.features.qr.boxes.size() == 3 && c2.features.holes.boxes.size() == 4;
        for (size_t i = 0; same && i < 3; ++i) same = c2.features.qr.boxes[i].cx_px == boxes[i].cx_px && c2.features.qr.boxes[i].cy_px == boxes[i].cy_px;
        for (size_t i = 0; same && i < 4; ++i) same = c2.features.holes.boxes[i].cx_px == boxes[3 + i].cx_px && c2.features.holes.boxes[i].cy_px == boxes[3 + i].cy_px;
        check(same, "ROI 7개 좌표 고쳐 쓰고 다시 읽기");
        check(std::abs(c2.zero.y_mm - z.y_mm) < 1e-9 && std::abs(c2.zero.t_deg - z.t_deg) < 1e-9, "영점 고쳐 쓰고 다시 읽기");
        check(count_lines(tmp) == n0, "다른 줄·주석 보존 (줄 수 " + std::to_string(n0) + ")");
        check(fs::exists(tmp + ".bak") && count_lines(tmp + ".bak") == n0, "원본 .bak 생성");
    }
    std::printf("== T8 alignment() 전체 흐름 (합성 장면 + 임시 설정)\n");
    {
        const fs::path dir = fs::temp_directory_path() / "ksh_align_test";
        fs::create_directories(dir);
        const std::string cfgp = (dir / "alignment.yml").string();
        fs::copy_file("config/alignment.yml", cfgp, fs::copy_options::overwrite_existing);
        // 합성 장면의 보정식 : 가로 이동 dx 만으로 만든 장면이므로 s = c + 14.3·Y + 80.16·Θ, d 는 홀 시차
        ksh::PoseModel m;
        m.s = {1907.65, 14.30, {80.16}, 0.28};
        m.d = {0.0, 3.128, {-1.268}, 0.05};
        const ksh::FeatureMeasurement f0 = ksh::measure_features(render(S, 0.0), P);
        m.d.c_px = f0.d_px;
        ksh::save_pose_model((dir / "model_s1d1.yml").string(), m, "test", 0);
        check(alignment_init(cfgp), "임시 설정 읽기");
        cv::Mat empty;
        const Alignment_t r0 = alignment(empty);
        check(std::isnan(r0.y) && std::isnan(r0.t) && !alignment_last_info().message.empty(), "빈 사진 → NaN : " + alignment_last_info().message);
        // 초기 세팅 : 박스가 몇 px 어긋난 설정에서 센터 사진 2장 → 박스 갱신 + 영점
        cv::Mat c1 = render(S, 0.0), c2 = render(S, 0.02);
        check(alignment_setup({c1, c2}, true), "초기 세팅 (센터 2장) " + alignment_setup_info().message);
        const SetupInfo &si = alignment_setup_info();
        check(si.warnings.empty(), "세팅 경고 없음 (s 차이 " + std::to_string(si.s_spread_px) + " px)");
        const Alignment_t rc = alignment(c1);
        ksh::AlignConfig tc;
        std::string te;
        ksh::load_align_config(cfgp, tc, te);
        const double ey = -tc.center_to_encoder.y_mm, et = -tc.center_to_encoder.t_deg;   // 센터 = −(센터 자세 − 엔코더 영점)
        char b[200];
        std::snprintf(b, sizeof(b), "센터 사진 추정 (%.5f mm, %.5f°) ≈ (%.5f, %.5f) = −center_to_encoder", rc.y, rc.t, ey, et);
        check(std::abs(rc.y - ey) < 0.002 && std::abs(rc.t - et) < 0.0005, b);
        // Y = 0.5 mm, Θ = 0.2° 에 해당하는 이동 : 플레이트 dx = 14.3·0.5 + 80.16·0.2, 홀 시차 = −(3.128·0.5 − 1.268·0.2)
        const double y = 0.5, t = 0.2, dx = 14.30 * y + 80.16 * t, hole = -(3.128 * y - 1.268 * t);
        cv::Mat mv = render(S, dx, hole);
        const Alignment_t rm = alignment(mv);
        std::snprintf(b, sizeof(b), "이동 사진 추정 − 센터 값 (%.4f mm, %.4f°) ≈ (0.5, 0.2)", rm.y - ey, rm.t - et);
        check(std::abs(rm.y - ey - y) < 0.01 && std::abs(rm.t - et - t) < 0.001, b);
        ksh::AlignConfig back;
        std::string e;
        ksh::load_align_config(cfgp, back, e);
        check(std::abs(back.zero.y_mm - si.zero_y_mm) < 1e-8 && fs::exists(cfgp + ".bak"), "세팅 결과가 설정 파일에 기록됨 (.bak 생성)");
    }
    std::printf("\n결과 : 통과 %d, 실패 %d\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
