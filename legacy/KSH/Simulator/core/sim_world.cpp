// =====================================================================================
// sim_world.cpp — 운동장 학습, 파노라마, 사진 합성
// =====================================================================================
#include "sim_world.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <regex>
#include <sstream>

namespace fs = std::filesystem;

namespace sim {

// =====================================================================================
// 사진 목록
// =====================================================================================
bool parseLabel(const std::string& stem, Photo& p) {
    static const std::regex re_center(R"(^center$)"), re_center_n(R"(^center\d+$)");
    static const std::regex re_mixed(R"(^Y\s*([+-]?)\s*(\d+(?:\.\d+)?)\s+T\s*([+-]?)\s*(\d+(?:\.\d+)?)\s*\(\d+\)$)");
    static const std::regex re_y(R"(^Y([+-]?\d+(?:\.\d+)?)$)"), re_t(R"(^T([+-]?\d+(?:\.\d+)?)$)");
    std::smatch m;
    p.stem = stem; p.pose = Pose(); p.set = 1; p.is_ref = false;
    if (std::regex_match(stem, re_center)) { p.is_ref = true; return true; }
    if (std::regex_match(stem, re_center_n)) { p.is_ref = true; p.set = 2; return true; }
    if (std::regex_match(stem, m, re_mixed)) {
        p.pose.y_mm = std::stod(m[2].str()) * (m[1].str() == "-" ? -1.0 : 1.0);
        p.pose.t_deg = std::stod(m[4].str()) * (m[3].str() == "-" ? -1.0 : 1.0);
        p.set = 2;
        return true;
    }
    if (std::regex_match(stem, m, re_y)) { p.pose.y_mm = std::stod(m[1].str()); return true; }
    if (std::regex_match(stem, m, re_t)) { p.pose.t_deg = std::stod(m[1].str()); return true; }
    return false;
}

std::vector<Photo> scanPhotos(const std::string& dir) {
    std::map<std::string, Photo> byStem;
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) return {};
    for (const auto& e : fs::directory_iterator(dir, ec)) {
        if (!e.is_regular_file()) continue;
        std::string ext = e.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return char(std::tolower(c)); });
        if (ext != ".bmp" && ext != ".png") continue;
        Photo p;
        if (!parseLabel(e.path().stem().string(), p)) continue;
        p.path = e.path().string();
        auto it = byStem.find(p.stem);
        if (it == byStem.end() || ext == ".bmp") byStem[p.stem] = p;   // 같은 이름이면 bmp 우선
    }
    std::vector<Photo> v;
    for (auto& kv : byStem) v.push_back(kv.second);
    return v;
}

const Photo* findPhoto(const std::vector<Photo>& v, const std::string& stem) {
    for (const auto& p : v) if (p.stem == stem) return &p;
    return nullptr;
}

bool firstSession(const Photo& p) {
    if (p.set != 1) return false;
    if (p.pose.y_mm == 0.0) return true;   // center, Θ 스윕
    const double k = p.pose.y_mm * 10.0;
    return std::abs(k - std::round(k)) < 1e-6;   // Y 0.1 mm 배수
}

// =====================================================================================
// 정합 도구 (alignment.cpp 와 같은 절차)
// =====================================================================================
namespace {

bool inside(const cv::Rect& r, const cv::Size& s) { return r.x >= 0 && r.y >= 0 && r.x + r.width <= s.width && r.y + r.height <= s.height; }
cv::Point roundPt(const cv::Point2d& p) { return cv::Point(int(std::lround(p.x)), int(std::lround(p.y))); }

cv::Point2d phaseShift(const cv::Mat& a, const cv::Mat& b) {
    cv::Mat win;
    cv::createHanningWindow(win, a.size(), CV_32F);
    cv::Mat a2 = a.clone(), b2 = b.clone();   // phaseCorrelate 가 창 함수를 제자리로 곱함
    return cv::phaseCorrelate(a2, b2, win);
}

int quietError(int, const char*, const char*, const char*, int, void*) { return 0; }

// 디버그 라이브러리의 오류 문구 출력을 이 객체가 살아 있는 동안 막음 (ECC 미수렴 = 정상적인 측정 실패)
struct QuietErrors {
    void* prevData = nullptr;
    cv::ErrorCallback prev = nullptr;
    QuietErrors() { prev = cv::redirectError(quietError, nullptr, &prevData); }
    ~QuietErrors() { cv::redirectError(prev, prevData); }
};

bool eccShift(const cv::Mat& tmpl, const cv::Mat& img, const cv::Point2d& init, cv::Point2d& out, double& cc, int maxIter = 100, double eps = 1e-7) {
    cv::Mat w = cv::Mat::eye(2, 3, CV_32F);
    w.at<float>(0, 2) = float(init.x);
    w.at<float>(1, 2) = float(init.y);
    try {
        cc = cv::findTransformECC(tmpl, img, w, cv::MOTION_TRANSLATION,
                                  cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, maxIter, eps), cv::noArray());
    } catch (const cv::Exception&) {
        return false;
    }
    out = cv::Point2d(w.at<float>(0, 2), w.at<float>(1, 2));
    return true;
}

cv::Mat loadGray32(const std::string& path) {
    cv::Mat g = cv::imread(path, cv::IMREAD_GRAYSCALE), f;
    if (!g.empty()) g.convertTo(f, CV_32F);
    return f;
}

std::string fmt(const char* f, double a = 0, double b = 0, double c = 0, double d = 0) {
    char buf[256];
    std::snprintf(buf, sizeof(buf), f, a, b, c, d);
    return buf;
}

} // namespace

// =====================================================================================
// 운동장 학습 + 파노라마
// =====================================================================================
bool World::build(const SimConfig& cfg, const cv::Point& coarseCenter, const cv::Size& coarseWin,
                  const Progress& progress, std::string& message) {
    const auto say = [&](double f, const std::string& s) { if (progress) progress(f, s); };
    step_ = cfg.field_step_px; mx_ = cfg.canvas_margin_x_px; my_ = cfg.canvas_margin_y_px;
    useOffset_ = cfg.render_reference_offset; noiseSigma_ = cfg.render_noise_sigma; gain_ = cfg.render_gain;

    // ---------------------------------------------------------------- 학습 사진 고르기 (1차 첫 번째 측정)
    const std::vector<Photo> all = scanPhotos(cfg.image_dir);
    const Photo* refP = findPhoto(all, "center");
    if (!refP) { message = "기준 사진 center 가 없음 : " + cfg.image_dir; return false; }
    std::vector<Photo> fit;
    for (const auto& p : all) {
        if (p.is_ref || !firstSession(p)) continue;
        if (p.pose.y_mm == 0.0) {   // Θ 스윕 : 간격 맞춰 고르기 (±1° 끝은 항상 포함)
            const double k = p.pose.t_deg / cfg.field_theta_step_deg;
            if (std::abs(k - std::round(k)) > 1e-6 && std::abs(std::abs(p.pose.t_deg) - 1.0) > 1e-9) continue;
        }
        fit.push_back(p);
    }
    if (fit.size() < 10) { message = "학습 사진이 너무 적음"; return false; }
    const cv::Mat ref = loadGray32(refP->path);
    frame_ = ref.size();
    if (frame_.width % step_ || frame_.height % step_) { message = "사진 크기가 격자 간격으로 나눠지지 않음"; return false; }
    const int gx = frame_.width / step_, gy = frame_.height / step_, NB = gx * gy, NP = int(fit.size());
    const cv::Rect coarseRect(coarseCenter.x - coarseWin.width / 2, coarseCenter.y - coarseWin.height / 2, coarseWin.width, coarseWin.height);
    if (!inside(coarseRect, frame_)) { message = "거친 이동 창이 사진 밖"; return false; }

    // 격자 창 (가장자리는 사진 안쪽으로 밀어 넣음)
    std::vector<cv::Rect> win(NB);
    const int W = cfg.field_window_px;
    for (int j = 0; j < gy; ++j)
        for (int i = 0; i < gx; ++i) {
            const int cx = int(std::lround((i + 0.5) * step_ - 0.5)), cy = int(std::lround((j + 0.5) * step_ - 0.5));
            int x = std::min(std::max(cx - W / 2, 0), frame_.width - W), y = std::min(std::max(cy - W / 2, 0), frame_.height - W);
            win[j * gx + i] = cv::Rect(x, y, W, W);
        }

    // ---------------------------------------------------------------- 질감 점수 (Shi–Tomasi 작은 고유값의 창 평균)
    //   같은 계산을 촬영 잡음만 있는 영상에 해서 '잡음 바닥' 을 구하고, 그 배수보다 낮은 창은 재지 않음 (어차피 이웃 값으로 채움)
    std::vector<unsigned char> textured(NB, 1);
    int nTextured = NB;
    {
        cv::Mat eig, eigN, noise(cv::Size(512, 512), CV_32F);
        cv::cornerMinEigenVal(ref, eig, 5, 3);
        cv::RNG nr(12345);
        nr.fill(noise, cv::RNG::NORMAL, 0.0, std::max(0.1, cfg.render_noise_sigma));
        cv::cornerMinEigenVal(noise, eigN, 5, 3);
        const double floor = cv::mean(eigN)[0];
        nTextured = 0;
        for (int b = 0; b < NB; ++b) {
            textured[b] = cv::mean(eig(win[b]))[0] >= cfg.field_min_texture * floor ? 1 : 0;
            nTextured += textured[b];
        }
    }

    // ---------------------------------------------------------------- 사진마다 격자 이동 측정 (가장 오래 걸림 → world/field_raw.bin 에 저장해 두고 같은 조건이면 다시 씀)
    std::vector<float> mdx(size_t(NP) * NB, 0.f), mdy(size_t(NP) * NB, 0.f);
    std::vector<unsigned char> mok(size_t(NP) * NB, 0);
    std::string key;   // 측정 조건 : 격자·창·질감 기준·사진 목록
    {
        std::ostringstream k;
        k << "v1 " << step_ << ' ' << W << ' ' << cfg.field_min_texture << ' ' << cfg.field_min_cc << ' ' << NP << ' ' << NB;
        for (const auto& p : fit) k << ' ' << p.stem;
        key = k.str();
    }
    const std::string rawPath = cfg.world_dir + "/field_raw.bin";
    bool cached = false;
    {
        std::ifstream in(rawPath, std::ios::binary);
        uint32_t len = 0;
        if (in && in.read(reinterpret_cast<char*>(&len), sizeof(len)) && len == key.size()) {
            std::string k(len, ' ');
            in.read(&k[0], len);
            if (k == key) {
                in.read(reinterpret_cast<char*>(mdx.data()), std::streamsize(mdx.size() * sizeof(float)));
                in.read(reinterpret_cast<char*>(mdy.data()), std::streamsize(mdy.size() * sizeof(float)));
                in.read(reinterpret_cast<char*>(mok.data()), std::streamsize(mok.size()));
                cached = bool(in);
            }
        }
    }
    if (cached) say(0.85, "운동장 측정 : 이전 측정 결과를 다시 씀 (" + rawPath + ")");
    else {
        {
            QuietErrors quiet;
            for (int k = 0; k < NP; ++k) {
                say(0.85 * k / NP, fmt("운동장 측정 %.0f / %.0f", k + 1.0, double(NP)) + " : " + fit[k].stem);
                const cv::Mat cur = loadGray32(fit[k].path);
                if (cur.size() != frame_) continue;
                const cv::Point2d coarse = phaseShift(ref(coarseRect), cur(coarseRect));
                const cv::Point g0 = roundPt(coarse);
                const double minCc = cfg.field_min_cc;
                cv::parallel_for_(cv::Range(0, NB), [&](const cv::Range& r) {
                    for (int b = r.start; b < r.end; ++b) {
                        if (!textured[b]) continue;
                        const cv::Rect A = win[b], B = A + g0;
                        if (!inside(B, frame_)) continue;
                        const cv::Point2d g = cv::Point2d(g0) + phaseShift(ref(A), cur(B));
                        const cv::Point gi = roundPt(g);
                        const cv::Rect C = A + gi;
                        if (!inside(C, frame_)) continue;
                        cv::Point2d fr; double cc = 0;
                        if (!eccShift(ref(A), cur(C), g - cv::Point2d(gi), fr, cc, 50, 1e-5)) continue;
                        const cv::Point2d s = fr + cv::Point2d(gi);
                        if (cc < minCc || cv::norm(s - g) > 2.0) continue;
                        mdx[size_t(k) * NB + b] = float(s.x); mdy[size_t(k) * NB + b] = float(s.y); mok[size_t(k) * NB + b] = 1;
                    }
                });
            }
        }
        std::error_code ec2;
        fs::create_directories(cfg.world_dir, ec2);
        std::ofstream out(rawPath, std::ios::binary);
        const uint32_t len = uint32_t(key.size());
        out.write(reinterpret_cast<const char*>(&len), sizeof(len));
        out.write(key.data(), len);
        out.write(reinterpret_cast<const char*>(mdx.data()), std::streamsize(mdx.size() * sizeof(float)));
        out.write(reinterpret_cast<const char*>(mdy.data()), std::streamsize(mdy.size() * sizeof(float)));
        out.write(reinterpret_cast<const char*>(mok.data()), std::streamsize(mok.size()));
    }

    // ---------------------------------------------------------------- 격자마다 맞춤 : d = c + aY·Y + bT·Θ + qT·Θ² + (사진별 공통 이동)
    //   실제 사진에는 라벨과 실제 자세의 작은 차이(스테이지 백래시 등)가 있어 사진 전체가 함께 0.2 px 정도 밀린다.
    //   이 '사진별 공통 이동' o_k, p_k 를 격자 잔차의 중앙값으로 추정해 빼고 맞춘다 (4회 반복).
    //   → 운동장 = 라벨 자세의 기하만, 공통 이동 = 로봇 자세 오차 (시뮬레이터에서는 로봇 오차로 따로 넣음)
    say(0.87, "운동장 맞춤");
    for (auto& m : coef_) m = cv::Mat::zeros(gy, gx, CV_32F);
    valid_ = cv::Mat::zeros(gy, gx, CV_8U);
    resid_ = cv::Mat::zeros(gy, gx, CV_32F);
    cv::Mat A(NP, 4, CV_64F);
    for (int k = 0; k < NP; ++k) {
        const Pose& P = fit[k].pose;
        A.at<double>(k, 0) = 1.0; A.at<double>(k, 1) = P.y_mm; A.at<double>(k, 2) = P.t_deg; A.at<double>(k, 3) = P.t_deg * P.t_deg;
    }
    std::vector<int> cand;
    for (int b = 0; b < NB; ++b) {
        if (!textured[b]) continue;
        int n = 0;
        for (int k = 0; k < NP; ++k) n += mok[size_t(k) * NB + b];
        if (n >= std::max(8, int(cfg.field_min_ok_ratio * NP))) cand.push_back(b);
    }
    std::vector<double> ox(NP, 0.0), oy(NP, 0.0);
    std::vector<double> rx(size_t(NP) * NB, 0.0), ry(size_t(NP) * NB, 0.0), rms(NB, 1e9);
    std::vector<cv::Mat> cxs(NB), cys(NB);
    const auto fitBlocks = [&]() {
        cv::parallel_for_(cv::Range(0, int(cand.size())), [&](const cv::Range& r) {
            for (int ci = r.start; ci < r.end; ++ci) {
                const int b = cand[ci];
                std::vector<int> rows;
                for (int k = 0; k < NP; ++k) if (mok[size_t(k) * NB + b]) rows.push_back(k);
                cv::Mat Ab(int(rows.size()), 4, CV_64F), bx(int(rows.size()), 1, CV_64F), by(int(rows.size()), 1, CV_64F);
                for (int q = 0; q < int(rows.size()); ++q) {
                    A.row(rows[q]).copyTo(Ab.row(q));
                    bx.at<double>(q) = mdx[size_t(rows[q]) * NB + b] - ox[rows[q]];
                    by.at<double>(q) = mdy[size_t(rows[q]) * NB + b] - oy[rows[q]];
                }
                cv::solve(Ab, bx, cxs[b], cv::DECOMP_SVD);
                cv::solve(Ab, by, cys[b], cv::DECOMP_SVD);
                const cv::Mat ex = Ab * cxs[b] - bx, ey = Ab * cys[b] - by;
                for (int q = 0; q < int(rows.size()); ++q) {
                    rx[size_t(rows[q]) * NB + b] = -ex.at<double>(q);   // 잔차 = 측정 − 식 (공통 이동 제외 전)
                    ry[size_t(rows[q]) * NB + b] = -ey.at<double>(q);
                }
                rms[b] = std::sqrt((ex.dot(ex) + ey.dot(ey)) / rows.size());
            }
        });
    };
    for (int it = 0; it < 4; ++it) {
        fitBlocks();
        // 사진별 공통 이동 = 이 사진에서 잰 격자 잔차의 중앙값 (+ 지금까지의 공통 이동)
        for (int k = 0; k < NP; ++k) {
            std::vector<double> vx, vy;
            for (int b : cand) if (mok[size_t(k) * NB + b]) { vx.push_back(rx[size_t(k) * NB + b]); vy.push_back(ry[size_t(k) * NB + b]); }
            if (vx.empty()) continue;
            std::nth_element(vx.begin(), vx.begin() + vx.size() / 2, vx.end());
            std::nth_element(vy.begin(), vy.begin() + vy.size() / 2, vy.end());
            ox[k] += vx[vx.size() / 2]; oy[k] += vy[vy.size() / 2];
        }
        // 공통 이동 중 자세의 식으로 설명되는 부분은 운동장 몫 → 빼서 직교화
        cv::Mat vo(NP, 1, CV_64F, ox.data()), vp(NP, 1, CV_64F, oy.data()), co, cp;
        cv::solve(A, vo, co, cv::DECOMP_SVD);
        cv::solve(A, vp, cp, cv::DECOMP_SVD);
        const cv::Mat fo = A * co, fp = A * cp;
        for (int k = 0; k < NP; ++k) { ox[k] -= fo.at<double>(k); oy[k] -= fp.at<double>(k); }
    }
    fitBlocks();
    int nValid = 0;
    for (int b : cand) {
        const int j = b / gx, i = b % gx;
        resid_.at<float>(j, i) = float(rms[b]);
        for (int q = 0; q < 4; ++q) { coef_[q].at<float>(j, i) = float(cxs[b].at<double>(q)); coef_[4 + q].at<float>(j, i) = float(cys[b].at<double>(q)); }
        if (rms[b] <= cfg.field_max_resid_px) { valid_.at<uchar>(j, i) = 255; ++nValid; }
    }
    double oRms = 0;
    for (int k = 0; k < NP; ++k) oRms += ox[k] * ox[k] + oy[k] * oy[k];
    oRms = std::sqrt(oRms / NP);
    if (nValid < 20) { message = "측정에 성공한 격자가 너무 적음 (" + std::to_string(nValid) + ")"; return false; }

    // ---------------------------------------------------------------- 질감 없는 격자 채우기 (이웃 평균을 바깥으로 번져 나감)
    {
        cv::Mat known;
        valid_.convertTo(known, CV_32F, 1.0 / 255.0);
        for (auto& m : coef_) m = m.mul(known);
        for (int it = 0; it < gx + gy; ++it) {
            if (cv::countNonZero(known) == NB) break;
            cv::Mat den, kNew = known.clone();
            cv::blur(known, den, cv::Size(3, 3), cv::Point(-1, -1), cv::BORDER_REPLICATE);
            for (auto& m : coef_) {
                cv::Mat num;
                cv::blur(m.mul(known), num, cv::Size(3, 3), cv::Point(-1, -1), cv::BORDER_REPLICATE);
                for (int j = 0; j < gy; ++j)
                    for (int i = 0; i < gx; ++i)
                        if (known.at<float>(j, i) == 0.f && den.at<float>(j, i) > 0.f) m.at<float>(j, i) = num.at<float>(j, i) / den.at<float>(j, i);
            }
            for (int j = 0; j < gy; ++j)
                for (int i = 0; i < gx; ++i)
                    if (known.at<float>(j, i) == 0.f && den.at<float>(j, i) > 0.f) kNew.at<float>(j, i) = 1.f;
            known = kNew;
        }
    }
    makeCanvasField();

    // ---------------------------------------------------------------- 파노라마 : 가운데 = center, 바깥 = 다른 자세 사진을 되돌려 평균
    say(0.9, "파노라마 합성");
    cv::Mat ref8 = cv::imread(refP->path, cv::IMREAD_GRAYSCALE);
    cv::copyMakeBorder(ref8, canvas_, my_, my_, mx_, mx_, cv::BORDER_REPLICATE);
    const cv::Size cs = canvas_.size();
    cv::Mat sum = cv::Mat::zeros(cs, CV_32F), cnt = cv::Mat::zeros(cs, CV_32F);
    cv::Mat inner = cv::Mat::zeros(cs, CV_8U);
    inner(cv::Rect(mx_, my_, frame_.width, frame_.height)).setTo(255);
    for (int k = 0; k < NP; ++k) {
        const Photo& P = fit[k];
        if (std::abs(P.pose.t_deg) < 0.25 && std::abs(P.pose.y_mm) < 1.0) continue;   // 가장자리 정보가 거의 없는 사진은 생략
        cv::Mat ph = cv::imread(P.path, cv::IMREAD_GRAYSCALE), phf;
        if (ph.size() != frame_) continue;
        ph.convertTo(phf, CV_32F);
        // 파노라마 화소 p → 이 사진에서의 위치 q = p_ref + 이동(p_ref, 자세)  (사진에는 상수항이 들어 있으므로 포함)
        cv::Mat dxL(coefCanvas_[0].size(), CV_32F), dyL(coefCanvas_[0].size(), CV_32F);
        const double y = P.pose.y_mm, t = P.pose.t_deg;
        dxL = coefCanvas_[0] + coefCanvas_[1] * y + coefCanvas_[2] * t + coefCanvas_[3] * (t * t);
        dyL = coefCanvas_[4] + coefCanvas_[5] * y + coefCanvas_[6] * t + coefCanvas_[7] * (t * t);
        cv::Mat dxF, dyF;
        cv::resize(dxL, dxF, cs, 0, 0, cv::INTER_LINEAR);
        cv::resize(dyL, dyF, cs, 0, 0, cv::INTER_LINEAR);
        cv::Mat mapx(cs, CV_32F), mapy(cs, CV_32F);
        for (int r = 0; r < cs.height; ++r) {
            float* px = mapx.ptr<float>(r); float* py = mapy.ptr<float>(r);
            const float* ax = dxF.ptr<float>(r); const float* ay = dyF.ptr<float>(r);
            for (int c = 0; c < cs.width; ++c) { px[c] = float(c - mx_) + ax[c]; py[c] = float(r - my_) + ay[c]; }
        }
        cv::Mat warped, ok;
        cv::remap(phf, warped, mapx, mapy, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
        cv::remap(cv::Mat(frame_, CV_32F, cv::Scalar(1)), ok, mapx, mapy, cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
        ok.setTo(0, inner);
        sum += warped.mul(ok);
        cnt += ok;
        say(0.9 + 0.08 * (k + 1) / NP, "파노라마 합성 : " + P.stem);
    }
    covered_ = inner.clone();
    for (int r = 0; r < cs.height; ++r)
        for (int c = 0; c < cs.width; ++c)
            if (cnt.at<float>(r, c) > 0.5f) {
                canvas_.at<uchar>(r, c) = cv::saturate_cast<uchar>(sum.at<float>(r, c) / cnt.at<float>(r, c));
                covered_.at<uchar>(r, c) = 255;
            }

    // ---------------------------------------------------------------- 요약
    std::ostringstream o;
    o << "# 가상 장면 (운동장 + 파노라마)\n\n";
    o << "- 학습 사진 : 1차 첫 번째 측정 " << NP << "장 (Θ 스윕 " << cfg.field_theta_step_deg << "° 간격 + Y 0.1 mm 배수), 기준 center\n";
    o << "- 격자 : " << gx << " × " << gy << " (간격 " << step_ << " px, 창 " << W << " px) — 질감 있음 " << nTextured << ", 측정 성공 " << nValid << " / " << NB
      << ", 나머지는 이웃 값으로 채움\n";
    std::vector<float> rv;
    for (int j = 0; j < gy; ++j) for (int i = 0; i < gx; ++i) if (valid_.at<uchar>(j, i)) rv.push_back(resid_.at<float>(j, i));
    std::sort(rv.begin(), rv.end());
    if (!rv.empty()) o << fmt("- 맞춤 잔차 RMS (측정 격자) : 중앙값 %.3f px, 90 %% %.3f px\n", rv[rv.size() / 2], rv[size_t(rv.size() * 0.9)]);
    o << fmt("- 사진별 공통 이동 (라벨 대비 실제 자세 차, 운동장에서 뺌) : RMS %.3f px\n", oRms);
    const auto probe = [&](const char* name, double u, double v, double expectY) {
        const int i = std::min(gx - 1, std::max(0, int(u / step_))), j = std::min(gy - 1, std::max(0, int(v / step_)));
        o << "- " << name << fmt(" 격자 : Y 감도 %.2f px/mm, Θ 감도 %.2f px/°", coef(0, 1).at<float>(j, i), coef(0, 2).at<float>(j, i))
          << fmt("  (ROI 실측 회귀 %.2f px/mm)\n", expectY);
    };
    probe("QR_C", 1907, 1065, 14.25);
    probe("하단 문 W2", 2867, 1357, 11.15);
    o << "- 파노라마 : " << cs.width << " × " << cs.height << " (기준 사진 + 좌우 " << mx_ << " px, 위아래 " << my_ << " px), 실제 사진으로 채운 비율 "
      << fmt("%.1f %%\n", 100.0 * cv::countNonZero(covered_) / double(cs.area()));
    report_ = o.str();
    say(1.0, "완료");
    return true;
}

void World::makeCanvasField() {
    const int bx = mx_ / step_, by = my_ / step_;
    for (int k = 0; k < 8; ++k) cv::copyMakeBorder(coef_[k], coefCanvas_[k], by, by, bx, bx, cv::BORDER_REPLICATE);
}

// =====================================================================================
// 저장·불러오기
// =====================================================================================
bool World::save(const std::string& dir) const {
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (!cv::imwrite(dir + "/panorama.png", canvas_)) return false;
    cv::imwrite(dir + "/panorama_covered.png", covered_);
    cv::FileStorage f(dir + "/field.yml", cv::FileStorage::WRITE);
    if (!f.isOpened()) return false;
    f << "step_px" << step_ << "margin_x_px" << mx_ << "margin_y_px" << my_ << "frame_width_px" << frame_.width << "frame_height_px" << frame_.height;
    const char* names[8] = {"dx_c", "dx_aY", "dx_bT", "dx_qT2", "dy_c", "dy_aY", "dy_bT", "dy_qT2"};
    for (int k = 0; k < 8; ++k) f << names[k] << coef_[k];
    f << "valid" << valid_ << "residual" << resid_;
    f.release();
    std::ofstream(dir + "/field_report.md") << report_;
    return true;
}

bool World::load(const std::string& dir, const SimConfig& cfg) {
    std::error_code ec;
    if (!fs::exists(dir + "/field.yml", ec)) return false;   // 없는 파일을 열면 OpenCV 가 오류 문구를 찍으므로 먼저 확인
    cv::FileStorage f;
    try {
        if (!f.open(dir + "/field.yml", cv::FileStorage::READ)) return false;
    } catch (const cv::Exception&) {
        return false;
    }
    step_ = int(f["step_px"]); mx_ = int(f["margin_x_px"]); my_ = int(f["margin_y_px"]);
    frame_ = cv::Size(int(f["frame_width_px"]), int(f["frame_height_px"]));
    const char* names[8] = {"dx_c", "dx_aY", "dx_bT", "dx_qT2", "dy_c", "dy_aY", "dy_bT", "dy_qT2"};
    for (int k = 0; k < 8; ++k) { f[names[k]] >> coef_[k]; if (coef_[k].empty()) return false; }
    f["valid"] >> valid_; f["residual"] >> resid_;
    canvas_ = cv::imread(dir + "/panorama.png", cv::IMREAD_GRAYSCALE);
    covered_ = cv::imread(dir + "/panorama_covered.png", cv::IMREAD_GRAYSCALE);
    if (canvas_.empty() || canvas_.cols != frame_.width + 2 * mx_ || canvas_.rows != frame_.height + 2 * my_) return false;
    if (covered_.empty()) covered_ = cv::Mat(canvas_.size(), CV_8U, cv::Scalar(255));
    std::ifstream rf(dir + "/field_report.md");
    std::stringstream ss; ss << rf.rdbuf(); report_ = ss.str();
    useOffset_ = cfg.render_reference_offset; noiseSigma_ = cfg.render_noise_sigma; gain_ = cfg.render_gain;
    makeCanvasField();
    return true;
}

// =====================================================================================
// 합성
// =====================================================================================
cv::Point2d World::sampleCanvas(double u, double v, const Pose& pose, bool withOffset) const {
    const int cgx = coefCanvas_[0].cols, cgy = coefCanvas_[0].rows;
    const double fx = (u + 0.5) / step_ - 0.5 + mx_ / step_, fy = (v + 0.5) / step_ - 0.5 + my_ / step_;
    const double gxc = std::min(std::max(fx, 0.0), double(cgx - 1)), gyc = std::min(std::max(fy, 0.0), double(cgy - 1));
    const int i0 = std::min(int(gxc), cgx - 2), j0 = std::min(int(gyc), cgy - 2);
    const double ax = gxc - i0, ay = gyc - j0;
    const double y = pose.y_mm, t = pose.t_deg;
    double out[2];
    for (int axis = 0; axis < 2; ++axis) {
        const cv::Mat* m = &coefCanvas_[axis * 4];
        const bool off = withOffset && useOffset_;   // 상수항 = 기준 사진과 스윕 원점의 실측 차이 (설정으로 끌 수 있음)
        const auto val = [&](int j, int i) {
            return (off ? m[0].at<float>(j, i) : 0.f) + m[1].at<float>(j, i) * y + m[2].at<float>(j, i) * t + m[3].at<float>(j, i) * t * t;
        };
        out[axis] = (1 - ay) * ((1 - ax) * val(j0, i0) + ax * val(j0, i0 + 1)) + ay * ((1 - ax) * val(j0 + 1, i0) + ax * val(j0 + 1, i0 + 1));
    }
    return cv::Point2d(out[0], out[1]);
}

cv::Point2d World::displacement(const cv::Point2d& p, const Pose& pose) const { return sampleCanvas(p.x, p.y, pose, true); }

void World::inverseLow(const Pose& pose, cv::Mat& ix, cv::Mat& iy) const {
    const int gx = frame_.width / step_, gy = frame_.height / step_;
    ix.create(gy, gx, CV_32F); iy.create(gy, gx, CV_32F);
    for (int j = 0; j < gy; ++j)
        for (int i = 0; i < gx; ++i) {
            const double qu = (i + 0.5) * step_ - 0.5, qv = (j + 0.5) * step_ - 0.5;
            double pu = qu, pv = qv;
            for (int it = 0; it < 6; ++it) {   // q = p + D(p) 를 고정점 반복으로 풂 (D 는 완만해 3~4 번이면 1e-4 px)
                const cv::Point2d d = sampleCanvas(pu, pv, pose, true);
                pu = qu - d.x; pv = qv - d.y;
            }
            ix.at<float>(j, i) = float(pu - qu); iy.at<float>(j, i) = float(pv - qv);
        }
}

cv::Mat World::render(const Pose& pose, cv::RNG* rng) const {
    cv::Mat ix, iy, ixF, iyF;
    inverseLow(pose, ix, iy);
    cv::resize(ix, ixF, frame_, 0, 0, cv::INTER_LINEAR);   // 격자 중심 = (i+0.5)·step − 0.5 와 resize 의 화소 대응이 같음
    cv::resize(iy, iyF, frame_, 0, 0, cv::INTER_LINEAR);
    cv::Mat mapx(frame_, CV_32F), mapy(frame_, CV_32F);
    for (int r = 0; r < frame_.height; ++r) {
        float* px = mapx.ptr<float>(r); float* py = mapy.ptr<float>(r);
        const float* ax = ixF.ptr<float>(r); const float* ay = iyF.ptr<float>(r);
        for (int c = 0; c < frame_.width; ++c) { px[c] = float(c + mx_) + ax[c]; py[c] = float(r + my_) + ay[c]; }
    }
    cv::Mat f, out;
    cv::remap(canvas_, f, mapx, mapy, cv::INTER_CUBIC, cv::BORDER_REPLICATE);
    f.convertTo(f, CV_32F, gain_);
    if (rng && noiseSigma_ > 0) {
        cv::Mat n(frame_, CV_32F);
        rng->fill(n, cv::RNG::NORMAL, 0.0, noiseSigma_);
        f += n;
    }
    f.convertTo(out, CV_8U);   // 반올림 + 0~255 자르기 (실제 카메라처럼 8 bit)
    return out;
}

cv::Mat World::renderReference(cv::RNG* rng) const {
    cv::Mat f, out;
    canvas_(cv::Rect(mx_, my_, frame_.width, frame_.height)).convertTo(f, CV_32F, gain_);
    if (rng && noiseSigma_ > 0) {
        cv::Mat n(frame_, CV_32F);
        rng->fill(n, cv::RNG::NORMAL, 0.0, noiseSigma_);
        f += n;
    }
    f.convertTo(out, CV_8U);
    return out;
}

std::vector<cv::Point2f> World::footprint(const Pose& pose) const {
    std::vector<cv::Point2f> v;
    const cv::Point2d q[4] = {{0, 0}, {double(frame_.width), 0}, {double(frame_.width), double(frame_.height)}, {0, double(frame_.height)}};
    for (const auto& c : q) {
        double pu = c.x, pv = c.y;
        for (int it = 0; it < 6; ++it) { const cv::Point2d d = sampleCanvas(pu, pv, pose, true); pu = c.x - d.x; pv = c.y - d.y; }
        v.emplace_back(float(pu + mx_), float(pv + my_));
    }
    return v;
}

} // namespace sim
