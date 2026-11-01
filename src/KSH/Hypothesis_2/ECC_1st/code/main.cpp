// =====================================================================================
// main.cpp — 가설 H2 (ECC 전역 원근 정합) 검증
//
//  실행 위치 : 프로젝트 폴더 (img/ 가 보이는 곳).  인수 : [img 폴더] (기본 "img")
//  순서
//    1) 1차 세트 (img/center.bmp 기준, Y 80장 + Θ 200장) 측정 → 짝수 번째로 학습, 홀수 번째로 검증
//    2) 2차 세트 (일부 데이터 추출/기준 위치/center1 기준, Y·Θ 동시 19 조합 × 3장) 측정
//       → 1차 세트로 학습한 모델로 추정 (다른 날 촬영 + 동시 이동 + Θ 2° 외삽 검증), 같은 자세 3장 반복성
//    3) 결과 : results/h2_ecc/ (CSV, 그래프, 요약),  모델 : ecc_model.yml (alignment() 와 MFC 앱용)
//  ※ 장당 ECC 가 수백 ms 걸리므로 Release 로 실행 권장
// =====================================================================================
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <map>
#include <cmath>
#include <chrono>
#include "alignment.hpp"
#include "ecc_eval.hpp"
#include "result_plot.hpp"
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>   // 콘솔 한글 출력 (UTF-8)
#endif

namespace fs = std::filesystem;

static const double TOL_Y_MM  = 0.09;   // Y 허용 오차 (mm)
static const double TOL_T_DEG = 0.01;   // Θ 허용 오차 (deg) — 사진마다 ±0.01°
static const char*  OUT_DIR   = "results/h2_ecc";
static const char*  MODEL     = "ecc_model.yml";

// 오차 통계
struct Stat { int n = 0, passT = 0, passY = 0, passBoth = 0; double rmsT = 0, maxT = 0, meanT = 0, rmsY = 0, maxY = 0; };
template <class Pred>
static Stat calcStat(const std::vector<ev::Item>& v, Pred keep) {
    Stat s;
    for (const ev::Item& it : v) {
        if (!it.hasTruth || !it.e.valid || !keep(it)) continue;
        double eT = it.e.t - it.trueT, eY = (it.e.y - it.trueY) * 1000;
        s.n++; s.rmsT += eT * eT; s.meanT += eT; s.rmsY += eY * eY;
        s.maxT = std::max(s.maxT, std::fabs(eT)); s.maxY = std::max(s.maxY, std::fabs(eY));
        bool pt = std::fabs(eT) <= TOL_T_DEG, py = std::fabs(eY) <= TOL_Y_MM * 1000;
        s.passT += pt; s.passY += py; s.passBoth += (pt && py);
    }
    if (s.n) { s.rmsT = std::sqrt(s.rmsT / s.n); s.meanT /= s.n; s.rmsY = std::sqrt(s.rmsY / s.n); }
    return s;
}
static std::string fmtStat(const char* label, const Stat& s) {
    std::ostringstream o;
    o << "  " << std::left << std::setw(26) << label << std::right << std::setw(4) << s.n << "장 | Θ RMS "
      << std::fixed << std::setprecision(4) << s.rmsT << "° 최대 " << s.maxT << "° 평균 " << std::showpos << s.meanT
      << std::noshowpos << "° ±0.01 통과 " << s.passT << "/" << s.n << " | Y RMS " << std::setprecision(1) << s.rmsY
      << " µm 최대 " << s.maxY << " µm ±90 통과 " << s.passY << "/" << s.n << " | 둘 다 " << s.passBoth << "/" << s.n << "\n";
    return o.str();
}

static void estimateAll(const ea::Model& md, std::vector<ev::Item>& v) {
    for (ev::Item& it : v) it.e = it.m.ok ? ea::estimate(md, it.m.f) : ea::Estimate();
}

static std::vector<rp::Rec> toRecs(const std::vector<ev::Item>& v, bool testOnly) {
    std::vector<rp::Rec> out;
    for (const ev::Item& it : v) {
        if (testOnly && it.train) continue;
        rp::Rec r;
        r.name = it.name; r.trueY = it.trueY; r.trueT = it.trueT; r.estY = it.e.y; r.estT = it.e.t;
        r.hasTruth = it.hasTruth; r.hasEst = it.e.valid;
        out.push_back(r);
    }
    return out;
}

int main(int argc, char** argv)
{
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    std::cout << "OpenCV version: " << CV_VERSION << "\n";
    // 인수 : [img 폴더] [--opencv-ecc]   (--opencv-ecc : 비교용 OpenCV findTransformECC 호모그래피)
    fs::path imgDir("img");
    bool opencvEcc = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--opencv-ecc") opencvEcc = true;
        else imgDir = fs::path(a);
    }
    const fs::path set2Dir = imgDir / fs::path(L"일부 데이터 추출");
    std::ostringstream sum;
    sum << "가설 H2 — ECC 전역 원근 정합 : 검증 결과\n"
        << "  H2 : QR 3개를 포함한 띠 전체를 ECC(정규화 상관) 정합 한 번으로 맞추면 원근 성분을 긴 기준선에서 직접 추정해\n"
        << "       Θ 측정 잡음이 위상(Gabor) 방식보다 작아지고 Y·Θ 를 더 정확히 구분한다.\n"
        << "  판정 기준 : 1차 세트 검증 Θ RMS < 위상 방식 0.018~0.020°,  사진마다 |Θ 오차| ≤ 0.01° 통과율 증가\n\n";

    // ------------------------------------------------------------ 1) 1차 세트
    cv::Mat ref1 = ev::readGray(imgDir / "center.bmp");
    ea::EccParams prm;                       // 기본 : method 1 (ZNCC Gauss-Newton 원근 다항 워프)
    if (opencvEcc) { prm.method = 0; prm.blurSigma = 0; prm.maxIter = 100; prm.eps = 1e-8; }
    const std::string outDir = std::string(OUT_DIR) + (opencvEcc ? "_opencv" : "");
    fs::create_directories(outDir);
    sum << "  방법 : " << (opencvEcc ? "OpenCV findTransformECC 호모그래피 (비교용)" : "심화 ECC — ZNCC Gauss-Newton 원근 다항 워프 (기본)") << "\n\n";
    ea::EccAligner al1(prm);
    if (ref1.empty() || !al1.setReference(ref1)) { std::cerr << "1차 세트 기준(img/center.bmp)을 읽거나 QR 을 찾지 못했습니다.\n"; return -1; }
    std::vector<ev::Item> set1 = ev::listSet1(imgDir);
    std::cout << "\n=== 1차 세트 측정 (" << set1.size() << "장, 기준 center.bmp) ===\n";
    ev::measureAll(al1, set1, "set1");

    std::vector<ea::Sample> train;
    for (const ev::Item& it : set1) {
        if (!it.train || !it.m.ok) continue;
        ea::Sample s; s.f.assign(it.m.f, it.m.f + ea::NUM_FEAT); s.y = it.trueY; s.t = it.trueT;
        train.push_back(s);
    }
    ea::Model mdQ = ea::fitModel(train, prm, true,  "set1 (2026-10-04) even samples, ref img/center.bmp, Theta^2 term");
    ea::Model mdL = ea::fitModel(train, prm, false, "set1 (2026-10-04) even samples, ref img/center.bmp, linear");
    if (!mdQ.ok) { std::cerr << "학습 실패 (학습 샘플 부족)\n"; return -1; }

    const char* fn[ea::NUM_FEAT] = { "tL", "sL", "tC", "sC", "tR", "sR" };
    sum << "[보정 모델] f = a*Y + b*Θ + q*Θ² + c  (학습 " << mdQ.nTrain << "장)\n";
    for (int k = 0; k < ea::NUM_FEAT; ++k)
        sum << "  " << fn[k] << " : a " << std::setw(9) << std::fixed << std::setprecision(4) << mdQ.a[k] << " /mm  b " << std::setw(9) << mdQ.b[k]
            << " /deg  q " << std::setw(8) << mdQ.q[k] << " /deg²  잡음 σ " << std::setprecision(5) << mdQ.sigma(k) << "\n";
    sum << "\n";

    auto isY = [](const ev::Item& it) { return it.set == "Y"; };
    auto isT = [](const ev::Item& it) { return it.set == "Theta"; };
    for (int v = 0; v < 2; ++v) {
        const ea::Model& md = v == 0 ? mdQ : mdL;
        estimateAll(md, set1);
        sum << "[1차 세트 검증 — 홀수 번째] 모델 " << (v == 0 ? "Θ² 포함" : "선형") << "\n";
        sum << fmtStat("Y 단독 (Y ±2 mm)", calcStat(set1, [&](const ev::Item& it) { return !it.train && isY(it); }));
        sum << fmtStat("Θ 단독 (Θ ±1°)", calcStat(set1, [&](const ev::Item& it) { return !it.train && isT(it); }));
        sum << "\n";
    }
    estimateAll(mdQ, set1);
    ev::writeCsv(fs::path(outDir) / "h2_set1_results.csv", set1, TOL_Y_MM, TOL_T_DEG);
    cv::imwrite(outDir + "/h2_set1.png",
                rp::drawReport(toRecs(set1, true), TOL_Y_MM, TOL_T_DEG, "H2 ECC  |  set 1 test (odd samples)  |  Theta^2 model"));

    // ------------------------------------------------------------ 2) 2차 세트
    std::vector<ev::Item> set2 = ev::listSet2(set2Dir);
    if (!set2.empty()) {
        fs::path c1 = set2Dir / fs::path(L"기준 위치") / "center1.bmp";
        if (!fs::exists(c1)) c1 = set2Dir / fs::path(L"기준 위치") / "center1.png";
        cv::Mat ref2 = ev::readGray(c1);
        ea::EccAligner al2(prm);
        if (ref2.empty() || !al2.setReference(ref2)) { std::cerr << "2차 세트 기준(center1)을 읽거나 QR 을 찾지 못했습니다.\n"; }
        else {
            std::cout << "\n=== 2차 세트 측정 (" << set2.size() << "장, 기준 center1) ===\n";
            ev::measureAll(al2, set2, "set2");
            auto notCenter = [](const ev::Item& it) { return it.set != "center"; };
            for (int v = 0; v < 2; ++v) {
                estimateAll(v == 0 ? mdQ : mdL, set2);
                sum << "[2차 세트 — Y·Θ 동시 이동, 1차 세트 모델, 기준 center1] 모델 " << (v == 0 ? "Θ² 포함" : "선형") << "\n";
                sum << fmtStat("전체 (center 제외)", calcStat(set2, notCenter));
                sum << fmtStat("Θ ≤ 0.20°", calcStat(set2, [&](const ev::Item& it) { return notCenter(it) && it.trueT <= 0.2 + 1e-9; }));
                sum << fmtStat("Θ = 1.00° (학습 범위 끝)", calcStat(set2, [&](const ev::Item& it) { return notCenter(it) && std::fabs(it.trueT - 1.0) < 1e-9; }));
                sum << fmtStat("Θ = 2.00° (학습 범위 밖)", calcStat(set2, [&](const ev::Item& it) { return notCenter(it) && std::fabs(it.trueT - 2.0) < 1e-9; }));
                sum << fmtStat("center2·3 (같은 기준 자세)", calcStat(set2, [&](const ev::Item& it) { return it.set == "center" && it.shot != 1; }));
                sum << "\n";
            }
            estimateAll(mdQ, set2);

            // 같은 자세 3장 반복성 : 묶음 안 표준편차를 묶음 수로 합침 (pooled)
            std::map<std::string, std::vector<const ev::Item*>> groups;
            for (const ev::Item& it : set2) if (it.set != "center" && it.e.valid) groups[it.set].push_back(&it);
            double ssT = 0, ssY = 0, ssTC = 0, ssDS = 0; int dof = 0;
            double maxRangeT = 0, maxRangeTC = 0;
            for (const auto& g : groups) {
                const auto& v = g.second;
                if (v.size() < 2) continue;
                double mT = 0, mY = 0, mTC = 0, mDS = 0;
                for (auto p : v) { mT += p->e.t; mY += p->e.y; mTC += p->m.f[2]; mDS += p->m.f[5] - p->m.f[1]; }
                mT /= v.size(); mY /= v.size(); mTC /= v.size(); mDS /= v.size();
                double lo = 1e9, hi = -1e9, loC = 1e9, hiC = -1e9;
                for (auto p : v) {
                    ssT += (p->e.t - mT) * (p->e.t - mT); ssY += (p->e.y - mY) * (p->e.y - mY);
                    ssTC += (p->m.f[2] - mTC) * (p->m.f[2] - mTC);
                    double ds = p->m.f[5] - p->m.f[1]; ssDS += (ds - mDS) * (ds - mDS);
                    lo = std::min(lo, p->e.t); hi = std::max(hi, p->e.t); loC = std::min(loC, p->m.f[2]); hiC = std::max(hiC, p->m.f[2]);
                }
                dof += (int)v.size() - 1;
                maxRangeT = std::max(maxRangeT, hi - lo); maxRangeTC = std::max(maxRangeTC, hiC - loC);
            }
            if (dof > 0) {
                sum << "[같은 자세 3장 반복성 — " << groups.size() << " 자세, 자유도 " << dof << "]\n" << std::fixed
                    << "  Θ 추정 σ " << std::setprecision(4) << std::sqrt(ssT / dof) << "°  (묶음 안 최대 폭 " << maxRangeT << "°)\n"
                    << "  Y 추정 σ " << std::setprecision(1) << std::sqrt(ssY / dof) * 1000 << " µm\n"
                    << "  가운데 QR 이동 tC σ " << std::setprecision(3) << std::sqrt(ssTC / dof) << " px (최대 폭 " << maxRangeTC << " px)\n"
                    << "  배율 차 sR−sL σ " << std::setprecision(4) << std::sqrt(ssDS / dof) << " px/100px\n\n";
            }
            ev::writeCsv(fs::path(outDir) / "h2_set2_results.csv", set2, TOL_Y_MM, TOL_T_DEG);
            cv::imwrite(outDir + "/h2_set2.png",
                        rp::drawReport(toRecs(set2, false), TOL_Y_MM, TOL_T_DEG, "H2 ECC  |  set 2 (Y+Theta, 3 shots, ref center1)  |  model from set 1"));
            cv::imwrite(outDir + "/h2_repeat.png", ev::drawRepeatability(set2, TOL_T_DEG));
        }
    }

    // ------------------------------------------------------------ 3) 모델 저장 + 요약
    if (!opencvEcc) mdQ.save(MODEL);
    mdQ.save(outDir + "/" + MODEL);
    mdL.save(outDir + "/ecc_model_linear.yml");
    sum << "모델 저장 : " << (opencvEcc ? outDir + "/" + MODEL : std::string(MODEL)) << " (Θ² 포함), " << outDir << "/ecc_model_linear.yml (선형)\n";
    std::cout << "\n" << sum.str();
    std::ofstream(fs::path(outDir) / "h2_summary.txt", std::ios::binary) << "\xEF\xBB\xBF" << sum.str();

    // ------------------------------------------------------------ 4) alignment() 사용 예
    if (!opencvEcc && alignment_init((imgDir / "center.bmp").string(), MODEL)) {
        cv::Mat img = ev::readGray(imgDir / "Theta" / "T+0.35.bmp");
        if (!img.empty()) {
            Alignment_t r = alignment(img);
            std::cout << "[alignment] Theta/T+0.35 -> y = " << std::setprecision(4) << r.y << " mm, t = " << r.t
                      << " deg  (로봇 보정값 : " << -r.y << " mm, " << -r.t << " deg)\n";
        }
    }
    return 0;
}
