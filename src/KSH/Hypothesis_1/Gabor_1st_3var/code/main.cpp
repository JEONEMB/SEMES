// =====================================================================================
// main.cpp — 위상 기반 Y·Θ 분리 : 학습(보정) → 검증 → 그래프 → alignment() 사용 예
//
// 실행 위치 : 프로젝트 폴더 (img/center.bmp, img/Y/*.bmp, img/Theta/*.bmp 가 보이는 곳)
// 실행 옵션 : --no-gui  창을 띄우지 않고 파일만 저장
//             --tune    필터 매개변수(λ, σx, σy) 조합을 비교 (논문 방식의 필터 선택, 시간이 걸림)
// 출력      : phase_calib.yml (보정 모델), phase_results.csv (샘플별 특징·추정값),
//             phase_features.png, phase_estimation.png, phase_quality.png, phase_overlay_*.png
// ※ Debug 빌드는 느리므로 Release 로 실행 권장
// =====================================================================================
#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <cmath>
#include "alignment.hpp"
#include "phase_plot.hpp"
using namespace std;

static const string IMG_DIR    = "img";
static const string CALIB_PATH = "phase_calib.yml";
static const double TOL_Y_MM   = 0.09;   // Y 허용 오차 (mm)
static const double TOL_T_DEG  = 0.09;   // Θ 허용 오차 (deg)

// 모든 샘플 측정 (영상은 한 장씩 읽고 바로 버림 → 메모리 절약). 없는 파일은 목록에서 뺌
static void measureAll(const pa::PhaseAligner& al, vector<pa::Sample>& samples, bool verbose) {
    vector<pa::Sample> kept;
    auto t0 = chrono::steady_clock::now();
    for (size_t i = 0; i < samples.size(); ++i) {
        Mat g = imread(samples[i].path, IMREAD_GRAYSCALE);
        if (g.empty()) continue;
        samples[i].feat = al.measure(g);
        if (verbose && kept.size() % 20 == 0)
            cout << "  [" << setw(3) << kept.size() + 1 << "] " << samples[i].path
                 << (samples[i].feat.valid ? "" : "  (측정 실패)") << "\n";
        kept.push_back(samples[i]);
    }
    samples.swap(kept);
    double sec = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
    if (verbose) cout << "  측정 " << samples.size() << "장, " << fixed << setprecision(1) << sec << " s ("
                      << setprecision(0) << 1000.0 * sec / std::max<size_t>(1, samples.size()) << " ms/장, 파일 읽기 포함)\n";
}

// 검증 통계
struct Stat { int n = 0, passY = 0, passT = 0; double rmsY = 0, rmsT = 0, maxY = 0, maxT = 0; };
static Stat evaluate(const vector<pa::Sample>& samples, char sweep) {
    Stat st; double ssY = 0, ssT = 0;
    for (auto& s : samples) {
        if (s.train || s.sweep != sweep || !s.feat.valid || std::isnan(s.estY)) continue;
        double eY = (s.estY - s.trueY) * 1000, eT = s.estT - s.trueT;
        st.n++; ssY += eY * eY; ssT += eT * eT;
        st.maxY = std::max(st.maxY, fabs(eY)); st.maxT = std::max(st.maxT, fabs(eT));
        if (fabs(eY) <= TOL_Y_MM * 1000) st.passY++;
        if (fabs(eT) <= TOL_T_DEG) st.passT++;
    }
    if (st.n) { st.rmsY = sqrt(ssY / st.n); st.rmsT = sqrt(ssT / st.n); }
    return st;
}

// 학습 → 추정까지 한 번에 (튜닝·본 실행 공용). useMag : 3변수(공통 배율 포함) 여부
static pa::Calibration trainAndEstimate(pa::PhaseAligner& al, vector<pa::Sample>& samples, bool useMag,
                                        vector<double>* magOut = nullptr) {
    double n[pa::NUM_FEAT];
    al.magnificationDirection(n);
    pa::Calibration cal = pa::fitCalibration(samples, useMag ? n : nullptr);
    al.setCalibration(cal);
    if (magOut) magOut->assign(samples.size(), NAN);
    for (size_t i = 0; i < samples.size(); ++i) {
        pa::Estimate e = al.estimate(samples[i].feat);
        samples[i].estY = e.valid ? e.y : NAN;
        samples[i].estT = e.valid ? e.t : NAN;
        if (magOut && e.valid) (*magOut)[i] = e.mag;
    }
    return cal;
}

// 필터 매개변수 비교 : 조합마다 전체 학습·검증을 다시 해서 Y·Θ 검증 오차를 비교
static void tuneFilter(const Mat& ref) {
    cout << "\n=== 필터 매개변수 비교 (검증 샘플 기준) ===\n";
    cout << " lambda sigmaX sigmaY |  Y RMS(um) [Y swp / T swp] | Theta RMS(deg) [Y swp / T swp]\n";
    const double lambdas[] = { 8, 10, 12, 16 };
    const double sxs[]     = { 3, 4, 5 };
    const double sys_[]    = { 4, 6, 10 };
    for (double lam : lambdas) for (double sx : sxs) for (double sy : sys_) {
        pa::PhaseParams p; p.lambda = lam; p.sigmaX = sx; p.sigmaY = sy;
        pa::PhaseAligner al(p);
        if (!al.setReference(ref)) continue;
        vector<pa::Sample> s = pa::buildSampleList(IMG_DIR);
        measureAll(al, s, false);
        trainAndEstimate(al, s, p.useMagnification);
        Stat a = evaluate(s, 'Y'), b = evaluate(s, 'T');
        cout << fixed << setprecision(1) << setw(7) << lam << setw(7) << sx << setw(7) << sy << " | "
             << setw(10) << a.rmsY << " / " << setw(7) << b.rmsY << "        | "
             << setprecision(4) << setw(8) << a.rmsT << " / " << setw(7) << b.rmsT << "\n";
    }
}

int main(int argc, char** argv)
{
    bool gui = true, tune = false;
    for (int i = 1; i < argc; ++i) {
        string a = argv[i];
        if (a == "--no-gui") gui = false;
        if (a == "--tune") tune = true;
    }
    cout << "OpenCV version: " << CV_VERSION << "\n";

    // ---------------------------------------------------------------- 1. 기준 영상
    Mat ref = imread(IMG_DIR + "/center.bmp", IMREAD_GRAYSCALE);
    if (ref.empty()) { cerr << "기준 영상(img/center.bmp)을 읽을 수 없습니다.\n"; return -1; }
    if (tune) tuneFilter(ref);

    pa::PhaseParams params;                 // 기본값 : λ 12, σx 4, σy 6 (phase_align.hpp)
    pa::PhaseAligner aligner(params);
    if (!aligner.setReference(ref)) { cerr << "기준 영상에서 QR 3개를 찾지 못했습니다.\n"; return -1; }

    // ---------------------------------------------------------------- 2. 전체 측정
    cout << "\n=== 측정 (Y ±2 mm, Θ ±1 deg) ===\n";
    vector<pa::Sample> samples = pa::buildSampleList(IMG_DIR);
    measureAll(aligner, samples, true);

    // ---------------------------------------------------------------- 3. 학습(짝수 번째) → 2변수 / 3변수 비교
    vector<pa::Sample> s2 = samples;                          // 2변수 결과 보관용
    trainAndEstimate(aligner, s2, false);
    vector<double> magEst;
    pa::Calibration cal = trainAndEstimate(aligner, samples, params.useMagnification, &magEst);
    if (!cal.ok) { cerr << "보정 모델을 만들 수 없습니다 (학습 샘플 부족).\n"; return -1; }
    cal.save(CALIB_PATH, aligner.params());

    const char* fn[pa::NUM_FEAT] = { "tL", "sL", "tC", "sC", "tR", "sR" };
    cout << "\n=== 보정 모델 (특징 = a*Y + b*Theta + c" << (cal.useMag ? " + n*m" : "") << ") ===   t: px, s: px/100px\n";
    for (int k = 0; k < pa::NUM_FEAT; ++k) {
        cout << "  " << fn[k] << " : a = " << fixed << setprecision(4) << setw(9) << cal.a[k] << " /mm,  b = "
             << setw(9) << cal.b[k] << " /deg,  noise = " << setw(7) << cal.sigma(k);
        if (cal.useMag) cout << ",  n = " << scientific << setprecision(2) << cal.n[k] << " /ppm" << fixed;
        cout << "\n";
    }
    cout << "  저장: " << CALIB_PATH << "\n";

    // ---------------------------------------------------------------- 4. 검증 (홀수 번째) : 2변수 vs 3변수
    cout << "\n=== 검증 결과 (학습에 쓰지 않은 홀수 번째 샘플) ===\n";
    const char* label[2] = { "2변수 [Y,Θ]      ", "3변수 [Y,Θ,배율] " };
    for (int v = 0; v < 2; ++v) {
        if (v == 1 && !cal.useMag) break;
        const vector<pa::Sample>& S = v == 0 ? s2 : samples;
        for (char sw : { 'Y', 'T' }) {
            Stat st = evaluate(S, sw);
            if (!st.n) continue;
            cout << "  " << label[v] << (sw == 'Y' ? "[Y 스윕] " : "[Θ 스윕] ") << setw(3) << st.n << "장 | Y RMS " << fixed
                 << setprecision(1) << setw(6) << st.rmsY << " um, 최대 " << setw(6) << st.maxY << " um, 통과 "
                 << setw(3) << st.passY << "/" << st.n << " | Θ RMS " << setprecision(4) << st.rmsT << " deg, 최대 "
                 << st.maxT << ", 통과 " << st.passT << "/" << st.n << "\n";
        }
    }
    if (cal.useMag) {
        double sm = 0, ss = 0; int nm = 0;
        for (double m : magEst) if (!std::isnan(m)) { sm += m; ss += m * m; nm++; }
        if (nm > 1) cout << "  추정된 공통 배율 변화 : 평균 " << setprecision(1) << sm / nm << " ppm, 표준편차 "
                         << sqrt(std::max(0.0, ss / nm - (sm / nm) * (sm / nm))) << " ppm\n";
    }

    // ---------------------------------------------------------------- 5. CSV
    ofstream csv("phase_results.csv");
    csv << "path,sweep,train,trueY_mm,trueT_deg,coarseDx,tL,sL,tC,sC,tR,sR,validQR,estY_mm,estT_deg,estMag_ppm,estY2_mm,estT2_deg";
    for (int q = 0; q < pa::NUM_QR; ++q) csv << ",q" << q << "_validRatio,q" << q << "_fitStd,q" << q << "_gradSpread";
    csv << "\n" << setprecision(6);
    for (size_t i = 0; i < samples.size(); ++i) {
        const pa::Sample& s = samples[i];
        csv << s.path << "," << s.sweep << "," << s.train << "," << s.trueY << "," << s.trueT << "," << s.feat.coarseDx;
        for (int k = 0; k < pa::NUM_FEAT; ++k) csv << "," << s.feat.f[k];
        csv << "," << s.feat.numValidQR << "," << s.estY << "," << s.estT << "," << magEst[i]
            << "," << s2[i].estY << "," << s2[i].estT;
        for (int q = 0; q < pa::NUM_QR; ++q)
            csv << "," << s.feat.qr[q].validRatio << "," << s.feat.qr[q].fitStd << "," << s.feat.qr[q].gradSpread;
        csv << "\n";
    }
    csv.close();
    cout << "\n저장: phase_results.csv\n";

    // ---------------------------------------------------------------- 6. 그래프
    pp::saveAndShow("features",   pp::plotFeatureSweeps(samples, cal), "phase_features.png", gui);
    pp::saveAndShow("estimation", pp::plotEstimation(samples, TOL_Y_MM, TOL_T_DEG), "phase_estimation.png", gui);
    pp::saveAndShow("quality",    pp::plotQRQuality(samples), "phase_quality.png", gui);
    if (cal.useMag) {
        pp::saveAndShow("compare", pp::plotModelCompare(s2, samples, TOL_Y_MM, TOL_T_DEG), "phase_compare_2v3.png", gui);
        pp::saveAndShow("magnification", pp::plotMagnification(samples, magEst), "phase_magnification.png", gui);
    }

    // ---------------------------------------------------------------- 7. alignment() 사용 예 + 오버레이
    if (alignment_init(IMG_DIR + "/center.bmp", CALIB_PATH)) {
        const string demo[] = { IMG_DIR + "/Theta/T+0.35.bmp", IMG_DIR + "/Y/Y+1.05.bmp" };
        const string out[]  = { "phase_overlay_T.png", "phase_overlay_Y.png" };
        for (int i = 0; i < 2; ++i) {
            Mat img = imread(demo[i], IMREAD_GRAYSCALE);
            if (img.empty()) continue;
            Alignment_t r = alignment(img);
            cout << "[alignment] " << demo[i] << " -> y = " << fixed << setprecision(4) << r.y
                 << " mm, t = " << r.t << " deg   (로봇 보정값 : " << -r.y << " mm, " << -r.t << " deg)\n";
            pa::Features f = alignment_engine().measure(img);
            pa::Estimate e = alignment_engine().estimate(f);
            pp::saveAndShow(out[i], pp::drawQROverlay(img, alignment_engine(), f, e, demo[i]), out[i], gui);
        }
    }
    if (gui) {
        cout << "\n그래프 창에서 아무 키나 누르면 종료합니다.\n";
        waitKey(0);
        destroyAllWindows();
    }
    return 0;
}
