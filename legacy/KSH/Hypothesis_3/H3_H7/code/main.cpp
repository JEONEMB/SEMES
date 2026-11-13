// =====================================================================================
// main.cpp — 가설 H3 (물리 제약) · H6 (서브픽셀 치우침) · H7 (세션 보정) 실험
//
//  실행 위치 : 프로젝트 폴더 (img/, legacy/ 가 보이는 곳)
//  인수      : [img 폴더] [--retrack] [--no-phys] [--legacy <legacy 폴더>]
//    --retrack : 대응점 캐시(h3_h7/results/tracks.bin)를 무시하고 다시 추적
//    --no-phys : 완전 물리 모델 학습(수십 초~수 분) 생략
//  비교·정답에 쓰는 이전 결과 (없으면 해당 항목만 건너뜀)
//    legacy/center_check/results/parallax_set1.csv, parallax_set2.csv   배경 시차 추정 (정답 확인용)
//    legacy/h2_ecc/results/h2_ecc/h2_set*_results.csv                   심화 ECC
//    legacy/h2_ecc/results/h2_ecc_opencv/h2_set*_results.csv            OpenCV ECC 호모그래피
//  출력 : h3_h7/results/ (h3_h7_summary.txt, h3_h7_summary.png, h3_set1.csv, h3_set2.csv,
//                         field_model.yml, phys_calib.yml, tracks.bin)
// =====================================================================================
#include <cstdarg>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include "dataset.hpp"
#include "qr_track.hpp"
#include "field_model.hpp"
#include "phys_model.hpp"
#include "bias_h6.hpp"
#include "session_h7.hpp"
#include "analysis.hpp"
#include "report_plot.hpp"
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

using namespace h3;

static const double TOL_T_DEG = 0.01, TOL_Y_MM = 0.09;

static std::string fmt(const char* f, ...)
{
    char b[512]; va_list ap; va_start(ap, f); std::vsnprintf(b, sizeof(b), f, ap); va_end(ap); return b;
}
static std::string statLine(const char* label, const EvalStat& s)
{
    return fmt("  %-26s %3d장 | Θ RMS %.4f° 최대 %.4f° 평균 %+.4f° ±0.01 통과 %3d | Y RMS %6.1f µm ±90 통과 %3d | 둘 다 %3d\n",
               label, s.n, s.rmsT, s.maxT, s.meanT, s.passT, s.rmsY, s.passY, s.passBoth);
}
static std::vector<SessionSample> samples(const std::vector<Item>& it, const std::vector<int>& idx, const std::vector<Estimate>& E)
{
    std::vector<SessionSample> s;
    for (int i : idx) if (E[i].valid) s.push_back({ it[i].y_mm, it[i].t_deg, E[i].y_mm, E[i].t_deg });
    return s;
}

int main(int argc, char** argv)
{
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    fs::path imgDir("img"), legacy("legacy");
    bool retrack = false, doPhys = true;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--retrack") retrack = true;
        else if (a == "--no-phys") doPhys = false;
        else if (a == "--legacy" && i + 1 < argc) legacy = fs::path(argv[++i]);
        else imgDir = fs::path(a);
    }
    const fs::path out = "h3_h7/results";
    fs::create_directories(out);
    std::ostringstream S;   // 요약

    // ------------------------------------------------------------ 0) 영상 목록
    std::vector<Item> items = listImages(imgDir);
    const int iRef = findItem(items, "center.bmp") >= 0 ? findItem(items, "center.bmp") : findItem(items, "center.png");
    const int iC1 = findItem(items, "center1.bmp") >= 0 ? findItem(items, "center1.bmp") : findItem(items, "center1.png");
    if (iRef < 0) { std::cerr << "기준 영상 img/center.bmp 가 없습니다\n"; return 1; }
    std::vector<int> train, testY, testT, set2, set1All;
    for (int i = 0; i < (int)items.size(); ++i) {
        const Item& it = items[i];
        if (it.train) train.push_back(i);
        if (it.test) (it.t_deg == 0 ? testY : testT).push_back(i);
        if (it.set == 2 && !it.isCenter) set2.push_back(i);
        if (it.set == 1 && !it.isCenter) set1All.push_back(i);
    }
    std::cout << "영상 " << items.size() << "장 (학습 " << train.size() << ", 1차 검증 " << testY.size() + testT.size()
              << ", 2차 " << set2.size() << ")\n";

    // ------------------------------------------------------------ 1) 대응점
    std::vector<cv::Point2d> P0;
    std::vector<Track> tracks;
    {
        std::vector<std::string> cachedNames;
        bool useCache = !retrack && loadTracks((out / "tracks.bin").string(), P0, cachedNames, tracks) && cachedNames.size() == items.size();
        if (useCache) for (size_t i = 0; i < items.size(); ++i) if (cachedNames[i] != items[i].name) { useCache = false; break; }
        if (useCache) std::cout << "대응점 : 캐시 사용 (" << P0.size() << "점)\n";
        else {
            QrTracker tk;
            if (!tk.setReference(readGray(items[iRef].path))) { std::cerr << "기준 영상 모서리 추출 실패\n"; return 1; }
            P0 = tk.refPoints();
            tracks.resize(items.size());
            double ms = 0;
            for (size_t i = 0; i < items.size(); ++i) {
                tracks[i] = tk.track(readGray(items[i].path));
                ms += tracks[i].ms;
                if (i % 40 == 0 || i + 1 == items.size())
                    std::cout << "  [" << std::setw(3) << i + 1 << "/" << items.size() << "] " << items[i].name << "  유효점 " << tracks[i].nValid() << "\n";
            }
            std::vector<std::string> names; for (const Item& it : items) names.push_back(it.name);
            saveTracks((out / "tracks.bin").string(), P0, names, tracks);
            std::cout << "  추적 평균 " << std::fixed << std::setprecision(0) << ms / items.size() << " ms/장\n";
        }
    }
    {
        std::vector<int> nv; for (const Track& t : tracks) nv.push_back(t.nValid());
        std::sort(nv.begin(), nv.end());
        double ms = 0; for (const Track& t : tracks) ms += t.ms;
        S << "QR 대응점 : 기준점 " << P0.size() << "개, 장당 유효점 중앙값 " << nv[nv.size() / 2] << " (최소 " << nv.front()
          << "), 추적 " << fmt("%.0f", ms / tracks.size()) << " ms/장\n\n";
    }

    // ------------------------------------------------------------ 2) H3 선형화 물리 모델
    S << "[H3 선형화 물리 모델 — 변위장 기저 분해 → GLS]\n";
    std::vector<FieldSpec> specs(4);
    specs[0].bx = { B_ONE, B_U, B_U2 }; specs[0].by = {};
    specs[1].bx = { B_ONE, B_U, B_U2, B_V, B_UV }; specs[1].by = { B_ONE, B_U, B_V, B_UV };
    specs[2].bx = { B_ONE, B_U, B_U2, B_V, B_UV, B_V2 }; specs[2].by = { B_ONE, B_U, B_U2, B_V, B_UV, B_V2 };
    specs[3].bx = { B_ONE, B_U, B_U2, B_V, B_UV, B_V2, B_U3 }; specs[3].by = specs[2].by;
    const int MAIN = 1;
    std::vector<Estimate> E(items.size()), E2(items.size());
    FieldModel mainModel;
    auto runField = [&](const FieldSpec& sp, const std::vector<Track>& tr, std::vector<Estimate>& e1, std::vector<Estimate>& e2, FieldModel& fm) {
        std::vector<std::vector<double>> K(items.size());
        for (size_t i = 0; i < items.size(); ++i) K[i] = fieldCoeffs(P0, tr[i], sp);
        std::vector<std::vector<double>> Kt; std::vector<double> yt, tt;
        for (int i : train) { Kt.push_back(K[i]); yt.push_back(items[i].y_mm); tt.push_back(items[i].t_deg); }
        fm.fit(Kt, yt, tt, sp.quad);
        e1.assign(items.size(), Estimate()); e2.assign(items.size(), Estimate());
        for (size_t i = 0; i < items.size(); ++i) {
            e1[i] = fm.estimate(K[i]);
            if (iC1 >= 0) e2[i] = fm.estimateRef(K[i], K[iC1]);
        }
    };
    for (int s = 0; s < (int)specs.size(); ++s) {
        std::vector<Estimate> e1, e2; FieldModel fm;
        runField(specs[s], tracks, e1, e2, fm);
        S << "  기저 " << specs[s].label() << (s == MAIN ? "   ← 기본" : "") << "\n";
        S << statLine("  1차 검증 Y 스윕", evalStat(samples(items, testY, e1)));
        S << statLine("  1차 검증 Θ 스윕", evalStat(samples(items, testT, e1)));
        S << statLine("  2차 (center1 기준)", evalStat(samples(items, set2, e2)));
        if (s == MAIN) { E = e1; E2 = e2; mainModel = fm; }
    }
    mainModel.save((out / "field_model.yml").string(), specs[MAIN], "set1 (2026-10-04) even samples, ref img/center.bmp");
    // Θ 행의 u² 감도 (잡음 한계 환산용)
    int iU2 = 2;   // specs[MAIN].bx 의 u² 위치
    const double sensU2 = mainModel.M().at<double>(1, iU2);

    // 물리 확인 (순수 카메라 회전 예측과 비교)
    {
        PhysicsCheck pc = physicsCheck(items, P0, tracks);
        S << "\n[변위장 항 — Θ 1° 당 (Θ 스윕), Y 1 mm 당 (Y 스윕)]  u, v = (x − 1920, y − 1080)/1000\n";
        for (size_t j = 0; j < pc.theta.size(); ++j)
            S << fmt("  %-6s Θ: 기울기 %9.4f  2차 %8.4f  잔차 %.4f   |  Y: 기울기 %9.4f  잔차 %.4f\n", pc.theta[j].name.c_str(),
                     pc.theta[j].slope, pc.theta[j].quad, pc.theta[j].resid, pc.y[j].slope, pc.y[j].resid);
        const double f = pc.theta[0].slope / (CV_PI / 180.0);
        S << fmt("  순수 회전 예측 (f = %.0f px) : dx-1 = f·θ = %.2f,  dx-u2 = dy-uv = f·θ·(1000/f)² = %.3f px/(1000px)²\n",
                 f, f * CV_PI / 180.0, f * CV_PI / 180.0 * (1000.0 / f) * (1000.0 / f));
    }

    // ------------------------------------------------------------ 3) H3 완전 물리 모델
    if (doPhys) {
        S << "\n[H3 완전 물리 모델 — 카메라·평면·Y 방향·회전축 + 라벨 원점·접근 방향 (LM, soft-L1)]\n";
        std::vector<double> yt, tt;
        for (int i : train) { yt.push_back(items[i].y_mm); tt.push_back(items[i].t_deg); }
        PhysParams init; init.v["cx"] = 1912.0; init.v["k1"] = -0.03;
        const std::vector<std::string> freeNames = { "f", "cx", "k1", "na", "nb", "e1", "e2", "w1", "w2", "ax", "az", "bY", "bT", "Y0", "T0" };
        std::cout << "완전 물리 모델 학습 중 ...\n";
        PhysFitResult pr = fitPhysModel(P0, tracks, train, yt, tt, freeNames, init, 100);
        S << fmt("  반복 %d회, %.0f s, 재투영 잔차 RMS x %.4f px, y %.4f px\n", pr.iters, pr.seconds, pr.rmsX, pr.rmsY);
        S << "  매개변수 :";
        for (const auto& k : freeNames) S << fmt(" %s=%.5g", k.c_str(), pr.g[k]);
        S << "\n  → 매개변수가 물리적으로 타당한지 확인할 것 (QR 이 화면 가운데 좁은 띠라 f·주점·왜곡·회전축이 서로 구분되기 어려움)\n";
        cv::FileStorage fsw((out / "phys_calib.yml").string(), cv::FileStorage::WRITE);
        fsw << "model_type" << "PHYS_CAMERA_PLATE_v1" << "note" << "set1 even samples; reference only (parameters poorly identifiable)";
        for (const auto& kv : pr.g.v) fsw << kv.first << kv.second;
        fsw << "rmsX_px" << pr.rmsX << "rmsY_px" << pr.rmsY;
        fsw.release();
        std::vector<ResidualRow> rd = residualDiag(P0, tracks, train, yt, tt, pr.g);
        S << "  잔차 진단 (영상별 dx 잔차 = 평균 + 기울기·s + 곡률·s², s = (x − 1900)/400) — Θ 스윕 일부\n";
        int cnt = 0;
        for (const auto& r : rd) if (r.y_mm == 0 && (cnt++ % 8 == 0))
            S << fmt("    Θ %+.2f : 평균 %+.4f  기울기 %+.4f  곡률 %+.4f  영상 안 σ %.4f\n", r.t_deg, r.mean, r.slope, r.curv, r.within);
    }

    // ------------------------------------------------------------ 4) H6 서브픽셀 치우침
    std::vector<Estimate> E6, E62;
    {
        S << "\n[H6 서브픽셀 치우침 (pixel-locking)]\n";
        H6Diag d = diagnoseH6(P0, tracks, set1All, specs[MAIN]);
        S << fmt("  dx 잔차 σ %.4f px = 점마다 고정된 성분 σ %.4f + 자세마다 바뀌는 성분 σ %.4f\n", d.residSd, d.pointConstSd, d.remainSd);
        S << fmt("  전역 곡선 설명력 %.3f (1차 조화 진폭 %.4f px), 점별 곡선 교차검증 설명력 %.3f\n", d.globalExplained, d.globalAmp, d.pointExplainedCV);
        std::vector<Track> tc = correctH6(P0, tracks, train, specs[MAIN]);
        FieldModel fm6;
        runField(specs[MAIN], tc, E6, E62, fm6);
        S << statLine("  보정 후 1차 검증 Y 스윕", evalStat(samples(items, testY, E6)));
        S << statLine("  보정 후 1차 검증 Θ 스윕", evalStat(samples(items, testT, E6)));
        S << statLine("  보정 후 2차 (center1 기준)", evalStat(samples(items, set2, E62)));
        S << fmt("  잔차 성분 → Θ 환산 (u² 감도 %.3f) : 시간 잡음·자세 의존·고정 성분은 아래 [잡음 한계] 참고\n", sensU2);
    }

    // ------------------------------------------------------------ 5) H7 세션 보정
    std::vector<H7Result> h7;
    {
        S << "\n[H7 세션 보정 — 2차 세트 Θ = 0.01°, 1.00° 자세로 보정, 나머지로 평가]\n";
        const fs::path pc = legacy / "center_check" / "results", ecc = legacy / "h2_ecc" / "results";
        struct Src { std::string name; std::vector<SessionSample> s; };
        std::vector<Src> src;
        bool ok = false;
        auto addCsv = [&](const std::string& nm, const fs::path& p) {
            std::vector<SessionSample> s = readSet2Csv(p.string(), &ok);
            if (ok && !s.empty()) src.push_back({ nm, s }); else S << "  (" << nm << " 결과 없음 : " << u8(p) << ")\n";
        };
        addCsv("Parallax (truth tool)", pc / "parallax_set2.csv");
        addCsv("OpenCV ECC homography", ecc / "h2_ecc_opencv" / "h2_set2_results.csv");
        addCsv("ECC-GN (H2)", ecc / "h2_ecc" / "h2_set2_results.csv");
        src.push_back({ "H3 field model", samples(items, set2, E2) });
        for (const Src& s : src) {
            H7Result r = runH7(s.name, s.s);
            h7.push_back(r);
            S << fmt("  %-24s 배율 %.4f 오프셋 %+.4f° Y오프셋 %+.1f µm | 보정 전 Θ RMS %.4f 통과 %2d/%d → 보정 후 Θ RMS %.4f 통과 %2d/%d, Y ±90 %d\n",
                     s.name.c_str(), r.cal.gainT, r.cal.offT, r.cal.offY * 1000, r.before.rmsT, r.before.passT, r.before.n,
                     r.after.rmsT, r.after.passT, r.after.n, r.after.passY);
        }
    }

    // ------------------------------------------------------------ 6) 한계 분석
    NoiseFloor nf = noiseFloor(testT, P0, tracks, specs[MAIN], sensU2);
    std::vector<TemporalRow> tn = temporalNoise(items, P0, tracks, specs[MAIN], sensU2);
    std::vector<Truth> truth = loadParallaxTruth(items, (legacy / "center_check" / "results" / "parallax_set1.csv").string(),
                                                 (legacy / "center_check" / "results" / "parallax_set2.csv").string());
    {
        S << "\n[잡음 한계]\n";
        S << fmt("  점 잔차만으로 생기는 u² 계수 σ %.4f px/(1000px)² → Θ σ ≈ %.4f° (u² 감도 %.3f)\n", nf.sigU2, nf.thetaSigma, sensU2);
        for (const auto& r : tn)
            S << fmt("  연속 촬영 %s : 촬영 간 점 잔차 σ %.4f px, u² 계수 Θ 환산 [%+.4f %+.4f %+.4f]°\n", r.pose.c_str(), r.pointSd, r.thetaEq[0], r.thetaEq[1], r.thetaEq[2]);
        int nt = 0; for (const Truth& t : truth) nt += t.valid;
        if (nt) {
            S << "\n[배경 시차 정답 대비 (H3 선형화 모델)]\n";
            VsTruth a = compareToTruth(testT, E, truth, items), b = compareToTruth(testY, E, truth, items), c = compareToTruth(set2, E2, truth, items);
            S << fmt("  1차 검증 Θ 스윕 : Θ RMS %.4f° (치우침 %+.4f) 통과 %d/%d | 정답 vs 지령값 RMS %.4f°\n", a.rms, a.bias, a.pass, a.n, a.truthVsLabelRms);
            S << fmt("  1차 검증 Y 스윕 : Θ RMS %.4f° (치우침 %+.4f) 통과 %d/%d | 정답 vs 지령값 RMS %.4f°\n", b.rms, b.bias, b.pass, b.n, b.truthVsLabelRms);
            S << fmt("  2차 세트        : Θ RMS %.4f° (치우침 %+.4f) 통과 %d/%d | 정답 vs 지령값 RMS %.4f°\n", c.rms, c.bias, c.pass, c.n, c.truthVsLabelRms);
            Backlash bl = backlashFromTruth(items, truth);
            S << fmt("  접근 방향 차 (정답 − 지령값, + 접근과 − 접근 평균 차의 절반) : Y ±%.1f µm (%d장), Θ ±%.4f° (%d장)"
                     "  — 하한 : 배경 시차 모델이 지령값으로 학습되어 일부를 흡수함\n",
                     std::fabs(bl.halfY_um), bl.nY, std::fabs(bl.halfT_deg), bl.nT);
        } else S << "  (배경 시차 결과 CSV 가 없어 정답 비교 생략)\n";
    }

    // ------------------------------------------------------------ 7) 저장
    auto writeCsv = [&](const fs::path& p, int set) {
        std::ofstream f(p, std::ios::binary);
        f << "\xEF\xBB\xBF" << "file,set,shot,train,trueY_mm,estY_mm,errY_um,trueT_deg,estT_deg,errT_deg,judge,estT_H6_deg,truthY_mm,truthT_deg,validPts\r\n";
        for (size_t i = 0; i < items.size(); ++i) {
            const Item& it = items[i];
            if (it.set != set) continue;
            const Estimate& e = (set == 2) ? E2[i] : E[i];
            const Estimate& e6 = (set == 2) ? E62[i] : E6[i];
            double eY = (e.y_mm - it.y_mm) * 1000, eT = e.t_deg - it.t_deg;
            bool py = std::fabs(eY) <= TOL_Y_MM * 1000, pt = std::fabs(eT) <= TOL_T_DEG;
            const char* j = !e.valid ? "" : (py && pt) ? "OK" : (!py && !pt) ? "Y.T NG" : (!py ? "Y NG" : "T NG");
            f << "\"" << it.name << "\"," << fmt("%d,%d,%d,%.4f,%.6f,%.2f,%.4f,%.6f,%.6f,%s,%.6f,", it.set, it.shot, it.train ? 1 : 0,
                                                 it.y_mm, e.y_mm, eY, it.t_deg, e.t_deg, eT, j, e6.t_deg);
            if (truth[i].valid) f << fmt("%.6f,%.6f,", truth[i].y_mm, truth[i].t_deg); else f << ",,";
            f << tracks[i].nValid() << "\r\n";
        }
    };
    writeCsv(out / "h3_set1.csv", 1);
    writeCsv(out / "h3_set2.csv", 2);
    const std::string sum = S.str();
    std::cout << "\n" << sum;
    std::ofstream(out / "h3_h7_summary.txt", std::ios::binary) << "\xEF\xBB\xBF" << sum;

    // 그림
    {
        cv::Mat img(560, 1860, CV_8UC3, cv::Scalar::all(255));
        std::vector<Bar> a;
        auto addSet1 = [&](const std::string& label, const EvalStat& st, cv::Scalar col) {
            Bar b; b.label = label; b.value = st.passT; b.text = fmt("%d/%d (RMS %.4f)", st.passT, st.n, st.rmsT); b.color = col; a.push_back(b); };
        EvalStat st;
        const fs::path pc = legacy / "center_check" / "results", ecc = legacy / "h2_ecc" / "results";
        if (set1ThetaTestStat((pc / "parallax_set1.csv").string(), st)) addSet1("Parallax (truth tool)", st, cv::Scalar(177, 184, 185));
        { Bar b; b.label = "Phase Gabor (H1 ref)"; b.value = 42; b.text = "42/100 (RMS 0.0184, H1)"; b.color = cv::Scalar(244, 197, 158); a.push_back(b); }
        addSet1("H3 field model", evalStat(samples(items, testT, E)), cv::Scalar(214, 120, 42));
        addSet1("H3 + H6 correction", evalStat(samples(items, testT, E6)), cv::Scalar(214, 120, 42));
        if (set1ThetaTestStat((ecc / "h2_ecc_opencv" / "h2_set1_results.csv").string(), st)) addSet1("OpenCV ECC homography", st, cv::Scalar(244, 197, 158));
        if (set1ThetaTestStat((ecc / "h2_ecc" / "h2_set1_results.csv").string(), st)) addSet1("ECC-GN (H2)", st, cv::Scalar(244, 197, 158));
        drawBarPanel(img, cv::Rect(0, 0, 620, 560), "(a) Same day: set1 Theta sweep test, pass |err|<=0.01 deg", "pass count", a, 125);
        std::vector<Bar> b;
        for (const H7Result& r : h7) { Bar x; x.label = r.method; x.value = r.before.passT; x.value2 = r.after.passT;
            x.text = fmt("%d", r.before.passT); x.text2 = fmt("%d/%d", r.after.passT, r.after.n); b.push_back(x); }
        drawBarPanel(img, cv::Rect(620, 0, 620, 560), "(b) Other day: set2 + H7 session calibration", "pass count (test poses)", b, 35, "before H7", "after H7");
        H6Diag d = diagnoseH6(P0, tracks, set1All, specs[MAIN]);
        double temporal = tn.empty() ? 0 : tn[0].pointSd;
        std::vector<Bar> c = {
            { "temporal (shot-to-shot)", temporal, -1, fmt("%.3f px", temporal), "", cv::Scalar(177, 184, 185) },
            { "pose-dependent", d.remainSd, -1, fmt("%.3f px", d.remainSd), "", cv::Scalar(52, 104, 235) },
            { "fixed per point", d.pointConstSd, -1, fmt("%.3f px", d.pointConstSd), "", cv::Scalar(200, 206, 207) },
            { "total residual", d.residSd, -1, fmt("%.3f px", d.residSd), "", cv::Scalar(42, 43, 43) } };
        drawBarPanel(img, cv::Rect(1240, 0, 620, 560), "(c) Point error budget (dx residual sigma)", "px", c, 0.10);
        cv::imwrite((out / "h3_h7_summary.png").string(), img);
    }
    std::cout << "결과 : " << u8(out) << "/\n";
    return 0;
}
