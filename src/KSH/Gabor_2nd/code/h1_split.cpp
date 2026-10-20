// =====================================================================================
// h1_split.cpp — 가설 H1 (QR 행 분할 시험) 구현. 2분할(위·아래), 3분할(상·중·하) 공용
// =====================================================================================
#include "h1_split.hpp"
#include "phase_align.hpp"
#include "phase_plot.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cmath>
#include <algorithm>

namespace h1 {

namespace {

// 조각 이름 (그래프는 영어, 콘솔·요약은 한국어)
std::string partNameEn(int k, int N) {
    if (N == 2) return k == 0 ? "top" : "bottom";
    if (N == 3) return k == 0 ? "top" : (k == 1 ? "middle" : "bottom");
    return "part" + std::to_string(k);
}
std::string partNameKo(int k, int N) {
    if (N == 2) return k == 0 ? "위" : "아래";
    if (N == 3) return k == 0 ? "상" : (k == 1 ? "중" : "하");
    return "조각" + std::to_string(k);
}

// 피어슨 상관
double corr(const std::vector<double>& a, const std::vector<double>& b) {
    size_t n = std::min(a.size(), b.size());
    if (n < 3) return NAN;
    double ma = 0, mb = 0;
    for (size_t i = 0; i < n; ++i) { ma += a[i]; mb += b[i]; }
    ma /= n; mb /= n;
    double sab = 0, saa = 0, sbb = 0;
    for (size_t i = 0; i < n; ++i) {
        sab += (a[i] - ma) * (b[i] - mb); saa += (a[i] - ma) * (a[i] - ma); sbb += (b[i] - mb) * (b[i] - mb);
    }
    return (saa > 0 && sbb > 0) ? sab / std::sqrt(saa * sbb) : NAN;
}
double cov(const std::vector<double>& a, const std::vector<double>& b) {
    size_t n = std::min(a.size(), b.size());
    if (n < 2) return 0;
    double ma = 0, mb = 0, s = 0;
    for (size_t i = 0; i < n; ++i) { ma += a[i]; mb += b[i]; }
    ma /= n; mb /= n;
    for (size_t i = 0; i < n; ++i) s += (a[i] - ma) * (b[i] - mb);
    return s / n;
}
double rms(const std::vector<double>& a) {
    if (a.empty()) return NAN;
    double s = 0; for (double v : a) s += v * v; return std::sqrt(s / a.size());
}

// 단서(κ, Δs) 잔차 : 단서 = c0 + c1·Y + c2·Θ + c3·Θ² 를 모든 유효 샘플로 맞춘 뒤 남은 값
std::vector<double> cueResidual(const std::vector<double>& cue, const std::vector<double>& Y, const std::vector<double>& T) {
    int n = (int)cue.size();
    cv::Mat A(n, 4, CV_64F), b(n, 1, CV_64F), x;
    for (int i = 0; i < n; ++i) {
        A.at<double>(i, 0) = 1; A.at<double>(i, 1) = Y[i]; A.at<double>(i, 2) = T[i]; A.at<double>(i, 3) = T[i] * T[i];
        b.at<double>(i) = cue[i];
    }
    cv::solve(A, b, x, cv::DECOMP_SVD);
    std::vector<double> r(n);
    for (int i = 0; i < n; ++i) r[i] = cue[i] - cv::Mat(A.row(i) * x).at<double>(0);
    return r;
}

double curvature(const pa::Features& f) { return f.f[0] + f.f[4] - 2 * f.f[2]; }   // tL + tR − 2·tC
double scaleDiff(const pa::Features& f) { return f.f[5] - f.f[1]; }                // sR − sL

// 조각 a(가로축) vs 조각 b(세로축) 산점도 한 칸
void scatterPanel(cv::Mat& img, cv::Rect area, const std::vector<double>& a, const std::vector<double>& b,
                  const std::string& title, const std::string& xName, const std::string& yName,
                  const std::string& unit, double rho, cv::Scalar color) {
    std::vector<double> all(a);
    all.insert(all.end(), b.begin(), b.end());
    double m = 0;
    for (double v : all) m = std::max(m, std::fabs(v));
    m = (m > 0 ? m : 1) * 1.1;
    pp::Panel P(img, area, -m, m, -m, m);
    P.frame(title, "x: " + xName + "  (" + unit + ")", "");
    P.line(-m, 0, m, 0, cv::Scalar(190, 190, 190), 1);
    P.line(0, -m, 0, m, cv::Scalar(190, 190, 190), 1);
    P.line(-m, -m, m, m, cv::Scalar(120, 120, 120), 1);           // y = x (완전 공통이면 여기에 모임)
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) P.point(a[i], b[i], color, 3, true);
    char buf[96];
    std::snprintf(buf, sizeof(buf), "rho = %.2f   (n = %d)", rho, (int)std::min(a.size(), b.size()));
    P.text(buf, 0, cv::Scalar(20, 20, 160));
    P.text("y: " + yName, 1, cv::Scalar(90, 90, 90));
}

} // namespace

Result runSplitTest(const cv::Mat& ref, const std::string& imgDir, const std::string& outDir, int nParts, bool gui) {
    Result R;
    const int N = std::max(2, nParts);
    const int M = N + 1;                 // 측정 방식 수 : [0] 전체, [1..N] 조각
    R.nParts = N;
    pp::makeDir(outDir);
    std::cout << "\n=== H1 분할 시험 : QR 행 " << N << "분할 독립 측정 ===\n";

    // ------------------------------------------------------------ 1. 측정기 (전체 + 조각 N개)
    std::vector<pa::PhaseAligner> al(M);
    for (int p = 0; p < M; ++p) {
        pa::PhaseParams prm;
        prm.useMagnification = false;    // 2변수 [Y, Θ]
        prm.rowSplit = (p == 0) ? 1 : N;
        prm.rowPart  = (p == 0) ? 0 : p - 1;
        al[p].setParams(prm);
        if (!al[p].setReference(ref)) { std::cerr << "기준 영상 설정 실패\n"; return R; }
    }
    // 조각별 사용 행 (보고용, phase_align.cpp 의 measureQR 과 같은 규칙)
    {
        const int H = al[0].roi(1).height, gap = al[0].params().halfY();
        const int L = std::max(0, (H - 2 * gap * (N - 1)) / N);
        for (int k = 0; k < N; ++k) {
            int b = std::min(H, k * (L + 2 * gap));
            R.rows.push_back(std::make_pair(b, std::min(H, b + L)));
        }
    }

    // ------------------------------------------------------------ 2. 사진 한 장을 M 방식으로 측정
    std::vector<std::vector<pa::Sample>> S(M);
    std::vector<pa::Sample> list = pa::buildSampleList(imgDir);
    auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < list.size(); ++i) {
        cv::Mat g = cv::imread(list[i].path, cv::IMREAD_GRAYSCALE);
        if (g.empty()) continue;
        for (int p = 0; p < M; ++p) {
            pa::Sample s = list[i];
            s.feat = al[p].measure(g);
            S[p].push_back(s);
        }
        if (S[0].size() % 40 == 1) std::cout << "  [" << std::setw(3) << S[0].size() << "] " << list[i].path << "\n";
    }
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "  측정 " << S[0].size() << "장 x " << M << "방식, " << std::fixed << std::setprecision(1) << sec << " s\n";

    // ------------------------------------------------------------ 3. 방식별로 따로 학습(짝수) → 추정
    for (int p = 0; p < M; ++p) {
        pa::Calibration cal = pa::fitCalibration(S[p], nullptr);
        if (!cal.ok) { std::cerr << "보정 실패 (방식 " << p << ")\n"; return R; }
        al[p].setCalibration(cal);
        for (auto& s : S[p]) {
            pa::Estimate e = al[p].estimate(s.feat);
            s.estY = e.valid ? e.y : NAN;
            s.estT = e.valid ? e.t : NAN;
        }
        std::cout << "  [" << std::setw(6) << (p == 0 ? std::string("full") : partNameEn(p - 1, N))
                  << "] 보정 잡음 sigma : sL " << std::setprecision(4) << cal.sigma(1)
                  << ", sR " << cal.sigma(5) << ", tC " << cal.sigma(2) << "\n";
    }
    auto okAll = [&](size_t i) {
        for (int p = 0; p < M; ++p)
            if (!S[p][i].feat.valid || std::isnan(S[p][i].estT)) return false;
        return true;
    };

    // ------------------------------------------------------------ 4. 단서 잔차 (모든 유효 샘플, 방식별 맞춤)
    std::vector<size_t> idx;
    for (size_t i = 0; i < S[0].size(); ++i) if (okAll(i)) idx.push_back(i);
    std::vector<double> Yv, Tv;
    for (size_t i : idx) { Yv.push_back(S[0][i].trueY); Tv.push_back(S[0][i].trueT); }
    std::vector<std::vector<double>> curvRes(M), dsRes(M);
    for (int p = 0; p < M; ++p) {
        std::vector<double> c, d;
        for (size_t i : idx) { c.push_back(curvature(S[p][i].feat)); d.push_back(scaleDiff(S[p][i].feat)); }
        curvRes[p] = cueResidual(c, Yv, Tv);
        dsRes[p]   = cueResidual(d, Yv, Tv);
    }

    // ------------------------------------------------------------ 5. 스윕별 통계
    // 그래프용 : [스윕][방식] 의 Θ 오차·κ 잔차·Δs 잔차
    std::vector<std::vector<double>> eTs[2], cvs[2], dss[2];
    for (int w = 0; w < 2; ++w) {
        char sw = w == 0 ? 'Y' : 'T';
        SweepStat& st = R.sweep[w];
        std::vector<std::vector<double>> eT(M), eY(M), cv_(M), ds_(M);
        std::vector<double> avg;
        for (size_t j = 0; j < idx.size(); ++j) {
            size_t i = idx[j];
            if (S[0][i].sweep != sw) continue;
            for (int p = 0; p < M; ++p) { cv_[p].push_back(curvRes[p][j]); ds_[p].push_back(dsRes[p][j]); }
            if (S[0][i].train) continue;                              // Θ·Y 오차는 검증 샘플만
            double a = 0;
            for (int p = 0; p < M; ++p) {
                eT[p].push_back(S[p][i].estT - S[p][i].trueT);
                eY[p].push_back((S[p][i].estY - S[p][i].trueY) * 1000);
                if (p > 0) a += eT[p].back();
            }
            avg.push_back(a / N);
        }
        st.n = (int)eT[0].size();
        st.fullT = rms(eT[0]); st.fullY = rms(eY[0]);
        double sumVar = 0;
        for (int p = 1; p < M; ++p) {
            st.partT.push_back(rms(eT[p])); st.partY.push_back(rms(eY[p]));
            sumVar += st.partT.back() * st.partT.back();
        }
        st.avgT = rms(avg);
        st.indepT = std::sqrt(sumVar) / N;
        for (int a = 1; a < M; ++a) for (int b = a + 1; b < M; ++b) {
            PairStat ps;
            ps.a = a - 1; ps.b = b - 1;
            ps.rhoT = corr(eT[a], eT[b]);
            ps.rhoY = corr(eY[a], eY[b]);
            ps.sharedT = std::sqrt(std::max(0.0, cov(eT[a], eT[b])));
            ps.rhoCurv = corr(cv_[a], cv_[b]);
            ps.rhoDs = corr(ds_[a], ds_[b]);
            st.pairs.push_back(ps);
        }
        eTs[w] = eT; cvs[w] = cv_; dss[w] = ds_;
    }
    R.ok = true;

    // ------------------------------------------------------------ 6. 판정
    double rhoAll = 0, rhoNear = 0, rhoFar = 0; int nAll = 0, nNear = 0, nFar = 0;
    for (int w = 0; w < 2; ++w)
        for (const PairStat& ps : R.sweep[w].pairs) {
            rhoAll += ps.rhoT; nAll++;
            if (ps.b - ps.a == 1) { rhoNear += ps.rhoT; nNear++; } else { rhoFar += ps.rhoT; nFar++; }
        }
    rhoAll /= std::max(1, nAll);
    std::ostringstream vd;
    vd << std::fixed << std::setprecision(2) << "mean rho(Theta err) = " << rhoAll << " -> ";
    if (rhoAll < 0.3)      vd << "PIXEL NOISE dominant";
    else if (rhoAll > 0.7) vd << "COMMON error dominant";
    else                   vd << "MIXED (part pixel noise, part common error)";
    if (nFar > 0) {
        rhoNear /= nNear; rhoFar /= nFar;
        vd << " | neighbor rho " << rhoNear << " vs top-bottom rho " << rhoFar
           << (rhoNear - rhoFar > 0.15 ? " -> common error VARIES top to bottom" : " -> common error UNIFORM top to bottom");
    }
    R.verdict = vd.str();

    // ------------------------------------------------------------ 7. 출력 (콘솔·요약·CSV)
    std::ostringstream sum;
    sum << "H1 분할 시험 결과 — QR 행 " << N << "분할 (검증 샘플 = 홀수 번째, 2변수 모델)\n"
        << "  방식: 전체 = 기존 QR 전체 행, 조각 = 위에서부터 " << N << "등분 (경계마다 ±" << al[0].params().halfY()
        << "행 제외, 조각 높이 동일, 화소 비공유)\n  조각별 사용 행 (QR 영역 높이 " << al[0].roi(1).height << "행 기준):";
    for (int k = 0; k < N; ++k)
        sum << "  " << partNameKo(k, N) << " " << R.rows[k].first << "~" << R.rows[k].second - 1
            << " (" << R.rows[k].second - R.rows[k].first << "행)";
    sum << "\n\n";
    for (int w = 0; w < 2; ++w) {
        const SweepStat& st = R.sweep[w];
        sum << std::fixed << (w == 0 ? "[Y 스윕]" : "[Θ 스윕]") << " n = " << st.n << "\n" << std::setprecision(4)
            << "  Θ RMS (deg)  전체 " << st.fullT;
        for (int k = 0; k < N; ++k) sum << " | " << partNameKo(k, N) << " " << st.partT[k];
        sum << " | 조각 평균 " << st.avgT << " (조각끼리 독립이면 " << st.indepT << ")\n" << std::setprecision(1)
            << "  Y RMS (um)   전체 " << st.fullY;
        for (int k = 0; k < N; ++k) sum << " | " << partNameKo(k, N) << " " << st.partY[k];
        sum << "\n";
        for (const PairStat& ps : st.pairs)
            sum << std::setprecision(2) << "  " << partNameKo(ps.a, N) << "–" << partNameKo(ps.b, N)
                << " : Θ 오차 ρ " << ps.rhoT << ", Y 오차 ρ " << ps.rhoY
                << " | 곡률 κ ρ " << ps.rhoCurv << ", 배율 차 Δs ρ " << ps.rhoDs
                << std::setprecision(4) << " | 공통 성분 √cov " << ps.sharedT << " deg\n";
        sum << "\n";
    }
    sum << "판정: " << R.verdict << "\n";
    std::cout << "\n" << sum.str();
    std::ofstream(outDir + "/h1_split_summary.txt") << sum.str();

    std::ofstream csv(outDir + "/h1_split_results.csv");
    csv << "path,sweep,train,trueY_mm,trueT_deg";
    for (int p = 0; p < M; ++p) {
        std::string nm = p == 0 ? "full" : partNameEn(p - 1, N);
        for (const char* c : { "tL", "sL", "tC", "sC", "tR", "sR", "valid", "estY_mm", "estT_deg" }) csv << "," << nm << "_" << c;
    }
    csv << "\n" << std::setprecision(6);
    for (size_t i = 0; i < S[0].size(); ++i) {
        const pa::Sample& s = S[0][i];
        csv << s.path << "," << s.sweep << "," << s.train << "," << s.trueY << "," << s.trueT;
        for (int p = 0; p < M; ++p) {
            const pa::Sample& q = S[p][i];
            for (int k = 0; k < pa::NUM_FEAT; ++k) csv << "," << q.feat.f[k];
            csv << "," << q.feat.valid << "," << q.estY << "," << q.estT;
        }
        csv << "\n";
    }
    csv.close();

    // ------------------------------------------------------------ 8. 그래프 (2행 = 스윕, 3열)
    //   2분할 : [Θ 오차, κ 잔차, Δs 잔차] 위 vs 아래
    //   3분할 : Θ 오차 [상 vs 중, 중 vs 하, 상 vs 하]
    const int W = 1440, pw = 360, ph = 320, lx[3] = { 90, 570, 1050 }, ty[2] = { 90, 540 };
    cv::Mat img(1060, W, CV_8UC3, cv::Scalar::all(255));
    std::string head = "H1 split test: QR rows split into " + std::to_string(N) + " parts (independent pixels)";
    cv::putText(img, head, cv::Point(20, 34), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(10, 10, 10), 2, cv::LINE_AA);
    const cv::Scalar colSw[2] = { cv::Scalar(200, 130, 40), cv::Scalar(40, 120, 230) };
    for (int w = 0; w < 2; ++w) {
        std::string tag = w == 0 ? "[Y] " : "[Theta] ";
        const SweepStat& st = R.sweep[w];
        if (N == 2) {
            const PairStat& ps = st.pairs[0];
            scatterPanel(img, cv::Rect(lx[0], ty[w], pw, ph), eTs[w][1], eTs[w][2],
                         tag + "Theta error, test (deg)", "top", "bottom", "deg", ps.rhoT, colSw[w]);
            scatterPanel(img, cv::Rect(lx[1], ty[w], pw, ph), cvs[w][1], cvs[w][2],
                         tag + "curvature tL+tR-2tC resid.", "top", "bottom", "px", ps.rhoCurv, colSw[w]);
            scatterPanel(img, cv::Rect(lx[2], ty[w], pw, ph), dss[w][1], dss[w][2],
                         tag + "scale diff sR-sL resid.", "top", "bottom", "px/100px", ps.rhoDs, colSw[w]);
        } else {
            for (int c = 0; c < 3 && c < (int)st.pairs.size(); ++c) {
                const PairStat& ps = st.pairs[c == 0 ? 0 : (c == 1 ? 2 : 1)];   // 상-중, 중-하, 상-하 순서
                std::string xa = partNameEn(ps.a, N), yb = partNameEn(ps.b, N);
                scatterPanel(img, cv::Rect(lx[c], ty[w], pw, ph), eTs[w][ps.a + 1], eTs[w][ps.b + 1],
                             tag + "Theta err: " + xa + " vs " + yb, xa, yb, "deg", ps.rhoT, colSw[w]);
            }
        }
    }
    cv::putText(img, "rho ~ 0 : independent pixel noise (cloud is round)     rho ~ 1 : common error (points on y = x line)",
                cv::Point(20, 985), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(60, 60, 60), 1, cv::LINE_AA);
    cv::putText(img, R.verdict, cv::Point(20, 1025), cv::FONT_HERSHEY_SIMPLEX, N == 2 ? 0.6 : 0.5, cv::Scalar(20, 20, 160),
                N == 2 ? 2 : 1, cv::LINE_AA);
    pp::saveAndShow("H1 split", img, outDir + "/h1_split.png", gui);
    std::cout << "저장: " << outDir << "/h1_split.png, h1_split_results.csv, h1_split_summary.txt\n";
    return R;
}

} // namespace h1
