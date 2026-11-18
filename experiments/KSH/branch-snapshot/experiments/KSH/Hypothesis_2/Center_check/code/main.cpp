// =====================================================================================
// main.cpp — 기준(center) 영상 특징 분석 + 깊이별 랜드마크 시차로 Y·Θ 분리
//
//  실행 위치 : 프로젝트 폴더 (img/ 가 보이는 곳)
//  인수      : [img 폴더] [--centers-only]
//  입력      : img/center.bmp (10/04, 기준), img/center1~3.bmp (10/15),
//              img/Y±a.bmp · img/T±b.bmp (1차 세트), img/Y+a T+b (n).bmp (2차 세트)
//  출력      : results/center/
//    1) center_landmarks.png   : 기준 영상 위 랜드마크 위치
//       center_features.csv    : center 4장의 랜드마크별 이동 (center.bmp 대비)
//       center_summary.csv     : center 4장의 묶음별 이동, 밝기·초점·잡음 (추정 Y·Θ 는 parallax_summary.txt)
//    2) parallax_model.yml     : 시차 모델 (프로젝트 폴더에도 저장 → MFC 앱용)
//       parallax_set1.csv / parallax_set2.csv / parallax_summary.txt
//       landmarks.yml          : 랜드마크 ROI (프로젝트 폴더, 수정 후 다시 실행 가능)
//  ※ 영상 한 장당 0.3 ~ 0.5 s → Release 로 실행 권장
// =====================================================================================
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <regex>
#include <map>
#include <cmath>
#include <algorithm>
#include "center_feature.hpp"
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>   // 콘솔 한글 출력 (UTF-8)
#endif

namespace fs = std::filesystem;

static const double TOL_Y_MM  = 0.09;
static const double TOL_T_DEG = 0.01;
static const char*  OUT_DIR   = "results/center";
static const char*  MODEL     = "parallax_model.yml";
static const char*  LM_FILE   = "landmarks.yml";

// ---------------------------------------------------------------------------------
struct Item {
    fs::path path;
    std::string name;
    int set = 1;            // 1 : 1차 세트 (10/04), 2 : 2차 세트 (10/15)
    int shot = 0;           // 2차 세트 (1)(2)(3), centerN 은 N
    bool isCenter = false;
    double trueY = 0, trueT = 0;
    bool train = false;
    cf::Measure m;
    cf::Estimate e;
    std::vector<double> dx;
};

static std::string u8(const fs::path& p) { auto s = p.u8string(); return std::string(s.begin(), s.end()); }

static cv::Mat readGray(const fs::path& p)
{
    std::ifstream f(p, std::ios::binary);
    if (!f) return cv::Mat();
    std::vector<uchar> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return buf.empty() ? cv::Mat() : cv::imdecode(buf, cv::IMREAD_GRAYSCALE);
}

static bool parseName(const std::string& stem, Item& it)
{
    std::string s = stem;
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    static const std::regex reC("^center(\\d*)$");
    static const std::regex reY("(?:^|[^a-z])y\\s*([+-])\\s*(\\d+(?:\\.\\d+)?)");
    static const std::regex reT("(?:^|[^a-z])(?:theta|t)\\s*([+-])\\s*(\\d+(?:\\.\\d+)?)");
    static const std::regex reS("\\((\\d+)\\)");
    std::smatch m;
    if (std::regex_match(s, m, reC)) {
        it.isCenter = true;
        it.shot = m[1].str().empty() ? 0 : std::stoi(m[1].str());
        it.set = it.shot == 0 ? 1 : 2;
        return true;
    }
    bool hy = false, ht = false;
    if (std::regex_search(s, m, reY)) { it.trueY = std::stod(m[2].str()) * (m[1].str() == "-" ? -1 : 1); hy = true; }
    if (std::regex_search(s, m, reT)) { it.trueT = std::stod(m[2].str()) * (m[1].str() == "-" ? -1 : 1); ht = true; }
    if (std::regex_search(s, m, reS)) { it.set = 2; it.shot = std::stoi(m[1].str()); }
    return hy || ht;
}

static double groupMean(const Item& it, const std::vector<cf::Landmark>& lm, const std::string& g)
{
    double s = 0; int n = 0;
    for (size_t k = 0; k < lm.size(); ++k) if (lm[k].group == g && it.m.s[k].ok) { s += it.m.s[k].dx; n++; }
    return n ? s / n : NAN;
}

struct Stat { int n = 0, pT = 0, pY = 0, pB = 0; double rmsT = 0, maxT = 0, meanT = 0, rmsY = 0, maxY = 0, meanY = 0; };
template <class Pred>
static Stat calcStat(const std::vector<Item>& v, Pred keep)
{
    Stat s;
    for (const Item& it : v) {
        if (!it.e.valid || !keep(it)) continue;
        double eT = it.e.t - it.trueT, eY = (it.e.y - it.trueY) * 1000;
        s.n++; s.rmsT += eT * eT; s.meanT += eT; s.rmsY += eY * eY; s.meanY += eY;
        s.maxT = std::max(s.maxT, std::fabs(eT)); s.maxY = std::max(s.maxY, std::fabs(eY));
        bool pt = std::fabs(eT) <= TOL_T_DEG, py = std::fabs(eY) <= TOL_Y_MM * 1000;
        s.pT += pt; s.pY += py; s.pB += (pt && py);
    }
    if (s.n) { s.rmsT = std::sqrt(s.rmsT / s.n); s.meanT /= s.n; s.rmsY = std::sqrt(s.rmsY / s.n); s.meanY /= s.n; }
    return s;
}
static std::string fmtStat(const char* label, const Stat& s)
{
    std::ostringstream o;
    o << "  " << std::left << std::setw(28) << label << std::right << std::setw(4) << s.n << "장 | Θ RMS "
      << std::fixed << std::setprecision(4) << s.rmsT << "° 최대 " << s.maxT << "° 평균 " << std::showpos << s.meanT
      << std::noshowpos << "° ±0.01 통과 " << s.pT << "/" << s.n << " | Y RMS " << std::setprecision(1) << s.rmsY
      << " µm 평균 " << std::showpos << s.meanY << std::noshowpos << " µm ±90 통과 " << s.pY << "/" << s.n
      << " | 둘 다 " << s.pB << "/" << s.n << "\n";
    return o.str();
}

static void writeEvalCsv(const fs::path& p, const std::vector<Item>& v, const std::vector<cf::Landmark>& lm)
{
    std::ofstream f(p, std::ios::binary);
    f << "\xEF\xBB\xBF" << "file,set,shot,train,trueY_mm,estY_mm,errY_um,trueT_deg,estT_deg,errT_deg,judge,"
                           "sdY_um,sdT_deg,plate_dx,wall_upper_dx,wall_lower_dx,coarseDx,ms\r\n";
    char b[640];
    for (const Item& it : v) {
        double eY = (it.e.y - it.trueY) * 1000, eT = it.e.t - it.trueT;
        bool py = std::fabs(eY) <= TOL_Y_MM * 1000, pt = std::fabs(eT) <= TOL_T_DEG;
        const char* j = !it.e.valid ? "" : (py && pt) ? "OK" : (!py && !pt) ? "Y.T NG" : (!py ? "Y NG" : "T NG");
        std::snprintf(b, sizeof(b), "%d,%d,%d,%.4f,%.6f,%.2f,%.4f,%.6f,%.6f,%s,%.2f,%.6f,%.4f,%.4f,%.4f,%.3f,%.0f\r\n",
                      it.set, it.shot, it.train ? 1 : 0, it.trueY, it.e.y, eY, it.trueT, it.e.t, eT, j,
                      it.e.sdY * 1000, it.e.sdT, groupMean(it, lm, "plate"), groupMean(it, lm, "wall_upper"),
                      groupMean(it, lm, "wall_lower"), it.m.coarseDx, it.m.ms);
        f << "\"" << it.name << "\"," << b;
    }
}

int main(int argc, char** argv)
{
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    std::cout << "OpenCV version: " << CV_VERSION << "\n";
    fs::path imgDir("img");
    bool centersOnly = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--centers-only") centersOnly = true; else imgDir = fs::path(a);
    }
    fs::create_directories(OUT_DIR);

    // ------------------------------------------------------------ 목록
    std::vector<Item> items;
    for (const auto& de : fs::directory_iterator(imgDir)) {
        if (!de.is_regular_file()) continue;
        std::string ext = u8(de.path().extension());
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return (char)std::tolower(c); });
        if (ext != ".bmp") continue;
        Item it; it.path = de.path(); it.name = u8(de.path().filename());
        if (parseName(u8(de.path().stem()), it)) items.push_back(it);
    }
    std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) {
        if (a.set != b.set) return a.set < b.set;
        if (a.isCenter != b.isCenter) return a.isCenter;
        if (a.trueY != b.trueY) return a.trueY < b.trueY;
        if (a.trueT != b.trueT) return a.trueT < b.trueT;
        return a.shot < b.shot; });
    std::cout << "영상 " << items.size() << "장\n";

    // ------------------------------------------------------------ 랜드마크 · 기준
    std::vector<cf::Landmark> lm = cf::defaultLandmarks();
    if (fs::exists(LM_FILE) && cf::loadLandmarks(LM_FILE, lm)) std::cout << "랜드마크 : " << LM_FILE << " 에서 " << lm.size() << "개\n";
    else { cf::saveLandmarks(LM_FILE, lm); std::cout << "랜드마크 : 기본값 " << lm.size() << "개 (" << LM_FILE << " 저장)\n"; }

    cv::Mat ref = readGray(imgDir / "center.bmp");
    cf::LandmarkTracker tr;
    if (ref.empty() || !tr.setReference(ref, lm)) { std::cerr << "기준 영상(img/center.bmp) 또는 랜드마크 설정 오류\n"; return 1; }

    // ------------------------------------------------------------ 1) center 4장 특징
    {
        cv::Mat vis; cv::cvtColor(ref, vis, cv::COLOR_GRAY2BGR);
        std::map<std::string, cv::Scalar> col = { {"plate", {40, 40, 230}}, {"station", {0, 160, 230}},
                                                  {"wall_upper", {40, 170, 40}}, {"wall_lower", {200, 110, 20}} };
        for (const cf::Landmark& l : lm) {
            cv::Scalar c = col.count(l.group) ? col[l.group] : cv::Scalar(128, 128, 128);
            cv::rectangle(vis, cv::Rect(l.cx - l.hw, l.cy - l.hh, 2 * l.hw, 2 * l.hh), c, l.use ? 4 : 2, cv::LINE_AA);
            cv::putText(vis, l.name + (l.use ? "" : " (off)"), cv::Point(l.cx - l.hw, l.cy - l.hh - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, c, 2, cv::LINE_AA);
        }
        cv::imwrite(std::string(OUT_DIR) + "/center_landmarks.png", vis);
    }
    std::vector<Item*> centers;
    for (Item& it : items) if (it.isCenter) centers.push_back(&it);
    std::cout << "\n[1] 기준(center) 영상 " << centers.size() << "장 특징 — center.bmp 대비\n";
    for (Item* c : centers) {
        cv::Mat g = readGray(c->path);
        c->m = tr.measure(g);
        c->dx.resize(lm.size());
        for (size_t k = 0; k < lm.size(); ++k) c->dx[k] = c->m.s[k].dx;
    }
    {
        std::ofstream f(fs::path(OUT_DIR) / "center_features.csv", std::ios::binary);
        f << "\xEF\xBB\xBF" << "file,landmark,group,use,dx_px,dy_px,peak\r\n";
        char b[256];
        for (Item* c : centers) for (size_t k = 0; k < lm.size(); ++k) {
            std::snprintf(b, sizeof(b), ",%s,%s,%d,%.4f,%.4f,%.3f\r\n", lm[k].name.c_str(), lm[k].group.c_str(),
                          lm[k].use ? 1 : 0, c->m.s[k].dx, c->m.s[k].dy, c->m.s[k].peak);
            f << "\"" << c->name << "\"" << b;
        }
    }
    std::cout << "  " << std::left << std::setw(12) << "영상" << std::right
              << "  plate dx   wall_upper dx  wall_lower dx   plate dy   밝기   초점(라플라시안)  잡음σ\n";
    std::ofstream cs(fs::path(OUT_DIR) / "center_summary.csv", std::ios::binary);
    cs << "\xEF\xBB\xBF" << "file,plate_dx,wall_upper_dx,wall_lower_dx,plate_dy,mean,sharp_laplacian_var,noise_sigma,sat_frac\r\n";
    for (Item* c : centers) {
        cv::Mat g = readGray(c->path);
        cf::Quality q = cf::quality(g, cv::Rect(0, 0, g.cols, g.rows));
        double pdy = 0; int n = 0;
        for (size_t k = 0; k < lm.size(); ++k) if (lm[k].group == "plate") { pdy += c->m.s[k].dy; n++; }
        std::cout << "  " << std::left << std::setw(12) << c->name << std::right << std::fixed << std::setprecision(3)
                  << std::setw(10) << groupMean(*c, lm, "plate") << std::setw(15) << groupMean(*c, lm, "wall_upper")
                  << std::setw(15) << groupMean(*c, lm, "wall_lower") << std::setw(11) << pdy / std::max(1, n)
                  << std::setprecision(1) << std::setw(8) << q.mean << std::setw(14) << q.sharp
                  << std::setprecision(2) << std::setw(12) << q.noise << "\n";
        char b[256];
        std::snprintf(b, sizeof(b), ",%.4f,%.4f,%.4f,%.4f,%.2f,%.2f,%.3f,%.5f\r\n", groupMean(*c, lm, "plate"), groupMean(*c, lm, "wall_upper"),
                      groupMean(*c, lm, "wall_lower"), pdy / std::max(1, n), q.mean, q.sharp, q.noise, q.satFrac);
        cs << "\"" << c->name << "\"" << b;
    }
    cs.close();
    if (centersOnly) return 0;

    // ------------------------------------------------------------ 2) 전체 측정
    std::cout << "\n[2] 전체 영상 랜드마크 측정\n";
    double sumMs = 0; int cnt = 0;
    for (size_t i = 0; i < items.size(); ++i) {
        Item& it = items[i];
        if (!it.isCenter) {
            cv::Mat g = readGray(it.path);
            if (g.empty()) { std::cout << "  읽기 실패: " << it.name << "\n"; continue; }
            it.m = tr.measure(g);
            it.dx.resize(lm.size());
            for (size_t k = 0; k < lm.size(); ++k) it.dx[k] = it.m.s[k].dx;
            sumMs += it.m.ms; cnt++;
        }
        if (i % 40 == 0 || i + 1 == items.size())
            std::cout << "  [" << std::setw(3) << i + 1 << "/" << items.size() << "] " << it.name << (it.m.ok ? "" : "  (일부 랜드마크 실패)") << "\n";
    }
    std::cout << "  평균 " << std::fixed << std::setprecision(0) << (cnt ? sumMs / cnt : 0) << " ms/장 (측정만)\n";

    // ------------------------------------------------------------ 3) 학습 (1차 세트 짝수 번째)
    std::vector<Item*> ySweep, tSweep;
    for (Item& it : items) if (it.set == 1 && !it.isCenter && it.m.ok) (it.trueT == 0 ? ySweep : tSweep).push_back(&it);
    std::sort(ySweep.begin(), ySweep.end(), [](Item* a, Item* b) { return a->trueY < b->trueY; });
    std::sort(tSweep.begin(), tSweep.end(), [](Item* a, Item* b) { return a->trueT < b->trueT; });
    std::vector<cf::Sample> train;
    for (auto* v : { &ySweep, &tSweep })
        for (size_t i = 0; i < v->size(); ++i) if (i % 2 == 0) {
            (*v)[i]->train = true;
            cf::Sample s; s.dx = (*v)[i]->dx; s.y = (*v)[i]->trueY; s.t = (*v)[i]->trueT; train.push_back(s);
        }
    std::vector<int> idx;
    for (size_t k = 0; k < lm.size(); ++k) if (lm[k].use) idx.push_back((int)k);
    cf::ParallaxModel md = cf::fitModel(train, idx, lm, "set1 (2026-10-04) even samples, ref img/center.bmp");
    if (!md.ok) { std::cerr << "모델 학습 실패\n"; return 1; }
    md.save(MODEL);
    md.save(std::string(OUT_DIR) + "/" + MODEL);

    // ------------------------------------------------------------ 4) 추정
    const Item* c1 = nullptr;
    for (const Item& it : items) if (it.isCenter && it.shot == 1) c1 = &it;
    for (Item& it : items) {
        if (it.dx.empty()) continue;
        const std::vector<double>* refDx = (it.set == 2 && !it.isCenter && c1 && !c1->dx.empty()) ? &c1->dx : nullptr;
        it.e = cf::estimate(md, it.dx, refDx);
    }

    std::ostringstream sum;
    sum << "기준(center) 영상 특징 + 깊이별 랜드마크 시차로 Y·Θ 분리\n"
        << "  관찰 : 카메라가 로봇과 함께 움직이고 QR 플레이트·벽은 고정. Y 이동은 깊이에 따라 다르게(시차), Θ 회전은 깊이와 무관하게 보임\n"
        << "  모델 : dx_k = a_k·Y + b_k·Θ + q_k·Θ² + c_k  (랜드마크 " << idx.size() << "개, 학습 " << md.nTrain << "장, GLS)\n\n";
    sum << "[랜드마크별 계수]\n";
    for (size_t k = 0; k < md.idx.size(); ++k) {
        char b[200];
        std::snprintf(b, sizeof(b), "  %-14s %-11s a %7.3f px/mm  b %7.3f px/deg  q %7.3f px/deg2  잔차 σ %.3f px\n",
                      md.names[k].c_str(), lm[md.idx[k]].group.c_str(), md.a[k], md.b[k], md.q[k], std::sqrt(md.cov.at<double>((int)k, (int)k)));
        sum << b;
    }
    sum << "\n[기준 영상 — center.bmp 대비 추정]\n";
    for (Item* c : centers) {
        char b[200];
        std::snprintf(b, sizeof(b), "  %-12s Y %+7.1f µm  Θ %+.4f°   (plate dx %+.3f px, wall_lower dx %+.3f px)\n", c->name.c_str(),
                      c->e.y * 1000, c->e.t, groupMean(*c, lm, "plate"), groupMean(*c, lm, "wall_lower"));
        sum << b;
    }
    auto isY = [](const Item& it) { return it.trueT == 0; };
    sum << "\n[1차 세트 검증 — 홀수 번째]\n";
    sum << fmtStat("Y 단독 (Y ±2 mm)", calcStat(items, [&](const Item& it) { return it.set == 1 && !it.isCenter && !it.train && isY(it); }));
    sum << fmtStat("Θ 단독 (Θ ±1°)", calcStat(items, [&](const Item& it) { return it.set == 1 && !it.isCenter && !it.train && !isY(it); }));
    sum << "\n[2차 세트 — 1차 세트 모델, center1 기준으로 다시 맞춤]\n";
    auto s2 = [](const Item& it) { return it.set == 2 && !it.isCenter; };
    sum << fmtStat("전체", calcStat(items, s2));
    sum << fmtStat("Θ ≤ 0.20°", calcStat(items, [&](const Item& it) { return s2(it) && it.trueT <= 0.2 + 1e-9; }));
    sum << fmtStat("Θ = 1.00°", calcStat(items, [&](const Item& it) { return s2(it) && std::fabs(it.trueT - 1) < 1e-9; }));
    sum << fmtStat("Θ = 2.00° (학습 범위 밖)", calcStat(items, [&](const Item& it) { return s2(it) && std::fabs(it.trueT - 2) < 1e-9; }));
    sum << "\n모델 저장 : " << MODEL << " (model_type PARALLAX_LANDMARK_v1), 랜드마크 : " << LM_FILE << "\n";
    std::cout << "\n" << sum.str();
    std::ofstream(fs::path(OUT_DIR) / "parallax_summary.txt", std::ios::binary) << "\xEF\xBB\xBF" << sum.str();

    std::vector<Item> v1, v2;
    for (const Item& it : items) (it.set == 1 ? v1 : v2).push_back(it);
    writeEvalCsv(fs::path(OUT_DIR) / "parallax_set1.csv", v1, lm);
    writeEvalCsv(fs::path(OUT_DIR) / "parallax_set2.csv", v2, lm);
    std::cout << "결과 : " << OUT_DIR << "/\n";
    return 0;
}
