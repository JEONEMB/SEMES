// A13: 2.1.1 수식 정리의 물리 모형을 실측 감도에 맞춰 f, 주점, Z, T 물리 단위를 식별한다.
//
//  Irani 평면 운동식 (문서 식 (1)(2)(5)), 주점 (x0,y0) 기준 x = X-x0, y = Y-y0
//    Y 이동:  u_Y = p                      v_Y = 0
//    T 회전:  u_T = A + C x^2              v_T = C x y        (C = -Omega/f, A = -f(1+d/Z)Omega)
//  u 의 2차항과 v 의 xy 항이 **같은 계수 C** 를 쓴다는 것이 이 모형의 핵심 제약이고,
//  그 제약 덕분에 f 를 자료에서 식별할 수 있다:  A/C = f^2 (1+d/Z)
//
//  QR 630점(u,v)과 선 48블록(v 만)을 함께 쓴다. 선은 QR 에서 세로로 멀리 떨어져 y0 와 C 를 강하게 묶는다.
#include "cache.hpp"
#include "noise.hpp"
#include "hye/plot.hpp"
#include <functional>
#include <iostream>

using namespace a0;
namespace {

struct Obs {
    double X, Yi;   // 영상 좌표
    double ju, jv;  // T 에 대한 감도 (px per T 명령단위)
    double ju_y, jv_y;  // Y 에 대한 감도 (px per mm)
    bool has_u;     // 선 블록은 세로(v) 만 있다
    int kind;       // 0 = QR 점, 1 = 선 블록
};

struct Block { double x, y; bool upper; };

std::vector<Block> read_blocks(const fs::path& p) {
    cv::FileStorage f(p.u8string(), cv::FileStorage::READ);
    if (!f.isOpened()) throw std::runtime_error("line_protocol.json 없음");
    std::vector<Block> out;
    for (auto it = f["blocks"].begin(); it != f["blocks"].end(); ++it) {
        auto n = *it;
        auto b = n["read_box"];
        out.push_back({0.5 * (double(int(b[0])) + double(int(b[2]))),
                       0.5 * (double(int(b[1])) + double(int(b[3]))),
                       std::string(n["line"]) == "upper"});
    }
    return out;
}

// 제약 모형을 (A, C, x0, y0) 로 맞춘다. x0,y0 는 비선형이므로 격자 탐색 후 국소 세분.
struct Phys {
    double A = 0, C = 0, x0 = 1920, y0 = 1080, rss = 0;
    int n = 0;
};

Phys fit_phys(const std::vector<Obs>& o, double x0lo, double x0hi, double y0lo, double y0hi, int steps) {
    Phys best;
    best.rss = 1e300;
    for (int i = 0; i <= steps; ++i) {
        double x0 = x0lo + (x0hi - x0lo) * i / std::max(1, steps);
        for (int j = 0; j <= steps; ++j) {
            double y0 = y0lo + (y0hi - y0lo) * j / std::max(1, steps);
            // A, C 는 선형: 설계행렬 [1, x^2] 는 u 행, [0, x*y] 는 v 행
            double s11 = 0, s12 = 0, s22 = 0, b1 = 0, b2 = 0, n = 0;
            for (auto& k : o) {
                double x = k.X - x0, y = k.Yi - y0;
                if (k.has_u) {  // u = A + C x^2
                    double g = x * x;
                    s11 += 1; s12 += g; s22 += g * g;
                    b1 += k.ju; b2 += g * k.ju;
                    ++n;
                }
                double h = x * y;  // v = C x y  (A 항 없음)
                s22 += h * h;
                b2 += h * k.jv;
                ++n;
            }
            double det = s11 * s22 - s12 * s12;
            if (std::abs(det) < 1e-30) continue;
            double A = (b1 * s22 - b2 * s12) / det;
            double C = (s11 * b2 - s12 * b1) / det;
            double rss = 0;
            for (auto& k : o) {
                double x = k.X - x0, y = k.Yi - y0;
                if (k.has_u) { double r = k.ju - (A + C * x * x); rss += r * r; }
                double r2 = k.jv - C * x * y;
                rss += r2 * r2;
            }
            if (rss < best.rss) { best = Phys{A, C, x0, y0, rss, int(n)}; }
        }
    }
    return best;
}

}  // namespace

int main() {
    try {
        const fs::path repo = fs::u8path(REPO_DIR);
        const fs::path result = fs::u8path(EXPERIMENT_DIR);
        const fs::path results = result / "results";
        fs::create_directories(results);
        hye::begin_results();
        Cache c = load(cache_dir());
        auto blocks = read_blocks(repo / fs::u8path(u8"experiments/hye/20261028_1258_final_Y_T_joint/assets/line_protocol.json"));

        // ---- 1014 단일축으로 감도 ----
        std::vector<int> ids;
        for (auto& s : c.shots) if (s.measured && s.set == "clean_20261014") ids.push_back(s.idx);
        const int n = int(ids.size());
        cv::Mat X(n, 3, CV_64F), Yp(n, kPoints * 2, CV_64F), Yl(n, kLines, CV_64F);
        for (int i = 0; i < n; ++i) {
            X.at<double>(i, 0) = c.shots[ids[i]].y;
            X.at<double>(i, 1) = c.shots[ids[i]].t;
            X.at<double>(i, 2) = 1;
            for (int p = 0; p < kPoints; ++p) {
                auto d = c.dz(ids[i], p);
                Yp.at<double>(i, p * 2) = d.x;
                Yp.at<double>(i, p * 2 + 1) = d.y;
            }
            for (int b = 0; b < kLines; ++b) Yl.at<double>(i, b) = c.line(ids[i])[b * 2];
        }
        cv::Mat Bp, Bl;
        cv::solve(X, Yp, Bp, cv::DECOMP_SVD);
        cv::solve(X, Yl, Bl, cv::DECOMP_SVD);

        std::vector<Obs> obs, obs_qr;
        for (int p = 0; p < kPoints; ++p) {
            if (!c.points[p].selected) continue;
            Obs k{c.points[p].x, c.points[p].y,
                  Bp.at<double>(1, p * 2), Bp.at<double>(1, p * 2 + 1),
                  Bp.at<double>(0, p * 2), Bp.at<double>(0, p * 2 + 1), true, 0};
            obs.push_back(k);
            obs_qr.push_back(k);
        }
        for (int b = 0; b < kLines; ++b)
            obs.push_back({blocks[b].x, blocks[b].y, 0, Bl.at<double>(1, b), 0, Bl.at<double>(0, b), false, 1});
        std::cout << "관측 " << obs.size() << " (QR 점 " << obs_qr.size() << ", 선 " << kLines << ")\n";

        // ---- 제약 없는 2차 적합 (모형 검증용) ----
        Table uc;
        uc.head = {"응답", "상수", "x", "y", "x^2", "xy", "y^2", "잔차 RMS"};
        auto quad = [&](std::function<double(const Obs&)> get, const std::vector<Obs>& o, const std::string& name) {
            cv::Mat Aq(int(o.size()), 6, CV_64F), Bq(int(o.size()), 1, CV_64F), Cq;
            for (size_t i = 0; i < o.size(); ++i) {
                double x = (o[i].X - 1920) / 1000, y = (o[i].Yi - 1080) / 1000;
                double b[] = {1, x, y, x * x, x * y, y * y};
                for (int j = 0; j < 6; ++j) Aq.at<double>(int(i), j) = b[j];
                Bq.at<double>(int(i)) = get(o[i]);
            }
            cv::solve(Aq, Bq, Cq, cv::DECOMP_SVD);
            cv::Mat R = Bq - Aq * Cq;
            double rms_ = 0;
            for (int i = 0; i < R.rows; ++i) rms_ += R.at<double>(i) * R.at<double>(i);
            rms_ = std::sqrt(rms_ / R.rows);
            std::vector<std::string> row{name};
            for (int j = 0; j < 6; ++j) row.push_back(num(Cq.at<double>(j), 5));
            row.push_back(num(rms_, 5));
            uc.add(row);
            std::vector<double> cf;
            for (int j = 0; j < 6; ++j) cf.push_back(Cq.at<double>(j));
            return cf;
        };
        auto cu_t = quad([](const Obs& o) { return o.ju; }, obs_qr, "T -> u (QR 536점)");
        auto cv_t = quad([](const Obs& o) { return o.jv; }, obs_qr, "T -> v (QR 536점)");
        quad([](const Obs& o) { return o.ju_y; }, obs_qr, "Y -> u (QR 536점)");
        quad([](const Obs& o) { return o.jv_y; }, obs_qr, "Y -> v (QR 536점)");
        uc.csv(results / "unconstrained_quadratic.csv");

        // ---- 제약 모형 적합 ----
        Phys ph = fit_phys(obs, 1400, 2400, 600, 1600, 50);
        ph = fit_phys(obs, ph.x0 - 40, ph.x0 + 40, ph.y0 - 40, ph.y0 + 40, 80);
        Phys ph_qr = fit_phys(obs_qr, 1400, 2400, 300, 2000, 50);
        ph_qr = fit_phys(obs_qr, ph_qr.x0 - 40, ph_qr.x0 + 40, ph_qr.y0 - 40, ph_qr.y0 + 40, 80);

        // Y 감도 (균일 가로 이동)
        double pY = 0;
        {
            double s = 0;
            for (auto& k : obs_qr) s += k.ju_y;
            pY = s / obs_qr.size();
        }

        auto derive = [&](const Phys& p) {
            struct D { double f, Omega, deg, Z, Req; };
            double ratio = p.A / p.C;          // = f^2 (1+d/Z)
            double f = ratio > 0 ? std::sqrt(ratio) : std::numeric_limits<double>::quiet_NaN();
            double Omega = -p.C * f;           // rad per T 명령단위
            double Z = f / pY;                 // mm (u_Y = (f/Z) per mm)
            return D{f, Omega, Omega * 180 / CV_PI, Z, std::abs(kTtoY / Omega)};
        };
        auto d1 = derive(ph), d2 = derive(ph_qr);

        Table pt;
        pt.head = {"적합", "관측", "A (px/T)", "C (px/T/px^2)", "주점 x0", "주점 y0", "잔차 RMS"};
        pt.add({"QR 536점 + 선 48", std::to_string(ph.n), num(ph.A, 4), num(ph.C, 9), num(ph.x0, 1), num(ph.y0, 1),
                num(std::sqrt(ph.rss / ph.n), 5)});
        pt.add({"QR 536점만", std::to_string(ph_qr.n), num(ph_qr.A, 4), num(ph_qr.C, 9), num(ph_qr.x0, 1),
                num(ph_qr.y0, 1), num(std::sqrt(ph_qr.rss / ph_qr.n), 5)});
        pt.csv(results / "physical_fit.csv");

        Table dt;
        dt.head = {"유도값", "QR + 선", "QR만", "비고"};
        dt.add({"초점거리 f (px)", num(d1.f, 1), num(d2.f, 1), "d=0 가정. 실제로는 f*sqrt(1+d/Z)"});
        dt.add({"주점 x0 (px)", num(ph.x0, 1), num(ph_qr.x0, 1), "영상 중심 1920"});
        dt.add({"주점 y0 (px)", num(ph.y0, 1), num(ph_qr.y0, 1), "영상 중심 1080"});
        dt.add({"T 1 명령단위 = (rad)", num(d1.Omega, 6), num(d2.Omega, 6), ""});
        dt.add({"T 1 명령단위 = (deg)", num(d1.deg, 4), num(d2.deg, 4), "**T 물리 단위**"});
        dt.add({"카메라-판 거리 Z (mm)", num(d1.Z, 1), num(d2.Z, 1), "Z = f / (14.25 px/mm)"});
        dt.add({"등가 회전 반경 (mm)", num(d1.Req, 1), num(d2.Req, 1), "문서 식 (7) 은 322 mm"});
        dt.add({"Y 감도 p (px/mm)", num(pY, 4), num(pY, 4), "팀 기준 14.25"});
        dt.csv(results / "derived.csv");

        // ---- 모형 검증: 제약 모형 대 제약 없는 2차 ----
        Table vt;
        vt.head = {"검증", "값", "뜻"};
        vt.add({"u 의 x^2 계수 (제약 없음)", num(cu_t[3] / 1e6, 9), "1e-6 px/T/px^2 단위로 환산"});
        vt.add({"v 의 xy 계수 (제약 없음)", num(cv_t[4] / 1e6, 9), "모형은 둘이 같아야 한다 (= C)"});
        vt.add({"두 계수의 비", num(cv_t[4] != 0 ? cu_t[3] / cv_t[4] : 0, 4), "모형이 맞으면 1"});
        vt.add({"제약 모형의 C", num(ph.C, 9), "QR + 선"});
        vt.add({"u 의 x 1차 계수", num(cu_t[1], 5), "모형은 0"});
        vt.add({"u 의 y 1차 계수", num(cu_t[2], 5), "모형은 0"});
        vt.add({"v 의 상수 계수", num(cv_t[0], 5), "모형은 0"});
        vt.csv(results / "model_check.csv");

        // ---- 그림 ----
        cv::Mat center = hye::imread_u8((repo / "data/clean/clean_20261014/center.bmp").u8string(), cv::IMREAD_GRAYSCALE);
        {
            hye::Chart ch("A13-131 T response: v (vertical) vs image x, by row", "image x [px]", "dv/dT [px per unit]");
            for (int k = 0; k < 2; ++k) {
                std::vector<double> x, y;
                for (auto& o : obs) if ((o.kind == 1) == (k == 1)) { x.push_back(o.X); y.push_back(o.jv); }
                ch.scatter(x, y, hye::color(k), k ? "line blocks" : "QR points", 3);
            }
            // 모형 예측
            for (double yy : {710.0, 1070.0, 1796.0}) {
                std::vector<double> x, y;
                for (double xx = 300; xx <= 3500; xx += 50) { x.push_back(xx); y.push_back(ph.C * (xx - ph.x0) * (yy - ph.y0)); }
                ch.line(x, y, hye::kGray, "");
            }
            ch.save("131_T세로응답.png");
        }
        {
            hye::Chart ch("A13-132 T response: u (horizontal) vs image x", "image x [px]", "du/dT [px per unit]");
            std::vector<double> x, y;
            for (auto& o : obs_qr) { x.push_back(o.X); y.push_back(o.ju); }
            ch.scatter(x, y, hye::color(0), "QR points", 3);
            std::vector<double> mx, my;
            for (double xx = 1500; xx <= 2300; xx += 10) { mx.push_back(xx); my.push_back(ph.A + ph.C * (xx - ph.x0) * (xx - ph.x0)); }
            ch.line(mx, my, hye::kRed, "model A + C x^2");
            ch.save("132_T가로응답.png");
        }
        {
            hye::Chart ch("A13-133 Y response: u and v vs image x", "image x [px]", "d/dY [px per mm]");
            std::vector<double> x, yu, yv;
            for (auto& o : obs_qr) { x.push_back(o.X); yu.push_back(o.ju_y); yv.push_back(o.jv_y); }
            ch.scatter(x, yu, hye::color(0), "u (model: constant)", 3);
            ch.scatter(x, yv, hye::color(1), "v (model: zero)", 3);
            ch.save("133_Y응답.png");
        }
        {  // 주점 위치를 영상 위에
            std::vector<hye::Mark> m{{{ph.x0, ph.y0}, {0, 0}, "principal point (fit)", hye::kRed},
                                     {{1920, 1080}, {0, 0}, "image center", hye::color(2)}};
            hye::save_png("134_주점.png", hye::zoom(center, cv::Rect(0, 0, 3840, 2160), 1400, m, 0,
                                                    "A13-134 identified principal point"));
        }

        std::ostringstream o;
        o << "# A13 결과 — 물리 모형 자기 보정: f, 주점, Z, T 물리 단위\n\n"
          << "`실험/src/main.cpp` 가 다시 쓴다. 해석은 [보고서](보고서.md).\n\n"
          << "근거 문서: `docs/hye/문헌연구/01_주제_관련/원인분석/논문/2.1.1_원인의_수식_정리.docx` 의 식 (1)(2)(5).\n\n"
          << "모형은 u 의 x^2 계수와 v 의 xy 계수가 **같은 C** 를 쓴다고 말한다. 이 제약이 있어야 f 가 식별된다.\n\n"
          << "## 1. 모형이 맞는가 (제약 없는 2차 적합)\n\n" << uc.md() << "\n"
          << "x, y 는 영상 중심 기준으로 1000 px 로 나눈 값이다.\n\n" << vt.md() << "\n"
          << "## 2. 제약 모형 적합\n\n" << pt.md() << "\n"
          << "## 3. 유도한 물리량\n\n" << dt.md() << "\n"
          << "## 4. 그림\n\n"
          << hye::img_md("131_T세로응답.png", "T vertical response") << "\n\n"
          << hye::img_md("132_T가로응답.png", "T horizontal response") << "\n\n"
          << hye::img_md("133_Y응답.png", "Y response") << "\n\n"
          << hye::img_md("134_주점.png", "principal point") << "\n";
        write_text(result / fs::u8path(u8"결과.md"), o.str());
        std::cout << "끝\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "오류: " << e.what() << "\n";
        return 1;
    }
}
