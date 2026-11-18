// 단일축 trial 6개의 원점(명령 0) 역추정: trial별 직선 절편을 630점 변위 1260성분으로 외삽하고, 축별 평균 기울기 J로 Y0/T0 를 최소제곱으로 푼다.
// 원본: eda_trial_origins_20261022/build.py (변위는 joint_center_630 measurement.py, trial 은 eda_capture_inventory 의 파일 시각 순서)
#include <hye/dataset.hpp>
#include <hye/plot.hpp>
#include <hye/report.hpp>
#include <omp.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include "struct630.hpp"
#include "zip_times.hpp"

using namespace hye;

namespace {

constexpr int kF = 1260;

cv::Mat read_img(const Shot& s, int flag) {
    std::ifstream f(fs::u8path(s.path), std::ios::binary);
    std::vector<uchar> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    cv::Mat m = cv::imdecode(buf, flag);
    if (m.empty()) throw std::runtime_error("read failed: " + s.rel);
    return m;
}

std::map<std::string, std::string> source_members(const std::string& set) {
    std::ifstream in(fs::u8path(clean_dir() + "/" + set + "/_metadata/file_inventory.csv"));
    std::string line;
    std::getline(in, line);
    std::map<std::string, std::string> out;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        auto f = split_csv(line);
        out[f[0]] = f[1];
    }
    return out;
}
std::string source_archive(const std::string& set) {
    std::ifstream in(fs::u8path(clean_dir() + "/" + set + "/_metadata/provenance.json"));
    std::string s((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    auto k = s.find("\"source_archive\"");
    auto q1 = s.find('"', s.find(':', k) + 1), q2 = s.find('"', q1 + 1);
    return s.substr(q1 + 1, q2 - q1 - 1);
}
std::string stem(const std::string& rel) {
    auto nm = rel.substr(rel.rfind('/') + 1);
    return nm.substr(0, nm.rfind('.'));
}

}  // namespace

static int run();
int main() {
    try {
        return run();
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}

static int run() {
    begin_results();
    cv::setNumThreads(1);
    omp_set_num_threads(4);
    const std::string set = "clean_20261014";
    auto all = load_1014();
    const Shot center = center_1014(all);
    std::vector<Shot> shots;
    for (auto& s : all) if (s.condition != "center") shots.push_back(s);
    const int N = (int)shots.size();

    // 파일 시각 (원본 zip 중앙 디렉터리 NTFS 시각, #79 와 같은 방법) -> 시간순, 그룹 -> S1..S6
    std::vector<int64_t> tm(N);
    {
        auto member = source_members(set);
        auto inner = ziptimes::extract_member(fs::u8path(clean_dir() + "/" + set) / fs::u8path(source_archive(set)), ".zip");
        std::map<std::string, int64_t> t;
        for (auto& e : ziptimes::central_directory(inner)) t[e.name] = ziptimes::kst_seconds(e.ntfs);
        for (int i = 0; i < N; ++i) tm[i] = t.at(member.at(shots[i].rel));
    }
    auto group = [&](const Shot& s) {
        if (s.condition == "T") return std::string("T") + (s.t > 0 ? "+" : "-");
        return std::string(std::lround(std::abs(s.y) / 0.05) % 2 ? "Y2" : "Y1") + (s.y > 0 ? "+" : "-");
    };
    std::vector<int> ord(N);
    std::iota(ord.begin(), ord.end(), 0);
    std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return std::tie(tm[a], shots[a].rel) < std::tie(tm[b], shots[b].rel); });
    std::vector<std::string> trial_group;
    std::vector<int> trial_of(N);
    for (int i : ord) {
        auto g = group(shots[i]);
        auto it = std::find(trial_group.begin(), trial_group.end(), g);
        if (it == trial_group.end()) { trial_group.push_back(g); it = trial_group.end() - 1; }
        trial_of[i] = int(it - trial_group.begin());
    }
    const int NT = (int)trial_group.size();

    // 630점 변위
    std::vector<std::vector<double>> D(N, std::vector<double>(kF, kNaN));
    std::vector<int> nvalid(N);
    {
        const cv::Mat c = read_img(center, cv::IMREAD_GRAYSCALE);
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < N; ++i) {
            try {
                auto R = s630::extract(c, read_img(shots[i], cv::IMREAD_GRAYSCALE));
                for (int p = 0; p < 630; ++p) { D[i][2 * p] = R.D[p][0]; D[i][2 * p + 1] = R.D[p][1]; }
                nvalid[i] = R.nvalid();
            } catch (const std::exception& e) {
#pragma omp critical
                std::cerr << "630 failed " << shots[i].rel << ": " << e.what() << "\n";
            }
        }
    }
    std::cout << "measured " << N << "\n";

    // trial 별 직선 D = b + a * cmd (1260 성분 각각)
    struct TrialFit { std::vector<int> ix; bool isY; std::vector<double> cmd, a, b; };
    std::vector<TrialFit> fits(NT);
    for (int k = 0; k < NT; ++k) {
        auto& F = fits[k];
        for (int i : ord) if (trial_of[i] == k) F.ix.push_back(i);
        F.isY = trial_group[k][0] == 'Y';
        for (int i : F.ix) F.cmd.push_back(F.isY ? shots[i].y : shots[i].t);
        const double mx = mean(F.cmd);
        double sxx = 0;
        for (double c : F.cmd) sxx += (c - mx) * (c - mx);
        F.a.assign(kF, 0);
        F.b.assign(kF, 0);
        for (int j = 0; j < kF; ++j) {
            double my = 0, sxy = 0;
            for (size_t r = 0; r < F.ix.size(); ++r) my += D[F.ix[r]][j];
            my /= F.ix.size();
            for (size_t r = 0; r < F.ix.size(); ++r) sxy += (F.cmd[r] - mx) * (D[F.ix[r]][j] - my);
            F.a[j] = sxy / sxx;
            F.b[j] = my - F.a[j] * mx;
        }
    }
    // J: 축별 trial 기울기 평균 (Y 4개, T 2개를 같은 무게로)
    cv::Mat J = cv::Mat::zeros(kF, 2, CV_64F);
    int ny = 0, nt = 0;
    for (auto& F : fits) {
        for (int j = 0; j < kF; ++j) J.at<double>(j, F.isY ? 0 : 1) += F.a[j];
        (F.isY ? ny : nt)++;
    }
    J.col(0) /= ny;
    J.col(1) /= nt;
    cv::Mat w;
    cv::SVD::compute(J, w);
    const double cond = w.at<double>(0) / w.at<double>(1);
    const double cosJ = J.col(0).dot(J.col(1)) / (cv::norm(J.col(0)) * cv::norm(J.col(1)));
    // Y 1mm 반응 중 T 와 직교하는 부분 (좌표 RMS, px)
    cv::Mat orthY = J.col(0) - J.col(0).dot(J.col(1)) / J.col(1).dot(J.col(1)) * J.col(1);
    const double orth_rms = cv::norm(orthY) / std::sqrt((double)kF);
    auto solveYT = [&](const std::vector<double>& v) {
        cv::Mat b(kF, 1, CV_64F), x;
        for (int j = 0; j < kF; ++j) b.at<double>(j) = v[j];
        cv::solve(J, b, x, cv::DECOMP_SVD);
        return std::array<double, 2>{x.at<double>(0), x.at<double>(1)};
    };

    struct Out { double Y, T, sY, sT, odx, ody, sdx, sdy, unexpl; int nearest; };
    std::vector<Out> res(NT);
    struct ShotZ { int i, k; double cmd, Y, T, dx, dy; };
    std::vector<ShotZ> pz;
    for (int k = 0; k < NT; ++k) {
        auto& F = fits[k];
        std::vector<double> origin(kF, 0), shared(kF, 0);
        const int n = (int)F.ix.size();
        for (int r = 0; r < n; ++r) {
            std::vector<double> z(kF);
            double zx = 0, zy = 0;
            for (int j = 0; j < kF; ++j) {
                z[j] = D[F.ix[r]][j] - F.cmd[r] * F.a[j];
                origin[j] += z[j] / n;
                shared[j] += (D[F.ix[r]][j] - F.cmd[r] * J.at<double>(j, F.isY ? 0 : 1)) / n;
                (j % 2 ? zy : zx) += z[j];
            }
            auto v = solveYT(z);
            pz.push_back({F.ix[r], k, F.cmd[r], v[0], v[1], zx / 630, zy / 630});
        }
        auto o = solveYT(origin), s = solveYT(shared);
        Out& R = res[k];
        R.Y = o[0];
        R.T = o[1];
        R.sY = s[0];
        R.sT = s[1];
        R.odx = R.ody = R.sdx = R.sdy = R.unexpl = 0;
        for (int j = 0; j < kF; ++j) {
            const double pr = J.at<double>(j, 0) * o[0] + J.at<double>(j, 1) * o[1];
            R.unexpl += (origin[j] - pr) * (origin[j] - pr);
            (j % 2 ? R.ody : R.odx) += origin[j] / 630;
            (j % 2 ? R.sdy : R.sdx) += F.a[j] / 630;
        }
        R.unexpl = std::sqrt(R.unexpl / kF);
        R.nearest = 0;
        for (int r = 1; r < n; ++r) if (std::abs(F.cmd[r]) < std::abs(F.cmd[R.nearest])) R.nearest = r;
    }

    // ---- 표 ----
    const char* gname[] = {"Y1+", "Y 0.1 간격 +", "Y1-", "Y 0.1 간격 -", "Y2+", "Y 0.05 홀수 +", "Y2-", "Y 0.05 홀수 -", "T+", "Θ +", "T-", "Θ -"};
    auto gk = [&](const std::string& g) { for (int i = 0; i < 12; i += 2) if (g == gname[i]) return std::string(gname[i + 1]); return g; };
    Table tt{{"trial", "group", "axis", "count", "start", "end", "first_command", "last_command", "nearest_zero_command", "slope_dx_px", "slope_dy_px", "origin_dx_px", "origin_dy_px",
              "origin_Y_mm", "origin_T", "shared_slope_origin_Y_mm", "shared_slope_origin_T", "unexplained_field_rms_px"}};
    for (int k = 0; k < NT; ++k) {
        auto& F = fits[k];
        auto& R = res[k];
        tt.add({"S" + std::to_string(k + 1), gk(trial_group[k]), F.isY ? "Y" : "T", std::to_string(F.ix.size()), ziptimes::format(tm[F.ix.front()]), ziptimes::format(tm[F.ix.back()]),
                fmt(F.cmd.front(), 2), fmt(F.cmd.back(), 2), fmt(F.cmd[R.nearest], 2), fmt(R.sdx, 6), fmt(R.sdy, 6), fmt(R.odx, 6), fmt(R.ody, 6), fmt(R.Y, 6), fmt(R.T, 6), fmt(R.sY, 6),
                fmt(R.sT, 6), fmt(R.unexpl, 6)});
    }
    tt.csv(out("trial_origins.csv"));
    Table st{{"trial", "file", "command", "taken_at", "zero_Y", "zero_T", "zero_dx_px", "zero_dy_px", "valid_points"}};
    for (auto& z : pz)
        st.add({"S" + std::to_string(z.k + 1), shots[z.i].rel, fmt(z.cmd, 2), ziptimes::format(tm[z.i]), fmt(z.Y, 6), fmt(z.T, 6), fmt(z.dx, 6), fmt(z.dy, 6), std::to_string(nvalid[z.i])});
    st.csv(out("per_shot_zero_estimates.csv"));

    int kmin = 0, kmax = 0;
    for (int k = 1; k < NT; ++k) { if (res[k].Y < res[kmin].Y) kmin = k; if (res[k].Y > res[kmax].Y) kmax = k; }

    // ---- 그림 ----
    auto origins_chart = [&](bool cloud) {
        Chart c(cloud ? "Per-shot zero estimates and trial means" : "Trial origins (command zero) vs fixed center", "Y origin offset (mm)", "T origin offset");
        c.W = 900;
        c.H = 620;
        for (int k = 0; k < NT; ++k) {
            if (cloud) {
                std::vector<double> x, y;
                for (auto& z : pz) if (z.k == k) { x.push_back(z.Y); y.push_back(z.T); }
                c.scatter(x, y, color(k), "", 2);
            }
            c.scatter({res[k].Y}, {res[k].T}, color(k), "S" + std::to_string(k + 1), 7);
        }
        c.hline(0).vline(0);
        c.note(0, 0, "center");
        if (!cloud)
            for (int k = 0; k < NT; ++k) c.note(res[k].Y, res[k].T, "S" + std::to_string(k + 1) + " (" + fmts(res[k].Y, 3) + ", " + fmts(res[k].T, 3) + ")");
        return c.render();
    };
    save_png("01_trial원점_YT.png", origins_chart(false));
    {
        Chart b("Slope assumption: trial slope (dot) vs shared axis slope (x line end)", "Y origin offset (mm)", "T origin offset");
        b.W = 900;
        b.H = 620;
        for (int k = 0; k < NT; ++k) {
            b.line({res[k].Y, res[k].sY}, {res[k].T, res[k].sT}, color(k), "S" + std::to_string(k + 1), true);
        }
        b.hline(0).vline(0);
        save_png("02_사진별산포_기울기가정.png", hstack({origins_chart(true), b.render()}));
    }
    // 튀는 값: 사진별 역추정에서 Y·T 가 가장 벗어난 사진
    size_t wy = 0, wt = 0;
    for (size_t q = 1; q < pz.size(); ++q) {
        if (std::abs(pz[q].Y - res[pz[q].k].Y) > std::abs(pz[wy].Y - res[pz[wy].k].Y)) wy = q;
        if (std::abs(pz[q].T - res[pz[q].k].T) > std::abs(pz[wt].T - res[pz[wt].k].T)) wt = q;
    }
    {
        std::vector<cv::Mat> panels;
        for (int ax = 0; ax < 2; ++ax) {
            Chart c(std::string("Per-shot zero estimate minus trial origin - ") + (ax ? "T" : "Y"), "command", ax ? "zero T - trial T" : "zero Y - trial Y (mm)");
            c.W = 900;
            c.H = 380;
            for (int k = 0; k < NT; ++k) {
                std::vector<double> x, y;
                for (auto& z : pz) if (z.k == k) { x.push_back(z.cmd); y.push_back((ax ? z.T : z.Y) - (ax ? res[k].T : res[k].Y)); }
                c.scatter(x, y, color(k), "S" + std::to_string(k + 1), 3);
            }
            auto& z = pz[ax ? wt : wy];
            c.hline(0).note(z.cmd, (ax ? z.T - res[z.k].T : z.Y - res[z.k].Y), stem(shots[z.i].rel));
            panels.push_back(c.render());
        }
        save_png("03_사진별_원점편차.png", vstack(panels));
    }
    auto shot_fig = [&](const ShotZ& z, const std::string& file, const std::string& cap) {
        cv::Mat im = read_img(shots[z.i], cv::IMREAD_COLOR);
        const auto& F = fits[z.k];
        std::vector<cv::Mat> tiles;
        for (int q = 0; q < 3; ++q) {
            const int* b = s630::kBox[q];
            // 기준(초록): 이 trial 직선이 이 명령에서 예측한 위치(평균), 측정(빨강): 이 사진의 평균 이동
            double mx = 0, my = 0, px = 0, py = 0;
            int n = 0;
            std::vector<Mark> mk;
            for (int p = 0; p < 630; ++p)
                if (fixed630::kRegion[p] == q) {
                    mx += D[z.i][2 * p]; my += D[z.i][2 * p + 1];
                    px += F.b[2 * p] + F.a[2 * p] * z.cmd; py += F.b[2 * p + 1] + F.a[2 * p + 1] * z.cmd;
                    ++n;
                    if (n % 25 == 1)
                        mk.push_back({{fixed630::kRef[p][0] + D[z.i][2 * p], fixed630::kRef[p][1] + D[z.i][2 * p + 1]},
                                      {D[z.i][2 * p] - (F.b[2 * p] + F.a[2 * p] * z.cmd), D[z.i][2 * p + 1] - (F.b[2 * p + 1] + F.a[2 * p + 1] * z.cmd)}, "", kRed});
                }
            mx /= n; my /= n; px /= n; py /= n;
            cv::Rect2d pred(b[0] + px, b[1] + py, b[2] - b[0], b[3] - b[1]), now(b[0] + mx, b[1] + my, b[2] - b[0], b[3] - b[1]);
            cv::Rect roi = cv::Rect(pred | now) + cv::Size(40, 40) - cv::Point(20, 20);
            tiles.push_back(zoom(im, roi, 420, mk, 100, q == 0 ? cap : "", {{pred, kGreen, ""}, {now, kRed, "q" + std::to_string(q) + " res dx " + fmts(mx - px, 3) + " dy " + fmts(my - py, 3)}}));
        }
        save_png(file, hstack(tiles));
    };
    const std::string fy = "04_튀는값_Y_" + stem(shots[pz[wy].i].rel) + ".png";
    const std::string ft = "05_튀는값_T_" + stem(shots[pz[wt].i].rel) + ".png";
    shot_fig(pz[wy], fy, stem(shots[pz[wy].i].rel) + " zero Y " + fmts(pz[wy].Y, 3) + " mm (trial S" + std::to_string(pz[wy].k + 1) + " " + fmts(res[pz[wy].k].Y, 3) + ")");
    if (wt != wy) shot_fig(pz[wt], ft, stem(shots[pz[wt].i].rel) + " zero T " + fmts(pz[wt].T, 3) + " (trial S" + std::to_string(pz[wt].k + 1) + " " + fmts(res[pz[wt].k].T, 3) + ")");

    Table outl{{"구분", "사진", "trial", "명령", "사진별 Y0 (mm)", "trial Y0", "사진별 T0", "trial T0", "명령 제거 후 평균 dx / dy (px)", "유효 점"}};
    auto addz = [&](const std::string& lab, size_t q) {
        auto& z = pz[q];
        outl.add({lab, shots[z.i].rel, "S" + std::to_string(z.k + 1), fmts(z.cmd, 2), fmts(z.Y, 4), fmts(res[z.k].Y, 4), fmts(z.T, 4), fmts(res[z.k].T, 4), fmts(z.dx, 3) + " / " + fmts(z.dy, 3),
                  std::to_string(nvalid[z.i])});
    };
    {
        std::vector<double> dev;
        for (auto& z : pz) dev.push_back(z.Y - res[z.k].Y);
        for (size_t q : worst(dev, 5)) addz("Y0 편차 상위", q);
        dev.clear();
        for (auto& z : pz) dev.push_back(z.T - res[z.k].T);
        for (size_t q : worst(dev, 3)) addz("T0 편차 상위", q);
    }

    // ---- 원래 결과와 대조 ----
    // 원래 값: eda_trial_origins_20261022/trial_origins.csv, audit.json, per_shot_zero_estimates.csv
    Parity par;
    const double oY[6] = {-0.005138, 0.086185, -0.124878, -0.357366, -0.460608, -0.232172};
    const double oT[6] = {-0.005789, -0.014016, 0.011923, 0.061387, 0.076644, 0.036125};
    const double oU[6] = {0.045833, 0.053803, 0.049512, 0.068071, 0.072535, 0.076572};
    {
        std::string a = "Y1+, Y1-, T+, T-, Y2+, Y2-", b;
        for (int k = 0; k < NT; ++k) b += (k ? ", " : "") + trial_group[k];
        par.add_text("S1~S6 순서 (시간순 그룹)", a, b, a == b);
    }
    for (int k = 0; k < 6 && k < NT; ++k) par.add("S" + std::to_string(k + 1) + " 원점 Y (mm)", oY[k], res[k].Y, 0.001, 6);
    for (int k = 0; k < 6 && k < NT; ++k) par.add("S" + std::to_string(k + 1) + " 원점 T", oT[k], res[k].T, 0.0002, 6);
    for (int k = 0; k < 6 && k < NT; ++k) par.add("S" + std::to_string(k + 1) + " 설명 안 된 잔차 RMS (px)", oU[k], res[k].unexpl, 0.0005, 6);
    par.add("S3 기울기 평균 dx (px/deg)", 80.298987, res[2].sdx, 0.001, 6);
    par.add("S1 기울기 평균 dx (px/mm)", 14.359770, res[0].sdx, 0.001, 6);
    par.add("S5 공통 기울기 원점 Y (mm)", -0.452674, res[4].sY, 0.001, 6);
    par.add("J 조건수", 2450.525276, cond, 1.0, 3);
    par.add("J 두 열 cos", 0.9999972174, cosJ, 1e-8, 10);
    par.add("사진별 역추정 장수", 280, (double)pz.size(), 0, 0);
    {
        size_t q = 0;
        for (size_t r = 1; r < pz.size(); ++r) if (std::abs(pz[r].Y) > std::abs(pz[q].Y)) q = r;
        par.add_text("|사진별 Y0| 최대 사진", "Theta/T-0.54.bmp", shots[pz[q].i].rel, shots[pz[q].i].rel == "Theta/T-0.54.bmp");
        par.add("|사진별 Y0| 최대 값 (mm)", -0.857597, pz[q].Y, 0.002, 6);
    }

    // ---- 결과.md ----
    std::ostringstream R;
    R << "# 결과: trial 원점 - 절편 외삽 역산\n\n"
      << "입력: `data/clean/clean_20261014`의 단일축 280장(center 제외). 변위는 center.bmp 기준 630점 경계 맞춤(현재 - 기준, px)의 1260성분이다.\n"
      << "trial: 파일명 그룹(Y 0.1 간격, Y 0.05 홀수, T 부호)을 원본 zip의 파일 시각 순으로 S1~S6으로 번호 붙였다(#79와 같은 시각 출처, clean 메타데이터에는 시각이 없다).\n\n"
      << "방법: trial마다 성분별 직선 D = b + a x 명령을 맞춘다. 사진별로 D - a x 명령을 계산해 명령 0으로 외삽하고 trial 안에서 평균한 것이 trial 원점 변위다. "
      << "J = [Y trial 4개 기울기 평균, T trial 2개 기울기 평균] (1260 x 2)에 최소제곱으로 맞춰 Y0, T0를 얻는다. '공통 기울기'는 a 대신 J의 해당 열을 쓴 경우다.\n\n"
      << "## trial 원점\n\n" << tt.md()
      << "\n## J 진단\n\n| 지표 | 값 |\n|---|---|\n| 조건수 (특이값 비) | " << fmt(cond, 3) << " |\n| Y 열과 T 열의 cos | " << fmt(cosJ, 10) << " |\n"
      << "| Y 1mm 반응 RMS (px) | " << fmt(cv::norm(J.col(0)) / std::sqrt((double)kF), 4) << " |\n| T 1 반응 RMS (px) | " << fmt(cv::norm(J.col(1)) / std::sqrt((double)kF), 4) << " |\n"
      << "| Y 1mm 반응 중 T와 직교하는 부분 RMS (px) | " << fmt(orth_rms, 5) << " |\n"
      << "\n## 원래 결과와 대조 (" << par.ok << "/" << par.total << " 일치)\n\n" << par.t.md()
      << "\n## 옮기지 않은 부분\n\n- 원래 report.html과 origin_fields.npz(원점 변위장 저장). 원점 변위장은 이 프로그램 안에서 다시 계산해 표로만 냈다.\n"
      << "- 630점 측정 자산(모서리 위치·모양)은 `실험/src/fixed630.hpp` 상수로 옮겼다(`실험/tools/gen_fixed630.py`).\n"
      << "\n## 파일\n\n- `trial_origins.csv`: trial 6개 원점\n- `per_shot_zero_estimates.csv`: 280장 사진별 명령 0 외삽값\n- `이미지/01~05`\n";
    write_text(out("결과.md"), R.str());

    // ---- 보고서.md ----
    double umin = 1e9, umax = 0;
    for (auto& r : res) { umin = std::min(umin, r.unexpl); umax = std::max(umax, r.unexpl); }
    auto& zy = pz[wy];
    std::ostringstream P;
    P << "# 보고서: trial 원점 - 절편 외삽 역산 (10-22 17:50)\n\n"
      << "## 문제\n\n단일축 trial마다 명령 0일 때의 원점이 고정 center에서 얼마나 어긋나는지 Y/T로 환산해야 했다.\n\n"
      << "## 해결 방법\n\ntrial별 직선 절편을 630점 변위 1260성분에서 명령 0으로 외삽하고, 그 변위를 축별 평균 기울기 J에 최소제곱으로 맞춰 Y0/T0를 구했다.\n\n"
      << "## 결과\n\ntrial 원점 Y는 S" << kmin + 1 << " " << fmt(res[kmin].Y, 6) << " mm부터 S" << kmax + 1 << " " << fmt(res[kmax].Y, 6) << " mm까지 나왔고, T는 "
      << fmt(std::min({res[0].T, res[1].T, res[2].T, res[3].T, res[4].T, res[5].T}), 4) << "~" << fmt(std::max({res[0].T, res[1].T, res[2].T, res[3].T, res[4].T, res[5].T}), 4) << "이다.\n\n"
      << "| trial | 축 | 장수 | 원점 Y (mm) | 원점 T | 원점 평균 dx / dy (px) | 공통 기울기 Y / T | 설명 안 된 잔차 RMS (px) |\n|---|---|---|---|---|---|---|---|\n";
    for (int k = 0; k < NT; ++k)
        P << "| S" << k + 1 << " (" << gk(trial_group[k]) << ") | " << (fits[k].isY ? "Y" : "T") << " | " << fits[k].ix.size() << " | " << fmts(res[k].Y, 4) << " | " << fmts(res[k].T, 4) << " | "
          << fmts(res[k].odx, 3) << " / " << fmts(res[k].ody, 3) << " | " << fmts(res[k].sY, 4) << " / " << fmts(res[k].sT, 4) << " | " << fmt(res[k].unexpl, 4) << " |\n";
    P << "\n" << img_md("01_trial원점_YT.png", "trial 원점") << "\n\n" << img_md("02_사진별산포_기울기가정.png", "사진별 산포") << "\n\n"
      << "## 튀는 값\n\ntrial 평균에서 가장 벗어난 사진은 " << shots[zy.i].rel << "(S" << zy.k + 1 << ", 명령 " << fmts(zy.cmd, 2) << ")로, 명령을 빼고 남은 변위가 Y0 " << fmts(zy.Y, 3)
      << " mm(trial " << fmts(res[zy.k].Y, 3) << ")로 환산됐고, 남은 평균 이동은 dx " << fmts(zy.dx, 3) << " px뿐인데 J가 T와 거의 평행해 작은 모양 차이가 Y로 크게 옮겨졌다.\n\n"
      << outl.md() << "\n" << img_md(fy, "Y 튀는 값");
    if (wt != wy) P << "\n\n" << img_md(ft, "T 튀는 값");
    else P << "\n\nT0 편차가 가장 큰 사진도 같은 사진이다.";
    P << "\n\n초록 상자는 그 trial 직선이 이 명령에서 예측한 QR 평균 위치, 빨간 상자는 이 사진에서 잰 평균 위치, 빨간 화살표는 25점마다 한 점의 (측정 - 직선 예측)을 100배 한 것이다.\n\n"
      << img_md("03_사진별_원점편차.png", "사진별 원점 편차") << "\n\n"
      << "## 한줄평\n\nJ의 Y 열과 T 열 cos가 " << fmt(cosJ, 7) << "(조건수 " << fmt(cond, 0) << ")라 Y 1mm 중 T와 구분되는 변위가 " << fmt(orth_rms, 4)
      << " px RMS뿐이어서, trial 원점 변위 중 J로 설명 안 되는 " << fmt(umin, 3) << "~" << fmt(umax, 3) << " px의 작은 모양 차이와 기울기 가정이 Y0를 수백 µm 흔들어 S" << kmin + 1 << "와 S"
      << kmax + 1 << "가 " << fmt(res[kmax].Y - res[kmin].Y, 3) << " mm 벌어졌다.\n\n"
      << "## 원래 결과와 대조\n\n원래 값은 `eda_trial_origins_20261022/trial_origins.csv`, `audit.json`, `per_shot_zero_estimates.csv`에서 가져왔다. " << par.ok << "/" << par.total << " 일치.\n\n"
      << par.t.md();
    write_text(out("보고서.md"), P.str());
    std::cout << "parity " << par.ok << "/" << par.total << "\n";
    return 0;
}
