// 같은 점 추적: center 고정 ID 점을 LK -> GN 으로 추적하고 재검출·합성 이동으로 검증
// (원본: codex_experiments_20261014/measurement/measure.py, refine.py, diagnostics.py, synthetic_affine.py)
#include <hye/dataset.hpp>
#include <hye/plot.hpp>
#include <hye/report.hpp>
#include <omp.h>
#include <iostream>
#include <map>
#include "qr205.hpp"

using namespace hye;
using namespace qr;

namespace {

const int SCREWS[2][2] = {{1492, 1063}, {2312, 1073}};

// measure.feature_set: QR 상자(경계 28px 침식) 600개·간격 4·품질 .005, 나사 반지름 21 원 60개·품질 .01
void feature_set(const cv::Mat& ref, std::vector<cv::Point2f>& pts, std::vector<int>& regions) {
    cv::Mat sm = blur8(ref, 0.65);
    for (int k = 0; k < 5; ++k) {
        cv::Mat mask = cv::Mat::zeros(ref.size(), CV_8U);
        int maxn;
        double qual;
        if (k < 3) {
            const int* b = BOXES[k];
            mask(cv::Range(b[1] + 28 - CROP[1], b[3] - 28 - CROP[1]), cv::Range(b[0] + 28 - CROP[0], b[2] - 28 - CROP[0])) = 255;
            maxn = 600; qual = 0.005;
        } else {
            cv::circle(mask, {SCREWS[k - 3][0] - CROP[0], SCREWS[k - 3][1] - CROP[1]}, 21, 255, -1);
            maxn = 60; qual = 0.01;
        }
        std::vector<cv::Point2f> p;
        cv::goodFeaturesToTrack(sm, p, maxn, qual, 4, mask, 5);
        cv::cornerSubPix(sm, p, cv::Size(2, 2), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 0.001));
        for (auto& q : p) {
            if (k < 3) {
                const int* b = BOXES[k];
                double gx = (double)q.x + CROP[0], gy = (double)q.y + CROP[1];
                if (!(b[0] + 27 <= gx && gx < b[2] - 27 && b[1] + 27 <= gy && gy < b[3] - 27)) continue;
            }
            pts.push_back(q);
            regions.push_back(k);
        }
    }
}

// measure.coarse(include_screws=True): QR 3개 + 나사 63x63
std::vector<cv::Point2f> coarse5(const cv::Mat& ref, const cv::Mat& cur) {
    std::vector<cv::Point2f> out;
    for (int k = 0; k < 5; ++k) {
        int x0, y0, x1, y1;
        if (k < 3) { x0 = BOXES[k][0]; y0 = BOXES[k][1]; x1 = BOXES[k][2]; y1 = BOXES[k][3]; }
        else { x0 = SCREWS[k - 3][0] - 31; y0 = SCREWS[k - 3][1] - 31; x1 = SCREWS[k - 3][0] + 32; y1 = SCREWS[k - 3][1] + 32; }
        x0 -= CROP[0]; x1 -= CROP[0]; y0 -= CROP[1]; y1 -= CROP[1];
        cv::Mat r;
        cv::matchTemplate(cur(cv::Range(y0 - 22, y1 + 22), cv::Range(x0 - 110, x1 + 110)), ref(cv::Range(y0, y1), cv::Range(x0, x1)), r, cv::TM_CCOEFF_NORMED);
        cv::Point pos;
        cv::minMaxLoc(r, nullptr, nullptr, nullptr, &pos);
        out.push_back(cv::Point2f((float)(pos.x - 110), (float)(pos.y - 22)));
    }
    return out;
}

// scipy map_coordinates(order 3, mode 'reflect') 로 만든 합성 사진. src(x, y) -> 원본 좌표
double reflect_coord(double in, int len) {
    double sz2 = 2.0 * len;
    if (in < 0) {
        if (in < -sz2) in = sz2 * (long long)(-in / sz2) + in;
        in = in < -len ? in + sz2 : -in - 1;
    } else if (in > len - 1) {
        in -= sz2 * (long long)(in / sz2);
        if (in >= len) in = sz2 - in - 1;
    }
    return in;
}
int reflect_idx(int i, int n) {
    while (i < 0 || i >= n) i = i < 0 ? -i - 1 : 2 * n - i - 1;
    return i;
}
template <class F>
cv::Mat synth(const cv::Mat& ref, F src) {
    cv::Mat c = spline_filter([&] { cv::Mat f; ref.convertTo(f, CV_64F); return f; }(), Bd::Reflect);
    cv::Mat out(ref.size(), CV_8U);
#pragma omp parallel for
    for (int y = 0; y < ref.rows; ++y)
        for (int x = 0; x < ref.cols; ++x) {
            cv::Point2d s = src(x, y);
            BW wy = bweights(reflect_coord(s.y, ref.rows)), wx = bweights(reflect_coord(s.x, ref.cols));
            double v = 0;
            for (int a = 0; a < 4; ++a) {
                const double* r = c.ptr<double>(reflect_idx(wy.i0 + a, c.rows));
                double t = 0;
                for (int b = 0; b < 4; ++b) t += wx.w[b] * r[reflect_idx(wx.i0 + b, c.cols)];
                v += wy.w[a] * t;
            }
            out.at<uchar>(y, x) = cv::saturate_cast<uchar>(v);
        }
    return out;
}

struct Err { double bdx, bdy, rmse, p95; int n; double vr; };
Err err_stats(const std::vector<cv::Point2d>& d, const std::vector<bool>& v, const std::vector<cv::Point2d>& truth, const std::vector<int>& reg) {
    Err e{0, 0, 0, 0, 0, 0};
    std::vector<double> nrm;
    int nqr = 0, nv = 0;
    for (size_t i = 0; i < d.size(); ++i) {
        if (reg[i] >= 3) continue;
        ++nqr;
        if (!v[i]) continue;
        ++nv;
        double ex = d[i].x - truth[i].x, ey = d[i].y - truth[i].y;
        e.bdx += ex; e.bdy += ey; e.rmse += ex * ex + ey * ey;
        nrm.push_back(std::hypot(ex, ey));
    }
    e.n = nv;
    e.vr = (double)nv / nqr;
    e.bdx /= nv; e.bdy /= nv;
    e.rmse = std::sqrt(e.rmse / (2.0 * nv));
    e.p95 = percentile(nrm, 0.95);
    return e;
}

double median(std::vector<double> v) { return percentile(std::move(v), 0.5); }

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
    omp_set_num_threads(4);
    auto shots = load_1014();
    // measure.py 순서: center, Theta 정렬, Y 정렬
    std::vector<size_t> order(shots.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    auto rank = [&](size_t i) { return shots[i].condition == "center" ? 0 : shots[i].condition == "T" ? 1 : 2; };
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return rank(a) != rank(b) ? rank(a) < rank(b) : shots[a].rel < shots[b].rel; });
    const int N = (int)order.size();
    cv::Mat ref = read_crop(shots[order[0]]);
    std::vector<cv::Point2f> pts;
    std::vector<int> reg;
    feature_set(ref, pts, reg);
    const int P = (int)pts.size();
    int nreg[5] = {};
    for (int r : reg) nreg[r]++;
    int nqr = nreg[0] + nreg[1] + nreg[2];
    std::cout << "points " << P << " QR " << nreg[0] << "/" << nreg[1] << "/" << nreg[2] << " screws " << nreg[3] + nreg[4] << "\n";

    struct Var { std::string name; int w; double s; cv::Mat r; };
    std::vector<Var> V{{"lk15", 15, 0, ref}, {"lk31", 31, 0, ref}, {"lk51", 51, 0, ref}, {"lk31_blur", 31, 0.8, blur8(ref, 0.8)}};
    Refiner gn17(ref, pts, 17, 0.8);

    // 사진마다: 4가지 LK, GN17(LK31 흐림 출발), 재검출 거리
    std::vector<std::vector<std::vector<double>>> fb(4, std::vector<std::vector<double>>(N)), le(4, std::vector<std::vector<double>>(N));
    std::vector<std::vector<int>> vqr(4, std::vector<int>(N));
    std::vector<std::vector<bool>> vall(4, std::vector<bool>(P, true));
    std::vector<std::vector<cv::Point2d>> Dgn(N), Dlk(N);
    std::vector<int> gnvalid(N);
    std::vector<double> maxinit(N);
    std::vector<std::vector<double>> dist(N, std::vector<double>(nqr));
    std::vector<std::vector<cv::Point2f>> found_all(N);
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N; ++i) {
        cv::Mat cur = read_crop(shots[order[i]]);
        auto init = coarse5(ref, cur);
        std::vector<cv::Point2f> ini(P);
        for (int k = 0; k < P; ++k) ini[k] = init[reg[k]];
        for (int v = 0; v < 4; ++v) {
            std::vector<cv::Point2d> d;
            std::vector<bool> ok;
            std::vector<double> f;
            std::vector<float> e;
            lk_track(V[v].r, V[v].s > 0 ? blur8(cur, V[v].s) : cur, pts, ini, V[v].w, d, ok, &f, &e);
            int c = 0;
            for (int k = 0; k < P; ++k)
                if (reg[k] < 3) {
                    c += ok[k];
                    fb[v][i].push_back(f[k]);
                    le[v][i].push_back(e[k]);
#pragma omp critical
                    if (!ok[k]) vall[v][k] = false;
                }
            vqr[v][i] = c;
            if (v == 3) Dlk[i] = d;
        }
        std::vector<double> last;
        std::vector<bool> gv;
        gn17.run(cur, Dlk[i], Dgn[i], last, gv);
        int c = 0;
        double mx = 0;
        for (int k = 0; k < P; ++k)
            if (reg[k] < 3) { c += gv[k]; mx = std::max(mx, std::hypot(Dgn[i][k].x - Dlk[i][k].x, Dgn[i][k].y - Dlk[i][k].y)); }
        gnvalid[i] = c;
        maxinit[i] = mx;
        // diagnostics: 현재 사진에서 점을 다시 검출해 추적 위치와 가장 가까운 점까지 거리
        cv::Mat sm = blur8(cur, 0.65);
        int j = 0;
        for (int k = 0; k < 3; ++k) {
            const int* b = BOXES[k];
            int dx = (int)init[k].x, dy = (int)init[k].y;
            cv::Mat mask = cv::Mat::zeros(cur.size(), CV_8U);
            mask(cv::Range(b[1] + 28 - CROP[1] + dy, b[3] - 28 - CROP[1] + dy), cv::Range(b[0] + 28 - CROP[0] + dx, b[2] - 28 - CROP[0] + dx)) = 255;
            std::vector<cv::Point2f> fd;
            cv::goodFeaturesToTrack(sm, fd, 600, 0.005, 4, mask, 5);
            cv::cornerSubPix(sm, fd, cv::Size(2, 2), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 0.001));
            found_all[i].insert(found_all[i].end(), fd.begin(), fd.end());
            for (int q = 0; q < P; ++q) {
                if (reg[q] != k) continue;
                double px = pts[q].x + Dgn[i][q].x, py = pts[q].y + Dgn[i][q].y, best = 1e300;
                for (auto& f : fd) best = std::min(best, std::hypot(f.x - px, f.y - py));
                dist[i][j++] = best;
            }
        }
    }
    std::cout << "tracked\n";

    // LK 요약
    Table lkt{{"방법", "QR 유효 비율", "사진별 최소 유효 비율", "항상 유효 점", "FB 중앙값 (px)", "FB 최대 (px)", "LK 밝기 오차 중앙값"}};
    std::map<std::string, std::array<double, 6>> LKS;
    for (int v = 0; v < 4; ++v) {
        double tot = 0, mn = 1;
        std::vector<double> f, e;
        for (int i = 0; i < N; ++i) {
            tot += vqr[v][i];
            mn = std::min(mn, (double)vqr[v][i] / nqr);
            f.insert(f.end(), fb[v][i].begin(), fb[v][i].end());
            e.insert(e.end(), le[v][i].begin(), le[v][i].end());
        }
        int always = 0;
        for (int k = 0; k < P; ++k) always += reg[k] < 3 && vall[v][k];
        LKS[V[v].name] = {tot / ((double)N * nqr), mn, (double)always, median(f), max_abs(f), median(e)};
        auto& s = LKS[V[v].name];
        lkt.add({V[v].name, fmt(s[0], 6), fmt(s[1], 6), std::to_string(always), fmt(s[3], 6), fmt(s[4], 4), fmt(s[5], 3)});
    }
    int gn_all = 0;
    for (int i = 0; i < N; ++i) gn_all += gnvalid[i];
    double gn_vf = (double)gn_all / ((double)N * nqr), gn_maxinit = max_abs(maxinit);

    // 재검출
    Table red{{"거리 기준", "평균 재검출 비율", "사진별 최소", "모든 사진에서 재검출된 점", "QR 점"}};
    std::map<int, std::array<double, 3>> RD;
    std::vector<double> frac1(N);
    for (int tol : {1, 2, 3}) {
        double tot = 0, mn = 1;
        int all = 0;
        for (int i = 0; i < N; ++i) {
            int c = 0;
            for (double d : dist[i]) c += d < tol;
            tot += c;
            mn = std::min(mn, (double)c / nqr);
            if (tol == 1) frac1[i] = (double)c / nqr;
        }
        for (int j = 0; j < nqr; ++j) {
            bool ok = true;
            for (int i = 0; i < N; ++i) ok &= dist[i][j] < tol;
            all += ok;
        }
        RD[tol] = {tot / ((double)N * nqr), mn, (double)all};
        red.add({std::to_string(tol) + " px", fmt(RD[tol][0], 5), fmt(mn, 5), std::to_string(all), std::to_string(nqr)});
    }

    // 전역 affine 잔차 (diagnostics)
    std::vector<int> qi;
    for (int k = 0; k < P; ++k) if (reg[k] < 3) qi.push_back(k);
    double mx = 0, my = 0;
    for (int k : qi) { mx += pts[k].x; my += pts[k].y; }
    mx /= qi.size(); my /= qi.size();
    cv::Mat Xa((int)qi.size(), 3, CV_64F), pinvX;
    for (int r = 0; r < Xa.rows; ++r) { Xa.at<double>(r, 0) = 1; Xa.at<double>(r, 1) = (pts[qi[r]].x - mx) / 100; Xa.at<double>(r, 2) = (pts[qi[r]].y - my) / 100; }
    cv::invert(Xa, pinvX, cv::DECOMP_SVD);
    std::vector<double> perpoint(qi.size(), 0), perimage(N, 0);
    for (int i = 0; i < N; ++i) {
        cv::Mat d((int)qi.size(), 2, CV_64F);
        for (int r = 0; r < d.rows; ++r) { d.at<double>(r, 0) = Dgn[i][qi[r]].x; d.at<double>(r, 1) = Dgn[i][qi[r]].y; }
        cv::Mat res = d - Xa * (pinvX * d);
        for (int r = 0; r < d.rows; ++r) {
            double s = res.at<double>(r, 0) * res.at<double>(r, 0) + res.at<double>(r, 1) * res.at<double>(r, 1);
            perpoint[r] += s;
            perimage[i] += s;
        }
    }
    for (auto& v : perpoint) v = std::sqrt(v / (2.0 * N));
    for (auto& v : perimage) v = std::sqrt(v / (2.0 * qi.size()));

    // 합성 이동: scipy 3차 평행이동, affine
    Refiner gn31(ref, pts, 31, 0.8);
    cv::Mat bref = blur8(ref, 0.8);
    auto evaluate = [&](const cv::Mat& cur, const std::vector<cv::Point2d>& truth, std::map<std::string, Err>& outm) {
        auto init = coarse5(ref, cur);
        std::vector<cv::Point2f> ini(P);
        for (int k = 0; k < P; ++k) ini[k] = init[reg[k]];
        std::vector<cv::Point2d> d, g17, g31;
        std::vector<bool> v, v17, v31;
        std::vector<double> last;
        lk_track(bref, blur8(cur, 0.8), pts, ini, 31, d, v);
        gn17.run(cur, d, g17, last, v17);
        gn31.run(cur, d, g31, last, v31);
        outm["lk31_blur"] = err_stats(d, v, truth, reg);
        outm["gn17"] = err_stats(g17, v17, truth, reg);
        outm["gn31"] = err_stats(g31, v31, truth, reg);
    };
    const double shifts[7][2] = {{0.1, 0.07}, {0.25, -0.3}, {0.5, 0.5}, {7.7, -1.2}, {28.35, 0.08}, {80.15, -0.45}, {-79.4, 0.3}};
    std::vector<std::map<std::string, Err>> SY(7);
    for (int c = 0; c < 7; ++c) {
        double dx = shifts[c][0], dy = shifts[c][1];
        cv::Mat cur = synth(ref, [&](int x, int y) { return cv::Point2d(x - dx, y - dy); });
        evaluate(cur, std::vector<cv::Point2d>(P, {dx, dy}), SY[c]);
    }
    struct AC { const char* kind; double dx, dy, ang, sc, sh; };
    const AC acases[6] = {{"rotation", 80.15, -.45, .03, 0, 0}, {"rotation", -79.4, .3, -.03, 0, 0}, {"rotation", .25, -.3, .10, 0, 0},
                          {"scale", 28.35, .08, 0, .001, 0}, {"shear", 7.7, -1.2, 0, 0, .001}, {"mixed", -10.2, .6, -.04, .0006, .0008}};
    std::vector<std::map<std::string, Err>> SA(6);
    for (int c = 0; c < 6; ++c) {
        auto& a = acases[c];
        double t = a.ang * CV_PI / 180;
        cv::Matx22d A(std::cos(t), -std::sin(t), std::sin(t), std::cos(t));
        A = A * (1 + a.sc);
        A(0, 1) += a.sh;
        cv::Vec2d cen(ref.cols / 2.0, ref.rows / 2.0);
        cv::Vec2d b = cen - A * cen + cv::Vec2d(a.dx, a.dy);
        cv::Matx22d inv = A.inv();
        cv::Mat cur = synth(ref, [&](int x, int y) { cv::Vec2d s = inv * (cv::Vec2d(x, y) - b); return cv::Point2d(s[0], s[1]); });
        std::vector<cv::Point2d> truth(P);
        for (int k = 0; k < P; ++k) {
            cv::Vec2d p(pts[k].x, pts[k].y), q = A * p + b - p;
            truth[k] = {q[0], q[1]};
        }
        evaluate(cur, truth, SA[c]);
    }
    std::cout << "synthetic done\n";

    Table syt{{"이동 (dx, dy px)", "LK31 흐림 RMSE (px)", "GN17 RMSE (px)", "GN31 RMSE (px)", "GN17 치우침 dx / dy (px)", "GN17 p95 (px)"}};
    std::vector<double> rl, r17, r31, sxv;
    for (int c = 0; c < 7; ++c) {
        auto& s = SY[c];
        syt.add({fmt(shifts[c][0], 2) + ", " + fmt(shifts[c][1], 2), fmt(s["lk31_blur"].rmse, 5), fmt(s["gn17"].rmse, 5), fmt(s["gn31"].rmse, 5),
                 fmts(s["gn17"].bdx, 5) + " / " + fmts(s["gn17"].bdy, 5), fmt(s["gn17"].p95, 5)});
        rl.push_back(s["lk31_blur"].rmse); r17.push_back(s["gn17"].rmse); r31.push_back(s["gn31"].rmse); sxv.push_back(c);
    }
    Table aft{{"종류", "dx, dy (px)", "회전 (deg)", "배율", "전단", "LK31 흐림 RMSE", "GN17 RMSE", "GN31 RMSE"}};
    for (int c = 0; c < 6; ++c) {
        auto& a = acases[c];
        aft.add({a.kind, fmt(a.dx, 2) + ", " + fmt(a.dy, 2), fmt(a.ang, 2), fmt(a.sc, 4), fmt(a.sh, 4), fmt(SA[c]["lk31_blur"].rmse, 5), fmt(SA[c]["gn17"].rmse, 5), fmt(SA[c]["gn31"].rmse, 5)});
    }

    // ---- 그림 ----
    auto name_of = [&](int i) { auto r = shots[order[i]].rel; r = r.substr(r.find('/') + 1); return r.substr(0, r.rfind('.')); };
    {
        cv::Mat c8;
        cv::cvtColor(ref, c8, cv::COLOR_GRAY2BGR);
        std::vector<Mark> mk;
        for (int k = 0; k < P; ++k) mk.push_back({cv::Point2d(pts[k].x, pts[k].y), {0, 0}, "", color(reg[k])});
        cv::Mat z = zoom(c8, cv::Rect(130, 140, 960, 260), 1400, mk, 0, "center: " + std::to_string(nqr) + " QR points + " + std::to_string(nreg[3] + nreg[4]) + " screw points");
        save_png("01_고정ID점_" + std::to_string(nqr) + "개.png", z);
    }
    {
        Chart c("Synthetic cubic shift: coordinate RMSE", "shift case (0.1,0.07) ... (-79.4,0.3)", "RMSE (px)");
        c.line(sxv, rl, color(0), "LK31 blur", true).line(sxv, r17, color(1), "GN17", true).line(sxv, r31, color(2), "GN31", true);
        c.save("02_합성이동_RMSE.png");
    }
    std::vector<double> idx(N), aval(N);
    for (int i = 0; i < N; ++i) { idx[i] = i; aval[i] = perimage[i]; }
    int worst_img = (int)(std::min_element(frac1.begin(), frac1.end()) - frac1.begin());
    {
        Chart c("Redetection within 1px of tracked GN17 position", "image (0 center, 1-200 T, 201-280 Y)", "fraction of 928 points");
        c.scatter(idx, frac1, color(0)).note(worst_img, frac1[worst_img], name_of(worst_img));
        c.save("03_재검출비율_사진별.png");
    }
    int worst_aff = (int)(std::max_element(perimage.begin(), perimage.end()) - perimage.begin());
    {
        Chart c("Per-image residual after global affine fit (GN17)", "image (0 center, 1-200 T, 201-280 Y)", "residual RMS (px)");
        c.scatter(idx, aval, color(1)).note(worst_aff, aval[worst_aff], name_of(worst_aff));
        c.save("04_사진별_affine잔차.png");
    }
    // 튀는 값 1: 재검출 비율 최저 사진, 가운데 QR. 초록 = 추적 위치(고정 ID), 빨강 = 다시 검출한 점
    std::string f5 = "05_튀는값_재검출최저_" + name_of(worst_img) + ".png";
    {
        cv::Mat cur = read_crop(shots[order[worst_img]]);
        cv::Mat c8;
        cv::cvtColor(cur, c8, cv::COLOR_GRAY2BGR);
        std::vector<Mark> mk;
        double ox = 0;
        int cnt = 0;
        for (int k = 0; k < P; ++k)
            if (reg[k] == 1) { mk.push_back({cv::Point2d(pts[k].x + Dgn[worst_img][k].x, pts[k].y + Dgn[worst_img][k].y), {0, 0}, "", kGreen}); ox += Dgn[worst_img][k].x; ++cnt; }
        for (auto& f : found_all[worst_img]) mk.push_back({cv::Point2d(f.x, f.y), {0, 0}, "", kRed});
        ox /= cnt;
        const int* b = BOXES[1];
        cv::Rect roi((int)(b[0] - CROP[0] + ox) + 20, b[1] - CROP[1] + 20, 80, 70);
        save_png(f5, zoom(c8, roi, 800, mk, 0, name_of(worst_img) + ": green tracked, red redetected, 1px " + fmt(frac1[worst_img] * 100, 1) + "%"));
    }
    // 튀는 값 2: affine 잔차가 가장 큰 점. 왼쪽 center, 오른쪽 잔차 최대 사진
    int wp = (int)(std::max_element(perpoint.begin(), perpoint.end()) - perpoint.begin());
    int kp = qi[wp];
    std::string f6 = "06_튀는값_affine잔차최대점.png";
    {
        cv::Mat c8, d8;
        cv::cvtColor(ref, c8, cv::COLOR_GRAY2BGR);
        cv::cvtColor(read_crop(shots[order[worst_aff]]), d8, cv::COLOR_GRAY2BGR);
        cv::Point2d p0(pts[kp].x, pts[kp].y), p1 = p0 + Dgn[worst_aff][kp];
        cv::Rect r0((int)p0.x - 20, (int)p0.y - 20, 40, 40), r1((int)p1.x - 20, (int)p1.y - 20, 40, 40);
        save_png(f6, hstack({zoom(c8, r0, 400, {{p0, {0, 0}, "", kGreen}}, 0, "center point " + std::to_string(kp)),
                             zoom(d8, r1, 400, {{p1, {0, 0}, "", kRed}}, 0, name_of(worst_aff) + " tracked")}));
    }

    // CSV
    {
        Table pq{{"point_index", "region", "x", "y", "affine_residual_rms_px", "redetection_fraction_1px"}};
        for (size_t r = 0; r < qi.size(); ++r) {
            int c1 = 0;
            for (int i = 0; i < N; ++i) c1 += dist[i][r] < 1;
            pq.add({std::to_string(qi[r]), std::to_string(reg[qi[r]]), fmt(pts[qi[r]].x + CROP[0], 4), fmt(pts[qi[r]].y + CROP[1], 4), fmt(perpoint[r], 6), fmt((double)c1 / N, 5)});
        }
        pq.csv(out("point_quality.csv"));
        Table im{{"file", "gn17_mean_dx", "gn17_mean_dy", "redetection_1px", "affine_residual_rms_px"}};
        for (int i = 0; i < N; ++i) {
            double sx = 0, sy = 0;
            for (int k : qi) { sx += Dgn[i][k].x - Dgn[0][k].x; sy += Dgn[i][k].y - Dgn[0][k].y; }
            im.add({shots[order[i]].rel, fmt(sx / qi.size(), 5), fmt(sy / qi.size(), 5), fmt(frac1[i], 5), fmt(perimage[i], 6)});
        }
        im.csv(out("image_summary.csv"));
    }

    // ---- 원래 결과와 대조 ----
    // 원래 값: measurement/features.npz, tracking_summary.json, gn17_metadata.json, corner_repeatability.json, affine_residual_summary.json,
    //          synthetic_cubic_validation.csv, synthetic_cubic_gn31.csv, synthetic_affine_validation.csv
    Parity par;
    par.add("QR 점 수", 928, nqr, 0, 0);
    par.add("QR 좌 점 수", 316, nreg[0], 0, 0);
    par.add("QR 중 점 수", 302, nreg[1], 0, 0);
    par.add("QR 우 점 수", 310, nreg[2], 0, 0);
    par.add("나사 점 수", 39, nreg[3] + nreg[4], 0, 0);
    const double lko[4][6] = {{0.999321235734446, 0.9978448275862069, 925, 0.00204718136228621, 28.30950355529785, 4.351110935211182},
                              {1.0, 1.0, 928, 0.0008676067227497697, 0.024784788489341736, 4.3715691566467285},
                              {1.0, 1.0, 928, 0.0006103515625, 0.010262803174555302, 4.44087028503418},
                              {1.0, 1.0, 928, 0.0005214845878072083, 0.01022794283926487, 2.6368041038513184}};
    for (int v = 0; v < 4; ++v) {
        auto& s = LKS[V[v].name];
        par.add(V[v].name + " QR 유효 비율", lko[v][0], s[0], 0.0005, 6);
        par.add(V[v].name + " 항상 유효 점", lko[v][2], s[2], v == 0 ? 2 : 0, 0);
        par.add(V[v].name + " FB 중앙값 (px)", lko[v][3], s[3], 0.0002, 6);
        if (v > 0) par.add(V[v].name + " FB 최대 (px)", lko[v][4], s[4], 0.005, 5);
        par.add(V[v].name + " 밝기 오차 중앙값", lko[v][5], s[5], 0.01, 4);
    }
    par.add("GN17 QR 유효 비율", 1.0, gn_vf, 0, 4);
    par.add("GN17 LK 출발점과 최대 차이 (px)", 0.41963112354278564, gn_maxinit, 0.005, 5);
    const double rdo[3][3] = {{0.5971399865014112, 0.5118534482758621, 70}, {0.6703928396122224, 0.5948275862068966, 74}, {0.8058235673088723, 0.7586206896551724, 245}};
    for (int t = 1; t <= 3; ++t) {
        par.add(std::to_string(t) + "px 평균 재검출 비율", rdo[t - 1][0], RD[t][0], 0.002, 5);
        par.add(std::to_string(t) + "px 사진별 최소", rdo[t - 1][1], RD[t][1], 0.005, 5);
        par.add(std::to_string(t) + "px 모든 사진 재검출 점", rdo[t - 1][2], RD[t][2], 3, 0);
    }
    par.add("점별 affine 잔차 중앙값 (px)", 0.0554064345210476, median(perpoint), 0.0005, 5);
    par.add("점별 affine 잔차 p95 (px)", 0.07837026513886254, percentile(perpoint, 0.95), 0.0005, 5);
    par.add("사진별 affine 잔차 중앙값 (px)", 0.042853497133634726, median(perimage), 0.0005, 5);
    par.add("사진별 affine 잔차 최대 (px)", 0.10063598499744941, max_abs(perimage), 0.0005, 5);
    const double sco[7][3] = {{0.006247186801311792, 0.0012798679146559147, 0.0007369765628885641}, {0.01216653148308077, 0.002571397447262905, 0.0015114700019494911},
                              {0.012654339502466874, 0.0035141148547978406, 0.0017634725136905392}, {0.011319692214139032, 0.0024615763219360937, 0.0013697794478622783},
                              {0.009416341673439188, 0.0021182905989805562, 0.0011802852528699315}, {0.011461718319820232, 0.0027624413444830246, 0.0015844523589046153},
                              {0.012107446816276925, 0.003021560953990939, 0.0016202209375844735}};
    for (int c = 0; c < 7; ++c) {
        std::string k = "합성 (" + fmt(shifts[c][0], 2) + ", " + fmt(shifts[c][1], 2) + ") ";
        par.add(k + "LK31 흐림 RMSE", sco[c][0], SY[c]["lk31_blur"].rmse, 0.0005, 5);
        par.add(k + "GN17 RMSE", sco[c][1], SY[c]["gn17"].rmse, 0.0001, 5);
        par.add(k + "GN31 RMSE", sco[c][2], SY[c]["gn31"].rmse, 0.0001, 5);
    }
    const double afo[6][3] = {{0.011356043924668123, 0.003082757727598857, 0.0024137812520883296}, {0.011836654580395772, 0.00322184219480375, 0.002369630508492845},
                              {0.012629011920827781, 0.005432760655354372, 0.006313250790560072}, {0.008995486334564649, 0.003040528607726353, 0.0032585810934897467},
                              {0.011396692551315568, 0.0030237946372206133, 0.0027407131410148895}, {0.011804338091709362, 0.004116240418804114, 0.004569026725169972}};
    for (int c = 0; c < 6; ++c) {
        std::string k = std::string("affine ") + acases[c].kind + " " + std::to_string(c) + " ";
        par.add(k + "LK31 흐림 RMSE", afo[c][0], SA[c]["lk31_blur"].rmse, 0.0005, 5);
        par.add(k + "GN17 RMSE", afo[c][1], SA[c]["gn17"].rmse, 0.0001, 5);
        par.add(k + "GN31 RMSE", afo[c][2], SA[c]["gn31"].rmse, 0.0001, 5);
    }

    double g17min = *std::min_element(r17.begin(), r17.end()), g17max = *std::max_element(r17.begin(), r17.end());
    double lkmin = *std::min_element(rl.begin(), rl.end()), lkmax = *std::max_element(rl.begin(), rl.end());
    // ---- 결과.md ----
    std::ostringstream R;
    R << "# 결과: 같은 점 추적 (고정 ID, LK -> GN)\n\n"
      << "입력: `data/clean/clean_20261014` 281장의 판 영역(x 1340~2470, y 800~1330). center에서 고른 점을 모든 사진에서 같은 ID로 추적했다.\n\n"
      << "- 점: 흐림 0.65 center에서 QR 상자(경계 28px 침식)마다 Shi-Tomasi(품질 .005, 간격 4) + cornerSubPix, QR 경계 27px 안쪽만. QR " << nqr << "개("
      << nreg[0] << "/" << nreg[1] << "/" << nreg[2] << "), 나사 " << nreg[3] + nreg[4] << "개.\n"
      << "- 추적: QR 전체 NCC(±110 x ±22)로 정수 이동 -> 단일 단계 LK(창 15/31/51, 흐림 0.8 선택) 왕복 -> GN17(흐림 0.8, 3차 B-spline, gain/offset, 12회).\n"
      << "- 재검출: 현재 사진에서 같은 규칙으로 점을 다시 뽑아 GN17 추적 위치에서 가장 가까운 점까지 거리.\n"
      << "- 합성: center를 scipy 3차 보간(reflect)으로 알려진 만큼 옮긴 사진 7장, affine 6장.\n\n"
      << "## LK 추적 요약 (281장, QR " << nqr << "점)\n\n" << lkt.md()
      << "\nGN17: QR 유효 비율 " << fmt(gn_vf, 4) << ", LK31 흐림 출발점과 최대 차이 " << fmt(gn_maxinit, 4) << " px.\n\n"
      << "## 재검출 (같은 점이 다시 검출되는가)\n\n" << red.md()
      << "\n## 전역 affine 잔차 (GN17)\n\n| 지표 | 값 (px) |\n|---|---|\n| 점별 잔차 RMS 중앙값 | " << fmt(median(perpoint), 5) << " |\n| 점별 p95 | "
      << fmt(percentile(perpoint, 0.95), 5) << " |\n| 사진별 중앙값 | " << fmt(median(perimage), 5) << " |\n| 사진별 최대 | " << fmt(max_abs(perimage), 5) << " |\n\n"
      << "## 합성 평행이동 (scipy 3차)\n\n" << syt.md() << "\n## 합성 affine\n\n" << aft.md()
      << "\n## 원래 결과와 대조 (" << par.ok << "/" << par.total << " 일치)\n\n" << par.t.md()
      << "\n## 옮기지 않은 부분\n\n- measure.py의 OpenCV warpAffine 합성(synthetic_validation.csv): OpenCV 3.4 warpAffine은 좌표를 1/32 px로 양자화해 Python cv2와 다르다. 원래 노트도 이 값을 scipy 합성과 섞지 않는다.\n"
      << "- refine.py GN31의 실사진 281장 측정(gn31.npz), extract_pair.py(PairExtractor API), spread.py(QR205 측정, 6후보 실험 폴더에서 옮김).\n\n"
      << "## 파일\n\n- `point_quality.csv`: QR 점별 affine 잔차, 1px 재검출 비율\n- `image_summary.csv`: 사진별 평균 이동, 재검출 비율, affine 잔차\n";
    write_text(out("결과.md"), R.str());

    // ---- 보고서.md ----
    std::ostringstream Pm;
    Pm << "# 보고서: 같은 점 추적, 고정 ID LK + GN (10-14 16:18)\n\n"
       << "## 문제\n\n사진마다 다시 검출한 점은 같은 점이 아니므로, center 기준 고정 ID 점의 소수점 이동을 정확히 재야 했다.\n\n"
       << "## 해결 방법\n\ncenter QR 안의 점 " << nqr << "개를 고정하고 NCC -> LK -> GN17(밝기 gain/offset 포함)로 추적한 뒤, 재검출과 합성 이동으로 검증했다.\n\n"
       << "## 결과\n\n다시 검출한 점이 추적 위치 1px 안에 있는 비율은 평균 " << fmt(RD[1][0] * 100, 2) << "%뿐이었고, 합성 평행이동에서 GN17 좌표 RMSE는 " << fmt(g17min, 5) << "~" << fmt(g17max, 5) << " px였다.\n\n"
       << "| 지표 | 값 |\n|---|---|\n"
       << "| QR 고정 ID 점 | " << nqr << " |\n"
       << "| GN17 유효 (281장) | " << fmt(gn_vf * 100, 2) << "% |\n"
       << "| 1px 안 재검출 평균 / 최저 사진 | " << fmt(RD[1][0] * 100, 2) << "% / " << fmt(RD[1][1] * 100, 2) << "% |\n"
       << "| 281장 모두 1px 안 재검출된 점 | " << (int)RD[1][2] << " / " << nqr << " |\n"
       << "| 합성 평행이동 GN17 RMSE (px) | " << fmt(g17min, 5) << " ~ " << fmt(g17max, 5) << " |\n"
       << "| 합성 평행이동 LK31 흐림 RMSE (px) | " << fmt(lkmin, 5) << " ~ " << fmt(lkmax, 5) << " |\n"
       << "| 사진별 전역 affine 잔차 중앙값 (px) | " << fmt(median(perimage), 4) << " |\n\n"
       << img_md("01_고정ID점_" + std::to_string(nqr) + "개.png", "고정 ID 점") << "\n\n" << img_md("02_합성이동_RMSE.png", "합성 RMSE") << "\n\n"
       << img_md("03_재검출비율_사진별.png", "재검출 비율") << "\n\n" << img_md("04_사진별_affine잔차.png", "affine 잔차") << "\n\n"
       << "## 튀는 값\n\n재검출이 가장 나쁜 사진은 " << name_of(worst_img) << "(1px 안 " << fmt(frac1[worst_img] * 100, 1) << "%)이고, affine 잔차가 가장 큰 점은 " << kp << "번(점별 RMS "
       << fmt(perpoint[wp], 4) << " px, 중앙값의 " << fmt(perpoint[wp] / median(perpoint), 1) << "배)이다.\n\n"
       << "| 구분 | 사진 / 점 | 값 | 기준 |\n|---|---|---|---|\n"
       << "| 재검출 최저 사진 | " << name_of(worst_img) << " | 1px 안 " << fmt(frac1[worst_img] * 100, 1) << "% | 평균 " << fmt(RD[1][0] * 100, 1) << "% |\n"
       << "| affine 잔차 최대 사진 | " << name_of(worst_aff) << " | " << fmt(perimage[worst_aff], 4) << " px | 중앙값 " << fmt(median(perimage), 4) << " px |\n"
       << "| affine 잔차 최대 점 | " << kp << " (QR " << reg[kp] << ", x " << fmt(pts[kp].x + CROP[0], 1) << ", y " << fmt(pts[kp].y + CROP[1], 1) << ") | " << fmt(perpoint[wp], 4)
       << " px | 중앙값 " << fmt(median(perpoint), 4) << " px |\n\n"
       << img_md(f5, "재검출 최저") << "\n\n" << img_md(f6, "affine 잔차 최대 점")
       << "\n\n위 그림의 초록 십자는 고정 ID로 추적한 위치, 빨간 십자는 그 사진에서 새로 검출한 점이다. 아래 그림은 같은 점의 center 위치(초록)와 잔차 최대 사진의 추적 위치(빨강)다.\n\n"
       << "## 한줄평\n\nShi-Tomasi 검출은 사진마다 모서리 강도 순위와 cornerSubPix 수렴점이 바뀌어 1px 안 재검출이 평균 " << fmt(RD[1][0] * 100, 1)
       << "%에 그쳤지만, 고정 템플릿 GN은 밝기 gain/offset까지 같이 풀어 합성 이동에서 " << fmt(g17max, 4) << " px 이하로 같은 점을 따라갔다.\n\n"
       << "## 원래 결과와 대조\n\n원래 값은 `codex_experiments_20261014/measurement/`의 `tracking_summary.json`, `gn17_metadata.json`, `corner_repeatability.json`, "
       << "`affine_residual_summary.json`, `synthetic_cubic_validation.csv`, `synthetic_cubic_gn31.csv`, `synthetic_affine_validation.csv`, `features.npz`에서 가져왔다. "
       << par.ok << "/" << par.total << " 일치.\n\n" << par.t.md();
    write_text(out("보고서.md"), Pm.str());
    std::cout << "parity " << par.ok << "/" << par.total << "\n";
    return 0;
}
