// GN17 추적 잔차에 밝기별 잡음의 역분산 가중을 넣으면 반복 분할 오분류가 줄어드는지 본다.
// 원본: codex_experiments_20261014/pixel_weighted_gn_20261015/experiment.py (prepare / estimate_noise / WeightedRefiner / run),
//       gls_comparison_20261014 의 분할과 zero_experiment.py 판정기, measurement/spread.py·refine.py·measure.py.
// 원래 cache/ 의 npy 는 쓰지 않고 사진에서 다시 만든다.
#include <hye/dataset.hpp>
#include <hye/plot.hpp>
#include <hye/report.hpp>
#include <omp.h>
#include <iostream>
#include "kfold.hpp"
#include "track.hpp"
#include "zero.hpp"

using namespace hye;

namespace {

// np.interp
double interp(double x, const std::vector<double>& xp, const std::vector<double>& fp) {
    if (x <= xp.front()) return fp.front();
    if (x >= xp.back()) return fp.back();
    size_t j = std::upper_bound(xp.begin(), xp.end(), x) - xp.begin();
    double t = (x - xp[j - 1]) / (xp[j] - xp[j - 1]);
    return fp[j - 1] + t * (fp[j] - fp[j - 1]);
}

struct Curve {
    std::vector<double> centers, sd;
    std::vector<long long> counts;
    std::vector<std::pair<int, int>> pairs;  // (a, b) 사진 번호
    std::vector<float> pair_err;
};

// 사진 한 장의 준비물 (원래 prepare 단계)
struct Prep {
    std::array<cv::Mat, 3> raw, coef;            // QR 조각 원본(uint8), 흐림 후 스플라인 계수
    std::array<std::vector<cv::Point2f>, 3> lk;  // LK15 초기 이동 (조각 좌표)
    std::array<std::vector<char>, 3> lkvalid;
    std::array<std::vector<float>, 3> gain, offset;  // 가중 없는 GN 의 gain / offset (float32)
    std::array<cv::Point, 3> coarse;
    std::vector<cv::Point2f> D;  // 기존 GN17 이동 (float32, center 안 뺌)
};

// 가중 GN: 템플릿 주변 23x23 원본 밝기와 흐림 커널 제곱으로 분산을 전파한다
struct Weighted {
    const trk::Refiner* old = nullptr;
    std::vector<std::array<double, 529>> tex;  // 점마다 23x23 원본 밝기 (3차 스플라인)
    std::vector<std::array<double, 289>> refvar;
    double k2[7];
    std::vector<double> centers, sd;

    Weighted(const trk::Refiner& r, const cv::Mat& template_u8) : old(&r) {
        cv::Mat t;
        template_u8.convertTo(t, CV_64F);
        cv::Mat c = trk::spl::filter_nearest(t);
        for (auto& p : r.pts) {
            std::array<double, 529> a;
            for (int y = 0; y < 23; ++y)
                for (int x = 0; x < 23; ++x) a[y * 23 + x] = trk::spl::eval(c, (double)p.y + (y - 11), (double)p.x + (x - 11), 12);
            tex.push_back(a);
        }
        cv::Mat g = cv::getGaussianKernel(7, 0.8, CV_64F);
        for (int i = 0; i < 7; ++i) k2[i] = g.at<double>(i) * g.at<double>(i);
    }
    // ndimage.convolve1d 두 축 (k2) 후 [3:-3, 3:-3]
    void variance(const double* in, double* out) const {
        double v1[17][23];
        for (int r = 0; r < 17; ++r)
            for (int c = 0; c < 23; ++c) {
                double s = 0;
                for (int j = 0; j < 7; ++j) s += k2[j] * in[(r + j) * 23 + c];
                v1[r][c] = s;
            }
        for (int r = 0; r < 17; ++r)
            for (int c = 0; c < 17; ++c) {
                double s = 0;
                for (int j = 0; j < 7; ++j) s += k2[j] * v1[r][c + j];
                out[r * 17 + c] = s;
            }
    }
    void configure(const Curve& cv_) {
        centers = cv_.centers;
        sd = cv_.sd;
        refvar.assign(tex.size(), {});
        for (size_t i = 0; i < tex.size(); ++i) {
            double tmp[529];
            for (int k = 0; k < 529; ++k) { double s = interp(tex[i][k], centers, sd); tmp[k] = s * s; }
            variance(tmp, refvar[i].data());
        }
    }
    struct Res {
        trk::Refiner::Out o;
        double wmin, wmax;
    };
    Res run(const cv::Mat& coef, const std::vector<cv::Point2f>& init, const std::vector<float>& gain, const std::vector<float>& offset) const {
        const size_t n = tex.size();
        std::vector<cv::Mat> pv(n);
        double wmin = 1e9, wmax = 0;
        for (size_t i = 0; i < n; ++i) {
            double a = std::max((double)gain[i], 0.25), b = offset[i];
            double tmp[529], cur[289];
            for (int k = 0; k < 529; ++k) { double s = interp(a * tex[i][k] + b, centers, sd); tmp[k] = s * s; }
            variance(tmp, cur);
            double w[289], sum = 0;
            for (int k = 0; k < 289; ++k) { w[k] = 1 / std::max(cur[k] + a * a * refvar[i][k], 1e-6); sum += w[k]; }
            double mean = sum / 289;
            for (int k = 0; k < 289; ++k) { w[k] /= mean; wmin = std::min(wmin, w[k]); wmax = std::max(wmax, w[k]); }
            const cv::Mat& A = old->design[i];
            cv::Mat WA = A.clone();
            for (int k = 0; k < 289; ++k)
                for (int j = 0; j < 4; ++j) WA.at<double>(k, j) *= w[k];
            cv::Mat normal = A.t() * WA, rhs = WA.t(), P;
            cv::solve(normal, rhs, P, cv::DECOMP_LU);
            pv[i] = P;
        }
        return {old->run_coef(coef, init, pv), wmin, wmax};
    }
};

// estimate_noise: 학습 Y 사진 중 정수 이동에 가까운 8쌍으로 밝기 구간별 잔차 분산
Curve estimate_noise(const std::vector<Prep>& prep, const std::vector<zx::Row>& m, const std::vector<int>& tr, const std::vector<int>& region) {
    std::vector<int> ids;
    for (int i : tr) if (m[i].axis == 'Y') ids.push_back(i);
    const int P = (int)region.size();
    std::vector<std::array<float, 2>> mean(ids.size());
    for (size_t a = 0; a < ids.size(); ++a) {  // float32 순차 합 / 205
        float sx = 0, sy = 0;
        for (int j = 0; j < P; ++j) { sx += prep[ids[a]].D[j].x; sy += prep[ids[a]].D[j].y; }
        mean[a] = {sx / (float)P, sy / (float)P};
    }
    struct C { float e; int a, b; };
    std::vector<C> cand;
    for (size_t a = 0; a < ids.size(); ++a)
        for (size_t b = a + 1; b < ids.size(); ++b) {
            float dx = mean[b][0] - mean[a][0], dy = mean[b][1] - mean[a][1];
            float e = std::abs(dx - std::nearbyint(dx)) + std::abs(dy - std::nearbyint(dy));
            cand.push_back({e, ids[a], ids[b]});
        }
    std::sort(cand.begin(), cand.end(), [](const C& x, const C& y) {
        if (x.e != y.e) return x.e < y.e;
        if (x.a != y.a) return x.a < y.a;
        return x.b < y.b;
    });
    Curve cu;
    std::vector<long long> count(16, 0);
    std::vector<double> sums(16, 0), sumsq(16, 0);
    for (int q = 0; q < 8; ++q) {
        int a = cand[q].a, b = cand[q].b;
        cu.pairs.push_back({a, b});
        cu.pair_err.push_back(cand[q].e);
        for (int k = 0; k < 3; ++k) {
            double lx = 0, ly = 0;
            int c = 0;
            for (int j = 0; j < P; ++j)
                if (region[j] == k) { lx += prep[b].D[j].x - prep[a].D[j].x; ly += prep[b].D[j].y - prep[a].D[j].y; ++c; }
            lx = lx / c - (prep[b].coarse[k].x - prep[a].coarse[k].x);
            ly = ly / c - (prep[b].coarse[k].y - prep[a].coarse[k].y);
            int dx = (int)std::nearbyint(lx), dy = (int)std::nearbyint(ly);
            cv::Mat aa, bb;
            prep[a].raw[k].convertTo(aa, CV_64F);
            prep[b].raw[k].convertTo(bb, CV_64F);
            int h = aa.rows, w = aa.cols;
            int x0 = std::max(3, 3 - dx), x1 = std::min(w - 3, w - 3 - dx), y0 = std::max(3, 3 - dy), y1 = std::min(h - 3, h - 3 - dy);
            cv::Mat A = aa(cv::Range(y0, y1), cv::Range(x0, x1)).clone(), B = bb(cv::Range(y0 + dy, y1 + dy), cv::Range(x0 + dx, x1 + dx)).clone();
            double ma = cv::mean(A)[0], mb = cv::mean(B)[0], sab = 0, saa = 0;
            for (int y = 0; y < A.rows; ++y)
                for (int x = 0; x < A.cols; ++x) {
                    double ac = A.at<double>(y, x) - ma;
                    sab += ac * (B.at<double>(y, x) - mb);
                    saa += ac * ac;
                }
            double gain = sab / saa, off = mb - gain * ma;
            cv::Mat dd = B - (gain * A + off);
            cv::Mat avg = (A + B) / 2, g;
            cv::GaussianBlur(avg, g, cv::Size(0, 0), 0.7);
            cv::Mat gy = trk::np_gradient(g, 0), gx = trk::np_gradient(g, 1);
            cv::Mat xx, yy, xy, xd, yd;
            cv::blur(cv::Mat(gx.mul(gx)), xx, cv::Size(9, 9));
            cv::blur(cv::Mat(gy.mul(gy)), yy, cv::Size(9, 9));
            cv::blur(cv::Mat(gx.mul(gy)), xy, cv::Size(9, 9));
            cv::blur(cv::Mat(gx.mul(dd)), xd, cv::Size(9, 9));
            cv::blur(cv::Mat(gy.mul(dd)), yd, cv::Size(9, 9));
            for (int y = 5; y < A.rows - 5; ++y)
                for (int x = 5; x < A.cols - 5; ++x) {
                    double XX = xx.at<double>(y, x), YY = yy.at<double>(y, x), XY = xy.at<double>(y, x), XD = xd.at<double>(y, x), YD = yd.at<double>(y, x);
                    double det = XX * YY - XY * XY + 1e-6, sx = (YY * XD - XY * YD) / det, sy = (XX * YD - XY * XD) / det;
                    double r = dd.at<double>(y, x) - gx.at<double>(y, x) * sx - gy.at<double>(y, x) * sy;
                    double br = avg.at<double>(y, x);
                    int bin = std::min((int)(std::min(std::max(br, 0.0), 255.0) / 16), 15);
                    count[bin]++;
                    sums[bin] += r;
                    sumsq[bin] += r * r;
                }
        }
    }
    std::vector<double> cx, vv;
    for (int i = 0; i < 16; ++i) {
        double var = std::max((sumsq[i] - sums[i] * sums[i] / std::max<long long>(count[i], 1)) / std::max<long long>(count[i] - 1, 1) / 2, 0.0);
        cu.centers.push_back(8 + 16.0 * i);
        if (count[i] >= 100) { cx.push_back(8 + 16.0 * i); vv.push_back(var); }
    }
    for (int i = 0; i < 16; ++i) cu.sd.push_back(std::sqrt(std::max(interp(cu.centers[i], cx, vv), 0.25)));
    cu.counts = count;
    return cu;
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
    omp_set_num_threads(4);
    auto shots = load_1014();
    // 원래 npz 순서: center, Theta 파일명 정렬, Y 파일명 정렬
    std::vector<size_t> order;
    for (size_t i = 0; i < shots.size(); ++i) if (shots[i].condition == "center") order.push_back(i);
    for (const char* c : {"T", "Y"}) {
        std::vector<size_t> v;
        for (size_t i = 0; i < shots.size(); ++i) if (shots[i].condition == c) v.push_back(i);
        std::sort(v.begin(), v.end(), [&](size_t a, size_t b) { return shots[a].rel < shots[b].rel; });
        order.insert(order.end(), v.begin(), v.end());
    }
    const int N = (int)order.size();
    std::vector<zx::Row> m(N);
    for (int k = 0; k < N; ++k) {
        auto& s = shots[order[k]];
        std::string f = s.rel.substr(s.rel.rfind('/') + 1);
        m[k].file = f;
        if (s.condition == "center") { m[k].axis = 'C'; m[k].value = 0; m[k].session = "C"; continue; }
        m[k].axis = f[0];
        m[k].value = std::stod(f.substr(1, f.size() - 5));
        std::string ss = m[k].axis == 'T' ? "T" : ((std::llabs((long long)std::nearbyint(m[k].value / .05)) % 2) ? "Y2" : "Y1");
        m[k].session = ss + (m[k].value > 0 ? "+" : "-");
    }

    // ---- 준비: GN17 spread14 기존 측정 + 조각·계수·LK 초기값 ----
    const cv::Mat ref = trk::crop(read_gray(shots[order[0]]));
    trk::Features f14 = trk::features_spread(ref, 14);
    trk::Spread sp(ref, f14);
    const int P = (int)f14.pts.size();
    std::vector<Prep> prep(N);
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N; ++i) {
        cv::Mat cur = trk::crop(read_gray(shots[order[i]]));
        Prep& p = prep[i];
        p.coarse = trk::coarse(ref, cur);
        p.D.assign(P, {0, 0});
        for (int k = 0; k < 3; ++k) {
            int dx = p.coarse[k].x, dy = p.coarse[k].y;
            auto& b = sp.box[k];
            p.raw[k] = cur(cv::Range(b[1] + dy, b[3] + dy), cv::Range(b[0] + dx, b[2] + dx)).clone();
            cv::Mat tb;
            cv::GaussianBlur(p.raw[k], tb, cv::Size(0, 0), 0.8);
            std::vector<cv::Point2f> zero(sp.local[k].size(), {0, 0});
            trk::Track lt = trk::lk_track(sp.tplblur[k], tb, sp.local[k], zero, 15);
            p.lk[k] = lt.d;
            p.lkvalid[k] = lt.valid;
            p.coef[k] = sp.model[k].coefficients(p.raw[k]);
            auto g = sp.model[k].run_coef(p.coef[k], lt.d, sp.model[k].pinv);
            for (size_t j = 0; j < sp.idx[k].size(); ++j) {
                p.D[sp.idx[k][j]] = g.d[j] + cv::Point2f((float)dx, (float)dy);
                p.gain[k].push_back((float)g.gain[j]);
                p.offset[k].push_back((float)g.offset[j]);
            }
        }
    }
    std::cout << "prepared " << N << "\n";
    // 공간 기저 [1, x, y, x^2, xy, y^2], x=(X-1906)/486, y=(Y-1070)/122
    auto g14 = f14.global();
    cv::Mat B(P, 6, CV_64F);
    for (int j = 0; j < P; ++j) {
        double x = (g14[j].x - 1906.0) / 486, y = (g14[j].y - 1070.0) / 122;
        double v[6] = {1, x, y, x * x, x * y, y * y};
        for (int c = 0; c < 6; ++c) B.at<double>(j, c) = v[c];
    }
    // project: center 를 빼고 유효점으로 P12. 거절 = 유효 95% 미만 또는 QR 하나라도 20점 미만
    auto project = [&](const std::vector<std::vector<cv::Point2f>>& D, const std::vector<std::vector<char>>& valid, std::vector<char>& good) {
        cv::Mat X = cv::Mat::zeros(N, 12, CV_64F);
        good.assign(N, 1);
        for (int i = 0; i < N; ++i) {
            int nv = 0, per[3] = {0, 0, 0};
            for (int j = 0; j < P; ++j) if (valid[i][j]) { ++nv; per[f14.region[j]]++; }
            good[i] = nv >= 0.95 * P && per[0] >= 20 && per[1] >= 20 && per[2] >= 20;
            if (!good[i]) continue;
            cv::Mat Bv, d;
            for (int j = 0; j < P; ++j)
                if (valid[i][j]) {
                    Bv.push_back(B.row(j));
                    cv::Mat r = (cv::Mat_<double>(1, 2) << (double)D[i][j].x - (double)D[0][j].x, (double)D[i][j].y - (double)D[0][j].y);
                    d.push_back(r);
                }
            cv::Mat co = zx::lstsq(Bv, d);  // 6 x 2
            co.reshape(1, 1).copyTo(X.row(i));
        }
        return X;
    };
    std::vector<std::vector<cv::Point2f>> Dbase(N);
    std::vector<std::vector<char>> allv(N, std::vector<char>(P, 1));
    for (int i = 0; i < N; ++i) Dbase[i] = prep[i].D;
    std::vector<char> good_base;
    cv::Mat Xbase = project(Dbase, allv, good_base);

    // 가중 GN (QR 마다)
    std::vector<Weighted> W;
    for (int k = 0; k < 3; ++k) {
        auto& b = sp.box[k];
        W.emplace_back(sp.model[k], ref(cv::Range(b[1], b[3]), cv::Range(b[0], b[2])));
    }

    // ---- 분할: gls_comparison splits.json 과 같은 묶음 (interleaved 제외) ----
    struct Proto { std::string name; std::vector<zx::Fold> folds; };
    std::vector<Proto> protos;
    auto ids = zx::noncenter(m);
    protos.push_back({"full280", {{"all", ids, ids}}});
    protos.push_back({"stratified5", zx::stratified(m, 5, 20261014)});
    protos.push_back({"blocked_within_session5", zx::blocked5(m)});
    for (int s = 0; s < 20; ++s) protos.push_back({"repeated5_seed" + std::to_string(s), zx::stratified(m, 5, 7000 + s)});
    protos.push_back({"session_out_missing_origin", zx::session_out(m)});

    struct PredR { zx::Pred p; bool rejected; };
    std::vector<std::array<std::vector<PredR>, 2>> preds(protos.size());
    struct Diag { double wmin = 1e9, wmax = 0, med = 0, mx = 0; int test_rej = 0, train_rej = 0; Curve cu; };
    std::vector<std::vector<Diag>> diags(protos.size());
    std::vector<std::vector<cv::Point2f>> Dfull;  // full280 가중 측정
    for (size_t pi = 0; pi < protos.size(); ++pi) {
        for (auto& f : protos[pi].folds) {
            Diag dg;
            dg.cu = estimate_noise(prep, m, f.tr, f14.region);
            for (auto& w : W) w.configure(dg.cu);
            std::vector<std::vector<cv::Point2f>> D(N, std::vector<cv::Point2f>(P));
            std::vector<std::vector<char>> valid(N, std::vector<char>(P, 0));
            std::vector<double> lo(N, 1e9), hi(N, 0);
#pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < N; ++i)
                for (int k = 0; k < 3; ++k) {
                    auto r = W[k].run(prep[i].coef[k], prep[i].lk[k], prep[i].gain[k], prep[i].offset[k]);
                    lo[i] = std::min(lo[i], r.wmin);
                    hi[i] = std::max(hi[i], r.wmax);
                    for (size_t j = 0; j < sp.idx[k].size(); ++j) {
                        int q = sp.idx[k][j];
                        D[i][q] = r.o.d[j] + cv::Point2f((float)prep[i].coarse[k].x, (float)prep[i].coarse[k].y);
                        valid[i][q] = r.o.valid[j] && prep[i].lkvalid[k][j];
                    }
                }
            for (int i = 0; i < N; ++i) { dg.wmin = std::min(dg.wmin, lo[i]); dg.wmax = std::max(dg.wmax, hi[i]); }
            std::vector<double> delta;
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < P; ++j) delta.push_back(std::hypot((double)D[i][j].x - prep[i].D[j].x, (double)D[i][j].y - prep[i].D[j].y));
            dg.med = percentile(delta, 0.5);
            dg.mx = *std::max_element(delta.begin(), delta.end());
            std::vector<char> good;
            cv::Mat X = project(D, valid, good);
            std::vector<int> fittr;
            for (int i : f.tr) (good[i] ? fittr.push_back(i) : (void)++dg.train_rej);
            zx::Model mb, mw;
            {
                cv::Mat R;
                int rank;
                mb = zx::fit_curves(m, Xbase, f.tr, &R, &rank);
                mb.W = zx::legacy_W(R, rank);
                mw = zx::fit_curves(m, X, fittr, &R, &rank);
                mw.W = zx::legacy_W(R, rank);
            }
            for (auto& p : zx::predict(m, Xbase, f.te, mb)) preds[pi][0].push_back({p, false});
            for (auto& p : zx::predict(m, X, f.te, mw)) {
                bool rej = !good[p.i];
                if (rej) { p.pred_axis = 'R'; p.pred = 0; p.branch = "REJECT"; ++dg.test_rej; }
                preds[pi][1].push_back({p, rej});
            }
            if (pi == 0) Dfull = D;
            diags[pi].push_back(dg);
        }
        if (pi % 5 == 0) std::cout << "protocol " << protos[pi].name << " done\n";
    }

    // ---- 집계 ----
    auto pidx = [&](const std::string& n) {
        for (size_t i = 0; i < protos.size(); ++i) if (protos[i].name == n) return (int)i;
        return -1;
    };
    auto score = [&](const std::vector<PredR>& v) {
        std::vector<zx::Pred> p;
        int rej = 0;
        for (auto& x : v) { p.push_back(x.p); rej += x.rejected; }
        auto s = zx::score(p);  // 거절은 pred_axis 'R' 이라 축 오류 = 실패
        return std::make_pair(s, rej);
    };
    struct Sum { zx::Score full, strat, blocked, sout; int cv_failed = 0, Yc = 0, Tc = 0, perfect = 0, rej = 0; double Ymax = 0, Tmax = 0; std::map<std::string, int> fails; };
    Sum S[2];
    int fixed = 0, newf = 0;
    for (int me = 0; me < 2; ++me) {
        S[me].full = score(preds[0][me]).first;
        S[me].strat = score(preds[pidx("stratified5")][me]).first;
        S[me].blocked = score(preds[pidx("blocked_within_session5")][me]).first;
        S[me].sout = score(preds[pidx("session_out_missing_origin")][me]).first;
        for (int sd = 0; sd < 20; ++sd) {
            int pi = pidx("repeated5_seed" + std::to_string(sd));
            auto sc = score(preds[pi][me]);
            S[me].cv_failed += sc.first.failed;
            S[me].Yc += sc.first.Yc;
            S[me].Tc += sc.first.Tc;
            S[me].perfect += sc.first.failed == 0;
            S[me].rej += sc.second;
            S[me].Ymax = std::max(S[me].Ymax, sc.first.Ymax);
            S[me].Tmax = std::max(S[me].Tmax, sc.first.Tmax);
            for (auto& x : preds[pi][me]) if (!x.p.pass()) S[me].fails[m[x.p.i].file]++;
            if (me == 1) {
                std::map<int, bool> bp;
                for (auto& x : preds[pi][0]) bp[x.p.i] = x.p.pass();
                for (auto& x : preds[pi][1]) { fixed += !bp[x.p.i] && x.p.pass(); newf += bp[x.p.i] && !x.p.pass(); }
            }
        }
    }
    const Diag& d0 = diags[0][0];

    // ---- 원래 결과 (pixel_weighted_gn_20261015/comparison_summary.csv, metrics.csv, fold_diagnostics.csv, folds/full280__all/noise_model.json) ----
    Parity par;
    const double orig_sd[16] = {1.2519255254256136, 1.2519255254256136, 1.2519255254256136, 2.433863099598901, 3.201912598402459, 4.844725403440139,
                                7.040293309210101,  6.805439924029978,  6.559407732032614,  5.92414983924053,  4.630984779147364, 4.743327953376244,
                                3.663050848682536,  2.799156382297573,  2.799156382297573,  2.799156382297573};
    double sd_max_diff = 0;
    for (int i = 0; i < 16; ++i) sd_max_diff = std::max(sd_max_diff, std::abs(d0.cu.sd[i] - orig_sd[i]));
    par.add("full280 잡음 곡선 16구간 최대 차이 (gray)", 0, sd_max_diff, 1e-3, 6);
    par.add("full280 잡음 구간 표본 수 (구간 4, 밝기 48~64)", 123858, (double)d0.cu.counts[3], 0, 0);
    par.add_text("full280 첫 잡음 쌍", "Y+0.40.bmp / Y-0.20.bmp", m[d0.cu.pairs[0].first].file + " / " + m[d0.cu.pairs[0].second].file,
                 m[d0.cu.pairs[0].first].file == "Y+0.40.bmp" && m[d0.cu.pairs[0].second].file == "Y-0.20.bmp");
    par.add("full280 첫 쌍 정수 이동 오차 (px)", 0.00856044515967369, d0.cu.pair_err[0], 1e-6, 6);
    par.add("full280 가중치 최소", 0.285121, d0.wmin, 1e-4, 6);
    par.add("full280 가중치 최대", 4.228531, d0.wmax, 1e-3, 6);
    par.add("full280 측정 변화 중앙값 (px)", 0.007347, d0.med, 1e-5, 6);
    par.add("full280 측정 변화 최대 (px)", 0.078199, d0.mx, 1e-4, 6);
    par.add("기존 full280 Y 최대 (mm)", 0.03624133635474813, S[0].full.Ymax, 1e-6, 6);
    par.add("가중 GN full280 Y 최대 (mm)", 0.008351659555412061, S[1].full.Ymax, 1e-5, 6);
    par.add("가중 GN full280 T 최대 (deg)", 0.0029143822870817994, S[1].full.Tmax, 1e-5, 6);
    par.add("가중 GN 층화 5분할 Y 최대 (mm)", 0.035988835497928195, S[1].strat.Ymax, 1e-5, 6);
    par.add("가중 GN 시행 내 블록 5분할 Y 최대 (mm)", 0.03611189662023151, S[1].blocked.Ymax, 1e-5, 6);
    par.add("기존 반복 5600건 실패", 13, S[0].cv_failed, 0, 0);
    par.add("가중 GN 반복 5600건 실패", 12, S[1].cv_failed, 0, 0);
    par.add("가중 GN 반복 Y 정답 / 1600", 1600, S[1].Yc, 0, 0);
    par.add("가중 GN 반복 T 정답 / 4000", 3988, S[1].Tc, 0, 0);
    par.add("가중 GN 시행 제외 실패", 222, S[1].sout.failed, 0, 0);

    // ---- 표 ----
    Table summ{{"방법", "full280 실패", "full280 Y 최대 (mm)", "full280 T 최대 (deg)", "층화 5분할 Y 최대", "블록 5분할 Y 최대", "반복 실패 / 5600", "Y 정답 / 1600",
                "T 정답 / 4000", "실패 0 회차 / 20", "반복 Y 최대", "거절", "시행 제외 실패 / 280"}};
    const char* mname[2] = {"baseline (기존 GN17)", "pixel_noise_gn (역분산 가중 GN17)"};
    for (int me = 0; me < 2; ++me)
        summ.add({mname[me], std::to_string(S[me].full.failed), fmt(S[me].full.Ymax, 6), fmt(S[me].full.Tmax, 6), fmt(S[me].strat.Ymax, 6), fmt(S[me].blocked.Ymax, 6),
                  std::to_string(S[me].cv_failed), std::to_string(S[me].Yc), std::to_string(S[me].Tc), std::to_string(S[me].perfect), fmt(S[me].Ymax, 4),
                  std::to_string(S[me].rej), std::to_string(S[me].sout.failed)});
    summ.csv(out("comparison_summary.csv"));
    Table noise{{"밝기 구간", "표본 수", "잡음 표준편차 (gray, full280)"}};
    for (int i = 0; i < 16; ++i) noise.add({std::to_string(16 * i) + "~" + std::to_string(16 * i + 15), std::to_string(d0.cu.counts[i]), fmt(d0.cu.sd[i], 3)});
    Table pairs{{"쌍", "사진 a", "사진 b", "정수 이동 오차 (px)"}};
    for (size_t q = 0; q < d0.cu.pairs.size(); ++q)
        pairs.add({std::to_string(q), m[d0.cu.pairs[q].first].file, m[d0.cu.pairs[q].second].file, fmt(d0.cu.pair_err[q], 5)});
    Table fold{{"평가", "분할", "가중치 최소", "가중치 최대", "측정 변화 중앙 (px)", "측정 변화 최대 (px)", "학습 거절", "시험 거절"}};
    for (size_t pi = 0; pi < protos.size(); ++pi)
        for (size_t k = 0; k < diags[pi].size(); ++k) {
            auto& d = diags[pi][k];
            fold.add({protos[pi].name, protos[pi].folds[k].name, fmt(d.wmin, 4), fmt(d.wmax, 4), fmt(d.med, 5), fmt(d.mx, 4), std::to_string(d.train_rej), std::to_string(d.test_rej)});
        }
    fold.csv(out("fold_diagnostics.csv"));
    Table freq{{"방법", "사진", "실패 회차 / 20"}};
    for (int me = 0; me < 2; ++me)
        for (auto& kv : S[me].fails) freq.add({me ? "pixel_noise_gn" : "baseline", kv.first, std::to_string(kv.second)});

    // ---- 그림 ----
    {
        Chart c("Noise model (full280 training pairs)", "brightness (gray)", "residual noise sd (gray)");
        std::vector<double> x(d0.cu.centers), y(d0.cu.sd), yo(orig_sd, orig_sd + 16);
        c.line(x, y, color(0), "C++", true).scatter(x, yo, kRed, "original", 4);
        c.save("01_밝기별_잡음곡선.png");
    }
    {
        // full280 Y 사진 오차: 기존 대 가중
        std::vector<double> ty, eb, ew;
        for (size_t k = 0; k < preds[0][0].size(); ++k) {
            auto& b = preds[0][0][k].p;
            if (b.axis != 'Y') continue;
            auto& w = preds[0][1][k].p;
            ty.push_back(b.value);
            eb.push_back((b.pred_axis == 'Y' ? b.pred : 0) - b.value);
            ew.push_back((w.pred_axis == 'Y' ? w.pred : 0) - w.value);
        }
        Chart c("full280 Y error: original GN17 vs weighted GN17", "true Y (mm)", "pred - true (mm)");
        c.scatter(ty, eb, kGray, "original GN17", 3).scatter(ty, ew, color(0), "weighted GN17", 3).hline(0);
        c.save("02_full280_Y오차_기존대가중.png");
    }
    {
        std::vector<std::string> cats{"baseline", "pixel_noise_gn"};
        bar_chart("03_반복5분할_실패.png", "Failures in 20 x 5-fold CV (5600)", cats, {(double)S[0].cv_failed, (double)S[1].cv_failed}, "failed", 1);
    }
    // 튀는 값: 가중 GN 이 반복에서 가장 자주 틀린 T 사진 / Y 는 full280 Y 오차 최대 사진
    std::string ft;
    int cft = 0;
    for (auto& kv : S[1].fails) if (kv.second > cft) { cft = kv.second; ft = kv.first; }
    int wy = -1;
    double ey = -1;
    for (auto& x : preds[0][1])
        if (x.p.axis == 'Y' && x.p.eY() > ey) { ey = x.p.eY(); wy = x.p.i; }
    zx::Pred wyb, wyw;
    for (auto& x : preds[0][0]) if (x.p.i == wy) wyb = x.p;
    for (auto& x : preds[0][1]) if (x.p.i == wy) wyw = x.p;
    // T 사진의 첫 실패 예측
    zx::Pred tb{}, tw{};
    int tseed = -1;
    for (int sd = 0; sd < 20 && tseed < 0; ++sd) {
        int pi = pidx("repeated5_seed" + std::to_string(sd));
        for (auto& x : preds[pi][1])
            if (m[x.p.i].file == ft && !x.p.pass()) {
                tw = x.p;
                tseed = sd;
                for (auto& y : preds[pi][0]) if (y.p.i == x.p.i) tb = y.p;
            }
    }
    auto photo = [&](int k, const std::vector<std::vector<cv::Point2f>>& Dw, const std::string& png, const std::string& cap) {
        cv::Mat im = read_color(shots[order[k]]);
        std::vector<cv::Mat> tiles;
        for (int q = 0; q < 3; ++q) {
            auto b = trk::BOXES[q];
            double mx = 0, my = 0;
            int cnt = 0;
            for (int j = 0; j < P; ++j)
                if (f14.region[j] == q) { mx += prep[k].D[j].x - prep[0].D[j].x; my += prep[k].D[j].y - prep[0].D[j].y; ++cnt; }
            mx /= cnt; my /= cnt;
            cv::Rect2d rb(b[0], b[1], b[2] - b[0], b[3] - b[1]), nb(b[0] + mx, b[1] + my, b[2] - b[0], b[3] - b[1]);
            std::vector<Mark> mk;
            for (int j = 0; j < P; ++j)
                if (f14.region[j] == q) {
                    // 가중 GN - 기존 GN 의 점별 차이 x 300 (파랑 화살표)
                    double ddx = Dw.empty() ? 0 : (double)Dw[k][j].x - prep[k].D[j].x, ddy = Dw.empty() ? 0 : (double)Dw[k][j].y - prep[k].D[j].y;
                    mk.push_back({{g14[j].x + mx, g14[j].y + my}, {ddx, ddy}, "", cv::Scalar(180, 119, 31)});
                }
            cv::Rect roi = cv::Rect(rb | nb) + cv::Size(40, 40) - cv::Point(20, 20);
            tiles.push_back(zoom(im, roi, 420, mk, 300, q == 0 ? cap : "", {{rb, kGreen, ""}, {nb, kRed, "q" + std::to_string(q) + " dx " + fmt(mx, 2) + " dy " + fmt(my, 3)}}));
        }
        save_png(png, hstack(tiles));
    };
    std::string ny = m[wy].file.substr(0, m[wy].file.size() - 4), nt = ft.substr(0, ft.size() - 4);
    photo(wy, Dfull, "04_튀는값_Y_" + ny + ".png", ny + " full280 weighted Y err " + fmts((wyw.pred_axis == 'Y' ? wyw.pred : 0) - wyw.value, 4));
    int ti = 0;
    for (int i = 0; i < N; ++i) if (m[i].file == ft) ti = i;
    photo(ti, Dfull, "05_튀는값_T_" + nt + ".png", nt + " seed " + std::to_string(7000 + tseed) + " -> " + std::string(1, tw.pred_axis) + " " + fmts(tw.pred, 3));
    Table outl{{"구분", "사진", "정답 Y (mm)", "정답 T (deg)", "기존 GN 추정 (축·값·분기·비용)", "가중 GN 추정 (축·값·분기·비용)", "가중 GN 반복 실패 / 20", "기존 반복 실패 / 20"}};
    auto row = [&](const std::string& lab, const zx::Pred& b, const zx::Pred& w) {
        auto f = m[b.i].file;
        outl.add({lab, f, fmt(b.axis == 'Y' ? b.value : 0, 2), fmt(b.axis == 'T' ? b.value : 0, 2),
                  std::string(1, b.pred_axis) + " " + fmts(b.pred, 4) + " " + b.branch + " " + fmt(b.cost, 1),
                  std::string(1, w.pred_axis) + " " + fmts(w.pred, 4) + " " + w.branch + " " + fmt(w.cost, 1),
                  std::to_string(S[1].fails.count(f) ? S[1].fails.at(f) : 0), std::to_string(S[0].fails.count(f) ? S[0].fails.at(f) : 0)});
    };
    row("full280 Y 오차 최대", wyb, wyw);
    if (tseed >= 0) row("반복 최다 실패 (seed " + std::to_string(7000 + tseed) + ")", tb, tw);

    // ---- 결과.md ----
    std::ostringstream R;
    R << "# 결과: 픽셀 잡음 역분산 가중 GN17\n\n"
      << "입력: `data/clean/clean_20261014` 281장. center.bmp 기준 GN17 spread14(QR 205점)를 사진에서 다시 재고, QR 조각·스플라인 계수·LK15 초기값·가중 없는 GN의 gain/offset을 준비했다.\n"
      << "분할마다: 학습 Y 사진 중 평균 이동이 정수에 가장 가까운 8쌍으로 밝기 16구간 잡음을 추정 -> 23x23 원본 밝기로 두 사진 분산을 흐림 커널 제곱으로 전파 -> 역분산 가중 GN17로 281장 재측정 -> P12, 6분기 2차 곡선, 기존 공분산 판정을 학습 행으로 다시 맞춘다.\n"
      << "평가: full280(재대입), 층화 5분할(seed 20261014), 시행 내 블록 5분할, 반복 5분할 20회(seed 7000~7019), 시행 제외 6분할.\n\n"
      << "## 요약\n\n" << summ.md() << "\n반복 20회에서 기존 대비 고친 예측 " << fixed << "건, 새로 틀린 예측 " << newf << "건.\n\n"
      << "## full280 잡음 곡선\n\n" << noise.md() << "\n## full280 잡음 추정 쌍\n\n" << pairs.md() << "\n## 반복 20회 사진별 실패\n\n" << freq.md()
      << "\n## 원래 결과와 대조 (" << par.ok << "/" << par.total << " 일치)\n\n" << par.t.md()
      << "\n## 옮기지 않은 부분\n\n- `interleaved` 2분할: 촬영 시각 순서가 필요한데 `data/clean`에는 촬영 시각이 없어 뺐다.\n"
      << "- 원래 cache/(QR 조각 npy 80MB)는 쓰지 않고 실행 때 메모리에서 다시 만든다. 보고서 HTML·패키징·held-out 변조 검사 스크립트는 옮기지 않았다.\n\n"
      << "## 파일\n\n- `comparison_summary.csv`: 두 방법 집계\n- `fold_diagnostics.csv`: 분할별 가중치 범위·측정 변화·거절\n";
    write_text(out("결과.md"), R.str());

    // ---- 보고서.md ----
    std::ostringstream Pp;
    Pp << "# 보고서: 픽셀 잡음 역분산 가중 GN (10-14 16:21)\n\n"
       << "## 문제\n\nGN 추적 잔차에 밝기별 잡음 가중을 주면 반복 분할 오분류가 줄어드는지 확인해야 했다.\n\n"
       << "## 해결 방법\n\n학습 분할의 Y 사진 8쌍으로 밝기별 잡음 분산을 추정하고, 그 역분산을 픽셀 가중으로 넣은 GN17로 281장을 다시 쟀다.\n\n"
       << "## 결과\n\nfull280 Y 최대 오차가 " << fmt(S[0].full.Ymax, 5) << " -> " << fmt(S[1].full.Ymax, 5) << " mm로 줄었으나 반복 실패는 " << S[0].cv_failed << " -> "
       << S[1].cv_failed << "/5600으로 거의 같았다.\n\n" << summ.md() << "\n"
       << img_md("01_밝기별_잡음곡선.png", "잡음 곡선") << "\n\n" << img_md("02_full280_Y오차_기존대가중.png", "full280 Y 오차") << "\n\n"
       << img_md("03_반복5분할_실패.png", "반복 실패") << "\n\n"
       << "## 튀는 값\n\nfull280에서 가중 GN의 Y 오차가 가장 큰 사진은 " << m[wy].file << "이고, 반복 20회에서 가장 자주 틀린 사진은 " << ft << "(" << cft
       << "회)로 기존 GN에서도 " << (S[0].fails.count(ft) ? S[0].fails.at(ft) : 0) << "회 틀린 사진이라 가중이 분기 선택을 바꾸지 못했다.\n\n"
       << outl.md() << "\n" << img_md("04_튀는값_Y_" + ny + ".png", "Y 튀는 값") << "\n\n" << img_md("05_튀는값_T_" + nt + ".png", "T 튀는 값")
       << "\n\n초록 상자는 기준 사진의 QR, 빨간 상자는 기존 GN17 점 평균 이동만큼 옮긴 QR이다. 파란 십자는 측정 점, 화살표는 full280 가중 GN과 기존 GN의 점별 차이 x300이다.\n\n"
       << "## 한줄평\n\n가중은 어두운·밝은 구간(잡음 sd " << fmt(*std::min_element(d0.cu.sd.begin(), d0.cu.sd.end()), 2) << "~" << fmt(*std::max_element(d0.cu.sd.begin(), d0.cu.sd.end()), 2)
       << " gray) 차이만큼 픽셀 비중을 바꿔 점 위치를 중앙값 " << fmt(d0.med, 4) << " px 옮기므로 재대입 Y 최대 오차는 " << fmt(S[0].full.Ymax / S[1].full.Ymax, 1)
       << "배 줄었지만, 실패의 원인인 " << ft << " 같은 사진의 분기 선택은 점 위치 0.01 px 수준 변화로 바뀌지 않아 반복 실패가 " << S[0].cv_failed << " -> " << S[1].cv_failed << "건에 그쳤다.\n\n"
       << "## 원래 결과와 대조\n\n원래 값은 `pixel_weighted_gn_20261015/comparison_summary.csv`, `metrics.csv`, `fold_diagnostics.csv`, `folds/full280__all/noise_model.json`에서 가져왔다. "
       << par.ok << "/" << par.total << " 일치.\n\n" << par.t.md();
    write_text(out("보고서.md"), Pp.str());
    std::cout << "parity " << par.ok << "/" << par.total << "\n";
    return 0;
}
