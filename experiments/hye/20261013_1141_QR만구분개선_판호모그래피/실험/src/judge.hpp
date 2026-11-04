// Y·Θ 구분 판정 (analysis_v2/judge.py) 과 직접 추정 (analysis_v2/estimate.py) 의 C++ 판.
// 짝 기준: QR 3개 가로 평균 m 이 같은 상대 축 지점을 보간으로 찾는다.
// 규칙1: 값마다 같은 축 LOO 보간 잔차 sd(두 축 중 큰 값) x 3 안이면 구분 불가. 규칙2: 마할라노비스 거리 <= chi2(0.9973, k) 면 구분 불가.
#pragma once
#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace judge {

struct Item {
    std::string file;  // 확장자 포함 파일명
    char ax;           // 'Y' 또는 'T'
    double value;      // 정답 (Y mm 또는 T deg)
    double m;          // QR 3개 가로 평균 이동 (px)
    std::vector<double> f;
    int src = -1;      // 원래 사진 인덱스
};

// np.interp (xp 오름차순, 범위 밖은 끝값)
inline double interp(double x, const std::vector<double>& xp, const std::vector<double>& fp) {
    if (x <= xp.front()) return fp.front();
    if (x >= xp.back()) return fp.back();
    size_t j = std::upper_bound(xp.begin(), xp.end(), x) - xp.begin();
    if (xp[j - 1] == x) return fp[j - 1];
    double t = (x - xp[j - 1]) / (xp[j] - xp[j - 1]);
    return fp[j - 1] + t * (fp[j] - fp[j - 1]);
}

// sheet: m 오름차순 정렬 (pandas sort_values 기본 quicksort; 같은 m 은 없다고 본다)
inline std::vector<Item> sheet(const std::vector<Item>& all, char ax) {
    std::vector<Item> s;
    for (auto& r : all) if (r.ax == ax) s.push_back(r);
    std::stable_sort(s.begin(), s.end(), [](const Item& a, const Item& b) { return a.m < b.m; });
    return s;
}

// row 의 값 - ref 를 m 으로 보간한 값 (skip: ref 에서 뺄 행)
inline std::vector<double> resid(const Item& row, const std::vector<Item>& ref, int skip = -1) {
    std::vector<double> xp;
    std::vector<std::vector<double>> fp(row.f.size());
    for (int i = 0; i < (int)ref.size(); ++i) {
        if (i == skip) continue;
        xp.push_back(ref[i].m);
        for (size_t k = 0; k < row.f.size(); ++k) fp[k].push_back(ref[i].f[k]);
    }
    std::vector<double> r(row.f.size());
    for (size_t k = 0; k < r.size(); ++k) r[k] = row.f[k] - interp(row.m, xp, fp[k]);
    return r;
}

// 짝: ref 에서 m 앞뒤 실제 사진 두 장 (범위 밖이면 -1)
inline std::pair<int, int> partners(double m, const std::vector<Item>& ref) {
    if (m < ref.front().m || m > ref.back().m) return {-1, -1};
    size_t j = std::lower_bound(ref.begin(), ref.end(), m, [](const Item& a, double v) { return a.m < v; }) - ref.begin();
    if (j == 0) return {0, std::min<int>(1, (int)ref.size() - 1)};
    return {(int)j - 1, (int)std::min(j, ref.size() - 1)};
}

// 정규화 하불완전 감마 P(a, x)
inline double gammap(double a, double x) {
    if (x <= 0) return 0;
    double gln = std::lgamma(a);
    if (x < a + 1) {
        double ap = a, sum = 1.0 / a, del = sum;
        for (int n = 0; n < 1000; ++n) { ap += 1; del *= x / ap; sum += del; if (std::abs(del) < std::abs(sum) * 1e-16) break; }
        return sum * std::exp(-x + a * std::log(x) - gln);
    }
    double b = x + 1 - a, c = 1e300, d = 1 / b, h = d;
    for (int i = 1; i < 1000; ++i) {
        double an = -i * (i - a);
        b += 2;
        d = an * d + b; if (std::abs(d) < 1e-300) d = 1e-300;
        c = b + an / c; if (std::abs(c) < 1e-300) c = 1e-300;
        d = 1 / d;
        double del = d * c;
        h *= del;
        if (std::abs(del - 1) < 1e-16) break;
    }
    return 1 - std::exp(-x + a * std::log(x) - gln) * h;
}
// scipy.stats.chi2.ppf(q, k)
inline double chi2_ppf(double q, int k) {
    double lo = 0, hi = 1000;
    for (int i = 0; i < 200; ++i) {
        double mid = (lo + hi) / 2;
        (gammap(k / 2.0, mid / 2) < q ? lo : hi) = mid;
    }
    return (lo + hi) / 2;
}

struct Decision {
    bool out = false;        // 상대 축 범위 밖
    bool r1 = false, r2 = false;  // 구분 불가 여부
    double d2 = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> r;   // 상대 축 보간 대비 차이
    int pa = -1, pb = -1;    // 짝 (상대 시트 인덱스)
    double partner_value = 0;
};

struct Result {
    std::vector<Item> Ys, Ts;
    std::vector<double> th;  // 규칙1 임계값
    double dthr = 0, fp1 = 0, fp2 = 0;
    std::vector<Decision> dy, dt;  // 시트 순서
    int n1Y = 0, n2Y = 0, n1T = 0, n2T = 0, inY = 0, inT = 0;
};

inline Result run(const std::vector<Item>& all) {
    Result R;
    R.Ys = sheet(all, 'Y');
    R.Ts = sheet(all, 'T');
    const size_t k = all[0].f.size();
    std::vector<std::vector<double>> L;
    auto loo = [&](const std::vector<Item>& s) {
        for (int i = 1; i + 1 < (int)s.size(); ++i) L.push_back(resid(s[i], s, i));
    };
    loo(R.Ys);
    const size_t nY = L.size();
    loo(R.Ts);
    auto sd = [&](size_t a, size_t b, size_t j) {
        double m = 0, s = 0;
        for (size_t i = a; i < b; ++i) m += L[i][j];
        m /= (b - a);
        for (size_t i = a; i < b; ++i) s += (L[i][j] - m) * (L[i][j] - m);
        return std::sqrt(s / (b - a));
    };
    for (size_t j = 0; j < k; ++j) R.th.push_back(3 * std::max(sd(0, nY, j), sd(nY, L.size(), j)));
    cv::Mat S((int)k, (int)k, CV_64F, cv::Scalar(0)), mu((int)k, 1, CV_64F, cv::Scalar(0));
    for (auto& r : L) for (size_t j = 0; j < k; ++j) mu.at<double>((int)j) += r[j] / L.size();
    for (auto& r : L)
        for (size_t a = 0; a < k; ++a)
            for (size_t b = 0; b < k; ++b) S.at<double>((int)a, (int)b) += (r[a] - mu.at<double>((int)a)) * (r[b] - mu.at<double>((int)b));
    S /= (k > 1 ? (double)(L.size() - 1) : (double)L.size());
    double tr = cv::trace(S)[0];
    for (size_t j = 0; j < k; ++j) S.at<double>((int)j, (int)j) += 1e-9 * tr / k;
    cv::Mat Si = S.inv(cv::DECOMP_LU);
    R.dthr = chi2_ppf(0.9973, (int)k);
    auto md = [&](const std::vector<double>& r) {
        double s = 0;
        for (size_t a = 0; a < k; ++a) for (size_t b = 0; b < k; ++b) s += r[a] * Si.at<double>((int)a, (int)b) * r[b];
        return s;
    };
    int c1 = 0, c2 = 0;
    for (auto& r : L) {
        bool any = false;
        for (size_t j = 0; j < k; ++j) any |= std::abs(r[j]) > R.th[j];
        c1 += any;
        c2 += md(r) > R.dthr;
    }
    R.fp1 = (double)c1 / L.size();
    R.fp2 = (double)c2 / L.size();
    auto decide = [&](const std::vector<Item>& s, const std::vector<Item>& ref, std::vector<Decision>& out, int& n1, int& n2, int& nin) {
        for (auto& row : s) {
            Decision d;
            if (row.m < ref.front().m || row.m > ref.back().m) { d.out = true; out.push_back(d); continue; }
            ++nin;
            d.r = resid(row, ref);
            d.d2 = md(d.r);
            d.r1 = true;
            for (size_t j = 0; j < k; ++j) d.r1 &= std::abs(d.r[j]) <= R.th[j];
            d.r2 = d.d2 <= R.dthr;
            auto pr = partners(row.m, ref);
            d.pa = pr.first; d.pb = pr.second;
            std::vector<double> xp, vp;
            for (auto& q : ref) { xp.push_back(q.m); vp.push_back(q.value); }
            d.partner_value = interp(row.m, xp, vp);
            n1 += d.r1; n2 += d.r2;
            out.push_back(d);
        }
    };
    decide(R.Ys, R.Ts, R.dy, R.n1Y, R.n2Y, R.inY);
    decide(R.Ts, R.Ys, R.dt, R.n1T, R.n2T, R.inT);
    return R;
}

// estimate.py: [Ys; Ts] 순서, 전체 표준화, 짝/홀 2겹, 릿지 lam = 1e-9 n (절편 벌점 없음)
struct Est {
    std::vector<const Item*> rows;
    std::vector<double> pT, pY, tT, tY;
    double yrms_um = 0, ymax_um = 0, trms = 0, small_um = 0;
    int over90 = 0;
};
inline Est estimate(const Result& R) {
    Est E;
    for (auto& r : R.Ys) E.rows.push_back(&r);
    for (auto& r : R.Ts) E.rows.push_back(&r);
    const int n = (int)E.rows.size(), k = (int)E.rows[0]->f.size();
    cv::Mat A(n, k + 1, CV_64F);
    for (int j = 0; j < k; ++j) {
        double m = 0, s = 0;
        for (auto r : E.rows) m += r->f[j] / n;
        for (auto r : E.rows) s += (r->f[j] - m) * (r->f[j] - m);
        s = std::sqrt(s / n);
        for (int i = 0; i < n; ++i) A.at<double>(i, j) = (E.rows[i]->f[j] - m) / s;
    }
    for (int i = 0; i < n; ++i) A.at<double>(i, k) = 1;
    for (auto r : E.rows) { E.tT.push_back(r->ax == 'T' ? r->value : 0); E.tY.push_back(r->ax == 'Y' ? r->value : 0); }
    E.pT.assign(n, 0); E.pY.assign(n, 0);
    const double lam = 1e-9 * n;
    for (int fold = 0; fold < 2; ++fold) {
        std::vector<int> tr, te;
        for (int i = 0; i < n; ++i) ((i % 2) != fold ? tr : te).push_back(i);
        cv::Mat At((int)tr.size(), k + 1, CV_64F);
        for (size_t i = 0; i < tr.size(); ++i) A.row(tr[i]).copyTo(At.row((int)i));
        cv::Mat G = At.t() * At;
        for (int j = 0; j < k; ++j) G.at<double>(j, j) += lam;
        for (int which = 0; which < 2; ++which) {
            auto& tgt = which ? E.tY : E.tT;
            auto& out = which ? E.pY : E.pT;
            cv::Mat y((int)tr.size(), 1, CV_64F);
            for (size_t i = 0; i < tr.size(); ++i) y.at<double>((int)i) = tgt[tr[i]];
            cv::Mat w;
            cv::solve(G, At.t() * y, w, cv::DECOMP_LU);
            for (int i : te) out[i] = A.row(i).dot(w.t());
        }
    }
    double sy = 0, st = 0, ss = 0;
    int ns = 0;
    for (int i = 0; i < n; ++i) {
        double ey = std::abs(E.pY[i] - E.tY[i]) * 1000, et = std::abs(E.pT[i] - E.tT[i]);
        sy += ey * ey; st += et * et;
        E.ymax_um = std::max(E.ymax_um, ey);
        E.over90 += ey > 90;
        if (std::abs(E.tY[i]) <= 0.5 && std::abs(E.tT[i]) <= 0.09) { ss += ey * ey; ++ns; }
    }
    E.yrms_um = std::sqrt(sy / n); E.trms = std::sqrt(st / n); E.small_um = std::sqrt(ss / ns);
    return E;
}

}  // namespace judge
