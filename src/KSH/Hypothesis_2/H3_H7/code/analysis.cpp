// =====================================================================================
// analysis.cpp — H3 · H6 · H7 오차 분석 도구
// =====================================================================================
#include "analysis.hpp"
#include "bias_h6.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>

namespace h3 {

// y = c0 + c1·x + c2·x² 최소제곱 → (기울기 c1, 2차 c2, 잔차 σ)
static TermFit polyfit2(const std::string& name, const std::vector<double>& x, const std::vector<double>& y)
{
    TermFit t; t.name = name;
    const int n = (int)x.size();
    if (n < 4) return t;
    cv::Mat A(n, 3, CV_64F), b(n, 1, CV_64F), c;
    for (int i = 0; i < n; ++i) { A.at<double>(i, 0) = 1; A.at<double>(i, 1) = x[i]; A.at<double>(i, 2) = x[i] * x[i]; b.at<double>(i) = y[i]; }
    cv::solve(A, b, c, cv::DECOMP_SVD);
    cv::Mat r = b - A * c; cv::Scalar mu, sd; cv::meanStdDev(r, mu, sd);
    t.slope = c.at<double>(1); t.quad = c.at<double>(2); t.resid = sd[0];
    return t;
}

PhysicsCheck physicsCheck(const std::vector<Item>& items, const std::vector<cv::Point2d>& P0, const std::vector<Track>& tr)
{
    // 분석용 기저 : u = (x − 1920)/1000, v = (y − 1080)/1000, dx·dy 모두 [1, u, u², v, uv]
    FieldSpec s; s.cx_px = 1920; s.cy_px = 1080;
    s.bx = { B_ONE, B_U, B_U2, B_V, B_UV }; s.by = s.bx;
    const char* nm[10] = { "dx-1", "dx-u", "dx-u2", "dx-v", "dx-uv", "dy-1", "dy-u", "dy-u2", "dy-v", "dy-uv" };
    std::vector<double> tT, yY; std::vector<std::vector<double>> kT(10), kY(10);
    for (size_t i = 0; i < items.size(); ++i) {
        const Item& it = items[i];
        if (it.set != 1 || it.isCenter) continue;
        // 일반 최소제곱 (로버스트 반복 없음)
        std::vector<double> k;
        for (int comp = 0; comp < 2; ++comp) {
            int n = 0; for (uchar o : tr[i].ok) n += o;
            cv::Mat A(n, 5, CV_64F), d(n, 1, CV_64F), c;
            for (size_t p = 0, r = 0; p < P0.size(); ++p) if (tr[i].ok[p]) {
                double u = (P0[p].x - 1920) / 1000, v = (P0[p].y - 1080) / 1000;
                for (int j = 0; j < 5; ++j) A.at<double>((int)r, j) = basisValue(s.bx[j], u, v);
                d.at<double>((int)r) = comp == 0 ? tr[i].q[p].x - P0[p].x : tr[i].q[p].y - P0[p].y; ++r;
            }
            cv::solve(A, d, c, cv::DECOMP_SVD);
            for (int j = 0; j < 5; ++j) k.push_back(c.at<double>(j));
        }
        if (it.y_mm == 0) { tT.push_back(it.t_deg); for (int j = 0; j < 10; ++j) kT[j].push_back(k[j]); }
        if (it.t_deg == 0) { yY.push_back(it.y_mm); for (int j = 0; j < 10; ++j) kY[j].push_back(k[j]); }
    }
    PhysicsCheck pc;
    for (int j = 0; j < 10; ++j) { pc.theta.push_back(polyfit2(nm[j], tT, kT[j])); pc.y.push_back(polyfit2(nm[j], yY, kY[j])); }
    return pc;
}

NoiseFloor noiseFloor(const std::vector<int>& idx, const std::vector<cv::Point2d>& P0, const std::vector<Track>& tr,
                      const FieldSpec& s, double sensU2)
{
    NoiseFloor nf; nf.sensU2 = sensU2;
    int iU2 = -1;
    for (size_t j = 0; j < s.bx.size(); ++j) if (s.bx[j] == B_U2) iU2 = (int)j;
    if (iU2 < 0 || idx.empty()) return nf;
    double sum = 0;
    for (int i : idx) {
        const int nb = (int)s.bx.size();
        int n = 0; for (uchar o : tr[i].ok) n += o;
        cv::Mat A(n, nb, CV_64F), d(n, 1, CV_64F), c;
        for (size_t p = 0, r = 0; p < P0.size(); ++p) if (tr[i].ok[p]) {
            double u = (P0[p].x - s.cx_px) / s.scale_px, v = (P0[p].y - s.cy_px) / s.scale_px;
            for (int j = 0; j < nb; ++j) A.at<double>((int)r, j) = basisValue(s.bx[j], u, v);
            d.at<double>((int)r) = tr[i].q[p].x - P0[p].x; ++r;
        }
        cv::solve(A, d, c, cv::DECOMP_SVD);
        cv::Mat r = d - A * c; cv::Scalar mu, sd; cv::meanStdDev(r, mu, sd);
        cv::Mat Ci = (A.t() * A).inv(cv::DECOMP_SVD) * (sd[0] * sd[0]);
        sum += std::sqrt(Ci.at<double>(iU2, iU2));
    }
    nf.sigU2 = sum / idx.size();
    nf.thetaSigma = nf.sigU2 / std::fabs(sensU2);
    return nf;
}

std::vector<TemporalRow> temporalNoise(const std::vector<Item>& items, const std::vector<cv::Point2d>& P0,
                                       const std::vector<Track>& tr, const FieldSpec& s, double sensU2)
{
    std::vector<TemporalRow> out;
    const double poses[4][2] = { {0.05, 0.01}, {0.05, 0.02}, {0.05, 0.03}, {0.50, 0.10} };
    int iU2 = -1;
    for (size_t j = 0; j < s.bx.size(); ++j) if (s.bx[j] == B_U2) iU2 = (int)j;
    for (const auto& ps : poses) {
        int ii[3];
        bool ok = true;
        for (int k = 0; k < 3; ++k) { ii[k] = findPose(items, 2, ps[0], ps[1], k + 1, false); if (ii[k] < 0) ok = false; }
        if (!ok) continue;
        TemporalRow row;
        char b[64]; std::snprintf(b, sizeof(b), "Y+%.2f T+%.2f", ps[0], ps[1]); row.pose = b;
        std::vector<double> r[3];
        for (int k = 0; k < 3; ++k) {
            r[k] = plainResidualX(P0, tr[ii[k]], s);
            // u² 계수 (일반 최소제곱)
            const int nb = (int)s.bx.size(); int n = 0; for (uchar o : tr[ii[k]].ok) n += o;
            cv::Mat A(n, nb, CV_64F), d(n, 1, CV_64F), c;
            for (size_t p = 0, q = 0; p < P0.size(); ++p) if (tr[ii[k]].ok[p]) {
                double u = (P0[p].x - s.cx_px) / s.scale_px, v = (P0[p].y - s.cy_px) / s.scale_px;
                for (int j = 0; j < nb; ++j) A.at<double>((int)q, j) = basisValue(s.bx[j], u, v);
                d.at<double>((int)q) = tr[ii[k]].q[p].x - P0[p].x; ++q;
            }
            cv::solve(A, d, c, cv::DECOMP_SVD);
            row.u2[k] = iU2 >= 0 ? c.at<double>(iU2) : 0.0;
        }
        double s1 = 0, s2 = 0; int n = 0;
        for (size_t p = 0; p < P0.size(); ++p) if (!std::isnan(r[0][p]) && !std::isnan(r[1][p])) { double d = r[0][p] - r[1][p]; s1 += d; s2 += d * d; ++n; }
        row.pointSd = n > 1 ? std::sqrt(std::max(0.0, s2 / n - (s1 / n) * (s1 / n))) / std::sqrt(2.0) : 0;
        double m = (row.u2[0] + row.u2[1] + row.u2[2]) / 3;
        for (int k = 0; k < 3; ++k) row.thetaEq[k] = (row.u2[k] - m) / sensU2;
        out.push_back(row);
    }
    return out;
}

// ------------------------------------------------------------------ 배경 시차 정답
static std::vector<std::string> splitCsv(const std::string& line)
{
    std::vector<std::string> out; std::string cur; bool q = false;
    for (char ch : line) {
        if (ch == '"') q = !q;
        else if (ch == ',' && !q) { out.push_back(cur); cur.clear(); }
        else if (ch != '\r') cur += ch;
    }
    out.push_back(cur);
    return out;
}

static bool readCsv(const std::string& path, std::vector<std::string>& head, std::vector<std::vector<std::string>>& rows)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::string line;
    if (!std::getline(f, line)) return false;
    if (line.size() >= 3 && (unsigned char)line[0] == 0xEF) line = line.substr(3);
    head = splitCsv(line);
    while (std::getline(f, line)) { auto v = splitCsv(line); if (v.size() >= head.size()) rows.push_back(v); }
    return true;
}

std::vector<Truth> loadParallaxTruth(const std::vector<Item>& items, const std::string& csv1, const std::string& csv2)
{
    std::vector<Truth> truth(items.size());
    for (const std::string& path : { csv1, csv2 }) {
        std::vector<std::string> h; std::vector<std::vector<std::string>> rows;
        if (!readCsv(path, h, rows)) continue;
        std::map<std::string, int> col;
        for (int i = 0; i < (int)h.size(); ++i) col[h[i]] = i;
        if (!col.count("file") || !col.count("estY_mm") || !col.count("estT_deg")) continue;
        for (const auto& v : rows) {
            Item it;
            std::string name = v[col["file"]];
            std::string stem = name.substr(0, name.find_last_of('.'));
            if (!parseName(stem, it) || v[col["estT_deg"]].empty()) continue;
            int i = findPose(items, it.set, it.y_mm, it.t_deg, it.shot, it.isCenter);
            if (i < 0) continue;
            truth[i].y_mm = std::stod(v[col["estY_mm"]]); truth[i].t_deg = std::stod(v[col["estT_deg"]]); truth[i].valid = true;
        }
    }
    return truth;
}

VsTruth compareToTruth(const std::vector<int>& idx, const std::vector<Estimate>& est,
                       const std::vector<Truth>& truth, const std::vector<Item>& items)
{
    VsTruth v; double s2 = 0, sb = 0, sl = 0;
    for (int i : idx) {
        if (!truth[i].valid || !est[i].valid) continue;
        double d = est[i].t_deg - truth[i].t_deg, dl = truth[i].t_deg - items[i].t_deg;
        v.n++; s2 += d * d; sb += d; sl += dl * dl; v.pass += std::fabs(d) <= 0.01;
    }
    if (v.n) { v.rms = std::sqrt(s2 / v.n); v.bias = sb / v.n; v.truthVsLabelRms = std::sqrt(sl / v.n); }
    return v;
}

Backlash backlashFromTruth(const std::vector<Item>& items, const std::vector<Truth>& truth)
{
    // 1차 세트 스윕 : (정답 − 지령값) 의 평균을 + 방향 / − 방향 접근으로 나눠 비교
    Backlash b; double yp = 0, yn = 0, tp = 0, tn = 0; int nyp = 0, nyn = 0, ntp = 0, ntn = 0;
    for (size_t i = 0; i < items.size(); ++i) {
        const Item& it = items[i];
        if (it.set != 1 || it.isCenter || !truth[i].valid) continue;
        if (it.t_deg == 0 && it.y_mm != 0) { double d = truth[i].y_mm - it.y_mm; if (it.y_mm > 0) { yp += d; nyp++; } else { yn += d; nyn++; } }
        if (it.y_mm == 0 && it.t_deg != 0) { double d = truth[i].t_deg - it.t_deg; if (it.t_deg > 0) { tp += d; ntp++; } else { tn += d; ntn++; } }
    }
    if (nyp && nyn) b.halfY_um = 0.5 * (yp / nyp - yn / nyn) * 1000;
    if (ntp && ntn) b.halfT_deg = 0.5 * (tp / ntp - tn / ntn);
    b.nY = nyp + nyn; b.nT = ntp + ntn;
    return b;
}

bool set1ThetaTestStat(const std::string& csv, EvalStat& st)
{
    std::vector<std::string> h; std::vector<std::vector<std::string>> rows;
    if (!readCsv(csv, h, rows)) return false;
    std::map<std::string, int> col;
    for (int i = 0; i < (int)h.size(); ++i) col[h[i]] = i;
    for (const char* need : { "train", "trueY_mm", "estY_mm", "trueT_deg", "estT_deg" }) if (!col.count(need)) return false;
    std::vector<SessionSample> s;
    for (const auto& v : rows) {
        if (v[col["train"]] != "0" || v[col["estT_deg"]].empty() || v[col["trueT_deg"]].empty()) continue;
        double ty = std::stod(v[col["trueY_mm"]]), tt = std::stod(v[col["trueT_deg"]]);
        if (ty != 0 || tt == 0) continue;                       // Θ 스윕만
        s.push_back({ ty, tt, std::stod(v[col["estY_mm"]]), std::stod(v[col["estT_deg"]]) });
    }
    st = evalStat(s);
    return st.n > 0;
}

} // namespace h3
