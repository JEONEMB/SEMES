// =====================================================================================
// session_h7.cpp — 가설 H7 : 세션 보정
// =====================================================================================
#include "session_h7.hpp"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>

namespace h3 {

EvalStat evalStat(const std::vector<SessionSample>& s, double tolT, double tolY)
{
    EvalStat e;
    for (const SessionSample& x : s) {
        double dT = x.estT_deg - x.trueT_deg, dY = (x.estY_mm - x.trueY_mm) * 1000.0;
        e.n++; e.rmsT += dT * dT; e.meanT += dT; e.rmsY += dY * dY; e.meanY += dY;
        e.maxT = std::max(e.maxT, std::fabs(dT));
        bool pt = std::fabs(dT) <= tolT, py = std::fabs(dY) <= tolY * 1000.0;
        e.passT += pt; e.passY += py; e.passBoth += (pt && py);
    }
    if (e.n) { e.rmsT = std::sqrt(e.rmsT / e.n); e.meanT /= e.n; e.rmsY = std::sqrt(e.rmsY / e.n); e.meanY /= e.n; }
    return e;
}

bool isCalibrationPose(double t) { return std::fabs(t - 0.01) < 1e-6 || std::fabs(t - 1.0) < 1e-6; }

SessionCal fitSession(const std::vector<SessionSample>& s)
{
    SessionCal c;
    const int n = (int)s.size();
    if (n < 3) return c;
    cv::Mat A(n, 2, CV_64F), b(n, 1, CV_64F), B(n, 2, CV_64F), d(n, 1, CV_64F), x;
    for (int i = 0; i < n; ++i) {
        A.at<double>(i, 0) = s[i].trueT_deg; A.at<double>(i, 1) = 1; b.at<double>(i) = s[i].estT_deg;
        B.at<double>(i, 0) = 1; B.at<double>(i, 1) = s[i].trueT_deg; d.at<double>(i) = s[i].estY_mm - s[i].trueY_mm;
    }
    cv::solve(A, b, x, cv::DECOMP_SVD); c.gainT = x.at<double>(0); c.offT = x.at<double>(1);
    cv::solve(B, d, x, cv::DECOMP_SVD); c.offY = x.at<double>(0); c.kY = x.at<double>(1);
    c.nCal = n; c.ok = std::fabs(c.gainT) > 1e-6;
    return c;
}

std::vector<SessionSample> applySession(const SessionCal& c, const std::vector<SessionSample>& s)
{
    std::vector<SessionSample> out = s;
    for (SessionSample& x : out) {
        x.estT_deg = (x.estT_deg - c.offT) / c.gainT;
        x.estY_mm = x.estY_mm - c.offY - c.kY * x.estT_deg;
    }
    return out;
}

H7Result runH7(const std::string& method, const std::vector<SessionSample>& all)
{
    H7Result r; r.method = method;
    std::vector<SessionSample> cal, test;
    for (const SessionSample& x : all) (isCalibrationPose(x.trueT_deg) ? cal : test).push_back(x);
    r.cal = fitSession(cal);
    r.before = evalStat(test);
    r.after = r.cal.ok ? evalStat(applySession(r.cal, test)) : r.before;
    return r;
}

// ------------------------------------------------------------------ CSV
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

std::vector<SessionSample> readSet2Csv(const std::string& path, bool* found)
{
    std::vector<SessionSample> out;
    std::ifstream f(path, std::ios::binary);
    if (found) *found = (bool)f;
    if (!f) return out;
    std::string line;
    std::getline(f, line);
    if (line.size() >= 3 && (unsigned char)line[0] == 0xEF) line = line.substr(3);
    std::vector<std::string> h = splitCsv(line);
    std::map<std::string, int> col;
    for (int i = 0; i < (int)h.size(); ++i) col[h[i]] = i;
    for (const char* need : { "file", "trueY_mm", "estY_mm", "trueT_deg", "estT_deg" }) if (!col.count(need)) { if (found) *found = false; return out; }
    while (std::getline(f, line)) {
        std::vector<std::string> v = splitCsv(line);
        if ((int)v.size() < (int)h.size()) continue;
        const std::string& name = v[col["file"]];
        if (name.rfind("center", 0) == 0) continue;
        if (name.find('(') == std::string::npos) continue;          // 2차 세트 = 이름에 촬영 번호 "(n)"
        if (v[col["estT_deg"]].empty() || v[col["trueT_deg"]].empty()) continue;
        SessionSample s;
        s.trueY_mm = std::stod(v[col["trueY_mm"]]); s.estY_mm = std::stod(v[col["estY_mm"]]);
        s.trueT_deg = std::stod(v[col["trueT_deg"]]); s.estT_deg = std::stod(v[col["estT_deg"]]);
        out.push_back(s);
    }
    return out;
}

} // namespace h3
