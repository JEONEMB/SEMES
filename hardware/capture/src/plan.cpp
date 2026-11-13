// plan.cpp — 촬영 계획 구현
#include "capture/plan.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <sstream>

namespace capture {

std::string fmtMilli(int32_t v)
{
    const char sign = v < 0 ? '-' : '+';
    const int64_t a = v < 0 ? -static_cast<int64_t>(v) : v;
    const long long ip = a / 1000, frac = a % 1000;
    char buf[32];
    if (frac % 10 == 0)
        std::snprintf(buf, sizeof(buf), "%c%lld.%02lld", sign, ip, frac / 10);
    else
        std::snprintf(buf, sizeof(buf), "%c%lld.%03lld", sign, ip, frac);
    return buf;
}

bool parseMilli(const std::string &s, int32_t &out)
{
    size_t i = 0;
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) ++i;
    size_t end = s.size();
    while (end > i && (s[end - 1] == ' ' || s[end - 1] == '\t' || s[end - 1] == '\r')) --end;
    bool neg = false;
    if (i < end && (s[i] == '+' || s[i] == '-')) neg = s[i++] == '-';
    int64_t ip = 0, frac = 0;
    int ndig = 0, nfrac = 0;
    while (i < end && s[i] >= '0' && s[i] <= '9') {
        ip = ip * 10 + (s[i++] - '0');
        if (++ndig > 6) return false;
    }
    if (i < end && s[i] == '.') {
        ++i;
        while (i < end && s[i] >= '0' && s[i] <= '9') {
            const int d = s[i++] - '0';
            if (nfrac < 3) frac = frac * 10 + d;
            else if (d != 0) return false;  // 0.0005 같은 값은 프로토콜 분해능 밖
            ++nfrac;
        }
    }
    if (i != end || (ndig == 0 && nfrac == 0)) return false;
    for (int k = nfrac; k < 3; ++k) frac *= 10;
    const int64_t v = ip * 1000 + frac;
    out = static_cast<int32_t>(neg ? -v : v);
    return true;
}

namespace {

std::vector<ShotSpec> planAxis(int32_t range, int32_t step, bool y_axis)
{
    std::vector<ShotSpec> out;
    if (range <= 0 || step <= 0) return out;
    const int32_t n = range / step;
    for (int32_t k = -n; k <= n; ++k) {
        if (k == 0) continue;
        ShotSpec s;
        const int32_t v = k * step;
        if (y_axis) {
            s.rel.y_um = v;
            s.dir = "Y";
            s.stem = "Y" + fmtMilli(v);
        } else {
            s.rel.t_mdeg = v;
            s.dir = "Theta";
            s.stem = "T" + fmtMilli(v);
        }
        out.push_back(s);
    }
    return out;
}

}  // namespace

std::vector<ShotSpec> planY(int32_t range_um, int32_t step_um) { return planAxis(range_um, step_um, true); }

std::vector<ShotSpec> planT(int32_t range_mdeg, int32_t step_mdeg) { return planAxis(range_mdeg, step_mdeg, false); }

std::vector<ShotSpec> planMixed(const std::vector<Rel> &poses, int repeat)
{
    std::vector<ShotSpec> out;
    for (const Rel &p : poses) {
        const std::string name = "Y" + fmtMilli(p.y_um) + " T" + fmtMilli(p.t_mdeg);
        for (int k = 1; k <= repeat; ++k) {
            ShotSpec s;
            s.rel = p;
            s.dir = name;
            s.stem = name + " (" + std::to_string(k) + ")";
            out.push_back(s);
        }
    }
    return out;
}

std::vector<ShotSpec> planGrid(int32_t y_range_um, int32_t y_step_um, int32_t t_range_mdeg, int32_t t_step_mdeg)
{
    std::vector<ShotSpec> out;
    if (y_range_um < 0 || t_range_mdeg < 0 || y_step_um <= 0 || t_step_mdeg <= 0) return out;
    const int32_t ny = y_range_um / y_step_um, nt = t_range_mdeg / t_step_mdeg;
    for (int32_t i = -nt; i <= nt; ++i) {
        const bool forward = ((i + nt) % 2) == 0;     // 첫 줄은 Y 오름차순, 다음 줄은 내림차순 ('ㄹ')
        for (int32_t j = -ny; j <= ny; ++j) {
            const int32_t k = forward ? j : -j;
            if (k == 0 && i == 0) continue;            // 센터는 따로 찍는다
            ShotSpec s;
            s.rel = {k * y_step_um, i * t_step_mdeg};
            s.dir = "Grid";
            s.stem = "Y" + fmtMilli(s.rel.y_um) + " T" + fmtMilli(s.rel.t_mdeg) + " (1)";
            out.push_back(s);
        }
    }
    return out;
}

std::vector<Rel> defaultMixedPoses()
{
    std::vector<Rel> out;
    for (int32_t t : {10, 50, 200, 500, 1000})
        for (int32_t y : {50, 200, 500, 1000}) out.push_back({y, t});
    return out;
}

bool parsePoseCsv(const std::string &text, std::vector<Rel> &out, std::string &err)
{
    std::istringstream in(text);
    std::string line;
    int lineno = 0;
    bool first = true;
    while (std::getline(in, line)) {
        ++lineno;
        const size_t hash = line.find('#');
        if (hash != std::string::npos) line.erase(hash);
        if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
        const size_t comma = line.find(',');
        Rel r;
        const bool ok = comma != std::string::npos && parseMilli(line.substr(0, comma), r.y_um) &&
                        parseMilli(line.substr(comma + 1), r.t_mdeg);
        if (!ok) {
            if (first) {  // 머리글 (y_mm,t_deg)
                first = false;
                continue;
            }
            err = std::to_string(lineno) + "번째 줄을 읽을 수 없음: " + line;
            return false;
        }
        first = false;
        out.push_back(r);
    }
    if (out.empty()) {
        err = "자세가 하나도 없음";
        return false;
    }
    return true;
}

void sortForTravel(std::vector<ShotSpec> &shots)
{
    std::stable_sort(shots.begin(), shots.end(), [](const ShotSpec &a, const ShotSpec &b) {
        if (a.rel.t_mdeg != b.rel.t_mdeg) return a.rel.t_mdeg < b.rel.t_mdeg;
        return a.rel.y_um < b.rel.y_um;
    });
}

}  // namespace capture
