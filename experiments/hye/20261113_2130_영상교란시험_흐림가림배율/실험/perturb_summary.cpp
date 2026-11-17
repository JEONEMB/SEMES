// perturb_test 출력(사진 × 교란)을 교란별 표로 요약한다. 지표는 계획.md 에 미리 정한 것.
// 사용: perturb_summary <출력 폴더> <요약.tsv> <점검효과.tsv> <자료 이름...>
//   <출력 폴더>/<자료 이름>.tsv 를 읽는다(예: semes_536 semes_ncc ssafy_ncc). 인자 경로는 ASCII.
//
// 요약.tsv (교란마다)
//   stopped: 예외로 값을 내지 않은 사진, silent_wrong: 교란 전 통과·교란 뒤 값은 냈는데 허용오차 밖
//   dY/dT: 교란 전 출력과의 차이 (중앙값, 95%, 최대, 평균), near_*: 센터 근처(|Y_abs|, |T_abs| ≤ 0.1) 사진만
//   zero_*/after_zero_*: 영점 업데이트 흉내(센터 근처 사진의 평균 ΔY, ΔT 를 뺌) 뒤의 흔들림과 조용히 틀림
// 점검효과.tsv (교란마다, health 열이 있을 때)
//   caught: 조용히 틀린 사진 중 측정값 일치 점검값이 기준을 넘은 사진, silent_after: 넘지 않은 사진(점검으로도 못 잡음)
//   needless_flag: 교란 뒤에도 통과인데 점검값이 기준을 넘은 사진
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {
const double kTolY = 0.045, kTolT = 0.01, kSmallY = 0.010, kSmallT = 0.002;
const double NaN = std::numeric_limits<double>::quiet_NaN();

struct Row { int id; std::string perturb; bool valid; double y, t, ya, ta, health, thr; std::string message; };

std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> v; std::string item; std::istringstream in(s);
    while (std::getline(in, item, '\t')) v.push_back(item);
    if (!s.empty() && s.back() == '\t') v.push_back("");
    return v;
}
double num(const std::string& s) { return s.empty() || s == "nan" ? NaN : std::stod(s); }

std::vector<Row> read_rows(const std::string& file) {
    std::ifstream f(file, std::ios::binary); if (!f) throw std::runtime_error("파일을 열 수 없다: " + file);
    std::string line; std::getline(f, line); if (!line.empty() && line.back() == '\r') line.pop_back();
    const auto head = split(line); std::map<std::string, int> col; for (int i = 0; i < int(head.size()); ++i) col[head[i]] = i;
    const bool has_health = col.count("health") > 0;
    std::vector<Row> rows;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        const auto v = split(line);
        Row r{std::stoi(v[col["id"]]), v[col["perturb"]], v[col["valid"]] == "1", num(v[col["Y"]]), num(v[col["T"]]), num(v[col["Y_abs"]]), num(v[col["T_abs"]]),
              has_health ? num(v[col["health"]]) : NaN, has_health ? num(v[col["health_threshold"]]) : NaN, v.size() > size_t(col["message"]) ? v[col["message"]] : ""};
        rows.push_back(r);
    }
    return rows;
}
// 빈 목록이면 NaN. 분위는 선형 보간(사이 값)
double quantile(std::vector<double> v, double q) {
    if (v.empty()) return NaN;
    std::sort(v.begin(), v.end()); const double pos = q * (v.size() - 1); const size_t lo = size_t(std::floor(pos));
    return lo + 1 < v.size() ? v[lo] + (pos - lo) * (v[lo + 1] - v[lo]) : v[lo];
}
double mean(const std::vector<double>& v) { if (v.empty()) return NaN; double s = 0; for (double x : v) s += x; return s / v.size(); }
double maxof(const std::vector<double>& v) { return v.empty() ? NaN : *std::max_element(v.begin(), v.end()); }
bool passes(double y, double t, double ya, double ta) { return std::abs(y - ya) <= kTolY && std::abs(t - ta) <= kTolT; }   // NaN 이면 false
std::string fmt(double v) { if (std::isnan(v)) return "nan"; std::ostringstream o; o << std::fixed << std::setprecision(5) << v; return o.str(); }
std::vector<double> absv(const std::vector<double>& v) { std::vector<double> a; for (double x : v) a.push_back(std::abs(x)); return a; }
}

int main(int argc, char** argv) {
    try {
        if (argc < 5) { std::cerr << "perturb_summary <출력 폴더> <요약.tsv> <점검효과.tsv> <자료 이름...>\n"; return 2; }
        std::ofstream sum(argv[2], std::ios::binary), eff(argv[3], std::ios::binary);
        sum << "data\tperturb\tn\tstopped\tsilent_wrong\tdY_med\tdY_p95\tdY_max\tdT_med\tdT_p95\tdT_max\tdY_mean\tdT_mean\tpass_rate\t"
               "near_n\tnear_stopped\tnear_silent\tnear_dY_p95\tnear_dT_p95\tsmall\tzero_dY\tzero_dT\tafter_zero_dY_p95\tafter_zero_dT_p95\tafter_zero_silent\tfirst_error\n";
        eff << "data\tperturb\tn\tstopped\tsilent_before\tcaught\tsilent_after\tneedless_flag\thealth_median\n";
        for (int a = 4; a < argc; ++a) {
            const std::string name = argv[a];
            const auto rows = read_rows(std::string(argv[1]) + "/" + name + ".tsv");
            std::map<int, std::pair<double, double>> base;
            for (const auto& r : rows) if (r.perturb == "none") base[r.id] = {r.y, r.t};
            std::vector<std::string> order; std::map<std::string, std::vector<const Row*>> groups;
            for (const auto& r : rows) { if (!groups.count(r.perturb)) order.push_back(r.perturb); groups[r.perturb].push_back(&r); }
            for (const auto& p : order) {
                const auto& g = groups[p];
                int stopped = 0, silent = 0, pass = 0, near_n = 0, near_stopped = 0, near_silent = 0, caught = 0, needless = 0;
                std::vector<double> dy, dt, ndy, ndt, health; std::map<std::string, int> errors;
                for (const Row* r : g) {
                    const auto b = base.at(r->id);
                    const bool pass0 = passes(b.first, b.second, r->ya, r->ta), ok = passes(r->y, r->t, r->ya, r->ta);
                    const bool near = std::abs(r->ya) <= 0.1 && std::abs(r->ta) <= 0.1;
                    const bool flag = r->valid && r->health > r->thr;
                    pass += ok; near_n += near;
                    if (!r->valid) { ++stopped; near_stopped += near; ++errors[r->message]; continue; }
                    dy.push_back(r->y - b.first); dt.push_back(r->t - b.second);
                    if (near) { ndy.push_back(r->y - b.first); ndt.push_back(r->t - b.second); }
                    if (!std::isnan(r->health)) health.push_back(r->health);
                    if (pass0 && !ok) { ++silent; near_silent += near; if (flag) ++caught; }
                    if (ok && flag) ++needless;
                }
                // 영점 업데이트 흉내: 센터 근처 사진의 평균 ΔY, ΔT 를 뺀다(센터 근처가 모두 멈추면 보정 없음)
                const double zy = ndy.empty() ? 0 : mean(ndy), zt = ndt.empty() ? 0 : mean(ndt);
                std::vector<double> cy, ct; int after_zero_silent = 0;
                for (const Row* r : g) {
                    if (!r->valid) continue;
                    const auto b = base.at(r->id);
                    cy.push_back(std::abs(r->y - b.first - zy)); ct.push_back(std::abs(r->t - b.second - zt));
                    if (passes(b.first, b.second, r->ya, r->ta) && !passes(r->y - zy, r->t - zt, r->ya, r->ta)) ++after_zero_silent;
                }
                const auto ay = absv(dy), at = absv(dt);
                const bool small = !dy.empty() && quantile(ay, .95) <= kSmallY && quantile(at, .95) <= kSmallT;
                std::string first_error; int most = 0; for (auto& e : errors) if (e.second > most) { most = e.second; first_error = e.first; }
                sum << name << '\t' << p << '\t' << g.size() << '\t' << stopped << '\t' << silent << '\t'
                    << fmt(quantile(ay, .5)) << '\t' << fmt(quantile(ay, .95)) << '\t' << fmt(maxof(ay)) << '\t'
                    << fmt(quantile(at, .5)) << '\t' << fmt(quantile(at, .95)) << '\t' << fmt(maxof(at)) << '\t'
                    << fmt(mean(dy)) << '\t' << fmt(mean(dt)) << '\t' << fmt(100.0 * pass / g.size()) << '\t'
                    << near_n << '\t' << near_stopped << '\t' << near_silent << '\t' << fmt(quantile(absv(ndy), .95)) << '\t' << fmt(quantile(absv(ndt), .95)) << '\t'
                    << (small ? "True" : "False") << '\t' << fmt(zy) << '\t' << fmt(zt) << '\t' << fmt(quantile(cy, .95)) << '\t' << fmt(quantile(ct, .95)) << '\t'
                    << after_zero_silent << '\t' << first_error << '\n';
                eff << name << '\t' << p << '\t' << g.size() << '\t' << stopped << '\t' << silent << '\t' << caught << '\t' << silent - caught << '\t' << needless << '\t'
                    << fmt(quantile(health, .5)) << '\n';
            }
        }
        std::cout << "written " << argv[2] << " " << argv[3] << "\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << "오류: " << e.what() << "\n"; return 1; }
}
