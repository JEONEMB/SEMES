// A23 집계: 같은 사진의 "없음(FFT 왕복)" 대비 교란 후 Y 변화 |ΔY| 의 평균과 최대를 버전별로 낸다.
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
namespace fs = std::filesystem;
int main() {
    const fs::path res = fs::u8path(EXPERIMENT_DIR) / "results";
    std::vector<std::string> order; std::map<std::string, std::map<std::string, std::pair<double, double>>> agg; std::map<std::string, std::map<std::string, int>> cnt, fail;
    for (std::string v : {"v1", "v2"}) {
        std::ifstream in(res / ("perturb_" + v + ".csv")); std::string l; std::getline(in, l); std::map<std::string, double> base;
        std::vector<std::vector<std::string>> rows; while (std::getline(in, l)) { std::vector<std::string> r; std::string c; std::stringstream ss(l); while (std::getline(ss, c, ',')) r.push_back(c); while (r.size() < 6) r.push_back(""); rows.push_back(r); }
        for (auto& r : rows) if (r[2].rfind("없음", 0) == 0 && !r[3].empty()) base[r[0] + "|" + r[1]] = std::stod(r[3]);
        for (auto& r : rows) { if (r[2].rfind("없음", 0) == 0) continue; if (v == "v1" && std::find(order.begin(), order.end(), r[2]) == order.end()) order.push_back(r[2]);
            auto it = base.find(r[0] + "|" + r[1]); if (r[3].empty() || it == base.end()) { fail[v][r[2]]++; continue; }
            double d = std::abs(std::stod(r[3]) - it->second); auto& a = agg[v][r[2]]; a.first += d; a.second = std::max(a.second, d); cnt[v][r[2]]++; }
    }
    std::ofstream o(res / "summary.csv"); o << "교란,v1 |ΔY| 평균,v1 |ΔY| 최대,v1 실패,v2 |ΔY| 평균,v2 |ΔY| 최대,v2 실패\n";
    for (auto& p : order) { o << p; for (std::string v : {"v1", "v2"}) o << "," << agg[v][p].first / std::max(1, cnt[v][p]) << "," << agg[v][p].second << "," << fail[v][p]; o << "\n"; }
    std::cout << "eval done\n";
}
