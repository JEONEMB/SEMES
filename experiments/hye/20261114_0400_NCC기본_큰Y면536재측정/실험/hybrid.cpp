// 혼합 방식 계산: 흐림 NCC 로 먼저 추정하고, 그 추정이 문턱을 넘으면 536점 추적 값으로 바꾼다.
// 입력: final00_tool range 가 낸 predictions_range.tsv 두 개(같은 사진, 같은 분할, 값 5개 = 바 없음)
// 사용: hybrid <ncc predictions_range.tsv> <536 predictions_range.tsv> <출력.txt>
// 시간 가정(노트북 4스레드, SEMES): NCC 한 장 약 15 ms, 536점 한 장 약 45 ms. 다시 재면 QR 값만 536점으로 다시 잰다(나사는 그대로)고 보고 +30 ms.
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {
struct Pred { std::string region; double y, t, py, pt; };
std::map<int, Pred> read(const std::string& file) {
    std::ifstream f(file); if (!f) throw std::runtime_error("파일을 열 수 없다: " + file);
    std::string line; std::getline(f, line);
    std::map<int, Pred> out;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::vector<std::string> c; std::string item; std::istringstream s(line); while (std::getline(s, item, '\t')) c.push_back(item);
        if (c.size() < 9) continue;
        // id source region Y_given T_given Y_pred eY T_pred eT ...
        out[std::stoi(c[0])] = {c[2], std::stod(c[3]), std::stod(c[4]), std::stod(c[5]), std::stod(c[7])};
    }
    return out;
}
bool passes(double ey, double et) { return std::abs(ey) <= 0.045 && std::abs(et) <= 0.01; }
}

int main(int argc, char** argv) {
    try {
        if (argc != 4) { std::cerr << "hybrid <ncc.tsv> <536.tsv> <출력.txt>\n"; return 2; }
        const auto ncc = read(argv[1]), t536 = read(argv[2]);
        const double ms_ncc = 15, ms_extra = 30;
        struct Rule { std::string name; double y_limit, t_limit; };   // 추정 |Y| > y_limit 또는 |T| > t_limit 이면 536점으로
        const std::vector<Rule> rules = {
            {"NCC 만", 1e9, 1e9}, {"|Y| > 0.5 면 536", 0.5, 1e9}, {"|Y| > 1.0 면 536", 1.0, 1e9}, {"|Y| > 1.5 면 536", 1.5, 1e9},
            {"|T| > 0.5 면 536", 1e9, 0.5}, {"|T| > 0.8 면 536", 1e9, 0.8}, {"|Y| > 1 또는 |T| > 0.5 면 536", 1.0, 0.5}, {"536 만", -1, -1}};
        std::ostringstream o; o << std::fixed;
        o << "사진 " << ncc.size() << "장 (안쪽은 5-fold, 바깥쪽은 안쪽만 학습한 모델로 예측)\n";
        o << "규칙 | 다시 잰 비율 | 평균 시간(ms) | 탈락 전체 | 탈락 안쪽 | 탈락 바깥쪽 | rms Y 바깥쪽 | rms T 바깥쪽\n";
        for (const auto& r : rules) {
            int n = 0, redo = 0, fail = 0, fail_in = 0, fail_out = 0, n_out = 0; double sy = 0, st = 0;
            for (const auto& [id, a] : ncc) {
                const auto it = t536.find(id); if (it == t536.end()) continue;
                const Pred& b = it->second; ++n;
                const bool use536 = std::abs(a.py) > r.y_limit || std::abs(a.pt) > r.t_limit;   // NCC 추정으로 판단
                const Pred& p = use536 ? b : a; redo += use536;
                const double ey = p.py - p.y, et = p.pt - p.t; const bool ok = passes(ey, et);
                fail += !ok; (a.region == "outer" ? fail_out : fail_in) += !ok;
                if (a.region == "outer") { ++n_out; sy += ey * ey; st += et * et; }
            }
            o << r.name << " | " << std::setprecision(1) << 100.0 * redo / n << "% | " << ms_ncc + ms_extra * redo / n << " | " << fail << " | " << fail_in << " | " << fail_out
              << " | " << std::setprecision(5) << std::sqrt(sy / n_out) << " | " << std::sqrt(st / n_out) << "\n";
        }
        std::ofstream(argv[3], std::ios::binary) << o.str();
        std::cout << o.str();
        return 0;
    } catch (const std::exception& e) { std::cerr << "오류: " << e.what() << "\n"; return 1; }
}
