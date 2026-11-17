// A17. control 측정기의 범위 밖 정수 이동 실패를 세 전략으로 비교한다.
// 입력 목록과 정답은 A0 캐시의 shots.csv 를 그대로 읽는다.
#include "tracker_variant.hpp"
#include "hye/plot.hpp"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>

namespace fs = std::filesystem;
using variant::Strategy;

namespace {
const char* kRepo = REPO_DIR;
const char* kOut = EXPERIMENT_DIR;
// A16 이 42,463 쌍에서 낸 판 가로 이동 감도.
constexpr double kPerY = 14.2491, kPerT = 80.0650;

struct Shot { int idx; std::string set, rel; double y, t; int valid_a0; };

std::vector<Shot> read_shots(const fs::path& csv) {
    std::ifstream in(csv);
    if (!in) throw std::runtime_error("A0 캐시 shots.csv 를 열 수 없다: " + csv.u8string());
    std::string line; std::getline(in, line);
    std::vector<std::string> head; { std::stringstream ss(line); std::string c; while (std::getline(ss, c, ',')) head.push_back(c); }
    auto column = [&](const std::string& name) {
        for (size_t i = 0; i < head.size(); ++i) if (head[i] == name) return int(i);
        throw std::runtime_error("shots.csv 에 " + name + " 열이 없다");
    };
    int ci = column("idx"), cs = column("set"), cr = column("rel"), cy = column("y"), ct = column("t"), cv_ = column("valid_selected");
    std::vector<Shot> out;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::vector<std::string> f; std::stringstream ss(line); std::string c;
        while (std::getline(ss, c, ',')) f.push_back(c);
        out.push_back({std::stoi(f[ci]), f[cs], f[cr], std::stod(f[cy]), std::stod(f[ct]), std::stoi(f[cv_])});
    }
    return out;
}

struct Row {
    std::string name; int n = 0, recovered = 0, gate_rejected = 0, shift_wrong = 0, matched_a0 = 0;
    double ms_total = 0, pixels_total = 0; int valid_min = 630; long long valid_sum = 0;
};

std::string f(double v, int p = 4) { std::ostringstream o; o << std::fixed << std::setprecision(p) << v; return o.str(); }
}

int main() {
    try {
        const fs::path repo = fs::u8path(kRepo), out = fs::u8path(kOut);
        fs::create_directories(out / "results");
        hye::begin_results();
        auto shots = read_shots(repo / "experiments" / "hye" /
                                fs::u8path("20261102_124309_A0_분할측정고정_계층분할과점측정캐시") /
                                fs::u8path("결과") / "cache" / "shots.csv");
        std::cout << "사진 " << shots.size() << "장\n";

        variant::Tracker tracker(repo / "experiments" / "hye" / "00000_final_Y_T control" / "assets");
        struct Case { const char* name; Strategy strategy; };
        const Case cases[] = {{"S_A 현재 단일 창 110x22", Strategy::Single},
                              {"S_B joint 식 단계 확장 110-260-512", Strategy::Escalate},
                              {"S_C 피라미드 + 간격 게이트", Strategy::Pyramid}};

        std::ofstream per(out / "results" / "per_shot.csv");
        per << "set,rel,y,t,predicted_dx,strategy,shift0,shift1,shift2,spread,ncc_min,valid_selected,valid_a0,gate_ok,gate_reason,matches,ms\n";
        std::ofstream gate(out / "results" / "gate_events.csv");
        gate << "set,rel,y,t,strategy,spread,ncc_min,valid_selected,reason\n";

        std::map<std::string, Row> rows;
        const fs::path clean = repo / "data" / "clean";

        for (size_t i = 0; i < shots.size(); ++i) {
            const auto& s = shots[i];
            cv::Mat img = legacy_control::read_gray(clean / fs::u8path(s.set) / fs::u8path(s.rel));
            double predicted = kPerY * s.y + kPerT * s.t;
            for (const auto& c : cases) {
                auto t0 = std::chrono::steady_clock::now();
                auto tr = tracker.measure(img, c.strategy);
                double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                int valid = tracker.valid_selected(tr);
                double ncc_min = *std::min_element(tr.ncc.begin(), tr.ncc.end());
                // 세 QR 전부를 본다. QR 하나만 보면 S_B 처럼 일부만 고친 경우를 놓친다.
                bool shift_ok = true;
                for (int k = 0; k < 3; ++k) if (std::abs(double(tr.shift[k].x) - predicted) > 8.0) shift_ok = false;
                auto& r = rows[c.name];
                r.name = c.name; ++r.n; r.ms_total += ms; r.pixels_total += tr.coarse_pixels;
                r.valid_sum += valid; r.valid_min = std::min(r.valid_min, valid);
                if (valid == 536) ++r.recovered;
                if (!tr.gate_ok) ++r.gate_rejected;
                if (!shift_ok) ++r.shift_wrong;
                if (valid == s.valid_a0) ++r.matched_a0;
                per << s.set << ',' << '"' << s.rel << '"' << ',' << f(s.y, 3) << ',' << f(s.t, 3) << ',' << f(predicted, 1) << ','
                    << '"' << c.name << '"' << ',' << tr.shift[0].x << ',' << tr.shift[1].x << ',' << tr.shift[2].x << ','
                    << tr.shift_spread << ',' << f(ncc_min, 3) << ',' << valid << ',' << s.valid_a0 << ','
                    << (tr.gate_ok ? 1 : 0) << ',' << '"' << tr.gate_reason << '"' << ',' << tr.template_matches << '\n';
                if (!tr.gate_ok)
                    gate << s.set << ',' << '"' << s.rel << '"' << ',' << f(s.y, 3) << ',' << f(s.t, 3) << ','
                         << '"' << c.name << '"' << ',' << tr.shift_spread << ',' << f(ncc_min, 3) << ',' << valid << ','
                         << '"' << tr.gate_reason << '"' << '\n';
            }
            if ((i + 1) % 50 == 0) std::cout << (i + 1) << "/" << shots.size() << std::flush << "\n";
        }

        // 그림 두 장
        {
            std::vector<std::string> cats; std::vector<double> valid_536, pixels, worst;
            for (const auto& c : cases) {
                const auto& r = rows[c.name];
                cats.push_back(c.name); valid_536.push_back(r.recovered);
                pixels.push_back(r.pixels_total / r.n / 1000.0); worst.push_back(r.valid_min);
            }
            hye::bar_chart("171_shots_with_all_536_valid.png",
                           "A17-171 shots with all 536 selected points valid (of 575)",
                           cats, valid_536, "shots", 2, 575);
            hye::bar_chart("172_worst_case_valid_points.png",
                           "A17-172 worst-case valid selected points (of 536)",
                           cats, worst, "points", 2, 536);
            hye::bar_chart("173_search_cost.png",
                           "A17-173 searched pixels per shot (thousands, lower is better)",
                           cats, pixels, "kilopixels", 2);
        }

        {   // 시간은 실행마다 달라 결과.md 와 summary.csv 에 넣지 않는다. 재현 대상이 아니다.
            std::ofstream tm(out / "results" / "timing.csv");
            tm << "strategy,n,ms_mean_this_run\n";
            for (const auto& c : cases) { const auto& r = rows[c.name]; tm << '"' << r.name << '"' << ',' << r.n << ',' << f(r.ms_total / r.n, 2) << '\n'; }
        }
        std::ofstream sum(out / "results" / "summary.csv");
        sum << "strategy,n,valid_536,valid_min,valid_mean,gate_rejected,shift_off_8px,matches_a0,pixels_per_shot\n";
        std::ostringstream md;
        md << "# A17 결과 - control 범위 밖 정수 이동 실패\n\n"
           << "`실험/src/main.cpp` 가 다시 쓴다. 해석은 [보고서](보고서.md). 계획은 [계획서](../계획서.md).\n\n"
           << "세 전략은 정수 이동 단계만 다르고 정밀화와 유효 판정은 같은 코드다. "
           << "참 정수 이동의 대리값은 `dx = 14.2491*Y + 80.0650*T` 다(A16).\n\n"
           << "## 1. 전략별 요약\n\n"
           << "| 전략 | 장수 | 선택점 536 전부 유효 | 최소 유효 | 평균 유효 | 게이트 거부 | 정수 이동 8 px 초과 | A0 과 같은 유효수 | 장당 평균 ms | 장당 탐색 화소 |\n"
           << "|---|---|---|---|---|---|---|---|---|---|\n";
        for (const auto& c : cases) {
            const auto& r = rows[c.name];
            sum << '"' << r.name << '"' << ',' << r.n << ',' << r.recovered << ',' << r.valid_min << ','
                << f(double(r.valid_sum) / r.n, 1) << ',' << r.gate_rejected << ',' << r.shift_wrong << ','
                << r.matched_a0 << ',' << f(r.pixels_total / r.n, 0) << '\n';
            md << "| " << r.name << " | " << r.n << " | " << r.recovered << " | " << r.valid_min << " | "
               << f(double(r.valid_sum) / r.n, 1) << " | " << r.gate_rejected << " | " << r.shift_wrong << " | "
               << r.matched_a0 << " | " << f(r.pixels_total / r.n, 0) << " |\n";
        }
        md << "\n![all 536 valid](이미지/171_shots_with_all_536_valid.png)\n\n"
              "![worst case](이미지/172_worst_case_valid_points.png)\n\n"
              "![search cost](이미지/173_search_cost.png)\n\n"
              "빨간 선은 각각 575장, 536점이다.\n\n"
           << "장당 평균 ms 는 세 전략을 한 프로세스에서 번갈아 돌린 값이라 캐시 효과가 섞여 있다. "
              "전략 사이의 **상대 비교**로만 읽는다. 장당 탐색 화소는 matchTemplate 에 넘긴 탐색 영역 넓이의 합이며 캐시와 무관한 비용 대리값이다.\n\n"
           << "## 2. 범위 밖 12장 (T+2.00)\n\n"
           << "| 사진 | 예측 dx | 전략 | QR1 | QR2 | QR3 | 불일치 | 최소 NCC | 유효/536 | 게이트 |\n"
           << "|---|---|---|---|---|---|---|---|---|---|\n";
        {
            std::ifstream in(out / "results" / "per_shot.csv"); std::string line; std::getline(in, line);
            while (std::getline(in, line)) {
                if (line.find("T+2.00") == std::string::npos && line.find("T+2.0 ") == std::string::npos) continue;
                std::vector<std::string> fcol; std::string cell; bool quoted = false;
                for (char ch : line) {
                    if (ch == '"') { quoted = !quoted; continue; }
                    if (ch == ',' && !quoted) { fcol.push_back(cell); cell.clear(); } else cell += ch;
                }
                fcol.push_back(cell);
                if (fcol.size() < 17) continue;
                md << "| " << fcol[1] << " | " << fcol[4] << " | " << fcol[5] << " | " << fcol[6] << " | " << fcol[7]
                   << " | " << fcol[8] << " | " << fcol[9] << " | " << fcol[10] << " | " << fcol[11] << "/536 | "
                   << (fcol[13] == "1" ? "통과" : "거부: " + fcol[14]) << " |\n";
            }
        }
        md << "\n전체 사진별 값은 [per_shot.csv](results/per_shot.csv), 게이트가 걸린 장은 [gate_events.csv](results/gate_events.csv) 에 있다.\n";
        std::ofstream(out / fs::u8path("결과.md")) << md.str();
        std::cout << "끝. 결과: " << (out / fs::u8path("결과.md")).u8string() << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "실패: " << e.what() << "\n";
        return 1;
    }
}
