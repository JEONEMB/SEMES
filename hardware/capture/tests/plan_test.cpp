// plan_test — 촬영 계획·파일 이름 시험 (카메라·스테이지 없이)
#include <cstdio>
#include <cstdlib>
#include <regex>
#include <set>
#include <string>

#include "capture/plan.hpp"

using namespace capture;

static int failures = 0;

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::fprintf(stderr, "실패 %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                   \
        }                                                                 \
    } while (0)

int main()
{
    // 이름 형식
    CHECK(fmtMilli(50) == "+0.05");
    CHECK(fmtMilli(-1200) == "-1.20");
    CHECK(fmtMilli(45) == "+0.045");
    CHECK(fmtMilli(0) == "+0.00");
    CHECK(fmtMilli(2000) == "+2.00");
    CHECK(fmtMilli(-10) == "-0.01");

    // 숫자 읽기 (실수 반올림 없이)
    int32_t v = 0;
    CHECK(parseMilli("0.05", v) && v == 50);
    CHECK(parseMilli("-1.2", v) && v == -1200);
    CHECK(parseMilli("+2", v) && v == 2000);
    CHECK(parseMilli(" 0.010 ", v) && v == 10);
    CHECK(parseMilli("0.0450", v) && v == 45);
    CHECK(!parseMilli("0.0005", v));
    CHECK(!parseMilli("abc", v));
    CHECK(!parseMilli("", v));
    CHECK(!parseMilli("1.2.3", v));

    // 세트 1: SEMES 1차와 같은 장수 (Y 80, Θ 200)
    const auto ys = planY(2000, 50);
    const auto ts = planT(1000, 10);
    CHECK(ys.size() == 80);
    CHECK(ts.size() == 200);
    CHECK(ys.front().stem == "Y-2.00" && ys.back().stem == "Y+2.00" && ys.front().dir == "Y");
    CHECK(ts.front().stem == "T-1.00" && ts.back().stem == "T+1.00" && ts.front().dir == "Theta");
    CHECK(ys.front().rel.t_mdeg == 0 && ts.front().rel.y_um == 0);

    // 김성현 dataset.cpp 의 라벨 정규식과 맞는지
    const std::regex re_mixed(R"(^Y([+-]?\d+(?:\.\d+)?)\s+T([+-]?\d+(?:\.\d+)?)\s*\(\d+\)$)");
    const std::regex re_y(R"(^Y([+-]?\d+(?:\.\d+)?)$)"), re_t(R"(^T([+-]?\d+(?:\.\d+)?)$)");
    std::set<std::string> names;
    for (const auto &s : ys) {
        CHECK(std::regex_match(s.stem, re_y));
        names.insert(s.stem);
    }
    for (const auto &s : ts) {
        CHECK(std::regex_match(s.stem, re_t));
        names.insert(s.stem);
    }
    CHECK(names.size() == 280);  // 이름 중복 없음

    // 세트 2
    const auto mixed = planMixed(defaultMixedPoses(), 3);
    CHECK(mixed.size() == 60);
    CHECK(mixed.front().dir == "Y+0.05 T+0.01");
    CHECK(mixed.front().stem == "Y+0.05 T+0.01 (1)");
    for (const auto &s : mixed) CHECK(std::regex_match(s.stem, re_mixed));

    // 격자: Y ±2 / 0.1 (41) × Θ ±1 / 0.05 (41) - 센터 = 1680, 'ㄹ'자 순서
    const auto grid = planGrid(2000, 100, 1000, 50);
    CHECK(grid.size() == 41 * 41 - 1);
    CHECK(grid.front().dir == "Grid");
    CHECK(grid.front().stem == "Y-2.00 T-1.00 (1)");
    CHECK(grid[40].rel.y_um == 2000 && grid[40].rel.t_mdeg == -1000);    // 첫 줄 끝: Y 오름차순
    CHECK(grid[41].rel.y_um == 2000 && grid[41].rel.t_mdeg == -950);     // 둘째 줄 시작: 같은 Y 에서 Θ 한 칸
    CHECK(grid[81].rel.y_um == -2000 && grid[81].rel.t_mdeg == -950);    // 둘째 줄 끝: Y 내림차순
    CHECK(grid.back().rel.t_mdeg == 1000);
    std::set<std::string> gnames;
    bool center_in_grid = false;
    for (size_t k = 0; k < grid.size(); ++k) {
        CHECK(std::regex_match(grid[k].stem, re_mixed));
        gnames.insert(grid[k].stem);
        if (grid[k].rel.y_um == 0 && grid[k].rel.t_mdeg == 0) center_in_grid = true;
        if (k > 0) {   // 이웃한 두 장은 한 축으로 한 칸만 (줄 바꿈 포함) — 센터를 뺀 자리만 두 칸
            const int32_t dy = std::abs(grid[k].rel.y_um - grid[k - 1].rel.y_um);
            const int32_t dt = std::abs(grid[k].rel.t_mdeg - grid[k - 1].rel.t_mdeg);
            CHECK((dy == 100 && dt == 0) || (dy == 0 && dt == 50) || (dy == 200 && dt == 0 && grid[k].rel.t_mdeg == 0));
        }
    }
    CHECK(!center_in_grid);
    CHECK(gnames.size() == grid.size());
    CHECK(planGrid(0, 100, 1000, 50).size() == 40);    // Y 범위 0 = Θ 한 줄
    CHECK(planGrid(2000, 0, 1000, 50).empty());

    // CSV
    std::vector<Rel> poses;
    std::string err;
    CHECK(parsePoseCsv("y_mm,t_deg\n0.5, 0.1\n# 주석\n\n-1,0.02\n", poses, err));
    CHECK(poses.size() == 2 && poses[0].y_um == 500 && poses[0].t_mdeg == 100 && poses[1].y_um == -1000);
    poses.clear();
    CHECK(!parsePoseCsv("0.5,0.1\nxx\n", poses, err));
    poses.clear();
    CHECK(!parsePoseCsv("# 비어 있음\n", poses, err));

    // 이동 순서
    std::vector<ShotSpec> order = {{{500, 100}, "", "a"}, {{-500, 100}, "", "b"}, {{0, -100}, "", "c"}};
    sortForTravel(order);
    CHECK(order[0].stem == "c" && order[1].stem == "b" && order[2].stem == "a");

    if (failures == 0) std::printf("plan_test 통과\n");
    return failures == 0 ? 0 : 1;
}
