// 예제: 사진 한 장이 단계마다 어떤 값을 내는지 모두 찍는다.
// 코드를 다른 곳으로 옮길 때, 옮긴 코드의 값과 한 단계씩 비교하는 데 쓴다 (docs/2_포팅안내서.md).
// 사용: show_steps <model 폴더> <사진>
#include "alignment.hpp"
#include "common/reference.hpp"
#include "steps/01_find_qr.hpp"
#include "steps/02_find_screws.hpp"
#include "steps/03_compute_yt.hpp"
#include "steps/04_check_result.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>

using namespace alignment;

int main(int argc, char** argv) {
    if (argc != 3) { std::cerr << "사용: show_steps <model 폴더> <사진>\n"; return 2; }
    const std::string data = argv[1];
    std::string error;
    Reference reference;
    std::vector<CoefficientSet> sets;
    std::vector<AgreementCheck> checks;
    cv::Mat gray;
    if (!load_reference(data, reference, error) || !load_coefficient_sets(data, sets, error) || !load_agreement_checks(data, checks, error) ||
        !to_gray(alignment::read_photo(argv[2]), gray, error)) {
        std::cerr << error << "\n";
        return 1;
    }
    std::cout << std::fixed << std::setprecision(6);

    // ---- 1단계 ----
    const QrFinder qr_finder(reference);
    std::cout << "[1단계] find_qr_shift\n";
    for (int k = 0; k < 3; ++k) {
        const QrFinder::Match match = qr_finder.find_one(gray, k, kNarrowHalfWidth);
        std::cout << "  QR " << k + 1 << " (창 ±" << kNarrowHalfWidth << "): 찾음 " << match.found << ", dx " << match.dx << ", dy " << match.dy << ", 닮은 정도 " << match.similarity << "\n";
    }
    const QrShift qr = qr_finder.find_qr_shift(gray);
    std::cout << "  결과: dx " << qr.dx << ", dy " << qr.dy << ", 찾은 QR " << qr.found_count << "개, 탐색 단계 " << int(qr.search) << " (0 좁게, 1 넓게, 2 전체 폭, 3 일부, 4 못 찾음)\n";
    if (qr.search == QrSearch::NotFound) return 1;

    // ---- 2단계 ----
    const ScrewFinder screw_finder(reference);
    const cv::Point expected(int(std::lround(qr.dx)), int(std::lround(qr.dy)));
    std::cout << "[2단계] find_screw_shifts  (QR 이동을 반올림한 예상 이동: " << expected.x << ", " << expected.y << ")\n";
    double values[5] = {qr.dx, 0, 0, 0, 0};
    std::array<bool, 4> found{};
    for (int k = 0; k < 4; ++k) {
        cv::Point located;
        double similarity = 0, rough_dx = 0, rough_dy = 0, dx = 0, dy = 0;
        std::cout << "  나사 " << k + 1 << ":";
        if (!screw_finder.locate_screw(gray, k, expected, located, similarity)) { std::cout << " locate_screw 실패\n"; continue; }
        std::cout << " locate_screw (" << located.x << ", " << located.y << ") 닮은 정도 " << similarity;
        if (!screw_finder.refine_by_brightness(gray, k, located, rough_dx, rough_dy)) { std::cout << " | refine_by_brightness 실패\n"; continue; }
        std::cout << " | refine_by_brightness (" << rough_dx << ", " << rough_dy << ")";
        if (!screw_finder.refine_by_edges(gray, k, cv::Point(cvRound(rough_dx), cvRound(rough_dy)), dx, dy)) { std::cout << " | refine_by_edges 실패\n"; continue; }
        std::cout << " | refine_by_edges (" << dx << ", " << dy << ")\n";
        found[k] = true;
        values[k + 1] = dx;
    }

    // ---- 3단계 ----
    const CoefficientSet* set = choose_coefficient_set(sets, found);
    std::cout << "[3단계] compute_yt  (계수 조합 screws_" << set->screws << ")\n";
    const char* names[5] = {"QR", "나사 1", "나사 2", "나사 3", "나사 4"};
    for (int i = 0; i < 5; ++i) {
        if (set->weight_y[i] == 0 && set->weight_t[i] == 0) continue;
        const double normalized = (values[i] - set->mean[i]) / set->scale[i];
        std::cout << "  " << names[i] << ": 값 " << values[i] << " → 정규화 " << normalized << " → Y 에 " << set->weight_y[i] * normalized << ", T 에 " << set->weight_t[i] * normalized << "\n";
    }
    const YT yt = compute_yt(*set, values);
    std::cout << "  절편 Y " << set->intercept_y << ", T " << set->intercept_t << "\n";
    std::cout << "  결과: Y " << yt.y_mm << " mm, T " << yt.t_deg << " 도  (영점을 빼기 전)\n";

    // ---- 4단계 ----
    std::cout << "[4단계] check_values_agree, check_in_trained_range\n";
    if (const AgreementCheck* check = find_agreement_check(checks, set->screws)) {
        double differences[5];
        expected_differences(*check, values, yt.y_mm, yt.t_deg, differences);
        for (const AgreementCheck::Row& row : check->rows)
            std::cout << "  " << names[row.value_index] << ": 측정값 − 기대값 " << differences[row.value_index] << " px, 평소 흔들림 " << row.usual_spread << " → " << std::abs(differences[row.value_index]) / row.usual_spread << "\n";
        const double no_baseline[5] = {};
        const AgreementResult agreement = check_values_agree(*check, values, yt.y_mm, yt.t_deg, no_baseline);
        std::cout << "  점검값 " << agreement.value << " (기준 " << agreement.limit << ") → " << (agreement.agrees() ? "서로 맞음" : "서로 맞지 않음") << "\n";
    }
    std::cout << "  학습 범위 안: " << (check_in_trained_range(reference, yt.y_mm, yt.t_deg) ? "예" : "아니오") << "\n";
    return 0;
}
