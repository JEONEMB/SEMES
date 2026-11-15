// 단계별 측정값 출력과 이식 결과 비교
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

namespace {

const char* const kValueLabels[kValueCount] = {"QR", "나사 1", "나사 2", "나사 3", "나사 4",
                                              "나사 1 둘레", "나사 2 둘레", "나사 3 둘레", "나사 4 둘레"};   // 화면에 출력하는 항목 이름

// print_qr_step: 1단계 값 표시
//   입력: qr_finder (기준 QR 탐색 도구), gray (회색 사진)
//   출력: QrShift, 대략 봉우리와 평균 이동 출력
QrShift print_qr_step(const QrFinder& qr_finder, const cv::Mat& gray) {
    std::cout << "[1단계] find_qr_shift\n";
    std::cout << "  대략: QR 띠 봉우리 (높은 순, 문턱 " << kPeakThreshold << ")\n";
    for (const QrFinder::Peak& peak : qr_finder.find_peaks(gray))
        std::cout << "    x " << peak.x << ", y " << peak.y << ", 닮은 정도 " << peak.similarity << (peak.similarity >= kPeakThreshold ? "  (QR)" : "") << "\n";
    const QrShift qr = qr_finder.find_qr_shift(gray);
    for (int k = 0; k < 3; ++k)
        if (qr.qr[k].found) std::cout << "  정밀 QR " << k + 1 << " (대략 위치 둘레 ±" << kFineHalfWidth << " px): dx " << qr.qr[k].dx << ", dy " << qr.qr[k].dy << ", 닮은 정도 " << qr.qr[k].similarity << "\n";
    if (!qr.found) {
        std::cout << "  결과: 실패, " << qr.problem << "\n";
        return qr;
    }
    std::cout << "  결과: dx " << qr.dx << ", dy " << qr.dy << ", 정밀 닮은 정도 최소 " << qr.lowest_similarity << "\n";
    return qr;
}

// print_screw_step: 2단계 값 표시
//   입력: screw_finder (나사 탐색 도구), gray (회색 사진), qr (QR 이동 결과)
//   출력: 나사별 검출 여부, values[1~4] 테두리와 values[5~8] 둘레 영역 이동량 저장, 단계별 값 표시
std::array<bool, 4> print_screw_step(const ScrewFinder& screw_finder, const cv::Mat& gray, const QrShift& qr, double values[kValueCount]) {
    const cv::Point expected(int(std::lround(qr.dx)), int(std::lround(qr.dy)));
    std::cout << "[2단계] find_screw_shifts  (QR 이동을 반올림한 예상 이동: " << expected.x << ", " << expected.y << ")\n";
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
        std::cout << " | refine_by_edges (" << dx << ", " << dy << ")";
        double area_dx = 0, area_dy = 0;
        if (!screw_finder.refine_by_area(gray, k, cv::Point(cvRound(dx), cvRound(dy)), area_dx, area_dy)) { std::cout << " | refine_by_area 실패\n"; continue; }
        std::cout << " | refine_by_area (" << area_dx << ", " << area_dy << ")\n";
        found[k] = true;
        values[k + 1] = dx;
        values[kAreaValueOffset + k] = area_dx;
    }
    return found;
}

// print_yt_step: 3단계 값 표시
//   입력: set (이동량 계산식), values (QR과 나사 이동량)
//   출력: YT, 각 측정값이 Y와 T에 더한 양 표시
YT print_yt_step(const CoefficientSet& set, const double values[kValueCount]) {
    std::cout << "[3단계] compute_yt  (계수 조합 screws_" << set.screws << ")\n";
    for (int i = 0; i < kValueCount; ++i) {
        if (set.weight_y[i] == 0 && set.weight_t[i] == 0) continue;
        const double normalized = (values[i] - set.mean[i]) / set.scale[i];
        std::cout << "  " << kValueLabels[i] << ": 값 " << values[i] << " → 정규화 " << normalized << " → Y 에 " << set.weight_y[i] * normalized << ", T 에 " << set.weight_t[i] * normalized << "\n";
    }
    const YT yt = compute_yt(set, values);
    std::cout << "  절편 Y " << set.intercept_y << ", T " << set.intercept_t << "\n";
    std::cout << "  결과: Y " << yt.y_mm << " mm, T " << yt.t_deg << " 도  (영점을 빼기 전)\n";
    return yt;
}

// print_check_step: 4단계 값 표시
//   입력: check (점검식, 없으면 nullptr), reference (기준), values (이동량), yt (계산한 Y와 T)
//   출력: 없음 (실제값과 예상값의 차이, 점검값, 학습 범위 출력)
void print_check_step(const AgreementCheck* check, const Reference& reference, const double values[kValueCount], const YT& yt) {
    std::cout << "[4단계] check_values_agree, check_in_trained_range\n";
    if (check) {
        double differences[kValueCount];
        expected_differences(*check, values, yt.y_mm, yt.t_deg, differences);
        for (const AgreementCheck::Row& row : check->rows)
            std::cout << "  " << kValueLabels[row.value_index] << ": 측정값 − 기대값 " << differences[row.value_index] << " px, 평소 흔들림 " << row.usual_spread << " → "
                      << std::abs(differences[row.value_index]) / row.usual_spread << "\n";
        const double no_baseline[kValueCount] = {};
        const AgreementResult agreement = check_values_agree(*check, values, yt.y_mm, yt.t_deg, no_baseline);
        std::cout << "  점검값 " << agreement.value << " (기준 " << agreement.limit << ") → " << (agreement.agrees() ? "서로 맞음" : "서로 맞지 않음") << "\n";
    }
    std::cout << "  학습 범위 안: " << (check_in_trained_range(reference, yt.y_mm, yt.t_deg) ? "예" : "아니오") << "\n";
}

}   // 파일 내부

// main: 단계별 측정값 출력과 이식 결과 비교
//   입력: 실행할 때 지정한 경로와 옵션(파일 상단 사용법)
//   출력: 실행 결과와 성공 여부를 나타내는 종료 코드
int main(int argc, char** argv) {
    if (argc != 3) { std::cerr << "사용: show_steps <model 폴더> <사진>\n"; return 2; }
    const std::string model_folder = argv[1];
    std::string error;
    Reference reference;
    std::vector<CoefficientSet> sets;
    std::vector<AgreementCheck> checks;
    cv::Mat gray;
    if (!load_reference(model_folder, reference, error) || !load_coefficient_sets(model_folder, sets, error) || !load_agreement_checks(model_folder, checks, error) ||
        !to_gray(alignment::read_photo(argv[2]), gray, error)) {
        std::cerr << error << "\n";
        return 1;
    }
    std::cout << std::fixed << std::setprecision(6);

    const QrShift qr = print_qr_step(QrFinder(reference), gray);
    if (!qr.found) return 1;
    double values[kValueCount] = {qr.dx};
    const std::array<bool, 4> found = print_screw_step(ScrewFinder(reference), gray, qr, values);
    const CoefficientSet* set = choose_coefficient_set(sets, found);
    const YT yt = print_yt_step(*set, values);
    print_check_step(find_agreement_check(checks, set->screws), reference, values, yt);
    return 0;
}
