// QR과 나사 이동량을 판의 이동량으로 바꾸는 계산식 생성과 저장
#include "coefficient_fit.hpp"
#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>

namespace tools {

const char* const kScrewSetNames[16] = {"1234", "234", "134", "124", "123", "12", "13", "14", "23", "24", "34", "1", "2", "3", "4", "none"};
const char* const kCheckSetNames[11] = {"1234", "234", "134", "124", "123", "12", "13", "14", "23", "24", "34"};

namespace {

const int kHuberRounds = 6;   // 오차가 큰 사진의 비중 조정 횟수
const double kRidgePerPhoto = 1e-7;   // 곱셈값이 과도하게 커지는 것을 줄일 보정량
const double kRobustScale = 1.4826;          // 중앙값 기준 흔들림을 일반적인 오차 크기로 환산하는 배율
const double kThresholdQuantile = 0.999;   // 이상 판정 기준으로 삼을 학습 점검값의 위치 비율
const double kThresholdMargin = 1.5;   // 학습 점검값에 곱할 여유 배율
const int kCheckTerms = 6;                   // 예상 이동량 계산에 쓰는 6가지 값: 1, Y, T, Y², YT, T²

// median_of: 크기순으로 정렬한 가운데 값 구하기
//   입력: v (비어 있지 않은 숫자 목록)
//   출력: 가운데 값. 짝수 개이면 가운데 두 값의 평균
double median_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// quantile_of: 정렬한 값에서 지정 비율의 경계 구하기
//   입력: v (숫자 목록), q (위치 비율, 0~1)
//   출력: 경계값. 예: q = 0.999는 아래 99.9%와 위 0.1%의 경계
double quantile_of(std::vector<double> v, double q) {
    std::sort(v.begin(), v.end());
    const double position = q * (v.size() - 1);
    const size_t low = size_t(std::floor(position));
    return low + 1 < v.size() ? v[low] + (position - low) * (v[low + 1] - v[low]) : v[low];
}

// used_values_of: 조합의 측정값 번호 선택
//   입력: screws (나사 조합)
//   출력: 계산에 사용할 번호. 0은 QR, 1~4는 나사
std::vector<int> used_values_of(const std::string& screws) {
    std::vector<int> used = {0};
    if (screws == "none") return used;
    for (char c : screws) used.push_back(c - '0');
    return used;
}

// standardize: 측정값의 크기 차이를 맞출 기준 계산
//   입력: used (사용할 QR과 나사 번호), values (사진별 이동량)
//   출력: 없음 (set에 항목별 평균과 퍼진 정도 저장)
void standardize(alignment::CoefficientSet& set, const std::vector<int>& used, const std::vector<Values>& values) {
    const int n = int(values.size());
    for (int v : used) {
        double sum = 0, square = 0;
        for (const Values& one : values) sum += one[v];
        set.mean[v] = sum / n;
        for (const Values& one : values) square += std::pow(one[v] - set.mean[v], 2);
        set.scale[v] = std::max(std::sqrt(square / n), 1e-6);
    }
}

// solve_huber: 오차가 큰 사진의 영향을 줄여 계산식 구하기
//   입력: design (사진별 계산 자료), truth_y와 truth_t (정답 이동량)
//   출력: Y와 T 계산용 숫자 표. 첫 행은 기본값, 나머지는 곱셈값
cv::Mat solve_huber(const cv::Mat& design, const std::vector<double>& truth_y, const std::vector<double>& truth_t) {
    const int n = design.rows;
    // 1. 사진별 정답 Y와 T 준비
    cv::Mat truth(n, 2, CV_64F);
    for (int i = 0; i < n; ++i) {
        truth.at<double>(i, 0) = truth_y[i];
        truth.at<double>(i, 1) = truth_t[i];
    }
    // 2. 계산용 곱셈값이 과도하게 커지는 것을 줄일 보정 준비
    cv::Mat ridge = cv::Mat::eye(design.cols, design.cols, CV_64F) * (kRidgePerPhoto * n);
    ridge.at<double>(0, 0) = 0;   // 기본값은 곱셈값 크기 제한에서 제외

    std::vector<double> weight(n, 1.0);
    cv::Mat coefficients;
    for (int round = 0; round < kHuberRounds; ++round) {
        cv::Mat weighted = design.clone();
        for (int i = 0; i < n; ++i) weighted.row(i) *= weight[i];
        // 3. 사진별 반영 비중으로 이동량 계산식 구하기
        cv::solve(weighted.t() * design + ridge, weighted.t() * truth, coefficients, cv::DECOMP_SVD);
        const cv::Mat predicted = design * coefficients;
        // 4. 정답과 차이가 큰 사진의 반영 비중 축소
        double weight_sum = 0;
        for (int i = 0; i < n; ++i) {
            const double badness = std::max(std::abs(predicted.at<double>(i, 0) - truth_y[i]) / kToleranceYmm,
                                            std::abs(predicted.at<double>(i, 1) - truth_t[i]) / kToleranceTdeg);
            weight[i] = std::min(1.0, 1.0 / std::max(badness, 1e-12));
            weight_sum += weight[i];
        }
        for (double& w : weight) w *= n / weight_sum;
    }
    return coefficients;
}

// screws_text: 사용한 나사 번호 표시
//   입력: screws (사용할 나사 번호)
//   출력: 표시 문자열. 예: QR + 나사 1, 3, 4
std::string screws_text(const std::string& screws) {
    if (screws == "none") return "QR 만";
    std::string text = "QR + 나사 ";
    for (size_t i = 0; i < screws.size(); ++i) text += (i ? ", " : "") + std::string(1, screws[i]);
    return text;
}

}   // 파일 내부

// passes: 두 방향의 오차가 허용 범위인지 확인
//   입력: error_y_mm (Y 오차), error_t_deg (T 오차)
//   출력: 두 오차 모두 허용 범위 안이면 true
bool passes(double error_y_mm, double error_t_deg) {
    return std::abs(error_y_mm) <= kToleranceYmm && std::abs(error_t_deg) <= kToleranceTdeg;
}

// has_agreement_check: 이상 여부 확인 대상인지 조회
//   입력: screws (사용할 나사 번호)
//   출력: 대상 조합이면 true
bool has_agreement_check(const std::string& screws) {
    return std::find(std::begin(kCheckSetNames), std::end(kCheckSetNames), screws) != std::end(kCheckSetNames);
}

// fit_coefficient_set: 사진별 정답에 맞는 이동량 계산식 생성
//   입력: screws (사용 나사), values (측정값), truth_y와 truth_t (사진별 정답)
//   출력: CoefficientSet (Y와 T 계산용 숫자)
alignment::CoefficientSet fit_coefficient_set(const std::string& screws, const std::vector<Values>& values,
                                              const std::vector<double>& truth_y, const std::vector<double>& truth_t) {
    alignment::CoefficientSet set;
    set.screws = screws;
    const std::vector<int> used = used_values_of(screws);
    for (int v : used)
        if (v >= 1) set.uses_screw[v - 1] = true;
    set.precise = (set.uses_screw[0] || set.uses_screw[2]) && (set.uses_screw[1] || set.uses_screw[3]);   // 왼쪽(1, 3)과 오른쪽(2, 4)이 모두 있음
    // 1. 측정 항목마다 다른 값의 크기 맞추기
    standardize(set, used, values);

    const int n = int(values.size()), p = int(used.size());
    cv::Mat design(n, p + 1, CV_64F);
    for (int i = 0; i < n; ++i) {
        design.at<double>(i, 0) = 1;
        for (int j = 0; j < p; ++j) design.at<double>(i, j + 1) = (values[i][used[j]] - set.mean[used[j]]) / set.scale[used[j]];
    }
    // 2. 정답과의 차이를 줄이는 곱셈값과 기본값 구하기
    const cv::Mat coefficients = solve_huber(design, truth_y, truth_t);
    // 3. 계산 결과를 나사 조합별로 보관
    set.intercept_y = coefficients.at<double>(0, 0);
    set.intercept_t = coefficients.at<double>(0, 1);
    for (int j = 0; j < p; ++j) {
        set.weight_y[used[j]] = coefficients.at<double>(j + 1, 0);
        set.weight_t[used[j]] = coefficients.at<double>(j + 1, 1);
    }
    return set;
}

// predict: 사진 속 이동량으로 판의 이동량 계산
//   입력: set (이동량 계산식), values (QR과 나사 이동량)
//   출력: YT (Y 이동량 mm, T 회전량 도)
alignment::YT predict(const alignment::CoefficientSet& set, const Values& values) {
    return alignment::compute_yt(set, values.data());
}

// fit_agreement_check: QR과 나사 측정값의 이상 여부 확인식 생성
//   입력: set (이동량 계산식), values (학습 사진의 측정값)
//   출력: AgreementCheck (예상 이동량 계산식과 이상 판정 기준)
alignment::AgreementCheck fit_agreement_check(const alignment::CoefficientSet& set, const std::vector<Values>& values) {
    const int n = int(values.size());
    // 1. 사진별 Y와 T로 예상 이동량 계산 자료 준비
    cv::Mat design(n, kCheckTerms, CV_64F);
    for (int i = 0; i < n; ++i) {
        const alignment::YT pose = predict(set, values[i]);
        const double terms[kCheckTerms] = {1, pose.y_mm, pose.t_deg, pose.y_mm * pose.y_mm, pose.y_mm * pose.t_deg, pose.t_deg * pose.t_deg};
        for (int q = 0; q < kCheckTerms; ++q) design.at<double>(i, q) = terms[q];
    }
    alignment::AgreementCheck check;
    check.screws = set.screws;
    std::vector<double> score(n, 0);   // 사진마다 점검값 = 가장 크게 어긋난 값의 (어긋난 양 ÷ 흔들림)
    // 2. QR과 나사마다 예상 이동량 계산식 구하기
    for (int v : used_values_of(set.screws)) {
        cv::Mat target(n, 1, CV_64F), solved;
        for (int i = 0; i < n; ++i) target.at<double>(i) = values[i][v];
        cv::solve(design, target, solved, cv::DECOMP_SVD);
        alignment::AgreementCheck::Row row;
        row.value_index = v;
        for (int q = 0; q < kCheckTerms; ++q) row.c[q] = solved.at<double>(q);
        std::vector<double> residual(n), spread(n);
        for (int i = 0; i < n; ++i) {
            double expected = 0;
            for (int q = 0; q < kCheckTerms; ++q) expected += row.c[q] * design.at<double>(i, q);
            residual[i] = values[i][v] - expected;
        }
        // 3. 실제값과 예상값 사이의 평소 흔들림 계산
        const double middle = median_of(residual);
        for (int i = 0; i < n; ++i) spread[i] = std::abs(residual[i] - middle);
        row.usual_spread = kRobustScale * median_of(spread);
        for (int i = 0; i < n; ++i) score[i] = std::max(score[i], std::abs(residual[i]) / row.usual_spread);
        check.rows.push_back(row);
    }
    // 4. 평소 흔들림보다 큰 차이를 가려낼 기준 설정
    check.limit = kThresholdMargin * quantile_of(score, kThresholdQuantile);
    return check;
}

// write_coefficient_file: 이동량 계산용 숫자 저장
//   입력: path (저장 경로), set (계산식), photo_count (학습 사진 수)
//   출력: 없음 (탭으로 열을 나눈 TSV 파일 저장)
void write_coefficient_file(const std::filesystem::path& path, const alignment::CoefficientSet& set, int photo_count) {
    std::ofstream file(path, std::ios::binary);
    file << std::setprecision(17);
    file << "# Y와 T 이동량 계산용 숫자. 사용 항목: " << screws_text(set.screws) << ". 학습 사진 " << photo_count << "장 (오차가 큰 사진의 영향 축소).\n"
         << "# mean: 평균, scale: 값의 퍼진 정도, weight: 곱셈값, intercept: 기본값\n"
         << "name\tmean\tscale\tweight_Y\tweight_T\n"
         << "intercept\t0\t1\t" << set.intercept_y << '\t' << set.intercept_t << '\n';
    for (int v : used_values_of(set.screws))
        file << kValueNames[v] << '\t' << set.mean[v] << '\t' << set.scale[v] << '\t' << set.weight_y[v] << '\t' << set.weight_t[v] << '\n';
}

// write_agreement_file: 이상 여부 확인용 숫자 저장
//   입력: path (저장 경로), check (확인식), photo_count (학습 사진 수)
//   출력: 없음 (탭으로 열을 나눈 TSV 파일 저장)
void write_agreement_file(const std::filesystem::path& path, const alignment::AgreementCheck& check, int photo_count) {
    std::ofstream file(path, std::ios::binary);
    file << std::setprecision(17);
    file << "# QR과 나사별 예상 이동량 계산용 숫자. Y와 T는 영점 차감 전 값. 학습 사진 " << photo_count << "장\n"
         << "# scale: 평소 흔들림 크기. threshold: 실제 이동량과 예상 이동량의 차이가 너무 큰지 판단할 기준.\n"
         << "threshold\t" << check.limit << '\n'
         << "name\tc1\tcY\tcT\tcYY\tcYT\tcTT\tscale\n";
    for (const alignment::AgreementCheck::Row& row : check.rows) {
        file << kValueNames[row.value_index];
        for (double c : row.c) file << '\t' << c;
        file << '\t' << row.usual_spread << '\n';
    }
}

}   // tools
