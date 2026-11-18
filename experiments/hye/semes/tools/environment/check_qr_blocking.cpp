#include "common/console.hpp"
// QR 가림과 반사에 대한 판정 규칙 비교
// 사용: check_qr_blocking <model 폴더> <tables 폴더> <출력 폴더>
// 출력: qr_blocking_report.txt, qr_blocking_cases.tsv, block_<사례>.png
// 1부: 실제 사진. QR 3개 이동량의 차이, QR 일부만 사용한 계산의 정확도, 탐색 시간
// 2부: 합성 가림. 현재 판정(띠 봉우리 방식)과, 이전 방식 탐색 결과에 대안 판정 5가지를 적용한 결과
// 사진 경로: tables/photos.tsv 의 경로 (패키지 폴더에서 실행)
#include "alignment.hpp"
#include "common/dataset.hpp"
#include "common/progress.hpp"
#include "common/tables.hpp"
#include "scene_effects.hpp"
#include "steps/01_find_qr.hpp"
#include "steps/02_find_screws.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>

namespace fs = std::filesystem;
using namespace alignment;
using tools::DatasetPhoto;
using tools::Values;

namespace {

const double kAgreeLimitPx = 1.0;      // QR끼리 이동량이 같다고 보는 한계(px). 1부의 정상 사진 최대 차이의 약 3배
const double kTrustedSimilarity = 0.95;   // 이보다 덜 닮은 QR은 가려진 것으로 보고 뺌. 1부의 정상 사진 최소값은 0.9855
const double kWrongPlacePx = 20.0;     // 가리기 전 위치와 이만큼 다르면 엉뚱한 자리로 판정
const int kBlockPhotoStep = 12;        // 2부에 쓰는 사진 간격 (usable 사진 12장마다 1장)
const int kCoverBrightness = 55;       // 가리는 물체의 밝기 (make_environment_images의 로봇 팔과 같은 값)
const double kGlareStrength = 230;     // 반사광 세기 (make_environment_images와 같은 값)

std::ostringstream report_text;

// print: 화면과 보고서에 한 줄 기록
//   입력: line (내용)
//   출력: 없음
void print(const std::string& line) { std::cout << line << "\n"; report_text << line << "\n"; }

// fixed: 소수 자릿수를 정한 숫자 표시
//   입력: value (숫자), digits (소수 자릿수)
//   출력: 숫자 문자열
std::string fixed(double value, int digits) { std::ostringstream o; o << std::fixed << std::setprecision(digits) << value; return o.str(); }

// percent: 비율 표시
//   입력: count (해당 수), total (전체 수)
//   출력: 백분율 문자열. 예: 96.4%
std::string percent(int count, int total) { return fixed(total ? 100.0 * count / total : 0, 1) + "%"; }

// median_of: 가운데값
//   입력: values (숫자 목록, 비어 있지 않음)
//   출력: 가운데값
double median_of(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

// milliseconds_since: 경과 시간
//   입력: start (시작 시각)
//   출력: 경과 시간(ms)
double milliseconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

//
// 이전 방식 QR 탐색 (2026-11-15 까지 패키지에서 쓰던 방식, 비교용): QR 마다 따로, ±110 -> ±250 -> 전체 폭
//

const int kPreviousNarrowHalfWidth = 110;   // 이전 방식 기본 탐색 반너비(px)
const int kPreviousWideHalfWidth = 250;     // 이전 방식 확대 탐색 반너비(px)
const int kPreviousHalfHeight = 22;         // 이전 방식 세로 탐색 반높이(px)

// PreviousQrFinder: 이전 방식의 QR 하나 탐색 (흐림 σ2, NCC, 문턱 0.8, 3점 포물선)
class PreviousQrFinder {
public:
    explicit PreviousQrFinder(const Reference& reference) {
        cv::GaussianBlur(reference.photo, blurred_reference_, cv::Size(0, 0), 2.0);
        for (int k = 0; k < 3; ++k) boxes_[k] = reference.qr[k];
    }
    // find_one: QR 하나 탐색
    //   입력: gray (회색 사진), qr_index (0~2), half_width (가로 반너비 px, 0이면 전체 폭)
    //   출력: Match (검출 여부, 이동량, 닮은 정도)
    QrFinder::Match find_one(const cv::Mat& gray, int qr_index, int half_width) const {
        QrFinder::Match match;
        const cv::Rect box = boxes_[qr_index];
        const int left = half_width > 0 ? box.x - half_width : 0, right = half_width > 0 ? box.x + box.width + half_width : kPhotoWidth;
        const cv::Rect window = cv::Rect(left, box.y - kPreviousHalfHeight, right - left, box.height + 2 * kPreviousHalfHeight) & kPhotoRect;
        const cv::Rect wider = cv::Rect(window.x - 8, window.y - 8, window.width + 16, window.height + 16) & kPhotoRect;
        cv::Mat blurred, similarity;
        cv::GaussianBlur(gray(wider), blurred, cv::Size(0, 0), 2.0);
        cv::matchTemplate(blurred(window - wider.tl()), blurred_reference_(box), similarity, cv::TM_CCOEFF_NORMED);
        double best = 0;
        cv::Point p;
        cv::minMaxLoc(similarity, nullptr, &best, nullptr, &p);
        if (best < kMinSimilarity || p.x <= 0 || p.y <= 0 || p.x >= similarity.cols - 1 || p.y >= similarity.rows - 1) return match;
        const auto offset = [](double a, double b, double c) { const double curvature = a - 2 * b + c; return curvature < 0 ? 0.5 * (a - c) / curvature : 0.0; };
        match.found = true;
        match.dx = window.x + p.x + offset(similarity.at<float>(p.y, p.x - 1), similarity.at<float>(p.y, p.x), similarity.at<float>(p.y, p.x + 1)) - box.x;
        match.dy = window.y + p.y + offset(similarity.at<float>(p.y - 1, p.x), similarity.at<float>(p.y, p.x), similarity.at<float>(p.y + 1, p.x)) - box.y;
        match.similarity = best;
        return match;
    }
private:
    cv::Mat blurred_reference_;
    cv::Rect boxes_[3];
};

struct QrMatches {
    QrFinder::Match match[3];

    // found_count: 검출 QR 수
    int found_count() const { return int(match[0].found) + int(match[1].found) + int(match[2].found); }
};

// find_three: 같은 범위에서 QR 3개 탐색 (이전 방식)
//   입력: finder (이전 방식 탐색기), gray (회색 사진), half_width (가로 반너비 px, 0이면 전체 폭)
//   출력: QR별 결과
QrMatches find_three(const PreviousQrFinder& finder, const cv::Mat& gray, int half_width) {
    QrMatches matches;
    for (int k = 0; k < 3; ++k) matches.match[k] = finder.find_one(gray, k, half_width);
    return matches;
}

// find_previous_way: 이전 방식 순서로 범위 확대
//   입력: finder (이전 방식 탐색기), gray (회색 사진)
//   출력: 마지막 탐색 범위의 QR별 결과
QrMatches find_previous_way(const PreviousQrFinder& finder, const cv::Mat& gray) {
    QrMatches matches = find_three(finder, gray, kPreviousNarrowHalfWidth);
    if (matches.found_count() == 3) return matches;
    matches = find_three(finder, gray, kPreviousWideHalfWidth);
    if (matches.found_count() == 3) return matches;
    return find_three(finder, gray, 0);
}
// agreeing_group: 이동량이 서로 같은 QR 묶음 중 가장 큰 것
//   입력: matches (QR별 결과)
//   출력: 묶음에 든 QR 표시. 2개 이상이 서로 kAgreeLimitPx 안일 때만 표시, 아니면 모두 false
std::array<bool, 3> agreeing_group(const QrMatches& matches) {
    std::array<bool, 3> best{};
    int best_size = 1;
    for (int anchor = 0; anchor < 3; ++anchor) {
        if (!matches.match[anchor].found) continue;
        std::array<bool, 3> group{};
        int size = 0;
        for (int k = 0; k < 3; ++k) {
            group[k] = matches.match[k].found && std::abs(matches.match[k].dx - matches.match[anchor].dx) <= kAgreeLimitPx;
            size += group[k];
        }
        if (size > best_size) { best = group; best_size = size; }
    }
    return best;
}

// drop_less_similar: 덜 닮은 QR 제외
//   입력: matches (QR별 결과)
//   출력: 닮은 정도가 kTrustedSimilarity 이상인 QR만 남긴 결과. 남는 것이 2개 미만이면 원래 결과 그대로
QrMatches drop_less_similar(const QrMatches& matches) {
    QrMatches trusted = matches;
    for (QrFinder::Match& match : trusted.match) match.found = match.found && match.similarity >= kTrustedSimilarity;
    return trusted.found_count() >= 2 ? trusted : matches;
}

//
// QR 일부로 Y, T 계산
//

struct Model {
    Reference reference;
    std::vector<CoefficientSet> sets;
};

struct Pose { bool has_value = false; double y_mm = 0, t_deg = 0; };

// pose_from: 고른 QR의 평균 이동량으로 Y, T 계산 (영점 차감 전)
//   입력: model (기준과 계산식), screw_finder (나사 탐색기), gray (회색 사진), matches (QR별 결과), use (계산에 쓸 QR),
//         own_set (그 QR 묶음으로 만든 계산식. nullptr이면 지금 계산식. 나사 4개를 찾았을 때만 적용)
//   출력: Pose. 쓸 QR 또는 계산식이 없으면 has_value = false
Pose pose_from(const Model& model, const ScrewFinder& screw_finder, const cv::Mat& gray, const QrMatches& matches, const std::array<bool, 3>& use,
               const CoefficientSet* own_set = nullptr) {
    double dx = 0, dy = 0;
    int count = 0;
    for (int k = 0; k < 3; ++k)
        if (use[k] && matches.match[k].found) { dx += matches.match[k].dx; dy += matches.match[k].dy; ++count; }
    if (count == 0) return {};
    dx /= count; dy /= count;

    const std::array<ScrewShift, 4> screws = screw_finder.find_screw_shifts(gray, dx, dy, {true, true, true, true});
    double values[kValueCount] = {dx};
    std::array<bool, 4> found{};
    for (int k = 0; k < 4; ++k) {
        found[k] = screws[k].found;
        values[k + 1] = found[k] ? screws[k].dx : 0.0;
        values[alignment::kAreaValueOffset + k] = found[k] ? screws[k].area_dx : 0.0;
    }
    const bool four_screws = found[0] && found[1] && found[2] && found[3];
    const CoefficientSet* set = own_set && four_screws ? own_set : choose_coefficient_set(model.sets, found);
    if (!set) return {};
    const YT yt = compute_yt(*set, values);
    return {true, yt.y_mm, yt.t_deg};
}

//
// 1부: 실제 사진
//

const std::array<bool, 3> kQrSubsets[] = {{true, true, true}, {true, true, false}, {true, false, true}, {false, true, true},
                                          {true, false, false}, {false, true, false}, {false, false, true}};
const char* const kQrSubsetNames[] = {"QR 1, 2, 3 (지금)", "QR 1, 2", "QR 1, 3", "QR 2, 3", "QR 1", "QR 2", "QR 3"};
const int kQrSubsetCount = 7;

struct RealPhoto {
    const DatasetPhoto* photo = nullptr;
    double qr_dx[3] = {};                 // QR별 가로 이동량 (px)
    double subset_qr_dx[kQrSubsetCount] = {};   // 묶음별 QR 평균 이동량
    Values screw_values{};                // [1~4] 나사 이동량. [0]은 쓰지 않음
};

// refit_pass_count: 묶음의 QR 이동량으로 계산식을 다시 만들었을 때 통과 수 (5묶음 교차 검증 1회)
//   입력: photos (1부에서 잰 사진), subset (QR 묶음 번호)
//   출력: 통과 수, total (시험한 사진 수)
int refit_pass_count(const std::vector<RealPhoto>& photos, int subset, int& total) {
    int passed = 0;
    total = 0;
    for (int fold = 0; fold < tools::kFoldCount; ++fold) {
        std::vector<Values> train_values;
        std::vector<double> train_y, train_t;
        for (const RealPhoto& p : photos) {
            if (p.photo->final_check || p.photo->fold[0] == fold) continue;
            Values values = p.screw_values; values[0] = p.subset_qr_dx[subset];
            train_values.push_back(values); train_y.push_back(p.photo->label_y_mm); train_t.push_back(p.photo->label_t_deg);
        }
        const CoefficientSet set = tools::fit_coefficient_set("1234", train_values, train_y, train_t);
        for (const RealPhoto& p : photos) {
            if (p.photo->final_check || p.photo->fold[0] != fold) continue;
            Values values = p.screw_values; values[0] = p.subset_qr_dx[subset];
            const YT yt = tools::predict(set, values);
            passed += tools::passes(yt.y_mm - p.photo->label_y_mm, yt.t_deg - p.photo->label_t_deg);
            ++total;
        }
    }
    return passed;
}

// run_real_photos: 1부. 가리지 않은 실제 사진
//   입력: model (기준과 계산식), usable (학습에 쓰는 사진)
//   출력: QR 묶음별 계산식 (1부의 모든 사진으로 학습, 나사 4개 조합). 보고서 기록
std::vector<CoefficientSet> run_real_photos(const Model& model, const std::vector<const DatasetPhoto*>& usable) {
    const QrFinder qr_finder(model.reference);
    const PreviousQrFinder previous_finder(model.reference);
    const ScrewFinder screw_finder(model.reference);
    std::vector<RealPhoto> photos;
    std::vector<double> spreads, similarities, search_ms;
    int pass_with_model[kQrSubsetCount] = {}, qr_failed = 0, four_screws = 0;
    double square_y[kQrSubsetCount] = {}, square_t[kQrSubsetCount] = {};

    tools::ProgressBar bar("1부 실제 사진", int(usable.size()));
    for (const DatasetPhoto* photo : usable) {
        bar.step();
        cv::Mat gray;
        std::string error;
        if (!to_gray(read_photo(photo->path), gray, error)) continue;

        const auto start = std::chrono::steady_clock::now();
        const QrShift library_shift = qr_finder.find_qr_shift(gray);
        search_ms.push_back(milliseconds_since(start));
        qr_failed += !library_shift.found;

        const QrMatches matches = find_previous_way(previous_finder, gray);
        if (matches.found_count() != 3) continue;
        RealPhoto real;
        real.photo = photo;
        double low = 1e9, high = -1e9;
        for (int k = 0; k < 3; ++k) {
            real.qr_dx[k] = matches.match[k].dx;
            low = std::min(low, real.qr_dx[k]); high = std::max(high, real.qr_dx[k]);
            similarities.push_back(matches.match[k].similarity);
        }
        spreads.push_back(high - low);

        const std::array<ScrewShift, 4> screws = screw_finder.find_screw_shifts(gray, library_shift.dx, library_shift.dy, {true, true, true, true});
        bool all_found = true;
        for (int k = 0; k < 4; ++k) { all_found &= screws[k].found; real.screw_values[k + 1] = screws[k].dx; real.screw_values[alignment::kAreaValueOffset + k] = screws[k].area_dx; }
        if (!all_found) continue;
        ++four_screws;

        const CoefficientSet* set = choose_coefficient_set(model.sets, {true, true, true, true});
        for (int s = 0; s < kQrSubsetCount; ++s) {
            double sum = 0; int count = 0;
            for (int k = 0; k < 3; ++k) if (kQrSubsets[s][k]) { sum += real.qr_dx[k]; ++count; }
            real.subset_qr_dx[s] = sum / count;
            Values values = real.screw_values; values[0] = real.subset_qr_dx[s];
            const YT yt = compute_yt(*set, values.data());
            const double error_y = yt.y_mm - photo->label_y_mm, error_t = yt.t_deg - photo->label_t_deg;
            pass_with_model[s] += tools::passes(error_y, error_t);
            square_y[s] += error_y * error_y; square_t[s] += error_t * error_t;
        }
        photos.push_back(real);
    }
    bar.finish();

    std::sort(spreads.begin(), spreads.end());
    std::sort(similarities.begin(), similarities.end());
    print("[1부] 가리지 않은 실제 사진 " + std::to_string(usable.size()) + "장");
    print("  QR 3개의 가로 이동량 차이 (가장 큰 값 - 가장 작은 값, px): 가운데값 " + fixed(median_of(spreads), 3) + ", 99% " + fixed(spreads[size_t(0.99 * (spreads.size() - 1))], 3) +
          ", 최대 " + fixed(spreads.back(), 3) + "   (간격 검사 한계 " + fixed(kAgreeLimitPx, 1) + " px 를 넘은 사진 " +
          std::to_string(std::count_if(spreads.begin(), spreads.end(), [](double v) { return v > kAgreeLimitPx; })) + "장)");
    print("  QR 닮은 정도: 최소 " + fixed(similarities.front(), 4) + ", 1% " + fixed(similarities[size_t(0.01 * (similarities.size() - 1))], 4) + ", 가운데값 " + fixed(median_of(similarities), 4));
    print("  QR 탐색 시간 (find_qr_shift, ms): 가운데값 " + fixed(median_of(search_ms), 2) + ", QR 을 찾지 못한 사진 " + std::to_string(qr_failed) + "장");
    print("  QR 일부만 쓴 계산 (나사 4개를 찾은 " + std::to_string(four_screws) + "장, 학습 정답으로 채점)");
    print("    쓴 QR               지금 계산식 통과율   Y RMS mm   T RMS 도   계산식을 다시 만들면 (교차 검증)");
    for (int s = 0; s < kQrSubsetCount; ++s) {
        int total = 0;
        const int passed = refit_pass_count(photos, s, total);
        std::ostringstream line;
        line << "    " << kQrSubsetNames[s] << "\t" << percent(pass_with_model[s], four_screws) << "\t" << fixed(std::sqrt(square_y[s] / four_screws), 4) << "\t"
             << fixed(std::sqrt(square_t[s] / four_screws), 4) << "\t" << percent(passed, total) << " (" << passed << "/" << total << ")";
        print(line.str());
    }

    std::vector<CoefficientSet> subset_sets;
    for (int s = 0; s < kQrSubsetCount; ++s) {
        std::vector<Values> values;
        std::vector<double> truth_y, truth_t;
        for (const RealPhoto& p : photos) {
            Values one = p.screw_values; one[0] = p.subset_qr_dx[s];
            values.push_back(one); truth_y.push_back(p.photo->label_y_mm); truth_t.push_back(p.photo->label_t_deg);
        }
        subset_sets.push_back(tools::fit_coefficient_set("1234", values, truth_y, truth_t));
    }
    return subset_sets;
}

//
// 2부: 합성 가림
//

struct BlockCase {
    std::string id, name;                              // id는 파일 이름과 그림 제목용 영어
    std::function<cv::Mat(const cv::Mat&, const QrMatches&)> make;   // 입력: 원본 사진, 가리기 전 QR 위치
};

// qr_box_in_photo: 사진에서 QR이 있는 영역
//   입력: reference (기준), matches (가리기 전 QR 위치), k (QR 번호 0~2)
//   출력: 사진 속 QR 사각 영역
cv::Rect qr_box_in_photo(const Reference& reference, const QrMatches& matches, int k) {
    return reference.qr[k] + cv::Point(cvRound(matches.match[k].dx), cvRound(matches.match[k].dy));
}

// cover_left_part: QR의 왼쪽 일부를 어두운 물체로 가리기
//   입력: gray (회색 사진), box (QR 영역), fraction (가리는 비율 0~1)
//   출력: 가린 사진
cv::Mat cover_left_part(const cv::Mat& gray, const cv::Rect& box, double fraction) {
    cv::Mat out = gray.clone();
    const cv::Rect cover(box.x, box.y - 30, cvRound(box.width * fraction), box.height + 60);   // 위아래로 30 px 더 덮음
    cv::rectangle(out, cover & kPhotoRect, cv::Scalar(kCoverBrightness), -1);
    return out;
}

// list_block_cases: 가림과 반사 사례 구성
//   입력: reference (기준)
//   출력: 사례 목록
std::vector<BlockCase> list_block_cases(const Reference& reference) {
    std::vector<BlockCase> cases;
    const int percents[] = {5, 10, 25, 50, 75, 100};
    for (int k = 0; k < 3; ++k) {
        for (int p : percents)
            cases.push_back({"qr" + std::to_string(k + 1) + "_cover" + std::to_string(p), "QR " + std::to_string(k + 1) + " 을 " + std::to_string(p) + "% 가림",
                             [=](const cv::Mat& g, const QrMatches& m) { return cover_left_part(g, qr_box_in_photo(reference, m, k), p / 100.0); }});
        for (int radius : {20, 45, 90})
            cases.push_back({"qr" + std::to_string(k + 1) + "_glare" + std::to_string(radius), "QR " + std::to_string(k + 1) + " 에 반사광 (반지름 " + std::to_string(radius) + " px)",
                             [=](const cv::Mat& g, const QrMatches& m) {
                                 const cv::Rect box = qr_box_in_photo(reference, m, k);
                                 return scene::glare(g, box.tl() + cv::Point(box.width / 2, box.height / 2), radius, kGlareStrength);
                             }});
    }
    const int pairs[3][2] = {{0, 1}, {0, 2}, {1, 2}};
    for (const auto& pair : pairs) {
        const int a = pair[0], b = pair[1];
        cases.push_back({"qr" + std::to_string(a + 1) + std::to_string(b + 1) + "_cover100", "QR " + std::to_string(a + 1) + ", " + std::to_string(b + 1) + " 을 모두 가림",
                         [=](const cv::Mat& g, const QrMatches& m) { return cover_left_part(cover_left_part(g, qr_box_in_photo(reference, m, a), 1.0), qr_box_in_photo(reference, m, b), 1.0); }});
    }
    cases.push_back({"qr123_cover100", "QR 3개를 모두 가림", [=](const cv::Mat& g, const QrMatches& m) {
                         cv::Mat out = g;
                         for (int k = 0; k < 3; ++k) out = cover_left_part(out, qr_box_in_photo(reference, m, k), 1.0);
                         return out;
                     }});
    return cases;
}

// 판정 하나의 결과 집계
struct Outcome {
    int correct = 0;   // 값을 냈고 가리기 전 값과 허용오차 안
    int wrong = 0;     // 값을 냈는데 허용오차 밖
    int failed = 0;    // 값을 내지 않음 (실패)
    double worst_y = 0, worst_t = 0;   // 값을 낸 사진 중 가리기 전 값과의 최대 차이

    // add: 사진 한 장의 결과 추가
    //   입력: has_value (값을 냈는지), dy와 dt (가리기 전 값과의 차이)
    //   출력: 없음
    void add(bool has_value, double dy, double dt) {
        if (!has_value) { ++failed; return; }
        (tools::passes(dy, dt) ? correct : wrong)++;
        worst_y = std::max(worst_y, std::abs(dy)); worst_t = std::max(worst_t, std::abs(dt));
    }
    // text: 보고서용 한 칸
    std::string text() const {
        return "맞음 " + std::to_string(correct) + ", 틀림 " + std::to_string(wrong) + ", 실패 " + std::to_string(failed) +
               (correct + wrong ? " (최대 Y " + fixed(worst_y, 3) + ", T " + fixed(worst_t, 4) + ")" : "");
    }
};

// run_block_cases: 2부. 합성 가림
//   입력: model (기준과 계산식), subset_sets (QR 묶음별 계산식), usable (학습에 쓰는 사진), output (출력 폴더)
//   출력: 없음 (보고서, 표, 사례 그림 기록)
void run_block_cases(const Model& model, const std::vector<CoefficientSet>& subset_sets, const std::vector<const DatasetPhoto*>& usable, const fs::path& output) {
    const QrFinder qr_finder(model.reference);
    const PreviousQrFinder previous_finder(model.reference);
    const ScrewFinder screw_finder(model.reference);
    const std::vector<BlockCase> cases = list_block_cases(model.reference);
    std::vector<const DatasetPhoto*> chosen;
    for (size_t i = 0; i < usable.size(); i += kBlockPhotoStep) chosen.push_back(usable[i]);

    struct CaseResult {
        Outcome now, count_rule, spacing_fail, use_agreeing, use_agreeing_own_set, use_similar_own_set;
        int now_warned_wrong = 0;      // 현재 판정에서 틀린 값에 경고나 실패가 붙은 수
        int wrong_place = 0;           // QR 하나 이상이 엉뚱한 자리에 맞은 사진 수
        int found[4] = {};             // 검출 QR 수별 사진 수
        std::map<std::string, int> now_codes;   // 현재 판정의 알림 코드별 사진 수 (알림이 없으면 "없음")
        double wrong_similarity_low = 1, wrong_similarity_high = 0;   // 엉뚱한 자리에 맞은 QR의 닮은 정도 범위
        std::vector<double> search_ms, spreads, lowest_similarities;   // 사진별 탐색 시간, 검출 QR의 이동량 차이, 최소 닮은 정도
    };
    std::vector<CaseResult> results(cases.size());
    std::ofstream table(output / "qr_blocking_cases.tsv", std::ios::binary);
    table << "case\tphoto\tfound\twrong_place\tnow_status\tnow_code\tnow_dy\tnow_dt\tagreeing\tagree_dy\tagree_dt\tsearch_ms\n";

    tools::ProgressBar bar("2부 합성 가림", int(chosen.size()));
    const tools::QuietWarnings quiet;
    for (size_t photo_index = 0; photo_index < chosen.size(); ++photo_index) {
        bar.step();
        const DatasetPhoto* photo = chosen[photo_index];
        cv::Mat gray;
        std::string error;
        if (!to_gray(read_photo(photo->path), gray, error)) continue;
        const QrMatches clean = find_previous_way(previous_finder, gray);
        if (clean.found_count() != 3) continue;
        const Result clean_result = measure_diagnostic(gray);                                   // 현재 판정의 기준값 (영점 포함)
        const Pose clean_pose = pose_from(model, screw_finder, gray, clean, {true, true, true});   // 대안 판정의 기준값 (영점 차감 전)
        if (clean_result.status == Status::Failed || !clean_pose.has_value) continue;

        for (size_t c = 0; c < cases.size(); ++c) {
            const cv::Mat blocked = cases[c].make(gray, clean);
            if (photo_index == 0) scene::save_png(output / ("block_" + cases[c].id + ".png"), scene::labeled_view(blocked, cases[c].id));
            CaseResult& r = results[c];

            const auto start = std::chrono::steady_clock::now();
            qr_finder.find_qr_shift(blocked);
            r.search_ms.push_back(milliseconds_since(start));

            const QrMatches matches = find_previous_way(previous_finder, blocked);
            bool wrong_place = false;
            for (int k = 0; k < 3; ++k) {
                if (!matches.match[k].found || std::abs(matches.match[k].dx - clean.match[k].dx) <= kWrongPlacePx) continue;
                wrong_place = true;
                r.wrong_similarity_low = std::min(r.wrong_similarity_low, matches.match[k].similarity);
                r.wrong_similarity_high = std::max(r.wrong_similarity_high, matches.match[k].similarity);
            }
            r.wrong_place += wrong_place;
            ++r.found[matches.found_count()];
            double low = 1e9, high = -1e9, lowest_similarity = 1;
            for (int k = 0; k < 3; ++k) {
                if (!matches.match[k].found) continue;
                low = std::min(low, matches.match[k].dx); high = std::max(high, matches.match[k].dx);
                lowest_similarity = std::min(lowest_similarity, matches.match[k].similarity);
            }
            if (matches.found_count() > 0) { r.spreads.push_back(high - low); r.lowest_similarities.push_back(lowest_similarity); }

            // 현재 판정
            const Result now = measure_diagnostic(blocked);
            const bool now_has_value = now.status != Status::Failed;
            const double now_dy = now.y_mm - clean_result.y_mm, now_dt = now.t_deg - clean_result.t_deg;
            r.now.add(now_has_value, now_dy, now_dt);
            ++r.now_codes[now.code.empty() ? "없음" : now.code];
            if (now_has_value && !tools::passes(now_dy, now_dt)) r.now_warned_wrong += now.status == Status::Warning;

            // 개수 규칙: 3개가 아니면 실패, 3개이면 그대로 사용
            const bool three = matches.found_count() == 3;
            const Pose all_pose = three ? pose_from(model, screw_finder, blocked, matches, {true, true, true}) : Pose{};
            r.count_rule.add(all_pose.has_value, all_pose.y_mm - clean_pose.y_mm, all_pose.t_deg - clean_pose.t_deg);

            // 간격 검사: 3개를 찾고 이동량이 서로 같을 때만 사용
            const std::array<bool, 3> group = agreeing_group(matches);
            const int group_size = int(group[0]) + int(group[1]) + int(group[2]);
            const bool all_agree = three && group_size == 3;
            r.spacing_fail.add(all_agree && all_pose.has_value, all_pose.y_mm - clean_pose.y_mm, all_pose.t_deg - clean_pose.t_deg);

            // 맞는 QR로 계산: 이동량이 같은 QR이 2개 이상이면 그것만으로 계산
            const Pose group_pose = group_size >= 2 ? pose_from(model, screw_finder, blocked, matches, group) : Pose{};
            r.use_agreeing.add(group_pose.has_value, group_pose.y_mm - clean_pose.y_mm, group_pose.t_deg - clean_pose.t_deg);

            // 맞는 QR + 묶음별 계산식: 위와 같되 그 QR 묶음으로 만든 계산식 사용
            int subset = 0;
            while (subset < kQrSubsetCount && kQrSubsets[subset] != group) ++subset;
            const Pose own_pose = group_size >= 2 ? pose_from(model, screw_finder, blocked, matches, group, &subset_sets[subset]) : Pose{};
            r.use_agreeing_own_set.add(own_pose.has_value, own_pose.y_mm - clean_pose.y_mm, own_pose.t_deg - clean_pose.t_deg);

            // 닮은 정도 검사 추가: 덜 닮은 QR을 먼저 빼고, 남은 QR에 간격 검사와 묶음별 계산식 적용
            const QrMatches similar = drop_less_similar(matches);
            const std::array<bool, 3> similar_group = agreeing_group(similar);
            const int similar_size = int(similar_group[0]) + int(similar_group[1]) + int(similar_group[2]);
            int similar_subset = 0;
            while (similar_subset < kQrSubsetCount && kQrSubsets[similar_subset] != similar_group) ++similar_subset;
            const Pose similar_pose = similar_size >= 2 ? pose_from(model, screw_finder, blocked, similar, similar_group, &subset_sets[similar_subset]) : Pose{};
            r.use_similar_own_set.add(similar_pose.has_value, similar_pose.y_mm - clean_pose.y_mm, similar_pose.t_deg - clean_pose.t_deg);

            table << cases[c].id << "\t" << photo->path << "\t" << matches.found_count() << "\t" << wrong_place << "\t" << int(now.status) << "\t" << now.code << "\t"
                  << (now_has_value ? fixed(now_dy, 4) : "") << "\t" << (now_has_value ? fixed(now_dt, 5) : "") << "\t" << group_size << "\t"
                  << (group_pose.has_value ? fixed(group_pose.y_mm - clean_pose.y_mm, 4) : "") << "\t" << (group_pose.has_value ? fixed(group_pose.t_deg - clean_pose.t_deg, 5) : "") << "\t"
                  << fixed(r.search_ms.back(), 2) << "\n";
        }
    }
    bar.finish();

    print("\n[2부] 합성 가림. 사진 " + std::to_string(chosen.size()) + "장 x 사례 " + std::to_string(cases.size()) + "가지");
    print("  맞음 = 값을 냈고 가리기 전 값과 |Y| <= 0.045 mm, |T| <= 0.01 도 안. 틀림 = 값을 냈는데 그 밖. 실패 = 값을 내지 않음");
    print("  개수 규칙 = QR 3개가 아니면 실패. 간격 검사 = 3개의 이동량 차이가 " + fixed(kAgreeLimitPx, 0) + " px 안일 때만 사용");
    print("  맞는 QR로 계산 = 이동량이 같은 2개 이상으로 계산 (지금 계산식). + 묶음별 계산식 = 그 QR 묶음으로 만든 계산식 사용");
    print("  + 닮은 정도 검사 = 닮은 정도 " + fixed(kTrustedSimilarity, 2) + " 미만인 QR을 먼저 뺀 뒤 위와 같이 계산");
    for (size_t c = 0; c < cases.size(); ++c) {
        const CaseResult& r = results[c];
        print("  " + cases[c].name + "  [" + cases[c].id + "]");
        print("    검출 QR 수 3/2/1/0: " + std::to_string(r.found[3]) + "/" + std::to_string(r.found[2]) + "/" + std::to_string(r.found[1]) + "/" + std::to_string(r.found[0]) +
              ", 엉뚱한 자리에 맞은 사진 " + std::to_string(r.wrong_place) + "장, QR 탐색 시간 가운데값 " + fixed(median_of(r.search_ms), 1) + " ms");
        if (!r.spreads.empty())
            print("    검출 QR의 이동량 차이 가운데값 " + fixed(median_of(r.spreads), 2) + " px, 최소 닮은 정도 가운데값 " + fixed(median_of(r.lowest_similarities), 3) +
                  " (범위 " + fixed(*std::min_element(r.lowest_similarities.begin(), r.lowest_similarities.end()), 3) + " ~ " +
                  fixed(*std::max_element(r.lowest_similarities.begin(), r.lowest_similarities.end()), 3) + ")");
        if (r.wrong_place) print("    엉뚱한 자리에 맞은 QR의 닮은 정도: " + fixed(r.wrong_similarity_low, 3) + " ~ " + fixed(r.wrong_similarity_high, 3));
        print("    지금:            " + r.now.text() + (r.now.wrong ? ", 틀린 값 중 경고가 붙은 것 " + std::to_string(r.now_warned_wrong) + "장" : ""));
        std::string codes;
        for (const auto& code : r.now_codes) codes += (codes.empty() ? "" : ", ") + code.first + " " + std::to_string(code.second);
        print("    지금의 알림 코드: " + codes);
        print("    개수 규칙:       " + r.count_rule.text());
        print("    간격 검사:       " + r.spacing_fail.text());
        print("    맞는 QR로 계산:  " + r.use_agreeing.text());
        print("    + 묶음별 계산식: " + r.use_agreeing_own_set.text());
        print("    + 닮은 정도 검사: " + r.use_similar_own_set.text());
    }
}

}   // 파일 내부

// main: QR 가림 판정 규칙 비교
//   입력: 실행할 때 지정한 경로(파일 상단 사용법)
//   출력: 실행 결과와 성공 여부를 나타내는 종료 코드
int main(int argc, char** argv) {
    tools::use_korean_console();
    if (argc != 4) { std::cerr << "사용: check_qr_blocking <model 폴더> <tables 폴더> <출력 폴더>\n"; return 2; }
    const std::string model_folder = argv[1];
    const fs::path output = fs::u8path(argv[3]);
    std::string error;
    Model model;
    std::vector<DatasetPhoto> photos;
    std::vector<tools::SessionZero> sessions;
    if (!load_reference(model_folder, model.reference, error) || !load_coefficient_sets(model_folder, model.sets, error) ||
        !tools::load_tables(fs::u8path(argv[2]), photos, sessions, error) || !tools::make_empty_folder(output, error)) {
        std::cerr << error << "\n";
        return 1;
    }
    const Result ready = initialize(model_folder);
    if (ready.status == Status::Failed) { std::cerr << ready.message << "\n"; return 1; }

    std::vector<const DatasetPhoto*> usable;
    for (const DatasetPhoto& photo : photos) if (photo.usable) usable.push_back(&photo);

    const std::vector<CoefficientSet> subset_sets = run_real_photos(model, usable);
    run_block_cases(model, subset_sets, usable, output);
    std::ofstream(output / "qr_blocking_report.txt", std::ios::binary) << report_text.str();
    return 0;
}
