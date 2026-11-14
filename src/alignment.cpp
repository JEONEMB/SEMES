// alignment: 사진 한 장을 Y(세로 밀림), T(기울기) 값으로 바꾸는 모듈
// measure_alignment 가 사진 한 장을 받으면 아래 1~4단계를 차례로 거친다.
//   1. find_qr_shift      QR 이 기준에서 얼마나 움직였는지 측정 (steps/01_find_qr)
//   2. find_screw_shifts  나사 4개가 기준에서 얼마나 움직였는지 측정 (steps/02_find_screws)
//   3. compute_yt         위 이동량들로 Y, T 계산 (steps/03_compute_yt)
//   4. check_result 계열  결과가 믿을 만한지 점검 (steps/04_check_result)
// 1~4단계 결과는 아래 Measured 에 담기고, 여러 번 측정해도 바뀌지 않는 값(설정, 기준, 계수 등)은
// State 에 따로 보관해서 다음 측정에도 그대로 쓴다.
#include "alignment.hpp"
#include "common/reference.hpp"
#include "setup/find_new_reference.hpp"
#include "setup/settings.hpp"
#include "steps/01_find_qr.hpp"
#include "steps/02_find_screws.hpp"
#include "steps/03_compute_yt.hpp"
#include "steps/04_check_result.hpp"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>

namespace alignment {

namespace {

const double kNaN = std::numeric_limits<double>::quiet_NaN();

// Measured: 사진 한 장을 1~3단계에 돌린 결과 (영점을 빼기 전의 값, 사진마다 새로 만들어짐)
struct Measured {
    bool has_value = false;
    QrShift qr;
    std::array<ScrewShift, 4> screws;
    const CoefficientSet* set = nullptr;
    double values[kValueCount] = {};   // [0] QR, [1~4] 나사 1~4 가로 이동
    YT raw;                  // 계수로 계산한 그대로의 Y, T
};

// State: 측정 여러 번에 걸쳐 계속 들고 있어야 하는 값들의 모음
//   (사진 한 장으로 끝나는 Measured 와 달리, initialize 에서 한 번 채워지고
//    이후 측정마다 읽고 갱신하며 계속 이어서 쓴다: 설정, 기준, 계수, 영점 등)
struct State {
    bool ready = false;
    Settings settings;
    Reference reference;
    std::unique_ptr<QrFinder> qr_finder;
    std::unique_ptr<ScrewFinder> screw_finder;
    std::vector<CoefficientSet> coefficient_sets;
    std::vector<AgreementCheck> agreement_checks;

    YT zero;               // 영점 (출력 = 계산값 + carry_over - zero)
    YT carry_over;         // 기준 사진을 다시 잡았을 때, 처음 영점을 잇기 위해 더하는 값
    std::map<std::string, std::array<double, kValueCount>> baselines;   // 조합마다 영점을 잡을 때의 (측정값 - 기대값)
};

State state;
std::mutex state_mutex;

// ------------------------------------------------------------------------------------------------
// 결과 만들기와 알림: Result 를 만들고 실패/경고/참고 메시지를 붙이는 공통 함수들
// ------------------------------------------------------------------------------------------------

// report: 로그 남기기
//   입력: result (message 가 비어있으면 아무것도 하지 않음)
//   출력: 없음 (stderr 와 설정된 로그 파일에 한 줄 기록)
void report(const Result& result) {
    if (result.message.empty()) return;
    const char* level = result.status == Status::Failed ? "[실패 " : result.status == Status::Warning ? "[경고 " : "[알림 ";
    const std::string line = level + result.code + "] " + result.message;
    std::cerr << line << std::endl;
    if (!state.settings.log_file.empty()) std::ofstream(state.settings.log_file, std::ios::app) << line << "\n";
}

// failed: 실패 Result 만들기
//   입력: code (에러 코드), message (에러 메시지)
//   출력: Result (y_mm/t_deg/점검값은 NaN, status = Failed), 로그도 남김
Result failed(const std::string& code, const std::string& message) {
    Result result;
    result.y_mm = result.t_deg = kNaN;
    result.check_value = result.check_limit = kNaN;
    for (double& shift : result.screw_shift_px) shift = kNaN;
    result.status = Status::Failed;
    result.code = code;
    result.message = message;
    report(result);
    return result;
}

// add_notice: Result 에 알림 추가
//   입력: result (수정 대상), status, code, message
//   출력: 없음 (result.status/code/message 를 직접 바꿈, 먼저 추가한 알림이 code 를 가짐)
void add_notice(Result& result, Status status, const std::string& code, const std::string& message) {
    if (result.code.empty()) result.code = code;
    if (status == Status::Failed || result.status == Status::Ok) result.status = status;
    result.message += (result.message.empty() ? "" : " / ") + message;
}

// add_info: Result 에 참고 알림 추가 (상태는 바꾸지 않음)
//   입력: result (수정 대상), code, message
//   출력: 없음 (result.message 만 덧붙임)
void add_info(Result& result, const std::string& code, const std::string& message) {
    if (result.code.empty()) result.code = code;
    result.message += (result.message.empty() ? "" : " / ") + message;
}

// ------------------------------------------------------------------------------------------------
// 1~3단계: QR 을 찾고, 나사를 찾고, 그 이동량들로 Y, T 를 계산하는 함수들
// ------------------------------------------------------------------------------------------------

// measure_values: 1~3단계 실행 (QR 찾기, 나사 찾기, Y,T 계산 순서로 진행)
//   입력: gray (흑백 사진)
//   출력: Measured (QR 을 못 찾으면 has_value = false)
Measured measure_values(const cv::Mat& gray) {
    Measured measured;

    // 1단계: QR 이 기준에서 얼마나 움직였는지 구한다 (못 찾으면 바로 리턴, has_value 는 false 로 남음)
    measured.qr = state.qr_finder->find_qr_shift(gray);
    if (measured.qr.search == QrSearch::NotFound) return measured;
    measured.values[0] = measured.qr.dx;

    // 2단계: 설정에서 쓰기로 한 나사만, QR 위치를 기준 삼아 이동량을 구한다
    measured.screws = state.screw_finder->find_screw_shifts(gray, measured.qr.dx, measured.qr.dy, state.settings.use_screw);
    std::array<bool, 4> found{};
    for (int k = 0; k < 4; ++k) {
        found[k] = measured.screws[k].found;
        measured.values[k + 1] = found[k] ? measured.screws[k].dx : 0.0;
    }

    // 3단계: 어떤 나사를 찾았는지에 맞는 계수 세트를 고르고, 그 계수로 Y, T 를 계산한다
    measured.set = choose_coefficient_set(state.coefficient_sets, found);
    if (!measured.set) return measured;
    measured.raw = compute_yt(*measured.set, measured.values);
    measured.has_value = true;
    return measured;
}

// reset_baselines: 나사 조합별 baseline 다시 잡기
//   입력: measured (해당 조합 나사를 모두 찾았을 때만 그 조합의 baseline 갱신)
//   출력: 없음 (state.baselines 를 새로 채움)
void reset_baselines(const Measured& measured) {
    state.baselines.clear();
    for (const AgreementCheck& check : state.agreement_checks) {
        const CoefficientSet* set = nullptr;
        for (const CoefficientSet& candidate : state.coefficient_sets)
            if (candidate.screws == check.screws) set = &candidate;
        bool all_found = set != nullptr;
        for (int k = 0; all_found && k < 4; ++k)
            if (set->uses_screw[k] && !measured.screws[k].found) all_found = false;
        if (!all_found) continue;
        const YT raw = compute_yt(*set, measured.values);
        std::array<double, kValueCount> baseline{};
        expected_differences(check, measured.values, raw.y_mm, raw.t_deg, baseline.data());
        state.baselines[check.screws] = baseline;
    }
}

// ------------------------------------------------------------------------------------------------
// 4단계와 알림 붙이기: 1~3단계 결과가 믿을 만한지 점검하고, 상황에 맞는 알림을 Result 에 붙인다
// ------------------------------------------------------------------------------------------------

// missing_screws: 못 찾은 나사 번호 모으기
//   입력: measured
//   출력: 쉼표로 구분한 나사 번호 문자열 (다 찾았으면 빈 문자열)
std::string missing_screws(const Measured& measured) {
    std::string names;
    for (int k = 0; k < 4; ++k)
        if (state.settings.use_screw[k] && !measured.screws[k].found) names += (names.empty() ? "" : ", ") + std::to_string(k + 1);
    return names;
}

// px_text: 정수 px 값을 문구에 넣을 글자로
std::string px_text(int px) { return std::to_string(px) + " px"; }

std::string exceeded_axis(const char* axis, double value, const double* range, const char* unit) {
    if (value >= range[0] && value <= range[1]) return "";
    char text[128];
    std::snprintf(text, sizeof(text), "%s %+.8g %s (학습 범위 %+.8g ~ %+.8g %s)", axis, value, unit, range[0], range[1], unit);
    return text;
}

std::string used_screws(const std::string& screws) {
    if (screws == "none") return "사용 나사 없음";
    std::string names;
    for (char screw : screws) names += (names.empty() ? "" : ", ") + std::string(1, screw);
    return "나사 " + names + "로 계산";
}

// add_notices: 상황에 맞는 알림 모두 추가
//   입력: result (수정 대상), measured, agreement (4단계 값 일치 결과), 영점 차감 전 Y, T
//   출력: 없음 (해당하는 경우마다 result 에 알림 추가)
void add_notices(Result& result, const Measured& measured, const AgreementResult& agreement, double absolute_y, double absolute_t) {
    const std::string missing = missing_screws(measured);
    const std::string used = used_screws(measured.set->screws);

    if (!measured.set->precise)
        add_notice(result, Status::Warning, "W_COARSE",
                   "좌우 나사 조합 미확보, 대략값 반환 (" + used + ")");
    if (measured.qr.search == QrSearch::Partial)
        add_notice(result, Status::Warning, "W_QR_PARTIAL",
                   "전체 폭 탐색에서 QR " + std::to_string(measured.qr.found_count) + "/3 검출");
    if (!agreement.agrees()) {
        const std::string which = agreement.worst_value_index == 0 ? "QR" : "나사 " + std::to_string(agreement.worst_value_index);
        char values[96];
        std::snprintf(values, sizeof(values), "%.8g, 한계 %.8g", agreement.value, agreement.limit);
        add_notice(result, state.settings.fail_on_disagreement ? Status::Failed : Status::Warning, "W_VALUES_DISAGREE",
                   "일치 점검값 " + std::string(values) + " 초과 (최대 편차 항목 " + which + ")");
    }
    if (measured.set->precise && !missing.empty())
        add_notice(result, Status::Warning, "W_SCREW_MISSING", "나사 " + missing + " 미검출, " + used);
    if (measured.qr.search == QrSearch::FullWidth)
        add_notice(result, Status::Warning, "W_FAR", "±" + px_text(kWideHalfWidth) + " 창에서 QR 일부 또는 전체 미검출, 전체 폭에서 3/3 검출");
    if (!check_in_trained_range(state.reference, absolute_y, absolute_t)) {
        const auto y = exceeded_axis("Y", absolute_y, state.reference.trained_y_mm, "mm");
        const auto t = exceeded_axis("T", absolute_t, state.reference.trained_t_deg, "도");
        add_notice(result, Status::Warning, "W_OUT_OF_RANGE",
                   "영점 차감 전 학습 범위 초과 (" + y + (y.empty() || t.empty() ? "" : ", ") + t + ")");
    }
    if (measured.qr.search == QrSearch::Wide)
        add_info(result, "N_WIDE_SEARCH", "±" + px_text(kNarrowHalfWidth) + " 창에서 QR 일부 또는 전체 미검출, ±" + px_text(kWideHalfWidth) + " 창에서 3/3 검출");
}

// make_result: 최종 Result 만들기
//   입력: measured (1~3단계 결과)
//   출력: Result (영점, carry_over 반영한 y_mm/t_deg 와 4단계 점검, 알림 포함)
Result make_result(const Measured& measured) {
    Result result;
    const double absolute_y = measured.raw.y_mm + state.carry_over.y_mm;
    const double absolute_t = measured.raw.t_deg + state.carry_over.t_deg;
    result.y_mm = absolute_y - state.zero.y_mm;
    result.t_deg = absolute_t - state.zero.t_deg;
    result.qr_shift_px = measured.qr.dx;
    for (int k = 0; k < 4; ++k) result.screw_shift_px[k] = measured.screws[k].found ? measured.screws[k].dx : kNaN;
    result.screws_used = measured.set->screws;

    // 4단계 (가): QR/나사 값들이 서로 앞뒤가 맞는지 점검한다 (안 맞으면 가림, 반사 등이 의심됨)
    AgreementResult agreement;
    result.check_value = result.check_limit = kNaN;
    if (const AgreementCheck* check = find_agreement_check(state.agreement_checks, measured.set->screws)) {
        const auto baseline = state.baselines.count(check->screws) ? state.baselines[check->screws] : std::array<double, kValueCount>{};
        agreement = check_values_agree(*check, measured.values, measured.raw.y_mm, measured.raw.t_deg, baseline.data());
        result.check_value = agreement.value;
        result.check_limit = agreement.limit;
    }
    // 4단계 (나): 학습 범위(reference.json 의 trained_y_mm, trained_t_deg) 안에 있는지 점검한다
    add_notices(result, measured, agreement, absolute_y, absolute_t);
    return result;
}

// use_reference: 기준 바꾸기
//   입력: reference (새 기준)
//   출력: 없음 (state.reference 와 QR, 나사 탐색기를 새로 만듦)
void use_reference(const Reference& reference) {
    state.reference = reference;
    state.qr_finder.reset(new QrFinder(reference));
    state.screw_finder.reset(new ScrewFinder(reference));
}

}   // namespace

// ------------------------------------------------------------------------------------------------
// 공개 함수: alignment.hpp 에 선언된, 이 모듈 밖에서 호출하는 함수들
// ------------------------------------------------------------------------------------------------

// initialize: 모듈 준비하기 (최초 또는 재설정 시 호출)
//   입력: data_folder (기준, 계수, 점검 규칙이 든 폴더), settings_file (설정 파일 경로)
//   출력: Result (성공하면 비어있음, 실패하면 코드와 메시지)
Result initialize(const std::string& data_folder, const std::string& settings_file) {
    std::lock_guard<std::mutex> lock(state_mutex);
    try {
        State fresh;
        std::string error;
        if (!load_settings(settings_file, fresh.settings, error)) return failed("E_SETTINGS", error);
        if (!load_reference(data_folder, fresh.reference, error)) return failed("E_DATA", error);
        if (!load_coefficient_sets(data_folder, fresh.coefficient_sets, error)) return failed("E_DATA", error);
        if (!load_agreement_checks(data_folder, fresh.agreement_checks, error)) return failed("E_DATA", error);
        state = std::move(fresh);
        use_reference(Reference(state.reference));
        state.ready = true;
        return Result();
    } catch (const std::exception& e) {
        return failed("E_DATA", std::string("초기화 중 오류: ") + e.what());
    }
}

// measure_alignment: 사진 한 장으로 Y, T 측정 (1~4단계 전체 수행)
//   입력: photo (컬러 또는 흑백 사진 한 장)
//   출력: Result (y_mm, t_deg, 점검값, 알림, 실패하면 에러 코드와 메시지)
Result measure_alignment(const cv::Mat& photo) {
    std::lock_guard<std::mutex> lock(state_mutex);
    if (!state.ready) return failed("E_NOT_INITIALIZED", "초기화 완료 전 호출");
    try {
        cv::Mat gray;
        std::string error;
        if (!to_gray(photo, gray, error)) return failed("E_PHOTO", error);

        const Measured measured = measure_values(gray);
        if (!measured.has_value)
            return failed("E_QR_NOT_FOUND", measured.qr.found_count == 0 ? "전체 폭 탐색에서 QR 0/3 검출" : "검출한 나사 조합의 계수 미확보");

        const Result result = make_result(measured);
        report(result);
        return result;
    } catch (const std::exception& e) {
        return failed("E_INTERNAL", std::string("측정 중 오류: ") + e.what());
    }
}

// set_zero_from_photos: 명령 0 에서 찍은 사진들로 영점 다시 잡기
//   입력: zero_photos (로봇을 명령 0 으로 보내 찍은 사진 여러 장)
//   출력: Result (성공하면 새 영점의 y_mm/t_deg, 실패하면 에러 코드와 메시지)
Result set_zero_from_photos(const std::vector<cv::Mat>& zero_photos) {
    std::lock_guard<std::mutex> lock(state_mutex);
    if (!state.ready) return failed("E_NOT_INITIALIZED", "초기화 완료 전 호출");
    try {
        cv::Mat gray;
        std::string error;
        if (!average_gray(zero_photos, gray, error)) return failed("E_PHOTO", error);

        const Measured measured = measure_values(gray);
        if (!measured.has_value || !measured.set->precise)
            return failed("E_ZERO", "영점 사진에서 측정값 또는 좌우 나사 조합 미확보, 기존 영점 유지");

        state.zero.y_mm = measured.raw.y_mm + state.carry_over.y_mm;
        state.zero.t_deg = measured.raw.t_deg + state.carry_over.t_deg;
        reset_baselines(measured);

        Result result;
        result.y_mm = state.zero.y_mm;
        result.t_deg = state.zero.t_deg;
        return result;
    } catch (const std::exception& e) {
        return failed("E_INTERNAL", std::string("영점 잡는 중 오류: ") + e.what());
    }
}

// set_zero_value: 영점 직접 지정
//   입력: y_mm, t_deg (새 영점 값)
//   출력: 없음 (state.zero 를 바로 바꿈)
void set_zero_value(double y_mm, double t_deg) {
    std::lock_guard<std::mutex> lock(state_mutex);
    state.zero.y_mm = y_mm;
    state.zero.t_deg = t_deg;
}

// clear_zero: 영점 (0, 0) 으로 되돌리기
//   입력: 없음
//   출력: 없음 (state.zero 를 초기화)
void clear_zero() {
    std::lock_guard<std::mutex> lock(state_mutex);
    state.zero = YT();
}

// set_reference_from_photos: 센터 사진들로 기준 다시 잡기 (가능하면 영점 이어줌)
//   입력: center_photos (센터에서 찍은 사진 여러 장)
//   출력: Result (성공하면 새 기준에서의 y_mm/t_deg, 실패하면 에러 코드와 메시지)
Result set_reference_from_photos(const std::vector<cv::Mat>& center_photos) {
    std::lock_guard<std::mutex> lock(state_mutex);
    if (!state.ready) return failed("E_NOT_INITIALIZED", "초기화 완료 전 호출");
    try {
        cv::Mat gray;
        std::string error;
        if (!average_gray(center_photos, gray, error)) return failed("E_PHOTO", error);

        // 바꾸기 전 기준으로 같은 사진 측정 (바뀐 뒤 같은 값 나오게 맞추면 영점 연결)
        const Measured before = measure_values(gray);
        const Reference old_reference = state.reference;

        Reference new_reference;
        if (!find_new_reference(old_reference, gray, new_reference, error)) return failed("E_REFERENCE", error + ", 기존 기준 유지");
        use_reference(new_reference);
        const Measured after = measure_values(gray);
        if (!after.has_value || !after.set->precise) {
            use_reference(old_reference);
            return failed("E_REFERENCE", "새 기준에서 측정값 또는 좌우 나사 조합 미확보, 기존 기준 유지");
        }

        Result result;
        const bool keeps_zero = before.has_value && before.set->precise;
        if (keeps_zero) {
            state.carry_over.y_mm += before.raw.y_mm - after.raw.y_mm;
            state.carry_over.t_deg += before.raw.t_deg - after.raw.t_deg;
        } else {
            // 바꾸기 전 기준으로 측정 불가할 만큼 큰 이동 (이 사진을 새 (0, 0) 으로 삼음)
            state.carry_over.y_mm = -after.raw.y_mm;
            state.carry_over.t_deg = -after.raw.t_deg;
            state.zero = YT();
            add_notice(result, Status::Warning, "W_ZERO_RESET", "기존 기준에서 측정값 또는 좌우 나사 조합 미확보, 새 기준 사진을 영점으로 설정");
        }
        reset_baselines(after);
        result.y_mm = after.raw.y_mm + state.carry_over.y_mm - state.zero.y_mm;
        result.t_deg = after.raw.t_deg + state.carry_over.t_deg - state.zero.t_deg;
        report(result);
        return result;
    } catch (const std::exception& e) {
        return failed("E_INTERNAL", std::string("기준 다시 잡는 중 오류: ") + e.what());
    }
}

}   // namespace alignment
