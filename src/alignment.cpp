// 측정 흐름: QR 이동 -> 나사 이동 -> Y, T 계산 -> 결과 점검
// 출력 = 계산값 + 기준 연결값(carry_over) - 영점(zero)
// 동시 호출 시 state_mutex로 한 번에 한 요청만 처리
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

// Measured: 사진별 1~3단계 결과. 영점 차감 전
struct Measured {
    bool has_value = false;
    QrShift qr;
    std::array<ScrewShift, 4> screws;
    const CoefficientSet* set = nullptr;
    double values[kValueCount] = {};   // [0] QR, [1~4] 나사 1~4 테두리, [5~8] 나사 1~4 둘레 영역 가로 이동
    YT raw;                  // 계산식으로 구한 그대로의 Y, T
};

// State: 다음 측정에도 사용할 설정, 기준 사진, 계산식, 영점
// 초기 구성: initialize. 갱신: 기준 또는 영점 설정
// 여러 요청의 동시 변경 방지: state_mutex 사용
struct State {
    bool ready = false;
    Settings settings;
    Reference reference;
    std::unique_ptr<QrFinder> qr_finder;
    std::unique_ptr<ScrewFinder> screw_finder;
    std::vector<CoefficientSet> coefficient_sets;
    std::vector<AgreementCheck> agreement_checks;

    YT zero;               // 영점 (출력 = 계산값 + carry_over - zero)
    YT carry_over;         // 기준 재등록 전후의 좌표 연결값
    std::map<std::string, std::array<double, kValueCount>> baselines;   // 나사 조합별 영점 설정 때의 차이(실제 이동량 - 예상 이동량)
};

State state;
std::mutex state_mutex;

//
// 결과 생성과 알림 처리
//

// report: 로그 기록
//   입력: result (message가 비어있으면 아무것도 하지 않음)
//   출력: 없음 (화면 오류 출력과 설정한 기록 파일에 한 줄 저장)
void report(const Result& result) {
    if (result.message.empty()) return;
    const char* level = result.status == Status::Failed ? "[실패 " : result.status == Status::Warning ? "[경고 " : "[알림 ";
    const std::string line = level + result.code + "] " + result.message;
    std::cerr << line << std::endl;
    if (!state.settings.log_file.empty()) std::ofstream(state.settings.log_file, std::ios::app) << line << "\n";
}

// failed: 실패 결과 생성
//   입력: code (오류 코드), message (오류 메시지)
//   출력: Result (y_mm/t_deg/점검값은 NaN, status = Failed), 로그도 남김
Result failed(const std::string& code, const std::string& message) {
    Result result;
    result.y_mm = result.t_deg = kNaN;
    result.check_value = result.check_limit = kNaN;
    for (double& shift : result.screw_shift_px) shift = kNaN;
    for (double& shift : result.screw_area_shift_px) shift = kNaN;
    result.status = Status::Failed;
    result.code = code;
    result.message = message;
    report(result);
    return result;
}

// add_notice: Result에 알림 추가
//   입력: result (수정할 결과), status (상태), code (알림 코드), message (내용)
//   출력: 없음 (result.status/code/message를 직접 바꿈, 처음 추가한 알림 코드를 유지)
void add_notice(Result& result, Status status, const std::string& code, const std::string& message) {
    if (result.code.empty()) result.code = code;
    if (status == Status::Failed || result.status == Status::Ok) result.status = status;
    result.message += (result.message.empty() ? "" : " / ") + message;
}

//
// 1~3단계: QR, 나사, Y와 T
//

// measure_values: QR, 나사 이동량과 Y, T 계산
//   입력: gray (회색 사진)
//   출력: Measured. QR 또는 계산식 미확보 시 has_value = false
Measured measure_values(const cv::Mat& gray) {
    Measured measured;

    // 1단계: QR 이동 측정. QR 3개를 모두 찾지 못하면 종료
    measured.qr = state.qr_finder->find_qr_shift(gray);
    if (!measured.qr.found) return measured;
    measured.values[0] = measured.qr.dx;

    // 2단계: QR 위치 주변에서 사용 나사 탐색
    measured.screws = state.screw_finder->find_screw_shifts(gray, measured.qr.dx, measured.qr.dy, state.settings.use_screw);
    std::array<bool, 4> found{};
    for (int k = 0; k < 4; ++k) {
        found[k] = measured.screws[k].found;
        measured.values[k + 1] = found[k] ? measured.screws[k].dx : 0.0;
        measured.values[kAreaValueOffset + k] = found[k] ? measured.screws[k].area_dx : 0.0;
    }

    // 3단계: 검출 나사 조합의 계산식으로 Y, T 계산
    measured.set = choose_coefficient_set(state.coefficient_sets, found);
    if (!measured.set) return measured;
    measured.raw = compute_yt(*measured.set, measured.values);
    measured.has_value = true;
    return measured;
}

// reset_baselines: 나사 조합별 현재 측정값과 예상값 차이 저장
//   입력: measured (해당 조합 나사를 모두 찾았을 때만 그 조합의 baseline 갱신)
//   출력: 없음 (state.baselines를 새로 채움)
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

//
// 4단계: 값 일치, 학습 범위, 결과 알림
//

// missing_screws: 검출하지 못한 나사 번호 수집
//   입력: measured (측정 결과)
//   출력: 쉼표로 구분한 나사 번호 문자열 (다 찾았으면 빈 문자열)
std::string missing_screws(const Measured& measured) {
    std::string names;
    for (int k = 0; k < 4; ++k)
        if (state.settings.use_screw[k] && !measured.screws[k].found && !measured.screws[k].partial) names += (names.empty() ? "" : ", ") + std::to_string(k + 1);
    return names;
}

// partial_screws: 머리 일부 가림 또는 오염 의심으로 뺀 나사와 사유
//   입력: measured (측정 결과)
//   출력: "나사 2: 사유" 를 " / " 로 이은 문자열 (없으면 빈 문자열)
std::string partial_screws(const Measured& measured) {
    std::string text;
    for (int k = 0; k < 4; ++k)
        if (state.settings.use_screw[k] && measured.screws[k].partial) text += (text.empty() ? "" : " / ") + std::string("나사 ") + std::to_string(k + 1) + ": " + measured.screws[k].problem;
    return text;
}

// exceeded_axis: 학습 범위를 벗어난 값 표시
//   입력: axis (축 이름), value (측정값), range (학습 범위), unit (단위)
//   출력: 측정값과 범위를 담은 알림 문자열
std::string exceeded_axis(const char* axis, double value, const double* range, const char* unit) {
    if (value >= range[0] && value <= range[1]) return "";
    char text[128];
    std::snprintf(text, sizeof(text), "%s %+.8g %s (학습 범위 %+.8g ~ %+.8g %s)", axis, value, unit, range[0], range[1], unit);
    return text;
}

// used_screws: 실제 계산에 쓴 나사 표시
//   입력: screws (사용 나사 번호)
//   출력: 사용 나사 안내 문자열
std::string used_screws(const std::string& screws) {
    if (screws == "none") return "사용 나사 없음";
    std::string names;
    for (char screw : screws) names += (names.empty() ? "" : ", ") + std::string(1, screw);
    return "나사 " + names + "로 계산";
}

// add_notices: 발생한 조건에 해당하는 알림을 순서대로 추가
//   입력: result (수정 대상), measured (측정 결과), agreement (4단계 값 일치 결과), 영점 차감 전 Y, T
//   출력: 없음 (해당하는 경우마다 result에 알림 추가)
void add_notices(Result& result, const Measured& measured, const AgreementResult& agreement, double absolute_y, double absolute_t) {
    const std::string missing = missing_screws(measured);
    const std::string used = used_screws(measured.set->screws);

    // 머리 일부 가림 또는 오염 의심 나사는 빼고 계산했다. 좌우 나사 조합이 남지 않았거나 설정에서 실패로 정했으면 실패
    const std::string partial = partial_screws(measured);
    if (!partial.empty()) {
        const bool fail = state.settings.fail_on_partial_screw || !measured.set->precise;
        add_notice(result, fail ? Status::Failed : Status::Warning, fail ? "E_SCREW_PARTIAL" : "W_SCREW_PARTIAL", partial + " (그 나사를 빼고 " + used + ")");
    }
    if (!measured.set->precise)
        add_notice(result, Status::Warning, "W_COARSE",
                   "좌우 나사 조합 미확보, 대략값 반환 (" + used + ")");
    if (!agreement.agrees()) {
        const std::string which = agreement.worst_value_index == 0 ? "QR" : "나사 " + std::to_string(agreement.worst_value_index);
        char values[96];
        std::snprintf(values, sizeof(values), "%.8g, 한계 %.8g", agreement.value, agreement.limit);
        add_notice(result, state.settings.fail_on_disagreement ? Status::Failed : Status::Warning, "W_VALUES_DISAGREE",
                   "일치 점검값 " + std::string(values) + " 초과 (최대 편차 항목 " + which + ")");
    }
    if (measured.set->precise && !missing.empty())
        add_notice(result, Status::Warning, "W_SCREW_MISSING", "나사 " + missing + " 미검출, " + used);
    if (!check_in_trained_range(state.reference, absolute_y, absolute_t)) {
        const auto y = exceeded_axis("Y", absolute_y, state.reference.trained_y_mm, "mm");
        const auto t = exceeded_axis("T", absolute_t, state.reference.trained_t_deg, "도");
        add_notice(result, Status::Warning, "W_OUT_OF_RANGE",
                   "영점 차감 전 학습 범위 초과 (" + y + (y.empty() || t.empty() ? "" : ", ") + t + ")");
    }
}

// make_result: 최종 결과 생성
//   입력: measured (1~3단계 결과)
//   출력: Result (영점, carry_over 반영한 y_mm/t_deg와 4단계 점검, 알림 포함)
Result make_result(const Measured& measured) {
    Result result;
    const double absolute_y = measured.raw.y_mm + state.carry_over.y_mm;
    const double absolute_t = measured.raw.t_deg + state.carry_over.t_deg;
    result.y_mm = absolute_y - state.zero.y_mm;
    result.t_deg = absolute_t - state.zero.t_deg;
    result.qr_shift_px = measured.qr.dx;
    for (int k = 0; k < 4; ++k) result.screw_shift_px[k] = measured.screws[k].found ? measured.screws[k].dx : kNaN;
    for (int k = 0; k < 4; ++k) result.screw_area_shift_px[k] = measured.screws[k].found ? measured.screws[k].area_dx : kNaN;
    result.screws_used = measured.set->screws;

    // 4단계 (가): QR과 나사 측정값의 일치 여부
    AgreementResult agreement;
    result.check_value = result.check_limit = kNaN;
    if (const AgreementCheck* check = find_agreement_check(state.agreement_checks, measured.set->screws)) {
        const auto baseline = state.baselines.count(check->screws) ? state.baselines[check->screws] : std::array<double, kValueCount>{};
        agreement = check_values_agree(*check, measured.values, measured.raw.y_mm, measured.raw.t_deg, baseline.data());
        result.check_value = agreement.value;
        result.check_limit = agreement.limit;
    }
    // 4단계 (나): reference.json의 학습 범위 확인
    add_notices(result, measured, agreement, absolute_y, absolute_t);
    return result;
}

// use_reference: 기준 교체
//   입력: reference (새 기준)
//   출력: 없음 (state.reference와 QR, 나사 탐색기를 새로 만듦)
void use_reference(const Reference& reference) {
    state.reference = reference;
    state.qr_finder.reset(new QrFinder(reference));
    state.screw_finder.reset(new ScrewFinder(reference));
}

}   // 파일 내부

//
// 공개 함수
//

// initialize: 측정에 필요한 자료 읽기
//   입력: data_folder (모델 폴더), settings_file (설정 파일 경로)
//   출력: 기본 Result. 실패 시 Failed와 오류 알림
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
//   출력: Result (y_mm, t_deg, 점검값, 알림, 실패하면 오류 코드와 메시지)
Result measure_alignment(const cv::Mat& photo) {
    std::lock_guard<std::mutex> lock(state_mutex);
    if (!state.ready) return failed("E_NOT_INITIALIZED", "초기화 완료 전 호출");
    try {
        cv::Mat gray;
        std::string error;
        if (!to_gray(photo, gray, error)) return failed("E_PHOTO", error);

        const Measured measured = measure_values(gray);
        if (!measured.has_value && measured.qr.found && !partial_screws(measured).empty())
            return failed("E_SCREW_PARTIAL", partial_screws(measured) + " (남은 나사로 계산할 계수 없음)");
        if (!measured.has_value)
            return failed("E_QR_NOT_FOUND", measured.qr.found ? "검출한 나사 조합의 계수 미확보" : measured.qr.problem);

        const Result result = make_result(measured);
        report(result);
        return result;
    } catch (const std::exception& e) {
        return failed("E_INTERNAL", std::string("측정 중 오류: ") + e.what());
    }
}

// set_zero_from_photos: 명령 0 에서 찍은 사진들로 영점 재설정
//   입력: zero_photos (로봇을 명령 0 으로 보내 찍은 사진 여러 장)
//   출력: Result (성공하면 새 영점의 y_mm/t_deg, 실패하면 오류 코드와 메시지)
Result set_zero_from_photos(const std::vector<cv::Mat>& zero_photos) {
    std::lock_guard<std::mutex> lock(state_mutex);
    if (!state.ready) return failed("E_NOT_INITIALIZED", "초기화 완료 전 호출");
    try {
        cv::Mat gray;
        std::string error;
        if (!average_gray(zero_photos, gray, error)) return failed("E_PHOTO", error);

        const Measured measured = measure_values(gray);
        if (!partial_screws(measured).empty())
            return failed("E_ZERO", "영점 사진의 " + partial_screws(measured) + ", 기존 영점 유지");
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
//   입력: y_mm, t_deg (새 영점)
//   출력: 없음 (state.zero 변경, baselines 유지)
void set_zero_value(double y_mm, double t_deg) {
    std::lock_guard<std::mutex> lock(state_mutex);
    state.zero.y_mm = y_mm;
    state.zero.t_deg = t_deg;
}

// clear_zero: 영점 값 초기화
//   입력: 없음
//   출력: 없음 (state.zero만 (0, 0), 기준과 baselines 유지)
void clear_zero() {
    std::lock_guard<std::mutex> lock(state_mutex);
    state.zero = YT();
}

// set_reference_from_photos: 센터 사진들로 기준 재등록 (가능하면 기존 영점 연결)
//   입력: center_photos (센터에서 찍은 사진 여러 장)
//   출력: Result (성공하면 새 기준에서의 y_mm/t_deg, 실패하면 오류 코드와 메시지)
Result set_reference_from_photos(const std::vector<cv::Mat>& center_photos) {
    std::lock_guard<std::mutex> lock(state_mutex);
    if (!state.ready) return failed("E_NOT_INITIALIZED", "초기화 완료 전 호출");
    try {
        cv::Mat gray;
        std::string error;
        if (!average_gray(center_photos, gray, error)) return failed("E_PHOTO", error);

        // 변경 전후 측정 차이를 carry_over에 반영
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
            // 기존 영점 연결 불가. 새 사진을 영점으로 설정
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

}   // alignment
