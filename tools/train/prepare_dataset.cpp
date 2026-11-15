// 사진 측정과 학습용 표 생성
// 사용: prepare_dataset <사진 폴더> [--data <model 폴더>] [--output <표 폴더>] [--rules <규칙.tsv>]
// 입력: <사진 폴더>/<회차>/사진. 기본 모델: data/model
// 출력: --output 지정 폴더. 생략하면 <사진 폴더>/tables/
// 순서: 사진 수집 -> 측정 -> 회차별 영점 추정 -> 정답 보정 -> 검증 분할
// 분할: 동일 회차와 명령값끼리 묶음. 최종 검증 20%, 나머지 5묶음 × 10회
#include "alignment.hpp"
#include "common/console.hpp"
#include "common/dataset.hpp"
#include "common/data_preprocessing.hpp"
#include "common/photo_files.hpp"
#include "common/progress.hpp"
#include "common/tables.hpp"
#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <set>

namespace fs = std::filesystem;
using namespace tools;

namespace {

const double kTrainedRangeYmm = 2.0;       // 학습에 쓰는 명령 범위
const double kTrainedRangeTdeg = 1.0;
const double kFinalCheckShare = 0.2;       // 마지막 확인용으로 떼는 묶음 비율
const unsigned kSplitSeed = 20261114;      // 같은 사진 묶음을 다시 나누기 위한 시작값
const int kOutlierRounds = 3;               // 차이가 큰 사진을 제외하고 영점을 다시 구할 횟수
const double kOutlierLimit = 10.0;          // 평소 흔들림 대비 제외 기준 배수
const double kCommandErrorYmm = 0.2;        // Y 명령 불일치 후보 판정 한계(mm)
const double kCommandErrorTdeg = 0.1;       // T 명령 불일치 후보 판정 한계(도)
const int kPoseTerms = 5;                   // 영점 추정 식의 자세 항: Y, T, Y², YT, T²
const char* const kTableFolder = "tables";   // 사진 폴더 안에 표를 쓰는 폴더 이름

struct PreparedPhoto {
    std::string session;
    PhotoFile file;
    std::string relative_path;
    PreprocessingDecision preprocessing;
    bool preprocessing_applied = false;
    bool measured = false;
    Values values{};
    bool all_screws = false;
    bool in_range = false;
    bool command_mismatch = false;   // 파일명 명령과 측정 자세의 불일치 후보
    // training_y_mm: 학습용 Y 명령 선택
    //   입력: 없음 (사진 자료 사용)
    //   출력: 정정값 우선, 없으면 원본 Y 명령
    double training_y_mm() const { return preprocessing_applied ? preprocessing.verified_y_mm : file.command_y_mm; }
    // training_t_deg: 학습용 T 명령 선택
    //   입력: 없음 (사진 자료 사용)
    //   출력: 정정값 우선, 없으면 원본 T 명령
    double training_t_deg() const { return preprocessing_applied ? preprocessing.verified_t_deg : file.command_t_deg; }
    // training_allowed: 제외 규칙 확인
    //   입력: 없음 (사진의 적용 규칙 사용)
    //   출력: 학습 허용 여부
    bool training_allowed() const { return !preprocessing_applied || preprocessing.training_allowed; }
    // usable: 학습에 쓸 사진인지 확인
    //   입력: 없음 (측정 결과와 적용 규칙 사용)
    //   출력: 측정, 나사, 범위, 명령, 제외 조건을 통과하면 true
    bool usable() const { return measured && all_screws && in_range && training_allowed() && !command_mismatch; }
};

// median_of: 크기순 가운데 값 계산
//   입력: v (비어 있지 않은 숫자 목록)
//   출력: 가운데 값. 짝수 개이면 가운데 두 값의 평균
double median_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// 사진 속 이동량 계산: 판의 자세 영향 + 촬영 회차별 차이
struct ZeroFit {
    cv::Mat coefficients;                                      // 측정 항목별 계산용 숫자. 자세 항목 5개와 회차별 차이
    std::map<std::string, std::pair<double, double>> zeros;    // 회차 -> 영점 어긋남 (Y mm, T 도)
};

// fit_terms: 사진 한 장의 계산 자료 준비
//   입력: photo (사진 자료), sessions (촬영 회차 목록)
//   출력: 자세값 5개와 해당 회차 표시
cv::Mat fit_terms(const PreparedPhoto& photo, const std::vector<std::string>& sessions) {
    const double y = photo.training_y_mm(), t = photo.training_t_deg();
    const double pose[kPoseTerms] = {y, t, y * y, y * t, t * t};
    cv::Mat terms = cv::Mat::zeros(1, kPoseTerms + int(sessions.size()), CV_64F);
    for (int q = 0; q < kPoseTerms; ++q) terms.at<double>(q) = pose[q];
    terms.at<double>(kPoseTerms + int(std::find(sessions.begin(), sessions.end(), photo.session) - sessions.begin())) = 1;
    return terms;
}

// fit_session_zeros: 촬영 회차마다 다른 영점 추정
//   입력: used (사용 사진), sessions (회차 목록)
//   출력: 사진 속 이동량 계산식과 회차별 영점 차이
ZeroFit fit_session_zeros(const std::vector<const PreparedPhoto*>& used, const std::vector<std::string>& sessions) {
    // 1. 사진별 자세와 촬영 회차를 계산 자료로 준비
    const int n = int(used.size()), s_count = int(sessions.size());
    cv::Mat design(n, kPoseTerms + s_count, CV_64F);
    for (int i = 0; i < n; ++i) fit_terms(*used[i], sessions).copyTo(design.row(i));
    ZeroFit fit;
    // 2. QR과 나사별로 자세 영향과 촬영 회차 차이 계산
    fit.coefficients = cv::Mat(kValueCount, kPoseTerms + s_count, CV_64F);
    for (int k = 0; k < kValueCount; ++k) {
        cv::Mat value(n, 1, CV_64F), fitted;
        for (int i = 0; i < n; ++i) value.at<double>(i) = used[i]->values[k];
        cv::solve(design, value, fitted, cv::DECOMP_SVD);
        cv::Mat(fitted.t()).copyTo(fit.coefficients.row(k));
    }
    // 3. 사진 속 이동량이 0이 되는 회차별 명령값 계산
    const cv::Mat slope = fit.coefficients.colRange(0, 2);   // 값이 Y, T에 따라 변하는 정도
    for (int s = 0; s < s_count; ++s) {
        cv::Mat zero;   // slope * zero = -회차 상수  (그 명령에서 모든 값이 0)
        cv::solve(slope, -fit.coefficients.col(kPoseTerms + s), zero, cv::DECOMP_SVD);
        fit.zeros[sessions[s]] = {zero.at<double>(0), zero.at<double>(1)};
    }
    return fit;
}

// residual_of: 사진별 실제 이동량과 예상 이동량 비교
//   입력: fit (회차별 계산식), photo (사진 자료), sessions (촬영 회차 목록)
//   출력: 항목별 실제 이동량 - 예상 이동량
cv::Mat residual_of(const ZeroFit& fit, const PreparedPhoto& photo, const std::vector<std::string>& sessions) {
    const cv::Mat predicted = fit.coefficients * fit_terms(photo, sessions).t();
    cv::Mat residual(kValueCount, 1, CV_64F);
    for (int k = 0; k < kValueCount; ++k) residual.at<double>(k) = photo.values[k] - predicted.at<double>(k);
    return residual;
}

// command_error: 명령값 대비 자세 차이 계산
//   입력: fit (회차별 계산식), photo (사진 자료), sessions (촬영 회차 목록)
//   출력: Y 차이(mm), T 차이(도)
std::pair<double, double> command_error(const ZeroFit& fit, const PreparedPhoto& photo, const std::vector<std::string>& sessions) {
    cv::Mat pose;
    cv::solve(fit.coefficients.colRange(0, 2), residual_of(fit, photo, sessions), pose, cv::DECOMP_SVD);
    return {pose.at<double>(0), pose.at<double>(1)};
}

// estimate_session_zeros: 회차별 영점 차이 추정
//   입력: photos (사진 목록), sessions (촬영 회차 목록)
//   출력: ZeroFit (회차별 영점과 계산식), left_out (제외 사진)
ZeroFit estimate_session_zeros(const std::vector<PreparedPhoto>& photos, const std::vector<std::string>& sessions, std::vector<int>& left_out) {
    left_out.assign(photos.size(), 0);
    ZeroFit fit;
    for (int round = 0; round < kOutlierRounds + 1; ++round) {
        std::vector<int> used_index;
        for (int i = 0; i < int(photos.size()); ++i) if (photos[i].usable() && !left_out[i]) used_index.push_back(i);
        std::vector<const PreparedPhoto*> used;
        for (int i : used_index) used.push_back(&photos[i]);
        fit = fit_session_zeros(used, sessions);
        if (round == kOutlierRounds) break;
        // 평소 흔들림 대비 차이가 kOutlierLimit을 넘는 사진 제외
        std::vector<cv::Mat> residual;
        for (const PreparedPhoto* p : used) residual.push_back(residual_of(fit, *p, sessions));
        bool removed = false;
        for (int k = 0; k < kValueCount; ++k) {
            std::vector<double> column;
            for (const cv::Mat& r : residual) column.push_back(std::abs(r.at<double>(k)));
            const double spread = std::max(1.4826 * median_of(column), 1e-9);
            for (size_t i = 0; i < used.size(); ++i)
                if (column[i] / spread > kOutlierLimit) { left_out[used_index[i]] = 1; removed = true; }
        }
        if (!removed) break;
    }
    return fit;
}

// make_splits: 학습과 검증 자료 분할
//   입력: photos (사진 목록)
//   출력: final_check (최종 시험용), folds (반복 시험용 묶음 번호)
void make_splits(const std::vector<PreparedPhoto>& photos, std::vector<int>& final_check, std::vector<std::array<int, kRepeatCount>>& folds) {
    std::map<std::string, std::vector<int>> groups;   // "회차|Y|T" -> 사진 번호
    for (int i = 0; i < int(photos.size()); ++i) {
        if (!photos[i].usable()) continue;
        groups[photos[i].session + "|" + format_number(photos[i].training_y_mm()) + "|" + format_number(photos[i].training_t_deg())].push_back(i);
    }
    std::vector<std::vector<int>> group_list;
    for (auto& g : groups) group_list.push_back(g.second);

    final_check.assign(photos.size(), 0);
    folds.assign(photos.size(), {});
    for (auto& f : folds) f.fill(-1);

    std::mt19937 random(kSplitSeed);
    std::vector<int> order(group_list.size());
    for (int g = 0; g < int(order.size()); ++g) order[g] = g;
    std::shuffle(order.begin(), order.end(), random);
    const int final_groups = int(std::round(kFinalCheckShare * order.size()));
    std::vector<int> development;
    for (int g = 0; g < int(order.size()); ++g) {
        if (g < final_groups) for (int i : group_list[order[g]]) final_check[i] = 1;
        else development.push_back(order[g]);
    }
    for (int r = 0; r < kRepeatCount; ++r) {
        std::mt19937 repeat_random(kSplitSeed + 1 + r);
        std::vector<int> shuffled = development;
        std::shuffle(shuffled.begin(), shuffled.end(), repeat_random);
        for (int g = 0; g < int(shuffled.size()); ++g)
            for (int i : group_list[shuffled[g]]) folds[i][r] = g % kFoldCount;
    }
}

// measure_all_photos: 회차별 사진 수집과 측정
//   입력: photo_root (사진 상위 폴더), sessions (촬영 회차 목록)
//   출력: 사진 목록, skipped (명령값 없는 파일)
std::vector<PreparedPhoto> measure_all_photos(const fs::path& photo_root, const std::vector<std::string>& sessions, std::vector<std::string>& skipped) {
    std::vector<PreparedPhoto> photos;
    for (const std::string& session : sessions) {
        for (const PhotoFile& file : find_photos(photo_root / fs::u8path(session))) {
            if (!file.has_command) { skipped.push_back(session + "/" + file.name + "  (이름에 명령값이 없음)"); continue; }
            PreparedPhoto photo;
            photo.session = session;
            photo.file = file;
            photo.relative_path = file.path.lexically_relative(photo_root / fs::u8path(session)).generic_u8string();
            photo.in_range = std::abs(file.command_y_mm) <= kTrainedRangeYmm + 1e-9 && std::abs(file.command_t_deg) <= kTrainedRangeTdeg + 1e-9;
            photos.push_back(photo);
        }
    }
    ProgressBar bar("사진 재는 중", int(photos.size()));
    const QuietWarnings quiet;   // 화면 경고 생략. 결과에 알림 유지
    for (PreparedPhoto& photo : photos) {
        alignment::clear_zero();
        const alignment::Result r = alignment::measure_diagnostic(alignment::read_photo(photo.file.path.u8string()));
        photo.measured = r.status != alignment::Status::Failed;
        photo.values[0] = r.qr_shift_px;
        photo.all_screws = photo.measured;
        for (int k = 0; k < 4; ++k) {
            photo.values[k + 1] = r.screw_shift_px[k];
            if (!std::isfinite(r.screw_shift_px[k])) photo.all_screws = false;
        }
        bar.step();
    }
    bar.finish();
    return photos;
}

// apply_dataset_preprocessing: 학습용 정정과 제외 규칙 적용
//   입력: photos (사진 자료), sessions (회차), rules (규칙)
//   출력: 성공 여부, error (실패 사유). 원본 명령은 유지
bool apply_dataset_preprocessing(std::vector<PreparedPhoto>& photos, const std::vector<std::string>& sessions,
                                 const std::vector<PreprocessingRule>& rules, std::string& error) {
    std::set<std::string> applied;
    for (PreparedPhoto& photo : photos) {
        if (!apply_preprocessing_rule(rules, photo.session, photo.relative_path, photo.file.command_y_mm, photo.file.command_t_deg,
                                      photo.preprocessing, error)) return false;
        photo.preprocessing_applied = true;
        photo.in_range = std::abs(photo.training_y_mm()) <= kTrainedRangeYmm + 1e-9 && std::abs(photo.training_t_deg()) <= kTrainedRangeTdeg + 1e-9;
        if (!photo.preprocessing.rule_id.empty()) applied.insert(photo.preprocessing.rule_id);
    }
    return validate_preprocessing_coverage(rules, std::set<std::string>(sessions.begin(), sessions.end()), applied, error);
}

// 영점 추정에서 빠진 사진을 설명하는 줄 (보고서용)
struct LeftOutNotes {
    std::vector<std::string> mismatch;   // 명령 불일치 후보로 판정하여 학습과 평가에서 제외한 사진
    std::vector<std::string> kept;       // 영점 추정에서만 빼고 학습, 평가에는 쓰는 사진
};

// mark_command_mismatch: 명령 불일치 후보 제외
//   입력: photos (사진 목록), fit (회차별 차이를 계산할 식), sessions (촬영 회차 목록), left_out (영점 추정에서 제외한 사진)
//   출력: 제외 사유, photos의 학습 사용 여부 갱신
LeftOutNotes mark_command_mismatch(std::vector<PreparedPhoto>& photos, const ZeroFit& fit, const std::vector<std::string>& sessions, const std::vector<int>& left_out) {
    LeftOutNotes notes;
    for (int i = 0; i < int(photos.size()); ++i) {
        if (!left_out[i]) continue;
        const auto error = command_error(fit, photos[i], sessions);
        const std::string line = photos[i].session + "/" + photos[i].file.name + "  (사진이 보여 주는 자세 − 명령: Y " + format_number(error.first, 3) + " mm, T " +
                                 format_number(error.second, 3) + " 도)";
        photos[i].command_mismatch = std::abs(error.first) > kCommandErrorYmm || std::abs(error.second) > kCommandErrorTdeg;
        (photos[i].command_mismatch ? notes.mismatch : notes.kept).push_back(line);
    }
    return notes;
}

using SessionZeros = std::map<std::string, std::pair<double, double>>;
using Folds = std::vector<std::array<int, kRepeatCount>>;

// write_tables: 학습용 표 5개 저장
//   입력: tables (저장 폴더), photos (사진 자료), sessions (회차), zeros (영점 차이), 시험용 분할
//   출력: 없음 (common/dataset.hpp 형식의 TSV 저장)
void write_tables(const fs::path& tables, const std::vector<PreparedPhoto>& photos, const std::vector<std::string>& sessions, const SessionZeros& zeros,
                  const std::vector<int>& final_check, const Folds& folds) {
    std::ofstream photo_table(tables / "photos.tsv", std::ios::binary), feature_table(tables / "features.tsv", std::ios::binary),
        label_table(tables / "labels.tsv", std::ios::binary), split_table(tables / "splits.tsv", std::ios::binary),
        session_table(tables / "sessions.tsv", std::ios::binary);
    photo_table << "id\tsession\tpath\tcommand_y_mm\tcommand_t_deg\n";
    feature_table << "id\tmeasured\tqr\tscrew1\tscrew2\tscrew3\tscrew4\n";
    label_table << "id\tlabel_y_mm\tlabel_t_deg\n";
    split_table << "id\tusable\tfinal_check";
    for (int r = 0; r < kRepeatCount; ++r) split_table << "\tcv" << r;
    split_table << "\n";
    for (int i = 0; i < int(photos.size()); ++i) {
        const PreparedPhoto& p = photos[i];
        const auto& zero = zeros.at(p.session);
        photo_table << i << '\t' << p.session << '\t' << p.file.path.u8string() << '\t' << format_number(p.file.command_y_mm) << '\t' << format_number(p.file.command_t_deg) << '\n';
        feature_table << i << '\t' << (p.measured ? 1 : 0);
        for (double v : p.values) feature_table << '\t' << (std::isfinite(v) ? format_number(v, 12) : "nan");
        feature_table << '\n';
        label_table << i << '\t' << format_number(p.training_y_mm() - zero.first, 12) << '\t' << format_number(p.training_t_deg() - zero.second, 12) << '\n';
        split_table << i << '\t' << (p.usable() ? 1 : 0) << '\t' << final_check[i];
        for (int r = 0; r < kRepeatCount; ++r) split_table << '\t' << folds[i][r];
        split_table << '\n';
    }
    session_table << "session\tzero_y_mm\tzero_t_deg\tphotos\n";
    for (const std::string& s : sessions) {
        int count = 0;
        for (const PreparedPhoto& p : photos) count += p.session == s;
        session_table << s << '\t' << format_number(zeros.at(s).first, 8) << '\t' << format_number(zeros.at(s).second, 8) << '\t' << count << '\n';
    }
}

// write_preprocessing_table: 정정과 제외 이력 저장
//   입력: path (저장 경로), photos (사진 자료), left_out (영점 추정 제외 표시)
//   출력: 없음 (제외 사진도 기록, 원본 명령 유지)
void write_preprocessing_table(const fs::path& path, const std::vector<PreparedPhoto>& photos, const std::vector<int>& left_out) {
    std::ofstream file(path, std::ios::binary);
    file << "id\tsession\trelative_path\traw_y_mm\traw_t_deg\tverified_y_mm\tverified_t_deg\ttraining_allowed\ttraining_used\texclusion_reason\trule_id\treason\tprovenance\tzero_fit_excluded\n";
    for (size_t i = 0; i < photos.size(); ++i) {
        const auto& p = photos[i];
        std::string reason;
        const auto add = [&](const char* text) { reason += (reason.empty() ? "" : ";") + std::string(text); };
        if (!p.measured) add("measurement_failed");
        if (!p.all_screws) add("missing_screw");
        if (!p.in_range) add("out_of_training_range");
        if (!p.training_allowed()) add("rule_excluded");
        if (p.command_mismatch) add("automatic_command_mismatch");
        file << i << '\t' << p.session << '\t' << p.relative_path << '\t' << format_number(p.file.command_y_mm) << '\t' << format_number(p.file.command_t_deg)
             << '\t' << format_number(p.training_y_mm()) << '\t' << format_number(p.training_t_deg()) << '\t' << p.training_allowed() << '\t' << p.usable()
             << '\t' << reason << '\t' << p.preprocessing.rule_id << '\t' << p.preprocessing.reason << '\t' << p.preprocessing.provenance
             << '\t' << (!p.usable() || left_out.at(i) != 0) << '\n';
    }
}

// write_report: 자료 준비 보고서 저장
//   입력: path (파일 경로), 사진과 회차 정보, 영점, 제외 내역
//   출력: 없음 (사진 수, 제외 사유, 회차별 영점 저장)
void write_report(const fs::path& path, const std::vector<PreparedPhoto>& photos, const std::vector<std::string>& sessions, const SessionZeros& zeros,
                  const std::vector<int>& final_check, const LeftOutNotes& notes, const std::vector<std::string>& skipped) {
    int usable = 0, failed = 0, missing_screw = 0, out_of_range = 0, final_count = 0, rule_excluded = 0, corrected = 0;
    for (int i = 0; i < int(photos.size()); ++i) {
        const PreparedPhoto& p = photos[i];
        usable += p.usable();
        failed += !p.measured;
        missing_screw += p.measured && !p.all_screws;
        out_of_range += !p.in_range;
        final_count += final_check[i];
        rule_excluded += !p.training_allowed();
        corrected += p.training_y_mm() != p.file.command_y_mm || p.training_t_deg() != p.file.command_t_deg;
    }
    std::ofstream report(path, std::ios::binary);
    report << "사진 " << photos.size() << "장 (회차 " << sessions.size() << "개)\n"
           << "  학습, 평가에 쓰는 사진 " << usable << "장 (그중 마지막 확인용 " << final_count << "장)\n"
           << "  빼는 사진: 측정 실패 " << failed << ", 나사를 다 못 찾음 " << missing_screw << ", 명령이 학습 범위(|Y| <= 2, |T| <= 1) 밖 " << out_of_range
           << ", 규칙에 따른 학습 제외 " << rule_excluded << ", 자동 명령 불일치 후보 " << notes.mismatch.size() << "\n"
           << "  확인된 학습 명령 정정 " << corrected << "장. 원본 파일명과 photos.tsv의 평가 명령은 변경하지 않음.\n"
           << "  모든 사진의 정정, 제외 사유와 출처: preprocessing.tsv\n\n"
           << "회차별 영점 어긋남 (명령 − 학습 정답. 그날 로봇 0 과 절대 영점의 차이)\n";
    for (const std::string& s : sessions) report << "  " << s << "  Y " << format_number(zeros.at(s).first, 5) << " mm, T " << format_number(zeros.at(s).second, 5) << " 도\n";
    report << "\n명령값이 틀려 보여 뺀 사진 (자세 차이가 Y " << format_number(kCommandErrorYmm) << " mm 또는 T " << format_number(kCommandErrorTdeg)
           << " 도를 넘음. 사진을 확인하세요) " << notes.mismatch.size() << "장\n";
    for (const std::string& s : notes.mismatch) report << "  " << s << "\n";
    report << "\n영점 맞추기에서만 뺀 사진 (값이 식에서 벗어났지만 명령은 맞음. 학습, 평가에는 씀) " << notes.kept.size() << "장\n";
    for (const std::string& s : notes.kept) report << "  " << s << "\n";
    report << "\n건너뛴 파일 " << skipped.size() << "개\n";
    for (const std::string& s : skipped) report << "  " << s << "\n";
}

}   // 파일 내부

// main: 사진 측정과 학습용 표 생성
//   입력: 실행할 때 지정한 경로와 옵션(파일 상단 사용법)
//   출력: 실행 결과와 성공 여부를 나타내는 종료 코드
int main(int argc, char** argv) {
    use_korean_console();
    if (argc < 2) { std::cerr << "사용: prepare_dataset <사진 폴더> [--data <model 폴더>] [--output <표 폴더>] [--rules <규칙.tsv>]\n"; return 2; }
    const fs::path photo_root = fs::u8path(argv[1]);
    fs::path tables = photo_root / kTableFolder, rules_path = "config/data_preprocessing.tsv";
    std::string model_folder = "data/model";
    for (int a = 2; a < argc; ++a) {
        const std::string option = argv[a];
        if (a + 1 >= argc || (option != "--data" && option != "--output" && option != "--rules")) {
            std::cerr << "잘못된 옵션 또는 빠진 값: " << option << "\n"; return 2;
        }
        const std::string value = argv[++a];
        if (option == "--data") model_folder = value;
        else if (option == "--output") tables = fs::u8path(value);
        else rules_path = fs::u8path(value);
    }

    std::string error;
    std::vector<PreprocessingRule> rules;
    if (!load_preprocessing_rules(rules_path, rules, error)) { std::cerr << error << "\n"; return 1; }
    if (!make_empty_folder(tables, error)) { std::cerr << error << "\n"; return 1; }
    if (alignment::initialize(model_folder).status == alignment::Status::Failed) return 1;
    std::vector<std::string> sessions = list_subfolders(photo_root);
    sessions.erase(std::remove(sessions.begin(), sessions.end(), std::string(kTableFolder)), sessions.end());   // 회차 목록에서 표 폴더 제외
    sessions.erase(std::remove_if(sessions.begin(), sessions.end(), [&](const std::string& session) {
        return fs::absolute(photo_root / fs::u8path(session)).lexically_normal() == fs::absolute(tables).lexically_normal();
    }), sessions.end());
    if (sessions.empty()) { std::cerr << "사진 폴더가 없습니다: " << photo_root.u8string() << " (회차마다 폴더를 만들어 넣으세요)\n"; return 1; }

    std::vector<std::string> skipped;
    std::vector<PreparedPhoto> photos = measure_all_photos(photo_root, sessions, skipped);   // 1, 2
    if (!apply_dataset_preprocessing(photos, sessions, rules, error)) { std::cerr << error << "\n"; return 1; }
    if (std::none_of(photos.begin(), photos.end(), [](const PreparedPhoto& p) { return p.usable(); })) {
        std::cerr << "전처리 후 학습할 사진이 없습니다\n"; return 1;
    }
    std::vector<int> left_out;
    const ZeroFit fit = estimate_session_zeros(photos, sessions, left_out);                  // 3, 4
    const LeftOutNotes notes = mark_command_mismatch(photos, fit, sessions, left_out);
    std::vector<int> final_check;
    Folds folds;
    make_splits(photos, final_check, folds);                                                 // 5

    write_tables(tables, photos, sessions, fit.zeros, final_check, folds);
    write_preprocessing_table(tables / "preprocessing.tsv", photos, left_out);
    if (!write_preprocessing_metadata(tables, rules_path, error)) { std::cerr << error << "\n"; return 1; }
    write_report(tables / "prepare_report.txt", photos, sessions, fit.zeros, final_check, notes, skipped);
    std::ifstream echo(tables / "prepare_report.txt", std::ios::binary);
    std::cout << echo.rdbuf();
    return 0;
}
