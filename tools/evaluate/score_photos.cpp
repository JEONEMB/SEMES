// 사진 측정값과 명령값 비교
// 사용: score_photos <사진 폴더> <출력 폴더 또는 auto> [--zero <영점 사진>] [--data <model 폴더>]
//       [--tolerance-y 0.045] [--tolerance-t 0.01] [--answers 정답표.tsv]
// 인자 생략: 대화형 실행
// 정답: 파일명 또는 --answers의 file, y_mm, t_deg 열
// 영점: 회차별 설정. --zero 우선, 없으면 폴더의 정답 (0, 0) 사진 사용
// 출력: results.csv, summary.txt. 영점 사진은 채점 제외
#include "alignment.hpp"
#include "photo_picker.hpp"
#include "common/coefficient_fit.hpp"   // 통과 기준 kToleranceYmm, kToleranceTdeg
#include "common/console.hpp"
#include "common/photo_files.hpp"
#include "common/progress.hpp"
#include "common/photo_log.hpp"
#include "common/tables.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <set>
#include <regex>
#include <stdexcept>

namespace fs = std::filesystem;
using namespace tools;

namespace {

const char* const kResultFolder = "data/results/eval";   // 대화형 실행할 때 결과를 두는 곳 (그 아래에 실행마다 폴더 하나)
const char* const kNoZeroWarning =
    "경고: 영점 사진이 없어 영점 없이 채점했다. 그날 로봇 0 이 절대 영점에서 어긋난 만큼(우리 사진에서 Y 0.04 ~ 0.13 mm) 치우칠 수 있다";

struct Arguments {
    fs::path photo_folder, output, answers_file;
    fs::path zero_path;                  // 영점 사진 경로. 미지정 시 정답 (0, 0) 사진
    std::string model_folder = "data/model";
    double tolerance_y = kToleranceYmm, tolerance_t = kToleranceTdeg;   // 기본은 학습, 평가와 같은 통과 기준
};

// read_arguments: 실행할 때 받은 설정 읽기
//   입력: argc (인자 수), argv (입력값 목록)
//   출력: 성공 여부, arguments (실행 설정)
bool read_arguments(int argc, char** argv, Arguments& arguments) {
    if (argc < 3) return false;
    arguments.photo_folder = fs::u8path(argv[1]);
    arguments.output = fs::u8path(argv[2]);
    if ((argc - 3) % 2 != 0) return false;
    for (int a = 3; a + 1 < argc; a += 2) {
        const std::string option = argv[a];
        if (option != "--data" && option != "--zero" && option != "--tolerance-y" && option != "--tolerance-t" && option != "--answers") return false;
        if (option == "--data") arguments.model_folder = argv[a + 1];
        if (option == "--zero") arguments.zero_path = fs::u8path(argv[a + 1]);
        if (option == "--tolerance-y") arguments.tolerance_y = std::stod(argv[a + 1]);
        if (option == "--tolerance-t") arguments.tolerance_t = std::stod(argv[a + 1]);
        if (option == "--answers") arguments.answers_file = fs::u8path(argv[a + 1]);
    }
    return std::isfinite(arguments.tolerance_y) && arguments.tolerance_y > 0
        && std::isfinite(arguments.tolerance_t) && arguments.tolerance_t > 0;
}

struct Counts { int passed = 0, total = 0; };

// csv_cell: CSV 한 칸에 넣을 글자 정리
//   입력: text (원본 문자열)
//   출력: 쉼표와 따옴표를 처리한 저장용 문자열
std::string csv_cell(const std::string& text) {
    std::string quoted = "\"";
    for (char c : text) quoted += (c == '"') ? std::string("\"\"") : std::string(1, c);
    return quoted + "\"";
}

// number: 측정값을 정해진 자리까지 표시
//   입력: value (측정값), digits (소수 자릿수)
//   출력: 표시용 숫자 문자열
std::string number(double value, int digits) {
    if (!std::isfinite(value)) return "";
    std::ostringstream text;
    text << std::fixed << std::setprecision(digits) << value;
    return text.str();
}

// status_name: 측정 상태를 글자로 표시
//   입력: status (정상, 경고, 실패 상태)
//   출력: 상태 이름 문자열
const char* status_name(alignment::Status status) {
    return status == alignment::Status::Ok ? "정상" : status == alignment::Status::Warning ? "경고" : "실패";
}

//
// 사진 모으기: 채점할 사진, 영점 사진, 정답 없는 사진
//

// read_answer_table: 별도 정답표 적용
//   입력: answers_file, photos (사진 목록)
//   출력: 성공 여부, 수정한 photos. 표에 없으면 정답 없음
bool read_answer_table(const fs::path& answers_file, std::vector<PhotoFile>& photos) {
    Table answers;
    if (!read_table(answers_file, answers)) return false;
    const int file_column = answers.column("file"), y_column = answers.column("y_mm"), t_column = answers.column("t_deg");
    if (file_column < 0 || y_column < 0 || t_column < 0 || answers.rows.empty())
        throw std::runtime_error("정답표에는 file, y_mm, t_deg 열과 한 줄 이상의 데이터가 필요합니다. data/sample/answers.tsv를 참고하세요.");
    std::map<std::string, std::pair<double, double>> by_name;
    size_t line = 1;
    for (const auto& row : answers.rows) {
        ++line;
        if (row.size() <= size_t(std::max({file_column, y_column, t_column})))
            throw std::runtime_error("정답표 " + std::to_string(line) + "행: 열이 부족합니다. 탭으로 구분해 저장하세요.");
        const auto value = [&](int column) {
            size_t consumed = 0;
            const double number = std::stod(row[column], &consumed);
            if (!std::isfinite(number) || row[column].find_first_not_of(" \t", consumed) != std::string::npos)
                throw std::runtime_error("정답표 " + std::to_string(line) + "행: 올바른 숫자가 아닙니다.");
            return number;
        };
        const std::string name = row[file_column];
        if (name.empty() || !by_name.emplace(name, std::make_pair(value(y_column), value(t_column))).second)
            throw std::runtime_error("정답표의 파일명이 비었거나 중복됩니다: " + name);
    }
    std::set<std::string> seen;
    for (PhotoFile& p : photos) {
        if (!seen.insert(p.name).second)
            throw std::runtime_error("사진 파일명이 중복되어 정답을 연결할 수 없습니다: " + p.name + ". 파일명을 구분해 주세요.");
        const auto it = by_name.find(p.name);
        p.has_command = it != by_name.end();
        if (p.has_command) {
            p.command_y_mm = it->second.first;
            p.command_t_deg = it->second.second;
        }
    }
    for (const auto& entry : by_name)
        if (!seen.count(entry.first)) throw std::runtime_error("정답표에 있는 사진을 찾지 못했습니다: " + entry.first);
    return true;
}

// photos_at: 사진 경로 수집
//   입력: path (폴더 또는 파일)
//   출력: 폴더의 사진 목록 또는 사진 한 장
std::vector<PhotoFile> photos_at(const fs::path& path) {
    if (fs::is_directory(path)) return find_photos(path);
    std::vector<PhotoFile> photos;
    if (fs::is_regular_file(path)) {
        PhotoFile photo;
        photo.path = path;
        photo.name = path.filename().u8string();
        photos.push_back(photo);
    }
    return photos;
}

struct PhotoGroups {
    std::vector<PhotoFile> scored, zero_photos, no_answer;
    bool zero_given = false;   // 영점 사진을 따로 받았는지 (아니면 정답 (0, 0) 사진을 골라 쓴 것)
};

// group_photos: 채점 사진과 영점 사진 분리
//   입력: photos (사진 목록), zero_path (미지정 시 정답 (0, 0) 사진)
//   출력: PhotoGroups. 영점 사진은 채점 제외
PhotoGroups group_photos(const std::vector<PhotoFile>& photos, const fs::path& zero_path, const std::vector<PhotoFile>* selected_zero = nullptr) {
    PhotoGroups groups;
    groups.zero_given = selected_zero || !zero_path.empty();
    if (selected_zero) groups.zero_photos = *selected_zero;
    else if (groups.zero_given) groups.zero_photos = photos_at(zero_path);
    auto is_zero_photo = [&](const PhotoFile& p) {
        if (!groups.zero_given) return is_zero_command(p);
        for (const PhotoFile& zero : groups.zero_photos)
            if (fs::equivalent(zero.path, p.path)) return true;
        return false;
    };
    for (const PhotoFile& p : photos) {
        if (is_zero_photo(p)) { if (!groups.zero_given) groups.zero_photos.push_back(p); }
        else (p.has_command ? groups.scored : groups.no_answer).push_back(p);
    }
    return groups;
}

//
// 영점 설정과 채점
//

struct Zero {
    bool is_set = false;          // 영점을 설정했는지
    double y_mm = 0, t_deg = 0;   // 잡은 영점 (못 잡았으면 0, 0)
    std::string note;             // 요약 맨 위에 적는 설명 또는 경고
};

// set_zero: 평가용 영점 설정
//   입력: groups (영점용과 평가용 사진)
//   출력: 영점 설정 여부, Y와 T, 안내 문구
Zero set_zero(const PhotoGroups& groups) {
    Zero zero;
    if (groups.zero_photos.empty()) { zero.note = kNoZeroWarning; return zero; }
    const std::string source = std::string(groups.zero_given ? "입력한 영점 사진 " : "정답 (0, 0) 사진 ") + std::to_string(groups.zero_photos.size()) + "장";
    std::vector<cv::Mat> images;
    for (const PhotoFile& p : groups.zero_photos) images.push_back(alignment::read_photo(p.path.u8string()));
    const alignment::Result result = alignment::set_zero_from_photos(images);
    if (result.status == alignment::Status::Failed) {
        zero.note = "경고: " + source + "으로 영점을 잡지 못했다 (" + result.message + "). 영점 없이 채점했다";
        return zero;
    }
    zero.is_set = true;
    zero.y_mm = result.y_mm;
    zero.t_deg = result.t_deg;
    zero.note = "영점: " + source + "으로 잡음 (Y " + number(zero.y_mm, 4) + " mm, T " + number(zero.t_deg, 5) + " 도). 이 사진들은 채점에서 뺐다";
    return zero;
}

struct Scores {
    Counts with_zero, without_zero;
    Counts normal, warned;      // 상태별 허용오차 이내 건수. 측정 실패는 별도 집계
    int failed_count = 0;       // 측정 실패 건수 (IN/OUT 판정 대상에서 제외)
    std::map<std::string, Counts> by_folder;
    std::map<std::string, int> by_status, by_code;
    double worst_y = 0, worst_t = 0;   // 최대 오차 (영점 잡음, 실패 제외)
    std::vector<double> times_ms;      // 사진마다 측정 함수에 소요 시간 (사진 읽기 제외)
    std::array<Counts, 2> comparison, comparison_normal; // 고정된 두 기준의 전체 및 경고 없는 통과율
};

const std::array<double, 2> kComparisonY = {{0.045, 0.09}};
const double kComparisonT = 0.01;

// comparison_passed: 두 고정 기준 중 하나로 오차 판정
//   입력: failed, error_y, error_t, index (0=Y 0.045, 1=Y 0.09)
//   출력: 두 축 모두 허용오차 이내이고 측정에 성공했는지
bool comparison_passed(bool failed, double error_y, double error_t, size_t index) {
    return !failed && std::abs(error_y) <= kComparisonY[index] && std::abs(error_t) <= kComparisonT;
}

// comparison_text: 같은 측정값의 두 기준 통과율 표시
//   입력: scores (평가 집계)
//   출력: 기준별 통과율, IN/OUT, 실패, 경고 없는 OUT과 통과율
std::string comparison_text(const Scores& scores) {
    std::ostringstream text;
    text << "\n[허용오차 두 기준 비교]\n";
    for (size_t i = 0; i < kComparisonY.size(); ++i) {
        const auto& all = scores.comparison[i];
        const auto& normal = scores.comparison_normal[i];
        text << "  기준 " << i + 1 << ": Y +/-" << kComparisonY[i] << " mm, T +/-" << kComparisonT << " 도\n"
             << "    전체 통과율: " << format_rate(all.passed, all.total)
             << " / IN " << all.passed << " / OUT " << all.total - all.passed - scores.failed_count
             << " / 실패 " << scores.failed_count << "\n"
             << "    경고 없는 통과율: " << format_rate(normal.passed, normal.total)
             << " / 경고 없이 OUT " << normal.total - normal.passed << "장\n";
    }
    text << "  전체 통과율의 분모는 평가 대상 전체입니다. 실패는 IN/OUT 판정에서 제외하고 별도 표시합니다.\n";
    return text.str();
}

// measure_photos: 사진 측정과 채점
//   입력: scored (채점 사진), zero (영점 설정 결과), arguments (실행 설정)
//   출력: Scores, results.csv 저장
Scores measure_photos(const std::vector<PhotoFile>& scored, const Zero& zero, const Arguments& arguments) {
    std::ofstream csv(arguments.output / "results.csv", std::ios::binary);
    if (!csv) throw std::runtime_error("results.csv를 만들 수 없습니다. 출력 폴더의 쓰기 권한을 확인하세요.");
    csv << "\xEF\xBB\xBF";   // Excel 한글 인식용 UTF-8 BOM
    csv << "사진,폴더,정답 Y,정답 T,측정 Y,측정 T,오차 Y,오차 T,통과,영점 없이 측정 Y,영점 없이 측정 T,영점 없이 통과,상태,코드,메시지,시간 ms,Y 판정,T 판정,종합 판정,Y 여유 mm,T 여유 도,경고와 실제 오차,상세 로그,판정 Y0.045 T0.01,판정 Y0.09 T0.01\n";
    Scores scores;
    ProgressBar bar("사진 재는 중", int(scored.size()));
    const QuietWarnings quiet;   // 화면 경고 생략. results.csv에 알림 유지
    int photo_index = 0;
    for (const PhotoFile& p : scored) {
        PhotoLog log;
        std::ostringstream log_name;
        log_name << "logs/" << std::setw(5) << std::setfill('0') << ++photo_index << ".txt";
        log.text << "사진별 측정 기록\n사진: " << fs::absolute(p.path).u8string()
                 << "\n모델: " << fs::absolute(fs::u8path(arguments.model_folder)).u8string()
                 << "\n정답 출처: " << (arguments.answers_file.empty() ? "파일명" : arguments.answers_file.u8string())
                 << "\n" << zero.note << "\n[0] 사진 읽기 및 입력 검사\n";
        const auto read_start = std::chrono::steady_clock::now();
        cv::Mat image;
        try { image = alignment::read_photo(p.path.u8string()); }
        catch (const std::exception& error) { log.text << "읽기 실패: " << error.what() << "\n"; }
        const double read_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - read_start).count();
        log.text << "크기: " << image.cols << " x " << image.rows << ", 채널: " << image.channels() << "\n";
        const auto start = std::chrono::steady_clock::now();
        const alignment::Result r = (zero.is_set ? alignment::measure_alignment(image, log.callback()) : alignment::measure_diagnostic(image, log.callback()));
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        log.finish(r, read_ms, ms);
        scores.times_ms.push_back(ms);
        const std::string folder = fs::relative(p.path.parent_path(), arguments.photo_folder).u8string();
        const bool failed = r.status == alignment::Status::Failed;
        const double error_y = r.y_mm - p.command_y_mm, error_t = r.t_deg - p.command_t_deg;
        const double raw_y = r.y_mm + zero.y_mm, raw_t = r.t_deg + zero.t_deg;   // 영점을 빼기 전 값
        const bool ok = !failed && std::abs(error_y) <= arguments.tolerance_y && std::abs(error_t) <= arguments.tolerance_t;
        const bool ok_raw = !failed && std::abs(raw_y - p.command_y_mm) <= arguments.tolerance_y && std::abs(raw_t - p.command_t_deg) <= arguments.tolerance_t;
        for (size_t i = 0; i < kComparisonY.size(); ++i) {
            const bool passed = comparison_passed(failed, error_y, error_t, i);
            ++scores.comparison[i].total;
            scores.comparison[i].passed += passed;
            if (r.status == alignment::Status::Ok) {
                ++scores.comparison_normal[i].total;
                scores.comparison_normal[i].passed += passed;
            }
            log.text << "비교 기준 " << i + 1 << ": Y +/-" << kComparisonY[i]
                     << " mm, T +/-" << kComparisonT << " 도: " << (failed ? "평가 불가" : passed ? "IN" : "OUT") << '\n';
        }
        log.text << std::setprecision(10) << "5-1 원래 라벨: Y=" << p.command_y_mm << " mm, T=" << p.command_t_deg << " 도\n";
        if (!failed) log.text << "5-2 오차(예측-정답): Y=" << error_y << " mm, T=" << error_t << " 도\n"
            << "5-3 허용오차: Y=" << arguments.tolerance_y << " mm, T=" << arguments.tolerance_t << " 도\n"
            << "    여유(음수면 초과): Y=" << arguments.tolerance_y - std::abs(error_y) << " mm, T=" << arguments.tolerance_t - std::abs(error_t) << " 도\n"
            << "5-4 경고와 실제 오차: " << (r.status == alignment::Status::Ok ? "경고 없음" : "경고 있음") << " / " << (ok ? "IN" : "OUT") << "\n";
        log.text << "정답은 추론 후 비교에만 사용. 경고는 실제 오차 크기의 예측값이 아님.\n";
        log.save(arguments.output / fs::path(log_name.str()));
        if (failed) ++scores.failed_count;
        else {
            Counts& group = r.status == alignment::Status::Ok ? scores.normal : scores.warned;
            ++group.total;
            group.passed += ok;
        }
        ++scores.with_zero.total;
        scores.with_zero.passed += ok;
        ++scores.without_zero.total;
        scores.without_zero.passed += ok_raw;
        ++scores.by_folder[folder].total;
        scores.by_folder[folder].passed += ok;
        ++scores.by_status[status_name(r.status)];
        if (!r.code.empty()) ++scores.by_code[r.code];
        if (!failed) {
            scores.worst_y = std::max(scores.worst_y, std::abs(error_y));
            scores.worst_t = std::max(scores.worst_t, std::abs(error_t));
        }
        csv << csv_cell(p.name) << ',' << csv_cell(folder) << ',' << number(p.command_y_mm, 4) << ',' << number(p.command_t_deg, 4) << ','
            << number(r.y_mm, 4) << ',' << number(r.t_deg, 5) << ',' << number(error_y, 4) << ',' << number(error_t, 5) << ',' << (ok ? 1 : 0) << ','
            << number(raw_y, 4) << ',' << number(raw_t, 5) << ',' << (ok_raw ? 1 : 0) << ',' << status_name(r.status) << ',' << csv_cell(r.code) << ','
            << csv_cell(r.message) << ',' << number(ms, 1) << ','
            << (failed ? "평가 불가" : std::abs(error_y) <= arguments.tolerance_y ? "IN" : "OUT") << ','
            << (failed ? "평가 불가" : std::abs(error_t) <= arguments.tolerance_t ? "IN" : "OUT") << ','
            << (failed ? "평가 불가" : ok ? "IN" : "OUT") << ','
            << (failed ? "" : number(arguments.tolerance_y - std::abs(error_y), 6)) << ','
            << (failed ? "" : number(arguments.tolerance_t - std::abs(error_t), 6)) << ','
            << (failed ? "측정 실패" : r.status == alignment::Status::Ok ? (ok ? "경고 없음 / IN" : "경고 없음 / OUT (미검출)") : (ok ? "경고 있음 / IN" : "경고 있음 / OUT")) << ',' << log_name.str();
        for (size_t i = 0; i < kComparisonY.size(); ++i)
            csv << ',' << (failed ? "평가 불가" : comparison_passed(false, error_y, error_t, i) ? "IN" : "OUT");
        csv << '\n';
        bar.step();
    }
    bar.finish();
    csv.flush();
    if (!csv) throw std::runtime_error("results.csv 저장 실패: 디스크 공간과 쓰기 권한을 확인하세요. 결과가 완전하지 않을 수 있습니다.");
    return scores;
}

// speed_text: 측정 시간 요약
//   입력: times_ms
//   출력: 중앙값과 최댓값 문자열
std::string speed_text(std::vector<double> times_ms) {
    if (times_ms.empty()) return "잰 사진이 없음";
    std::sort(times_ms.begin(), times_ms.end());
    return "한 장에 가운데값 " + number(times_ms[times_ms.size() / 2], 1) + " ms, 가장 오래 걸린 사진 " + number(times_ms.back(), 1) + " ms";
}

// summary_text: 평가 결과 요약
//   입력: groups (사진 묶음), zero (영점), scores (점수), arguments (설정)
//   출력: 통과율과 오차 요약 문자열
std::string summary_text(const PhotoGroups& groups, const Zero& zero, const Scores& scores, const Arguments& arguments) {
    std::ostringstream summary;
    summary << zero.note << "\n\n"
            << "통과 기준: |Y 오차| <= " << arguments.tolerance_y << " mm 그리고 |T 오차| <= " << arguments.tolerance_t << " 도\n"
            << "채점한 사진 " << groups.scored.size() << "장 (영점 사진 " << groups.zero_photos.size() << "장, 정답이 없어 뺀 사진 " << groups.no_answer.size() << "장)\n\n";
    summary << comparison_text(scores) << '\n';
    const char* const scored_with = zero.is_set ? "영점 잡음" : "영점 없음";   // 영점 미설정 시 중복 점수 생략
    if (zero.is_set) summary << "통과율 (영점 잡음): " << format_rate(scores.with_zero.passed, scores.with_zero.total) << "\n";
    summary << "통과율 (영점 없음): " << format_rate(scores.without_zero.passed, scores.without_zero.total) << "\n"
            << "최대 오차 (" << scored_with << ", 실패 제외): Y " << (scores.normal.total + scores.warned.total ? number(scores.worst_y, 4) : "평가 불가")
            << " mm, T " << (scores.normal.total + scores.warned.total ? number(scores.worst_t, 5) : "평가 불가") << " 도\n\n폴더별 (" << scored_with << ")\n";
    for (const auto& f : scores.by_folder) summary << "  " << f.first << "  " << format_rate(f.second.passed, f.second.total) << "\n";
    summary << "\n상태별\n";
    for (const auto& s : scores.by_status) summary << "  " << s.first << " " << s.second << "\n";
    summary << "\n경고와 실제 오차 (영점 적용 상태는 위 평가 조건과 동일)\n"
            << "  경고 없음 / IN: " << scores.normal.passed << "장\n"
            << "  경고 없음 / OUT (미검출): " << scores.normal.total - scores.normal.passed << "장\n"
            << "  경고 있음 / IN: " << scores.warned.passed << "장\n"
            << "  경고 있음 / OUT: " << scores.warned.total - scores.warned.passed << "장\n"
            << "  측정 실패 (IN/OUT 평가 불가): " << scores.failed_count << "장\n"
            << "  경고 없는 결과의 통과율: " << format_rate(scores.normal.passed, scores.normal.total) << "\n"
            << "  IN: 두 축 모두 허용오차 이내. 경고는 실제 오차 크기의 예측값이 아닙니다.\n";
    summary << "\n코드별 (docs/03_상태와_알림.md)\n";
    for (const auto& c : scores.by_code) summary << "  " << c.first << " " << c.second << "\n";
    summary << "\n속도 (측정 함수만, 사진 읽기 제외)\n  " << speed_text(scores.times_ms) << "\n";
    if (!groups.no_answer.empty()) {
        summary << "\n정답이 없어 뺀 사진\n";
        for (const PhotoFile& p : groups.no_answer) summary << "  " << p.name << "\n";
    }
    return summary.str();
}

//
// 대화형 실행 (인자 없이 실행했을 때)
//

// initialize_model: 모델 준비
//   입력: arguments (실행 설정), guided (대화형 여부)
//   출력: 성공 여부. 대화형 실패 시 경로 재입력
bool initialize_model(Arguments& arguments, bool guided) {
    while (alignment::initialize(arguments.model_folder).status == alignment::Status::Failed) {
        if (!guided) return false;
        std::cout << "  model 폴더(center.bmp, reference.json, coefficients, checks 가 든 폴더)를 읽지 못했습니다: " << arguments.model_folder << "\n";
        arguments.model_folder = ask("  model 폴더 경로 (그만두려면 Enter) > ");
        if (arguments.model_folder.empty()) return false;
    }
    return true;
}

// collect_selection: 선택한 파일과 폴더에서 사진 수집
//   입력: paths (파일 선택 창의 경로 목록)
//   출력: 중복 경로를 제외한 사진 목록
std::vector<PhotoFile> collect_selection(const std::vector<fs::path>& paths) {
    std::vector<PhotoFile> result;
    for (const auto& path : paths) {
        for (auto photo : photos_at(path)) {
            bool duplicate = false;
            for (const auto& existing : result)
                if (fs::equivalent(existing.path, photo.path)) { duplicate = true; break; }
            if (duplicate) continue;
            photo.has_command = parse_command_from_file_name(photo.name, photo.command_y_mm, photo.command_t_deg);
            result.push_back(photo);
        }
    }
    return result;
}

// check_sessions: 경로에서 확인되는 촬영 회차 혼합 차단
//   입력: photos, centers (선택한 원본 사진)
//   출력: 혼합 시 오류. 경로에 회차 표시가 없으면 자동 판별하지 않음
void check_sessions(const std::vector<PhotoFile>& photos, const std::vector<PhotoFile>& centers) {
    const auto sessions = [](const std::vector<PhotoFile>& files) {
        std::set<std::string> names;
        const std::regex date("^(?:clean_)?([0-9]{8})$");
        const std::regex round("^clean_(round[0-9]+)$");
        for (const auto& file : files) {
            for (const auto& part : file.path.parent_path()) {
                const std::string name = part.u8string();
                std::smatch match;
                if (std::regex_match(name, match, date) || std::regex_match(name, match, round))
                    names.insert(match[1]);
            }
        }
        return names;
    };
    const auto photo_sessions = sessions(photos), center_sessions = sessions(centers);
    const auto describe = [](const std::set<std::string>& names) {
        std::string text;
        for (const auto& name : names) { if (!text.empty()) text += ", "; text += name; }
        return text;
    };
    if (photo_sessions.size() > 1 || center_sessions.size() > 1)
        throw std::runtime_error("촬영 회차가 섞여 있습니다. 평가 사진: " + describe(photo_sessions)
            + ", 기준 사진: " + describe(center_sessions)
            + ". 같은 회차의 기준 사진과 평가 사진만 다시 선택하세요. 평가를 시작하지 않았습니다.");
    if (photo_sessions.size() == 1 && center_sessions.size() == 1) {
        const auto& p = *photo_sessions.begin();
        const auto& c = *center_sessions.begin();
        if (p.size() == 8 && c.size() == 8 && p != c)
            throw std::runtime_error("기준 사진 날짜 " + c + "와 평가 사진 날짜 " + p
                + "가 다릅니다. 같은 날짜의 사진을 선택하세요. 평가를 시작하지 않았습니다.");
    }
}

// save_inputs: 평가에 선택한 입력 보관
//   입력: arguments, groups (실행 설정과 사진 목록)
//   출력: 원본 사본과 경로 기록 저장, 사진 경로를 사본으로 변경
void save_inputs(Arguments& arguments, PhotoGroups& groups) {
    std::ofstream manifest(arguments.output / "inputs.txt", std::ios::binary);
    if (!manifest) throw std::runtime_error("입력 경로 기록 파일 생성 실패");
    manifest << "선택한 원본과 보관한 사본\n";
    size_t photo_number = 0, zero_number = 0;
    const auto copy = [&](std::vector<PhotoFile>& photos, bool zero) {
        for (auto& photo : photos) {
            const size_t index = zero ? ++zero_number : ++photo_number;
            const auto destination = arguments.output / "inputs" / (zero ? "center" : "photos")
                / std::to_string(index) / photo.path.parent_path().filename() / photo.path.filename();
            fs::create_directories(destination.parent_path());
            fs::copy_file(photo.path, destination);
            manifest << "\n원본: " << fs::absolute(photo.path).u8string()
                     << "\n사본: " << fs::absolute(destination).u8string() << '\n';
            photo.path = destination;
        }
    };
    copy(groups.zero_photos, true);
    copy(groups.scored, false);
    copy(groups.no_answer, false);
    if (!arguments.answers_file.empty()) {
        const auto destination = arguments.output / "inputs/answers.tsv";
        fs::copy_file(arguments.answers_file, destination);
        manifest << "\n정답표 원본: " << fs::absolute(arguments.answers_file).u8string() << '\n';
        arguments.answers_file = destination;
    }
    manifest.close();
    if (!manifest) throw std::runtime_error("입력 경로 기록 저장 실패");
}

// default_output_folder: 기본 결과 경로 생성
//   입력: 없음
//   출력: data/results/eval/<날짜_시각>
fs::path default_output_folder() {
    const std::time_t now = std::time(nullptr);
    char stamp[32];
    std::strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%S", std::localtime(&now));
    const auto base = fs::path(kResultFolder) / fs::u8path(std::string(stamp));
    fs::path output = base;
    for (int suffix = 2; fs::exists(output); ++suffix) output = fs::u8path(base.u8string() + "_" + std::to_string(suffix));
    return output;
}

// confirm: 평가 시작 확인
//   입력: arguments (실행 설정), groups (용도별 사진 묶음)
//   출력: 진행 동의 여부
bool confirm(const Arguments& arguments, const PhotoGroups& groups) {
    std::cout << "\n[4] 평가 시작 확인\n"
              << "  평가할 사진: " << groups.scored.size() << "장 (" << arguments.photo_folder.u8string() << ")\n";
    if (!groups.no_answer.empty()) std::cout << "  이름에 명령값이 없어 건너뛰는 사진: " << groups.no_answer.size() << "장\n";
    if (groups.zero_photos.empty()) std::cout << "  영점 사진: 없음 → 영점 없이 채점합니다. 통과율이 크게 낮게 나올 수 있습니다.\n";
    else std::cout << "  영점 사진: " << groups.zero_photos.size() << "장 (" << (groups.zero_given ? arguments.zero_path.u8string() : "평가할 폴더 안의 Y 0, T 0 사진") << ")\n";
    std::cout << "  통과 기준: |Y 오차| <= " << arguments.tolerance_y << " mm 그리고 |T 오차| <= " << arguments.tolerance_t << " 도\n"
              << "  결과를 저장할 곳: " << fs::absolute(arguments.output).u8string() << "\n";
    return ask_yes("  시작할까요? (Enter 또는 y = 시작, n = 그만) > ");
}

}   // 파일 내부

// run_evaluation: 사진 측정값과 명령값 비교
//   입력: 실행할 때 지정한 경로와 옵션(파일 상단 사용법)
//   출력: 실행 결과와 성공 여부를 나타내는 종료 코드
int run_evaluation(int argc, char** argv) {
    use_korean_console();
    const bool guided = argc == 1;
    Arguments arguments;
    std::vector<PhotoFile> chosen_photos, chosen_zero;
    if (guided) {
        std::cout << "정렬 측정 평가: 사진을 재서 파일 이름의 명령값과 비교합니다.\n";
        std::cout << "영점은 평가 원점 (Y 0, T 0)에서 촬영한 사진을 선택하세요.\n";
        chosen_zero = collect_selection(pick_photos(true));
        if (chosen_zero.empty()) { std::cout << "기준 사진(center) 선택을 취소했거나 사진이 없습니다.\n"; return 0; }
        chosen_photos = collect_selection(pick_photos(false));
        if (chosen_photos.empty()) { std::cout << "평가 사진 선택을 취소했거나 사진이 없습니다.\n"; return 0; }
        check_sessions(chosen_photos, chosen_zero);
        arguments.answers_file = pick_answers();
        arguments.output = default_output_folder();
        arguments.photo_folder = arguments.output / "inputs/photos";
        arguments.zero_path = arguments.output / "inputs/center";
    } else if (!read_arguments(argc, argv, arguments)) {
        std::cerr << "사용: score_photos                                  (물어보며 진행)\n"
                  << "      score_photos <사진 폴더> <출력 폴더 또는 auto> [--zero <영점 사진 폴더 또는 파일>] [--data <model 폴더>] [--tolerance-y 0.045] [--tolerance-t 0.01] [--answers 정답표.tsv]\n"
                  << "      auto: data/results/eval 아래에 새 결과 폴더를 자동 생성\n";
        return 2;
    }

    if (arguments.output == "auto") arguments.output = default_output_folder();
    if (!initialize_model(arguments, guided)) return 1;

    std::vector<PhotoFile> photos = guided ? chosen_photos : find_photos(arguments.photo_folder);
    if (!arguments.answers_file.empty() && !read_answer_table(arguments.answers_file, photos)) {
        std::cerr << "정답표를 읽지 못했습니다: " << arguments.answers_file.u8string() << "\n";
        return 1;
    }
    PhotoGroups groups = group_photos(photos, arguments.zero_path, guided ? &chosen_zero : nullptr);
    if (groups.scored.empty()) {
        std::cerr << "평가할 사진이 없습니다. 사진 폴더와 정답표의 file 열을 확인하세요. 영점 사진은 평가에서 제외됩니다.\n";
        return 1;
    }
    if (groups.zero_given && groups.zero_photos.empty()) {
        std::cerr << "지정한 영점 경로에서 사진을 찾지 못했습니다. 경로를 확인하세요.\n";
        return 1;
    }
    if (guided && !confirm(arguments, groups)) { std::cout << "그만두었습니다. 아무것도 저장하지 않았습니다.\n"; return 0; }

    std::string error;
    const Zero zero = set_zero(groups);
    if (!groups.zero_photos.empty() && !zero.is_set) {
        std::cerr << "영점 설정 실패로 평가를 중단합니다. 영점 사진의 크기, QR, 나사 가림과 촬영 자세를 확인하세요.\n";
        return 1;
    }
    if (!make_empty_folder(arguments.output, error)) { std::cerr << error << "\n"; return 1; }
    if (guided) save_inputs(arguments, groups);
    std::cout << "\n평가 시작: " << groups.scored.size() << "장\n"
              << "정답 출처: " << (arguments.answers_file.empty() ? "사진 파일명" : arguments.answers_file.u8string()) << "\n"
              << "저장할 폴더: " << fs::absolute(arguments.output).u8string() << "\n";
    const Scores scores = measure_photos(groups.scored, zero, arguments);

    const std::string summary = "사진 폴더: " + fs::absolute(arguments.photo_folder).u8string()
        + "\n모델 폴더: " + fs::absolute(fs::u8path(arguments.model_folder)).u8string()
        + "\n정답 출처: " + (arguments.answers_file.empty() ? "사진 파일명" : fs::absolute(arguments.answers_file).u8string())
        + "\n\n" + summary_text(groups, zero, scores, arguments);
    std::cout << "\n============================================================\n[1] 평가 결과\n"
              << comparison_text(scores)
              << "\n[설정 기준 결과] Y +/-" << arguments.tolerance_y << " mm, T +/-" << arguments.tolerance_t << " 도\n"
              << "  사진: " << groups.scored.size() << "장 / IN: " << scores.with_zero.passed
              << " / OUT: " << scores.normal.total + scores.warned.total - scores.with_zero.passed
              << " / 실패: " << scores.failed_count << "\n"
              << "  경고 없이 OUT: " << scores.normal.total - scores.normal.passed << "장\n"
              << "  경고 없는 결과의 통과율: " << format_rate(scores.normal.passed, scores.normal.total) << "\n"
              << "  속도(측정 및 로그 작성, 사진 읽기와 파일 저장 제외): " << speed_text(scores.times_ms) << "\n"
              << "  영점: " << (zero.is_set ? "적용" : "미설정 - 평가 기준 확인 필요") << "\n";
    std::ofstream report(arguments.output / "summary.txt", std::ios::binary);
    report << summary;
    report.flush();
    if (!report) throw std::runtime_error("summary.txt 저장 실패. 디스크 공간과 쓰기 권한을 확인하세요.");
        std::cout << "\n[2] 결과 파일\n"
                  << "  결과를 저장한 곳: " << fs::absolute(arguments.output).u8string() << "\n"
                  << "    results.csv   엑셀로 열기: 정답, 예측, 오차, IN/OUT, 여유, 경고\n"
                  << "    summary.txt   위 요약 (통과율, 최대 오차, 상태별 개수)\n"
                  << "    logs/         사진별 단계, 하위 단계, 수치, 경고, 실패 기록 (CSV의 상세 로그 열 참조)\n"
                  << "  먼저 results.csv의 '경고와 실제 오차'에서 '경고 없음 / OUT (미검출)'을 확인하세요.\n"
                  << "  여유가 양수면 기준 안, 음수면 그 절댓값만큼 초과입니다. 측정 실패는 평가 불가입니다.\n"
                  << "\n[3] 폴더 열기\n  explorer.exe \"" << fs::absolute(arguments.output).u8string() << "\"\n"
                  << "  다른 날 사진을 평가하려면 이 프로그램을 다시 실행하세요.\n";
    if (guided) {
        ask("Enter 를 누르면 끝납니다.");
    }
    return 0;
}

// main: 입력 및 파일 처리 오류를 안내하고 종료
//   입력: 명령 인자 (파일 상단 사용법)
//   출력: 0=평가 완료(모든 사진의 IN을 뜻하지 않음), 1=처리 실패, 2=사용법 오류
int main(int argc, char** argv) {
    try { return run_evaluation(argc, argv); }
    catch (const std::exception& error) {
        std::cerr << "평가를 완료하지 못했습니다: " << error.what()
                  << "\n입력 경로와 정답표(file, y_mm, t_deg)를 확인하세요. 기존 결과는 덮어쓰지 않습니다.\n";
        return 1;
    }
}
