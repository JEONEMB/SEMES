// 사진 측정값과 명령값 비교
// 사용: score_photos <사진 폴더> <출력 폴더> [--zero <영점 사진>] [--data <model 폴더>]
//       [--tolerance-y 0.045] [--tolerance-t 0.01] [--answers 정답표.tsv]
// 인자 생략: 대화형 실행
// 정답: 파일명 또는 --answers의 file, y_mm, t_deg 열
// 영점: 회차별 설정. --zero 우선, 없으면 폴더의 정답 (0, 0) 사진 사용
// 출력: results.csv, summary.txt. 영점 사진은 채점 제외
#include "alignment.hpp"
#include "common/coefficient_fit.hpp"   // 통과 기준 kToleranceYmm, kToleranceTdeg
#include "common/console.hpp"
#include "common/photo_files.hpp"
#include "common/progress.hpp"
#include "common/tables.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>

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
    for (int a = 3; a + 1 < argc; ++a) {
        const std::string option = argv[a];
        if (option == "--data") arguments.model_folder = argv[a + 1];
        if (option == "--zero") arguments.zero_path = fs::u8path(argv[a + 1]);
        if (option == "--tolerance-y") arguments.tolerance_y = std::stod(argv[a + 1]);
        if (option == "--tolerance-t") arguments.tolerance_t = std::stod(argv[a + 1]);
        if (option == "--answers") arguments.answers_file = fs::u8path(argv[a + 1]);
    }
    return true;
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
    std::map<std::string, std::pair<double, double>> by_name;
    for (const auto& row : answers.rows) by_name[row.at(0)] = {std::stod(row.at(1)), std::stod(row.at(2))};
    for (PhotoFile& p : photos) {
        const auto it = by_name.find(p.name);
        p.has_command = it != by_name.end();
        if (p.has_command) {
            p.command_y_mm = it->second.first;
            p.command_t_deg = it->second.second;
        }
    }
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
PhotoGroups group_photos(const std::vector<PhotoFile>& photos, const fs::path& zero_path) {
    PhotoGroups groups;
    groups.zero_given = !zero_path.empty();
    if (groups.zero_given) groups.zero_photos = photos_at(zero_path);
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
    std::map<std::string, Counts> by_folder;
    std::map<std::string, int> by_status, by_code;
    double worst_y = 0, worst_t = 0;   // 최대 오차 (영점 잡음, 실패 제외)
    std::vector<double> times_ms;      // 사진마다 측정 함수에 소요 시간 (사진 읽기 제외)
};

// measure_photos: 사진 측정과 채점
//   입력: scored (채점 사진), zero (영점 설정 결과), arguments (실행 설정)
//   출력: Scores, results.csv 저장
Scores measure_photos(const std::vector<PhotoFile>& scored, const Zero& zero, const Arguments& arguments) {
    std::ofstream csv(arguments.output / "results.csv", std::ios::binary);
    csv << "\xEF\xBB\xBF";   // Excel 한글 인식용 UTF-8 BOM
    csv << "사진,폴더,정답 Y,정답 T,측정 Y,측정 T,오차 Y,오차 T,통과,영점 없이 측정 Y,영점 없이 측정 T,영점 없이 통과,상태,코드,메시지,시간 ms\n";
    Scores scores;
    ProgressBar bar("사진 재는 중", int(scored.size()));
    const QuietWarnings quiet;   // 화면 경고 생략. results.csv에 알림 유지
    for (const PhotoFile& p : scored) {
        const cv::Mat image = alignment::read_photo(p.path.u8string());
        const auto start = std::chrono::steady_clock::now();
        const alignment::Result r = alignment::measure_alignment(image);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        scores.times_ms.push_back(ms);
        const std::string folder = fs::relative(p.path.parent_path(), arguments.photo_folder).u8string();
        const bool failed = r.status == alignment::Status::Failed;
        const double error_y = r.y_mm - p.command_y_mm, error_t = r.t_deg - p.command_t_deg;
        const double raw_y = r.y_mm + zero.y_mm, raw_t = r.t_deg + zero.t_deg;   // 영점을 빼기 전 값
        const bool ok = !failed && std::abs(error_y) <= arguments.tolerance_y && std::abs(error_t) <= arguments.tolerance_t;
        const bool ok_raw = !failed && std::abs(raw_y - p.command_y_mm) <= arguments.tolerance_y && std::abs(raw_t - p.command_t_deg) <= arguments.tolerance_t;
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
            << csv_cell(r.message) << ',' << number(ms, 1) << '\n';
        bar.step();
    }
    bar.finish();
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
    const char* const scored_with = zero.is_set ? "영점 잡음" : "영점 없음";   // 영점 미설정 시 중복 점수 생략
    if (zero.is_set) summary << "통과율 (영점 잡음): " << format_rate(scores.with_zero.passed, scores.with_zero.total) << "\n";
    summary << "통과율 (영점 없음): " << format_rate(scores.without_zero.passed, scores.without_zero.total) << "\n"
            << "최대 오차 (" << scored_with << ", 실패 제외): Y " << number(scores.worst_y, 4) << " mm, T " << number(scores.worst_t, 5) << " 도\n\n폴더별 (" << scored_with << ")\n";
    for (const auto& f : scores.by_folder) summary << "  " << f.first << "  " << format_rate(f.second.passed, f.second.total) << "\n";
    summary << "\n상태별\n";
    for (const auto& s : scores.by_status) summary << "  " << s.first << " " << s.second << "\n";
    summary << "\n코드별 (README 5절)\n";
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

// ask_zero_photos: 영점 사진 경로 입력
//   입력: 키보드 입력
//   출력: 경로. Enter이면 미지정
fs::path ask_zero_photos() {
    std::cout << "\n[1/3] 영점 사진\n"
              << "  영점을 잡지 않으면 그날 로봇의 0 위치와 기준의 차이만큼 모든 값이 치우칩니다 (우리가 받은 사진에서 Y 0.04 ~ 0.13 mm).\n"
              << "  우리가 받은 4회차 사진으로 재 보면 통과율이 영점 없이 31.9%, 영점 사진 5장으로 95.3% 였습니다.\n"
              << "  영점 사진 = 로봇을 명령 0 (Y 0, T 0) 으로 보내 찍은 사진 5~10장. 평가할 사진과 같은 날 찍은 것이어야 합니다.\n"
              << "  - 영점 사진이 든 폴더(또는 사진 한 장)의 경로를 입력하세요.\n"
              << "  - 평가할 폴더 안에 \"Y+0.00 T+0.00 (1).bmp\" 처럼 이름으로 들어 있으면 그냥 Enter 를 누르세요.\n";
    while (true) {
        const std::string answer = ask("  영점 사진 경로 > ");
        if (answer.empty()) return fs::path();
        const fs::path path = fs::u8path(answer);
        const size_t count = photos_at(path).size();
        if (count > 0) { std::cout << "  → 사진 " << count << "장을 찾았습니다.\n"; return path; }
        std::cout << "  그 경로에서 사진(.bmp, .png)을 찾지 못했습니다. 다시 입력하세요 (영점 없이 하려면 Enter).\n";
    }
}

// ask_photo_folder: 파일명 규칙 안내와 사진 폴더 입력
//   입력: 키보드 입력
//   출력: 사진 폴더 경로
fs::path ask_photo_folder() {
    std::cout << "\n[2/3] 평가할 사진\n"
              << "  사진 파일 이름에 로봇 명령값이 들어 있어야 합니다. 이 값을 정답으로 씁니다 (Y 는 mm, T 는 도, 부호를 꼭 붙임).\n"
              << "    Y+0.05.bmp              → Y 0.05, T 0\n"
              << "    T-0.30.bmp              → Y 0, T -0.30\n"
              << "    Y+0.05 T+0.01 (2).bmp   → Y 0.05, T 0.01 (괄호 안 번호는 같은 자세를 여러 번 찍은 순번)\n"
              << "  - 하루(한 회차)에 찍은 사진만 넣으세요. 영점이 날마다 달라서 여러 날 사진은 날마다 따로 평가해야 합니다.\n"
              << "  - 하위 폴더가 있어도 됩니다. 이름에 명령값이 없는 사진은 건너뛰고 결과에 적습니다.\n";
    while (true) {
        const fs::path folder = fs::u8path(ask("  평가할 폴더 경로 > "));
        if (fs::is_directory(folder) && !find_photos(folder).empty()) return folder;
        std::cout << "  그 폴더에서 사진(.bmp, .png)을 찾지 못했습니다. 다시 입력하세요.\n";
    }
}

// default_output_folder: 기본 결과 경로 생성
//   입력: photo_folder (사진 폴더)
//   출력: data/results/eval/<사진 폴더명>_<날짜_시각>
fs::path default_output_folder(const fs::path& photo_folder) {
    const std::time_t now = std::time(nullptr);
    char stamp[32];
    std::strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%S", std::localtime(&now));
    return fs::path(kResultFolder) / fs::u8path(fs::absolute(photo_folder).lexically_normal().filename().u8string() + "_" + stamp);
}

// confirm: 평가 시작 확인
//   입력: arguments (실행 설정), groups (용도별 사진 묶음)
//   출력: 진행 동의 여부
bool confirm(const Arguments& arguments, const PhotoGroups& groups) {
    std::cout << "\n[3/3] 확인\n"
              << "  평가할 사진: " << groups.scored.size() << "장 (" << arguments.photo_folder.u8string() << ")\n";
    if (!groups.no_answer.empty()) std::cout << "  이름에 명령값이 없어 건너뛰는 사진: " << groups.no_answer.size() << "장\n";
    if (groups.zero_photos.empty()) std::cout << "  영점 사진: 없음 → 영점 없이 채점합니다. 통과율이 크게 낮게 나올 수 있습니다.\n";
    else std::cout << "  영점 사진: " << groups.zero_photos.size() << "장 (" << (groups.zero_given ? arguments.zero_path.u8string() : "평가할 폴더 안의 Y 0, T 0 사진") << ")\n";
    std::cout << "  통과 기준: |Y 오차| <= " << arguments.tolerance_y << " mm 그리고 |T 오차| <= " << arguments.tolerance_t << " 도\n"
              << "  결과를 저장할 곳: " << fs::absolute(arguments.output).u8string() << "\n";
    return ask_yes("  시작할까요? (Enter 또는 y = 시작, n = 그만) > ");
}

}   // 파일 내부

// main: 사진 측정값과 명령값 비교
//   입력: 실행할 때 지정한 경로와 옵션(파일 상단 사용법)
//   출력: 실행 결과와 성공 여부를 나타내는 종료 코드
int main(int argc, char** argv) {
    use_korean_console();
    const bool guided = argc == 1;
    Arguments arguments;
    if (guided) {
        std::cout << "정렬 측정 평가: 사진을 재서 파일 이름의 명령값과 비교합니다.\n";
        arguments.zero_path = ask_zero_photos();
        arguments.photo_folder = ask_photo_folder();
        arguments.output = default_output_folder(arguments.photo_folder);
    } else if (!read_arguments(argc, argv, arguments)) {
        std::cerr << "사용: score_photos                                  (물어보며 진행)\n"
                  << "      score_photos <사진 폴더> <출력 폴더> [--zero <영점 사진 폴더 또는 파일>] [--data <model 폴더>] [--tolerance-y 0.045] [--tolerance-t 0.01] [--answers 정답표.tsv]\n";
        return 2;
    }

    if (!initialize_model(arguments, guided)) return 1;

    std::vector<PhotoFile> photos = find_photos(arguments.photo_folder);
    if (!arguments.answers_file.empty() && !read_answer_table(arguments.answers_file, photos)) {
        std::cerr << "정답표를 읽지 못했습니다: " << arguments.answers_file.u8string() << "\n";
        return 1;
    }
    const PhotoGroups groups = group_photos(photos, arguments.zero_path);
    if (guided && !confirm(arguments, groups)) { std::cout << "그만두었습니다. 아무것도 저장하지 않았습니다.\n"; return 0; }

    std::string error;
    if (!make_empty_folder(arguments.output, error)) { std::cerr << error << "\n"; return 1; }
    if (guided) std::cout << "\n평가 중입니다 (사진 한 장에 약 0.1초)...\n\n";
    const Zero zero = set_zero(groups);
    const Scores scores = measure_photos(groups.scored, zero, arguments);

    const std::string summary = summary_text(groups, zero, scores, arguments);
    std::cout << summary;
    std::ofstream(arguments.output / "summary.txt", std::ios::binary) << summary;
    if (guided) {
        std::cout << "\n평가가 완료되었습니다.\n"
                  << "  결과를 저장한 곳: " << fs::absolute(arguments.output).u8string() << "\n"
                  << "    results.csv   사진마다 한 줄 (정답, 측정값, 오차, 통과 여부). 엑셀로 열면 됩니다\n"
                  << "    summary.txt   위 요약 (통과율, 최대 오차, 상태별 개수)\n"
                  << "  다른 날 사진을 평가하려면 이 프로그램을 다시 실행하세요.\n";
        ask("Enter 를 누르면 끝납니다.");
    }
    return 0;
}
