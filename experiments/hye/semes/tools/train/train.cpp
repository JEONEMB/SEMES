// 대화형 학습 실행. 기존 data/model 유지
// 사용: train
// 순서: prepare_dataset -> fit_coefficients -> evaluate_accuracy
// 출력: data/results/train/<날짜_시각>/
#include "common/console.hpp"
#include "common/data_preprocessing.hpp"
#include "common/photo_files.hpp"
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace tools;

namespace {

const char* const kDefaultPhotoFolder = "data/new_photos";
const char* const kTableFolder = "tables";                  // prepare_dataset이 사진 폴더 안에 표를 쓰는 곳
const char* const kResultFolder = "data/results/train";     // 새 계산식을 두는 곳 (그 아래에 실행마다 폴더 하나)
const double kSecondsPerPhoto = 0.15;                       // 사진 한 장을 읽고 측정하는 데 걸리는 대략의 시간 (예상 시간 안내용)

// 촬영 회차 이름과 사진 수
struct Session {
    std::string name;
    int photos = 0;
};

// list_sessions: 촬영 회차 수집
//   입력: photo_folder (사진 폴더)
//   출력: 회차별 명령 사진 수. tables 폴더 제외
std::vector<Session> list_sessions(const fs::path& photo_folder) {
    std::vector<Session> sessions;
    for (const std::string& name : list_subfolders(photo_folder)) {
        if (name == kTableFolder) continue;
        Session session;
        session.name = name;
        for (const PhotoFile& photo : find_photos(photo_folder / fs::u8path(name))) session.photos += photo.has_command;
        if (session.photos > 0) sessions.push_back(session);
    }
    return sessions;
}

// total_photos: 전체 사진 수 합계
//   입력: sessions (회차별 사진 수)
//   출력: 전체 사진 수
int total_photos(const std::vector<Session>& sessions) {
    int total = 0;
    for (const Session& session : sessions) total += session.photos;
    return total;
}

// ask_photo_folder: 사진 배치 안내와 경로 입력
//   입력: 키보드 입력
//   출력: 사진 폴더 경로
fs::path ask_photo_folder() {
    std::cout << "\n[1/2] 학습할 사진\n"
              << "  사진 폴더 안에 촬영한 날(회차)마다 폴더를 하나씩 만들고, 받은 사진을 이름 그대로 넣습니다.\n"
              << "    data/new_photos/20261014/Y+0.05.bmp              → Y 0.05, T 0\n"
              << "    data/new_photos/20261014/T-0.30.bmp              → Y 0, T -0.30\n"
              << "    data/new_photos/20261020/Y+0.05 T+0.01 (2).bmp   → Y 0.05, T 0.01 (괄호 안 번호는 같은 자세의 순번)\n"
              << "  - 파일 이름의 원본 명령(Y 는 mm, T 는 도)은 보존하고, 확인된 학습 정정과 제외 규칙을 적용합니다.\n"
              << "  - 날마다 로봇의 0 위치가 조금씩 달라서 회차를 폴더로 나눠야 합니다. 그 차이는 프로그램이 알아서 구합니다.\n"
              << "  - 이름에 명령값이 없는 사진은 건너뛰고 보고서에 적습니다. 자세한 설명은 data/new_photos/README.md\n";
    while (true) {
        std::string answer = ask(std::string("  사진 폴더 경로 (Enter = ") + kDefaultPhotoFolder + ") > ");
        if (answer.empty()) answer = kDefaultPhotoFolder;
        const fs::path folder = fs::u8path(answer);
        if (!list_sessions(folder).empty()) return folder;
        std::cout << "  그 폴더에서 회차 폴더와 사진(.bmp, .png)을 찾지 못했습니다. 다시 입력하세요.\n";
    }
}

// new_result_folder: 이번 실행의 결과 경로 생성
//   입력: 없음
//   출력: 현재 날짜와 시각을 붙인 저장 경로
fs::path new_result_folder() {
    const std::time_t now = std::time(nullptr);
    char stamp[32];
    std::strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%S", std::localtime(&now));
    return fs::path(kResultFolder) / stamp;
}

// confirm: 학습 시작 확인
//   입력: photo_folder (사진 폴더), result_folder (결과 폴더)
//   출력: 진행 동의 여부, measure_again (재측정 여부)
bool confirm(const fs::path& photo_folder, const fs::path& result_folder, bool& measure_again) {
    const std::vector<Session> sessions = list_sessions(photo_folder);
    std::cout << "\n[2/2] 확인\n  회차 " << sessions.size() << "개, 사진 " << total_photos(sessions) << "장 (" << photo_folder.u8string() << ")\n";
    for (const Session& session : sessions) std::cout << "    " << session.name << "  " << session.photos << "장\n";
    measure_again = true;
    if (fs::exists(photo_folder / kTableFolder / "photos.tsv")) {
        std::string error;
        if (validate_preprocessing_cache(photo_folder / kTableFolder, "config/data_preprocessing.tsv", error)) {
            std::cout << "  현재 전처리 규칙으로 만든 표가 있습니다 (" << (photo_folder / kTableFolder).u8string() << ").\n"
                      << "  사진을 더 넣거나 바꾸지 않았으면 다시 재지 않아도 됩니다.\n";
            measure_again = !ask_yes("  있는 표를 그대로 쓸까요? (Enter 또는 y = 그대로 씀, n = 새 폴더에 다시 잼) > ");
        } else {
            std::cout << "  기존 표를 재사용할 수 없습니다: " << error << "\n"
                      << "  이번 결과 폴더에 표를 새로 만듭니다. 기존 표는 보존합니다.\n";
        }
    }
    const int seconds = int(kSecondsPerPhoto * total_photos(sessions));
    std::cout << "  할 일: " << (measure_again ? "사진 재기 (약 " + std::to_string(seconds / 60) + "분 " + std::to_string(seconds % 60) + "초) → " : "") << "계수 만들기 → 정확도 확인\n"
              << "  새 계수를 저장할 곳: " << fs::absolute(result_folder).u8string() << "\n"
              << "  지금 쓰는 계수(data/model)는 바꾸지 않습니다.\n";
    return ask_yes("  시작할까요? (Enter 또는 y = 시작, n = 그만) > ");
}

// run_program: 같은 폴더의 도구 실행
//   입력: program_folder (도구 폴더), name (실행 파일명), arguments (전달할 경로 목록)
//   출력: 실행 성공 여부
bool run_program(const fs::path& program_folder, const std::string& name, const std::vector<fs::path>& arguments) {
#ifdef _WIN32
    std::wstring command = L"\"\"" + (program_folder / (name + ".exe")).wstring() + L"\"";
    for (const fs::path& argument : arguments) command += L" \"" + argument.wstring() + L"\"";
    command += L"\"";   // cmd 실행용 전체 명령 따옴표
    return _wsystem(command.c_str()) == 0;
#else
    std::string command = "\"" + (program_folder / name).string() + "\"";
    for (const fs::path& argument : arguments) command += " \"" + argument.string() + "\"";
    return std::system(command.c_str()) == 0;
#endif
}

// print_finished: 학습 완료 후 사용할 경로 안내
//   입력: photo_folder (사진 폴더), result_folder (결과 폴더)
//   출력: 없음 (결과 위치와 후속 실행 방법 표시)
void print_finished(const fs::path& tables, const fs::path& result_folder) {
    const std::string model = (result_folder / "model").generic_u8string();
    std::cout << "\n학습이 완료되었습니다.\n"
              << "  새 계수:        " << fs::absolute(result_folder / "model").u8string() << "\n"
              << "                  (coefficients/ 계수 16개, checks/ 점검식 11개, reference.json, center.bmp, fit_report.txt)\n"
              << "  정확도 보고서:  " << fs::absolute(result_folder / "accuracy" / "accuracy_report.txt").u8string() << "\n"
              << "  사진을 잰 표:   " << fs::absolute(tables).u8string() << "  (preprocessing.tsv 에 정정, 제외 사유와 출처)\n\n"
              << "  다음에 할 수 있는 일\n"
              << "  - 새 계수로 하루치 사진을 평가:  score_photos <사진 폴더> <출력 폴더> --data " << model << "\n"
              << "  - 새 계수를 쓰기로 했으면:  data/model 폴더를 다른 이름으로 보관한 뒤, " << model << " 안의 파일을 data/model 로 복사\n"
              << "  속도는 계수를 바꿔도 달라지지 않습니다 (재는 방법이 같아서). 속도만 재려면 measure_speed 를 씁니다.\n";
}

}   // 파일 내부

// main: 대화형 학습 실행. 기존 data/model 유지
//   입력: 키보드로 지정할 사진 폴더와 실행 여부
//   출력: 실행 결과와 성공 여부를 나타내는 종료 코드
int main(int, char** argv) {
    use_korean_console();
    const fs::path program_folder = fs::absolute(fs::path(argv[0])).parent_path();
    std::cout << "학습: 받은 사진으로 계수를 새로 만듭니다.\n";
    const fs::path photo_folder = ask_photo_folder();
    const fs::path result_folder = new_result_folder();
    fs::path tables = photo_folder / kTableFolder;
    bool measure_again = true;
    if (!confirm(photo_folder, result_folder, measure_again)) { std::cout << "그만두었습니다. 아무것도 바꾸지 않았습니다.\n"; return 0; }

    if (measure_again) {
        tables = result_folder / "tables";   // 기존 표를 지우지 않고 이번 실행의 결과 폴더에 준비
        std::cout << "\n[1/3] 사진 재기\n" << std::flush;
        if (!run_program(program_folder, "prepare_dataset", {photo_folder, "--output", tables})) { std::cout << "사진 재기에서 멈췄습니다. 위 메시지를 확인하세요.\n"; return 1; }
    }
    std::cout << "\n[2/3] 계수 만들기\n" << std::flush;
    if (!run_program(program_folder, "fit_coefficients", {tables, result_folder / "model"})) { std::cout << "계수 만들기에서 멈췄습니다. 위 메시지를 확인하세요.\n"; return 1; }
    std::cout << "\n[3/3] 정확도 확인 (학습에 안 쓴 사진으로 시험)\n" << std::flush;
    if (!run_program(program_folder, "evaluate_accuracy", {tables, result_folder / "accuracy"})) { std::cout << "정확도 확인에서 멈췄습니다. 위 메시지를 확인하세요.\n"; return 1; }

    print_finished(tables, result_folder);
    ask("Enter 를 누르면 끝납니다.");
    return 0;
}
