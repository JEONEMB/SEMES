// 학습: 받은 사진으로 계수를 새로 만든다 (물어보며 진행). 지금 쓰는 data/model 은 바꾸지 않는다.
//
// 사용: train        (인자 없이 실행. 사진 폴더를 묻고 확인한 뒤 아래 세 프로그램을 차례로 돌린다)
//
//   1. prepare_dataset    사진을 모두 재서 표로 저장                  <사진 폴더>/tables/
//   2. fit_coefficients   표로 새 계수와 점검식을 만든다              data/results/train/<날짜_시각>/model/
//   3. evaluate_accuracy  학습에 안 쓴 사진으로 새 방식의 정확도 확인  data/results/train/<날짜_시각>/accuracy/
//
// 한 단계만 다시 돌리고 싶으면 세 프로그램을 따로 실행해도 된다 (각 프로그램 파일 맨 위의 사용법).
#include "common/console.hpp"
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
const char* const kTableFolder = "tables";                  // prepare_dataset 이 사진 폴더 안에 표를 쓰는 곳
const char* const kResultFolder = "data/results/train";     // 새 계수를 두는 곳 (그 아래에 실행마다 폴더 하나)
const double kSecondsPerPhoto = 0.15;                       // 사진 한 장을 읽고 재는 데 걸리는 대략의 시간 (예상 시간 안내용)

// 한 회차(촬영한 날)의 이름과 사진 수
struct Session {
    std::string name;
    int photos = 0;
};

// 사진 폴더 바로 아래의 회차 폴더들 (표 폴더는 뺀다). 이름에 명령값이 있는 사진만 센다.
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

int total_photos(const std::vector<Session>& sessions) {
    int total = 0;
    for (const Session& session : sessions) total += session.photos;
    return total;
}

// [1/2] 사진 넣는 규칙을 알려 주고 사진 폴더를 묻는다.
fs::path ask_photo_folder() {
    std::cout << "\n[1/2] 학습할 사진\n"
              << "  사진 폴더 안에 촬영한 날(회차)마다 폴더를 하나씩 만들고, 받은 사진을 이름 그대로 넣습니다.\n"
              << "    data/new_photos/20261014/Y+0.05.bmp              → Y 0.05, T 0\n"
              << "    data/new_photos/20261014/T-0.30.bmp              → Y 0, T -0.30\n"
              << "    data/new_photos/20261020/Y+0.05 T+0.01 (2).bmp   → Y 0.05, T 0.01 (괄호 안 번호는 같은 자세의 순번)\n"
              << "  - 파일 이름의 명령값(Y 는 mm, T 는 도, 부호를 꼭 붙임)을 정답으로 씁니다.\n"
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

fs::path new_result_folder() {
    const std::time_t now = std::time(nullptr);
    char stamp[32];
    std::strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%S", std::localtime(&now));
    return fs::path(kResultFolder) / stamp;
}

// [2/2] 무엇을 할지 보여 주고 시작해도 되는지 묻는다. measure_again: 사진을 다시 잴지 (전에 만든 표가 있으면 묻는다).
bool confirm(const fs::path& photo_folder, const fs::path& result_folder, bool& measure_again) {
    const std::vector<Session> sessions = list_sessions(photo_folder);
    std::cout << "\n[2/2] 확인\n  회차 " << sessions.size() << "개, 사진 " << total_photos(sessions) << "장 (" << photo_folder.u8string() << ")\n";
    for (const Session& session : sessions) std::cout << "    " << session.name << "  " << session.photos << "장\n";
    measure_again = true;
    if (fs::exists(photo_folder / kTableFolder / "photos.tsv")) {
        std::cout << "  전에 사진을 재서 만든 표가 있습니다 (" << (photo_folder / kTableFolder).u8string() << ").\n"
                  << "  사진을 더 넣거나 바꾸지 않았으면 다시 재지 않아도 됩니다.\n";
        measure_again = !ask_yes("  있는 표를 그대로 쓸까요? (Enter 또는 y = 그대로 씀, n = 지우고 다시 잼) > ");
    }
    const int seconds = int(kSecondsPerPhoto * total_photos(sessions));
    std::cout << "  할 일: " << (measure_again ? "사진 재기 (약 " + std::to_string(seconds / 60) + "분 " + std::to_string(seconds % 60) + "초) → " : "") << "계수 만들기 → 정확도 확인\n"
              << "  새 계수를 저장할 곳: " << fs::absolute(result_folder).u8string() << "\n"
              << "  지금 쓰는 계수(data/model)는 바꾸지 않습니다.\n";
    return ask_yes("  시작할까요? (Enter 또는 y = 시작, n = 그만) > ");
}

// 같은 폴더에 있는 다른 프로그램을 실행한다. 성공하면 true.
bool run_program(const fs::path& program_folder, const std::string& name, const std::vector<fs::path>& arguments) {
#ifdef _WIN32
    std::wstring command = L"\"\"" + (program_folder / (name + ".exe")).wstring() + L"\"";
    for (const fs::path& argument : arguments) command += L" \"" + argument.wstring() + L"\"";
    command += L"\"";   // cmd 는 따옴표로 시작하는 명령을 통째로 한 번 더 감싸야 한다
    return _wsystem(command.c_str()) == 0;
#else
    std::string command = "\"" + (program_folder / name).string() + "\"";
    for (const fs::path& argument : arguments) command += " \"" + argument.string() + "\"";
    return std::system(command.c_str()) == 0;
#endif
}

void print_finished(const fs::path& photo_folder, const fs::path& result_folder) {
    const std::string model = (result_folder / "model").generic_u8string();
    std::cout << "\n학습이 완료되었습니다.\n"
              << "  새 계수:        " << fs::absolute(result_folder / "model").u8string() << "\n"
              << "                  (coefficients/ 계수 16개, checks/ 점검식 11개, reference.json, center.bmp, fit_report.txt)\n"
              << "  정확도 보고서:  " << fs::absolute(result_folder / "accuracy" / "accuracy_report.txt").u8string() << "\n"
              << "  사진을 잰 표:   " << fs::absolute(photo_folder / kTableFolder).u8string() << "  (prepare_report.txt 에 뺀 사진과 이유)\n\n"
              << "  다음에 할 수 있는 일\n"
              << "  - 새 계수로 하루치 사진을 평가:  score_photos <사진 폴더> <출력 폴더> --data " << model << "\n"
              << "  - 새 계수를 쓰기로 했으면:  data/model 폴더를 다른 이름으로 보관한 뒤, " << model << " 안의 파일을 data/model 로 복사\n"
              << "  속도는 계수를 바꿔도 달라지지 않습니다 (재는 방법이 같아서). 속도만 재려면 measure_speed 를 씁니다.\n";
}

}   // namespace

int main(int, char** argv) {
    use_korean_console();
    const fs::path program_folder = fs::absolute(fs::path(argv[0])).parent_path();
    std::cout << "학습: 받은 사진으로 계수를 새로 만듭니다.\n";
    const fs::path photo_folder = ask_photo_folder();
    const fs::path tables = photo_folder / kTableFolder, result_folder = new_result_folder();
    bool measure_again = true;
    if (!confirm(photo_folder, result_folder, measure_again)) { std::cout << "그만두었습니다. 아무것도 바꾸지 않았습니다.\n"; return 0; }

    if (measure_again) {
        fs::remove_all(tables);   // 프로그램이 만든 표만 지운다 (사진은 그대로)
        std::cout << "\n[1/3] 사진 재기\n" << std::flush;
        if (!run_program(program_folder, "prepare_dataset", {photo_folder})) { std::cout << "사진 재기에서 멈췄습니다. 위 메시지를 확인하세요.\n"; return 1; }
    }
    std::cout << "\n[2/3] 계수 만들기\n" << std::flush;
    if (!run_program(program_folder, "fit_coefficients", {tables, result_folder / "model"})) { std::cout << "계수 만들기에서 멈췄습니다. 위 메시지를 확인하세요.\n"; return 1; }
    std::cout << "\n[3/3] 정확도 확인 (학습에 안 쓴 사진으로 시험)\n" << std::flush;
    if (!run_program(program_folder, "evaluate_accuracy", {tables, result_folder / "accuracy"})) { std::cout << "정확도 확인에서 멈췄습니다. 위 메시지를 확인하세요.\n"; return 1; }

    print_finished(photo_folder, result_folder);
    ask("Enter 를 누르면 끝납니다.");
    return 0;
}
