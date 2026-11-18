#include "photo_picker.hpp"
#include <stdexcept>
#ifdef _WIN32
#include <windows.h>
#include <shobjidl.h>
#include <commctrl.h>
#include <wrl/client.h>
#pragma comment(linker, "\"/manifestdependency:type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'\"")

namespace {
std::filesystem::path next_photo_folder;

// example_folder: 파일 선택 창의 시작 위치 결정
//   입력: 없음
//   출력: 동봉 예시 폴더. 없으면 현재 작업 폴더
std::filesystem::path example_folder() {
    const auto examples = std::filesystem::absolute("data/examples");
    return std::filesystem::is_directory(examples) ? examples : std::filesystem::current_path();
}

// matching_photos: 기준 사진과 같은 세트의 평가 폴더 찾기
//   입력: selected (기준 사진 또는 기준 폴더)
//   출력: 인접 photos 폴더. 없으면 빈 경로
std::filesystem::path matching_photos(const std::filesystem::path& selected) {
    auto folder = std::filesystem::is_directory(selected) ? selected : selected.parent_path();
    if (folder.filename() == "center") {
        const auto photos = folder.parent_path() / "photos";
        if (std::filesystem::is_directory(photos)) return photos;
    }
    return {};
}
struct ComSession {
    HRESULT result = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    ComSession() { if (FAILED(result)) throw std::runtime_error("파일 선택 창 초기화 실패"); }
    ~ComSession() { CoUninitialize(); }
};

// choose: 마우스로 선택 방법 지정
//   입력: title, first, second (창과 버튼 문구)
//   출력: 선택 번호. 취소는 0
int choose(const wchar_t* title, const wchar_t* first, const wchar_t* second) {
    TASKDIALOG_BUTTON buttons[] = {{100, first}, {101, second}};
    TASKDIALOGCONFIG config{};
    config.cbSize = sizeof(config);
    config.pszWindowTitle = L"정렬 측정 평가";
    config.pszMainInstruction = title;
    config.dwFlags = TDF_USE_COMMAND_LINKS | TDF_ALLOW_DIALOG_CANCELLATION;
    config.dwCommonButtons = TDCBF_CANCEL_BUTTON;
    config.cButtons = 2;
    config.pButtons = buttons;
    config.nDefaultButton = 100;
    int selected = 0;
    if (FAILED(TaskDialogIndirect(&config, &selected, nullptr, nullptr)))
        throw std::runtime_error("선택 창을 열지 못했습니다.");
    return selected == 100 ? 1 : selected == 101 ? 2 : 0;
}

// open_paths: Windows 파일 선택 창 실행
//   입력: title, folder (폴더 선택), answers (정답표 선택), initial (시작 폴더)
//   출력: 선택한 경로들. 취소하면 빈 목록
std::vector<std::filesystem::path> open_paths(const wchar_t* title, bool folder, bool answers,
                                             const std::filesystem::path& initial) {
    ComSession session;
    Microsoft::WRL::ComPtr<IFileOpenDialog> dialog;
    if (FAILED(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&dialog))))
        throw std::runtime_error("파일 선택 창을 만들지 못했습니다.");
    DWORD flags = 0;
    dialog->GetOptions(&flags);
    flags |= FOS_FORCEFILESYSTEM | FOS_PATHMUSTEXIST | FOS_FILEMUSTEXIST | FOS_NOCHANGEDIR;
    if (!answers) flags |= FOS_ALLOWMULTISELECT;
    if (folder) flags |= FOS_PICKFOLDERS;
    dialog->SetOptions(flags);
    dialog->SetTitle(title);
    dialog->SetOkButtonLabel(L"선택 완료");
    Microsoft::WRL::ComPtr<IShellItem> initial_item;
    if (std::filesystem::is_directory(initial)
        && SUCCEEDED(SHCreateItemFromParsingName(initial.c_str(), nullptr, IID_PPV_ARGS(&initial_item))))
        dialog->SetFolder(initial_item.Get());
    const COMDLG_FILTERSPEC filter = answers
        ? COMDLG_FILTERSPEC{L"정답표 (TSV)", L"*.tsv"}
        : COMDLG_FILTERSPEC{L"사진 (BMP, PNG)", L"*.bmp;*.png"};
    if (!folder) dialog->SetFileTypes(1, &filter);
    const HRESULT shown = dialog->Show(nullptr);
    if (shown == HRESULT_FROM_WIN32(ERROR_CANCELLED)) return {};
    if (FAILED(shown)) throw std::runtime_error("파일 선택 실패");
    Microsoft::WRL::ComPtr<IShellItemArray> items;
    if (FAILED(dialog->GetResults(&items))) throw std::runtime_error("선택 목록 읽기 실패");
    DWORD count = 0;
    items->GetCount(&count);
    std::vector<std::filesystem::path> paths;
    for (DWORD i = 0; i < count; ++i) {
        Microsoft::WRL::ComPtr<IShellItem> item;
        PWSTR name = nullptr;
        if (FAILED(items->GetItemAt(i, &item)) || FAILED(item->GetDisplayName(SIGDN_FILESYSPATH, &name)))
            throw std::runtime_error("선택 경로 읽기 실패");
        std::filesystem::path path(name);
        CoTaskMemFree(name);
        paths.push_back(path);
    }
    return paths;
}
}
#endif

namespace tools {
std::vector<std::filesystem::path> pick_photos(bool zero) {
#ifdef _WIN32
    const wchar_t* title = zero ? L"[1] 기준 사진(center)을 선택하세요" : L"[2] 평가 사진을 선택하세요";
    const int choice = choose(title, L"사진 선택\nCtrl 또는 Shift로 여러 사진을 선택할 수 있습니다.", L"폴더 선택\n선택한 폴더의 하위 폴더까지 사진을 모읍니다.");
    if (!choice) return {};
    const auto initial = zero || next_photo_folder.empty() ? example_folder() : next_photo_folder;
    const auto selected = open_paths(title, choice == 2, false, initial);
    if (zero) {
        next_photo_folder.clear();
        for (const auto& path : selected) {
            const auto candidate = matching_photos(path);
            if (candidate.empty() || (!next_photo_folder.empty() && candidate != next_photo_folder)) {
                next_photo_folder.clear();
                break;
            }
            next_photo_folder = candidate;
        }
    }
    return selected;
#else
    throw std::runtime_error("파일 선택 창은 Windows에서 지원합니다. 명령 인자로 경로를 지정하세요.");
#endif
}

std::filesystem::path pick_answers() {
#ifdef _WIN32
    const int choice = choose(L"[3] 정답을 어떻게 읽을까요?", L"사진 파일명 사용\n예: Y+0.05 T+0.01.bmp", L"정답표 선택\nfile, y_mm, t_deg 열이 있는 TSV 파일");
    if (choice == 1) return {};
    if (choice == 2) {
        const auto paths = open_paths(L"정답표를 선택하세요", false, true,
            next_photo_folder.empty() ? example_folder() : next_photo_folder.parent_path());
        if (!paths.empty()) return paths.front();
    }
    throw std::runtime_error("선택을 취소했습니다. 평가를 시작하지 않았습니다.");
#else
    return {};
#endif
}
}
