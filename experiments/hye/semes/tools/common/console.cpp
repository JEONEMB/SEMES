// Windows 콘솔 인코딩과 사용자 입력 처리
#include "console.hpp"
#include <cstdlib>
#include <iostream>
#ifdef _WIN32
#include <windows.h>
#endif

namespace tools {

namespace {

// to_utf8: 콘솔 입력 인코딩 변환
//   입력: typed (현재 문자 저장 방식의 문자열)
//   출력: 한글을 지원하는 UTF-8 문자열
std::string to_utf8(const std::string& typed) {
#ifdef _WIN32
    UINT code_page = GetConsoleCP();
    if (code_page == 0) code_page = GetACP();   // 콘솔이 아니면(파일, 다른 프로그램에서 넘겨받음) 시스템 기본 문자 집합
    if (code_page == CP_UTF8 || typed.empty()) return typed;
    const int wide_length = MultiByteToWideChar(code_page, 0, typed.data(), int(typed.size()), nullptr, 0);
    std::wstring wide(wide_length, L'\0');
    MultiByteToWideChar(code_page, 0, typed.data(), int(typed.size()), &wide[0], wide_length);
    const int length = WideCharToMultiByte(CP_UTF8, 0, wide.data(), wide_length, nullptr, 0, nullptr, nullptr);
    std::string utf8(length, '\0');
    WideCharToMultiByte(CP_UTF8, 0, wide.data(), wide_length, &utf8[0], length, nullptr, nullptr);
    return utf8;
#else
    return typed;
#endif
}

// trimmed: 입력 경로의 불필요한 문자 제거
//   입력: text (입력 문자열)
//   출력: 앞뒤 공백과 바깥 따옴표를 제거한 문자열
std::string trimmed(const std::string& text) {
    const char* const ignored = " \t\r\n\"";
    const size_t first = text.find_first_not_of(ignored);
    if (first == std::string::npos) return "";
    return text.substr(first, text.find_last_not_of(ignored) - first + 1);
}

}   // 파일 내부

// use_korean_console: Windows 명령 창의 한글 표시 설정
//   입력: 없음
//   출력: 없음 (Windows 명령 창의 한글 출력 방식 변경)
void use_korean_console() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
}

// read_answer: 사용자 입력 한 줄 읽기
//   입력: 키보드 입력
//   출력: 한글을 지원하는 UTF-8 문자열. 앞뒤 공백과 경로 따옴표 제거
std::string read_answer() {
    std::string line;
    if (!std::getline(std::cin, line)) {   // 입력 종료
        std::cout << "\n입력이 끝나 그만둡니다.\n";
        std::exit(1);
    }
    return trimmed(to_utf8(line));
}

// ask: 질문과 답변 입력
//   입력: question (질문)
//   출력: 입력 문자열
std::string ask(const std::string& question) {
    std::cout << question << std::flush;
    return read_answer();
}

// ask_yes: 실행 여부 확인
//   입력: question (질문)
//   출력: y 또는 Enter이면 true
bool ask_yes(const std::string& question) {
    const std::string answer = ask(question);
    return answer.empty() || answer[0] == 'y' || answer[0] == 'Y';
}

}   // tools
