#include "console.hpp"
#include <cstdlib>
#include <iostream>
#ifdef _WIN32
#include <windows.h>
#endif

namespace tools {

namespace {

// 입력받은 글자를 UTF-8 로 바꾼다. Windows 콘솔은 한글을 UTF-8 이 아닌 문자 집합(보통 CP949)으로 넘겨준다.
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

std::string trimmed(const std::string& text) {
    const char* const ignored = " \t\r\n\"";
    const size_t first = text.find_first_not_of(ignored);
    if (first == std::string::npos) return "";
    return text.substr(first, text.find_last_not_of(ignored) - first + 1);
}

}   // namespace

void use_korean_console() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
}

std::string read_answer() {
    std::string line;
    if (!std::getline(std::cin, line)) {   // 입력이 끝남 (창이 닫혔거나 넘겨받은 입력이 다 떨어짐)
        std::cout << "\n입력이 끝나 그만둡니다.\n";
        std::exit(1);
    }
    return trimmed(to_utf8(line));
}

std::string ask(const std::string& question) {
    std::cout << question << std::flush;
    return read_answer();
}

bool ask_yes(const std::string& question) {
    const std::string answer = ask(question);
    return answer.empty() || answer[0] == 'y' || answer[0] == 'Y';
}

}   // namespace tools
