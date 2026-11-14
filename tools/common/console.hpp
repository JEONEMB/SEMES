#pragma once
// 화면에 묻고 답을 받는 도구 (물어보며 진행하는 프로그램에서 쓴다).
#include <string>

namespace tools {

// 한글이 깨지지 않게 화면 출력을 UTF-8 로 맞춘다 (Windows 콘솔). 프로그램을 시작할 때 한 번 부른다.
void use_korean_console();

// 한 줄을 입력받는다 (UTF-8 로 돌려준다). 앞뒤 빈칸과, 경로를 끌어다 놓을 때 붙는 따옴표는 뗀다.
std::string read_answer();

// question 을 보여 주고 한 줄을 입력받는다.
std::string ask(const std::string& question);

// question 을 보여 주고 y 또는 n 을 받는다 (그냥 Enter 는 y).
bool ask_yes(const std::string& question);

}   // namespace tools
