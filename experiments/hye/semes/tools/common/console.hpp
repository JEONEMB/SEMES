#pragma once
// 대화형 도구의 한글 명령 창 입력과 출력
#include <string>

namespace tools {

// use_korean_console: Windows 명령 창의 한글 표시 설정
//   입력: 없음
//   출력: 없음 (Windows 명령 창의 한글 출력 방식 변경)
void use_korean_console();

// read_answer: 사용자 입력 한 줄 읽기
//   입력: 키보드 입력
//   출력: 한글을 지원하는 UTF-8 문자열. 앞뒤 공백과 경로 따옴표 제거
std::string read_answer();

// ask: 질문과 답변 입력
//   입력: question (질문)
//   출력: 입력 문자열
std::string ask(const std::string& question);

// ask_yes: 실행 여부 확인
//   입력: question (질문)
//   출력: y 또는 Enter이면 true
bool ask_yes(const std::string& question);

}   // tools
