// 진행률과 남은 시간 표시. 화면과 파일 출력 구분
#include "progress.hpp"
#include <algorithm>
#include <cstdio>
#include <streambuf>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

namespace tools {

namespace {

const int kBarWidth = 24;                 // 막대 칸 수
const double kRedrawSeconds = 0.2;        // 화면을 고쳐 쓰는 간격

// output_is_screen: 출력 대상 확인
//   입력: 없음
//   출력: 명령 창 화면이면 true, 파일 등이면 false
bool output_is_screen() {
#ifdef _WIN32
    return _isatty(_fileno(stdout)) != 0;
#else
    return isatty(fileno(stdout)) != 0;
#endif
}

// minutes_seconds: 초를 분과 초로 표시
//   입력: seconds (경과 시간)
//   출력: 시간 문자열. 예: 1분 20초
std::string minutes_seconds(double seconds) {
    const int whole = int(seconds + 0.5);
    return std::to_string(whole / 60) + "분 " + std::to_string(whole % 60) + "초";
}

// QuietWarnings용 빈 출력
class NullBuffer : public std::streambuf {
protected:
    // overflow: 숨긴 알림 문자 처리
    //   입력: c (문자 또는 입력 끝 표시)
    //   출력: 정상 처리 표시. 화면 출력 없음
    int overflow(int c) override { return c == EOF ? 0 : c; }
};
NullBuffer null_buffer;

}   // 파일 내부

// ProgressBar: 진행 표시 시작
//   입력: label (작업명), total (전체 건수)
//   출력: 시작 시각과 출력 대상 저장
ProgressBar::ProgressBar(const std::string& label, int total)
    : label_(label), total_(std::max(total, 1)), on_screen_(output_is_screen()), start_(std::chrono::steady_clock::now()), last_draw_(start_) {
    if (on_screen_) draw(false);
}

// step: 처리 건수 증가
//   입력: 없음
//   출력: 없음 (진행 표시 갱신)
void ProgressBar::step() {
    ++count_;
    const auto now = std::chrono::steady_clock::now();
    if (on_screen_) {
        if (std::chrono::duration<double>(now - last_draw_).count() < kRedrawSeconds && count_ < total_) return;
        last_draw_ = now;
        draw(false);
    } else {
        const int tenth = 10 * count_ / total_;
        if (tenth == last_tenth_) return;
        last_tenth_ = tenth;
        draw(false);
    }
}

// finish: 진행 표시 완료
//   입력: 없음
//   출력: 없음 (경과 시간 표시와 줄바꿈)
void ProgressBar::finish() {
    draw(true);
}

// draw: 진행 표시 출력
//   입력: done (완료 여부)
//   출력: 없음 (화면 또는 파일 출력)
void ProgressBar::draw(bool done) {
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
    const int filled = kBarWidth * count_ / total_;
    std::string line = label_ + "  [" + std::string(filled, '#') + std::string(kBarWidth - filled, '-') + "]  " + std::to_string(count_) + "/" + std::to_string(total_) + "장  " +
                       std::to_string(100 * count_ / total_) + "%  ";
    if (done) line += "걸린 시간 " + minutes_seconds(elapsed);
    else if (count_ > 0) line += "남은 시간 약 " + minutes_seconds(elapsed / count_ * (total_ - count_));
    if (on_screen_) std::cout << '\r' << line << "      " << (done ? "\n" : "") << std::flush;   // 이전 출력의 남은 글자 제거용 공백
    else std::cout << line << "\n" << std::flush;
}

// QuietWarnings: 화면 오류 알림 숨기기
//   입력: 없음
//   출력: 기존 출력 위치 저장 후 임시 출력으로 변경
QuietWarnings::QuietWarnings() : saved_(std::cerr.rdbuf(&null_buffer)) {}

// ~QuietWarnings: 화면 오류 알림 복원
//   입력: 없음
//   출력: 오류 출력을 원래 위치로 복원
QuietWarnings::~QuietWarnings() {
    std::cerr.rdbuf(saved_);
}

}   // tools
