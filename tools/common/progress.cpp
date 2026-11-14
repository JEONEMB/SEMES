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

bool output_is_screen() {
#ifdef _WIN32
    return _isatty(_fileno(stdout)) != 0;
#else
    return isatty(fileno(stdout)) != 0;
#endif
}

std::string minutes_seconds(double seconds) {
    const int whole = int(seconds + 0.5);
    return std::to_string(whole / 60) + "분 " + std::to_string(whole % 60) + "초";
}

// 아무것도 쓰지 않는 출력 (QuietWarnings 가 쓴다)
class NullBuffer : public std::streambuf {
protected:
    int overflow(int c) override { return c == EOF ? 0 : c; }
};
NullBuffer null_buffer;

}   // namespace

ProgressBar::ProgressBar(const std::string& label, int total)
    : label_(label), total_(std::max(total, 1)), on_screen_(output_is_screen()), start_(std::chrono::steady_clock::now()), last_draw_(start_) {
    if (on_screen_) draw(false);
}

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

void ProgressBar::finish() {
    draw(true);
}

void ProgressBar::draw(bool done) {
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
    const int filled = kBarWidth * count_ / total_;
    std::string line = label_ + "  [" + std::string(filled, '#') + std::string(kBarWidth - filled, '-') + "]  " + std::to_string(count_) + "/" + std::to_string(total_) + "장  " +
                       std::to_string(100 * count_ / total_) + "%  ";
    if (done) line += "걸린 시간 " + minutes_seconds(elapsed);
    else if (count_ > 0) line += "남은 시간 약 " + minutes_seconds(elapsed / count_ * (total_ - count_));
    if (on_screen_) std::cout << '\r' << line << "      " << (done ? "\n" : "") << std::flush;   // 뒤 빈칸: 앞에 쓴 긴 글자를 지운다
    else std::cout << line << "\n" << std::flush;
}

QuietWarnings::QuietWarnings() : saved_(std::cerr.rdbuf(&null_buffer)) {}

QuietWarnings::~QuietWarnings() {
    std::cerr.rdbuf(saved_);
}

}   // namespace tools
