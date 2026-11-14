#pragma once
// 오래 걸리는 일의 진행 상황을 한 줄로 보여 준다.
//
//   사진 재는 중  [############------------]  352/715장  49%  남은 시간 약 0분 40초
//
// 쓰는 법
//   ProgressBar bar("사진 재는 중", 사진 수);
//   for (사진마다) { …; bar.step(); }
//   bar.finish();
#include <chrono>
#include <iostream>
#include <string>

namespace tools {

class ProgressBar {
public:
    ProgressBar(const std::string& label, int total);
    void step();     // 하나 끝날 때마다 부른다
    void finish();   // 다 끝나면 부른다 (걸린 시간을 적고 줄을 바꾼다)

private:
    void draw(bool done);

    std::string label_;
    int total_ = 0, count_ = 0;
    int last_tenth_ = 0;                                    // 화면이 아닐 때(파일로 저장) 10% 마다 한 줄씩 쓰려고
    bool on_screen_ = false;                                // 출력이 화면인지 (화면이면 같은 줄을 고쳐 쓴다)
    std::chrono::steady_clock::time_point start_, last_draw_;
};

// 살아 있는 동안 측정 함수의 알림(경고, 실패 한 줄씩)이 화면에 나오지 않게 한다. 진행 줄이 깨지지 않게 하려고 쓴다.
// 알림은 결과(Result 의 code, message)에는 그대로 들어 있다.
class QuietWarnings {
public:
    QuietWarnings();
    ~QuietWarnings();

private:
    std::streambuf* saved_ = nullptr;
};

}   // namespace tools
