// 학습에 사용하지 않은 사진으로 정확도 평가
// 사용: evaluate_accuracy <tables 폴더> <출력 폴더>
// 시험: 5묶음 × 10회, 최종 검증, 회차 제외, 학습 범위 밖
// 정답: 파일명 명령값과 영점 보정 정답(labels.tsv)
// 회차 영점: 명령 (0, 0) 사진 우선. 없으면 근접 명령 사진 5장, 200회 추출
// 영점 사진: 채점 제외
#include "common/coefficient_fit.hpp"
#include "common/console.hpp"
#include "common/dataset.hpp"
#include "common/tables.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>

namespace fs = std::filesystem;
using namespace tools;

namespace {

const char* const kEvaluatedScrews = "1234";            // 평가하는 조합 (나사 4개 모두)
const double kInnerYmm = 1.0, kInnerTdeg = 0.5;        // 범위 밖 시험의 안쪽
const double kNearCenterYmm = 0.25, kNearCenterTdeg = 0.05;
const int kZeroPhotoCount = 5, kZeroDraws = 200;
const unsigned kZeroSeed = 20261114;

using Photos = std::vector<const DatasetPhoto*>;

// train: 시험에 쓸 이동량 계산식 생성
//   입력: photos (학습용 사진)
//   출력: 나사 4개 조합의 이동량 계산식
alignment::CoefficientSet train(const Photos& photos) {
    std::vector<Values> values;
    std::vector<double> y, t;
    for (const DatasetPhoto* p : photos) {
        values.push_back(p->values);
        y.push_back(p->label_y_mm);
        t.push_back(p->label_t_deg);
    }
    return fit_coefficient_set(kEvaluatedScrews, values, y, t);
}

// passes_learned_answer: 학습 정답 기준 채점
//   입력: set (이동량 계산식), photo (시험 사진 자료)
//   출력: 허용오차 이내이면 true
bool passes_learned_answer(const alignment::CoefficientSet& set, const DatasetPhoto& photo) {
    const alignment::YT pose = predict(set, photo.values);
    return passes(pose.y_mm - photo.label_y_mm, pose.t_deg - photo.label_t_deg);
}

struct Count {
    int passed = 0, total = 0;
    // add: 채점 결과 한 건 반영
    //   입력: ok (통과 여부)
    //   출력: 없음 (전체 수와 통과 수 갱신)
    void add(bool ok) { ++total; passed += ok; }
};

std::ostringstream report_text;
// print: 평가 결과 한 줄 기록
//   입력: line (결과 문자열)
//   출력: 없음 (화면과 보고서 출력)
void print(const std::string& line) {
    std::cout << line << "\n";
    report_text << line << "\n";
}

// percent: 백분율 표시
//   입력: value (백분율 값)
//   출력: 백분율 문자열
std::string percent(double value) {
    std::ostringstream text;
    text << std::fixed << std::setprecision(1) << value << "%";
    return text.str();
}

// test_mixed_folds: 5묶음을 번갈아 시험에 사용, 10회 반복
//   입력: development (개발용 사진)
//   출력: 없음 (검증 결과 출력)
void test_mixed_folds(const Photos& development) {
    std::vector<int> fails;
    Count total;
    std::map<std::string, Count> by_session;
    for (int r = 0; r < kRepeatCount; ++r) {
        int repeat_fails = 0;
        for (int f = 0; f < kFoldCount; ++f) {
            Photos train_set;
            for (const DatasetPhoto* p : development)
                if (p->fold[r] != f) train_set.push_back(p);
            const alignment::CoefficientSet set = train(train_set);
            for (const DatasetPhoto* p : development) {
                if (p->fold[r] != f) continue;
                const bool ok = passes_learned_answer(set, *p);
                repeat_fails += !ok;
                total.add(ok);
                by_session[p->session].add(ok);
            }
        }
        fails.push_back(repeat_fails);
    }
    double mean = 0;
    for (int f : fails) mean += f;
    mean /= fails.size();
    std::ostringstream line;
    line << "[1] 섞음 (" << development.size() << "장, 5묶음 × 10번, 학습 정답으로 채점)\n    10번 평균 탈락 " << std::fixed << std::setprecision(1) << mean
         << "장, 가장 많을 때 " << *std::max_element(fails.begin(), fails.end()) << "장, 통과율 " << format_rate(total.passed, total.total);
    print(line.str());
    for (const auto& s : by_session) print("      " + s.first + "  " + format_rate(s.second.passed, s.second.total));
}

// test_final_check: 분리한 사진으로 최종 검증
//   입력: development (학습용), final_check (검증용)
//   출력: 없음 (검증 결과 출력)
void test_final_check(const Photos& development, const Photos& final_check) {
    const alignment::CoefficientSet set = train(development);
    Count count;
    for (const DatasetPhoto* p : final_check) count.add(passes_learned_answer(set, *p));
    print("\n[2] 마지막 확인 (" + std::to_string(final_check.size()) + "장, 학습 정답으로 채점)\n    탈락 " + std::to_string(count.total - count.passed) + "장, 통과율 " +
          format_rate(count.passed, count.total));
}

// 3. 처음 보는 날

// 한 회차의 측정값 (사진 -> 계산식으로 구한 Y, T)
using Measured = std::map<const DatasetPhoto*, alignment::YT>;

// rate_with_zero: 회차 영점 보정 후 채점
//   입력: test (시험 사진), measured (측정 결과), used_for_zero (영점 추정용 사진)
//   출력: 영점 사진을 제외한 통과율(%)
double rate_with_zero(const Photos& test, const Measured& measured, const Photos& used_for_zero) {
    double zero_y = 0, zero_t = 0;
    for (const DatasetPhoto* p : used_for_zero) {
        zero_y += measured.at(p).y_mm - p->command_y_mm;
        zero_t += measured.at(p).t_deg - p->command_t_deg;
    }
    zero_y /= used_for_zero.size();
    zero_t /= used_for_zero.size();
    Count count;
    for (const DatasetPhoto* p : test) {
        if (std::find(used_for_zero.begin(), used_for_zero.end(), p) != used_for_zero.end()) continue;
        count.add(passes(measured.at(p).y_mm - zero_y - p->command_y_mm, measured.at(p).t_deg - zero_t - p->command_t_deg));
    }
    return 100.0 * count.passed / std::max(count.total, 1);
}

struct SessionResult {
    Count learned_answer, received_no_zero;
    double received_with_zero = -1;   // 통과율 (%). 영점을 못 잡았으면 -1
    std::string zero_note;
};

// score_session: 한 촬영 회차의 정확도 평가
//   입력: set (계산식), test (시험 사진)
//   출력: 영점 적용 전후의 통과율
SessionResult score_session(const alignment::CoefficientSet& set, const Photos& test) {
    SessionResult result;
    Measured measured;
    Photos zero_photos, near_center;
    for (const DatasetPhoto* p : test) {
        const alignment::YT pose = predict(set, p->values);
        measured[p] = pose;
        result.learned_answer.add(passes(pose.y_mm - p->label_y_mm, pose.t_deg - p->label_t_deg));
        result.received_no_zero.add(passes(pose.y_mm - p->command_y_mm, pose.t_deg - p->command_t_deg));
        if (p->command_y_mm == 0 && p->command_t_deg == 0) zero_photos.push_back(p);
        else if (std::abs(p->command_y_mm) <= kNearCenterYmm && std::abs(p->command_t_deg) <= kNearCenterTdeg) near_center.push_back(p);
    }
    if (!zero_photos.empty()) {
        result.received_with_zero = rate_with_zero(test, measured, zero_photos);
        result.zero_note = "명령 (0, 0) 사진 " + std::to_string(zero_photos.size()) + "장으로 영점";
    } else if (int(near_center.size()) > kZeroPhotoCount) {
        std::mt19937 random(kZeroSeed);
        double sum = 0;
        for (int d = 0; d < kZeroDraws; ++d) {
            Photos chosen = near_center;
            std::shuffle(chosen.begin(), chosen.end(), random);
            chosen.resize(kZeroPhotoCount);
            sum += rate_with_zero(test, measured, chosen);
        }
        result.received_with_zero = sum / kZeroDraws;
        result.zero_note = "명령 0 사진이 없어 센터 근처 사진 5장으로 영점 (200번 평균)";
    } else {
        result.zero_note = "영점을 잡을 사진이 부족함";
    }
    return result;
}

// test_unseen_sessions: 학습에 빠진 촬영 회차로 시험
//   입력: usable (사용 가능 사진), sessions (회차 목록)
//   출력: 없음 (회차별 시험 결과 출력)
void test_unseen_sessions(const Photos& usable, const std::vector<SessionZero>& sessions) {
    print("\n[3] 처음 보는 날 (회차 하나를 빼고 학습)");
    print("    회차        사진  받은 정답, 영점 없음   받은 정답, 그날 영점   학습 정답");
    for (const SessionZero& session : sessions) {
        Photos train_set, test;
        for (const DatasetPhoto* p : usable) (p->session == session.session ? test : train_set).push_back(p);
        if (test.empty() || train_set.empty()) continue;
        const SessionResult r = score_session(train(train_set), test);
        std::ostringstream line;
        line << "    " << std::left << std::setw(10) << session.session << std::right << std::setw(6) << test.size() << "  " << std::left << std::setw(22)
             << format_rate(r.received_no_zero.passed, r.received_no_zero.total) << std::setw(22) << (r.received_with_zero < 0 ? "-" : percent(r.received_with_zero))
             << format_rate(r.learned_answer.passed, r.learned_answer.total) << "   (" << r.zero_note << ")";
        print(line.str());
    }
}

// test_outside_range: 학습 범위 밖 평가
//   입력: usable (사용 가능한 사진)
//   출력: 없음 (범위별 오차와 통과율 출력)
void test_outside_range(const Photos& usable) {
    Photos inner, outer;
    for (const DatasetPhoto* p : usable) (std::abs(p->label_y_mm) <= kInnerYmm && std::abs(p->label_t_deg) <= kInnerTdeg ? inner : outer).push_back(p);
    const alignment::CoefficientSet set = train(inner);
    Count count;
    double yy = 0, ye = 0, tt = 0, te = 0;   // 명령값이 커질수록 오차가 커지는 비율
    for (const DatasetPhoto* p : outer) {
        const alignment::YT pose = predict(set, p->values);
        count.add(passes(pose.y_mm - p->label_y_mm, pose.t_deg - p->label_t_deg));
        yy += p->label_y_mm * p->label_y_mm;
        ye += p->label_y_mm * (pose.y_mm - p->label_y_mm);
        tt += p->label_t_deg * p->label_t_deg;
        te += p->label_t_deg * (pose.t_deg - p->label_t_deg);
    }
    std::ostringstream line;
    line << "\n[4] 범위 밖 (안쪽 " << inner.size() << "장으로 학습, 바깥 " << outer.size() << "장 시험, 학습 정답으로 채점)\n    통과율 " << format_rate(count.passed, count.total)
         << ", 비례 오차 Y " << std::fixed << std::setprecision(2) << 100.0 * ye / std::max(yy, 1e-12) << "%, T " << 100.0 * te / std::max(tt, 1e-12) << "%";
    print(line.str());
}

}   // 파일 내부

// main: 학습에 사용하지 않은 사진으로 정확도 평가
//   입력: 실행할 때 지정한 경로와 옵션(파일 상단 사용법)
//   출력: 실행 결과와 성공 여부를 나타내는 종료 코드
int main(int argc, char** argv) {
    use_korean_console();
    if (argc < 3) { std::cerr << "사용: evaluate_accuracy <tables 폴더> <출력 폴더>\n"; return 2; }
    const fs::path tables = fs::u8path(argv[1]), output = fs::u8path(argv[2]);
    std::vector<DatasetPhoto> photos;
    std::vector<SessionZero> sessions;
    std::string error;
    if (!load_tables(tables, photos, sessions, error)) { std::cerr << error << "\n"; return 1; }
    if (!make_empty_folder(output, error)) { std::cerr << error << "\n"; return 1; }

    Photos usable, development, final_check;
    for (const DatasetPhoto& p : photos) {
        if (!p.usable) continue;
        usable.push_back(&p);
        (p.final_check ? final_check : development).push_back(&p);
    }
    print("통과 기준: |Y 오차| <= " + format_number(kToleranceYmm) + " mm 그리고 |T 오차| <= " + format_number(kToleranceTdeg) + " 도");
    print("쓴 사진 " + std::to_string(usable.size()) + "장 (마지막 확인용 " + std::to_string(final_check.size()) + "장)\n");

    test_mixed_folds(development);
    test_final_check(development, final_check);
    test_unseen_sessions(usable, sessions);
    test_outside_range(usable);

    std::ofstream(output / "accuracy_report.txt", std::ios::binary) << report_text.str();
    return 0;
}
