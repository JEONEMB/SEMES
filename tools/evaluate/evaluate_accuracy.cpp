// 평가: 정확도 (학습에 안 쓴 사진으로만 시험한다)
//
// 사용: evaluate_accuracy <tables 폴더> <출력 폴더>
//
// 시험 (나사 4개 조합의 계수, 시험마다 새로 학습)
//   1. 섞음:        마지막 확인 사진을 뺀 나머지를 5묶음으로 나눠 돌아가며 시험, 나누기를 바꿔 10번
//   2. 마지막 확인: 나머지 전부로 학습, 떼어 둔 마지막 확인 사진으로 한 번
//   3. 처음 보는 날: 회차 하나를 통째로 빼고 학습, 그 회차로 시험 (회차마다)
//   4. 범위 밖:     안쪽(|Y| <= 1, |T| <= 0.5)으로만 학습, 바깥을 시험
//
// 채점 정답 두 가지
//   - 받은 정답 (파일 이름의 명령값 그대로). 영점 없이 / 그 회차에서 영점을 잡고
//       영점: 그 회차의 명령 (0, 0) 사진이 있으면 그 사진들, 없으면 명령값을 아는 센터 근처(|Y| <= 0.25, |T| <= 0.05) 사진 5장
//             으로 평균(측정 − 명령)을 구한다. 5장은 200번 뽑아 평균. 영점에 쓴 사진은 채점에서 뺀다
//   - 학습 정답 (명령 − 그 회차의 영점 어긋남, tables/labels.tsv)
#include "common/coefficient_fit.hpp"
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

// 학습 정답으로 채점해서 통과인지
bool passes_learned_answer(const alignment::CoefficientSet& set, const DatasetPhoto& photo) {
    const alignment::YT pose = predict(set, photo.values);
    return passes(pose.y_mm - photo.label_y_mm, pose.t_deg - photo.label_t_deg);
}

struct Count {
    int passed = 0, total = 0;
    void add(bool ok) { ++total; passed += ok; }
};

std::ostringstream report_text;
void print(const std::string& line) {
    std::cout << line << "\n";
    report_text << line << "\n";
}

std::string percent(double value) {
    std::ostringstream text;
    text << std::fixed << std::setprecision(1) << value << "%";
    return text.str();
}

// ---- 1. 섞음 ----
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

// ---- 2. 마지막 확인 ----
void test_final_check(const Photos& development, const Photos& final_check) {
    const alignment::CoefficientSet set = train(development);
    Count count;
    for (const DatasetPhoto* p : final_check) count.add(passes_learned_answer(set, *p));
    print("\n[2] 마지막 확인 (" + std::to_string(final_check.size()) + "장, 학습 정답으로 채점)\n    탈락 " + std::to_string(count.total - count.passed) + "장, 통과율 " +
          format_rate(count.passed, count.total));
}

// ---- 3. 처음 보는 날 ----

// 한 회차의 측정값 (사진 → 계수로 낸 Y, T)
using Measured = std::map<const DatasetPhoto*, alignment::YT>;

// used_for_zero 사진들로 영점(평균(측정 − 명령))을 잡고, 나머지를 받은 정답으로 채점한 통과율 (%)
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
    double received_with_zero = -1;   // 통과율 (%). 영점을 못 잡았으면 −1
    std::string zero_note;
};

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

// ---- 4. 범위 밖 ----
void test_outside_range(const Photos& usable) {
    Photos inner, outer;
    for (const DatasetPhoto* p : usable) (std::abs(p->label_y_mm) <= kInnerYmm && std::abs(p->label_t_deg) <= kInnerTdeg ? inner : outer).push_back(p);
    const alignment::CoefficientSet set = train(inner);
    Count count;
    double yy = 0, ye = 0, tt = 0, te = 0;   // 오차 = k × 정답 의 기울기 k (비례 오차)
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

}   // namespace

int main(int argc, char** argv) {
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
