// =====================================================================================
// sim_runner.hpp — 영점 맞춤 시뮬레이터 : 로봇 + 카메라(합성 사진) + 모델(alignment) + 제어 반복
//
//  한 번의 시험 (trial)
//    0. 시작 자세(참값)로 로봇을 놓음
//    1. 현재 자세의 사진을 합성 (촬영 잡음 포함)
//    2. 추정기(사진) → 추정 어긋남 (y, t)   — 내장 ksh 또는 플러그인 DLL (sim_estimator.hpp)
//    3. 추정값이 '멈춤 범위'(허용 범위 × stop_ratio) 안이면 멈춤
//    4. 아니면 보정 이동 (−y, −t) 명령 → 한 번 최대 Y 2 mm / Θ 1° (장비 제원) 로 자름 → 로봇 오차(배율·반복 정밀도 실측)를 더해 실제 이동
//    5. 1 로 돌아감. 이동 20회를 넘기면 실패
//    판정 : 멈춘 뒤 '참값' 이 허용 범위(Y ±90 µm, Θ ±0.01°) 안이면 성공
//  기준(티칭) 사진 : 실제 center 사진 또는 자세 (0, 0) 합성 사진 (시험마다 새 잡음, 설정 render.teach_center_photo)
// =====================================================================================
#ifndef SIM_RUNNER_HPP
#define SIM_RUNNER_HPP

#include "sim_estimator.hpp"
#include "sim_world.hpp"

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace sim {

struct StepRecord {
    int k = 0;                 // 0 = 시작 측정, 1.. = 이동 뒤 측정
    Pose truth;                // 참 자세
    Pose est;                  // 모델 추정 (영점 맞춤 뒤, NaN = 실패) — 제어·멈춤 판단에 쓰는 값
    Pose est_raw;              // 추정기가 낸 값 그대로 (영점 맞춤 전)
    bool est_ok = false;
    Pose cmd;                  // 보정 명령 (자르기 뒤)
    Pose applied;              // 실제 이동 (로봇 오차 포함)
    bool stop = false;         // 이 측정에서 멈춤 판단
    bool truth_in_tol = false; // 참값이 허용 범위 안
    double ms_render = 0, ms_align = 0;
};

struct TrialResult {
    Pose start;
    std::vector<StepRecord> steps;
    int moves = 0;             // 실제 이동 횟수
    bool stopped = false;      // 멈춤 조건으로 끝남 (아니면 이동 한도 초과 또는 측정 실패)
    bool success = false;      // 멈춘 뒤 참값이 허용 범위 안
    std::string reason;
    Pose final_truth;
    Pose zero;                 // 이 시험의 영점 차이 (영점 맞춤을 안 하면 0)
    bool zero_applied = false;
};

struct MonteSummary {
    int n = 0, success = 0;
    double moves_mean = 0;
    int moves_max = 0;
    double final_y_rms_um = 0, final_y_max_um = 0, final_t_rms_deg = 0, final_t_max_deg = 0;
    std::vector<int> moves_hist;   // 이동 횟수별 시험 수 (0 ~ max_moves + 1)
    double ms_mean_per_step = 0;
    std::vector<TrialResult> trials;
};

struct ValidationRow {
    std::string stem, group;
    Pose label;
    Pose est_real, est_sim;
    bool ok_real = false, ok_sim = false;
};
struct ValidationSummary {
    std::vector<ValidationRow> rows;
    std::string table;   // 마크다운 표 (콘솔·보고서용)
};

class Simulator {
public:
    // 설정 파일을 읽음 (없으면 기본값으로 만들어 저장). 추정기·가상 장면을 불러옴
    //  작업 폴더 = 시뮬레이터 루트 상태에서 부를 것 (enterRoot)
    bool init(const std::string& configPath, std::string& message);
    bool worldReady() const { return world_.ready(); }
    bool estimatorReady() const { return est_ != nullptr; }
    // 추정기 바꾸기 : type = "ksh" 또는 "plugin" (plugin 은 DLL 경로). 실패하면 이전 추정기 유지
    bool setEstimator(const std::string& type, const std::string& plugin, std::string& message);
    std::string estimatorName() const { return est_ ? est_->name() : std::string("(없음)"); }
    Estimator* estimator() { return est_.get(); }                // 직접 시험할 때 (없으면 nullptr)
    // 마지막 추정의 진단 (ksh 만, 없으면 nullptr)
    const AlignmentInfo* lastInfo() const { return est_ ? est_->info() : nullptr; }
    // 마지막 teach 결과 설명 (플러그인이 기준 사진 교체를 지원하지 않을 때 등)
    const std::string& referenceNote() const { return referenceNote_; }
    // 현재 영점 차이 (teach 때 정함). 영점 맞춤을 안 했으면 applied = false
    Pose zeroOffset() const { return zero_; }
    bool zeroApplied() const { return zeroApplied_; }
    // 실제 사진으로 가상 장면을 새로 만들고 world 폴더에 저장 (수 분). 이전 장면은 world/_backup/날짜/ 로 복사해 둠
    bool buildWorld(const Progress& progress, std::string& message);
    // 영점 사진 바꾸기 : 설정과 설정 파일(paths.reference_photo)만 바꿈 → 이어서 buildWorld 로 장면을 다시 만들어야 적용됨
    bool setReferencePhoto(const std::string& stem, std::string& message);
    // 영점 사진 후보 : 데이터 폴더의 기준 사진 (center, center1, …). 데이터가 없으면 지금 장면의 영점 사진만
    std::vector<std::string> referenceCandidates() const;
    // 설정의 영점 사진과 지금 장면의 영점 사진이 같은가 (다르면 장면을 다시 만들어야 함)
    bool referenceMatchesWorld() const { return !world_.ready() || world_.referencePhoto() == cfg_.reference_photo; }

    const SimConfig& config() const { return cfg_; }
    const World& world() const { return world_; }

    // 기준(티칭) 사진 등록 + (설정에 따라) 영점 맞춤. 추정기가 기준 사진 교체를 지원하지 않으면 false (자체 기준 사진 사용)
    bool teach(cv::RNG& rng);
    // 한 시험. onStep(기록, 합성 사진) 이 false 를 돌려주거나 cancel 이 true 가 되면 중단
    using StepCallback = std::function<bool(const TrialResult&, const cv::Mat&)>;
    // 한 단계씩 진행 (화면에서 [1 단계]) : tr.start 에서 시작. 단계를 하나 더하고, 끝났으면 false (tr 판정이 채워짐)
    bool stepTrial(TrialResult& tr, cv::RNG& rng, cv::Mat* image);
    static bool finished(const TrialResult& tr);
    TrialResult runTrial(const Pose& start, cv::RNG& rng, const StepCallback& onStep, const std::atomic<bool>* cancel = nullptr);
    // 여러 시험 (시작 자세 : 균일 분포). onTrial(번호, 결과) 는 시험이 끝날 때마다
    MonteSummary monteCarlo(int n, unsigned int seed, const std::function<void(int, const TrialResult&)>& onTrial,
                            const std::atomic<bool>* cancel = nullptr);
    // 합성 사진이 실제 사진을 대신할 수 있는지 : 같은 라벨에서 모델 추정값 비교
    ValidationSummary validate(const Progress& progress, const std::atomic<bool>* cancel = nullptr);

    Pose randomStart(cv::RNG& rng) const;
    bool inTol(double ey_mm, double et_deg, double ratio = 1.0) const;

    // 결과 파일 (result_dir 아래)
    bool writeTrialCsv(const TrialResult& t, const std::string& path) const;
    bool writeMonteCsv(const MonteSummary& m, const std::string& path) const;
    std::string monteText(const MonteSummary& m) const;   // 요약 문단

private:
    SimConfig cfg_;
    World world_;
    std::unique_ptr<Estimator> est_;
    std::string configPath_, referenceNote_;
    Pose zero_;                // 영점 차이 (추정기가 참 자세 (0, 0) 에서 낸 값)
    bool zeroApplied_ = false;
    cv::Point coarseCenter_{1901, 1068};
    cv::Size coarseWin_{900, 400};
};

std::string poseText(const Pose& p);   // "Y +0.1234 mm, Θ +0.0123°"

} // namespace sim

#endif
