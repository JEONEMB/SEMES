// =====================================================================================
// sim_runner.hpp — 영점 맞춤 시뮬레이터 : 로봇 + 카메라(합성 사진) + 모델(alignment) + 제어 반복
//
//  한 번의 시험 (trial)
//    0. 시작 자세(참값)로 로봇을 놓음
//    1. 현재 자세의 사진을 합성 (촬영 잡음 포함)
//    2. alignment(사진) → 추정 어긋남 (y, t)
//    3. 추정값이 '멈춤 범위'(허용 범위 × stop_ratio) 안이면 멈춤
//    4. 아니면 보정 이동 (−y, −t) 명령 → 한 번 최대 2 mm / 2° 로 자름 → 로봇 오차(배율·반복 정밀도)를 더해 실제 이동
//    5. 1 로 돌아감. 이동 20회를 넘기면 실패
//    판정 : 멈춘 뒤 '참값' 이 허용 범위(Y ±90 µm, Θ ±0.01°) 안이면 성공
//  기준(티칭) 사진 : 자세 (0, 0) 에서 합성한 사진 (시험마다 새 잡음)
// =====================================================================================
#ifndef SIM_RUNNER_HPP
#define SIM_RUNNER_HPP

#include "sim_world.hpp"

#include <atomic>
#include <functional>
#include <string>
#include <vector>

namespace sim {

struct StepRecord {
    int k = 0;                 // 0 = 시작 측정, 1.. = 이동 뒤 측정
    Pose truth;                // 참 자세
    Pose est;                  // 모델 추정 (NaN = 실패)
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
    // 설정 파일을 읽음 (없으면 기본값으로 만들어 저장). alignment 설정·가상 장면을 불러옴
    bool init(const std::string& configPath, std::string& message);
    bool worldReady() const { return world_.ready(); }
    // 실제 사진으로 가상 장면을 새로 만들고 world 폴더에 저장 (수 분)
    bool buildWorld(const Progress& progress, std::string& message);

    const SimConfig& config() const { return cfg_; }
    const World& world() const { return world_; }

    // 기준(티칭) 사진 등록 : 자세 (0,0) 합성 사진
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
    std::string configPath_;
    cv::Point coarseCenter_{1901, 1068};
    cv::Size coarseWin_{900, 400};
};

std::string poseText(const Pose& p);   // "Y +0.1234 mm, Θ +0.0123°"

} // namespace sim

#endif
