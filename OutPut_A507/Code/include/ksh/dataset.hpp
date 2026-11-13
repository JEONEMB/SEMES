// =====================================================================================
// ksh/dataset.hpp — 라벨이 붙은 사진 목록 (학습·평가 도구용)
//
//  파일 이름 = 라벨 (정답 명령값)
//   1차(10/04) : center, Y±n.nn (Y 단독), T±n.nn (Θ 단독)
//   2차(10/15) : center1~3, "Y+n.nn T+n.nn (k)" (Y·Θ 동시, 같은 자세 k번째 촬영)
//   A507(11/12): rig_1st = center, Y±, T± / rig_grid = center, "Y± T± (1)" / _check/center_after_NNNN·center_end (센터 재촬영)
//  추론(alignment)에는 라벨을 쓰지 않는다. 라벨은 학습 정답과 평가 채점에만 쓴다
// =====================================================================================
#ifndef KSH_DATASET_HPP
#define KSH_DATASET_HPP

#include "ksh/pose_model.hpp"

#include <string>
#include <vector>

namespace ksh {

struct LabeledPhoto {
    std::string stem;          // 확장자 뺀 이름
    std::string path;
    double y_mm = 0.0, t_deg = 0.0;
    int set = 1;               // 촬영 세트 (1 = 10/04, 2 = 10/15)
    bool is_reference = false; // center 류 (기준 사진)
};

// 이름 → 라벨. 'T+0.01' 의 '.01' 을 확장자로 오인하지 않도록 확장자를 뗀 이름을 넘김
bool parse_label(const std::string &stem, LabeledPhoto &photo);

// 폴더의 .bmp/.png 중 라벨을 읽을 수 있는 사진 (같은 이름이면 bmp 우선), 이름 순
std::vector<LabeledPhoto> list_labeled_photos(const std::string &dir);

// 이름으로 찾기 (없으면 nullptr)
const LabeledPhoto *find_photo(const std::vector<LabeledPhoto> &photos, const std::string &stem);

// 학습 사진 선택 : 지정 세트의 단독 축 사진. Y 단독은 y_grid_mm 의 배수만
bool is_training_photo(const LabeledPhoto &photo, int set, double y_grid_mm);

// 특징 표본 CSV (측정 결과 저장·재사용). 열 : stem,y_mm,t_deg,s_px,d_px
bool save_feature_csv(const std::string &path, const std::vector<FeatureSample> &samples);
bool load_feature_csv(const std::string &path, std::vector<FeatureSample> &samples);

} // namespace ksh

#endif
