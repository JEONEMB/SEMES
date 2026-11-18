// =====================================================================================
// ksh/dataset.hpp — 라벨이 붙은 사진 목록 · 특징 CSV (학습·평가 도구용, 추론에는 쓰지 않음)
//
//  파일 이름 = 라벨 (지령값, 센터 대비 상대 이동)
//   center, center1 ~ (숫자)         : 기준(센터) 사진 — 학습 표본에서 빼고 영점 계산에만 씀
//   Y±a                              : Y 단독         T±b : Θ 단독
//   Y±a T±b  /  Y±a T±b (k)          : Y·Θ 동시 (k = 같은 자세 k번째 촬영)
//  폴더 이름이 라벨(Y±a T±b)이고 파일 라벨과 다르면 폴더를 따른다 (3차 'Y+0.05 T+0.32' 폴더 안 'T+0.33 (3)' 오기)
// =====================================================================================
#ifndef KSH_DATASET_HPP
#define KSH_DATASET_HPP

#include "ksh/pose_model.hpp"

#include <string>
#include <vector>

namespace ksh {

struct LabeledPhoto {
    std::string stem;          // 확장자 뺀 이름 (폴더 보정 후)
    std::string path;
    double y_mm = 0.0, t_deg = 0.0;
    bool is_reference = false; // center 류
};

// 이름 → 라벨 ('T+0.01' 의 '.01' 을 확장자로 오인하지 않도록 확장자를 뗀 이름을 넘김)
bool parse_label(const std::string &stem, LabeledPhoto &photo);

// 폴더(하위 폴더 포함)의 .bmp/.png 중 라벨을 읽을 수 있는 사진, 이름 순 (같은 이름이면 bmp 우선)
std::vector<LabeledPhoto> list_labeled_photos(const std::string &dir);

// 특징 CSV — 열 : stem,y_mm,t_deg,s_px,d_px,is_ref  (is_ref = 1 : 센터 사진)
struct FeatureRow {
    FeatureSample sample;
    bool is_reference = false;
};
bool save_feature_csv(const std::string &path, const std::vector<FeatureRow> &rows);
bool load_feature_csv(const std::string &path, std::vector<FeatureRow> &rows);

} // namespace ksh

#endif
