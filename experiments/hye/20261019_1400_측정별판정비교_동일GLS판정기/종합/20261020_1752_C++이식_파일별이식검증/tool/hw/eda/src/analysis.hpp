// =====================================================================================
// analysis.hpp  -  측정값으로 분석하고 슬라이드를 만든다
//
//   sweep 묶음 (1차): v4 발표자료 구성
//     표지 / 기초 정보 / 기준 위치(QR, 특징점, ECC) / 0점 / 이동 경로(특징점, ECC) / 한 칸 이동량 /
//     노이즈 표 / 노이즈 그림 2장 / 해석(notes.md)
//   grid 묶음 (2차): 기준 묶음(--ref) 결과로 평가
//     표지 / 기초 정보 / 기준 위치 / center 반복 / 같은 조합 반복 흔들림 / 기준 모델 예측 vs 측정 /
//     세로 이동 / 조합별 결과표 / 해석(notes.md)
// =====================================================================================
#pragma once

#include "core.hpp"
#include "measurements.hpp"
#include "pptx.hpp"

namespace hw {

struct Context {
    const DataSet &ds;
    const Measurements &m;
    fs::path out;             // docs/validation-runs/hw/eda/<묶음>
    Deck &deck;
    Summary &sum;             // 이 묶음의 핵심 숫자 (summary.yml)
    const Summary *ref;       // 기준 묶음의 핵심 숫자 (grid 평가용, 없으면 nullptr)
    std::string refName;
    int page = 0;
};

void buildSweepDeck(Context &ctx);
void buildGridDeck(Context &ctx);
void addNotesSlides(Context &ctx);   // docs/hw/eda/sets/<묶음>/notes.md -> 글 카드 슬라이드

}  // namespace hw
