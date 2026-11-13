#pragma once
#include "reference.hpp"
#include <opencv2/core.hpp>
#include <vector>

// joint() 의 기준 다시 잡기. initialize_joint(assets) 다음에, 그날 센터에서 찍은 사진으로 부른다.
//   - 측정 기준(센터 사진, QR·나사·바 위치)을 그 사진으로 다시 만든다. 계수(model7.tsv)는 그대로다.
//   - 이후 joint() 의 출력은 등록한 센터가 (0, 0) 인 자세다(등록 사진 자체는 정확히 0, 0).
//   - 여러 장을 주면 화소 평균으로 등록한다(흔들림을 줄인다).
//   - 실패하면 예외를 던지고 이전 상태가 그대로 남는다.
//   - initialize_joint 를 다시 부르면 등록이 지워지고 assets 의 절대 영점으로 돌아간다.
// 확인 앱(verify_web) 의 alignment_set_reference 에 이어 붙일 때 이 함수를 부른다.
final00::ReferenceReport set_joint_reference(const cv::Mat& center);
final00::ReferenceReport set_joint_reference(const std::vector<cv::Mat>& centers);
