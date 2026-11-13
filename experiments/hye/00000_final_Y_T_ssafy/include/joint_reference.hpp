#pragma once
#include "alignment.hpp"
#include "reference.hpp"
#include <opencv2/core.hpp>
#include <vector>

// joint() 의 영점과 측정 기준. 모두 initialize_joint(assets) 다음에 부른다. 실패하면 예외를 던지고 이전 상태가 그대로 남는다.
//
// [기본] 절대 영점: assets 의 center.bmp 와 같은 화면이 (0, 0). 아무것도 부르지 않으면 이것이다.
//        initialize_joint 를 다시 부르면 아래 등록이 모두 지워지고 절대 영점으로 돌아간다.
//
// [영점 업데이트] set_joint_zero(그날 센터 사진들)
//   그 사진(여러 장이면 화소 평균)의 지금 출력을 새 영점으로 삼아 이후 joint() 출력에서 뺀다. 측정 기준은 그대로다.
//   돌려주는 값 = 그 사진의 절대 영점 기준 자세(그날 센터가 절대 영점에서 떨어진 양). 저장해 두면 set_joint_zero_value 로 복원할 수 있다.
//   센터에는 정렬과 같은 방향(−쪽 접근점을 지나)으로 들어가 흔들림이 멎은 뒤 찍는다. 그 순간의 스테이지 오차가 영점에 들어간다.
Alignment_t set_joint_zero(const cv::Mat& center);
Alignment_t set_joint_zero(const std::vector<cv::Mat>& centers);
void set_joint_zero_value(double y, double t);   // 절대 영점 기준 영점 값을 직접 넣는다
void clear_joint_zero();                         // 절대 영점으로 되돌린다
Alignment_t joint_zero();                        // 지금 영점 (절대 영점 기준, 기본 0, 0)
//
// [측정 기준 다시 잡기] set_joint_reference(그날 센터 사진들)
//   화면이 크게 밀려 측정이 실패할 때만 쓴다(QR 탐색 범위: 가로 ±125, 세로 ±22 px). 센터 사진, QR 상자, 630점, 나사 중심을 그 사진으로 다시 찾는다.
//   계수와 영점은 그대로이고, 같은 사진을 바꾸기 전 기준으로도 재서 절대 영점을 잇는다(report.absolute_kept = true).
//   바꾸기 전 기준으로 잴 수 없을 만큼 밀렸으면 절대 영점을 잃는다: 그 사진이 (0, 0) 이 되고 영점은 지워진다(report.absolute_kept = false).
final00::ReferenceReport set_joint_reference(const cv::Mat& center);
final00::ReferenceReport set_joint_reference(const std::vector<cv::Mat>& centers);
