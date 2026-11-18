// 모든 소스 맨 앞에 강제로 넣는 헤더 (CMake: -include linux_prelude.hpp).
//
// 저장소 include/alignment.hpp 의 `typedef double error_t;` 는 리눅스 glibc 의
// `typedef int error_t;` (bits/types/error_t.h) 와 이름이 겹쳐 빌드가 안 된다 (윈도우는 문제없음).
// glibc 쪽을 먼저 다 읽어 둔 뒤, 그 뒤에 나오는 error_t 만 다른 이름으로 바꿔 읽게 한다.
// 팀 헤더를 고치면(예: alignment_error_t 로 이름 변경) 이 파일은 필요 없다.
#pragma once
#include <cerrno>
#include <opencv2/opencv.hpp>
#define error_t alignment_error_t
