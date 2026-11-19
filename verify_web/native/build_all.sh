#!/bin/bash
# 확인 앱 알고리즘 실행기 3개를 저장소 안 알고리즘 소스로 빌드한다.
#   KSH  = experiments/KSH/branch-snapshot/OutPut_A507/Code    (QR 3개 + 나사 2개, 영점 등록)
#   LHW  = experiments/hye/ssafy                               (QR + 나사 4개, 영점 사진)
#   SHIN = experiments/shin/branch-snapshot/experiments/shin/qr_only_multiple_exe/native (QR 3개, 기준 사진 대비)
# 결과: verify_web/native/build/{ksh,hye,shin}/align_worker  (verify_web/algorithms.json 이 이 경로를 씀)
# 실행: 저장소 어디서든  bash verify_web/native/build_all.sh   (하나만: bash ... ksh)
set -euo pipefail
NATIVE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$NATIVE/../.." && pwd)"
GEN=(); command -v ninja >/dev/null && GEN=(-G Ninja)

build() {   # 이름, cmake 옵션
    local name=$1; shift
    echo "== $name"
    cmake -S "$NATIVE" -B "$NATIVE/build/$name" "${GEN[@]}" -DCMAKE_BUILD_TYPE=Release "$@" >/dev/null
    cmake --build "$NATIVE/build/$name" -j "$(nproc)"
}

ARGS=("$@")
pick() { [ ${#ARGS[@]} -eq 0 ] || [[ " ${ARGS[*]} " == *" $1 "* ]]; }

pick ksh  && build ksh  -DALGO_DIR="$REPO/experiments/KSH/branch-snapshot/OutPut_A507/Code"
pick hye  && build hye  -DHYE2_DIR="$REPO/experiments/hye/ssafy"
pick shin && build shin -DQR_RIG_DIR="$REPO/experiments/shin/branch-snapshot/experiments/shin/qr_only_multiple_exe/native"
echo "완료: $NATIVE/build/{ksh,hye,shin}/align_worker"
