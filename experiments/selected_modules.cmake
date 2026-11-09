# 빌드에 사용할 1차·2차 모듈을 고른다. 경로는 저장소 루트 기준이다.
# 이 파일을 고치면 다음 빌드 때 CMake가 자동으로 다시 구성한다.
#
# 후보 모듈 조건
#   1차(joint):   CMakeLists.txt가 semes_joint 대상을 만들고 initialize_joint()·joint()를 구현한다.
#   2차(control): CMakeLists.txt가 semes_control 대상을 만들고 initialize_control()·control()을 구현한다.
#   두 모듈 모두 실행 자료를 assets/ 폴더에 둔다.

set(SEMES_JOINT_DIR   "experiments/hye/00000_final_QR_bar_pin66")
set(SEMES_CONTROL_DIR "experiments/hye/00000_final_Y_T control")
