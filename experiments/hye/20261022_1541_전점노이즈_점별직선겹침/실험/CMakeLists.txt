cmake_minimum_required(VERSION 3.15)
project(eda_all_points LANGUAGES CXX)
# 원래 이 파일은 src/export.cpp (hw_eda 측정기 호출부)를 빌드했다. 그 경로(../hw-analysis)가 없어져서
# 측정기를 src/hw_measure.hpp 로 옮기고 분석 전체를 src/main.cpp 한 프로그램으로 다시 만들었다. 원래 내용은 git 기록에 있다.
include("${CMAKE_CURRENT_SOURCE_DIR}/../../00001_common/hye_experiment.cmake")
hye_experiment(eda_all_points src/main.cpp)
