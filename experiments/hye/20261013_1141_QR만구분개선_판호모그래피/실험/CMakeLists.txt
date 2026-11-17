cmake_minimum_required(VERSION 3.15)
project(qr_plate_homography LANGUAGES CXX)
include("${CMAKE_CURRENT_SOURCE_DIR}/../../00001_common/hye_experiment.cmake")
hye_experiment(qr_plate_homography src/main.cpp)
