cmake_minimum_required(VERSION 3.15)
project(ssafy_qr_pose LANGUAGES CXX)

find_package(OpenCV 3.4 REQUIRED)
add_executable(ssafy_qr_pose src/alignment.cpp src/main.cpp)
target_compile_features(ssafy_qr_pose PRIVATE cxx_std_17)
target_include_directories(ssafy_qr_pose PRIVATE include ${OpenCV_INCLUDE_DIRS})
target_compile_definitions(ssafy_qr_pose PRIVATE
    QR_POSE_DEFAULT_MODEL="${CMAKE_CURRENT_SOURCE_DIR}/config/qr_pose.yml")
target_link_libraries(ssafy_qr_pose PRIVATE ${OpenCV_LIBS})

if(WIN32)
    add_custom_command(TARGET ssafy_qr_pose POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OpenCV_DIR}/x64/vc15/bin/opencv_world3415$<$<CONFIG:Debug>:d>.dll"
            "$<TARGET_FILE_DIR:ssafy_qr_pose>")
endif()
