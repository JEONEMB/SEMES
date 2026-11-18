# 실제 장치 구현 (카메라·로봇). 지금은 카메라 자리만 있고 연결되지 않았다.
# 상위 CMake가 OpenCV를 찾은 뒤 add_subdirectory(hardware/device)로 사용한다.
add_library(semes_device STATIC src/device_camera.cpp)
target_include_directories(semes_device PUBLIC include "${PROJECT_SOURCE_DIR}/include" ${OpenCV_INCLUDE_DIRS})
target_link_libraries(semes_device PUBLIC ${OpenCV_LIBS})
if(MSVC)
 target_compile_options(semes_device PRIVATE /utf-8 /EHsc)
 target_compile_definitions(semes_device PRIVATE NOMINMAX)
endif()
