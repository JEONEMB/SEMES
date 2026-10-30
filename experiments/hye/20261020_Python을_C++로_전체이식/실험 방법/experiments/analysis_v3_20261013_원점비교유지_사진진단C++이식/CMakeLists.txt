add_executable(historical_zero_offset src/zero_offset_photos.cpp)
target_link_libraries(historical_zero_offset PRIVATE alignment gdi32)
add_custom_command(TARGET historical_zero_offset POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
  "${OpenCV_DIR}/x64/vc15/bin/opencv_world3415$<$<CONFIG:Debug>:d>.dll" "$<TARGET_FILE_DIR:historical_zero_offset>")
