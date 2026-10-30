add_executable(historical_v3_outputs src/make_v3_outputs.cpp)
target_link_libraries(historical_v3_outputs PRIVATE alignment gdi32)
add_custom_command(TARGET historical_v3_outputs POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
  "${OpenCV_DIR}/x64/vc15/bin/opencv_world3415$<$<CONFIG:Debug>:d>.dll" "$<TARGET_FILE_DIR:historical_v3_outputs>")
