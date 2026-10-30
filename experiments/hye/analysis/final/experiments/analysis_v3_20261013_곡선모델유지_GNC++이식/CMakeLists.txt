add_executable(historical_v3_evaluate src/run_v3.cpp)
target_link_libraries(historical_v3_evaluate PRIVATE alignment)
add_custom_command(TARGET historical_v3_evaluate POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
  "${OpenCV_DIR}/x64/vc15/bin/opencv_world3415$<$<CONFIG:Debug>:d>.dll" "$<TARGET_FILE_DIR:historical_v3_evaluate>")
