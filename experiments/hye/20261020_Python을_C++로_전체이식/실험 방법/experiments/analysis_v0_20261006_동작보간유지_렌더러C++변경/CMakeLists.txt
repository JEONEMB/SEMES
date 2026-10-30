add_executable(historical_robot_motion src/make_robot_motion.cpp)
target_link_libraries(historical_robot_motion PRIVATE alignment windowscodecs ole32 gdi32 user32)
add_custom_command(TARGET historical_robot_motion POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
  "${OpenCV_DIR}/x64/vc15/bin/opencv_world3415$<$<CONFIG:Debug>:d>.dll" "$<TARGET_FILE_DIR:historical_robot_motion>")
