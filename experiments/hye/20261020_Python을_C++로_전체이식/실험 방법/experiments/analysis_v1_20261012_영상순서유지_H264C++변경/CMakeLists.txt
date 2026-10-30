add_executable(historical_video_preview src/make_preview.cpp)
target_link_libraries(historical_video_preview PRIVATE alignment mfplat mfreadwrite mfuuid ole32)
add_custom_command(TARGET historical_video_preview POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  "${OpenCV_DIR}/x64/vc15/bin/opencv_world3415$<$<CONFIG:Debug>:d>.dll" "$<TARGET_FILE_DIR:historical_video_preview>"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  "${OpenCV_DIR}/x64/vc15/bin/opencv_ffmpeg3415_64.dll" "$<TARGET_FILE_DIR:historical_video_preview>")
