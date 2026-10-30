add_executable(historical_v2_estimate src/estimate.cpp)
target_link_libraries(historical_v2_estimate PRIVATE alignment)
add_custom_command(TARGET historical_v2_estimate POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
  "${OpenCV_DIR}/x64/vc15/bin/opencv_world3415$<$<CONFIG:Debug>:d>.dll" "$<TARGET_FILE_DIR:historical_v2_estimate>")
add_executable(historical_v2_homography src/extract_homography.cpp)
target_link_libraries(historical_v2_homography PRIVATE alignment)
add_custom_command(TARGET historical_v2_homography POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${OpenCV_DIR}/x64/vc15/bin/opencv_world3415$<$<CONFIG:Debug>:d>.dll" "$<TARGET_FILE_DIR:historical_v2_homography>")
add_executable(historical_v2_judge src/judge.cpp)
target_link_libraries(historical_v2_judge PRIVATE alignment)
add_custom_command(TARGET historical_v2_judge POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${OpenCV_DIR}/x64/vc15/bin/opencv_world3415$<$<CONFIG:Debug>:d>.dll" "$<TARGET_FILE_DIR:historical_v2_judge>")
add_executable(historical_v2_outputs src/make_outputs.cpp)
target_link_libraries(historical_v2_outputs PRIVATE alignment gdi32)
add_executable(historical_v2_workbook_check src/workbook_check.cpp)
target_link_libraries(historical_v2_workbook_check PRIVATE alignment)
foreach(app historical_v2_outputs historical_v2_workbook_check)
  add_custom_command(TARGET ${app} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${OpenCV_DIR}/x64/vc15/bin/opencv_world3415$<$<CONFIG:Debug>:d>.dll" "$<TARGET_FILE_DIR:${app}>")
endforeach()
