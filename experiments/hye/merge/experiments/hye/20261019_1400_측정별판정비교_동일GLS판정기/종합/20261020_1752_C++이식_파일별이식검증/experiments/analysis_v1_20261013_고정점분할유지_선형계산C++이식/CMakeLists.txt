add_executable(historical_v1_grid src/grid.cpp)
target_link_libraries(historical_v1_grid PRIVATE alignment)
add_custom_command(TARGET historical_v1_grid POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
  "${OpenCV_DIR}/x64/vc15/bin/opencv_world3415$<$<CONFIG:Debug>:d>.dll" "$<TARGET_FILE_DIR:historical_v1_grid>")
add_executable(historical_v1_track src/track_all.cpp)
target_link_libraries(historical_v1_track PRIVATE alignment)
add_custom_command(TARGET historical_v1_track POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
  "${OpenCV_DIR}/x64/vc15/bin/opencv_world3415$<$<CONFIG:Debug>:d>.dll" "$<TARGET_FILE_DIR:historical_v1_track>")
add_executable(historical_v1_combos src/combos.cpp)
target_link_libraries(historical_v1_combos PRIVATE alignment)
add_executable(historical_v1_stable src/stable.cpp)
target_link_libraries(historical_v1_stable PRIVATE alignment gdi32)
foreach(app historical_v1_combos historical_v1_stable)
  add_custom_command(TARGET ${app} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${OpenCV_DIR}/x64/vc15/bin/opencv_world3415$<$<CONFIG:Debug>:d>.dll" "$<TARGET_FILE_DIR:${app}>")
endforeach()
