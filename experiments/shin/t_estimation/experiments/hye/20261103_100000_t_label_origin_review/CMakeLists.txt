add_executable(t_label_review src/review.cpp)
target_link_libraries(t_label_review PRIVATE six_methods)
target_compile_options(t_label_review PRIVATE /utf-8 /EHsc)
add_custom_command(TARGET t_label_review POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
 "${OpenCV_DIR}/x64/vc15/bin/opencv_world3415$<$<CONFIG:Debug>:d>.dll" "$<TARGET_FILE_DIR:t_label_review>")
