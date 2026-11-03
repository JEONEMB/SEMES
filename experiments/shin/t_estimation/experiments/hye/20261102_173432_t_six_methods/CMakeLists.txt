add_library(six_methods STATIC src/measure.cpp src/models.cpp src/direct.cpp)
target_include_directories(six_methods PUBLIC include ${PROJECT_SOURCE_DIR}/modules/qrwithbar_lanczos/src)
target_link_libraries(six_methods PUBLIC qrwithbar_lanczos t_model)
target_compile_options(six_methods PRIVATE /utf-8 /EHsc)
add_executable(six_extract tools/extract.cpp)
target_link_libraries(six_extract PRIVATE six_methods)
target_compile_options(six_extract PRIVATE /utf-8 /EHsc)
add_custom_command(TARGET six_extract POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
 "${OpenCV_DIR}/x64/vc15/bin/opencv_world3415$<$<CONFIG:Debug>:d>.dll" "$<TARGET_FILE_DIR:six_extract>")


add_executable(six_compare tools/compare.cpp)
target_link_libraries(six_compare PRIVATE six_methods)
target_compile_options(six_compare PRIVATE /utf-8 /EHsc)
add_executable(six_checks tools/checks.cpp)
target_link_libraries(six_checks PRIVATE six_methods)
target_compile_options(six_checks PRIVATE /utf-8 /EHsc)
foreach(target six_compare six_checks)
 add_custom_command(TARGET ${target} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
 "${OpenCV_DIR}/x64/vc15/bin/opencv_world3415$<$<CONFIG:Debug>:d>.dll" "$<TARGET_FILE_DIR:${target}>")
endforeach()
add_test(NAME six_method_isolation COMMAND six_checks)
add_executable(six_estimate tools/main.cpp)
target_link_libraries(six_estimate PRIVATE six_methods)
target_compile_options(six_estimate PRIVATE /utf-8 /EHsc)
add_custom_command(TARGET six_estimate POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
 "${OpenCV_DIR}/x64/vc15/bin/opencv_world3415$<$<CONFIG:Debug>:d>.dll" "$<TARGET_FILE_DIR:six_estimate>")
add_executable(six_sessions tools/sessions.cpp)
target_link_libraries(six_sessions PRIVATE six_methods)
target_compile_options(six_sessions PRIVATE /utf-8 /EHsc)
add_custom_command(TARGET six_sessions POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
 "${OpenCV_DIR}/x64/vc15/bin/opencv_world3415$<$<CONFIG:Debug>:d>.dll" "$<TARGET_FILE_DIR:six_sessions>")
add_executable(six_enrich tools/enrich.cpp)
target_link_libraries(six_enrich PRIVATE six_methods)
target_compile_options(six_enrich PRIVATE /utf-8 /EHsc)
add_custom_command(TARGET six_enrich POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
 "${OpenCV_DIR}/x64/vc15/bin/opencv_world3415$<$<CONFIG:Debug>:d>.dll" "$<TARGET_FILE_DIR:six_enrich>")
add_executable(six_combo tools/combine.cpp)
target_link_libraries(six_combo PRIVATE six_methods)
target_compile_options(six_combo PRIVATE /utf-8 /EHsc)
add_custom_command(TARGET six_combo POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
 "${OpenCV_DIR}/x64/vc15/bin/opencv_world3415$<$<CONFIG:Debug>:d>.dll" "$<TARGET_FILE_DIR:six_combo>")
add_executable(six_diagnose tools/diagnose.cpp)
target_link_libraries(six_diagnose PRIVATE six_methods)
target_compile_options(six_diagnose PRIVATE /utf-8 /EHsc)
add_custom_command(TARGET six_diagnose POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
 "${OpenCV_DIR}/x64/vc15/bin/opencv_world3415$<$<CONFIG:Debug>:d>.dll" "$<TARGET_FILE_DIR:six_diagnose>")
add_executable(six_report tools/report.cpp)
target_link_libraries(six_report PRIVATE six_methods)
target_compile_options(six_report PRIVATE /utf-8 /EHsc)
add_custom_command(TARGET six_report POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
 "${OpenCV_DIR}/x64/vc15/bin/opencv_world3415$<$<CONFIG:Debug>:d>.dll" "$<TARGET_FILE_DIR:six_report>")
