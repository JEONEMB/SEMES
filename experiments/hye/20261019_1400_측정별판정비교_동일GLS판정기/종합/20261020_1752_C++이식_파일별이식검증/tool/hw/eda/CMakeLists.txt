file(GLOB EDA_SOURCES CONFIGURE_DEPENDS "src/*.cpp")
add_executable(hw_eda ${EDA_SOURCES})
target_include_directories(hw_eda PRIVATE ${OpenCV_INCLUDE_DIRS} "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(hw_eda PRIVATE ${OpenCV_LIBS})
target_compile_definitions(hw_eda PRIVATE WIN32_LEAN_AND_MEAN SEMES_FINAL_DIR="${PROJECT_SOURCE_DIR}")
add_custom_command(TARGET hw_eda POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
  "${OpenCV_DIR}/x64/vc15/bin/opencv_world3415$<$<CONFIG:Debug>:d>.dll" "$<TARGET_FILE_DIR:hw_eda>")
