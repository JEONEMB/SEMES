# Separate OpenCV 4 runtime; these targets never link the OpenCV 3 alignment library.
get_filename_component(opencv4_source "${SEMES_OPENCV4_DIR}/../opencv-4.11.0" ABSOLUTE)
# A distinct imported target avoids binding OpenCV 4 headers to the root's
# already-imported OpenCV 3 target with the same upstream name.
add_library(semes_opencv411 SHARED IMPORTED GLOBAL)
set_target_properties(semes_opencv411 PROPERTIES
  IMPORTED_LOCATION "${SEMES_OPENCV4_DIR}/bin/Release/opencv_world4110.dll"
  IMPORTED_IMPLIB "${SEMES_OPENCV4_DIR}/lib/Release/opencv_world4110.lib")
set(opencv4_headers "${SEMES_OPENCV4_DIR}" "${opencv4_source}/include")
foreach(module core imgproc imgcodecs objdetect calib3d features2d flann highgui video videoio)
  list(APPEND opencv4_headers "${opencv4_source}/modules/${module}/include")
endforeach()
set(SEMES_OPENCV4_HEADERS "${opencv4_headers}" CACHE INTERNAL "OpenCV4 headers for isolated historical targets")
foreach(app historical_qr_only historical_ecc_tune)
  if(app STREQUAL historical_qr_only)
    set(source src/qr_only.cpp)
  else()
    set(source src/ecc_tune.cpp)
  endif()
  add_executable(${app} ${source})
  target_include_directories(${app} PRIVATE ${opencv4_headers})
  target_link_libraries(${app} PRIVATE semes_io semes_opencv411)
  add_custom_command(TARGET ${app} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${SEMES_OPENCV4_DIR}/bin/Release/opencv_world4110.dll" "$<TARGET_FILE_DIR:${app}>")
endforeach()
