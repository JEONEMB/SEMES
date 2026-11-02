cmake_minimum_required(VERSION 3.15)
project(reference_line_methods LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV 3.4 REQUIRED)
set(TP ${PROJECT_SOURCE_DIR}/../third_party)

# Third-party detectors (see third_party/*/PROVENANCE.md). Kept in separate libraries so their
# warnings and licences stay separated from the tool sources.
add_library(tp_c STATIC ${TP}/lsd_1.6/lsd.c ${TP}/devernay_1.0/devernay.c)
target_include_directories(tp_c PUBLIC ${TP}/lsd_1.6 ${TP}/devernay_1.0)

add_library(tp_cpp STATIC
    ${TP}/ED_Lib/ED.cpp ${TP}/ED_Lib/EDLines.cpp ${TP}/ED_Lib/EDColor.cpp ${TP}/ED_Lib/NFA.cpp
    ${TP}/ELSED/ELSED.cpp ${TP}/ELSED/EdgeDrawer.cpp ${TP}/ELSED/FullSegmentInfo.cpp
    ${TP}/fld_port/fld_port.cpp)
target_include_directories(tp_cpp PUBLIC ${TP}/ED_Lib ${TP}/ELSED ${TP}/fld_port ${OpenCV_INCLUDE_DIRS})
target_link_libraries(tp_cpp PUBLIC ${OpenCV_LIBS})
if(MSVC)
    target_compile_definitions(tp_cpp PRIVATE _USE_MATH_DEFINES _CRT_SECURE_NO_WARNINGS)
    target_compile_definitions(tp_c PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_compile_options(tp_cpp PRIVATE /utf-8 /W1 /bigobj)
    target_compile_options(tp_c PRIVATE /W1)
endif()

add_executable(line_methods
    src/main.cpp src/io_utils.cpp src/line_fit.cpp src/registry.cpp
    src/segment_support.cpp src/effective_settings.cpp
    src/methods_points.cpp src/methods_hough.cpp src/methods_segments.cpp src/methods_qr.cpp)
target_include_directories(line_methods PRIVATE include ${TP}/nlohmann_json/include)
target_link_libraries(line_methods PRIVATE tp_c tp_cpp ${OpenCV_LIBS})
if(MSVC)
    target_compile_definitions(line_methods PRIVATE _USE_MATH_DEFINES NOMINMAX _CRT_SECURE_NO_WARNINGS)
    target_compile_options(line_methods PRIVATE /utf-8 /W4 /bigobj /wd4244 /wd4267)
endif()

# Known geometry / configuration contract checks. Does not run native image detectors.
add_executable(pipeline_contract_test tests/pipeline_contract_test.cpp src/segment_support.cpp
    src/effective_settings.cpp src/methods_segments.cpp src/line_fit.cpp src/io_utils.cpp)
target_include_directories(pipeline_contract_test PRIVATE include ${TP}/nlohmann_json/include)
target_link_libraries(pipeline_contract_test PRIVATE tp_c tp_cpp ${OpenCV_LIBS})
if(MSVC)
    target_compile_definitions(pipeline_contract_test PRIVATE _USE_MATH_DEFINES NOMINMAX _CRT_SECURE_NO_WARNINGS)
    target_compile_options(pipeline_contract_test PRIVATE /utf-8 /W4 /bigobj /wd4244 /wd4267)
endif()
if(WIN32)
    add_custom_command(TARGET pipeline_contract_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OpenCV_DIR}/x64/vc15/bin/opencv_world3415$<$<CONFIG:Debug>:d>.dll"
            "$<TARGET_FILE_DIR:pipeline_contract_test>")
endif()

if(WIN32)
    add_custom_command(TARGET line_methods POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OpenCV_DIR}/x64/vc15/bin/opencv_world3415$<$<CONFIG:Debug>:d>.dll"
            "$<TARGET_FILE_DIR:line_methods>")
endif()

# Known-point checks of the line/angle conventions (no image data).
add_executable(line_selftest tests/line_selftest.cpp src/line_fit.cpp src/io_utils.cpp)
target_include_directories(line_selftest PRIVATE include ${TP}/nlohmann_json/include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(line_selftest PRIVATE ${OpenCV_LIBS})
if(WIN32)
    add_custom_command(TARGET line_selftest POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OpenCV_DIR}/x64/vc15/bin/opencv_world3415$<$<CONFIG:Debug>:d>.dll"
            "$<TARGET_FILE_DIR:line_selftest>")
endif()
if(MSVC)
    target_compile_definitions(line_selftest PRIVATE _USE_MATH_DEFINES NOMINMAX)
    target_compile_options(line_selftest PRIVATE /utf-8 /W4)
endif()
