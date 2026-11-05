# examples/joint_style/plugin.cmake — 예시 : 팀원 코드가 추가 라이브러리를 쓸 때 (team/<이름>/plugin.cmake 로 복사)
#   변수 name = 폴더 이름, 대상 이름 = sim_plugin_${name}
if(WIN32)
    target_link_libraries(sim_plugin_${name} PRIVATE bcrypt)   # Windows 해시 함수 (BCrypt*)
endif()
# 자산 폴더를 DLL 옆으로 복사 (plugins/<이름>/assets) — sim_hooks.cpp 의 기본 자산 경로
if(EXISTS "${dir}/assets")
    add_custom_command(TARGET sim_plugin_${name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${dir}/assets" "${SIM_PLUGIN_OUT_DIR}/${name}/assets")
endif()
