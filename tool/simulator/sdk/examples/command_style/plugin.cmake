# examples/command_style/plugin.cmake — 설정 파일(configs/*.json)을 DLL 옆으로 복사 (team/<이름>/plugin.cmake 로 복사)
if(EXISTS "${dir}/configs")
    add_custom_command(TARGET sim_plugin_${name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${dir}/configs" "${SIM_PLUGIN_OUT_DIR}/${name}/configs")
endif()
