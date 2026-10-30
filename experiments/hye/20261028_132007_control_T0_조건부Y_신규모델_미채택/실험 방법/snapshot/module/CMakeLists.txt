# The common library owns both stages. This file adds only the stage2 body.
# No application/tool main is compiled into the inference module.
target_sources(semes_alignment PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/control.cpp")
