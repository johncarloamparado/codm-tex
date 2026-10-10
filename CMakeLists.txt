cmake_minimum_required(VERSION 3.18)
project(codm_tex)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/libs)
set(DEBUG_SYMBOLS_PATH ${CMAKE_BINARY_DIR}/symbols)

# LSPlt (PLT hook library) is cloned next to this file by the CI workflow.
add_subdirectory(LSPlt/lsplt/src/main/jni lsplt)

# Zygisk module library; packaged as zygisk/<abi>.so.
add_library(codm_tex SHARED main.cpp)
target_include_directories(codm_tex PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/LSPlt/lsplt/src/main/jni/include)
target_link_libraries(codm_tex lsplt_static log android)
