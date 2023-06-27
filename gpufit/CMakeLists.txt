
# 设置 GPUFit 库文件路径
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(GPUFIT_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/debug")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(GPUFIT_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/release")
else()
    message(FATAL_ERROR "Invalid build type. Set CMAKE_BUILD_TYPE to Debug or Release.")
endif()
message(STATUS "GPUFIT_LIB_DIR = ${GPUFIT_LIB_DIR}")

# 设置 GPUFit 相关的库文件
file(GLOB GPUFIT_LIBRARIES "${GPUFIT_LIB_DIR}/*.dll")
file(GLOB GPUFIT_LIBS "${GPUFIT_LIB_DIR}/*.lib")

# 构建 GPUFit 动态库
add_library(gpufit SHARED IMPORTED)
set_target_properties(gpufit PROPERTIES 
    IMPORTED_IMPLIB ${GPUFIT_LIBS}
    IMPORTED_LOCATION "${GPUFIT_LIBRARIES}")

# 添加 GPUFit 的头文件目录
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include) # 需使用相对路径来包含gpufit.h
# 而 target_include_directories 将 ${CMAKE_CURRENT_SOURCE_DIR}/include 添加到 ${PROJECT_NAME} 
# 目标的包含路径中，只对 ${PROJECT_NAME} 目标及其依赖目标的源文件生效，因此直接包含gpufit.h
target_include_directories(${PROJECT_NAME} PRIVATE	${CMAKE_CURRENT_SOURCE_DIR}/include) #

# 将 GPUFit 库文件链接到主项目
target_link_libraries(${PROJECT_NAME} gpufit)

# 复制相应版本的 DLL 文件到输出目录
message(STATUS "CMAKE_BINARY_DIR = ${CMAKE_BINARY_DIR}")
file(COPY 
    ${GPUFIT_LIB_DIR}/
    DESTINATION ${CMAKE_BINARY_DIR}
    FILES_MATCHING
    PATTERN "*.dll"
)

# CMake Error at gpufit/CMakeLists.txt:39 (add_custom_command):
# TARGET 'test_cmake_gpufit' was not created in this directory
#add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
#    COMMAND ${CMAKE_COMMAND} -E copy_directory
#    ${GPUFIT_LIB_DIR}
#    $<TARGET_FILE_DIR:${PROJECT_NAME}>
#)