# CMakeList.txt: test_cmake_gpufit 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required (VERSION 3.18)

project ("test_cmake_gpufit")

enable_language(C)
enable_language(CXX)
set(CMAKE_CXX_STANDARD 11)

set(PRJ_SRC_LIST)
set(PRJ_HEADER_LIST)
file(GLOB PRJ_HEADER_LIST ${PROJECT_SOURCE_DIR}/include/*.h)
file(GLOB PRJ_SRC_LIST ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cc ${PROJECT_SOURCE_DIR}/src/*.cu)
add_executable(${PROJECT_NAME} ${PRJ_SRC_LIST} ${PRJ_HEADER_LIST})
target_include_directories(${PROJECT_NAME} PRIVATE	${PROJECT_SOURCE_DIR}/include)
message(STATUS "PROJECT_SOURCE_DIR = ${PROJECT_SOURCE_DIR}")

option(USE_CUDA "Use CUDA" ON)
if(USE_CUDA)      
	set(CUDA_TOOLKIT_ROOT_DIR  "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8")
    enable_language(CUDA)
    # set(CMAKE_CUDA_ARCHITECTURES 75) # https://discourse.cmake.org/t/correct-use-of-cmake-cuda-architectures/1250/4
	set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "75")
	set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
		
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14")
	if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -g -G -lineinfo)
    endif()
    message(STATUS "Final CUDA_NVCC_FLASG: ${CUDA_NVCC_FLAGS}")	
endif(USE_CUDA)

add_subdirectory(gpufit)

