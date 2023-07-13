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
file(GLOB PRJ_HEADER_LIST ${PROJECT_SOURCE_DIR}/include/*.h, ${PROJECT_SOURCE_DIR}/include/*.hpp)
file(GLOB PRJ_SRC_LIST ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cc ${PROJECT_SOURCE_DIR}/src/*.cu)
add_executable(${PROJECT_NAME} ${PRJ_SRC_LIST} ${PRJ_HEADER_LIST})
target_include_directories(${PROJECT_NAME} PRIVATE	${PROJECT_SOURCE_DIR}/include)
message(STATUS "PROJECT_SOURCE_DIR = ${PROJECT_SOURCE_DIR}")

option(USE_OPENCV "Use OpenCV" ON)
if(USE_OPENCV)
	set(OpenCV_DIR "D:/packages/opencv/opencv4.5.3-gpu-build/build2/install")
	find_package(OpenCV REQUIRED)
	message(STATUS "OpenCV_VERSION: ${OpenCV_VERSION}")
	if(OpenCV_CUDA_VERSION) # 检查 CUDA 的支持情况
	  message(STATUS "OpenCV supports CUDA")	  
	else()
	  message(STATUS "OpenCV does not support CUDA")	  
	endif()	
    include_directories(${OpenCV_INCLUDE_DIRS}) # Not needed for CMake >= 2.8.11
    target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})		
endif(USE_OPENCV)

option(USE_CUDA "Use CUDA" ON)
if(USE_CUDA)      
	set(CUDA_TOOLKIT_ROOT_DIR  "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8")
    enable_language(CUDA)
    # set(CMAKE_CUDA_ARCHITECTURES 75) # https://discourse.cmake.org/t/correct-use-of-cmake-cuda-architectures/1250/4
	set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "75")
	set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
		
	#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++14)
	if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -g -G -lineinfo)
    endif()
    message(STATUS "Final CUDA_NVCC_FLASG: ${CUDA_NVCC_FLAGS}")	
endif(USE_CUDA)

add_subdirectory(gpufit)

# -------------- libtorch package: start --------------

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(Torch_DIR "E:/package/pytorch/install/Debug/share/cmake/Torch")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(Torch_DIR "E:/package/pytorch/install/Release/share/cmake/Torch")
else()
    message(FATAL_ERROR "Invalid build type. Set CMAKE_BUILD_TYPE to Debug or Release.")
endif()
find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
target_link_libraries(${PROJECT_NAME} PRIVATE "${TORCH_LIBRARIES}")
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET ${PROJECT_NAME}
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endif (MSVC)

# -------------- libtorch package: end --------------

#
find_package(Matplot++ CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE Matplot++::cimg  Matplot++::matplot)