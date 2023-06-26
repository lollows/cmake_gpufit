
# ���� GPUFit ���ļ�·��
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(GPUFIT_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/debug")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(GPUFIT_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/release")
else()
    message(FATAL_ERROR "Invalid build type. Set CMAKE_BUILD_TYPE to Debug or Release.")
endif()
message(STATUS "GPUFIT_LIB_DIR = ${GPUFIT_LIB_DIR}")

# ���� GPUFit ��صĿ��ļ�
file(GLOB GPUFIT_LIBRARIES "${GPUFIT_LIB_DIR}/*.dll")
file(GLOB GPUFIT_LIBS "${GPUFIT_LIB_DIR}/*.lib")

# ���� GPUFit ��̬��
add_library(gpufit SHARED IMPORTED)
set_target_properties(gpufit PROPERTIES 
    IMPORTED_IMPLIB ${GPUFIT_LIBS}
    IMPORTED_LOCATION "${GPUFIT_LIBRARIES}")

# ���� GPUFit ��ͷ�ļ�Ŀ¼
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include) # ��ʹ�����·��������gpufit.h
# �� target_include_directories �� ${CMAKE_CURRENT_SOURCE_DIR}/include ���ӵ� ${PROJECT_NAME} 
# Ŀ��İ���·���У�ֻ�� ${PROJECT_NAME} Ŀ�꼰������Ŀ���Դ�ļ���Ч�����ֱ�Ӱ���gpufit.h
target_include_directories(${PROJECT_NAME} PRIVATE	${CMAKE_CURRENT_SOURCE_DIR}/include) #

# �� GPUFit ���ļ����ӵ�����Ŀ
target_link_libraries(${PROJECT_NAME} gpufit)

# ������Ӧ�汾�� DLL �ļ������Ŀ¼
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