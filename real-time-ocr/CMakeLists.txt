cmake_minimum_required(VERSION 3.10.2)

STRING(REGEX REPLACE ".*/(.*)" "\\1" CURRENT_FOLDER ${CMAKE_CURRENT_SOURCE_DIR})
MESSAGE("current project: " ${CURRENT_FOLDER})

set(CMAKE_SYSTEM_NAME Linux)

cmake_host_system_information(RESULT arch_value QUERY OS_PLATFORM)
message(STATUS "architecture: " ${arch_value})

if("${arch_value}" STREQUAL "armv7l")
	set(CMAKE_C_COMPILER "gcc")
	set(CMAKE_CXX_COMPILER "g++")
else()
	include ($ENV{HOME}/configs/cross.cmake)
endif()


project(${CURRENT_FOLDER})

add_definitions(-Wall)

set(toolkit_root ${CMAKE_CURRENT_SOURCE_DIR}/../easyeai-api)

# -I
set(api_inc
		${CMAKE_SYSROOT}/usr/include/rkmedia
		${toolkit_root}/peripheral_api/camera
		${toolkit_root}/peripheral_api/display
		${CMAKE_SYSROOT}/usr/include/rkfacial
		${toolkit_root}/algorithm_api/ocr/
)

link_directories(${toolkit_root}/peripheral_api/camera)
link_directories(${toolkit_root}/peripheral_api/display)
link_directories(${toolkit_root}/algorithm_api/ocr/)

find_package(OpenCV REQUIRED)
add_executable(real-time-ocr real-time-ocr.cpp)
target_link_libraries(real-time-ocr camera display easymedia rga rkaiq pthread ocr rknn_api  ${OpenCV_LIBS} )
target_include_directories(real-time-ocr PRIVATE ${api_inc} ${OpenCV_INCLUDE_DIRS})
