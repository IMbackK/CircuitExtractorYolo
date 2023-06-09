cmake_minimum_required(VERSION 3.0)

project(circutextractoryolo)

set(SRC_FILES
	src/log.cpp
	src/popplertocv.cpp
	src/yolo.cpp
	src/circut.cpp
	src/document.cpp
	src/linedetection.cpp
	src/utils.cpp
	src/randomgen.cpp
	src/element.cpp
	src/net.cpp
	src/graph.cpp
	src/thinning.cpp
	src/tokenize.cpp
	src/resources.cpp
	)

set(RESOURCE_LOCATION data)

find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(POPPLER REQUIRED poppler-cpp)

link_directories(${CMAKE_CURRENT_BINARY_DIR})

add_executable(${PROJECT_NAME} ${SRC_FILES} src/main.cpp)
target_link_libraries( ${PROJECT_NAME} pthread ${OpenCV_LIBS} ${POPPLER_LINK_LIBRARIES})
target_include_directories(${PROJECT_NAME} PRIVATE  ${OpenCV_INCLUDE_DIRS} ${POPPLER_INCLUDE_DIRS} ${RESOURCE_LOCATION})
add_dependencies(${PROJECT_NAME} ${PROJECT_NAME})
target_compile_options(${PROJECT_NAME} PRIVATE "-std=c++2a" "-Wall" "-O2" "-g" "-fno-strict-aliasing" "-Wfatal-errors" "-Wno-reorder")

add_executable(${PROJECT_NAME}_test ${SRC_FILES} src/test.cpp)
target_link_libraries( ${PROJECT_NAME}_test pthread ${OpenCV_LIBS} ${POPPLER_LINK_LIBRARIES})
target_include_directories(${PROJECT_NAME}_test PRIVATE  ${OpenCV_INCLUDE_DIRS} ${POPPLER_INCLUDE_DIRS} ${RESOURCE_LOCATION})
add_dependencies(${PROJECT_NAME}_test ${PROJECT_NAME})
target_compile_options(${PROJECT_NAME}_test PRIVATE "-std=c++2a" "-Wall" "-O0" "-g" "-fno-strict-aliasing" "-Wfatal-errors" "-Wno-reorder")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -s")

set(CMAKE_INSTALL_PREFIX "/usr")
install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
