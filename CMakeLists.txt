cmake_minimum_required(VERSION 3.0)

project(circutextractoryolo)

set(SRC_FILES_APP
	src/main.cpp
	src/log.cpp
	src/popplertocv.cpp
	src/yolo.cpp
	src/circut.cpp
	src/document.cpp
	src/linedetection.cpp
	)

find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(POPPLER REQUIRED poppler-cpp)

link_directories(${CMAKE_CURRENT_BINARY_DIR})

add_executable(${PROJECT_NAME} ${SRC_FILES_APP})
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} ${POPPLER_LINK_LIBRARIES})
target_include_directories(${PROJECT_NAME} PRIVATE  ${OpenCV_INCLUDE_DIRS} ${POPPLER_INCLUDE_DIRS})
add_dependencies(${PROJECT_NAME} ${PROJECT_NAME})
target_compile_options(${PROJECT_NAME} PRIVATE "-std=c++17" "-Wall" "-O2" "-g" "-fno-strict-aliasing" "-Wfatal-errors")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -s")

set(CMAKE_INSTALL_PREFIX "/usr")
install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
