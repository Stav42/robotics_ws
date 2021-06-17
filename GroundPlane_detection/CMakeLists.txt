cmake_minimum_required (VERSION 3.5)

project (GroundPlaneDetection)


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-fpermissive")
set(CMAKE_CXX_COMPILER /usr/bin/g++)
set(CMAKE_BUILD_TYPE "Release")

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED QUIET COMPONENTS core features filters io segmentation surface)

set(dependencies
  OpenCV
  EIGEN3
  PCL
)

include_directories(
    include
    ${OpenCV_INCLUDE_DIRS} 
    ${PCL_INCLUDE_DIRS}
    /usr/include/eigen3
)


file(GLOB sources "*.cpp")

add_executable(${PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/src/detection.cpp)
target_link_libraries(GroundPlaneDetection ${OpenCV_LIBS} ${PCL_LIBRARIES})
