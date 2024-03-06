cmake_minimum_required(VERSION 3.22)
project(pcaImplementations)

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Sophus REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Pangolin REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(basicPca basic_pca.cc)
target_link_libraries(basicPca Eigen3::Eigen)

add_executable(dualPca dual_pca.cc)
target_link_libraries(dualPca Eigen3::Eigen)
