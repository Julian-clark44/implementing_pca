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

add_executable(imageCompression image_compression.cc)
target_link_libraries(imageCompression
	Eigen3::Eigen
	${OpenCV_LIBS})

add_executable(planeReconstruction plane_reconstruction.cc)
target_link_libraries(planeReconstruction
	Eigen3::Eigen
	Sophus::Sophus
	${Pangolin_LIBRARIES})
