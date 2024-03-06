# Implementing PCA

Requirements:

* Cmake (sudo apt-get install cmake)
* Eigen3 (sudo apt install libeigen3-dev)
* Sophus (https://github.com/strasdat/Sophus)
* OpenCV (https://github.com/opencv/opencv)
* Pangolin (https://github.com/stevenlovegrove/Pangolin)

Four Different Executables:

* basicPca: Manually performs the PCA algorithm.
* dualPca: Manually performs the Dual PCA algorithm.
* imageCompression: Takes in an image file, converts it to grayscale, and compresses it using PCA.
* planeReconstruction: Starts with a grid of the integer-valued coordinates from (-5, -5) to (5, 5). Adds a small amount of random noise to each point. Applies the same random rotation and translation to all points, embedding the noisy flat 2D plane in 3D space. Estimates the plane using PCA with d=2. Theoretically, this should be very close to perfect since the rotation and translation are linear transformations of a surface that can be parametrized with 2 coordinates.
