#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <random>
#include <unistd.h>
#include <cmath>

#include <pangolin/pangolin.h>
#include <sophus/so3.hpp>
#include <Eigen/Core>
#include <Eigen/SVD>

using namespace Eigen;

// Credit: "Introduction to Visual SLAM: From Theory to Practice"
void showPointClouds(const std::vector<Vector3f, aligned_allocator<Vector3f>>& truth_cloud,
		const std::vector<Vector3f, aligned_allocator<Vector3f>>& estimated_cloud) {

    pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glPointSize(15);
        glBegin(GL_POINTS);
        for (auto &p: truth_cloud) {
            glColor3f(0, 255, 0);
            glVertex3d(p[0], p[1], p[2]);
        }
				for (auto &p: estimated_cloud) {
						glColor3f(0, 0, 255);
						glVertex3d(p[0], p[1], p[2]);
				}
        glEnd();
        pangolin::FinishFrame();
				usleep(5000); // sleep 5 ms
    }
    return;
}

int main(int argc, char **argv) {
	// Create a uniform distribution from -1 to 1.
	std::random_device r;
	std::default_random_engine e(r());
	std::uniform_real_distribution<float> dist(-1, 1);

	// Generate a random SO(3) object.
	Vector3f so3 = {10 * dist(e), 10 * dist(e), 10 * dist(e)};
	Sophus::SO3f R = Sophus::SO3f::exp(so3);

	// Generate the surface of a plane with a small amount of noise, and rotate it by the
	// random SO(3) object.
	std::vector<Vector3f, aligned_allocator<Vector3f>> truth_cloud;
	for (int i = -5; i <= 5; i++)
	for (int j = -5; j <= 5; j++) {
		Vector3f p = {i, j, 0};
		Vector3f noise = {dist(e) * 0.35, dist(e) * 0.35, dist(e) * 0.35};
		Vector3f p_final = R * (p + noise);
		truth_cloud.push_back(std::move(p_final));
	};

	// Convert the truth point cloud into a matrix.
	int num_points = truth_cloud.size();
	Matrix3Xf truth_cloud_mat;
	truth_cloud_mat.resize(3, num_points);
	for (int i = 0; i < num_points; i++) {
		truth_cloud_mat.col(i) = truth_cloud[i];
	}

	// Calculate the SVD.
	BDCSVD<Matrix3Xf> svd(truth_cloud_mat, ComputeThinV);
	MatrixXf V = svd.matrixV();

	// Resize V.
	V.conservativeResize(num_points, 2);

	// Reconstruct the point cloud from the projection.
	Matrix3Xf estimated_cloud_mat
		= truth_cloud_mat * V * V.transpose();

	// Get the estimated point cloud into a vector.
	std::vector<Vector3f, aligned_allocator<Vector3f>> estimated_cloud;
	for (int i = 0; i < num_points; i++) {
		estimated_cloud.push_back(estimated_cloud_mat.col(i));
	}

	// Display the results.
	showPointClouds(truth_cloud, estimated_cloud);
}
