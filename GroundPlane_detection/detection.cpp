#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cmath>
#include <eigen3/Eigen/Core>


using namespace cv;
using namespace std;
using namespace Eigen;

/* GroundDetection 
    Input format: Depth Image.
    Output Image: Backprojected image with plane
    Parameters: fx, fy, cx, cy, camera factor (intrinsic parameters)

    
    Methodology:
        1. RGBD -> Point Cloud
        2. Median Weight Filter
        3. Occupancy grid construction
        4. Ground plane thresholding and segmentation
        5. Back projection

*/

/* Set fx, fy, cx, cy*/

float fx = 2;
float fy = 2;
float cx = 2;
float cy = 5;


class camera_prop{
    public:
    float fx;
    float fy;
    float cx;
    float cy;
};

Matrix3f intr_matrix(float fx, float fy, float cx, float cy){

    Matrix3f cam = Matrix3f::Zero();  
    cam(0,0) = fx;
    cam(1,1) = fy;
    cam(0,2) = cx;
    cam(1,2) = cy;
    cam(2,2) = 1;
    return cam;
}


Point3f cart_cood(camera_prop cam, Point2f point, float d){
    
    Point3f point_3d;
    float x_over_z = (cam.cx - point.x) / cam.fx;
    float y_over_z = (cam.cy - point.y) / cam.fy;
    float z = d / sqrt(1. + pow(x_over_z,2) + pow(y_over_z,2));
    point_3d.x = x_over_z*z;
    point_3d.y = y_over_z*z;
    point_3d.z = z;
    
    return point_3d;

}


int main()
{   


    // Import depth image 
    string image_path = "/home/aditya/Desktop/Personal/Profile_Pic/Profile";
    Mat img = imread(image_path);
    if(img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
    }

    Point2f test_point = Point2f(5,7);
    test_point.x = 4;

    // Set camera properties:
    camera_prop camera;
    camera.fx = fx;
    camera.fy = fy;
    camera.cx = cx;
    camera.cy = cy;

    Matrix3f cam_matrix = intr_matrix(fx, fy, cx, cy);

    Point3f test = cart_cood(camera, test_point, 3);
    
    cout<<cam_matrix(1,2)<<" "<<cam_matrix(1,1);

    return 1;
}