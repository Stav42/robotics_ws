//OpenCV Libraries
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ximgproc.hpp>

#include <string>

// PCL Library
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>


#include <iostream>
#include <cmath>
#include <eigen3/Eigen/Core>

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloud;


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

const double camera_factor = 700;
const double camera_cx = 256;
const double camera_cy = 212;
const double camera_fx = 363.0;
const double camera_fy = 363;


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




int main(int argc, char *argv[])
{   


    // Import depth image 
    string depth_path = "/home/aditya/Desktop/Team Humanoid/robotics_ws/GroundPlane_detection/test_images/0062.png";
    string rgb_path = "/home/aditya/Desktop/Team Humanoid/robotics_ws/GroundPlane_detection/test_images/0062rgb.png";
    Mat depth = imread(depth_path, -1);
    Mat rgb = imread(rgb_path);    

    // Set camera properties:
    camera_prop camera;
    camera.fx = camera_fx;
    camera.fy = camera_fy;
    camera.cx = camera_cx;
    camera.cy = camera_cy;

    Matrix3f cam_matrix = intr_matrix(camera_fx, camera_fy, camera_cx, camera_cy);

    //Point3f test = cart_cood(camera, test_point, 3);
    
    // Data pre-processing
    // Weight Median Filter
    Mat F;
    depth.convertTo(F, CV_8U);
    depth.convertTo(depth, CV_32F);
    ximgproc::weightedMedianFilter(F,depth,depth, 10);

    //create a null cloud using smart pointers.
    PointCloud::Ptr cloud(new PointCloud);

    for(int m=0;m<depth.rows;m++){
        for(int n=0;n<depth.cols;n++){

            // Get the value at (m,n) in the depth map. The following is an efficient way to access the pixel
            ushort d = depth.ptr<ushort> (m)[n];

            //d may have no value, skip in that case
            if(d==0) continue;

            PointT p;

            // Calculate the space coordinate of this point
            p.z = double(d)/ camera_factor;
            p.x = (n - camera_cx)*p.z/camera_fx;
            p.y = (m - camera_cy)*p.z/camera_fy;

            p.b = rgb.ptr<uchar>(m)[n*3];
            p.g = rgb.ptr<uchar>(m)[n*3+1];
            p.r = rgb.ptr<uchar>(m)[n*3+2];

            cloud->points.push_back(p);
        }

    }


    cloud->height = 1;
    cloud->width = cloud->points.size();
    cout<< "point cloud size = "<<cloud->points.size()<<endl;
    cloud->is_dense = false;
    pcl::io::savePCDFile("/home/aditya/Desktop/Team Humanoid/robotics_ws/GroundPlane_detection/results/0062.pcd", *cloud);
    
    cloud->points.clear();
    cout<< "point cloud saved."<<endl;

    return 1;
}