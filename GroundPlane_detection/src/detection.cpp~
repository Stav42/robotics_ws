//OpenCV Libraries
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ximgproc.hpp>

#include <string>

// PCL Library
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/common/io.h> // for copyPointCloud
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <iostream>
#include <cmath>
#include <eigen3/Eigen/Core>

typedef pcl::PointXYZ PointT;
typedef pcl::PointXYZRGB PointTRGB;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointCloud<PointTRGB> PointCloud_RGB;

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
const double camera_cx = 40;
const double camera_cy = 30;
const double camera_fx = 363.0;
const double camera_fy = 363;

void  PreProcess(PointCloud_RGB::Ptr cloud_rgb, PointCloud_RGB::Ptr cloud_voxel){
   
    PointCloud_RGB::Ptr cloud_filtered (new PointCloud_RGB);

    pcl::PassThrough<pcl::PointXYZRGB> pass;
    pass.setInputCloud(cloud_rgb);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(0.0,20);
    pass.filter(*cloud_filtered);
    
    pcl::VoxelGrid<pcl::PointXYZRGB> sor;
    sor.setInputCloud (cloud_filtered);
    sor.setLeafSize (0.4, 0.4, 0.4);
    sor.filter (*cloud_voxel);

}  

int main(int argc, char *argv[])
{   
    

    // Import depth image 
    string depth_path = "/home/aditya/Desktop/Team_Humanoid/robotics_ws/GroundPlane_detection/test_images/0062.png";
    string rgb_path = "/home/aditya/Desktop/Team_Humanoid/robotics_ws/GroundPlane_detection/test_images/0062rgb.png";
    


    Mat depth = imread(depth_path, -1);
    Mat rgb = imread(rgb_path);    
    

    //create a null cloud using smart pointers.
    PointCloud::Ptr cloud(new PointCloud);
    PointCloud::Ptr cloud_filtered (new PointCloud);
    PointCloud::Ptr cloud_voxel (new PointCloud);
    PointCloud::Ptr final (new PointCloud);
    PointCloud_RGB::Ptr cloud_rgb(new PointCloud_RGB);
    PointCloud_RGB::Ptr voxel_rgb(new PointCloud_RGB);

    for(int m=0;m<depth.rows;m++){
        for(int n=0;n<depth.cols;n++){

            // Get the value at (m,n) in the depth map. The following is an efficient way to access the pixel
            ushort d = depth.ptr<ushort> (m)[n];

            //d may have no value, skip in that case
            if(d==0) continue;

            PointT p;
            PointTRGB c;

            // Calculate the space coordinate of this point
            p.z = double(d)/ camera_factor;
            p.x = (n - camera_cx)*p.z/camera_fx;
            p.y = (m - camera_cy)*p.z/camera_fy;

            //p.b = rgb.ptr<uchar>(m)[n*3];
            //p.g = rgb.ptr<uchar>(m)[n*3+1];
            //p.r = rgb.ptr<uchar>(m)[n*3+2];

            c.z = double(d)/camera_factor;
            c.x = (n-camera_cx)*c.z/camera_fx;
            c.y = (m-camera_cy)*c.z/camera_fy;

            c.b = rgb.ptr<uchar>(m)[n*3];
            c.g = rgb.ptr<uchar>(m)[n*3+1];
            c.r = rgb.ptr<uchar>(m)[n*3+2];

            cloud_rgb->points.push_back(c);
            cloud->points.push_back(p);
        }

    }

    PreProcess(cloud_rgb, voxel_rgb);

    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(0.0,20);
    pass.filter(*cloud_filtered);
    
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud (cloud_filtered);
    sor.setLeafSize (0.4, 0.4, 0.4);
    sor.filter (*cloud_voxel);

    // Position of inliers
    std::vector<int> inliers;

    // RandomSampleConsensus object and compute appropriated model
    pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr model_p(new pcl::SampleConsensusModelPlane<pcl::PointXYZ> (cloud_voxel));

    pcl::RandomSampleConsensus<pcl::PointXYZ> ransac (model_p);
    ransac.setDistanceThreshold(0.2);
    ransac.computeModel();
    ransac.getInliers(inliers);
    
    // copies all inliers of the model computed to another PointCloud
    pcl::copyPointCloud(*cloud_voxel, inliers, *final);

    cloud->height = 1;
    cloud->width = cloud->points.size();
    cout<< "point cloud size = "<<cloud->points.size()<<endl;
    cout<< "Filtered point cloud size = "<<cloud_filtered->points.size()<<endl;
    cout<< "Voxel point cloud size = "<<cloud_voxel->points.size()<<endl;
    cout<< "RANSAC plane detected. Size = "<<final->points.size()<<endl;
    cout<< "Voxel plane detected. Size = "<<voxel_rgb->points.size()<<endl;

    cloud->is_dense = false;
    pcl::io::savePCDFile("/home/aditya/Desktop/Team_Humanoid/robotics_ws/GroundPlane_detection/results/original.pcd", *cloud);
    pcl::io::savePCDFile("/home/aditya/Desktop/Team_Humanoid/robotics_ws/GroundPlane_detection/results/originalfiltered.pcd", *cloud_filtered);
    pcl::io::savePCDFile("/home/aditya/Desktop/Team_Humanoid/robotics_ws/GroundPlane_detection/results/originalvoxel.pcd", *cloud_voxel);
    pcl::io::savePCDFile("/home/aditya/Desktop/Team_Humanoid/robotics_ws/GroundPlane_detection/results/RANSAC.pcd", *final);
    pcl::io::savePCDFile("/home/aditya/Desktop/Team_Humanoid/robotics_ws/GroundPlane_detection/results/voxel_rgb.pcd", *voxel_rgb);
 


    cloud->points.clear();
    cloud_filtered->clear();
    cout<< "point cloud saved."<<endl;

    return 1;
}
