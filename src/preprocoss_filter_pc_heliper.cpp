//Author:   Weixin Ma       weixin.ma@connect.polyu.hk
//cpp file for preprocossing the point cloud data for HeLiPR dataset to fit the requirement of the SPVANS
#include <ros/ros.h>
#include <iostream>
#include <fstream>
#include <sys/types.h>
#include <dirent.h>
#include <string.h>
#include <numeric>
#include <vector>
#include <algorithm>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>


pcl::PointCloud<pcl::PointXYZI> readBinFile_and_filter(const std::string &point_file, std::string save_dir)  //for HeLiPR
{
  std::ofstream out(save_dir, std::ios::binary);

  pcl::PointCloud<pcl::PointXYZI> output;
  std::ifstream input_point;
  input_point.open(point_file, std::ios::in | std::ios::binary);

  int i;
  for (i=0; input_point.good() && input_point.peek()!=EOF; i++) 
  {
    pcl::PointXYZI my_point;
    float intensity;  
    uint32_t t;
    uint16_t reflectivity,ring, ambient;
    input_point.read(reinterpret_cast<char *>(&my_point.x), sizeof(float));
    input_point.read(reinterpret_cast<char *>(&my_point.y), sizeof(float));
    input_point.read(reinterpret_cast<char *>(&my_point.z), sizeof(float));
    input_point.read(reinterpret_cast<char *>(&my_point.intensity), sizeof(float));
    input_point.read(reinterpret_cast<char *>(&t), sizeof(uint32_t));
    input_point.read(reinterpret_cast<char *>(&reflectivity), sizeof(uint16_t));
    input_point.read(reinterpret_cast<char *>(&ring), sizeof(uint16_t));
    input_point.read(reinterpret_cast<char *>(&ambient), sizeof(uint16_t));

    float dis = sqrt(my_point.x * my_point.x + my_point.y * my_point.y);
    if (dis<=150)
    {
      out.write(reinterpret_cast<char*>(&my_point.x), sizeof(float));
      out.write(reinterpret_cast<char*>(&my_point.y), sizeof(float));
      out.write(reinterpret_cast<char*>(&my_point.z), sizeof(float));
      out.write(reinterpret_cast<char*>(&my_point.intensity), sizeof(float));
      out.write(reinterpret_cast<char*>(&t), sizeof(uint32_t));
      out.write(reinterpret_cast<char*>(&reflectivity), sizeof(uint16_t));
      out.write(reinterpret_cast<char*>(&ring), sizeof(uint16_t));
      out.write(reinterpret_cast<char*>(&ambient), sizeof(uint16_t));
    }
    
    output.push_back(my_point);
  }

  input_point.close();
  out.close();

  return output;
}

std::vector<std::string> GetFiles(const char* folderPath) {
    
  DIR* directory;
  struct dirent* file;

  std::vector<std::string> fileNames;
    
  if ((directory = opendir(folderPath)) != nullptr) 
  {
    while ((file = readdir(directory)) != nullptr) 
    {
      if (file->d_type == DT_REG) 
      {  // Check if it is a regular file
        fileNames.push_back(file->d_name);
      }
    }
    closedir(directory);
        
    std::sort(fileNames.begin(), fileNames.end());

  } 
  else 
  {
    std::cerr << "Cannot open directory: " << folderPath << std::endl;

  }
  return fileNames;
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "tripletloc");
  ros::NodeHandle nh("~");

  ros::Publisher  pc_ref_pub_;
  
  pc_ref_pub_  = nh.advertise<sensor_msgs::PointCloud2>("PC_in_ref_gt", 1000);

  //get all the files in the folder
  std::string pc_dirs, save_pc_dir;
  nh.getParam("seq_tobe_processed",pc_dirs);
  nh.getParam("save_path",save_pc_dir);
  
  std::vector<std::string> total_ref_frames = GetFiles(pc_dirs.c_str());
  
  sleep(2);

  for (int i = 0; i < total_ref_frames.size(); ++i)
  {
    //test 
    std::cout<<"**~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"<<std::endl;
    std::cout<<"*Refs: frame-"<<i<<", id: "<<total_ref_frames[i]<<std::endl;


    std::string pc_dir = pc_dirs+ '/' + total_ref_frames[i];
    std::string pc_filtered_dir = save_pc_dir + total_ref_frames[i];

    pcl::PointCloud<pcl::PointXYZI> pc = readBinFile_and_filter(pc_dir,pc_filtered_dir);
 
    sensor_msgs::PointCloud2 pc_ref;
    pcl::toROSMsg(pc, pc_ref);                   
    pc_ref.header.frame_id = "tripletloc";               
    pc_ref.header.stamp=ros::Time::now();       
    pc_ref_pub_.publish(pc_ref);
  }
  
  return 0;
}