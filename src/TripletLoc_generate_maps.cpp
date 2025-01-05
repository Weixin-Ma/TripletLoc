//Author:   Weixin Ma       weixin.ma@connect.polyu.hk
//Cpp file for generating the instance-level map and RSN map
#include <ros/ros.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <yaml-cpp/yaml.h>
#include <random>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include <pcl/io/io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

#include <pcl/filters/extract_indices.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/ModelCoefficients.h>

#include <omp.h>
#include <cstdlib>

#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include "tic_toc.h"


std::string save_path_RSN_map_;
std::string save_path_instance_map_;

struct semantic_configs{       
  std::string obj_class;
  int color_b;
  int color_g;
  int color_r;
};

struct configs_helipr{
  float frames_interval_refs;
  std::string cloud_path_ref;
  std::string label_path_ref;
  std::string pose_gt_file_ref;

  //paras for RSN map
  int mini_p_num;
  double grid_size;
  double xy_range;
  double fuse_radius;
  double SACS_thr;

  //paras for instance map
  float voxel_leaf_size_ref;
  float ins_fuse_radius;   

  std::vector<__int16_t> classes_for_graph;      
  std::map<__int16_t,semantic_configs> semantic_name_rgb;
  std::map<__int16_t,float> EuCluster_para;
  std::map<__int16_t,int> minimun_point_in_one_instance;
};

struct instance_centriod
{
  float x;
  float y;
  float z;
};

struct objects_and_pointclouds
{
  std::map<__int16_t, pcl::PointCloud<pcl::PointXYZRGB>> pc;
  std::map<__int16_t,std::map<__int16_t, instance_centriod>> instance_cens;
  std::map<__int16_t, int> instance_numbers;
};

struct GridCell {
  Eigen::Vector3f center;
  Eigen::Vector3f roadNormal; // Normal vector of the road surface for this cell
  bool noraml_ok;
  float dir_std; // Standard deviation of the direction of the loacal road surface, based on the fused normals
};

std::vector<GridCell> valid_grid_cells_total_;


ros::Publisher pc_current_pub_, ins_cen_fused_pub_, pc_total_pub_;
ros::Publisher odom_pub_, grid_normal_pub_, grid_normal_fused_pub_, road_grid_pub_;

configs_helipr configs_h_;

configs_helipr get_config(std::string config_file_dir)
{
  configs_helipr output;

  auto data_cfg = YAML::LoadFile(config_file_dir);    

  output.frames_interval_refs     = data_cfg["ref_related"]["frames_interval"].as<float>();      //frame interval for building maps
  
  //paras for isntance map
  output.voxel_leaf_size_ref      = data_cfg["instance_paras"]["voxel_leaf_size"].as<float>();      
  output.ins_fuse_radius          = data_cfg["instance_paras"]["ins_fuse_radius"].as<float>();      

  //paras for RSN map
  output.mini_p_num               = data_cfg["rsn_paras"]["mini_p_num"].as<int>();
  output.grid_size                = data_cfg["rsn_paras"]["grid_size"].as<double>();
  output.xy_range                 = data_cfg["rsn_paras"]["xy_range"].as<double>();
  output.fuse_radius              = data_cfg["rsn_paras"]["fuse_radius"].as<double>();
  output.SACS_thr                 = data_cfg["rsn_paras"]["sac_threshold"].as<double>();

  auto classes_for_graph          =data_cfg["classes_for_graph"];
  auto color_map                  =data_cfg["color_map"];
  auto class_name                 =data_cfg["labels"];
  auto instance_seg_para          =data_cfg["instance_seg_para"];
  auto minimun_point_one_instance =data_cfg["mini_point_one_instance"];

  std::vector<__int16_t> class_for_graph;
  std::map<__int16_t,std::string> label_name;
  std::map<__int16_t,semantic_configs> semantic_name_rgb;
  std::map<__int16_t,float> EuCluster_para;
  std::map<__int16_t,int> minimun_point_in_one_instance;

  YAML::Node::iterator iter,iter1, iter2, iter3;
  iter  =classes_for_graph.begin();
  iter1 =class_name.begin();
  iter2 =instance_seg_para.begin();
  iter3 =minimun_point_one_instance.begin();

  //class names
  for (iter1;iter1!=class_name.end();iter1++) {
    label_name[iter1->first.as<__int16_t>()] = iter1->second.as<std::string>();
  }

  //classes used for TripletLoc
  for (iter;iter!=classes_for_graph.end();iter++) {
    if(iter->second.as<bool>())
    {
      class_for_graph.push_back(iter->first.as<__int16_t>());
    }
  }
  output.classes_for_graph = class_for_graph;

  //class name and rgb
  YAML::Node::iterator it;
  for (it = color_map.begin(); it != color_map.end(); ++it)
  {
    semantic_configs single_semnatic;
    single_semnatic.obj_class = label_name[it->first.as<__int16_t>()];
    single_semnatic.color_b = it->second[0].as<int>();
    single_semnatic.color_g = it->second[1].as<int>();
    single_semnatic.color_r = it->second[2].as<int>();
    semantic_name_rgb[it->first.as<__int16_t>()] = single_semnatic;
  }
  output.semantic_name_rgb = semantic_name_rgb;

  //paras for clustering
  for (iter2;iter2!=instance_seg_para.end();iter2++) {
    EuCluster_para[iter2->first.as<__int16_t>()] = iter2->second.as<float>();
  }
  output.EuCluster_para = EuCluster_para;

  //paras for clustering
  for (iter3; iter3!=minimun_point_one_instance.end(); iter3++)
  {
    minimun_point_in_one_instance[iter3->first.as<__int16_t>()] = iter3->second.as<int>();
  }
  output.minimun_point_in_one_instance = minimun_point_in_one_instance;

  return output;
}



std::vector<std::pair<std::string, Eigen::Matrix4d>> get_pose_gt(std::string pose_gt_file) 
{
  std::vector<std::pair<std::string, Eigen::Matrix4d>> output;

  std::vector<std::vector<float>> trah_original;

  std::ifstream in(pose_gt_file);
  std::string line;
  std::vector<std::vector <std::string>> traj;
  if(in) // if the fiel exist
  {

    while (getline (in, line))
    {
      std::istringstream ss(line);
      std::string word;
      std::vector<std::string> single_pose;
      while ( ss >> word)
      {
        single_pose.push_back(word);
      }
      traj.push_back(single_pose);
    }
  }

  else //if the fiel not exist
  {
    std::cout <<"\033[33mCould not load the ground truth traj file!\033[0m"  << std::endl;
  }

  for (int i=0; i<traj.size();++i)
  {
    std::vector<float> pose;
    for (size_t j = 1; j < traj[i].size(); ++j)
      {
        std::stringstream s;
        float f;
        s<<std::fixed<<std::setprecision(9)<<traj[i][j];
        s>>f;
        pose.push_back(f);
      }
      trah_original.push_back(pose);
  }
  in.close();


  for (size_t i = 0; i < trah_original.size(); ++i)
  {
    Eigen::Quaterniond pose_Q(trah_original[i][6],trah_original[i][3],trah_original[i][4],trah_original[i][5]);
    Eigen::Matrix3d pose_rot = pose_Q.matrix();

    Eigen::Matrix4d Tcurrent;
    Tcurrent << pose_rot(0,0), pose_rot(0,1), pose_rot(0,2) , trah_original[i][0] ,
                pose_rot(1,0), pose_rot(1,1), pose_rot(1,2) , trah_original[i][1] ,
                pose_rot(2,0), pose_rot(2,1), pose_rot(2,2) , trah_original[i][2] ,
                0.0          , 0.0          , 0.0           , 1.0                 ;

    std::pair<std::string, Eigen::Matrix4d> single_one;

    single_one.first = traj[i][0];
    single_one.second = Tcurrent;

    output.push_back(single_one);
  }

  return output;
}


std::map<__int16_t, pcl::PointCloud<pcl::PointXYZRGB>> get_pointcloud_with_sem(std::string bin_dir, std::string label_file)
{
  std::map<__int16_t, pcl::PointCloud<pcl::PointXYZRGB>> output;

  pcl::PointCloud<pcl::PointXYZRGB> all_points;
  std::ifstream input_point, input_label;
  input_point.open(bin_dir, std::ios::in | std::ios::binary);
  input_label.open(label_file, std::ios::in | std::ios::binary);

  int i;
  for (i=0; input_point.good() && input_label.good() && input_label.peek()!=EOF &&input_point.peek()!=EOF; i++)
  {
    pcl::PointXYZRGB my_point;
    float intensity;
    uint32_t t;
    uint16_t reflectivity,ring, ambient;
    input_point.read(reinterpret_cast<char *>(&my_point.x), sizeof(float));
    input_point.read(reinterpret_cast<char *>(&my_point.y), sizeof(float));
    input_point.read(reinterpret_cast<char *>(&my_point.z), sizeof(float));
    input_point.read(reinterpret_cast<char *>(&intensity), sizeof(float));
    input_point.read(reinterpret_cast<char *>(&t), sizeof(uint32_t));
    input_point.read(reinterpret_cast<char *>(&reflectivity), sizeof(uint16_t));
    input_point.read(reinterpret_cast<char *>(&ring), sizeof(uint16_t));
    input_point.read(reinterpret_cast<char *>(&ambient), sizeof(uint16_t));

    //uint32_t label;
    uint16_t label;
    uint16_t id;

    input_label.read((char *) &label,sizeof(__int16_t));                        
    input_label.read((char *) &id,sizeof(__int16_t));

    // std::cout<<" "<<label;

    my_point.r = configs_h_.semantic_name_rgb[label].color_r;
    my_point.g = configs_h_.semantic_name_rgb[label].color_g;
    my_point.b = configs_h_.semantic_name_rgb[label].color_b;

    output[label].push_back(my_point);
    all_points.push_back(my_point);
  }

  output[-1] = all_points;

  input_point.close();
  input_label.close();

  return output;
}

void computeNormalForGridCell(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud, std::vector<int>& indices, Eigen::Vector3f& roadNormal,Eigen::Vector3f& grid_cen, Eigen::Matrix4d current_pose, double SACS_thr) {
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cellCloud(new pcl::PointCloud<pcl::PointXYZRGB>());
  float z_sum = 0.0; //actually not used, only for better visualizatin of the normal grid cell
  for (int idx : indices) {
    cellCloud->points.push_back(cloud->points[idx]);
    z_sum = z_sum + cloud->points[idx].z;
  }
  float z_ave = z_sum / indices.size();
  Eigen::Vector4d pose(grid_cen[0],grid_cen[1], z_ave,1.0);
  Eigen::Vector4d pose_new = current_pose * pose;
  grid_cen[0] = pose_new.x();
  grid_cen[1] = pose_new.y();
  grid_cen[2] = pose_new.z();

  pcl::SACSegmentation<pcl::PointXYZRGB> seg;
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setDistanceThreshold(SACS_thr); // Adjust threshold as needed

  seg.setInputCloud(cellCloud);

  pcl::ModelCoefficients coefficients;
  pcl::PointIndices inliers;
  seg.segment(inliers, coefficients);

  if (coefficients.values.size() == 4) {
    roadNormal = Eigen::Vector3f(coefficients.values[0], coefficients.values[1], coefficients.values[2]).normalized();
    //std::cout<<"Normal: "<<roadNormal<<std::endl;
  } else {
    //std::cout<<" + ";
    roadNormal = Eigen::Vector3f(0, 0, 1); // Default normal if plane fitting fails
  }
} 


std::vector<GridCell> computeNormalsForGridCells(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud, double xy_range, float grid_size, int mini_p_num,Eigen::Matrix4d current_pose, double SACS_thr ) {
  const float minLimit = -xy_range;
  const float maxLimit = xy_range;
  const float range = maxLimit - minLimit;
    
  int numCols = static_cast<int>(std::ceil(range / grid_size));
  int numRows = static_cast<int>(std::ceil(range / grid_size));
  int numCells = numRows * numCols;

  std::vector<GridCell> gridCells(numCells);

  // Initialize grid cell centers
  #pragma omp parallel for
  for (int row = 0; row < numRows; ++row) {
    for (int col = 0; col < numCols; ++col) {
      int index = row * numCols + col;
      float centerX = minLimit + (col + 0.5f) * grid_size;
      float centerY = minLimit + (row + 0.5f) * grid_size;
      gridCells[index].center = Eigen::Vector3f(centerX,centerY,0.0);
    }
  }

  // Assign points to the respective grid cells
  std::vector<std::vector<int>> pointIndices_all(numCells);
  for (int i = 0; i < cloud->points.size(); ++i) {
    const auto& point = cloud->points[i];
    if (point.x >= minLimit && point.x <= maxLimit && point.y >= minLimit && point.y <= maxLimit) {
      int col = static_cast<int>((point.x - minLimit) / grid_size);
      int row = static_cast<int>((point.y - minLimit) / grid_size);
      int index = row * numCols + col;

      if (index >= 0 && index < numCells) {
        pointIndices_all[index].push_back(i);
      }
    }
  }

  // Compute normals for each grid cell
  #pragma omp parallel for
  for (int index = 0; index < numCells; ++index) {
    if (pointIndices_all[index].size()>=mini_p_num) {
      gridCells[index].noraml_ok = true;
      computeNormalForGridCell(cloud, pointIndices_all[index], gridCells[index].roadNormal,gridCells[index].center,current_pose, SACS_thr);
    }
    else
    {
      gridCells[index].noraml_ok = false;
    }
  }

  return gridCells;
}


visualization_msgs::MarkerArray visNormals( const std::vector<GridCell>& gridCells, std::string ns, Eigen::Vector3f color, float z_offest) {
  int normal_id = 0;
  visualization_msgs::MarkerArray markerArray;
  for (size_t i = 0; i < gridCells.size(); ++i) {
    const GridCell& cell = gridCells[i];
    
    if (!cell.noraml_ok) {
      continue;
    }

    visualization_msgs::Marker marker;
    marker.header.frame_id = "tripletloc"; // Set to your appropriate frame
    marker.header.stamp = ros::Time::now();
    marker.ns = ns; //"road_normals";
    marker.id = normal_id; // Unique ID for each marker
    marker.type = visualization_msgs::Marker::ARROW;
    marker.action = visualization_msgs::Marker::ADD;

    // Set the start point (center of the grid cell)
    geometry_msgs::Point start;
    start.x = cell.center.x();
    start.y = cell.center.y();
    start.z = cell.center.z() + z_offest; // Set Z as needed
    marker.points.push_back(start);

    // Set the end point (in the direction of the normal)
    geometry_msgs::Point end;
    end.x = start.x + cell.roadNormal.x()*5;
    end.y = start.y + cell.roadNormal.y()*5;
    end.z = start.z + cell.roadNormal.z()*5;
    marker.points.push_back(end);

    // Set color (you can adjust this as needed)
    marker.color.r = color[0];
    marker.color.g = color[1];
    marker.color.b = color[2];
    marker.color.a = 1.0f;

    // Set scale (size of the arrow)
    marker.scale.x = 1.0; // Shaft diameter
    marker.scale.y = 1.5; // Head diameter

    markerArray.markers.push_back(marker);

    normal_id += 1;
  }

  return markerArray;
}

std::vector<GridCell> normals_fusion(pcl::PointCloud<pcl::PointXYZ> grid_cens_total, float fuse_radius)
{
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);   
  tree->setInputCloud(grid_cens_total.makeShared());                                     
  std::vector<pcl::PointIndices> cluster_indices;                                        
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;                             

  float cluster_tolerance = fuse_radius;
  ec.setClusterTolerance (cluster_tolerance);                                            
  ec.setMinClusterSize (1);
  ec.setMaxClusterSize ( grid_cens_total.size() );
  ec.setSearchMethod (tree);
  ec.setInputCloud ( grid_cens_total.makeShared());
  ec.extract (cluster_indices);

  TicToc cluster_instance_t;

  //cluster result saving
  std::vector<pcl::PointIndices>::const_iterator it;
  it = cluster_indices.begin();

  __int16_t new_id=0;
  int instance_number_fused = 0;

  std::vector<GridCell> fused_grid_cells;

  for (it; it!=cluster_indices.end();++it)
  {
    Eigen::Vector3f roadNormal_sum;
    roadNormal_sum.setZero();

    Eigen::Vector3f center_sum;
    center_sum.setZero();

    for (size_t k = 0; k < it->indices.size(); ++k)
    {
      roadNormal_sum = roadNormal_sum + valid_grid_cells_total_[it->indices[k]].roadNormal;
      center_sum = center_sum + valid_grid_cells_total_[it->indices[k]].center;
    }
    
    Eigen::Vector3f roadNormal_ave = roadNormal_sum / it->indices.size();     //average normal vector
    //normaliz the road normal vector
    roadNormal_ave.normalize();

    Eigen::Vector3f center_ave = center_sum / it->indices.size();             //average center


    //calculate the standard deviation of the direction of the loacal road surface, based on the fused normals
    float angle_diff_sum = 0.0;
    for (size_t k = 0; k < it->indices.size(); ++k)
    {
      //calculate the angle between the roadNormal_ave and the the normals belongs to it->indices
      Eigen::Vector3f normal_b = valid_grid_cells_total_[it->indices[k]].roadNormal;
      float dot_product = roadNormal_ave.dot(normal_b);

      // Calculate the magnitudes (norms)
      float norm_a = roadNormal_ave.norm();
      float norm_b = normal_b.norm();

      // Calculate the cosine of the angle
      float cos_theta = dot_product / (norm_a * norm_b);
      cos_theta = std::max(-1.0f, std::min(1.0f, cos_theta)); // Clamp to [-1, 1]

      // Calculate the angle in radians
      float angle_rad = std::acos(cos_theta);

      // Convert the angle to degrees (optional)
      float angle_deg = angle_rad * 180.0f / M_PI;

      angle_diff_sum = angle_diff_sum + angle_deg * angle_deg;
    } 
    float angle_diff_std = sqrt(angle_diff_sum/it->indices.size() );

    GridCell fused_cell;
    fused_cell.center     = center_ave;
    fused_cell.roadNormal = roadNormal_ave;    //average normal vector
    fused_cell.dir_std    = angle_diff_std;    //unit: degree
    fused_cell.noraml_ok  = true;
    fused_grid_cells.push_back(fused_cell);
  }

  return fused_grid_cells;
}


visualization_msgs::MarkerArray visRoadGrid(const std::vector<GridCell>& gridCells, std::string ns, double grid_size, Eigen::Matrix4d current_pose)
{
  int grid_id = 0; // only show grid cells for the current frame
  
  //generate random color for road grid
  std::random_device rd; // Seed for the random number engine
  std::mt19937 gen(rd()); // Mersenne Twister random number engine
  std::uniform_real_distribution<> dis(0.0, 1.0); // Uniform distribution between 0.0 and 1.0

  visualization_msgs::MarkerArray markerArray;
  for (size_t i = 0; i < gridCells.size(); ++i) {
    const GridCell& cell = gridCells[i];
    
    if (!cell.noraml_ok) {   //only show the valid grid cells
      continue;
    }

    visualization_msgs::Marker marker;
    marker.header.frame_id = "tripletloc"; // Set to your appropriate frame
    marker.header.stamp = ros::Time::now();
    marker.ns = ns; //"road_normals";
    marker.id = grid_id; // Unique ID for each marker
    marker.type = visualization_msgs::Marker::CUBE;
    marker.action = visualization_msgs::Marker::ADD;

    // Set the position of the cube center (x, y, z)
    marker.pose.position.x = cell.center.x();
    marker.pose.position.y = cell.center.y();
    marker.pose.position.z = cell.center.z();

    // value the rotation matrix
    Eigen::Matrix3d R = current_pose.block<3,3>(0,0);
    Eigen::Quaterniond q(R);

    // Set the orientation of the cube
    marker.pose.orientation.x = q.x();
    marker.pose.orientation.y = q.y();
    marker.pose.orientation.z = q.z();
    marker.pose.orientation.w = q.w();

    // Set the scale of the cube (x, y, z)
    marker.scale.x = grid_size;
    marker.scale.y = grid_size;
    marker.scale.z = 0.5;

    // Generate random normalized RGB values
    double red = dis(gen);
    double green = dis(gen);
    double blue = dis(gen);

    marker.color.r = red;
    marker.color.g = green;
    marker.color.b = blue;
    marker.color.a = 0.8f;

    markerArray.markers.push_back(marker);

    grid_id += 1;
  }

  return markerArray;  
}


void generate_RSN_map() {
  double grid_size = configs_h_.grid_size;
  int mini_p_num = configs_h_.mini_p_num;
  double xy_range = configs_h_.xy_range;
  double fuse_radius = configs_h_.fuse_radius;
  double SACS_thr = configs_h_.SACS_thr;

  sleep(1);

  nav_msgs::Odometry pose_stamped;

  Eigen::Matrix4d last_pose;
  last_pose << 1, 0, 0, 0,
               0, 1, 0, 0,
               0, 0, 1, 0,
               0, 0, 0, 1;

  int frame_number_for_global_map = 0;
  int frame_count = 0;

  std::vector<std::pair<std::string, Eigen::Matrix4d>> stamp_with_pose = get_pose_gt(configs_h_.pose_gt_file_ref);

  pcl::PointCloud<pcl::PointXYZRGB> concatenate_pc;

  //for grid normals fusion
  pcl::PointCloud<pcl::PointXYZ> grid_cens_total_;

  for (size_t i = 0; i < stamp_with_pose.size(); i++)
  {
    std::cout<<"Current frame ID-"<<i;
    Eigen::Matrix4d current_pose = stamp_with_pose[i].second;

    float d_x = current_pose(0,3) - last_pose(0,3);
    float d_y = current_pose(1,3) - last_pose(1,3);
    float d_z = current_pose(2,3) - last_pose(2,3);
    float dis = sqrt( pow(d_x,2) + pow(d_y,2) + pow(d_z,2) );

    if ( (dis>= configs_h_.frames_interval_refs) || (i==0))
    {
      std::cout<<" is used for building RSN map!"<<std::endl;
      std::string bin_dir = configs_h_.cloud_path_ref+ "/"+stamp_with_pose[i].first +".bin";
      std::string label_dir = configs_h_.label_path_ref+ "/"+stamp_with_pose[i].first +".label";
      std::cout<<bin_dir<<std::endl;
      std::cout<<label_dir<<std::endl;

      std::map<__int16_t, pcl::PointCloud<pcl::PointXYZRGB>> pc_rgb =get_pointcloud_with_sem(bin_dir, label_dir);

      pcl::PointCloud<pcl::PointXYZRGB>::Ptr road_pc = pc_rgb[8].makeShared();

      //calculate the road normal vector and visualize the road normal vector
      std::vector<GridCell> road_grids = computeNormalsForGridCells(road_pc,xy_range, grid_size, mini_p_num,current_pose,SACS_thr);
      
      //store all vaild grid cells for further fusion
      for (int grid_i = 0; grid_i < road_grids.size(); ++grid_i)
      {
        if (road_grids[grid_i].noraml_ok)
        {
          valid_grid_cells_total_.push_back(road_grids[grid_i]);
          pcl::PointXYZ grid_cen;
          grid_cen.x = road_grids[grid_i].center.x();
          grid_cen.y = road_grids[grid_i].center.y();
          grid_cen.z = 0.0;
          grid_cens_total_.points.push_back(grid_cen);
        }        
      }
      
      //delete the grid normals and road grid in last frame
      visualization_msgs::MarkerArray grid_normals_delte;
      visualization_msgs::Marker marker_delete;
      marker_delete.id = 0;
      marker_delete.ns = "road_normals";
      marker_delete.action = visualization_msgs::Marker::DELETEALL;
      grid_normals_delte.markers.push_back(marker_delete);
      grid_normal_pub_.publish(grid_normals_delte);

      visualization_msgs::MarkerArray road_grid_delte;
      visualization_msgs::Marker marker_delete_road;
      marker_delete_road.id = 0;
      marker_delete_road.ns = "road_grid";
      marker_delete_road.action = visualization_msgs::Marker::DELETEALL;
      road_grid_delte.markers.push_back(marker_delete_road);
      road_grid_pub_.publish(road_grid_delte);

      //visualize the grid normals and road grid in current frame
      std::string ns_grid_normal = "road_normals";
      Eigen::Vector3f color(0.0, 0.0, 1.0);
      visualization_msgs::MarkerArray grid_normals = visNormals(road_grids,ns_grid_normal,color,0.0);
      grid_normal_pub_.publish(grid_normals);

      std::string ns_grid = "road_grid";
      visualization_msgs::MarkerArray road_grid = visRoadGrid(road_grids, ns_grid, grid_size,current_pose);
      road_grid_pub_.publish(road_grid);


      pcl::PointCloud<pcl::PointXYZRGB> transformed_pc;
      pcl::transformPointCloud(*road_pc, transformed_pc, current_pose);
      sensor_msgs::PointCloud2 pc_ref;
      pcl::toROSMsg(transformed_pc, pc_ref);           
      pc_ref.header.frame_id = "tripletloc";               
      pc_ref.header.stamp=ros::Time::now();            
      pc_current_pub_.publish(pc_ref);

      pcl::PointCloud<pcl::PointXYZRGB> filtered_pc;
      pcl::VoxelGrid<pcl::PointXYZRGB> sor;
      sor.setInputCloud(transformed_pc.makeShared());                           
      sor.setLeafSize(1.0, 1.0, 1.0);               
      sor.filter(filtered_pc);

      concatenate_pc = concatenate_pc + filtered_pc;

      sensor_msgs::PointCloud2 pc_total_;
      pcl::toROSMsg(concatenate_pc, pc_total_);           
      pc_total_.header.frame_id = "tripletloc";               
      pc_total_.header.stamp=ros::Time::now();            
      pc_total_pub_.publish(pc_total_);


      pose_stamped.header.stamp = ros::Time::now();
      pose_stamped.header.frame_id = "tripletloc";
      pose_stamped.pose.pose.position.x = current_pose(0,3);
      pose_stamped.pose.pose.position.y = current_pose(1,3);
      pose_stamped.pose.pose.position.z = current_pose(2,3);

      Eigen::Quaterniond pose_Q(current_pose.block<3,3>(0,0));
      pose_stamped.pose.pose.orientation.x = pose_Q.x();
      pose_stamped.pose.pose.orientation.y = pose_Q.y();
      pose_stamped.pose.pose.orientation.z = pose_Q.z();
      pose_stamped.pose.pose.orientation.w = pose_Q.w();

      odom_pub_.publish(pose_stamped);

      last_pose = current_pose;

      frame_number_for_global_map = frame_number_for_global_map + 1;
    }

    else
    {
      std::cout<<" isn't used for building RSN map!"<<std::endl;
    }

    frame_count = frame_count +1;

  }

  //perform the fusion of grid normals
  std::vector<GridCell> fused_grid_cells = normals_fusion(grid_cens_total_,fuse_radius);
  std::string ns_grid_normal_fused = "road_normals_fused";
  Eigen::Vector3f color(0.0, 1.0, 0.0);
  visualization_msgs::MarkerArray grid_normals_fused = visNormals(fused_grid_cells,ns_grid_normal_fused,color,1.0);
  grid_normal_fused_pub_.publish(grid_normals_fused);

  //store the fused grid normals
  std::ofstream road_normal_out_;
  road_normal_out_.open(save_path_RSN_map_+ "road_normal.txt");

  for (int i = 0; i < fused_grid_cells.size(); ++i)
  {
    GridCell tmp = fused_grid_cells[i];
    road_normal_out_ <<std::fixed<<std::setprecision(9)<<tmp.center.x()<<" "<<tmp.center.y()<<" "
                     <<tmp.roadNormal.x()<<" "<<tmp.roadNormal.y()<<" "<<tmp.roadNormal.z()<<" "<<tmp.dir_std<<std::endl;
  }
  road_normal_out_.close();  

  std::cout<<"Total frame used for building RSN map: "<<frame_number_for_global_map<<" out of "<<frame_count<<std::endl;
  std::cout<<"Total vaild grid cells before fusion: "<<valid_grid_cells_total_.size()<<std::endl;
  std::cout<<"Total vaild grid cells after fusion: "<<fused_grid_cells.size()<<std::endl;
}


void generate_instance_map() { 
  objects_and_pointclouds output;

  std::vector<__int16_t> used_classes = configs_h_.classes_for_graph;

  Eigen::Matrix4d last_pose;
  last_pose << 1, 0, 0, 0,
               0, 1, 0, 0,
               0, 0, 1, 0,
               0, 0, 0, 1;

  int frame_number_for_global_map = 0;
  int frame_count = 0;

  std::vector<std::pair<std::string, Eigen::Matrix4d>> stamp_with_pose = get_pose_gt(configs_h_.pose_gt_file_ref);

  pcl::PointCloud<pcl::PointXYZ> instances_xyz_;
  std::vector<int> each_instance_point_num_;
  std::vector<__int16_t> each_instance_label_;
  std::map<__int16_t, std::vector<int>> id_in_tree_each_class_;

  int total_instance_number = 0;

  for (size_t i = 0; i < stamp_with_pose.size(); i++)
  {
    std::cout<<"Current frame ID-"<<i;
    Eigen::Matrix4d current_pose = stamp_with_pose[i].second;

    float d_x = current_pose(0,3) - last_pose(0,3);
    float d_y = current_pose(1,3) - last_pose(1,3);
    float d_z = current_pose(2,3) - last_pose(2,3);
    float dis = sqrt( pow(d_x,2) + pow(d_y,2) + pow(d_z,2) );


    if ( (dis>= configs_h_.frames_interval_refs) || (i==0))
    {
      std::cout<<" is used for building instance map!"<<std::endl;
      std::string bin_dir = configs_h_.cloud_path_ref+ "/"+stamp_with_pose[i].first +".bin";
      std::string label_dir = configs_h_.label_path_ref+ "/"+stamp_with_pose[i].first +".label";

      std::map<__int16_t, pcl::PointCloud<pcl::PointXYZRGB>> pc_rgb =get_pointcloud_with_sem(bin_dir, label_dir);

      pcl::PointCloud<pcl::PointXYZRGB> transformed_pc;
      pcl::transformPointCloud(pc_rgb[-1], transformed_pc, current_pose);
      sensor_msgs::PointCloud2 pc_ref;
      pcl::toROSMsg(transformed_pc, pc_ref);           
      pc_ref.header.frame_id = "tripletloc";               
      pc_ref.header.stamp=ros::Time::now();            
      pc_current_pub_.publish(pc_ref);

      std::map<__int16_t, pcl::PointCloud<pcl::PointXYZRGB>>::iterator iter;
      iter = pc_rgb.begin();

      float leaf_size = configs_h_.voxel_leaf_size_ref;
      for (iter; iter != pc_rgb.end(); ++iter)
      {
        if (std::find(used_classes.begin(),used_classes.end(), iter->first) != used_classes.end())
        {
          pcl::PointCloud<pcl::PointXYZRGB> original_pc = iter->second;
          pcl::PointCloud<pcl::PointXYZRGB> filtered_pc;

          pcl::VoxelGrid<pcl::PointXYZRGB> sor;
          sor.setInputCloud(original_pc.makeShared());                            
          sor.setLeafSize(leaf_size, leaf_size, leaf_size);               
          sor.filter(filtered_pc);

          pcl::PointCloud<pcl::PointXYZRGB> transformed_pc;
          pcl::transformPointCloud(filtered_pc, transformed_pc, current_pose);

          //perform clustering
          pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);   
          tree->setInputCloud(transformed_pc.makeShared());                                               
          std::vector<pcl::PointIndices> cluster_indices;                                                
          pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;                                         

          float cluster_tolerance = configs_h_.EuCluster_para[ iter->first ];
          ec.setClusterTolerance (cluster_tolerance);                                               
          ec.setMinClusterSize (20);
          ec.setMaxClusterSize ( transformed_pc.size() );
          ec.setSearchMethod (tree);
          ec.setInputCloud ( transformed_pc.makeShared());
          ec.extract (cluster_indices);


          //cluster result saving
          std::vector<pcl::PointIndices>::const_iterator it;
          it = cluster_indices.begin();

          __int16_t new_id=0;
          int one_class_instance_number = 0;
          std::vector<int> one_class_ids_in_tree;

          for (it; it!=cluster_indices.end();++it)
          {
            pcl::PointCloud<pcl::PointXYZRGB> cloud_cluster;
            float sum_x = 0.0;
            float sum_y = 0.0;
            float sum_z = 0.0;

            for (size_t k = 0; k < it->indices.size(); ++k)
            {
              sum_x = sum_x + transformed_pc[it->indices[k]].x;
              sum_y = sum_y + transformed_pc[it->indices[k]].y;
              sum_z = sum_z + transformed_pc[it->indices[k]].z;
              cloud_cluster.push_back(transformed_pc[it->indices[k]]);
            }

            int minimun_points_amount = configs_h_.minimun_point_in_one_instance[iter->first];
            if(cloud_cluster.points.size() >= minimun_points_amount)                   
            {
              instance_centriod centriod;
              pcl::PointXYZ centriod_one;
              centriod.x = sum_x/cloud_cluster.points.size();
              centriod.y = sum_y/cloud_cluster.points.size();
              centriod.z = sum_z/cloud_cluster.points.size();

              centriod_one.x = centriod.x;
              centriod_one.y = centriod.y;
              centriod_one.z = centriod.z;

              instances_xyz_.push_back(centriod_one);

              each_instance_point_num_.push_back(cloud_cluster.points.size());
              each_instance_label_.push_back(iter->first);
              one_class_ids_in_tree.push_back(total_instance_number);

              one_class_instance_number = one_class_instance_number + 1;
              total_instance_number = total_instance_number + 1;
              new_id++;
            }
          }
          std::cout<<"Instance number for class-"<<configs_h_.semantic_name_rgb[iter->first].obj_class<<" : "<<one_class_instance_number<<std::endl;
          id_in_tree_each_class_[iter->first] = one_class_ids_in_tree;
        }

      }

      last_pose = current_pose;

      frame_number_for_global_map = frame_number_for_global_map + 1;
    }

    else
    {
      std::cout<<" isn't used for building instance map!"<<std::endl;
    }

    frame_count = frame_count +1;
  }
  std::cout<<"Total frame used for building instance map: "<<frame_number_for_global_map<<", total instance number= "<<total_instance_number<<std::endl;


  //instance funsion
  std::map<__int16_t, std::vector<instance_centriod>> fused_ins_tmp;

  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);   
  tree->setInputCloud(instances_xyz_.makeShared());                                               
  std::vector<pcl::PointIndices> cluster_indices;                                                
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;                                          

  float cluster_tolerance = configs_h_.ins_fuse_radius;
  ec.setClusterTolerance (cluster_tolerance);                                                      
  ec.setMinClusterSize (1);
  ec.setMaxClusterSize ( instances_xyz_.size() );
  ec.setSearchMethod (tree);
  ec.setInputCloud ( instances_xyz_.makeShared());
  ec.extract (cluster_indices);

  TicToc cluster_instance_t;
  float leaf_size = configs_h_.voxel_leaf_size_ref;

  //cluster result saving
  std::vector<pcl::PointIndices>::const_iterator it;
  it = cluster_indices.begin();

  __int16_t new_id=0;
  int instance_number_fused = 0;

  for (it; it!=cluster_indices.end();++it)
  {
    pcl::PointCloud<pcl::PointXYZ> cloud_cluster;
    float sum_x = 0.0;
    float sum_y = 0.0;
    float sum_z = 0.0;

    int max_point_num = -10000000;
    int max_pc_num_id = -1;
    for (size_t k = 0; k < it->indices.size(); ++k)
    {
      int ins_index_in_tree = it->indices[k];
      sum_x = sum_x + instances_xyz_[ins_index_in_tree].x;
      sum_y = sum_y + instances_xyz_[ins_index_in_tree].y;
      sum_z = sum_z + instances_xyz_[ins_index_in_tree].z;
      cloud_cluster.push_back(instances_xyz_[ins_index_in_tree]);

      if (each_instance_point_num_[ ins_index_in_tree ] > max_point_num)
      {
        max_point_num = each_instance_point_num_[ ins_index_in_tree ];
        max_pc_num_id = ins_index_in_tree;
      }
    }

    instance_centriod centriod;
    centriod.x = sum_x/cloud_cluster.points.size();
    centriod.y = sum_y/cloud_cluster.points.size();
    centriod.z = sum_z/cloud_cluster.points.size();


    //decide the label of the fused instance
    __int16_t label_current_ins = each_instance_label_[max_pc_num_id];
    fused_ins_tmp[label_current_ins].push_back(centriod);

    instance_number_fused = instance_number_fused +1;
  }

  //set id for each fused instance
  std::map<__int16_t, std::map<__int16_t,instance_centriod>> fused_ins;
  std::map<__int16_t, std::vector<instance_centriod>>::iterator iter_fused;
  iter_fused = fused_ins_tmp.begin();

  for (iter_fused; iter_fused != fused_ins_tmp.end(); ++iter_fused)
  {
    std::cout<<"Fused: Instance number for class-"<<configs_h_.semantic_name_rgb[iter_fused->first].obj_class<<" : "<<iter_fused->second.size()<<std::endl;
    std::ofstream ins_info_out;
    ins_info_out.open(save_path_instance_map_+ std::to_string(iter_fused->first) + ".txt");

    for (int i = 0; i < iter_fused->second.size(); ++i)
    {
      fused_ins[iter_fused->first][i] = iter_fused->second[i];
      ins_info_out<<i<<" ";
      ins_info_out <<std::fixed<<std::setprecision(9)<<iter_fused->second[i].x<<" "<<iter_fused->second[i].y<<" "<<iter_fused->second[i].z<<std::endl;
    }
    ins_info_out.close();
  }

  //visualization
  visualization_msgs::MarkerArray ince_show;
  std::map<__int16_t,std::map<__int16_t,instance_centriod>>::iterator iter1111;
  iter1111 = fused_ins.begin();

  int id_index = 0;

  for (iter1111; iter1111 != fused_ins.end(); iter1111++)
  {
    std::map<__int16_t,instance_centriod>::iterator iter222;
    iter222 = iter1111->second.begin();

    for (iter222; iter222 != iter1111->second.end(); ++iter222)
    {
      visualization_msgs::Marker marker;
      marker.header.frame_id = "tripletloc";
      marker.header.stamp    =ros::Time::now();
      marker.ns = "ins_cen_fused";
      marker.type = visualization_msgs::Marker::SPHERE;

      //set marker action
      marker.action   = visualization_msgs::Marker::ADD;
      marker.lifetime = ros::Duration();//(sec,nsec),0 forever

      //~~~~~~~~~~~~~~~~instance center
      marker.id=  id_index;               

      //set marker position
      marker.pose.position.x = iter222->second.x;
      marker.pose.position.y = iter222->second.y;
      marker.pose.position.z = iter222->second.z;

      marker.pose.orientation.x = 0.0;
      marker.pose.orientation.y = 0.0;
      marker.pose.orientation.z = 0.0;
      marker.pose.orientation.w = 1.0;

      //set marker scale
      marker.scale.x = 2.0;
      marker.scale.y = 2.0;
      marker.scale.z = 2.0;

      //decide the color of the marker
      marker.color.a = 1.0; // Don't forget to set the alpha!
      marker.color.r = (float)configs_h_.semantic_name_rgb[iter1111->first].color_r/255;
      marker.color.g = (float)configs_h_.semantic_name_rgb[iter1111->first].color_g/255;
      marker.color.b = (float)configs_h_.semantic_name_rgb[iter1111->first].color_b/255;

      ince_show.markers.push_back(marker);

      id_index = id_index + 1;
    }

  }

  ins_cen_fused_pub_.publish(ince_show);

}



int main(int argc, char **argv) { 
  ros::init(argc, argv, "tripletloc");
  ros::NodeHandle nh("~");

  int rsn_or_instance = 0;
  nh.getParam("rsn_or_instance",rsn_or_instance);

  nh.getParam("save_path_RSN_map",save_path_RSN_map_);
  nh.getParam("save_path_instance_map",save_path_instance_map_);

  std::string config_path;
  nh.getParam("config_file",config_path);
  configs_h_ = get_config(config_path);

  std::string seq_for_maps_dir;
  nh.getParam("seq_for_maps_dir",seq_for_maps_dir);
  configs_h_.cloud_path_ref     = seq_for_maps_dir + "LiDAR/Ouster_filterd";          
  configs_h_.label_path_ref     = seq_for_maps_dir + "labels";          
  configs_h_.pose_gt_file_ref   = seq_for_maps_dir + "LiDAR_GT/Ouster_gt.txt";   

  pc_current_pub_  = nh.advertise<sensor_msgs::PointCloud2>("current_pc", 1000);
  pc_total_pub_     = nh.advertise<sensor_msgs::PointCloud2>("total_pc", 1000);

  //for instance map
  ins_cen_fused_pub_ = nh.advertise<visualization_msgs::MarkerArray>("ins_cen_fused", 1000);

  //for RSN map
  odom_pub_ = nh.advertise<nav_msgs::Odometry>("odom", 10000);
  grid_normal_pub_       = nh.advertise<visualization_msgs::MarkerArray>("grid_normals", 1000);
  grid_normal_fused_pub_ = nh.advertise<visualization_msgs::MarkerArray>("grid_normals_fused", 1000);
  road_grid_pub_ = nh.advertise<visualization_msgs::MarkerArray>("road_grid", 1000);

  if (rsn_or_instance==0)
  {
    generate_RSN_map();
  }
  else if (rsn_or_instance==1)
  {
    generate_instance_map();
  }
  
  return 0;
}
