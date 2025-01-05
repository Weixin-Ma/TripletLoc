//Author:   Weixin Ma       weixin.ma@connect.polyu.hk
//"Triplet-Graph: Global Metric Localization Based on Semantic Triplet Graph for Autonomous Vehicles", IEEE RA-L, 2024.
#ifndef TRIPLETGRAPH_H
#define TRIPLETGRAPH_H

#define PCL_NO_PRECOMPILE

#include <iostream>
#include <fstream>
#include <sys/types.h>
#include <dirent.h>
#include <string.h>
#include <numeric>
#include <vector>
#include <algorithm>
#include <yaml-cpp/yaml.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <random>
#include <omp.h>

#include <pcl/io/io.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/search/kdtree.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h> 

#include "tic_toc.h"
#include "CostFunction.h"

#include "omp.h"

struct semantic_config{        //semantic info for kitti
    std::string obj_class;
    int color_b;
    int color_g;
    int color_r;
};

struct config_para{
    int angle_resolution;
    float edge_dis_thr;
    float ransca_threshold;         
    
    std::map<__int16_t,float> voxel_leaf_size;
    std::vector<__int16_t> classes_for_graph;
    std::map<__int16_t,float> EuCluster_para;
    std::map<__int16_t,semantic_config> semantic_name_rgb;
    std::map<__int16_t,int> minimun_point_one_instance;
    std::map<__int16_t,float> weights_for_class;
    std::map<__int16_t,double> weights_cere_cost;

    int maxIterations_ransac;
    int cere_opt_iterations_max;
    float percentage_matches_used;
    float similarity_refine_thre;
    float ins_fuse_radius;
};

struct class_combination
{
    __int16_t l_f;
    __int16_t l_m;
    __int16_t l_t;
}; 

struct class_combination_info
{
    std::vector<class_combination> triplet_classes;                                                //{C}_{l^m}
    int dim_combination;                                                                           //N1
    std::map<__int16_t,std::map<__int16_t, std::map<__int16_t,int> > > triplet_to_descriptor_bit;  //C to index of C in {C}_{l^m}
    std::map<int, std::vector<class_combination>> bit_to_triplet;                                  //index in {C}_{l^m} to C
}; 

struct instance_center    //geometric centriod of instance
{
    float x;
    float y;
    float z;
};

struct label_id
{
    __int16_t label;
    __int16_t id;
};

struct instance_result
{
    std::map<__int16_t,std::map<__int16_t, instance_center>>  instance_centriods;
    std::pair< std::map<__int16_t, int>, int> instance_number;   //first: instance_number for each class,  second: total instance number
    pcl::PointCloud<pcl::PointXYZRGB> instance_as_pc;            //convert instance centriods to pointcloud
};

struct Descriptors 
{
    std::map<__int16_t,std::map<__int16_t, Eigen::MatrixXi>>  vertex_descriptors;    //set of vertex descriptors, {Des_{v_j}}
    std::map<__int16_t, Eigen::MatrixXi>  global_descriptor;                         //global descriptor, {Des^{l}}
};

struct match
{
    bool available;          //flage for a match
    __int16_t id;            //id of the matched vertex
    double similarity;       
};


struct match_xyz_label
{
   double x1;
   double y1;
   double z1;
   double x2;
   double y2;
   double z2;
    __int16_t label;
};

struct pose_est
{
    Eigen::Quaterniond ori_Q;
    Eigen::Vector3d trans;

};

class TripletGraph
{     //semantic graph extractor for lidar   
public:
    int frame_count_;

    int omp_num_threads_ransc_ = 8;
    int omp_num_threads_vetex_match_ =4;
    
    double load_pc_label_time=0;

    //Constructor
    TripletGraph();   

    //used parameters
    config_para conf_para_;    
    
    //setting
    void set_config(std::string config_file_dir);

    //load config
    config_para get_config(std::string config_file_dir);

    ////for building {C}_{lm}, i.e., first-level bins
    void get_class_combination();

    //load pointcloud with semantic from Helipr dataset
    std::map<__int16_t, pcl::PointCloud<pcl::PointXYZRGB>> get_pointcloud_with_semantic_helipr(std::string bin_dir, std::string label_file);

    //get instance
    instance_result get_ins_cen(std::string bin_flie, std::string label_file);

    //get vertex and global descriptors, fast version (without distance matrix calculation, using kd-tree instead)
    double get_angle(double x1, double y1, double x2, double y2, double x3, double y3);      //untli function, for calcuating \alpha
    Descriptors get_descriptor_fast(std::map<__int16_t,std::map<__int16_t, instance_center>> ins_cens);   //build vertex descriptor for global lidar map, fast version

    //vertex matching 
    std::map<__int16_t,std::map<__int16_t, match>> get_vertex_matches(Descriptors descriptor1, Descriptors descriptor2);

    //pose estimation, result.first is coarse pose ~T, result.second is optimized pose T*
    void get_random_samples(const std::vector<match_xyz_label>& matched_pairs, std::vector<match_xyz_label>& samples, int sample_amount, std::mt19937& rng);
    std::pair<Eigen::Matrix4d, Eigen::Matrix4d> pose_estimate_omp(std::map<__int16_t,std::map<__int16_t, match>> matches, std::map<__int16_t,std::map<__int16_t, instance_center>> ins_cens1, std::map<__int16_t,std::map<__int16_t, instance_center>> ins_cens2);

    //6-DoF pose solver，optimization
    pose_est pose_solver(std::vector<match_xyz_label> matches, Eigen::Quaterniond init_Q, Eigen::Vector3d init_xyz); 

    //6-DoF-pose solver, coarse
    pose_est solver_svd(std::vector<match_xyz_label> matches);     

    //similarity between two graphs without projection-based selection
    float cal_similarity(Descriptors descriptor1, Descriptors descriptor2);

    //projection-based selection 
    std::map<__int16_t,std::map<__int16_t, match>> select_matches(std::map<__int16_t,std::map<__int16_t, match>> origianl_matches, Eigen::Matrix4d ested_pose, std::map<__int16_t,std::map<__int16_t, instance_center>>ins_cen1,std::map<__int16_t,std::map<__int16_t, instance_center>> ins_cen2);

    //similarity between two graphs after projection-based selection
    float cal_refined_similarity(Descriptors descriptor1, Descriptors descriptor2, std::map<__int16_t,std::map<__int16_t, match>> filtered_match);

    //****************function for sort a vector and return the index************** 
    template <typename T>
    std::vector<int> argsort(const std::vector<T> &v) 
    { 
        std::vector<int> idx(v.size()); 
        iota(idx.begin(), idx.end(), 0); 
        sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v[i1] > v[i2];}); 
        return idx; 
    }

    template <typename D>
    std::vector<int> argsort_s(const std::vector<D> &v) 
    { 
        std::vector<int> idx(v.size());
        iota(idx.begin(), idx.end(), 0); 
        sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v[i1] < v[i2];}); 
        return idx; 
    }
    //****************function for sort a vector and return the index************** 

private:
                                            
};

#endif 
 