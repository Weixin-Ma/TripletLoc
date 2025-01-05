#ifndef TripletLoc_H
#define TripletLoc_H

#include <ros/ros.h>
#include <iostream>
#include <fstream>
#include <sys/types.h>
#include <dirent.h>
#include <string.h>
#include <numeric>
#include <vector>
#include <algorithm>
#include <random>
#include <yaml-cpp/yaml.h>
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

#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include "nanoflann/nanoflann.hpp"                                  //for kd-tree
#include "nanoflann/KDTreeVectorOfVectorsAdaptor.h"                 //cosine similarity serach using kd-tree
#include "nanoflann/nanoflann_utils.h"                              

#include "tic_toc.h"
#include "./../include/teaser/registration.h"                       //for pose_estimation_teaser() and mcq search in pose_estimate_gtsam()


struct semantic_configs{        
    std::string obj_class;
    int color_b;
    int color_g;
    int color_r;
};

struct configs{
    float frames_interval_ques;
    float voxel_leaf_size_que;     

    float ins_fuse_radius;

    std::string cloud_path_que;
    std::string label_path_que;
    std::string pose_gt_file_que;

    int des_type;

    std::string ins_map_txts_path;                 //instance-level map txt
    std::string rsn_map_txts_path;                 //RSN map txt

    std::vector<__int16_t> classes_for_graph;      //i.e., {l_m}

    std::map<__int16_t,semantic_configs> semantic_name_rgb;

    std::map<__int16_t,float> EuCluster_para;

    std::map<__int16_t,int> minimun_point_in_one_instance;

    int angle_resolution;          //i.e., /theta (degree) in Triplet-Graph paper
    float edge_dis_thr;            //i.e., /tau_{edge} in paper
    float dis_resolution;          //i.e., /tau in papaer

    int knn_match_number;                    //top-k mathces for vertex matching
    float vertex_dis_thre_for_correct_match; //beyond paper, evaluate whether a vertex match is "correct" "acceptable"

    float RRE_thr;
    float RTE_thr;

    //solver realated
    int solver_type;    //0: Gtsam (point-to-point with/without rotation constraint); 1: Teaser (point-to-point)
    bool use_rsn;       //whether use rotation constraint from RSN map, only for Gtsam solver
    float noise_bound;  //(m) para for Teaser (in fun pose_estimate() Teaser is used as the solver; in fun pose_estimate_new() Gtsam is used as solver, we only use MCQ search from Teaser)
    double rot_delta;   //(degree) i.e., \delta: additional perturbation for rotation measurement noise in Eq.3

};

struct instance_centriod
{
    float x;
    float y;
    float z;
};

struct id_label
{
    __int16_t label;
    __int16_t id;
};

struct objects_and_pointclouds
{
    std::map<__int16_t, pcl::PointCloud<pcl::PointXYZRGB>> pc;
    std::map<__int16_t,std::map<__int16_t, instance_centriod>> instance_cens;
    std::map<__int16_t, int> instance_numbers;
};

struct Des_and_AdjMat   //vertex descriptors and adjacency matrix for a graph
{
    std::map<__int16_t,std::map<__int16_t, std::vector<int> >> Descriptors_vec_angle;    //relative angle based (Des_{v_j}^{\alpha})
    std::map<__int16_t,std::map<__int16_t, std::vector<int> >> Descriptors_vec_length;   //edge length based    (Des_{v_j}^{d})
    std::map<__int16_t,std::map<__int16_t, std::vector<int> >> Descriptors_vec_comb;     //combination          (Des_{v_j})

    std::map<__int16_t,std::map<__int16_t, bool>> single_vertex_flags;        

    std::map<__int16_t,std::vector<__int16_t> > single_vertex_index;         
    std::map<__int16_t,std::vector<__int16_t> > non_single_vertex_index;  


    Eigen::MatrixXi adj_mat;                   //for viusalization
    std::map<int, id_label> index_2_label_id;
    int triplet_number;                        //how many triplets in the graph
};

struct class_comb
{
    __int16_t l_f;     //i.e., l^f
    __int16_t l_m;     //i.e., l^m
    __int16_t l_t;     //i.e., l^t
}; 

struct class_comb_info
{
    std::vector<class_comb> triplet_classes;                                                       //i.e., C_{lm} for lm
    int number_combinations;                                                                       //i.e., N1
    std::map<__int16_t,std::map<__int16_t, std::map<__int16_t,int> > > triplet_to_descriptor_bit;  //triplet combination to descriptor bit index, e.g., triplet wiht class combination {l^f,l^m,l^t} -> triplet_to_descriptor_bit[l^f][l^m][l^t]
    std::map<int, std::vector<class_comb>> bit_to_triplet;                                         //descriptor bit index to triplet combination, e.g., descriptor bit 1 -> bit_to_triplet[1]
}; 

struct matches_info
{
    bool invalid;                         //true: the match results for current vertex is invalid, will not be used for pose estimation
    std::vector<size_t> matches_index;    //top-k matches id in the kd-tree
    std::vector<float> matches_sims;      //cosine similarities for top-k matches 
    int max_sim;                          //max similarity, not used in the current version
    std::vector<bool> correct_flags;      //beyond paper, flage for "correct" "acceptable" matches in the top-k matches
};

struct pose_with_clique
{
    bool valid;
    std::vector<int> max_clique;
    Eigen::Matrix4d pose;

    double mcq_time;
    double pose_time;
};

class TripletLoc
{     
public:
    //Constructor
    TripletLoc(ros::NodeHandle &nh);   

    //load config
    configs get_config(std::string config_file_dir);

    //extract class combination for first-level bins
    void get_class_combination();

    //get pose ground truth 
    std::vector<std::pair<std::string, Eigen::Matrix4d>> get_pose_gt_Helipr(std::string pose_gt_file);  //for HeliPR dataset

    //main function
    void run();

    //load pointcloud and label file
    std::map<__int16_t, pcl::PointCloud<pcl::PointXYZRGB>> get_pointcloud_with_semantic_helipr(std::string bin_dir, std::string label_file);

    //load txt file to string vector
    std::vector<std::vector <std::string>> load_txt_2_string_vector(std::string dir_txt);

    //load and build instance-level map, and extract descriptors for each vertex
    void load_and_build_ins_map(std::string dir_global_map);
    
    //vertex descriptor extraction
    Des_and_AdjMat get_descriptor_fast(std::map<__int16_t,std::map<__int16_t, instance_centriod>> ins_cens);   
    double get_angle(double x1, double y1, double x2, double y2, double x3, double y3);            //get relative angle of a triplet
    double get_edge_length_ave(double x1, double y1, double x2, double y2, double x3, double y3);  //get average edge length of of a triplet,xy only
    
    //get instance center
    objects_and_pointclouds get_ins_cen(std::string bin_flie, std::string label_file);  
    
    //vertex matching
    std::vector<float> des_normalization(std::vector<int> input);                                           //normalize the descriptor
    std::map<__int16_t,std::map<__int16_t, matches_info>> vertex_match(Des_and_AdjMat query_vertex_des); //kd-tree based, vertex matching
    matches_info knn_search_kd(__int16_t class_label, std::vector<int> query_Des);                          //top-k matching using cosine similarity and kd-tree

    //pose estimation
    pose_with_clique pose_estimate_teaser(std::map<__int16_t,std::map<__int16_t, matches_info>> matches,std::map<__int16_t,std::map<__int16_t,instance_centriod>> ins_cen);  //using teaser++ to estimate pose, only point-to-point registration
    pose_with_clique pose_estimate_gtsam(std::map<__int16_t,std::map<__int16_t, matches_info>> matches,std::map<__int16_t,std::map<__int16_t,instance_centriod>> ins_cen);   //using gtsam to estimate pose, with/without rotation constraint

    //for vertex a in the query, we check the top-k matched vertices, to see whether each matched vertex is close to the vertex a, if yes: this matched pair is considered as correct(acceptable); if not: wrong(unacceptable) 
    //configs_.vertex_dis_thre_for_correct_match is used as the threshold for determining corrrect or wrong; these determation will be stored in the input "matches.correct_flags"
    //once there is at least one correct match pair for a vertex, the match result for this vertex is considered as positive, otherwise, negative 
    //noted that, this function is just used for better visualization and provide an additional evaluation for the vertex matching quality
    std::map<__int16_t, std::pair<int,int>> vertex_match_recall_N(std::map<__int16_t,std::map<__int16_t, matches_info>> &matches, std::map<__int16_t,std::map<__int16_t,instance_centriod>> ins_cen);    
    
    //**functions for visualization**
    //function to delete instance center and edge in the last frame
    std::vector<visualization_msgs::MarkerArray> clean_last_visualization();  

    //visualize instance center
    visualization_msgs::MarkerArray instances_visual(std::map<__int16_t,std::map<__int16_t,instance_centriod>>  ins_cens);

    //visualize edges, global_or_local: 0: global; 1: local; we do not visualize the edge in the global map(too many, making rviz slow)
    visualization_msgs::MarkerArray edges_visual(std::map<__int16_t,std::map<__int16_t,instance_centriod>> ins_cen, Eigen::MatrixXi adj_mat,std::map<int, id_label> index_2_label_id, int global_or_local);

    //visualize vertex matching results
    std::pair<visualization_msgs::MarkerArray, visualization_msgs::MarkerArray> vertex_matches_visual(std::map<__int16_t,std::map<__int16_t, matches_info>> matches,std::map<__int16_t,std::map<__int16_t,instance_centriod>> ins_cen); 

    //visualize the vertices in query frame using the pose estimation result
    visualization_msgs::MarkerArray vertex_using_est_pose_visual(pose_with_clique solution,std::map<__int16_t,std::map<__int16_t,instance_centriod>> ins_cens);

    //visualize assoication between vertices in the max clique
    visualization_msgs::MarkerArray MCQ_visual(std::vector<int> max_clique,std::map<__int16_t,std::map<__int16_t,instance_centriod>> ins_cens);

    //funciton to visualize the position of the current frame in the reference map, green: success; red: fail
    visualization_msgs::Marker local_area_visual(bool success_locate);

private:
    ros::NodeHandle nh_;

    //kd-tree for vertex matching, for other classes, you can make the kd-tree in the same way
    std::unique_ptr< KDTreeVectorOfVectorsAdaptor< std::vector< std::vector<float>> , float> > kd_tree_fence_;  
    std::unique_ptr< KDTreeVectorOfVectorsAdaptor< std::vector< std::vector<float>> , float> > kd_tree_vegetation_;
    std::unique_ptr< KDTreeVectorOfVectorsAdaptor< std::vector< std::vector<float>> , float> > kd_tree_trunk_;
    std::unique_ptr< KDTreeVectorOfVectorsAdaptor< std::vector< std::vector<float>> , float> > kd_tree_pole_;
    std::unique_ptr< KDTreeVectorOfVectorsAdaptor< std::vector< std::vector<float>> , float> > kd_tree_traffic_sign_;

    bool kd_tree_fence_ready_ = false;
    bool kd_tree_vegetation_ready_ = false;
    bool kd_tree_trunk_ready_ = false;
    bool kd_tree_pole_ready_ = false;
    bool kd_tree_traffic_sign_ready_ = false;

    pcl::KdTreeFLANN<pcl::PointXY> kdtree_road_grid_cen_;      //road grid cells centers, for search the anchor point p_a
    std::vector<Eigen::Vector3d> road_grid_normal_;            //road grid cells normals, i.e., vector of n
    std::vector<double> road_grid_normal_std_;                 //road grid cells normals std, i.e., vector of sigma_n 
};

#endif 