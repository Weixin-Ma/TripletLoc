//Author:   Weixin Ma       weixin.ma@connect.polyu.hk
//one of the baseline in TripletLoc paper, using coarse-to-fine matching strategy for Triplet-Graph
//"Triplet-Graph: Global Metric Localization Based on Semantic Triplet Graph for Autonomous Vehicles", IEEE RA-L, 2024.
#include <ros/ros.h>
#include "./../include/tripletgraph.h"

#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <yaml-cpp/yaml.h>


#include "./../include/nanoflann/nanoflann.hpp"                                  
#include "./../include/nanoflann/KDTreeVectorOfVectorsAdaptor.h"                 
#include "./../include/nanoflann/nanoflann_utils.h"

typedef KDTreeVectorOfVectorsAdaptor<std::vector< std::vector<float>>, float>my_kd_tree_t;  

std::vector<int> ref_frames_index_;
std::vector<std::map<__int16_t,std::map<__int16_t, instance_center>> > ref_frames_ins_;
std::vector<Descriptors> ref_frames_TG_descriptors_;
std::vector<std::vector<float>> ref_frames_des_;  //concated normalized descriptors

int omp_num_threads_ = 8;

TripletGraph TG_maneger_;
int top_k_candidates_ =1;
int top_k_search_method_=0; //0: force search, 1: kd-tree search

float offset_z = 30.0;

std::vector<std::pair<std::string, Eigen::Matrix4d>> get_pose_gt_Helipr(std::string pose_gt_file)  //for mulran
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

std::vector<int> get_topk_cand(Descriptors que_des)
{ 
  omp_set_num_threads(omp_num_threads_);
  std::vector<int> topk_cand;

  if (top_k_candidates_ == -1)   // not use top-k, all frames are candidates
  {
    for (int i = 0; i < ref_frames_TG_descriptors_.size(); ++i)
    {
      topk_cand.push_back(i);
    }
  }
  else    // using top-k candidates
  {
    std::vector<float> sim_scores(ref_frames_TG_descriptors_.size());

    // Parallelize the computation of similarity scores
    #pragma omp parallel for
    for (int i = 0; i < ref_frames_TG_descriptors_.size(); ++i)
    {
      sim_scores[i] = TG_maneger_.cal_similarity(que_des, ref_frames_TG_descriptors_[i]);
    }

    std::vector<int> sort_indx = TG_maneger_.argsort<float>(sim_scores);

    for (int i = 0; i < top_k_candidates_; ++i)
    {
      topk_cand.push_back(sort_indx[i]);
    }
  }

  return topk_cand;
}

std::pair<int, Eigen::Matrix4d> get_refine_result(Descriptors que_des, std::map<__int16_t,std::map<__int16_t, instance_center>> ins_cens_que, std::vector<int> topk_cand)
{
  int best_match = -1;
  Eigen::Matrix4d best_pose;
  float best_sim = -1000000.0;
  double match_t=0, est_t=0;

  for (int i = 0; i < topk_cand.size(); ++i)
  {
    TicToc match_tic;
    std::map<__int16_t,std::map<__int16_t, match>> matches = TG_maneger_.get_vertex_matches(ref_frames_TG_descriptors_[topk_cand[i]], que_des);
    match_t = match_t + match_tic.toc();

    TicToc est_tic;
    std::pair<Eigen::Matrix4d, Eigen::Matrix4d> est_T_s = TG_maneger_.pose_estimate_omp(matches, ref_frames_ins_[topk_cand[i]], ins_cens_que);
    Eigen::Matrix4d est_T        = est_T_s.second;            //T*
    Eigen::Matrix4d est_T_coarse = est_T_s.first;             //~T
    est_t = est_t + est_tic.toc();

    //get the refined similarity
    std::map<__int16_t,std::map<__int16_t, match>> filtered_matches = TG_maneger_.select_matches(matches,est_T,ref_frames_ins_[topk_cand[i]],ins_cens_que);

    float similarity_refined = TG_maneger_.cal_refined_similarity(ref_frames_TG_descriptors_[topk_cand[i]],que_des,filtered_matches);
    
    if (similarity_refined>best_sim)
    {
      best_sim = similarity_refined;
      best_match = topk_cand[i];
      best_pose = est_T;
    }
  }

  std::cout<<"Match time: "<<match_t/(double)topk_cand.size()<<", Estimation time: "<<est_t/(double)topk_cand.size()<<std::endl;

  return std::make_pair(best_match, best_pose);
}


visualization_msgs::Marker match_visual(Eigen::Matrix4d que_pose, Eigen::Matrix4d ref_pose, int id, bool correct_or_false)
{
  visualization_msgs::Marker edge;
  
  if (correct_or_false)
  {
    edge.color.r=0.0;         
    edge.color.g=1.0;         
    edge.color.b=0.0;  

    edge.ns = "matches";
  }
  else
  {
    edge.color.r=1.0;         
    edge.color.g=0.0;         
    edge.color.b=0.0;  
    edge.ns = "wrong_matches";   
  }

  edge.color.a =1.0;
  edge.pose.orientation.w=1.0;
  edge.scale.x =1.0;         
  edge.header.frame_id = "tripletgraph";
  edge.type = visualization_msgs::Marker::LINE_LIST;     
  edge.header.stamp=ros::Time::now();
  //set marker action
  edge.action = visualization_msgs::Marker::ADD;
  edge.lifetime = ros::Duration();//(sec,nsec),0 forever


  geometry_msgs::Point p1, p2;
  p1.x = que_pose(0,3);
  p1.y = que_pose(1,3);
  p1.z = que_pose(2,3)+offset_z;

  p2.x = ref_pose(0,3);
  p2.y = ref_pose(1,3);
  p2.z = ref_pose(2,3);

  edge.id= id;
  edge.points.push_back(p1);
  edge.points.push_back(p2);

  return edge;  
}

std::vector<float> concate_normal_des(Descriptors des)
{
  //concate {Des^{l}} as a single and normalized vector, then Eq.2 is simplified as the cosine similarity between these vectors
  int totalSize = 0;

  std::vector<Eigen::VectorXf> vecs;
  for(auto it = des.global_descriptor.begin(); it != des.global_descriptor.end(); ++it)
  {
    // Flatten the float matrix to a vector (row-major order)
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> floatMatrix =  it->second.cast<float>();
    Eigen::VectorXf vec = Eigen::Map<Eigen::VectorXf>(floatMatrix.data(), floatMatrix.size());
    vecs.push_back(vec);
    totalSize += vec.size();
  }     

  //concate the des.vertex_descriptors
  Eigen::VectorXf concatenated(totalSize);

  // Concatenate each vector in the std::vector into the concatenated Eigen::VectorXf
  int offset = 0;
  for (const auto& vec : vecs) {
    concatenated.segment(offset, vec.size()) = vec;
    offset += vec.size();
  }

  //normalize the descriptor
  concatenated.normalize();

  std::vector<float>query_Des = std::vector<float>(concatenated.data(), concatenated.data() + concatenated.size());

  return query_Des;
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "sgloop");
  ros::NodeHandle nh("~");

  ros::Publisher ref_frames_pub_, query_frame_pub_, match_pub_, wrong_match_pub_, instance_pub_;
  ref_frames_pub_  = nh.advertise<sensor_msgs::PointCloud2>("ref_frames", 10000);
  query_frame_pub_ = nh.advertise<sensor_msgs::PointCloud2>("que_frames", 1000);
  instance_pub_    = nh.advertise<sensor_msgs::PointCloud2>("instances", 1000);
  match_pub_       = nh.advertise<visualization_msgs::Marker>("matches", 1000);
  wrong_match_pub_ = nh.advertise<visualization_msgs::Marker>("wrong_matches", 1000);

  //load configures
  std::string config_path;
  nh.getParam("config_file",config_path);
  auto data_cfg = YAML::LoadFile(config_path);

  std::string save_dir;
  nh.getParam("save_dir",save_dir);

  std::string ref_seq_dir, que_seq_dir;
  nh.getParam("ref_seq_dir",ref_seq_dir);
  nh.getParam("que_seq_dir",que_seq_dir);

  std::string resutl_txt_dir = save_dir +"result.txt";
  std::ofstream result_txt(resutl_txt_dir);
  result_txt<<"[loop_frame_distance] [loop_que2ref_RTE] [loop_que2ref_RRE] [loop_solve_time]"<<std::endl;

  TG_maneger_.set_config(config_path);

  std::string pose_gt_file_ref_ = ref_seq_dir + "LiDAR_GT/Ouster_gt.txt";      //traj gt for the ref seq
  std::string cloud_path_ref_   = ref_seq_dir + "LiDAR/Ouster_filterd/";
  std::string label_path_ref_   = ref_seq_dir + "labels/";

  std::string pose_gt_file_query_  = que_seq_dir + "LiDAR_GT/Ouster_gt.txt";   //traj gt for the query seq
  std::string cloud_path_query_    = que_seq_dir + "LiDAR/Ouster_filterd/";
  std::string label_path_query_    = que_seq_dir + "labels/";

  omp_num_threads_ = data_cfg["omp_num_threads_knn"].as<int>();

  float frames_interval_que_   = data_cfg["query_related"]["frames_interval"].as<float>(); 
  float frames_interval_refs_  = data_cfg["ref_related"]["frames_interval"].as<float>();      

  float RRE_thr_    = data_cfg["metric_para"]["RRE_thr"].as<float>();        
  float RTE_thr_    = data_cfg["metric_para"]["RTE_thr"].as<float>();        

  top_k_candidates_ = data_cfg["tripletgraph_para"]["top_k_candidates"].as<int>();        //top-k candidates 
  top_k_search_method_ = data_cfg["tripletgraph_para"]["top_k_search_method"].as<int>();  //0: force search, 1: kd-tree search

  std::vector<std::pair<std::string, Eigen::Matrix4d>>gt_poses_ref_= get_pose_gt_Helipr(pose_gt_file_ref_);
  std::vector<std::pair<std::string, Eigen::Matrix4d>>gt_poses_query_= get_pose_gt_Helipr(pose_gt_file_query_);

  //**** extract descriptors for the reference frames
  TicToc build_ref_t;
  std::cout<<"*****Start extracting Triplet-Graph descriptors for the reference frames******"<<std::endl;

  int total_ins_num_ref=0;

  Eigen::Matrix4d last_pose_ref;
  last_pose_ref <<  1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1;  
  
  pcl::PointCloud<pcl::PointXYZ> ref_kf_traj, que_kf_traj; 
  pcl::PointXYZ one_kf_pose;
  int ref_cout=0;
  for (int i = 0; i < gt_poses_ref_.size(); ++i)
  {      

    Eigen::Matrix4d current_pose_ref = gt_poses_ref_[i].second;
    float d_x = current_pose_ref(0,3) - last_pose_ref(0,3);
    float d_y = current_pose_ref(1,3) - last_pose_ref(1,3);
    float d_z = current_pose_ref(2,3) - last_pose_ref(2,3);
    float dis = sqrt( pow(d_x,2) + pow(d_y,2) + pow(d_z,2) );

    if ( (dis>=frames_interval_refs_) || (i==0))
    {
      std::cout<<"*Refs: frame-"<<i<<", id: "<<gt_poses_ref_[i].first<<" ; ";
      ref_frames_index_.push_back(i);
      std::string pc_dir = cloud_path_ref_  + gt_poses_ref_[i].first+".bin";
      std::string label_dir = label_path_ref_ + gt_poses_ref_[i].first+".label";

      TicToc TG_des_time;
      instance_result ins_ref = TG_maneger_.get_ins_cen(pc_dir, label_dir);
      std::map<__int16_t,std::map<__int16_t, instance_center>> ins_cens_ref = ins_ref.instance_centriods;
      Descriptors ref_descriptors = TG_maneger_.get_descriptor_fast(ins_cens_ref);
      std::cout<<"Load pc & label: "<<std::fixed<<std::setprecision(2)<<TG_maneger_.load_pc_label_time<<" ms ; ";
      double extract_des_time = TG_des_time.toc() - TG_maneger_.load_pc_label_time;
      std::cout<<"Extract des: "<<std::fixed<<std::setprecision(2)<<extract_des_time<<" ms"<<std::endl;

      ref_frames_ins_.push_back(ins_cens_ref);
      ref_frames_TG_descriptors_.push_back(ref_descriptors);

      std::vector<float> ref_des_nor_concat = concate_normal_des(ref_descriptors);   //for kd-tree
      ref_frames_des_.push_back(ref_des_nor_concat);      //for kd-tree
      
      last_pose_ref = current_pose_ref;    
      ref_cout=ref_cout + 1; 

      //store and visual kf traj
      one_kf_pose.x = current_pose_ref(0,3);
      one_kf_pose.y = current_pose_ref(1,3);
      one_kf_pose.z = current_pose_ref(2,3);
      ref_kf_traj.push_back(one_kf_pose);
      sensor_msgs::PointCloud2 ref_kf_traj_msg;
      pcl::toROSMsg(ref_kf_traj, ref_kf_traj_msg);
      ref_kf_traj_msg.header.frame_id = "tripletgraph";
      ref_kf_traj_msg.header.stamp = ros::Time::now();
      ref_frames_pub_.publish(ref_kf_traj_msg);

      // //visualize the instance center
      pcl::PointCloud<pcl::PointXYZRGB> instance_cloud_trans;
      pcl::transformPointCloud(ins_ref.instance_as_pc,instance_cloud_trans,current_pose_ref); 
      sensor_msgs::PointCloud2 instance_cloud;
      pcl::toROSMsg(instance_cloud_trans, instance_cloud);
      instance_cloud.header.frame_id = "tripletgraph";
      instance_cloud.header.stamp = ros::Time::now();
      instance_pub_.publish(instance_cloud);

      total_ins_num_ref = total_ins_num_ref + ins_ref.instance_number.second;
    }
  }
  float average_ins_num = (float)total_ins_num_ref/(float)ref_cout;
  std::cout<<"ref_des_size="<<ref_frames_TG_descriptors_.size()<<", total instance num="<<total_ins_num_ref<<", average instance num="<<average_ins_num <<std::endl;
  std::cout<<"Successfully extract Triplet-Graph descriptors for the reference frames, consuming time: "<<build_ref_t.toc()<<" ms"<<std::endl;

  //getchar();

  // std::cout<<"~~~~~~~~~~~~~~~~~~~Start performing loop detection~~~~~~~~~~~~~~~~~~~~~~"<<std::endl;
  int dim = ref_frames_des_[0].size();
  my_kd_tree_t kd_tree(dim /*dim*/, ref_frames_des_, 10 /* max leaf */);

  //******result for evaluation
  std::vector<double> rres;
  std::vector<double> rtes;
  std::vector<double> solve_times;

  Eigen::Matrix4d last_pose_que;
  last_pose_que << 1, 0, 0, 0,
                   0, 1, 0, 0,
                   0, 0, 1, 0,
                   0, 0, 0, 1;  

  //loop closure detection for query frames
  int success_local_frame =0;
  int total_que_frame = 0;

  for (int i = 0; i < gt_poses_query_.size(); ++i)
  {
    Eigen::Matrix4d current_pose_que = gt_poses_query_[i].second;
    float d_x = current_pose_que(0,3) - last_pose_que(0,3);
    float d_y = current_pose_que(1,3) - last_pose_que(1,3);
    float d_z = current_pose_que(2,3) - last_pose_que(2,3);
    float dis = sqrt( pow(d_x,2) + pow(d_y,2) + pow(d_z,2) );

    if ( (dis>=frames_interval_que_) || (i==0))
    {

      std::cout<<"**~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"<<std::endl;
      std::cout<<">Current query frame-"<<gt_poses_query_[i].first<<"-"<<std::endl;
      std::string pc_dir = cloud_path_query_  + gt_poses_query_[i].first+".bin";
      std::string label_dir = label_path_query_ + gt_poses_query_[i].first+".label";

      TicToc TG_total_time;
      TicToc TG_des_t;
      instance_result ins_que = TG_maneger_.get_ins_cen(pc_dir, label_dir);
      std::map<__int16_t,std::map<__int16_t, instance_center>> ins_cens_que = ins_que.instance_centriods;
      Descriptors que_descriptors = TG_maneger_.get_descriptor_fast(ins_cens_que);
      double des_time = TG_des_t.toc();
      std::cout<<"Extract Triplet-Graph descriptors, consuming time: "<<des_time<<" ms"<<std::endl;

      //get top-k candidates
      std::vector<int> topk_candidates;
      if (top_k_search_method_==1)   //kd-tree search, simiplified the Eq.2
      {
        TicToc TG_kd_t;
        std::vector<float> ref_des_normalized = concate_normal_des(que_descriptors);
        std::vector<size_t> ret_indexes(top_k_candidates_);
        std::vector<float> out_dists_sqr(top_k_candidates_);   
        nanoflann::KNNResultSet<float> resultSet(top_k_candidates_);
        resultSet.init(&ret_indexes[0], &out_dists_sqr[0]); 
        kd_tree.index->findNeighbors(resultSet, &ref_des_normalized[0],nanoflann::SearchParams(10));
        std::vector<int> topk_candidates_kd(ret_indexes.begin(), ret_indexes.end());
        topk_candidates = topk_candidates_kd;
        double kd_time = TG_kd_t.toc();
        std::cout<<"Get top-k candidates by kd-tree, consuming time: "<<kd_time<<" ms"<<std::endl;
      }
      else if (top_k_search_method_==0)  //force search, using Eq.2
      {
        TicToc TG_topk_t;
        topk_candidates = get_topk_cand(que_descriptors);
        double topk_time = TG_topk_t.toc();
        std::cout<<"Get top-k candidates, consuming time: "<<topk_time<<" ms"<<std::endl;
      }

      //refine the top-k candidates and get the final result
      TicToc TG_refine_t;
      std::pair<int, Eigen::Matrix4d>final_loop= get_refine_result(que_descriptors, ins_cens_que, topk_candidates);
      double refine_time = TG_refine_t.toc();
      double tg_time = TG_total_time.toc()-TG_maneger_.load_pc_label_time;
      std::cout<<"Refine time for loop closure: "<<refine_time<<" ms"<<std::endl;
      std::cout<<"Total time: "<<tg_time<<" ms"<<std::endl;
      solve_times.push_back(tg_time);

      //*****evaluate the result
      int loop_id_in_ref = ref_frames_index_[final_loop.first];
      Eigen::Matrix4d ref_loop_pose = gt_poses_ref_[loop_id_in_ref].second;
      Eigen::Matrix4d query_pose = current_pose_que;

      // //check whether the loop detection is success
      double dis_que2ref =  (query_pose.topRightCorner(3, 1) - ref_loop_pose.topRightCorner(3, 1)).norm();  //dis between the loop frame and the que frame

      Eigen::Matrix4d gt_que2ref = ref_loop_pose.inverse()*query_pose;    //ground truth relative pose between the loop frame and the que frame
      double RTE =  (final_loop.second.topRightCorner(3, 1) - gt_que2ref.topRightCorner(3, 1)).norm();
      Eigen::Matrix3d est_R = final_loop.second.block<3,3>(0,0);
      Eigen::Matrix3d gt_R  = gt_que2ref.block<3,3>(0,0);      
      double a1 = ((est_R * gt_R.inverse()).trace() - 1) * 0.5;
      double aa1= std::max(std::min(a1,1.0), -1.0);
      double RRE = acos(aa1)*180/M_PI;

      result_txt<<std::fixed<<std::setprecision(6)<<dis_que2ref<<" "<<RTE<<" "<<RRE<<" "<<tg_time<<std::endl;

      bool success_loop = false;
      std::cout<<">Matched frame-"<<loop_id_in_ref<<std::endl;
      double corase_RTE_thr = 50.0; //since the 6-DoF relative pose is availiable, so we set a larger RTE threshold for the retrieval frame
      if (dis_que2ref<corase_RTE_thr && RTE<RTE_thr_ && RRE<RRE_thr_)        
      {
        std::cout<<"\033[36;1m ****Success Localization \033[0m"<< "RTE="<<RTE<<", RRE="<<RRE <<std::endl;
        success_local_frame = success_local_frame +1;
        success_loop = true;
        rres.push_back(RRE);
        rtes.push_back(RTE);
      }
      else  //error loop 
      {
        std::cout<<"\033[31;1m ****Faile Localization \033[0m"<< "RTE="<<RTE<<", RRE="<<RRE <<std::endl;
      }

      //visual the query keyframe traj
      one_kf_pose.x = current_pose_que(0,3);
      one_kf_pose.y = current_pose_que(1,3);
      one_kf_pose.z = current_pose_que(2,3)+offset_z;
      que_kf_traj.push_back(one_kf_pose);
      sensor_msgs::PointCloud2 que_kf_traj_msg;
      pcl::toROSMsg(que_kf_traj, que_kf_traj_msg);
      que_kf_traj_msg.header.frame_id = "tripletgraph";
      que_kf_traj_msg.header.stamp = ros::Time::now();
      query_frame_pub_.publish(que_kf_traj_msg);

      // //visualize the quey keyframe instance center
      pcl::PointCloud<pcl::PointXYZRGB> instance_cloud_trans;
      Eigen::Matrix4d mat = current_pose_que;
      mat(2,3) = mat(2,3) + offset_z;
      pcl::transformPointCloud(ins_que.instance_as_pc,instance_cloud_trans,mat); 
      sensor_msgs::PointCloud2 instance_cloud;
      pcl::toROSMsg(instance_cloud_trans, instance_cloud);
      instance_cloud.header.frame_id = "tripletgraph";
      instance_cloud.header.stamp = ros::Time::now();
      instance_pub_.publish(instance_cloud);

      //visualize the match
      visualization_msgs::Marker match = match_visual(query_pose,ref_loop_pose,i,success_loop);

      if (success_loop)
      {
        match_pub_.publish(match);
      }
      else
      {
        wrong_match_pub_.publish(match);
      }

      last_pose_que = current_pose_que; 
      total_que_frame = total_que_frame + 1;
    }

  }

  std::cout<<std::endl;
  double success_rate = (double)success_local_frame/total_que_frame;
  std::cout<<std::fixed<<std::setprecision(6)<<"\033[32;1m**Total global localization rate** :\033[0m"<< (double)success_local_frame/total_que_frame<<" , "<<success_local_frame<<" out of "<<total_que_frame<<std::endl;

  //calculate average and std
  Eigen::VectorXd rre_E = Eigen::Map<Eigen::VectorXd, Eigen::Aligned>(rres.data(), rres.size());
  Eigen::VectorXd rte_E = Eigen::Map<Eigen::VectorXd, Eigen::Aligned>(rtes.data(), rtes.size());
  Eigen::VectorXd pose_solve_time_E = Eigen::Map<Eigen::VectorXd, Eigen::Aligned>(solve_times.data(), solve_times.size());
  
  double rre_ave = rre_E.mean();
  double rre_std = std::sqrt((rre_E.array() - rre_ave).square().sum() / rre_E.size());

  double rte_ave = rte_E.mean();
  double rte_std = std::sqrt((rte_E.array() - rte_ave).square().sum() / rte_E.size());

  double pose_solve_time_ave = pose_solve_time_E.mean();
  double pose_solve_time_std = std::sqrt((pose_solve_time_E.array() - pose_solve_time_ave).square().sum() / pose_solve_time_E.size());

  std::cout<<std::fixed<<std::setprecision(3)<<"RTE= "<<rte_ave<<"±"<<rte_std<<std::endl;
  std::cout<<std::fixed<<std::setprecision(3)<<"RRE= "<<rre_ave<<"±"<<rre_std<<std::endl;
  std::cout<<std::fixed<<std::setprecision(3)<<"Pose solve time= "<<pose_solve_time_ave<<"±"<<pose_solve_time_std<<std::endl;

  result_txt<<"~~~~~~~~~~~~~~~~~~~~~~~~~~"<<std::endl;
  result_txt<<"Total global localization rate: "<< success_rate<<" , "<<success_local_frame<<" out of "<<total_que_frame<<std::endl;
  result_txt<<"RTE_thre="<<RTE_thr_<<", RRE_thre="<<RRE_thr_<<std::endl;
  result_txt<<std::fixed<<std::setprecision(6)<<"RTE= "<<rte_ave<<"±"<<rte_std<<std::endl;
  result_txt<<std::fixed<<std::setprecision(6)<<"RRE= "<<rre_ave<<"±"<<rre_std<<std::endl;
  result_txt<<std::fixed<<std::setprecision(6)<<"Pose solve time= "<<pose_solve_time_ave<<"±"<<pose_solve_time_std<<std::endl;

  result_txt.close();

  return 0;
}