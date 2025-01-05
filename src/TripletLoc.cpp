//Author:   Weixin Ma       weixin.ma@connect.polyu.hk
#include "TripletLoc.h"

//for gtasam based pose estimation
#include "Point2PointFactor.h"
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/GncOptimizer.h>
#include <gtsam/slam/PriorFactor.h>

using namespace gtsam;
using symbol_shorthand::X; 

configs configs_;     //configurations for tripletloc

std::vector<std::pair<std::string, Eigen::Matrix4d>> gt_poses_que_;

std::map<__int16_t, class_comb_info> class_combs_infos_;  

//results for global map
//std::map<int, id_label> index_2_label_id_ins_map_;                                      //for instance map, index of matrix to label and id, not used currently, can be used to visualize edges in refernce map, not recommended, can be slow and overflow
//Eigen::MatrixXi adj_mat_in_ins_map_;                                                    //adjacency matrix for instance map, not used currently, can be used to visualize edges in refernce map

std::map<__int16_t,std::map<__int16_t, instance_centriod>> instance_cens_in_ins_map_;
//std::map<__int16_t,std::map<__int16_t, bool>> single_vertex_flags_in_ins_map_;          //not used in current version of TripeltLoc

//std::map<__int16_t,std::vector<__int16_t> > single_vertex_index_in_ins_map_;            
std::map<__int16_t,std::vector<__int16_t> > non_single_vertex_index_in_ins_map_;  

int query_frame_count_;                         
double z_offest_ = 30;      //for visualization

int current_single_vertex_num_     = 0;
int current_non_single_vertex_num_ = 0;

int current_association_num_ = 0;
int current_max_clique_num_  = 0;

Eigen::MatrixXd association_query_;   //for visualization
Eigen::MatrixXd association_match_;   //for visualization


double instance_cluster_t_;

std::map<__int16_t,std::vector< std::vector<float> > > des_vec_all_class_f_for_kd_tree_;   //normalized


TripletLoc::TripletLoc(ros::NodeHandle &nh)
{
  nh_=nh;

  std::string config_path;
  nh_.getParam("config_file",config_path);

  configs_ = get_config(config_path);

  get_class_combination();
}

configs TripletLoc::get_config(std::string config_file_dir)
{ 
  configs output;

  auto data_cfg = YAML::LoadFile(config_file_dir);

  std::string map_path, que_seq_path, pose_gt_file_que;
  nh_.getParam("prior_maps_dir",map_path);
  nh_.getParam("query_seq_dir",que_seq_path);

  output.ins_map_txts_path = map_path +"instances";
  output.rsn_map_txts_path = map_path +"RSN/road_normal.txt";

  output.pose_gt_file_que = que_seq_path + "LiDAR_GT/Ouster_gt.txt";
  output.cloud_path_que = que_seq_path + "LiDAR/Ouster_filterd";
  output.label_path_que = que_seq_path + "labels";

  output.frames_interval_ques              = data_cfg["query_related"]["frames_interval"].as<float>();
  output.knn_match_number                  = data_cfg["query_related"]["knn_match_number"].as<int>();
  output.vertex_dis_thre_for_correct_match = data_cfg["query_related"]["vertex_dis_thre_for_correct_match"].as<float>();

  output.des_type                = data_cfg["query_related"]["des_type"].as<int>();   
  output.voxel_leaf_size_que     = data_cfg["query_related"]["voxel_leaf_size"].as<float>();                             
  output.ins_fuse_radius         = data_cfg["query_related"]["ins_fuse_radius"].as<float>();                     

  output.angle_resolution        = data_cfg["angel_resolution"].as<int>();                               // \theta
  output.edge_dis_thr            = data_cfg["edge_dis_thr"].as<float>();                                 // \tau_{edge}
  output.dis_resolution          = data_cfg["dis_resolution"].as<float>();                               // \tau

  output.RRE_thr               = data_cfg["query_related"]["RRE_thr"].as<float>();
  output.RTE_thr               = data_cfg["query_related"]["RTE_thr"].as<float>();

  output.solver_type             = data_cfg["solver_related"]["solver_type"].as<int>();       //0: gtsam, 1: Teaser
  output.use_rsn                 = data_cfg["solver_related"]["use_rsn"].as<bool>();          //whether use rotation constraint from RSN map, only for Gtsam solver
  output.noise_bound             = data_cfg["solver_related"]["noise_bound"].as<float>();
  output.rot_delta               = data_cfg["solver_related"]["rot_delta"].as<float>();       //additional perturbation for rotation measurement noise in Eq.3

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

  //used class, i.e., L
  for (iter;iter!=classes_for_graph.end();iter++) {
    if(iter->second.as<bool>())
    {
      class_for_graph.push_back(iter->first.as<__int16_t>());
    }
  }
  output.classes_for_graph = class_for_graph;

  //class name & rgb
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

  //paras for instance clustering 
  for (iter2;iter2!=instance_seg_para.end();iter2++) {
    EuCluster_para[iter2->first.as<__int16_t>()] = iter2->second.as<float>();
  }
  output.EuCluster_para = EuCluster_para;

  //paras for instance clustering 
  for (iter3; iter3!=minimun_point_one_instance.end(); iter3++)
  {
    minimun_point_in_one_instance[iter3->first.as<__int16_t>()] = iter3->second.as<int>();
  }
  output.minimun_point_in_one_instance = minimun_point_in_one_instance;

  return output;
}


std::map<__int16_t, pcl::PointCloud<pcl::PointXYZRGB>> TripletLoc::get_pointcloud_with_semantic_helipr(std::string bin_dir, std::string label_file)
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

    input_label.read((char *) &label,sizeof(__int16_t));                        //16bit label data，for voxel floder:16bit，for velodyne folder:32bit
    input_label.read((char *) &id,sizeof(__int16_t));

    my_point.r = configs_.semantic_name_rgb[label].color_r;
    my_point.g = configs_.semantic_name_rgb[label].color_g;
    my_point.b = configs_.semantic_name_rgb[label].color_b;

    output[label].push_back(my_point);
    all_points.push_back(my_point);
  }

  output[-1] = all_points;

  input_point.close();
  input_label.close();

  return output;
}

std::vector<std::pair<std::string, Eigen::Matrix4d>> TripletLoc::get_pose_gt_Helipr(std::string pose_gt_file)  
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

void TripletLoc::get_class_combination()                   //for building {C}_{l^m}, i.e., first-level bins
{
  std::vector<__int16_t> classes_for_graph = configs_.classes_for_graph;

  //generate triplet
  if(classes_for_graph.size()>=2)                          //at least two classes are required
  {
    class_comb single_combine;                             //a C in {C}_{l^m}

    int combination_amount;                                //combination number for {C}_{l^m}, equals to N1

    for (int i = 0; i < classes_for_graph.size(); ++i)     //pick l^m
    {
      combination_amount=0;
      std::vector<class_comb> triplet_classes;                                                          //{C}_{l^m}, l^m = classes_for_graph[m]
      std::map<__int16_t,std::map<__int16_t, std::map<__int16_t,int> > > triplet_to_descriptor_bit;     //C to index of {C}_{l^m} (i.e., first-level bin index)
      std::map<int, std::vector<class_comb>> bit_to_triplet;                                            //index of {C}_{l^m}  to C
      triplet_to_descriptor_bit.clear();
      bit_to_triplet.clear();
      

      //pick l^f and pick l^m are the same
      for (int j = 0; j < classes_for_graph.size(); ++j)       
      {
        single_combine.l_m = classes_for_graph[i];   //i.e., l^m
        single_combine.l_f = classes_for_graph[j];   //i.e., l^f
        single_combine.l_t = classes_for_graph[j];   //i.e., l^t

        triplet_classes.push_back(single_combine);                       
        triplet_to_descriptor_bit[single_combine.l_f][single_combine.l_m][single_combine.l_t] = combination_amount; 
        bit_to_triplet[combination_amount].push_back(single_combine);   
        //std::cout<<combination_amount<<std::endl;
        combination_amount = combination_amount+1;         
      }

      //pick l^f and pick l^m are different
      for (int k = 0; k < classes_for_graph.size(); ++k)   //pick l^f
      {
        std::vector<__int16_t> diff_vertex1 = classes_for_graph;
        for (int l=0; l<k+1; ++l)                          //delet used l^f, for picking l^t
        {
          diff_vertex1.erase(diff_vertex1.begin());
        }

        for (int m = 0; m < diff_vertex1.size(); ++m)      //pick l^t
        {
          //{l^f, l^m, l^t}, e.g., {fence, trunk, pole}
          single_combine.l_m = classes_for_graph[i];
          single_combine.l_f = classes_for_graph[k];
          single_combine.l_t = diff_vertex1[m];

          triplet_classes.push_back(single_combine);                
          triplet_to_descriptor_bit[single_combine.l_f][single_combine.l_m][single_combine.l_t]=combination_amount;   
          bit_to_triplet[combination_amount].push_back(single_combine);     

          //{l^t, l^m, l^f}, e.g., {pole, trunk, fence}
          single_combine.l_m = classes_for_graph[i];
          single_combine.l_t = classes_for_graph[k];
          single_combine.l_f = diff_vertex1[m];

          triplet_classes.push_back(single_combine);                
          triplet_to_descriptor_bit[single_combine.l_f][single_combine.l_m][single_combine.l_t]=combination_amount;   
          bit_to_triplet[combination_amount].push_back(single_combine);     

          combination_amount =combination_amount + 1;  //for vertex
        }
      }

      class_combs_infos_[classes_for_graph[i]].triplet_classes          = triplet_classes;
      class_combs_infos_[classes_for_graph[i]].triplet_to_descriptor_bit= triplet_to_descriptor_bit;
      class_combs_infos_[classes_for_graph[i]].number_combinations      = combination_amount;   
      class_combs_infos_[classes_for_graph[i]].bit_to_triplet           = bit_to_triplet;
    }

    std::map<__int16_t, class_comb_info>::iterator iter;
    iter=class_combs_infos_.begin();
    for ( iter; iter != class_combs_infos_.end(); ++iter)
    {
      std::cout<<"Class-"<<iter->first<<" : "<<"combin amouts-"<<iter->second.number_combinations<<";"<<std::endl;
      for (size_t i = 0; i < iter->second.triplet_classes.size(); ++i)
      {
        std::cout<<"    combines: "<<iter->second.triplet_classes[i].l_f<<"-"<<iter->second.triplet_classes[i].l_m<<"-"<<iter->second.triplet_classes[i].l_t<<
        ", correspoidng bit: "<<iter->second.triplet_to_descriptor_bit[iter->second.triplet_classes[i].l_f][iter->second.triplet_classes[i].l_m][iter->second.triplet_classes[i].l_t]<<std::endl;
      }
    }

  }
}

double TripletLoc::get_angle(double x1, double y1, double x2, double y2, double x3, double y3)
/*get angle ACB, point C is the center point A(x1,y1) B(x2,y2) C(x3,y3), 夹角角度为[0, 180]*/
{
  double theta = atan2(x1 - x3, y1 - y3) - atan2(x2 - x3, y2 - y3);
  if (theta >= M_PI)
  theta -= 2 * M_PI;
  if (theta <= -M_PI)
  theta += 2 * M_PI;
  theta = abs(theta * 180.0 / M_PI);
  return theta;
}

double TripletLoc::get_edge_length_ave(double x1, double y1, double x2, double y2, double x3, double y3) // only xy-coordinate
/*get the average of the length of two edge of triplet ACB, point C is the middle vertex, A(x1,y1) B(x2,y2) C(x3,y3)*/
{
  double len_1 = sqrt( (x1-x3)*(x1-x3) + (y1-y3)*(y1-y3) );
  double len_2 = sqrt( (x2-x3)*(x2-x3) + (y2-y3)*(y2-y3) );

  double len_ave = (len_1+len_2)*0.5;
  return len_ave;
}

Des_and_AdjMat TripletLoc::get_descriptor_fast(std::map<__int16_t,std::map<__int16_t, instance_centriod>> ins_cens)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr centriods(new pcl::PointCloud<pcl::PointXYZ>);
  std::map<__int16_t,std::map<__int16_t, instance_centriod>>::iterator iter;
  iter = ins_cens.begin();

  int index=0;
  std::map<int, id_label> index_2_label_id;                        //index of matrix to label and id

  for (iter; iter!=ins_cens.end(); iter++)
  {
    std::map<__int16_t,instance_centriod>::iterator iter1;
    iter1=iter->second.begin();

    for (iter1; iter1!=iter->second.end(); ++iter1)
    {
      index_2_label_id[index].label = iter->first;
      index_2_label_id[index].id    = iter1->first;

      pcl::PointXYZ one_ins;
      one_ins.x = ins_cens[iter->first][iter1->first].x;
      one_ins.y = ins_cens[iter->first][iter1->first].y;
      one_ins.z = ins_cens[iter->first][iter1->first].z;

      centriods->points.push_back(one_ins);
      index = index +1;
    }
  }

  int ins_amount = index_2_label_id.size();
  int bin_amount = 180 / configs_.angle_resolution;                       //i.e., N2
  int bin_amount_edge = configs_.edge_dis_thr/configs_.dis_resolution;    //i.e., N3

  //set sources points for knn search
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	kdtree.setInputCloud(centriods);

  //perform descriptor extraction
  Des_and_AdjMat output;
  Eigen::MatrixXi adj_mat;
  std::map<__int16_t,std::map<__int16_t, Eigen::MatrixXi>> descriptors_angle;        //i.e., all Des^{a}_{v_j}
  std::map<__int16_t,std::map<__int16_t, std::vector<int> >> descriptors_angle_vec;  

  std::map<__int16_t,std::map<__int16_t, Eigen::MatrixXi>> descriptors_len;          //i.e., all Des^{d}_{v_j}
  std::map<__int16_t,std::map<__int16_t, std::vector<int> >> descriptors_len_vec;  

  std::map<__int16_t,std::map<__int16_t, std::vector<int> >> descriptors_comb;       //i.e., all Des_{v_j}

  adj_mat.resize(ins_amount, ins_amount);
  adj_mat.setZero();

  int triplet_number = 0;                                               //number of constructed triplets, i.e., size of {\Delta}_{v_{j}}
  float radius = configs_.edge_dis_thr;

  std::map<__int16_t,std::map<__int16_t, bool>> single_vertex_flags;    //judge whether the current vertex is a single vertex, i.e., not connected with other vertex

  std::map<__int16_t,std::vector<__int16_t> > single_vertex_index;
  std::map<__int16_t,std::vector<__int16_t> > non_single_vertex_index;

  int single_vertex_count=0;
  int non_single_vertex_count=0;

  if(ins_amount>=3) 
  {
    for (size_t i = 0; i < ins_amount; ++i)            //pick v^m
    {
      pcl::PointXYZ query_point;
      query_point.x = centriods->points[i].x;
      query_point.y = centriods->points[i].y;
      query_point.z = centriods->points[i].z;

      std::vector<int> pointIdxRadiusSearch;
      std::vector<float> pointRadiusSquaredDistance;

      size_t nConnected_vertices = kdtree.radiusSearch(query_point, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance);
      std::vector<int> instance_index = pointIdxRadiusSearch;

      instance_centriod first_vertex, mid_vertex, third_vertex;          //i.e., v^f, v^m, v^t

      __int16_t class_label = index_2_label_id[instance_index[0]].label;                            //here, i = instance_index[0]
      __int16_t instance_id = index_2_label_id[instance_index[0]].id;                              


      int class_combines_amout=class_combs_infos_[class_label].number_combinations;                 //i.e., N1, actually same for all classes

      Eigen::MatrixXi vertex_descriptor_angle;                             //i.e., Des^{a}_{v_j}
      std::vector<int> vertex_des_angle_vec;                               //for kd-tree
      vertex_des_angle_vec.resize(class_combines_amout*bin_amount);
      vertex_descriptor_angle.setZero(class_combines_amout, bin_amount); 

      Eigen::MatrixXi vertex_descriptor_len;                               //i.e., Des^{d}_{v_j}
      std::vector<int> vertex_des_len_vec;                                 //for kd-tree
      vertex_des_len_vec.resize(class_combines_amout*bin_amount_edge);
      vertex_descriptor_len.setZero(class_combines_amout, bin_amount_edge);   


      if (instance_index.size()>=3) 
      {
        single_vertex_flags[class_label][instance_id] = false;

        non_single_vertex_index[class_label].push_back(instance_id);   

        std::vector<int> id_1 = instance_index;
        id_1.erase(id_1.begin());                                  //delete picked mid-vertex from id_1
        for (int j=0; j<id_1.size()-1; ++j)                        //pick v^t
        {
          std::vector<int> id_2 = id_1;
          for (int ll=0; ll<j+1; ++ll)                             //delete picked third-vertex from id_2
          {
            id_2.erase(id_2.begin());
          }

          for (int k=0; k<id_2.size(); ++k)                       //pick v^{f}
          {
            first_vertex= ins_cens[index_2_label_id[id_2[k]]          .label][index_2_label_id[id_2[k]]          .id];
            mid_vertex  = ins_cens[index_2_label_id[instance_index[0]].label][index_2_label_id[instance_index[0]].id];
            third_vertex= ins_cens[index_2_label_id[id_1[j]]          .label][index_2_label_id[id_1[j]]          .id];

            //calculate relative angle
            double angle=get_angle(first_vertex.x, first_vertex.y,third_vertex.x, third_vertex.y, mid_vertex.x, mid_vertex.y);
            int row = class_combs_infos_[class_label].triplet_to_descriptor_bit[index_2_label_id[id_2[k]].label] [index_2_label_id[instance_index[0]].label] [index_2_label_id[id_1[j]].label];

            int col ;
            if (angle==180)
            {
              col = (int)angle/configs_.angle_resolution -1;
            }
            else
            {
              col = (int)angle/configs_.angle_resolution;
            }

            vertex_descriptor_angle(row, col) = vertex_descriptor_angle(row, col)+1;

            //calculate relative edge length
            double edge_ave = get_edge_length_ave(first_vertex.x, first_vertex.y,third_vertex.x, third_vertex.y, mid_vertex.x, mid_vertex.y);

            int col_len;
            if (edge_ave == configs_.edge_dis_thr)
            {
              col_len=(int)(edge_ave/configs_.dis_resolution)-1;
            }
            else
            {
              col_len=(int)(edge_ave/configs_.dis_resolution);
            }

            vertex_descriptor_len(row, col_len)=vertex_descriptor_len(row, col_len) +1;

            adj_mat(id_2[k],instance_index[0])=1;                                        
            adj_mat(id_1[j],instance_index[0])=1;

            adj_mat(instance_index[0], id_2[k])=1;                                        
            adj_mat(instance_index[0], id_1[j])=1;

            triplet_number = triplet_number + 1;
          }
        }

        non_single_vertex_count = non_single_vertex_count+1;
      }

      else
      {
        single_vertex_flags[class_label][instance_id] =  true;

        single_vertex_index[class_label].push_back(instance_id);                           

        single_vertex_count = single_vertex_count + 1;
      }


      //descriptor eigen format to vector format, angle related
      for (int row_des = 0; row_des < vertex_descriptor_angle.rows(); ++row_des)
      {
        for (int col_des = 0; col_des < vertex_descriptor_angle.cols(); ++col_des)
        {
          vertex_des_angle_vec[row_des * bin_amount + col_des] = vertex_descriptor_angle(row_des, col_des);
        }
      }
      descriptors_angle[class_label][instance_id] = vertex_descriptor_angle;
      descriptors_angle_vec[class_label][instance_id] = vertex_des_angle_vec;


      //edge length related
      for (int row_des = 0; row_des < vertex_descriptor_len.rows(); ++row_des)
      {
        for (int col_des = 0; col_des < vertex_descriptor_len.cols(); ++col_des)
        {
          vertex_des_len_vec[row_des * bin_amount_edge + col_des] = vertex_descriptor_len(row_des, col_des);
        }
      }
      descriptors_len[class_label][instance_id] = vertex_descriptor_len;
      descriptors_len_vec[class_label][instance_id] = vertex_des_len_vec;
      // std::cout<<"\033[32mThe total amount of triplet: \033[0m"<<triplet_number<<std::endl;

      //combine angle and length descriptor
      std::vector<int> vertex_des_comb;
      for (size_t i_1 = 0; i_1 < vertex_des_angle_vec.size(); ++i_1)
      {
        vertex_des_comb.push_back(vertex_des_angle_vec[i_1]);
      }
      for (size_t i_2 = 0; i_2 < vertex_des_len_vec.size(); ++i_2)
      {
        vertex_des_comb.push_back(vertex_des_len_vec[i_2]);
      }

      descriptors_comb[class_label][instance_id] = vertex_des_comb;
    }
  }

  else
  {
    for (size_t i = 0; i < ins_amount; ++i)
    {
      __int16_t class_label  = index_2_label_id[i].label;                           
      __int16_t instance_id = index_2_label_id[i].id;                               

      single_vertex_flags[class_label][instance_id] =  true;

      single_vertex_index[class_label].push_back(instance_id);                                
    }
  }

  current_non_single_vertex_num_ = non_single_vertex_count;
  current_single_vertex_num_ = single_vertex_count;

  output.adj_mat             = adj_mat;                //adjection matrix, for visualization only
  output.index_2_label_id    = index_2_label_id;
  output.triplet_number      = triplet_number;         
  output.single_vertex_flags = single_vertex_flags;       
  output.Descriptors_vec_angle   = descriptors_angle_vec;
  output.Descriptors_vec_length  = descriptors_len_vec;
  output.non_single_vertex_index = non_single_vertex_index;
  output.single_vertex_index     = single_vertex_index;

  output.Descriptors_vec_comb = descriptors_comb;
  return output;
}

std::vector<std::vector <std::string>> TripletLoc::load_txt_2_string_vector(std::string dir_txt)
{
  std::ifstream in(dir_txt);
  std::string line;
  std::vector<std::vector <std::string>> lines;

  if(in) // if the fiel exist
  {
    while (getline (in, line))
    {
      std::istringstream ss(line);
      std::string word;
      std::vector<std::string> single_line;
      while ( ss >> word)
      {
        single_line.push_back(word);
      }
      lines.push_back(single_line);
    }
  }
  in.close();

  return lines;
}

std::vector<float> TripletLoc::des_normalization(std::vector<int> input) 
{
  // Step 1: Convert std::vector<int> to Eigen::VectorXf
  Eigen::VectorXf eigenVec = Eigen::VectorXi::Map(input.data(), input.size()).cast<float>();

  // Step 2: Normalize the Eigen::VectorXf
  Eigen::VectorXf normalizedVec = eigenVec.normalized();

  // Step 3: Convert the Eigen::VectorXf back to std::vector<float>
  std::vector<float> output(normalizedVec.data(), normalizedVec.data() + normalizedVec.size());

  return output;
}

void TripletLoc::load_and_build_ins_map(std::string dir_global_map)
{
  //load instance centriods
  std::map<__int16_t,std::map<__int16_t, instance_centriod>> instance_cens;
  std::vector<__int16_t> used_classes = configs_.classes_for_graph;
  pcl::PointCloud<pcl::PointXYZRGB> ins_in_map;

  for (int i = 0; i < used_classes.size(); ++i)
  {
    std::string dir_ins_cen    = dir_global_map + "/" + std::to_string(used_classes[i])+".txt";
    std::vector<std::vector <std::string>> lines_1 = load_txt_2_string_vector(dir_ins_cen);

    int instance_id = 0;
    for (int row_ins = 0; row_ins < lines_1.size(); ++row_ins)
    {
      std::vector <std::string> one_pose = lines_1[row_ins];
      instance_centriod centroid;
      pcl::PointXYZRGB one_ins;

      for (int col_ins = 0; col_ins < one_pose.size(); ++col_ins)
      {
        std::stringstream s_ins;
        float coord;
        s_ins<<std::fixed<<std::setprecision(9)<<one_pose[col_ins];
        s_ins>>coord;


        if (col_ins==0)
        {
          instance_id = coord;
        }

        else if (col_ins==1)
        {
          centroid.x = coord;
        }
        else if (col_ins==2)
        {
          centroid.y = coord;
        }
        else if (col_ins==3)
        {
          centroid.z = coord;
        }
      }
      instance_cens[used_classes[i]][(__int16_t)instance_id] = centroid;

      one_ins.x = centroid.x;
      one_ins.y = centroid.y;
      one_ins.z = centroid.z;
      one_ins.r = configs_.semantic_name_rgb[used_classes[i]].color_r;
      one_ins.g = configs_.semantic_name_rgb[used_classes[i]].color_g;
      one_ins.b = configs_.semantic_name_rgb[used_classes[i]].color_b;
      ins_in_map.points.push_back(one_ins);
    }
  }

  //get descriptors for each vertex in the instance map
  TicToc get_des_t;
  std::cout<<"Building graph and extracting vertex descriptors for instance map ..."<<std::endl;
  Des_and_AdjMat des_adjmat_for_map = get_descriptor_fast(instance_cens);
  std::cout<<"\033[40;35m[Extract vertices descriptors in Global Map] consuming time: \033[0m"<<get_des_t.toc()<<"ms"<<std::endl;

  //get non-single vertex descriptors for all classes for building kd-tree
  std::map<__int16_t,std::vector< std::vector<float> > > des_vec_all_class_for_kd;  //normalized float
  std::map<__int16_t,std::vector<__int16_t >>::iterator non_single_vertex_index_iter;
  non_single_vertex_index_iter = des_adjmat_for_map.non_single_vertex_index.begin();

  std::map<__int16_t,std::map<__int16_t, std::vector<int> >> des_type;
  if (configs_.des_type==0)       //relative angle based (Des_{v_j}^{\alpha})
  {
    des_type = des_adjmat_for_map.Descriptors_vec_angle;
  }
  else if (configs_.des_type==1)  //edge length based (Des_{v_j}^{d})
  {
    des_type = des_adjmat_for_map.Descriptors_vec_length;
  }
  else if (configs_.des_type==2)  //combination (Des_{v_j})
  {
    des_type = des_adjmat_for_map.Descriptors_vec_comb;
  }

  int non_single_vertex_num =0;
  for (non_single_vertex_index_iter; non_single_vertex_index_iter != des_adjmat_for_map.non_single_vertex_index.end();  ++non_single_vertex_index_iter)
  {
    std::vector<__int16_t > non_single_indexs = non_single_vertex_index_iter->second;
    std::vector< std::vector<int> > des_vec_one_class;
    std::vector <std::vector<float>> des_vec_one_class_f_for_kd_tree;

    for (size_t i = 0; i < non_single_indexs.size(); ++i)
    {
      std::vector<int> des_vec =des_type[non_single_vertex_index_iter->first][non_single_indexs[i]];
      des_vec_one_class.push_back( des_vec  );
      non_single_vertex_num =non_single_vertex_num + 1;

      std::vector<float> des_normalized =  des_normalization(des_vec);
      des_vec_one_class_f_for_kd_tree.push_back(des_normalized);
    }
    des_vec_all_class_for_kd[non_single_vertex_index_iter->first] = des_vec_one_class_f_for_kd_tree; 
  }

  //single vertex
  std::map<__int16_t,std::vector<__int16_t >>::iterator single_vertex_index_iter;
  single_vertex_index_iter = des_adjmat_for_map.single_vertex_index.begin();
  int single_vertex_num =0;
  for (single_vertex_index_iter; single_vertex_index_iter != des_adjmat_for_map.single_vertex_index.end();  ++single_vertex_index_iter)
  {
    std::vector<__int16_t > single_indexs = single_vertex_index_iter->second;

    for (size_t i = 0; i < single_indexs.size(); ++i)
    {
      single_vertex_num =single_vertex_num + 1;
    }
  }

  std::cout<<"Vertex number in global map= "<<single_vertex_num+non_single_vertex_num<<" , non-single vertex number= "<<non_single_vertex_num<<std::endl;

  instance_cens_in_ins_map_ = instance_cens;

  //index_2_label_id_ins_map_ = des_adjmat_for_map.index_2_label_id;
  //adj_mat_in_ins_map_ = des_adjmat_for_map.adj_mat;

  //single_vertex_flags_in_ins_map_    = des_adjmat_for_map.single_vertex_flags;
  //single_vertex_index_in_ins_map_    = des_adjmat_for_map.single_vertex_index;
  non_single_vertex_index_in_ins_map_= des_adjmat_for_map.non_single_vertex_index;

  des_vec_all_class_f_for_kd_tree_ = des_vec_all_class_for_kd; //normaized float

  ros::Publisher ins_pub;
  ins_pub = nh_.advertise<sensor_msgs::PointCloud2>("instance_map", 1000);
  sensor_msgs::PointCloud2 ins_map;
  pcl::toROSMsg(ins_in_map,ins_map);
  ins_map.header.frame_id = "tripletloc"; 
  ins_map.header.stamp=ros::Time::now();    

  ros::Rate rate(1);

  int run=0;

  //publish instance map, we do not publish edges in graph since it is too dense (very slow and overflow)
  while (ros::ok())
  {
    ins_pub.publish(ins_map);

    rate.sleep();

    if (run>0)     //only show once
    {
      break;
    }
    run = run + 1;
  }

}

objects_and_pointclouds TripletLoc::get_ins_cen(std::string bin_flie, std::string label_file)
{
  objects_and_pointclouds output;

  //TicToc load_pc;
  std::map<__int16_t, pcl::PointCloud<pcl::PointXYZRGB>> pc_rgb =  get_pointcloud_with_semantic_helipr(bin_flie, label_file);
  //std::cout<<"Load pc : "<<load_pc.toc()<<"ms"<<std::endl;

  TicToc instance_cluster_t;
  //downsample and cluster
  std::map<__int16_t, pcl::PointCloud<pcl::PointXYZRGB>>::iterator iter;
  iter = pc_rgb.begin();

  float leaf_size = configs_.voxel_leaf_size_que;

  std::map<__int16_t,std::map<__int16_t, instance_centriod>> instance_cens;
  std::map<__int16_t, int> instance_numbers;


  std::vector<__int16_t> class_for_graph = configs_.classes_for_graph;
  std::vector<__int16_t>::iterator t;

  pcl::PointCloud<pcl::PointXYZ> instances_xyz_;
  std::vector<int> each_instance_point_num_;
  std::vector<__int16_t> each_instance_label_;

  //TicToc cluster_first_t;
  for (iter; iter != pc_rgb.end(); ++iter)    //for each class
  {
    t = find(class_for_graph.begin(),class_for_graph.end(),iter->first);

    if(t != class_for_graph.end())
    {
      //downsample
      pcl::PointCloud<pcl::PointXYZRGB> filtered_pc; //= iter->second;
      pcl::VoxelGrid<pcl::PointXYZRGB> sor;
      sor.setInputCloud(iter->second.makeShared());                            
      sor.setLeafSize(leaf_size, leaf_size, leaf_size);              
      sor.filter(filtered_pc);

      //cluster
      pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);   
      tree->setInputCloud(filtered_pc.makeShared());                                               
      std::vector<pcl::PointIndices> cluster_indices;                                                
      pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;                                          

      float cluster_tolerance = configs_.EuCluster_para[iter->first];
      ec.setClusterTolerance (cluster_tolerance);                                                      
      ec.setMinClusterSize (20);
      ec.setMaxClusterSize ( filtered_pc.size() );
      ec.setSearchMethod (tree);
      ec.setInputCloud ( filtered_pc.makeShared());
      ec.extract (cluster_indices);

      //cluster result saving
      std::vector<pcl::PointIndices>::const_iterator it;
      it = cluster_indices.begin();

      __int16_t new_id=0;
      for (it; it!=cluster_indices.end();++it)
      {
        int minimun_points_amount = configs_.minimun_point_in_one_instance[iter->first];
        int cloud_cluster_point_num = it->indices.size();

        if(cloud_cluster_point_num >= minimun_points_amount)    //filter out small clusters
        {
          float sum_x = 0.0;
          float sum_y = 0.0;
          float sum_z = 0.0;

          for (size_t k = 0; k < it->indices.size(); ++k)
          {
            sum_x = sum_x + filtered_pc[it->indices[k]].x;
            sum_y = sum_y + filtered_pc[it->indices[k]].y;
            sum_z = sum_z + filtered_pc[it->indices[k]].z;
          }

          pcl::PointXYZ one_centriod;
          instance_centriod centriod;
          centriod.x = sum_x/(float)cloud_cluster_point_num;
          centriod.y = sum_y/(float)cloud_cluster_point_num;
          centriod.z = sum_z/(float)cloud_cluster_point_num;

          one_centriod.x = centriod.x;
          one_centriod.y = centriod.y;
          one_centriod.z = centriod.z;
          instances_xyz_.push_back(one_centriod);

          each_instance_point_num_.push_back(cloud_cluster_point_num);
          each_instance_label_.push_back(iter->first);
        }
      }
    }
  }

  //instances fusion, fusing instances that are close to each other
  std::map<__int16_t, std::vector<instance_centriod>> fused_ins_tmp;

  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);   
  tree->setInputCloud(instances_xyz_.makeShared());                                              
  std::vector<pcl::PointIndices> cluster_indices;                                           
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;                                     

  float cluster_tolerance = configs_.ins_fuse_radius;
  ec.setClusterTolerance (cluster_tolerance);                                                  
  ec.setMinClusterSize (1);
  ec.setMaxClusterSize ( instances_xyz_.size() );
  ec.setSearchMethod (tree);
  ec.setInputCloud ( instances_xyz_.makeShared());
  ec.extract (cluster_indices);

  //cluster result saving
  std::vector<pcl::PointIndices>::const_iterator it;
  it = cluster_indices.begin();

  __int16_t new_id=0;

  for (it; it!=cluster_indices.end();++it)
  {
    int pc_cluster_point_num = it->indices.size();

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

      if (each_instance_point_num_[ ins_index_in_tree ] > max_point_num)
      {
        max_point_num = each_instance_point_num_[ ins_index_in_tree ];
        max_pc_num_id = ins_index_in_tree;
      }
    }

    instance_centriod centriod;
    centriod.x = sum_x/(float)pc_cluster_point_num;
    centriod.y = sum_y/(float)pc_cluster_point_num;
    centriod.z = sum_z/(float)pc_cluster_point_num;


    //decide the label for the fused instance
    __int16_t label_current_ins = each_instance_label_[max_pc_num_id];
    fused_ins_tmp[label_current_ins].push_back(centriod);
  }


  int total_instance_number = 0;  

  std::map<__int16_t, std::map<__int16_t,instance_centriod>> fused_ins;
  std::map<__int16_t, std::vector<instance_centriod>>::iterator iter_fused;
  iter_fused = fused_ins_tmp.begin();

  for (iter_fused; iter_fused != fused_ins_tmp.end(); ++iter_fused)
  {
    std::cout<<"Instance number for class-"<<configs_.semantic_name_rgb[iter_fused->first].obj_class<<" : "<<iter_fused->second.size()<<std::endl;

    instance_numbers[iter_fused->first]=iter_fused->second.size();

    for (int i = 0; i < iter_fused->second.size(); ++i)
    {
      fused_ins[iter_fused->first][i] = iter_fused->second[i];
      total_instance_number = total_instance_number +1;
    }

  }

  instance_numbers[-1] = total_instance_number;                

  std::cout<<"Instance number in total: "<<total_instance_number<<std::endl;

  output.pc   = pc_rgb;
  output.instance_cens    = fused_ins;
  output.instance_numbers = instance_numbers;

  instance_cluster_t_ = instance_cluster_t.toc();

  return output;
}

matches_info TripletLoc::knn_search_kd(__int16_t class_label, std::vector<int> query_Des) 
{
  matches_info output;

  std::vector<float> query_Des_f = des_normalization(query_Des); 
  std::vector<size_t> result_indexes(configs_.knn_match_number);
  std::vector<float> out_dists_sqr(configs_.knn_match_number);
  nanoflann::KNNResultSet<float> resultSet(configs_.knn_match_number);
  resultSet.init(&result_indexes[0], &out_dists_sqr[0]);            

  if (class_label == 13)
  {
    if (kd_tree_fence_ready_)
    {
      output.invalid = false;
      kd_tree_fence_->index->findNeighbors(resultSet, &query_Des_f[0],nanoflann::SearchParams(10));
    }
    else
    {
      output.invalid = true;         
    }
  }
  else if (class_label == 14)
  {
    if (kd_tree_vegetation_ready_)
    {
      output.invalid = false;
      kd_tree_vegetation_->index->findNeighbors(resultSet, &query_Des_f[0],nanoflann::SearchParams(10));
    }
    else
    {
      output.invalid = true;         
    }
  }
  else if (class_label == 15)
  {
    if (kd_tree_trunk_ready_)
    {
      output.invalid = false;
      kd_tree_trunk_->index->findNeighbors(resultSet, &query_Des_f[0],nanoflann::SearchParams(10));
    }
    else
    {
      output.invalid = true;         
    }
  }
  else if (class_label == 17)
  {
    if (kd_tree_pole_ready_)
    {
      output.invalid = false;
      kd_tree_pole_->index->findNeighbors(resultSet, &query_Des_f[0],nanoflann::SearchParams(10));
    }
    else
    {
      output.invalid = true;         
    }
  }
  else if (class_label == 18)
  {
    if (kd_tree_traffic_sign_ready_)
    {
      output.invalid = false;
      kd_tree_traffic_sign_->index->findNeighbors(resultSet, &query_Des_f[0],nanoflann::SearchParams(10));
    }
    else
    {
      output.invalid = true;         
    }
  }

  int size = result_indexes.size();
  int match_num = std::min(size, configs_.knn_match_number);

  std::vector<size_t> matches_index;    
  std::vector<float> matches_sims;

  for (size_t i = 0; i < match_num; ++i)
  {
    matches_index.push_back(result_indexes[i]);
    matches_sims.push_back(out_dists_sqr[i]);
  }

  output.matches_index    = matches_index;     
  output.matches_sims     = matches_sims;
  output.max_sim          = out_dists_sqr[0];  

  return output;
}


std::map<__int16_t,std::map<__int16_t, matches_info>> TripletLoc::vertex_match(Des_and_AdjMat query_vertex_des)
{
  std::map<__int16_t,std::map<__int16_t, matches_info>> matche_results;

  std::map<__int16_t,std::vector<__int16_t> >::iterator iter;
  iter = query_vertex_des.non_single_vertex_index.begin();

  //#pragma omp parallel for
  for (iter; iter != query_vertex_des.non_single_vertex_index.end(); ++iter) //for each class
  {
    std::vector<__int16_t> non_single_query_vertex_id = iter->second;      

    //#pragma omp parallel for
    for (size_t i = 0; i < non_single_query_vertex_id.size(); ++i)
    {
      __int16_t vertex_id = non_single_query_vertex_id[i];                 

      std::vector<int>query;
      if (configs_.des_type==0)            //relative angle based (Des_{v_j}^{\alpha})
      {
        query = query_vertex_des.Descriptors_vec_angle[iter->first][vertex_id];
      }
      else if (configs_.des_type==1)       //edge length based (Des_{v_j}^{d})
      {
        query = query_vertex_des.Descriptors_vec_length[iter->first][vertex_id];
      }      
      else if (configs_.des_type==2)       //combination (Des_{v_j})
      { 
        query = query_vertex_des.Descriptors_vec_comb[iter->first][vertex_id];
      }

      matches_info matches_for_one_vertex = knn_search_kd(iter->first, query); 

      matche_results[iter->first][vertex_id] = matches_for_one_vertex;
    }
  }

  //for single-vertex, TODO in furture version
  std::map<__int16_t,std::vector<__int16_t> >::iterator iter1;
  iter1 = query_vertex_des.single_vertex_index.begin();

  for (iter1; iter1 != query_vertex_des.single_vertex_index.end(); ++iter1)
  {
    std::vector<__int16_t> single_query_vertex_id = iter1->second;

    for (size_t i = 0; i < single_query_vertex_id.size(); ++i)
    {
      matches_info matches_for_one_vertex;
      __int16_t vertex_id = single_query_vertex_id[i];

      matches_for_one_vertex.invalid = true;

      //TODO in furture version

      matche_results[iter1->first][vertex_id] = matches_for_one_vertex;
    }

  }

  return matche_results;
}


void TripletLoc::run()
{
  std::string save_dir;
  nh_.getParam("save_dir",save_dir);

  bool by_step = false;
  nh_.getParam("by_step",by_step);

  std::string time_record_txt_dir = save_dir +"run_time_breakdown.txt";  //[instance_cluster] [get_des] [vertex_matching] [mcq_search] [pose_solve] [one_loc]  

  std::string ret_rre_p_txt_dir = save_dir +"RTE_RRE_P.txt";             //[RTE] [RRE]

  std::ofstream time_record, ret_rre_p_record;
  time_record.open(time_record_txt_dir);
  ret_rre_p_record.open(ret_rre_p_txt_dir);

  omp_set_num_threads(4);

  //load ground truth for the qurey sequence poses
  gt_poses_que_ = get_pose_gt_Helipr(configs_.pose_gt_file_que); 

  //load RSN map
  if (configs_.use_rsn && (configs_.solver_type==0))
  {
    std::vector<std::vector <std::string>> road_grids = load_txt_2_string_vector(configs_.rsn_map_txts_path);
    pcl::PointCloud<pcl::PointXY>::Ptr road_grid_cen_xy(new pcl::PointCloud<pcl::PointXY>);
    for (size_t i = 0; i < road_grids.size(); i++)
    {
      pcl::PointXY one_grid;
      one_grid.x = std::stof(road_grids[i][0]);
      one_grid.y = std::stof(road_grids[i][1]);
      road_grid_cen_xy->points.push_back(one_grid);

      Eigen::Vector3d normal;
      normal << std::stof(road_grids[i][2]), std::stof(road_grids[i][3]), std::stof(road_grids[i][4]);
      road_grid_normal_.push_back(normal);

      double grid_normal_std = std::stof(road_grids[i][5]);
      road_grid_normal_std_.push_back(grid_normal_std);
    }
    kdtree_road_grid_cen_.setInputCloud(road_grid_cen_xy);
  }

  //load and build instance-level map, and extract descriptors for each vertex
  load_and_build_ins_map(configs_.ins_map_txts_path);

  //initial the kd-trees for vertex search
  int dim = des_vec_all_class_f_for_kd_tree_.begin()->second[0].size();
  if (des_vec_all_class_f_for_kd_tree_.find(13) != des_vec_all_class_f_for_kd_tree_.end())
  {
    kd_tree_fence_.reset();
    kd_tree_fence_ = std::make_unique<KDTreeVectorOfVectorsAdaptor<std::vector<std::vector<float>>, float>>(dim, des_vec_all_class_f_for_kd_tree_[13], 10);
    std::cout << "[KD-Tree] Constructed for fence, size=" << kd_tree_fence_->kdtree_get_point_count() << std::endl;
    kd_tree_fence_ready_ = true;
  }
  if (des_vec_all_class_f_for_kd_tree_.find(14) != des_vec_all_class_f_for_kd_tree_.end())
  {
    kd_tree_vegetation_.reset();
    kd_tree_vegetation_ = std::make_unique<KDTreeVectorOfVectorsAdaptor<std::vector<std::vector<float>>, float>>(dim, des_vec_all_class_f_for_kd_tree_[14], 10);
    std::cout << "[KD-Tree] Constructed for vegetation, size=" << kd_tree_vegetation_->kdtree_get_point_count() << std::endl;
    kd_tree_vegetation_ready_ = true;
  }
  if (des_vec_all_class_f_for_kd_tree_.find(15) != des_vec_all_class_f_for_kd_tree_.end())
  {
    kd_tree_trunk_.reset();
    kd_tree_trunk_ = std::make_unique<KDTreeVectorOfVectorsAdaptor<std::vector<std::vector<float>>, float>>(dim, des_vec_all_class_f_for_kd_tree_[15], 10);
    std::cout << "[KD-Tree] Constructed for trunk, size=" << kd_tree_trunk_->kdtree_get_point_count() << std::endl;
    kd_tree_trunk_ready_ = true;
  }
  if (des_vec_all_class_f_for_kd_tree_.find(17) != des_vec_all_class_f_for_kd_tree_.end())
  {
    kd_tree_pole_.reset();
    kd_tree_pole_ = std::make_unique<KDTreeVectorOfVectorsAdaptor<std::vector<std::vector<float>>, float>>(dim, des_vec_all_class_f_for_kd_tree_[17], 10);
    std::cout << "[KD-Tree] Constructed for pole, size=" << kd_tree_pole_->kdtree_get_point_count() << std::endl;
    kd_tree_pole_ready_ = true;
  }
  if (des_vec_all_class_f_for_kd_tree_.find(18) != des_vec_all_class_f_for_kd_tree_.end())
  {
    kd_tree_traffic_sign_.reset();
    kd_tree_traffic_sign_ = std::make_unique<KDTreeVectorOfVectorsAdaptor<std::vector<std::vector<float>>, float>>(dim, des_vec_all_class_f_for_kd_tree_[18], 10);
    std::cout << "[KD-Tree] Constructed for traffic-sign, size=" << kd_tree_traffic_sign_->kdtree_get_point_count() << std::endl;
    kd_tree_traffic_sign_ready_ = true;
  }


  //main process
  int frame_number_for_query = 0;
  int success_loc_frame = 0;

  Eigen::Matrix4d last_pose;
  last_pose << 1, 0, 0, 0,
               0, 1, 0, 0,
               0, 0, 1, 0,
               0, 0, 0, 1;

  ros::Publisher edges_pub, instances_pub, correct_match_pub, match_wrong_pub, ins_cen_est_pub, inlier_association_pub, local_area_pub, query_pc_pub, query_pc_pub_est;
  edges_pub   = nh_.advertise<visualization_msgs::MarkerArray>("que_edges", 1000);              //transformed graph edges using gt pose for visualization
  instances_pub = nh_.advertise<visualization_msgs::MarkerArray>("que_instances_gt", 1000);     //transformed query instance using gt pose with offest for visualization
  correct_match_pub   = nh_.advertise<visualization_msgs::MarkerArray>("correct_matches", 1000);//correct matches
  match_wrong_pub = nh_.advertise<visualization_msgs::MarkerArray>("wrong_matches", 1000);      //wrong matches
  ins_cen_est_pub = nh_.advertise<visualization_msgs::MarkerArray>("que_instance_est", 1000);   //transformed query instance using estimated pose, with offest for visualization
  inlier_association_pub = nh_.advertise<visualization_msgs::MarkerArray>("max_clique", 1000);  //associations from max clique
  local_area_pub = nh_.advertise<visualization_msgs::Marker>("local_area", 1000);               //position and area of the current frame in the instance map
  query_pc_pub = nh_.advertise<sensor_msgs::PointCloud2>("que_pc", 1000);                       //transformed query pc using gt pose for visualization, with offest for visualization
  query_pc_pub_est = nh_.advertise<sensor_msgs::PointCloud2>("que_pc_est", 1000);               //transformed query pc using estimated pose 

  //for result statistics
  int total_match_num = 0;
  int total_correct_match_num = 0;

  std::vector<int> vertex_num;                            //record the number of vertices in each query frame
  std::vector<int> non_single_vertex_num;                 //record the number of non-single vertices in each query frame
  std::vector<int> acceptable_matched_n_sing_vertex_num;  //record the number of acceptable matched non-single vertices in each query frame
  std::vector<double> rres;                               //record the rre of each successfully located query frame
  std::vector<double> rtes;                               //record the rte of each successfully located query frame
  std::vector<double> one_loc_times;                      //record the time of each query frame
  std::vector<int> association_num;                       //record the number of associations in each query frame
  std::vector<int> max_clique_num;                        //record the size of max clique in each query frame

  for (int i = 0; i < gt_poses_que_.size(); ++i)
  {
    Eigen::Matrix4d current_pose = gt_poses_que_[i].second;
    float d_x = current_pose(0,3) - last_pose(0,3);
    float d_y = current_pose(1,3) - last_pose(1,3);
    float d_z = current_pose(2,3) - last_pose(2,3);
    float dis = sqrt( pow(d_x,2) + pow(d_y,2) + pow(d_z,2) );

    if ( (dis>= configs_.frames_interval_ques) || (i==0))
    {
      query_frame_count_ = i;

      std::cout<<"\033[33;1m**************Current  query frame: \033[0m"<<i<<" out of "<<gt_poses_que_.size()<<"\033[33;1m**********************\033[0m"<<std::endl;

      TicToc one_shot_t;

      //load pointcloud and label
      std::string bin_dir   = configs_.cloud_path_que+ "/"+gt_poses_que_[i].first +".bin";
      std::string label_dir = configs_.label_path_que + "/"+gt_poses_que_[i].first +".label";

      //extract instances
      TicToc get_ins_t;
      objects_and_pointclouds  ins_cens_and_pc =  get_ins_cen(bin_dir, label_dir); 
      double get_ins_time = get_ins_t.toc();
      double load_pc_time = get_ins_time - instance_cluster_t_;
      std::cout<<"\033[40;35m[Load pc & label] consuming time: \033[0m"<<load_pc_time<<"ms"<<std::endl;
      std::cout<<"\033[40;35m[Extract instances] consuming time: \033[0m"<<instance_cluster_t_<<"ms"<<std::endl;

      //extract vertices descriptors
      TicToc one_locate_t;
      TicToc get_des_fast_t;
      Des_and_AdjMat des_adjmat = get_descriptor_fast(ins_cens_and_pc.instance_cens);
      double get_des_time = get_des_fast_t.toc();
      std::cout<<"\033[40;35m[Extract vertices descriptors fast] consuming time: \033[0m"<<get_des_time<<"ms"<<std::endl;

      //vertex matching
      TicToc vertex_matching_t;
      std::map<__int16_t,std::map<__int16_t, matches_info>> matche_results = vertex_match(des_adjmat);
      double vertex_matching_time = vertex_matching_t.toc();
      std::cout<<"\033[40;35m[Vertex matching] consuming time: \033[0m"<<vertex_matching_time<<"ms"<<std::endl;

      //pose estimation
      TicToc solve_t;
      pose_with_clique solution;
      if (configs_.solver_type==1)        //teaser, point-to-point
      {
        solution = pose_estimate_teaser(matche_results, ins_cens_and_pc.instance_cens);
      }
      else if (configs_.solver_type==0)  //gtsam, with/without RSN
      {
        solution = pose_estimate_gtsam(matche_results, ins_cens_and_pc.instance_cens);
      }
      double sovle_time = solve_t.toc();

      //print
      std::cout<<"\033[40;35m[Mcq solve] consuming time: \033[0m"<<solution.mcq_time<<"ms"<<std::endl;
      std::cout<<"\033[40;35m[Pose solve] consuming time: \033[0m"<<solution.pose_time<<"ms"<<std::endl;
      std::cout<<"\033[40;35m[Pose estimation] consuming time: \033[0m"<<sovle_time<<"ms"<<std::endl;
      std::cout<<"\033[34;1m[One shot localization] consuming time: \033[0m"<<one_shot_t.toc()-load_pc_time<<"ms"<<std::endl;
      double one_loc_time = one_locate_t.toc() + instance_cluster_t_;
      std::cout<<"Maximun Clique size= "<<current_max_clique_num_<<std::endl;

      one_loc_times.push_back(one_loc_time);
      association_num.push_back(current_association_num_);
      max_clique_num.push_back(current_max_clique_num_);

      //record time of each step as a txt file
      time_record<<std::fixed<<std::setprecision(9)<<instance_cluster_t_<<" "<<get_des_time<<" "<<vertex_matching_time<<" "<<solution.mcq_time<<" "<<solution.pose_time<<" "<<one_loc_time<<std::endl;  

      //publish query scan, based on ground truth with a z-offest
      Eigen::Matrix4d trans_pose_que_pc = current_pose;
      trans_pose_que_pc(2,3) =trans_pose_que_pc(2,3) + z_offest_;
      pcl::PointCloud<pcl::PointXYZRGB> transformed_pc;
      pcl::transformPointCloud(ins_cens_and_pc.pc[-1], transformed_pc, trans_pose_que_pc);
      sensor_msgs::PointCloud2 pc_query;
      pcl::toROSMsg(transformed_pc,pc_query);
      pc_query.header.frame_id = "tripletloc"; 
      pc_query.header.stamp=ros::Time::now();    
      query_pc_pub.publish(pc_query);

      //publish transformed query scan, based on the estimation pose
      pcl::PointCloud<pcl::PointXYZRGB> transformed_pc_est;
      pcl::transformPointCloud(ins_cens_and_pc.pc[-1], transformed_pc_est, solution.pose);
      sensor_msgs::PointCloud2 pc_query_est;
      pcl::toROSMsg(transformed_pc_est,pc_query_est);
      pc_query_est.header.frame_id = "tripletloc";               
      pc_query_est.header.stamp=ros::Time::now();      
      query_pc_pub_est.publish(pc_query_est);

      //calculate RTE and RRE
      double RTE =  (current_pose.topRightCorner(3, 1) - solution.pose.topRightCorner(3, 1)).norm();
      Eigen::Matrix3d est_R = solution.pose.block<3,3>(0,0);
      Eigen::Matrix3d gt_R  = current_pose.block<3,3>(0,0);
      double a = ((est_R * gt_R.inverse()).trace() - 1) * 0.5;
      double aa= std::max(std::min(a,1.0), -1.0);
      double RRE = acos(aa)*180/M_PI;

      //record time of each query frame as a txt file
      ret_rre_p_record<<std::fixed<<std::setprecision(6)<<RTE<<" "<<RRE<<std::endl;

      //visualize the position of the current frame in the reference map, green: success; red: fail
      visualization_msgs::Marker current_area;

      if (RTE<configs_.RTE_thr && RRE<configs_.RRE_thr && solution.valid)
      {
        std::cout<<"\033[36;1m ****Success Localization \033[0m"<< "RTE="<< RTE<<", RRE="<<RRE <<std::endl;
        success_loc_frame = success_loc_frame +1;
        rres.push_back(RRE);
        rtes.push_back(RTE);

        current_area = local_area_visual(true);
      }
      else
      {
        std::cout<<"\033[31;1m ****Faile Localization \033[0m"<< "RTE="<< RTE<<", RRE="<<RRE <<std::endl;
        current_area = local_area_visual(false);
      }

      //beyond paper, calculate the recall@N of the vertex matching result for current query frame (also used for matching visualization)
      std::map<__int16_t, std::pair<int,int>> match_nums = vertex_match_recall_N(matche_results, ins_cens_and_pc.instance_cens);
      float recall_N = (float)match_nums[0].second/(match_nums[0].first + 1e-15) ;
      std::cout<<"Current acceptable vertex matches: "<<match_nums[0].second<<" out of "<<match_nums[0].first<<" Recall@N= "<<recall_N<<std::endl;

      //beyond paper, calculate the accumulated recall@N of the vertex matches so far
      total_match_num = total_match_num + match_nums[0].first;
      total_correct_match_num =  total_correct_match_num + match_nums[0].second;
      float recall_N_accum = (float)total_correct_match_num/total_match_num;
      std::cout<<"Accumulative acceptable vertex matches: "<<total_correct_match_num<<" out of "<<total_match_num<<" Recall@N= "<<recall_N_accum<<std::endl;

      //delete the last visualization
      std::vector<visualization_msgs::MarkerArray>  ins_dete = clean_last_visualization();
      instances_pub.publish(ins_dete[0]);
      edges_pub.publish(ins_dete[1]);
      correct_match_pub.publish(ins_dete[2]);
      match_wrong_pub.publish(ins_dete[3]);
      ins_cen_est_pub.publish(ins_dete[4]);
      inlier_association_pub.publish(ins_dete[5]);

      //publish instances and edges in current frame
      visualization_msgs::MarkerArray ins_show;
      visualization_msgs::MarkerArray edges;
      ins_show = instances_visual(ins_cens_and_pc.instance_cens);
      edges = edges_visual(ins_cens_and_pc.instance_cens, des_adjmat.adj_mat, des_adjmat.index_2_label_id, 1);

      edges_pub.publish(edges);
      instances_pub.publish(ins_show);
      local_area_pub.publish(current_area);

      //publish associations
      if (match_nums[0].first>0 )
      {
        std::pair<visualization_msgs::MarkerArray, visualization_msgs::MarkerArray> match_show =vertex_matches_visual(matche_results, ins_cens_and_pc.instance_cens);
        correct_match_pub.publish(match_show.first);        //acceptable matches
        match_wrong_pub.publish(match_show.second);         //unacceptable matches
      }

      //publish vertices transformed by the estimated pose
      visualization_msgs::MarkerArray que_vertices_with_est_pose = vertex_using_est_pose_visual(solution,ins_cens_and_pc.instance_cens);
      ins_cen_est_pub.publish(que_vertices_with_est_pose); 

      //publish association from max clique
      visualization_msgs::MarkerArray mcq_associations = MCQ_visual(solution.max_clique,ins_cens_and_pc.instance_cens);
      inlier_association_pub.publish(mcq_associations); 

      last_pose = current_pose;

      frame_number_for_query = frame_number_for_query + 1;

      //for statistics
      int vertex_num_cur = current_single_vertex_num_ + current_non_single_vertex_num_;
      vertex_num .push_back(vertex_num_cur);
      non_single_vertex_num.push_back(current_non_single_vertex_num_);
      acceptable_matched_n_sing_vertex_num.push_back(match_nums[0].second);

      //pause for each query frame or not
      if (by_step)
      {
        std::cout<<"Press any key to continue..."<<std::endl;
        getchar();
      }
    }
  }

  //calculate success rate and print
  std::cout<<std::endl;
  double success_rate = (double)success_loc_frame/frame_number_for_query;
  std::cout<<std::fixed<<std::setprecision(6)<<"\033[32;1m**Total global localization rate** :\033[0m"<< (double)success_loc_frame/frame_number_for_query<<" , "<<success_loc_frame<<" out of "<<frame_number_for_query<<std::endl;

  //statistics beyond paper
  Eigen::VectorXi vertex_num_E = Eigen::Map<Eigen::VectorXi, Eigen::Aligned>(vertex_num.data(), vertex_num.size());
  Eigen::VectorXi non_single_vertex_num_E = Eigen::Map<Eigen::VectorXi, Eigen::Aligned>(non_single_vertex_num.data(), non_single_vertex_num.size());
  Eigen::VectorXi correct_n_single_vertex_num_E = Eigen::Map<Eigen::VectorXi, Eigen::Aligned>(acceptable_matched_n_sing_vertex_num.data(), acceptable_matched_n_sing_vertex_num.size());
  Eigen::VectorXi association_num_E = Eigen::Map<Eigen::VectorXi, Eigen::Aligned>(association_num.data(), association_num.size());
  Eigen::VectorXi max_cliaue_num_E = Eigen::Map<Eigen::VectorXi, Eigen::Aligned>(max_clique_num.data(), max_clique_num.size());

  //statistics for RTE, RRE, and one loc time
  Eigen::VectorXd rre_E = Eigen::Map<Eigen::VectorXd, Eigen::Aligned>(rres.data(), rres.size());
  Eigen::VectorXd rte_E = Eigen::Map<Eigen::VectorXd, Eigen::Aligned>(rtes.data(), rtes.size());
  Eigen::VectorXd one_loc_time_E = Eigen::Map<Eigen::VectorXd, Eigen::Aligned>(one_loc_times.data(), one_loc_times.size());

  //calculate the average and std
  int vertex_ave = vertex_num_E.mean();
  double vertex_std = std::sqrt((vertex_num_E.array() - vertex_ave).square().sum() / vertex_num_E.size());

  int non_single_vertex_ave = non_single_vertex_num_E.mean();
  double non_single_vertex_std = std::sqrt((non_single_vertex_num_E.array() - non_single_vertex_ave).square().sum() / non_single_vertex_num_E.size());

  int correct_non_single_vertex_ave = correct_n_single_vertex_num_E.mean();
  double correct_non_single_vertex_std = std::sqrt((correct_n_single_vertex_num_E.array() - correct_non_single_vertex_ave).square().sum() / correct_n_single_vertex_num_E.size());

  int association_ave = association_num_E.mean();
  double association_std = std::sqrt((association_num_E.array() - association_ave).square().sum() / association_num_E.size());

  int max_cliaue_ave = max_cliaue_num_E.mean();
  double max_cliaue_std = std::sqrt((max_cliaue_num_E.array() - max_cliaue_ave).square().sum() / max_cliaue_num_E.size());

  double rre_ave = rre_E.mean();
  double rre_std = std::sqrt((rre_E.array() - rre_ave).square().sum() / rre_E.size());

  double rte_ave = rte_E.mean();
  double rte_std = std::sqrt((rte_E.array() - rte_ave).square().sum() / rte_E.size());

  double pose_solve_time_ave = one_loc_time_E.mean();
  double pose_solve_time_std = std::sqrt((one_loc_time_E.array() - pose_solve_time_ave).square().sum() / one_loc_time_E.size());

  
  std::cout<<"Vertex num= "<<vertex_ave<<"±"<<vertex_std<<std::endl;
  std::cout<<"Non single vertex num= "<<non_single_vertex_ave<<"±"<<non_single_vertex_std<<std::endl;
  std::cout<<"Corrext matched non single vertex num= "<<correct_non_single_vertex_ave<<"±"<<correct_non_single_vertex_std<<std::endl;
  std::cout<<"Asscoiation num= "<<association_ave<<"±"<<association_std<<std::endl;
  std::cout<<"Max clique num= "<<max_cliaue_ave<<"±"<<max_cliaue_std<<std::endl;

  std::cout<<std::fixed<<std::setprecision(3)<<"RTE= "<<rte_ave<<"±"<<rte_std<<std::endl;
  std::cout<<std::fixed<<std::setprecision(3)<<"RRE= "<<rre_ave<<"±"<<rre_std<<std::endl;
  std::cout<<std::fixed<<std::setprecision(3)<<"One loc time= "<<pose_solve_time_ave<<"±"<<pose_solve_time_std<<std::endl;

  //record the rte_thr, rre_thr, success rate, RTE, RRE, and one-loc time as a txt file
  ret_rre_p_record<<"~~~~~~~~~~~~~~~~~~~~~~~~~~"<<std::endl;
  ret_rre_p_record<<"RTE_thr="<<configs_.RTE_thr<<", RRE_thr="<<configs_.RRE_thr<<std::endl;
  ret_rre_p_record<<"Total global localization rate: "<< success_rate<<" , "<<success_loc_frame<<" out of "<<frame_number_for_query<<std::endl;
  ret_rre_p_record<<std::fixed<<std::setprecision(6)<<"RTE= "<<rte_ave<<"±"<<rte_std<<std::endl;
  ret_rre_p_record<<std::fixed<<std::setprecision(6)<<"RRE= "<<rre_ave<<"±"<<rre_std<<std::endl;
  ret_rre_p_record<<std::fixed<<std::setprecision(6)<<"Pose solve time= "<<pose_solve_time_ave<<"±"<<pose_solve_time_std<<std::endl;

  ret_rre_p_record.close();
}

pose_with_clique TripletLoc::pose_estimate_teaser(std::map<__int16_t,std::map<__int16_t, matches_info>> matches, std::map<__int16_t,std::map<__int16_t,instance_centriod>> ins_cen)
{
  pose_with_clique output;
  output.valid = false;   //initial the status of the solution

  //set values
  std::vector<instance_centriod> query_vertices;
  std::vector<instance_centriod> match_vertices;

  int association_num = 0;

  std::map<__int16_t,std::map<__int16_t, matches_info>>::iterator iter;
  iter = matches.begin();

  for (iter; iter != matches.end(); ++iter)
  {
    std::map<__int16_t, matches_info>::iterator iter1;
    iter1 = iter->second.begin();

    for (iter1; iter1 != iter->second.end(); ++iter1)
    {
      if (!(iter1->second.invalid))
      {
        instance_centriod query_vertex = ins_cen[iter->first][iter1->first];
        matches_info one_match = iter1->second;
        int top_k_match = one_match.matches_index.size();

        for (size_t i = 0; i < top_k_match; ++i)
        {
          int match_vertex_id_in_kd_tree = one_match.matches_index[i];

          __int16_t vertex_id_in_instance_center_list =  non_single_vertex_index_in_ins_map_[iter->first][match_vertex_id_in_kd_tree];

          instance_centriod match_point = instance_cens_in_ins_map_[iter->first][vertex_id_in_instance_center_list];   

          query_vertices.push_back(query_vertex);
          match_vertices.push_back(match_point);

          association_num = association_num + 1;
        }
      }
    }
  }

  if (association_num>=3) //at least 3 associations
  {

    Eigen::Matrix<double, 3, Eigen::Dynamic> local_vertices(3, association_num);
    Eigen::Matrix<double, 3, Eigen::Dynamic> global_vertices(3, association_num);

    for (size_t i = 0; i < association_num; ++i)
    {
      local_vertices.col(i) << query_vertices[i].x, query_vertices[i].y, query_vertices[i].z;
      global_vertices.col(i) << match_vertices[i].x, match_vertices[i].y, match_vertices[i].z;
    }

    // Run TEASER++ registration
    // Prepare solver parameters
    teaser::RobustRegistrationSolver::Params params;
    params.kcore_heuristic_threshold = 1.0;            //always use max_clique searching
    params.estimate_scaling = false;                   //no scaling in LiDAR-based localization
    params.noise_bound = configs_.noise_bound;       
    params.cbar2 = 1;
    params.rotation_max_iterations = 100;
    params.rotation_gnc_factor = 1.4;
    params.rotation_estimation_algorithm =teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
    params.rotation_cost_threshold = 0.005;

    // Solve with TEASER++
    teaser::RobustRegistrationSolver solver(params);

    solver.solve(local_vertices, global_vertices);

    auto solution = solver.getSolution();

    //get max clique
    std::vector<int> mcq = solver.getInlierMaxClique();   

    association_query_=local_vertices;
    association_match_=global_vertices;

    //get pose
    Eigen::Vector3d translation = solution.translation;
    Eigen::Matrix3d rotation = solution.rotation;
    Eigen::Matrix4d pose;

    pose << rotation(0,0), rotation(0,1), rotation(0,2), translation.x(),
            rotation(1,0), rotation(1,1), rotation(1,2), translation.y(),
            rotation(2,0), rotation(2,1), rotation(2,2), translation.z(),
            0.0          , 0.0          , 0.0          , 1.0            ;

    output.max_clique = mcq;
    output.pose = pose;
    output.valid = solution.valid;
    output.mcq_time = solution.mcq_time;
    output.pose_time = solution.solve_time;

    current_association_num_ = association_num;
    current_max_clique_num_ = mcq.size();
  }

  return output;
}

//pose estimation using pmc and gtsam
pose_with_clique TripletLoc::pose_estimate_gtsam(std::map<__int16_t,std::map<__int16_t, matches_info>> matches,std::map<__int16_t,std::map<__int16_t,instance_centriod>> ins_cen)
{
  TicToc mcq_timer;
  pose_with_clique output;
  output.valid = false;   //initial the status of the solution

  std::vector<instance_centriod> query_vertices;
  std::vector<instance_centriod> match_vertices;

  int association_num = 0;

  std::map<__int16_t,std::map<__int16_t, matches_info>>::iterator iter;
  iter = matches.begin();

  for (iter; iter != matches.end(); ++iter)
  {
    std::map<__int16_t, matches_info>::iterator iter1;
    iter1 = iter->second.begin();

    for (iter1; iter1 != iter->second.end(); ++iter1)
    {
      if (!(iter1->second.invalid))
      {
        instance_centriod query_vertex = ins_cen[iter->first][iter1->first];
        matches_info one_match = iter1->second;
        int top_k_match = one_match.matches_index.size();

        for (size_t i = 0; i < top_k_match; ++i)
        {
          int match_vertex_id_in_kd_tree = one_match.matches_index[i];

          __int16_t vertex_id_in_instance_center_list =  non_single_vertex_index_in_ins_map_[iter->first][match_vertex_id_in_kd_tree];

          instance_centriod match_point = instance_cens_in_ins_map_[iter->first][vertex_id_in_instance_center_list];         //global map 中和query_point匹配的顶点

          query_vertices.push_back(query_vertex);
          match_vertices.push_back(match_point);

          association_num = association_num + 1;
        }
      }
    }
  }

  if (association_num>=3) //at least 3 associations
  {
    Eigen::Matrix<double, 3, Eigen::Dynamic> local_vertices(3, association_num);
    Eigen::Matrix<double, 3, Eigen::Dynamic> global_vertices(3, association_num);

    for (size_t i = 0; i < association_num; ++i)
    {
      local_vertices.col(i) << query_vertices[i].x, query_vertices[i].y, query_vertices[i].z;
      global_vertices.col(i) << match_vertices[i].x, match_vertices[i].y, match_vertices[i].z;
    }


    // TEASER++ setting (only max clique searching is used)
    // Prepare solver parameters
    teaser::RobustRegistrationSolver::Params params;
    params.use_max_clique = true;
    params.kcore_heuristic_threshold = 1.0;    
    params.estimate_scaling = false;            
    params.noise_bound = configs_.noise_bound;                      
    params.cbar2 = 1;

    // only solve max clique using TEASER++
    teaser::RobustRegistrationSolver solver(params);
    std::pair<std::vector<int>,double> mcq_and_time = solver.solveMcq(local_vertices, global_vertices);
    double mcq_time = mcq_timer.toc();

    //perform gtsam based pose estimation
    TicToc pose_solve_timer;
    bool solve_valid = true;
    double one_loc_time =0;

    if (mcq_and_time.first.size() <3)
    {
      solve_valid = false;
      output.pose = Eigen::Matrix4d::Identity();
    }

    else   //inlier association should more than 3
    {
      NonlinearFactorGraph graph;
      Values initial;
      initial.insert(X(0), Pose3(Eigen::Matrix4d::Identity()));

      //*************For rotation constraint from RSN map
      if (configs_.use_rsn) //use rotation constraint from RSN map
      {
        //calculate the anchor point
        int mcq_size = mcq_and_time.first.size();
        Eigen::Matrix<double, 3, Eigen::Dynamic> matched_vertices(3, mcq_size);
        for (size_t i = 0; i < mcq_size; ++i)
        {
          matched_vertices.col(i) = global_vertices.col(mcq_and_time.first[i]);
        }
        Eigen::Vector3d xyz_mean_matched_vertices = matched_vertices.rowwise().sum() / matched_vertices.cols();  //get the xyz mean of all matched vertices in instance map
        pcl::PointXY anchor_point;
        anchor_point.x = xyz_mean_matched_vertices[0];
        anchor_point.y = xyz_mean_matched_vertices[1];

        //get the road grid normal from the RSN map
        int K = 1; 
        std::vector<int> pointIdxNKNSearch(K);
        std::vector<float> pointNKNSquaredDistance(K);
        kdtree_road_grid_cen_.nearestKSearch(anchor_point, K, pointIdxNKNSearch, pointNKNSquaredDistance);
        Eigen::Vector3d road_normal = road_grid_normal_[pointIdxNKNSearch[0]];  //get the surfance normal from the nearest road grid
        road_normal.normalize();

        // Create an arbitrary x-axis direction
        Eigen::Vector3d x = Eigen::Vector3d::UnitX();
        // Create a y-axis direction
        Eigen::Vector3d y = Eigen::Vector3d::UnitY();

        // Compute y and x axes orthogonal to road_normal
        y = (x.cross(road_normal)).normalized();
        x = (y.cross(road_normal)).normalized();

        // value the rotation matrix
        Eigen::Matrix3d R;
        R.col(0) = x;
        R.col(1) = y;
        R.col(2) = road_normal;

        Eigen::Matrix4d prior_pose;
        prior_pose.block<3,3>(0,0) = R;  
        prior_pose(0,3) = 0;     //no prior for x, y, z
        prior_pose(1,3) = 0;
        prior_pose(2,3) = 0;
        prior_pose(3,3) = 1;

        double sigma_n = road_grid_normal_std_[pointIdxNKNSearch[0]];       //std for the normal of the road grid in RSN map
        double epsilon_n = (sigma_n + configs_.rot_delta) * M_PI / 180.0;   //additional perturbation

        double roll_variance = std::pow(epsilon_n, 2);   
        double pitch_variance = std::pow(epsilon_n, 2); 
        double yaw_variance = std::pow(M_PI, 2);       //yaw using a big variance, since we have no prior for yaw

        //set the prior factor
        gtsam::Vector6 variances;
        variances << roll_variance, pitch_variance, yaw_variance, 1e6, 1e6, 1e6;
        auto priorNoise = gtsam::noiseModel::Diagonal::Variances(variances);
        gtsam::PriorFactor<gtsam::Pose3> priorFactor(X(0), gtsam::Pose3(prior_pose), priorNoise);

        graph.add(priorFactor);  //add the prior factor to the graph
      }
    

      //*************For point to point registration
      noiseModel::Diagonal::shared_ptr noise = noiseModel::Unit::Create(3);
      std::vector<int> mcq_id = mcq_and_time.first;

      for (int i = 0; i < mcq_id.size(); ++i) {
        graph.add(Point2PointFactor(X(0), local_vertices.col(mcq_id[i]), global_vertices.col(mcq_id[i]), noise));
      }

      // Set options for the non-minimal solver
      LevenbergMarquardtParams lmParams;
      lmParams.setMaxIterations(300);
      lmParams.setRelativeErrorTol(1e-4);
      // lmParams.setVerbosityLM("TRYLAMBDA"); // Print detailed output during optimization

      // Set GNC-specific options
      GncParams<LevenbergMarquardtParams> gncParams(lmParams);
      gncParams.setLossType(GncLossType::TLS);  //GM or TLS
      gncParams.setVerbosityGNC(GncParams<LevenbergMarquardtParams>::Verbosity::SILENT);  //SILENT, SUMMARY, VALUES, print the final optimizatin result

      // Create the optimizer
      auto gnc =GncOptimizer<GncParams<LevenbergMarquardtParams>>(graph, initial, gncParams);
      Values estimate = gnc.optimize();
      Eigen::Matrix4d pose_gtsam = estimate.at<Pose3>(X(0))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                .matrix();    

      one_loc_time = pose_solve_timer.toc(); 

      output.pose = pose_gtsam;
    }
    
    association_query_=local_vertices;
    association_match_=global_vertices;

    output.max_clique = mcq_and_time.first;
    output.valid = solve_valid;
    output.mcq_time = mcq_time;
    output.pose_time = one_loc_time;

    current_association_num_ = association_num;
    current_max_clique_num_ = mcq_and_time.first.size();
  }

  return output;
}

std::map<__int16_t, std::pair<int,int>> TripletLoc::vertex_match_recall_N(std::map<__int16_t,std::map<__int16_t, matches_info>> &matches, std::map<__int16_t,std::map<__int16_t,instance_centriod>> ins_cen)
{
  std::map<__int16_t, std::pair<int,int>> match_nums;   //matches number for each class, .first refers to number of matches，.second refers to number of acceptable matches

  std::map<__int16_t,std::map<__int16_t, matches_info>> valid_matches;
  std::map<__int16_t,std::map<__int16_t, matches_info>>::iterator iter1;
  iter1 = matches.begin();

  for (iter1; iter1 != matches.end(); ++iter1)
  {
    std::map<__int16_t, matches_info>::iterator iter2;
    iter2 = iter1->second.begin();
    for (iter2; iter2 != iter1->second.end(); ++iter2)
    {
      if (!(iter2->second.invalid))
      {
        valid_matches[iter1->first][iter2->first] = iter2->second;
      }
    }
  }

  //get the gt pose of the current query frame
  Eigen::Matrix4d current_pose = gt_poses_que_[query_frame_count_].second;

  std::map<__int16_t,std::map<__int16_t, matches_info>>::iterator iter3;
  iter3 = valid_matches.begin();

  int correct_match_num = 0;            //for all classes, equals to the number of instances in the query frame based on vaild matches, which also has at least one correct(acceptable) match from the top-k matches
  int match_num = 0;                    //for all classes, equals to the number of instances in the query frame based on valid matches

  for (iter3; iter3 != valid_matches.end(); ++iter3)   //for each class
  {
    std::map<__int16_t, matches_info>::iterator iter4;
    iter4 = iter3->second.begin();

    int correct_match_num_one_class = 0;  
    int match_num_one_class = 0;          

    for (iter4; iter4!=iter3->second.end(); ++iter4)
    {
      instance_centriod query_point = ins_cen[iter3->first][iter4->first];  //one vertex in the query frame

      matches_info one_match = iter4->second;   //top-k matches for the query vertex

      int top_k_match = one_match.matches_index.size();

      bool match_flag = false;     //whether the query vertex has at least one correct match

      std::vector<bool> match_flags;
      match_flags.resize(top_k_match);

      for (size_t i = 0; i < top_k_match; ++i)
      {
        int match_vertex_id_in_kd_tree = one_match.matches_index[i];
        __int16_t vertex_id_in_instance_center_list =  non_single_vertex_index_in_ins_map_[iter3->first][match_vertex_id_in_kd_tree];
        instance_centriod match_point = instance_cens_in_ins_map_[iter3->first][vertex_id_in_instance_center_list];         //global map 中和query_point匹配的顶点

        Eigen::Vector4d que_point(query_point.x,query_point.y, query_point.z,1.0);
        Eigen::Vector4d que_point_new = current_pose * que_point;

        float dx = match_point.x - que_point_new.x();
        float dy = match_point.y - que_point_new.y();
        float dz = match_point.z - que_point_new.z();

        float dis = sqrt(dx*dx + dy*dy + dz*dz);

        bool single_match_flag = false;

        if (dis<= configs_.vertex_dis_thre_for_correct_match)      //consider as a correct(acceptable) match
        {
          match_flag = true;
          single_match_flag =true;
        }
        match_flags[i] = single_match_flag;
      }

      matches[iter3->first][iter4->first].correct_flags=match_flags; 

      if (match_flag)       //at least one correct match, matching result for the current query vertex is considered as positive
      {
        correct_match_num_one_class = correct_match_num_one_class + 1;
      }

      match_num_one_class = match_num_one_class + 1;

    }

    match_num = match_num + match_num_one_class;
    correct_match_num = correct_match_num + correct_match_num_one_class;

    std::pair<int, int> tmp;   //for one class
    tmp.first  = match_num_one_class;
    tmp.second = correct_match_num_one_class;

    match_nums[iter3->first] = tmp;
  }

  std::pair<int, int> total_nums;  //for the whole query frame
  total_nums.first  = match_num;
  total_nums.second = correct_match_num;

  match_nums[0] = total_nums;

  return match_nums;
}

//functions for visualization
visualization_msgs::MarkerArray TripletLoc::instances_visual(std::map<__int16_t,std::map<__int16_t,instance_centriod>>  ins_cens)
{
  visualization_msgs::MarkerArray output;

  Eigen::Matrix4d current_pose = gt_poses_que_[query_frame_count_].second;
  current_pose(2,3) = current_pose(2,3) + z_offest_;


  std::map<__int16_t,std::map<__int16_t,instance_centriod>>::iterator iter;
  iter = ins_cens.begin();

  int id_index = 0;

  for (iter; iter != ins_cens.end(); iter++)
  {
    std::map<__int16_t,instance_centriod>::iterator iter1;
    iter1 = iter->second.begin();

    for (iter1; iter1 != iter->second.end(); ++iter1)
    {
      visualization_msgs::Marker marker, maker_text;
      marker.header.frame_id = "tripletloc";
      marker.header.stamp    =ros::Time::now();
      marker.ns = "instance_query";
      marker.type = visualization_msgs::Marker::SPHERE;
      marker.action   = visualization_msgs::Marker::ADD;
      marker.lifetime = ros::Duration();//(sec,nsec),0 forever

      marker.id= id_index;     

      //set marker position
      Eigen::Vector4d pose(iter1->second.x,iter1->second.y, iter1->second.z,1.0);
      Eigen::Vector4d pose_new = current_pose * pose;
      marker.pose.position.x = pose_new.x();
      marker.pose.position.y = pose_new.y();
      marker.pose.position.z = pose_new.z();

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
      marker.color.r = (float)configs_.semantic_name_rgb[iter->first].color_r/255;
      marker.color.g = (float)configs_.semantic_name_rgb[iter->first].color_g/255;
      marker.color.b = (float)configs_.semantic_name_rgb[iter->first].color_b/255;

      output.markers.push_back(marker);

      id_index = id_index + 1;
    }
  }
  return output;
}

visualization_msgs::MarkerArray TripletLoc::edges_visual(std::map<__int16_t,std::map<__int16_t,instance_centriod>> ins_cen, Eigen::MatrixXi adj_mat,std::map<int, id_label> index_2_label_id, int global_or_local)
{
  visualization_msgs::MarkerArray output_edges;
  visualization_msgs::Marker edge;

  edge.color.r=1.0;
  edge.color.g=0.0;
  edge.color.b=0.0;
  edge.color.a=1.0;

  Eigen::Matrix4d current_pose;

  std::string name_space;
  if (global_or_local == 0)   //not suggested to use this function for global map, since there are too many edges
  {
    name_space = "edges_global";
    current_pose << 1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0;
  }
  else if (global_or_local == 1)
  {
    name_space = "edges_query";
    current_pose = gt_poses_que_[query_frame_count_].second;
    current_pose(2,3) = current_pose(2,3) + z_offest_;
  }

  int id_indx =0;
  for (int row = 0; row < adj_mat.rows(); ++row)
  {
    int col_start_id = row;    //because the matrix is symmetric
    for (int col = col_start_id; col < adj_mat.cols(); ++col)    
    {
      if(adj_mat(row,col)==1)
      {
        edge.header.frame_id = "tripletloc";
        edge.header.stamp=ros::Time::now();
        edge.ns = name_space;
        edge.type = visualization_msgs::Marker::LINE_LIST;     

        edge.pose.orientation.w=1.0;
        edge.scale.x=0.1;    

        //set marker action
        edge.action = visualization_msgs::Marker::ADD;
        edge.lifetime = ros::Duration();//(sec,nsec),0 forever
        edge.id= id_indx ;

        instance_centriod edge_p1 = ins_cen[ index_2_label_id[row].label ][ index_2_label_id[row].id ];
        instance_centriod edge_p2 = ins_cen[ index_2_label_id[col].label ][ index_2_label_id[col].id ];

        Eigen::Vector4d point1(edge_p1.x,edge_p1.y, edge_p1.z,1.0);
        Eigen::Vector4d point1_new = current_pose * point1;
        Eigen::Vector4d point2(edge_p2.x,edge_p2.y, edge_p2.z,1.0);
        Eigen::Vector4d point2_new = current_pose * point2;

        geometry_msgs::Point p1, p2;
        p1.x = point1_new.x();
        p1.y = point1_new.y();
        p1.z = point1_new.z();

        p2.x = point2_new.x();
        p2.y = point2_new.y();
        p2.z = point2_new.z();

        edge.points.push_back(p1);
        edge.points.push_back(p2);

        output_edges.markers.push_back(edge);

        id_indx = id_indx +1;
      }
    }
  }

  return output_edges;
}


std::pair<visualization_msgs::MarkerArray, visualization_msgs::MarkerArray> TripletLoc::vertex_matches_visual(std::map<__int16_t,std::map<__int16_t, matches_info>> matches, std::map<__int16_t,std::map<__int16_t,instance_centriod>> ins_cen)
{
  std::pair<visualization_msgs::MarkerArray, visualization_msgs::MarkerArray> ouputs;
  visualization_msgs::MarkerArray output_edges, output_edges_wrong;
  visualization_msgs::Marker edge, edge_wrong;

  std::map<__int16_t,std::map<__int16_t, matches_info>> show_matches;

  std::map<__int16_t,std::map<__int16_t, matches_info>>::iterator iter1;
  iter1 = matches.begin();

  for (iter1; iter1 != matches.end(); ++iter1)
  {
    std::map<__int16_t, matches_info>::iterator iter2;
    iter2 = iter1->second.begin();

    for (iter2; iter2 != iter1->second.end(); ++iter2)
    {
      if (!(iter2->second.invalid))
      {
        show_matches[iter1->first][iter2->first] = iter2->second;
      }
    }
  }

  //load current gt pose
  Eigen::Matrix4d current_pose = gt_poses_que_[query_frame_count_].second;
  current_pose(2,3) = current_pose(2,3) + z_offest_;

  //set value
  edge.color.r=0.0;
  edge.color.g=1.0;
  edge.color.b=0.0;
  edge.color.a =1.0;
  edge.pose.orientation.w=1.0;
  edge.scale.x=0.5;         
  edge.header.frame_id = "tripletloc";
  edge.ns = "matches_correct";
  edge.type = visualization_msgs::Marker::LINE_LIST;      
  edge.header.stamp=ros::Time::now();
  //set marker action
  edge.action = visualization_msgs::Marker::ADD;
  edge.lifetime = ros::Duration();//(sec,nsec),0 forever

  edge_wrong.color.r=0.0;
  edge_wrong.color.g=0.0;
  edge_wrong.color.b=1.0;
  edge_wrong.color.a =1.0;
  edge_wrong.pose.orientation.w=1.0;
  edge_wrong.scale.x=0.1;         
  edge_wrong.header.frame_id = "tripletloc";
  edge_wrong.ns = "matches_wrong";
  edge_wrong.type = visualization_msgs::Marker::LINE_LIST;      
  edge_wrong.header.stamp=ros::Time::now();
  //set marker action
  edge_wrong.action = visualization_msgs::Marker::ADD;
  edge_wrong.lifetime = ros::Duration();//(sec,nsec),0 forever

  int id_indx = 0;

  std::map<__int16_t,std::map<__int16_t, matches_info>>::iterator iter3;
  iter3 = show_matches.begin();
  for (iter3; iter3 != show_matches.end(); ++iter3)
  {
    std::map<__int16_t, matches_info>::iterator iter4;
    iter4 = iter3->second.begin();

    for (iter4; iter4!=iter3->second.end(); ++iter4)
    {
      matches_info one_match = iter4->second;

      int top_k_match = one_match.matches_index.size();

      for (size_t i = 0; i < top_k_match; ++i)
      {
        bool match_correct_flag = one_match.correct_flags[i];

        instance_centriod query_point = ins_cen[iter3->first][iter4->first];

        int match_vertex_id_in_kd_tree = one_match.matches_index[i];

        __int16_t vertex_id_in_instance_center_list =  non_single_vertex_index_in_ins_map_[iter3->first][match_vertex_id_in_kd_tree];

        instance_centriod match_point = instance_cens_in_ins_map_[iter3->first][vertex_id_in_instance_center_list];

        Eigen::Vector4d que_point(query_point.x,query_point.y, query_point.z,1.0);
        Eigen::Vector4d que_point_new = current_pose * que_point;


        geometry_msgs::Point p1, p2;
        p1.x = que_point_new.x();
        p1.y = que_point_new.y();
        p1.z = que_point_new.z();

        p2.x = match_point.x;
        p2.y = match_point.y;
        p2.z = match_point.z;


        if (match_correct_flag)
        {
          edge.id= id_indx ;
          edge.points.push_back(p1);
          edge.points.push_back(p2);

          output_edges.markers.push_back(edge);
        }
        else if(!match_correct_flag)
        {
          edge_wrong.id= id_indx ;
          edge_wrong.points.push_back(p1);
          edge_wrong.points.push_back(p2);

          output_edges_wrong.markers.push_back(edge_wrong);
        }

        id_indx = id_indx + 1;
      }

    }

  }

  ouputs.first =output_edges;
  ouputs.second =output_edges_wrong;

  return ouputs;
}


visualization_msgs::MarkerArray TripletLoc::vertex_using_est_pose_visual(pose_with_clique solution,std::map<__int16_t,std::map<__int16_t,instance_centriod>> ins_cens)
{
  visualization_msgs::MarkerArray vertices;

  if (solution.valid) 
  {
    Eigen::Matrix4d current_pose_est = solution.pose;
    current_pose_est(2,3) = current_pose_est(2,3)+ z_offest_;

    std::map<__int16_t,std::map<__int16_t,instance_centriod>>::iterator iter;
    iter = ins_cens.begin();

    int id_index = 0;

    for (iter; iter != ins_cens.end(); iter++)
    {
      std::map<__int16_t,instance_centriod>::iterator iter1;
      iter1 = iter->second.begin();

      for (iter1; iter1 != iter->second.end(); ++iter1)
      {
        visualization_msgs::Marker marker;
        marker.header.frame_id = "tripletloc";
        marker.header.stamp    =ros::Time::now();
        marker.ns = "instance_est";
        marker.type = visualization_msgs::Marker::SPHERE;
        marker.action   = visualization_msgs::Marker::ADD;
        marker.lifetime = ros::Duration();//(sec,nsec),0 forever

        marker.id=  id_index;

        //set marker position
        Eigen::Vector4d pose(iter1->second.x,iter1->second.y, iter1->second.z,1.0);
        Eigen::Vector4d pose_new = current_pose_est * pose;

        marker.pose.position.x = pose_new.x();
        marker.pose.position.y = pose_new.y();
        marker.pose.position.z = pose_new.z();

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
        marker.color.r = 1.0;
        marker.color.g = 1.0;
        marker.color.b = 1.0;

        vertices.markers.push_back(marker);

        id_index = id_index + 1;
      }
    }
  }

  return vertices;
}

visualization_msgs::MarkerArray TripletLoc::MCQ_visual(std::vector<int> max_clique,std::map<__int16_t,std::map<__int16_t,instance_centriod>> ins_cens)
{
  visualization_msgs::MarkerArray associations;
  visualization_msgs::Marker assoc;
  assoc.color.r=1.0;
  assoc.color.g=(float)128.0/255.0;
  assoc.color.b=0.0;
  assoc.color.a =1.0;
  assoc.pose.orientation.w=1.0;
  assoc.scale.x=0.1;   
  assoc.header.frame_id = "tripletloc";
  assoc.ns = "max_clique";
  assoc.type = visualization_msgs::Marker::LINE_LIST;     
  assoc.header.stamp=ros::Time::now();
  assoc.action = visualization_msgs::Marker::ADD;
  assoc.lifetime = ros::Duration();//(sec,nsec),0 forever

  //load current gt pose
  Eigen::Matrix4d current_pose = gt_poses_que_[query_frame_count_].second;
  current_pose(2,3) = current_pose(2,3) + z_offest_;

  int index =0;
  for (size_t i = 0; i < max_clique.size(); ++i)
  {
    int association_id = max_clique[i];
    Eigen::Vector4d que_point(association_query_(0,association_id),association_query_(1,association_id), association_query_(2,association_id),1.0);

    Eigen::Vector4d que_point_new = current_pose * que_point;

    assoc.id= index;

    geometry_msgs::Point p1, p2;
    p1.x = que_point_new.x();
    p1.y = que_point_new.y();
    p1.z = que_point_new.z();

    p2.x = association_match_(0, association_id);
    p2.y = association_match_(1, association_id);
    p2.z = association_match_(2, association_id);

    assoc.points.push_back(p1);
    assoc.points.push_back(p2);

    associations.markers.push_back(assoc);

    index = index + 1;
  }

  return associations;
}

visualization_msgs::Marker TripletLoc::local_area_visual(bool success_locate)
{
  visualization_msgs::Marker marker;
  marker.header.frame_id = "tripletloc";
  marker.header.stamp    =ros::Time::now();
  marker.ns = "local_area";
  marker.type = visualization_msgs::Marker::CYLINDER;
  marker.action   = visualization_msgs::Marker::ADD;
  marker.lifetime = ros::Duration();//(sec,nsec),0 forever

  if (success_locate)
  {
    marker.color.r=0.0;
    marker.color.g=1.0;
  }
  else
  {
    marker.color.r=1.0;
    marker.color.g=0.0;
  }

  marker.color.b=0.0;
  marker.color.a =0.3;

  marker.id= 0;
  marker.pose.position.x = gt_poses_que_[query_frame_count_].second(0,3);
  marker.pose.position.y = gt_poses_que_[query_frame_count_].second(1,3);
  marker.pose.position.z = 0;

  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;
  marker.pose.orientation.w = 1.0;

  //set marker scale
  marker.scale.x = 150;
  marker.scale.y = 150;
  marker.scale.z = 1.0;

  return marker;
}

std::vector<visualization_msgs::MarkerArray> TripletLoc::clean_last_visualization()
{
  std::vector<visualization_msgs::MarkerArray> output;
  visualization_msgs::MarkerArray output_ins, output_edge, output_matches,output_matches_wrong, output_ins_est, output_consistent_association, output_local_area;
  visualization_msgs::Marker msg_ins, msg_edge, msg_matche, msg_matche_wrong, msg_ins_est, msg_consis_assoc, msg_local_area;

  msg_ins.id = 0;
  msg_ins.ns = "instance_query";
  msg_ins.action = visualization_msgs::Marker::DELETEALL;
  output_ins.markers.push_back(msg_ins);

  msg_edge.id = 0;
  msg_edge.ns = "edges_query";
  msg_edge.action = visualization_msgs::Marker::DELETEALL;
  output_edge.markers.push_back(msg_edge);

  msg_matche.id = 0;
  msg_matche.ns = "matches_correct";
  msg_matche.action = visualization_msgs::Marker::DELETEALL;
  output_matches.markers.push_back(msg_matche);

  msg_matche_wrong.id = 0;
  msg_matche_wrong.ns = "matches_wrong";
  msg_matche_wrong.action = visualization_msgs::Marker::DELETEALL;
  output_matches_wrong.markers.push_back(msg_matche_wrong);

  msg_ins_est.id = 0;
  msg_ins_est.ns = "instance_est";
  msg_ins_est.action = visualization_msgs::Marker::DELETEALL;
  output_ins_est.markers.push_back(msg_ins_est);

  msg_consis_assoc.id = 0;
  msg_consis_assoc.ns = "max_clique";
  msg_consis_assoc.action = visualization_msgs::Marker::DELETEALL;
  output_consistent_association.markers.push_back(msg_consis_assoc);

  msg_local_area.id = 0;
  msg_local_area.ns = "local_area";
  msg_local_area.action = visualization_msgs::Marker::DELETEALL;
  output_local_area.markers.push_back(msg_local_area);

  output.push_back(output_ins);
  output.push_back(output_edge);
  output.push_back(output_matches);
  output.push_back(output_matches_wrong);
  output.push_back(output_ins_est);
  output.push_back(output_consistent_association);
  output.push_back(output_local_area);

  return output;
}