//Author:   Weixin Ma       weixin.ma@connect.polyu.hk
//"Triplet-Graph: Global Metric Localization Based on Semantic Triplet Graph for Autonomous Vehicles", IEEE RA-L, 2024.
#include "./../include/tripletgraph.h"

std::map<__int16_t, class_combination_info> class_combination_infos_;     //all infos for {C}_{l^m}, including {C}_{l^m}, and related mapping matrices 

TripletGraph::TripletGraph(){  
}

void TripletGraph::set_config(std::string config_file_dir)
{
  conf_para_ = get_config(config_file_dir);
  get_class_combination();
}

config_para TripletGraph::get_config(std::string config_file_dir){
  config_para output;

  auto data_cfg = YAML::LoadFile(config_file_dir);

  omp_num_threads_ransc_ = data_cfg["tripletgraph_para"]["omp_num_threads_ransc"].as<int>();
  omp_num_threads_vetex_match_ = data_cfg["tripletgraph_para"]["omp_num_threads_vertex_match"].as<int>();

  output.ins_fuse_radius         = data_cfg["tripletgraph_para"]["ins_fuse_radius"].as<float>();                     //for instance fusion
  output.angle_resolution        = data_cfg["tripletgraph_para"]["angel_resolution"].as<int>();                      // \theta
  output.edge_dis_thr            = data_cfg["tripletgraph_para"]["edge_dis_thr"].as<float>();                        // \tau_{edge}
  output.maxIterations_ransac    = data_cfg["tripletgraph_para"]["max_iterations_ransac"].as<int>();                 
  output.ransca_threshold        = data_cfg["tripletgraph_para"]["ransac_threshold"].as<float>();    
  output.cere_opt_iterations_max = data_cfg["tripletgraph_para"]["cere_opt_iterations_max"].as<int>();                           
  output.percentage_matches_used = data_cfg["tripletgraph_para"]["percentage_matches_used"].as<float>();                       
  output.similarity_refine_thre  = data_cfg["tripletgraph_para"]["project_select_thresold"].as<float>();    

  auto classes_for_graph          =data_cfg["classes_for_graph"];
  auto instance_seg_para          =data_cfg["instance_seg_para"];
  auto class_name                 =data_cfg["labels"];
  auto color_map                  =data_cfg["color_map"];
  auto minimun_point_one_instance =data_cfg["mini_point_one_instance"];
  auto weights_for_class          =data_cfg["weights_for_class"];
  auto weights_for_cere_cost      =data_cfg["weights_for_cere_cost"];
  auto voxel_leaf_sizes           =data_cfg["voxel_leaf_size"];


  std::map<__int16_t,std::string> label_name;
  std::vector<__int16_t> class_for_graph;
  std::map<__int16_t,float> EuCluster_para;
  std::map<__int16_t,semantic_config> semantic_name_rgb;
  std::map<__int16_t,int> minimun_point_a_instance;
  std::map<__int16_t,float> classes_weights;
  std::map<__int16_t,double> cere_cost_weights;
  std::map<__int16_t,float> voxel_leaf_size;


  YAML::Node::iterator iter,iter1, iter2, iter4, iter5, iter6, iter7;
  iter  =classes_for_graph.begin();
  iter1 =class_name.begin();
  iter2 =instance_seg_para.begin();
  iter4 =minimun_point_one_instance.begin();
  iter5 =weights_for_class.begin();
  iter6 =weights_for_cere_cost.begin();
  iter7 =voxel_leaf_sizes.begin();

  //used class, i.e., L
  for (iter;iter!=classes_for_graph.end();iter++) {
    if(iter->second.as<bool>())
    {
      class_for_graph.push_back(iter->first.as<__int16_t>());
    }
  }
  output.classes_for_graph = class_for_graph;

  //label names
  for (iter1;iter1!=class_name.end();iter1++) {
    label_name[iter1->first.as<__int16_t>()] = iter1->second.as<std::string>();
  }  
  
  //paras for instance clustering 
  for (iter2;iter2!=instance_seg_para.end();iter2++) {
    EuCluster_para[iter2->first.as<__int16_t>()] = iter2->second.as<float>();
  }
  output.EuCluster_para = EuCluster_para;

  //paras for instance clustering 
  for (iter4; iter4!=minimun_point_one_instance.end(); iter4++)
  {
    minimun_point_a_instance[iter4->first.as<__int16_t>()] = iter4->second.as<int>();
  }
  output.minimun_point_one_instance = minimun_point_a_instance;

  //weights for different class when calculating similarity between two graphs, we simply let all as 1.0
  float weight_sum = 0.0;
  for (iter5;iter5!=weights_for_class.end();iter5++) {

    if (std::find(class_for_graph.begin(), class_for_graph.end(), iter5->first.as<__int16_t>()) != class_for_graph.end())
    {
      classes_weights[iter5->first.as<__int16_t>()] = iter5->second.as<float>();
      weight_sum = weight_sum + iter5->second.as<float>();
    }
  }
  //normalization
  std::map<__int16_t,float>::iterator iter_weigth;
  iter_weigth = classes_weights.begin();
  for (iter_weigth; iter_weigth!=classes_weights.end(); ++iter_weigth)
  { 
    iter_weigth->second = iter_weigth->second/weight_sum;
  }  
  output.weights_for_class = classes_weights;


  //weights for different class when conducting pose estimation, we simply let all as 1.0
  double weight_sum1 = 0.0;
  for (iter6;iter6!=weights_for_cere_cost.end();iter6++) {

    if (std::find(class_for_graph.begin(), class_for_graph.end(), iter6->first.as<__int16_t>()) != class_for_graph.end())
    {
      cere_cost_weights[iter6->first.as<__int16_t>()] = iter6->second.as<double>();
      weight_sum1 = weight_sum1 + iter6->second.as<double>();
    }
  }

  //normalization
  std::map<__int16_t,double>::iterator iter_weigth1;
  iter_weigth1 = cere_cost_weights.begin();
  for (iter_weigth1; iter_weigth1!=cere_cost_weights.end(); ++iter_weigth1)
  { 
    iter_weigth1->second = iter_weigth1->second/weight_sum1;
  }  
  output.weights_cere_cost = cere_cost_weights;

  //class name in string, and rgb values
  YAML::Node::iterator it;
  for (it = color_map.begin(); it != color_map.end(); ++it)
  {
    semantic_config single_semnatic;
    single_semnatic.obj_class = label_name[it->first.as<__int16_t>()];
    single_semnatic.color_b = it->second[0].as<int>();
    single_semnatic.color_g = it->second[1].as<int>();
    single_semnatic.color_r = it->second[2].as<int>();
    semantic_name_rgb[it->first.as<__int16_t>()] = single_semnatic;
  }
  output.semantic_name_rgb = semantic_name_rgb;

  //voxel leaf size for each class
  for (iter7;iter7!=voxel_leaf_sizes.end();iter7++) {
    voxel_leaf_size[iter7->first.as<__int16_t>()] = iter7->second.as<float>();
  }
  output.voxel_leaf_size = voxel_leaf_size;

  return output;
}

void TripletGraph::get_class_combination()                  //for building {C}_{l^m}, i.e., first-level bins
{
  std::vector<__int16_t> classes_for_graph = conf_para_.classes_for_graph;

  //generate triplet
  if(classes_for_graph.size()>=2)                          //at least two classes are required
  {
    class_combination single_combine;                      //a C in {C}_{l^m}

    int combination_amount;                                //combination number for {C}_{l^m}, equals to N1
    

    for (int m = 0; m < classes_for_graph.size(); ++m)     //pick l^m
    {
      combination_amount=0;
      std::vector<class_combination> triplet_class_combs;  //{C}_{l^m}, lm = classes_for_graph[m]
      std::map<__int16_t,std::map<__int16_t, std::map<__int16_t,int> > > triplet_to_descriptor_bit;     //C to index of {C}_{l^m} (i.e., first-level bin index)
      std::map<int, std::vector<class_combination>> bit_to_triplet;                                     //index of {C}_{l^m}  to C 
      triplet_to_descriptor_bit.clear();
      bit_to_triplet.clear();

      //pick l^f and pick l^m are the same
      for (int j = 0; j < classes_for_graph.size(); ++j)       
      {
        single_combine.l_m = classes_for_graph[m];         
        single_combine.l_f = classes_for_graph[j];
        single_combine.l_t = classes_for_graph[j];

        triplet_class_combs.push_back(single_combine);                       
        triplet_to_descriptor_bit[single_combine.l_f][single_combine.l_m][single_combine.l_t] = combination_amount; 
        bit_to_triplet[combination_amount].push_back(single_combine);  
        //std::cout<<combination_amount<<std::endl;
        combination_amount = combination_amount+1;         
      }
      

      //pick l^f and pick l^m are different
      for (int f = 0; f < classes_for_graph.size(); ++f)          //pick l^f
      {
        std::vector<__int16_t> diff_vertex1 = classes_for_graph;
        for (int l=0; l<f+1; ++l)                                 //delet used l^f, for picking l^t
        {
          diff_vertex1.erase(diff_vertex1.begin());
        }

        for (int t = 0; t < diff_vertex1.size(); ++t)            //pick l^t
        {
          //{l^f, l^m, l^t}, e.g., {fence, trunk, pole}
          single_combine.l_m = classes_for_graph[m];
          single_combine.l_f = classes_for_graph[f];
          single_combine.l_t = diff_vertex1[t];

          triplet_class_combs.push_back(single_combine);               
          triplet_to_descriptor_bit[single_combine.l_f][single_combine.l_m][single_combine.l_t]=combination_amount; 
          bit_to_triplet[combination_amount].push_back(single_combine);     
          
          //{l^t, l^m, l^f}, e.g., {pole, trunk, fence}
          single_combine.l_m = classes_for_graph[m];
          single_combine.l_t = classes_for_graph[f];
          single_combine.l_f = diff_vertex1[t];
        
          triplet_class_combs.push_back(single_combine);                 
          triplet_to_descriptor_bit[single_combine.l_f][single_combine.l_m][single_combine.l_t]=combination_amount;   
          bit_to_triplet[combination_amount].push_back(single_combine);     


          combination_amount =combination_amount + 1;  //for vertex
        }
      }
      class_combination_infos_[classes_for_graph[m]].triplet_classes=triplet_class_combs;
      class_combination_infos_[classes_for_graph[m]].triplet_to_descriptor_bit=triplet_to_descriptor_bit;
      class_combination_infos_[classes_for_graph[m]].dim_combination=combination_amount;   
      class_combination_infos_[classes_for_graph[m]].bit_to_triplet =bit_to_triplet;
    }

    std::cout<<std::endl;
    
    // // print out first-level bins for different classes
    // std::map<__int16_t, class_combination_info>::iterator iter; 
    // iter=class_combination_infos_.begin();
    // for ( iter; iter != class_combination_infos_.end(); ++iter)
    // {
    //   std::cout<<"Class-"<<iter->first<<" : "<<"combination numbers (i.e., N1)="<<iter->second.dim_combination<<";"<<std::endl;
    //   for (size_t i = 0; i < iter->second.triplet_classes.size(); ++i)
    //   {
    //     std::cout<<"    combin: "<<iter->second.triplet_classes[i].l_f<<"-"<<iter->second.triplet_classes[i].l_m<<"-"<<iter->second.triplet_classes[i].l_t<<
    //     ", correspoidng bit: "<<iter->second.triplet_to_descriptor_bit[iter->second.triplet_classes[i].l_f][iter->second.triplet_classes[i].l_m][iter->second.triplet_classes[i].l_t]<<std::endl;
    //   }
    // }
  }
}


std::map<__int16_t, pcl::PointCloud<pcl::PointXYZRGB>> TripletGraph::get_pointcloud_with_semantic_helipr(std::string bin_dir, std::string label_file)
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
 
    my_point.r = conf_para_.semantic_name_rgb[label].color_r;
    my_point.g = conf_para_.semantic_name_rgb[label].color_g;
    my_point.b = conf_para_.semantic_name_rgb[label].color_b;

    output[label].push_back(my_point);
    all_points.push_back(my_point);
  }

  output[-1] = all_points;

  input_point.close();
  input_label.close();

  return output;  
}


instance_result TripletGraph::get_ins_cen(std::string bin_flie, std::string label_file)
{
  instance_result output;

  TicToc load_pc;
  std::map<__int16_t, pcl::PointCloud<pcl::PointXYZRGB>> pc_rgb =  get_pointcloud_with_semantic_helipr(bin_flie, label_file);
  load_pc_label_time = load_pc.toc();
  //std::cout<<"Load pc : "<<load_pc.toc()<<"ms"<<std::endl;

  TicToc instance_cluster_t;
  //downsample and cluster 
  std::map<__int16_t, pcl::PointCloud<pcl::PointXYZRGB>>::iterator iter;
  iter = pc_rgb.begin();
      
  int total_instance_number = 0;   

  std::map<__int16_t,std::map<__int16_t, instance_center>>  instance_cens;
  std::pair< std::map<__int16_t, int>, int> instance_number; 
  std::map<__int16_t, int> instance_numbers_each_class;

  std::vector<__int16_t> class_for_graph = conf_para_.classes_for_graph;
  std::vector<__int16_t>::iterator t;

  //for further instance fusion  
  pcl::PointCloud<pcl::PointXYZ> instances_xyz_;
  std::vector<int> each_instanc_point_num_;
  std::vector<__int16_t> each_instan_label_;

  //TicToc cluster_first_t;
  for (iter; iter != pc_rgb.end(); ++iter)    //for each class
  {
    t = find(class_for_graph.begin(),class_for_graph.end(),iter->first);

    if(t != class_for_graph.end())
    {
      pcl::PointCloud<pcl::PointXYZRGB> filtered_pc;

      float leaf_size = conf_para_.voxel_leaf_size[iter->first];

      pcl::VoxelGrid<pcl::PointXYZRGB> sor;
      sor.setInputCloud(iter->second.makeShared());                            
      sor.setLeafSize(leaf_size, leaf_size, leaf_size);                
      sor.filter(filtered_pc);  

      //Euclidean cluster
      pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);   
      tree->setInputCloud(filtered_pc.makeShared());                                               
      std::vector<pcl::PointIndices> cluster_indices;                                                
      pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;                                         

      float cluster_tolerance = conf_para_.EuCluster_para[iter->first];
      ec.setClusterTolerance (cluster_tolerance);                                                        
      ec.setMinClusterSize (20);
      ec.setMaxClusterSize ( filtered_pc.size() );
      ec.setSearchMethod (tree);
      ec.setInputCloud ( filtered_pc.makeShared());
      ec.extract (cluster_indices);

      //cluster saving
      std::vector<pcl::PointIndices>::const_iterator it;
      it = cluster_indices.begin();

      __int16_t new_id=0;
      int one_class_instance_number = 0;

      for (it; it!=cluster_indices.end();++it)
      {     
        int minimun_points_amount = conf_para_.minimun_point_one_instance[iter->first];
        int cloud_cluster_point_num = it->indices.size();

        if(cloud_cluster_point_num >= minimun_points_amount)              
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
          instance_center centriod;
          centriod.x = sum_x/(float)cloud_cluster_point_num;
          centriod.y = sum_y/(float)cloud_cluster_point_num;
          centriod.z = sum_z/(float)cloud_cluster_point_num;

          one_centriod.x = centriod.x;
          one_centriod.y = centriod.y;
          one_centriod.z = centriod.z;
          instances_xyz_.push_back(one_centriod);

          each_instanc_point_num_.push_back(cloud_cluster_point_num);
          each_instan_label_.push_back(iter->first);
        }  
      }
    }
  }
  //std::cout<<"First cluster : "<<cluster_first_t.toc()<<"ms"<<std::endl;

  //instances fusion, fusing instances that are close to each other
  //TicToc second_cluster_t;
  std::map<__int16_t, std::vector<instance_center>> fused_ins_tmp;
  
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);   
  tree->setInputCloud(instances_xyz_.makeShared());                                                  
  std::vector<pcl::PointIndices> cluster_indices;                                                
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;                                         

  float cluster_tolerance = conf_para_.ins_fuse_radius;
  ec.setClusterTolerance (cluster_tolerance);                                             
  ec.setMinClusterSize (1);
  ec.setMaxClusterSize ( instances_xyz_.size() );
  ec.setSearchMethod (tree);
  ec.setInputCloud ( instances_xyz_.makeShared());
  ec.extract (cluster_indices);

  //cluster saving
  std::vector<pcl::PointIndices>::const_iterator it;
  it = cluster_indices.begin();

  __int16_t new_id=0;
  int instance_number_fused = 0;

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

      if (each_instanc_point_num_[ ins_index_in_tree ] > max_point_num)
      {
        max_point_num = each_instanc_point_num_[ ins_index_in_tree ];
        max_pc_num_id = ins_index_in_tree;
      }
    }

    instance_center centriod;
    centriod.x = sum_x/(float)pc_cluster_point_num;
    centriod.y = sum_y/(float)pc_cluster_point_num;
    centriod.z = sum_z/(float)pc_cluster_point_num;


    //decide the label for the fused instance
    __int16_t label_current_ins = each_instan_label_[max_pc_num_id];
    fused_ins_tmp[label_current_ins].push_back(centriod);

    instance_number_fused = instance_number_fused +1;
  }
  //std::cout<<"Second cluster : "<<second_cluster_t.toc()<<"ms"<<std::endl;


  //TicToc value_t;
  //set new id for each instance
  std::map<__int16_t, std::map<__int16_t, instance_center>> fused_ins;
  std::map<__int16_t, std::vector<instance_center>>::iterator iter_fused;
  iter_fused = fused_ins_tmp.begin();

  pcl::PointCloud<pcl::PointXYZRGB> instance_as_pc;
  pcl::PointXYZRGB one_point;
  
  for (iter_fused; iter_fused != fused_ins_tmp.end(); ++iter_fused)
  {
    //std::cout<<"Instance number for class-"<<conf_para_.semantic_name_rgb[iter_fused->first].obj_class<<" : "<<iter_fused->second.size()<<std::endl;

    instance_numbers_each_class[iter_fused->first]=iter_fused->second.size();

    for (int i = 0; i < iter_fused->second.size(); ++i)
    {
      fused_ins[iter_fused->first][i] = iter_fused->second[i];
      total_instance_number = total_instance_number +1;

      one_point.x = iter_fused->second[i].x;
      one_point.y = iter_fused->second[i].y;
      one_point.z = iter_fused->second[i].z;
      one_point.r = conf_para_.semantic_name_rgb[iter_fused->first].color_r;
      one_point.g = conf_para_.semantic_name_rgb[iter_fused->first].color_g;
      one_point.b = conf_para_.semantic_name_rgb[iter_fused->first].color_b;
      instance_as_pc.push_back(one_point);
    }

  }

  instance_number.first = instance_numbers_each_class;
  instance_number.second = total_instance_number;

  //std::cout<<"Instance number in total: "<<total_instance_number<<std::endl;

  output.instance_centriods = fused_ins;
  output.instance_number = instance_number;
  output.instance_as_pc = instance_as_pc;
  
  return output;
}

Descriptors TripletGraph::get_descriptor_fast(std::map<__int16_t,std::map<__int16_t, instance_center>> ins_cens)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr centriods(new pcl::PointCloud<pcl::PointXYZ>);
  std::map<__int16_t,std::map<__int16_t, instance_center>>::iterator iter;
  iter = ins_cens.begin();

  int index=0;
  std::map<int, label_id> index_2_label_id;                        //index of matrix to label and id

  for (iter; iter!=ins_cens.end(); iter++)
  {
    std::map<__int16_t,instance_center>::iterator iter1;
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
  int bin_amount = 180 / conf_para_.angle_resolution;   //i.e., N2

  //set sources points for knn search
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	kdtree.setInputCloud(centriods);


  //perform descriptor extraction
  Descriptors output;
  std::map<__int16_t,std::map<__int16_t, Eigen::MatrixXi>> descriptors_angle;        
  std::map<__int16_t, Eigen::MatrixXi> global_descriptor;     
  std::vector<__int16_t> classes_for_graph = conf_para_.classes_for_graph;
  for (size_t i = 0; i < classes_for_graph.size(); ++i)
  {
    Eigen::MatrixXi single_class_overall_des;          //i.e., Des^{l}
    single_class_overall_des.resize(class_combination_infos_[classes_for_graph[i]].dim_combination, bin_amount);
    single_class_overall_des.setZero();
    global_descriptor[classes_for_graph[i]] = single_class_overall_des;
  }

  
  int triplet_number =0;                             //number of constructed triplets, i.e., size of {\Delta}_{v_{j}} 
  float radius = conf_para_.edge_dis_thr;

  if(ins_amount>=3)      
  {
    for (size_t i = 0; i < ins_amount; ++i)          //pick v^m
    {
      pcl::PointXYZ query_point;
      query_point.x = centriods->points[i].x;
      query_point.y = centriods->points[i].y;
      query_point.z = centriods->points[i].z;

      std::vector<int> pointIdxRadiusSearch;
      std::vector<float> pointRadiusSquaredDistance;

      size_t nConnected_vertices = kdtree.radiusSearch(query_point, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance);

      //buid descriptor
      std::vector<int> instance_index = pointIdxRadiusSearch;              

      instance_center first_vertex, mid_vertex, third_vertex;               //i.e., v^f, v^m, v^t

      __int16_t class_label = index_2_label_id[instance_index[0]].label;    //here, i = instance_index[0]
      __int16_t instance_id = index_2_label_id[instance_index[0]].id;                             

      int class_combines_amout = class_combination_infos_[class_label].dim_combination;   //i.e., N1, actually same for all classes               
      
      Eigen::MatrixXi vertex_descriptor_angle;                             
      vertex_descriptor_angle.setZero(class_combines_amout, bin_amount);   //vertex descriptor for（class_label，instnace_id）

      if (instance_index.size()>=3) 
      {
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

            double angle=get_angle(first_vertex.x, first_vertex.y,third_vertex.x, third_vertex.y, mid_vertex.x, mid_vertex.y);
            int row = class_combination_infos_[class_label].triplet_to_descriptor_bit[index_2_label_id[id_2[k]].label] [index_2_label_id[instance_index[0]].label] [index_2_label_id[id_1[j]].label];

            int col ;
            if (angle==180)
            {
              col = (int)angle/conf_para_.angle_resolution -1;

            }
            else
            {
              col = (int)angle/conf_para_.angle_resolution;
            }

            //std::cout<<"check point  "<<first_vertex.x<<" " <<first_vertex.y<<" "<<third_vertex.x<<" "<<third_vertex.y<<" "<<mid_vertex.x<<" "<<mid_vertex.y<< " angle:"<<angle<< " ; "<<row <<" "<< col<<std::endl;
            vertex_descriptor_angle(row, col) = vertex_descriptor_angle(row, col)+1;

            triplet_number = triplet_number + 1;
          }
        }
      }

      descriptors_angle[class_label][instance_id] = vertex_descriptor_angle;

      global_descriptor[class_label] = global_descriptor[class_label] + vertex_descriptor_angle;   

    }
  }
  
  output.vertex_descriptors = descriptors_angle;
  output.global_descriptor  = global_descriptor;  
  return output;  
}

double TripletGraph::get_angle(double x1, double y1, double x2, double y2, double x3, double y3)
/*get angle ACB, point C is the center point A(x1,y1) B(x2,y2) C(x3,y3), range: [0, 180]*/
{
  double theta = atan2(x1 - x3, y1 - y3) - atan2(x2 - x3, y2 - y3);
  if (theta >= M_PI)
  theta -= 2 * M_PI;
  if (theta <= -M_PI)
  theta += 2 * M_PI;
  theta = abs(theta * 180.0 / M_PI);
  return theta;
}

std::map<__int16_t, std::map<__int16_t, match>> TripletGraph::get_vertex_matches(Descriptors descriptor1, Descriptors descriptor2)
{
  omp_set_num_threads(omp_num_threads_vetex_match_);
  std::map<__int16_t, std::map<__int16_t, match>> output;

  // Create a vector of keys to iterate over in parallel
  std::vector<__int16_t> keys;
  for (const auto& pair : descriptor1.vertex_descriptors) {
    keys.push_back(pair.first);
  }

  // Parallelize the outer loop
  #pragma omp parallel for
  for (int i = 0; i < keys.size(); ++i) {
    __int16_t key = keys[i];
    auto& vertex_desc_map1 = descriptor1.vertex_descriptors[key];
    match match_result;   // single match

    // Check whether instances with class-(key) exist in frame-2
    if (descriptor2.vertex_descriptors.count(key) == 1)              // trunk to trunk, pole to pole, etc.
    {
      auto& vertex_desc_map2 = descriptor2.vertex_descriptors[key];  // vertex descriptors in graph-2 with label-(key)

      for (const auto& pair1 : vertex_desc_map1)                     // vertex in graph-1 (class: key, id: pair1.first)
      {
        Eigen::MatrixXi des_1 = pair1.second;                        // i.e., Des_{v^{1}_{j}}

        std::vector<double> sims;
        std::vector<__int16_t> ids;

        // Compare current vertex in graph-1 with all vertices in graph-2 that have the same class label
        for (const auto& pair2 : vertex_desc_map2) {
          Eigen::MatrixXi des_2 = pair2.second;                     // i.e., Des_{v^{2}_{t}}
          Eigen::MatrixXi dot_multiply;

          dot_multiply = des_1.cwiseProduct(des_2);
          int sum_1 = dot_multiply.sum();

          int sum_square1 = des_1.squaredNorm();
          int sum_square2 = des_2.squaredNorm();

          double sim = sum_1 / ((double)std::sqrt(sum_square1) * (double)std::sqrt(sum_square2) + 1e-10);

          sims.push_back(sim);
          ids.push_back(pair2.first);
        }

        if (sims.size() > 1) {
          std::vector<int> sort_indx = argsort(sims);  // descending sort
          match_result.id = ids[sort_indx[0]];         // top-1 match
          match_result.available = true;
          match_result.similarity = sims[sort_indx[0]];
        }
        else if (sims.size() == 1) {
          match_result.id = ids[0];
          match_result.available = true;
          match_result.similarity = sims[0];
        }
        else {
          match_result.available = false;
        }

        // Use critical section to avoid race conditions when writing to output
        #pragma omp critical
        {
          output[key][pair1.first] = match_result;
        }
      }
    }
    else {
      for (const auto& pair1 : vertex_desc_map1) {
        match_result.available = false;

        // Use critical section to avoid race conditions when writing to output
        #pragma omp critical
        {
          output[key][pair1.first] = match_result;
        }
      }
    }
  }

  return output;
}

pose_est TripletGraph::solver_svd(std::vector<match_xyz_label> matches)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr vertexs_1(new pcl::PointCloud<pcl::PointXYZ>());            //vertices in graph-1
  pcl::PointCloud<pcl::PointXYZ>::Ptr vertexs_2(new pcl::PointCloud<pcl::PointXYZ>());            //vertices in graph-2
  pcl::PointXYZ vertex_in_1, vertex_in_2;

  for (size_t i = 0; i < matches.size(); ++i)
  {
    vertex_in_1.x = matches[i].x1;
    vertex_in_1.y = matches[i].y1;
    vertex_in_1.z = matches[i].z1;
    vertex_in_2.x = matches[i].x2;
    vertex_in_2.y = matches[i].y2;
    vertex_in_2.z = matches[i].z2;
    vertexs_1->points.push_back(vertex_in_1);
    vertexs_2->points.push_back(vertex_in_2);
  }


  pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ> TESVD;
  pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ>::Matrix4 transformation;
  TESVD.estimateRigidTransformation( *vertexs_2, *vertexs_1, transformation);  

  pose_est output;
  Eigen::Matrix4d relative_T_svd;
  relative_T_svd   << transformation(0,0), transformation(0,1), transformation(0,2), transformation(0,3),
                      transformation(1,0), transformation(1,1), transformation(1,2), transformation(1,3),
                      transformation(2,0), transformation(2,1), transformation(2,2), transformation(2,3),
                      0.0                , 0.0                , 0.0                , 1.0                ;

  Eigen::Quaterniond Q_svd;
  Q_svd = relative_T_svd.block<3,3>(0,0);

  Eigen::Quaterniond rot_q(Q_svd.w(), Q_svd.x(),Q_svd.y(), Q_svd.z());
  Eigen::Vector3d trans(transformation(0,3),transformation(1,3),transformation(2,3));

  output.ori_Q = rot_q;
  output.trans = trans;
  
  return output;
} 


pose_est TripletGraph::pose_solver(std::vector<match_xyz_label> matches, Eigen::Quaterniond init_Q, Eigen::Vector3d init_xyz)
{
  std::map<__int16_t, double> residual_wieght = conf_para_.weights_cere_cost;     

  double para_q[4] = {init_Q.x(), init_Q.y(), init_Q.z(), init_Q.w()};    // set initial value
  double para_t[3] = {init_xyz.x(), init_xyz.y(), init_xyz.z()};          // set initial value
  Eigen::Map<Eigen::Quaterniond> q_2_to_1(para_q);                        
  Eigen::Map<Eigen::Vector3d> t_2_to_1(para_t);                           

  //setting
  ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);                                  
  //ceres::LocalParameterization *q_parameterization = new ceres::EigenQuaternionParameterization();  //set LocalParameterization，adding operation for Quaternion
  ceres::Manifold *q_parameterization = new ceres::EigenQuaternionManifold();
  ceres::Problem::Options problem_options;                                                          
  ceres::Problem problem(problem_options);

  //add redisual error
  for (size_t i = 0; i < matches.size(); i++)
  {
    double weight = residual_wieght[matches[i].label];
    Eigen::Vector3d vertex_frame1(matches[i].x1, matches[i].y1, matches[i].z1);
    Eigen::Vector3d vertex_frame2(matches[i].x2, matches[i].y2, matches[i].z2);
    ceres::CostFunction *cost_function = p2pFactor::Create(vertex_frame1, vertex_frame2, weight);   //factory moudel to build cost_function 
    problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);                         //add redisual
  }
  

  //when use LocalParameterization，these are required 
  problem.AddParameterBlock(para_q, 4, q_parameterization); // para_q，dim=4
  problem.AddParameterBlock(para_t, 3);                     // para_t，dim=3

  //set colver
  TicToc t_cere_solve;  //solving time 
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;    
  options.max_num_iterations = conf_para_.cere_opt_iterations_max;
  options.minimizer_progress_to_stdout = false; 

  ceres::Solver::Summary summary;

  //start optimization
  ceres::Solve(options, &problem, &summary);

  pose_est pose_result;
  Eigen::Quaterniond rot_q(q_2_to_1.w(), q_2_to_1.x(),q_2_to_1.y(), q_2_to_1.z());
  Eigen::Vector3d trans(t_2_to_1.x(),t_2_to_1.y(),t_2_to_1.z());

  pose_result.ori_Q = rot_q;
  pose_result.trans = trans;
  
  return pose_result;
}

void TripletGraph::get_random_samples(const std::vector<match_xyz_label>& matched_pairs, std::vector<match_xyz_label>& samples, int sample_amount, std::mt19937& rng) {
    std::vector<int> indices(matched_pairs.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);
    samples.clear();
    for (int i = 0; i < sample_amount; ++i) {
        samples.push_back(matched_pairs[indices[i]]);
    }
}

std::pair<Eigen::Matrix4d, Eigen::Matrix4d> TripletGraph::pose_estimate_omp(std::map<__int16_t,std::map<__int16_t, match>> matches, std::map<__int16_t,std::map<__int16_t, instance_center>> ins_cens1, std::map<__int16_t,std::map<__int16_t, instance_center>> ins_cens2)
{
  omp_set_num_threads(omp_num_threads_ransc_);

  std::map<__int16_t,std::map<__int16_t, match>>::iterator iter;
  iter = matches.begin();                                 

  //load matches
  std::vector<int> match_ids;                     
  std::vector<match_xyz_label> matched_pairs;
  int match_xyz_count = 0;
  for (iter; iter != matches.end(); ++iter)
  {
    std::map<__int16_t, match>::iterator iter1;
    iter1 = iter->second.begin();
    //std::cout<<"class: "<<iter11->first<<std::endl;

    for (iter1; iter1 != iter->second.end(); ++iter1)
    {
      if (iter1->second.available)
      {
        match_xyz_label one_matche;
        one_matche.label = iter->first;

        //vertex's coordinate in lidar frame1
        one_matche.x1 =ins_cens1[iter->first][iter1->first].x;
        one_matche.y1 =ins_cens1[iter->first][iter1->first].y;
        one_matche.z1 =ins_cens1[iter->first][iter1->first].z;

        //matched vertex's coordinate in lidar frame2
        one_matche.x2 =  ins_cens2[iter->first][iter1->second.id].x;
        one_matche.y2 =  ins_cens2[iter->first][iter1->second.id].y;
        one_matche.z2 =  ins_cens2[iter->first][iter1->second.id].z;

        match_ids.push_back(match_xyz_count);
        matched_pairs.push_back(one_matche);
        match_xyz_count = match_xyz_count + 1;
      }
    }
  //std::cout<<std::endl;  
  }

  // Create a random number generator
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(0, match_xyz_count - 1);

  //number of matches needed for RANSAC solving
  int sample_amount = int (conf_para_.percentage_matches_used * match_xyz_count);
  if (sample_amount < 3)
  {
    std::cout<<"Error, more matches are needed!"<<std::endl;       
  }

  // Initialize best results
  int best_count = 0;
  Eigen::Matrix4d best_pose;  
  best_pose << 1.0, 0.0, 0.0, 0.0,
              0.0, 1.0, 0.0, 0.0,
              0.0, 0.0, 1.0, 0.0,
              0.0, 0.0, 0.0, 1.0;

  std::vector<match_xyz_label> best_inliners;

  // RANSAC + SVD
  TicToc t_ransac_svd;
  int max_iterations = conf_para_.maxIterations_ransac;
  double ransac_threshold = conf_para_.ransca_threshold;

  #pragma omp parallel
  {
      // Thread-local variables
      int local_best_count = 0;
      Eigen::Matrix4d local_best_pose = Eigen::Matrix4d::Identity();
      std::vector<match_xyz_label> local_best_inliners;
      std::mt19937 rng(omp_get_thread_num()); // Thread-local random number generator

      #pragma omp for
      for (int i = 0; i < max_iterations; ++i) 
      {
          std::vector<match_xyz_label> matche_samples;
          get_random_samples(matched_pairs, matche_samples, sample_amount, rng);

          // SVD solving
          pose_est pose_2_to_1 = solver_svd(matche_samples);
          Eigen::Matrix3d rot_mat = pose_2_to_1.ori_Q.matrix();
          Eigen::Matrix4d pose_mat;
          pose_mat << rot_mat(0,0), rot_mat(0,1), rot_mat(0,2), pose_2_to_1.trans.x(),
                      rot_mat(1,0), rot_mat(1,1), rot_mat(1,2), pose_2_to_1.trans.y(),
                      rot_mat(2,0), rot_mat(2,1), rot_mat(2,2), pose_2_to_1.trans.z(),
                      0.0         , 0.0         , 0.0         , 1.0;

          // Mark inliers and outliers
          int count = 0;
          std::vector<match_xyz_label> current_inliners;
          #pragma omp parallel for reduction(+:count)
          for (size_t j = 0; j < matched_pairs.size(); ++j)
          {
              Eigen::Vector4d vertex_frame1(matched_pairs[j].x1, matched_pairs[j].y1, matched_pairs[j].z1, 1.0);
              Eigen::Vector4d vertex_frame2_t(matched_pairs[j].x2, matched_pairs[j].y2, matched_pairs[j].z2, 1.0);
              vertex_frame2_t = pose_mat * vertex_frame2_t;

              double project_dis = sqrt(pow(vertex_frame1[0] - vertex_frame2_t[0], 2) +
                                        pow(vertex_frame1[1] - vertex_frame2_t[1], 2) +
                                        pow(vertex_frame1[2] - vertex_frame2_t[2], 2));
              if (project_dis < ransac_threshold)
              {
                  #pragma omp critical
                  {
                      count++;
                      current_inliners.push_back(matched_pairs[j]);
                  }
              }
          }

          // Update local best results
          if (count > local_best_count)
          {
              local_best_pose = pose_mat; // Coarse pose
              local_best_count = count;
              local_best_inliners.swap(current_inliners);
          }
      }

      // Update global best results
      #pragma omp critical
      {
          if (local_best_count > best_count)
          {
              best_pose = local_best_pose;
              best_count = local_best_count;
              best_inliners.swap(local_best_inliners);
          }
      }
  }

  //************* Pose optimization
  Eigen::Matrix3d initial_rot_mat  = best_pose.block<3,3>(0,0);
  Eigen::Vector3d initial_eulerAngle = initial_rot_mat.eulerAngles(0,1,2); //Z-Y-X, RPY
  Eigen::Quaterniond initial_rot_Q = Eigen::Quaterniond(initial_rot_mat);
  Eigen::Vector3d initial_xyz (best_pose(0,3), best_pose(1,3), best_pose(2,3));

  //with best inliners，use ceres to further optimize
  TicToc t_ceres;
  pose_est pose_optimized = pose_solver(best_inliners, initial_rot_Q, initial_xyz);

  Eigen::Matrix4d optimized_pose; 
  Eigen::Matrix3d optimized_rot = pose_optimized.ori_Q.matrix();
  Eigen::Vector3d optimized_eulerAngle = optimized_rot.eulerAngles(0,1,2); //Z-Y-X, RPY
  optimized_pose << optimized_rot(0,0), optimized_rot(0,1), optimized_rot(0,2), pose_optimized.trans.x(),
                    optimized_rot(1,0), optimized_rot(1,1), optimized_rot(1,2), pose_optimized.trans.y(),
                    optimized_rot(2,0), optimized_rot(2,1), optimized_rot(2,2), pose_optimized.trans.z(),
                    0.0               , 0.0               , 0.0               , 1.0                     ;

  std::pair<Eigen::Matrix4d, Eigen::Matrix4d> output;
  output.first  = best_pose;
  output.second = optimized_pose;

  //std::cout<<"\033[34;1m Ceres_opt: Runtime:\033[0m "<<t_ceres.toc()<<"ms"<<std::endl;

  return output;   
}

std::map<__int16_t,std::map<__int16_t, match>> TripletGraph::select_matches(std::map<__int16_t,std::map<__int16_t, match>> origianl_matches, Eigen::Matrix4d ested_pose, std::map<__int16_t,std::map<__int16_t, instance_center>>ins_cen1,std::map<__int16_t,std::map<__int16_t, instance_center>> ins_cen2)
{
  std::map<__int16_t,std::map<__int16_t, match>> refined_match;

  std::map<__int16_t,std::map<__int16_t, match>>::iterator iter;
  iter = origianl_matches.begin();
  int original_matches_amount = 0;
  int filtered_matches_amount = 0;
  for (iter; iter != origianl_matches.end(); ++iter)
  {
    std::map<__int16_t, match>::iterator iter1;
    iter1 = iter->second.begin();
    match refine_match;
    for (iter1; iter1 != iter->second.end(); ++iter1)
    {
      refine_match = iter1->second;
      if (iter1->second.available)
      {
        original_matches_amount = original_matches_amount +1;
        instance_center ins_in_1 = ins_cen1[iter->first][iter1->first];
        Eigen::Vector4d vertex1(ins_in_1.x, ins_in_1.y, ins_in_1.z ,1.0);

        instance_center ins_in_2 = ins_cen2[iter->first][iter1->second.id];
        Eigen::Vector4d vertex2(ins_in_2.x, ins_in_2.y, ins_in_2.z ,1.0);

        Eigen::Vector4d ver2_in_frame1 = ested_pose * vertex2;

        double diff =sqrt( (vertex1.x()-ver2_in_frame1.x())*(vertex1.x()-ver2_in_frame1.x()) + (vertex1.y()-ver2_in_frame1.y())*(vertex1.y()-ver2_in_frame1.y()) + (vertex1.z()-ver2_in_frame1.z())*(vertex1.z()-ver2_in_frame1.z())  );

        if (diff>= conf_para_.similarity_refine_thre)
        {
          refine_match.available = false;
          filtered_matches_amount = filtered_matches_amount +1;
        }
      }

      refined_match[iter->first][iter1->first] = refine_match;
    }
  }
  //std::cout<<"original matches amount = "<<original_matches_amount<<"; matches amount for global descriptor="<< original_matches_amount-filtered_matches_amount<<std::endl;
  return refined_match;
}


float TripletGraph::cal_similarity(Descriptors descriptor1, Descriptors descriptor2)
{
  std::map<__int16_t, float> same_class_similarities;
  std::map<__int16_t, Eigen::MatrixXi>::iterator  iter;
  iter = descriptor1.global_descriptor.begin();

  for (iter; iter != descriptor1.global_descriptor.end(); ++iter)
  {
    Eigen::MatrixXi descri1 = iter->second;
    Eigen::MatrixXi descri2 = descriptor2.global_descriptor[iter->first];

    Eigen::MatrixXi dot_multipy;
          
    dot_multipy = descri1.cwiseProduct(descri2); 
    int sum_1   = dot_multipy.sum();

    int sum_square1 = descri1.squaredNorm();
    int sum_square2 = descri2.squaredNorm();

    float sim = sum_1/( (float)sqrt(sum_square1) * (float)sqrt(sum_square2) + 1e-10);

    same_class_similarities[iter->first] = sim;
  }
  
  //fianl similarity
  std::map<__int16_t, float>::iterator sim_iter;
  sim_iter = same_class_similarities.begin();

  float sim = 0.0;
  for (sim_iter; sim_iter != same_class_similarities.end(); ++sim_iter)
  {
    sim = sim + conf_para_.weights_for_class[sim_iter->first] * sim_iter->second;
  }

  return sim;
}


float TripletGraph::cal_refined_similarity(Descriptors descriptor1, Descriptors descriptor2, std::map<__int16_t,std::map<__int16_t, match>> filtered_match)
{
  int bin_amount = 180 / conf_para_.angle_resolution;

  std::map<__int16_t, int> qualified_counts;                         

  std::map<__int16_t, Eigen::MatrixXi>  overall_descriptors_1;    
  std::map<__int16_t, Eigen::MatrixXi>  overall_descriptors_2;    
  for (size_t i = 0; i < conf_para_.classes_for_graph.size(); ++i)  
  {
    qualified_counts[conf_para_.classes_for_graph[i]] = 0;

    Eigen::MatrixXi single_class_overall_des;
    single_class_overall_des.resize(class_combination_infos_[conf_para_.classes_for_graph[i]].dim_combination, bin_amount);
    single_class_overall_des.setZero();
    overall_descriptors_1[conf_para_.classes_for_graph[i]] = single_class_overall_des;
    overall_descriptors_2[conf_para_.classes_for_graph[i]] = single_class_overall_des;
  }

  //claculate {Des^l} for matches after projection selection operation
  std::map<__int16_t,std::map<__int16_t, match>>::iterator iter;
  iter = filtered_match.begin();
  int qualified_count_total = 0;

  for (iter; iter != filtered_match.end(); ++iter)
  {
    std::map<__int16_t, match>::iterator iter1;
    iter1 = iter->second.begin();
    int qualified_count_one_class =0;

    for (iter1; iter1 != iter->second.end(); ++iter1)
    {
      if (iter1->second.available)
      {   
        overall_descriptors_1[iter->first] = overall_descriptors_1[iter->first] + descriptor1.vertex_descriptors[iter->first][iter1->first];
        overall_descriptors_2[iter->first] = overall_descriptors_2[iter->first] + descriptor2.vertex_descriptors[iter->first][iter1->second.id];

        qualified_count_one_class = qualified_count_one_class + 1;
        qualified_count_total     = qualified_count_total + 1;
      } 
    }

    qualified_counts[iter->first] = qualified_count_one_class;
  }

  //for each class, calculate overall similarity 
  std::map<__int16_t, float> overall_similarities;
  std::map<__int16_t, Eigen::MatrixXi>::iterator  iter_2;
  iter_2 = overall_descriptors_1.begin();

  for (iter_2; iter_2 != overall_descriptors_1.end(); ++iter_2)
  {
    Eigen::MatrixXi des_1 = iter_2->second;
    Eigen::MatrixXi des_2 = overall_descriptors_2[iter_2->first];

    Eigen::MatrixXi dot_multipy;
          
    dot_multipy = des_1.cwiseProduct(des_2);  
    int sum_1   = dot_multipy.sum();

    int sum_square1 = des_1.squaredNorm();
    int sum_square2 = des_2.squaredNorm();

    float sim = sum_1/( (float)sqrt(sum_square1) * (float)sqrt(sum_square2) + 1e-10);

    overall_similarities[iter_2->first] = sim;

    //std::cout<<"class-"<<iter_2->first<<" , sim="<<sim <<" , quali count="<<qualified_counts[iter_2->first]<<std::endl;
  }

  //cal final similarity
  std::map<__int16_t, float>::iterator sim_iter;
  sim_iter = overall_similarities.begin();

  float sim = 0.0;
  for (sim_iter; sim_iter != overall_similarities.end(); ++sim_iter)
  {
    float penalty_factor = 1.0;

    if (qualified_counts[sim_iter->first] == 0)         
    {
      penalty_factor = 0.0;
    }
    //std::cout<<"class-"<<sim_iter->first<<" , penalty="<<penalty_factor<<std::endl;
    sim = sim + penalty_factor * conf_para_.weights_for_class[sim_iter->first] * sim_iter->second;
  }

  return sim;
}