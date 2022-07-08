#pragma once
#include "branch_tree_node.hpp"
#include "ioctree_node.hpp"
#include "leaf_tree_node.hpp"
#include "marching_cubes.hpp"
#include "object.hpp"
#include "tree_config.hpp"
#include "tree_node.hpp"
#include "triangle.hpp"
#include "vector.hpp"
#include <algorithm>
#include <array>
#include <boost/pool/object_pool.hpp>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <functional>
#include <memory>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <random>
#include <vector>
#include "update_tree.h"
#include <dirent.h>
#include <cstring>
#include <iostream>
#include <fstream>

#define PI 3.14159265358979323846
#define print(x) std::cout << x << std::endl;
#define printl(x, y) std::cout << x << " " << y << std::endl;

namespace dmc
{
//	template <class Scalar>
	class ioctree
	{
	public:
//		typedef Scalar scalar_type;
		typedef vector<float, 3> vector_type;
		typedef vector<float, 4> vector4_type;
		typedef vertex<float> vertex_type;
		typedef ioctree_config config_type;
		typedef ioctree_node<float> ioctree_node_type;
		typedef triangle<vector_type> triangle_type;


		config_type config_;
		ioctree_node_type root_;
		std::vector<ioctree_node_type*> leaf_vec;
		std::vector<cv::Mat> depth_map_vec;
		std::vector<cv::Mat> weight_map_vec;
		std::vector<Eigen::Matrix4f> frame_rt_vec;
		std::vector<int> frame_id;
		cv::Mat mask_depth;

		float* temp_xyz;

		float min_voxel_size;
		float max_voxel_size;
		float mean_depth_value;
		float median_depth_value;
		double depth_value_cnt;
		float voxel_scale;
		float width;
		float user_max_idepth;
		float depth_scale;
		int test_cnt;
		float smallest_radius;
		std::string voxel_path;
		std::string node_distribution_path;
		std::ofstream writeFile;
		int min_sdf_cnt;

		explicit ioctree(const config_type& config = config_type())
			:config_(config)
			,root_(ioctree_node_type(
				  vertex_type(vector_type(-1,-1,-1),-1),
				  true,config.min_phi,config.max_phi,config.min_theta,config.max_theta,config.min_idepth,config.max_idepth))
			,user_max_idepth(config.user_max_idepth)
			,voxel_scale(config.voxel_scale)
			,depth_scale(config.depth_scale)
			,voxel_path(config.voxel_path)
			,node_distribution_path(config.node_distribution_path)
			,min_sdf_cnt(config.min_sdf_cnt)
		{

			test_cnt = 0;
			float mid_min_depth = 1.0f;
			float mid_max_depth = 1.1f;

			mean_depth_value = 0.0f;
			median_depth_value = 0.0f;
			depth_value_cnt = 0.0f;
			smallest_radius = 1.0f;
			width = 0.0f;
			temp_xyz = (float*)malloc(sizeof(float) * 3);
			root_.set_type(0);
			//type 0 => root node
			//type 1 => balanced
			//type 2 => need to be balanced
			cv::Mat mask_depth_ = cv::imread(config.mask_path, -1);
			print(config.mask_path)
			mask_depth_.convertTo(mask_depth_, CV_32F);
			mask_depth = mask_depth_;
			writeFile.open(voxel_path);
		}

		void load_depth_map(std::vector<Eigen::Matrix4f>* frame_rt, std::string depth_path, std::string weight_path, int interval, float sky_depth_threshold){
			// search all the depth files in depth_path
			DIR *dir;
			struct dirent *ent;
			char delimiter = ' ';
			std::string tok;
			int elem_cnt = 0;
			int cnt = 0;
			double ret = 0;
			int id;
			int w, h;
			if ((dir = opendir (depth_path.c_str())) != NULL) {
				while ((ent = readdir (dir)) != NULL) {
					char *token = strtok(ent->d_name, ".");
					cnt = 0;
					while (token != NULL) {
						if(cnt != 0) break;
						std::string str(token);
						frame_id.push_back(std::stoi(str));
						token = strtok(NULL, "-");
						cnt++;
					}
				}
				closedir (dir);
			} else {
				/* could not open directory */
				perror ("");
				return;
			}
			// sort depth maps by name
			std::sort(frame_id.begin(), frame_id.end());

			Eigen::Matrix4f temp_rt;
			std::vector<float> median_vec;
			std::vector<float> depth_median_vec;
			for(int i = 0; i < frame_id.size(); i++){

			    if(interval == 0){
					// do nothing
				}
			    else{
                    if(i%interval != 0) continue;
			    }

				printl("loading... ",i);
				std::string path = depth_path + "/" + std::to_string(frame_id[i]) + ".exr";
				std::string path_w = weight_path + "/" + std::to_string(frame_id[i]) + ".exr";
				cv::Mat depth_map = cv::imread(path, -1);
				cv::Mat weight_map = cv::imread(path_w, -1);

				depth_map = depth_map / depth_scale;

				// if depth is large than threshold, set the value to the threshold
				// need to convert inv depth to depth first
				depth_map = 1.0f / depth_map;
				cv::threshold(depth_map, depth_map, (double)sky_depth_threshold, (double)-1, cv::THRESH_TRUNC);
				depth_map = 1.0f / depth_map;

				float* depth_map_ = (float*)depth_map.data;
				temp_rt = (*frame_rt)[frame_id[i]];
				temp_rt(0,3) = depth_scale * temp_rt(0,3);
				temp_rt(1,3) = depth_scale * temp_rt(1,3);
				temp_rt(2,3) = depth_scale * temp_rt(2,3);
				h = depth_map.size[0];
				w = depth_map.size[1];
				width = w;

				depth_map_vec.push_back(depth_map);
				weight_map_vec.push_back(weight_map);
				frame_rt_vec.push_back(temp_rt);
			}
		}


		void update_tree(Eigen::Matrix4f center_rt, float trunc, float weight_trunc_scale, float trunc_offset, float trunc_change){

			collect_leaf(&root_, 0);

			unsigned int len = leaf_vec.size();
			printl("node num: ", len);

			cuda cu = cuda();
			
			int h, w;
			h = depth_map_vec[0].size[0];
			w = depth_map_vec[0].size[1];


			float* mask_depth_ = (float*)mask_depth.data;
			cu.update_cuda_alloc(len, h * w, depth_map_vec.size());
			cu.update_cuda_alloc(len, h * w, weight_map_vec.size());
			cu.update_cuda_load(6, 0, &trunc, sizeof(float));
			cu.update_cuda_load(11, 0, mask_depth_, h * w * sizeof(float));
			cu.update_cuda_load(13, 0, &weight_trunc_scale, sizeof(float));
			cu.update_cuda_load(14, 0, &trunc_offset, sizeof(float));
			cu.update_cuda_load(16, 0, &trunc_change, sizeof(float));


			float cell_depth;

			float* sdf_value_list = (float*)malloc(len * sizeof(float));
			float* sdf_cnt_list = (float*)malloc(len * sizeof(float));
			float* sdf_weight_list = (float*)malloc(len * sizeof(float));
			float* sdf_center_list = (float*)malloc(3 * len * sizeof(float));


			for(int i = 0; i < len; i++){
                if(i%100000000 == 0) std::cout << "upload node num: " << i << " / " << len-1 << std::endl;
			    memcpy(sdf_value_list + i, &(leaf_vec[i]->sdf_value), sizeof(float));
			    memcpy(sdf_cnt_list + i, &(leaf_vec[i]->sdf_cnt), sizeof(float));
			    memcpy(sdf_weight_list + i, &(leaf_vec[i]->sdf_cnt), sizeof(float));
                vector_type xyz = leaf_vec[i]->get_center();


			    for(int j = 0; j < 3; j++){

                    float e = xyz[j];
                    memcpy(sdf_center_list + 3 * i + j, &e, sizeof(float));
			    }
			}

			cu.update_cuda_load(8, 0, sdf_value_list, len * sizeof(float));
			cu.update_cuda_load(9, 0, sdf_cnt_list, len * sizeof(float));
			cu.update_cuda_load(10, 0, sdf_center_list, 3 * len * sizeof(float));
			cu.update_cuda_load(12, 0, sdf_weight_list, 3 * len * sizeof(float));

			Eigen::Matrix4f center_inv = center_rt.inverse();


#pragma omp parallel for schedule(dynamic)
			for(int i = 0; i < depth_map_vec.size(); i++){
				float* depth_map = (float*)depth_map_vec[i].data;
				float* weight_map = (float*)weight_map_vec[i].data;
				cu.update_cuda_load(4, h * w * i, depth_map, h * w * sizeof(float));
				cu.update_cuda_load(15, h * w * i, depth_map, h * w * sizeof(float));
			}

#pragma omp parallel for schedule(dynamic)
			for(int k = 0; k < depth_map_vec.size(); k++){
				Eigen::Matrix4f frame_rt = frame_rt_vec[k];
				frame_rt = frame_rt * center_inv;
				for(int i = 0; i < 4; i++){
					for(int j = 0; j < 4; j++){
						float elem = frame_rt(i,j);
						cu.update_cuda_load(5, 16 * k + (i * 4 + j), &elem, sizeof(float));
					}
				}
			}

			// update
			print("Start update");
			cu.update_cuda_sdf((unsigned int)len, depth_map_vec.size(), h, w, (float*)mask_depth.data);
			print("End update");

			cu.update_cuda_unload(3, sdf_value_list, 0, len * sizeof(float));
			cu.update_cuda_unload(4, sdf_cnt_list, 0, len * sizeof(float));
			cu.update_cuda_unload(5, sdf_center_list, 0, 3 * len * sizeof(float));

			for(int i = 0; i < len; i++){
                if(i%10000000 == 0) std::cout << "download num: " << i << " / " << len-1 << std::endl;
                memcpy(&(leaf_vec[i]->sdf_value), sdf_value_list + i, sizeof(float));
                memcpy(&(leaf_vec[i]->sdf_cnt), sdf_cnt_list + i, sizeof(float));
				vector_type xyz(sdf_center_list[3 * i + 0], sdf_center_list[3 * i + 1], sdf_center_list[3 * i + 2]);

				leaf_vec[i]->set_vertex(xyz, leaf_vec[i]->sdf_value);

				if(leaf_vec[i]->sdf_value > 0){
					writeFile << "v " << std::to_string(xyz[0]) << " " << std::to_string(xyz[1]) << " " << std::to_string(xyz[2]) << " 255 0 0\n";
				}
				else{
					writeFile << "v " << std::to_string(xyz[0]) << " " << std::to_string(xyz[1]) << " " << std::to_string(xyz[2]) << " 0 0 255\n";
				}
			}
			free(sdf_value_list);
			free(sdf_cnt_list);
			free(sdf_center_list);
			writeFile.close();

			print("Update Done")
		}

		Eigen::Vector2i to_uv(Eigen::Vector3f xyz, int h, int w){
			xyz = xyz / xyz.norm();
			Eigen::Vector2i uv;
			int u,v;
			float phi = (float)(3 * PI) / 2 - atan2f(xyz(2,0), xyz(0,0));
			float theta = acosf(-xyz(1,0));
			u = round(theta * (float)h / PI - 0.5f);
			v = round(phi * (float)w / (2 * PI) - 0.5f);
			u = u < 0? (h + u) : u; u = u >= h ? (u - h) : u;
			v = v < 0? (w + v) : v; v = v >= w ? (v - w) : v;
			uv(0,0) = u;
			uv(1,0) = v;
			return uv;
		}


		void collect_leaf(ioctree_node_type* node, int level){
			if(node->get_is_leaf()){

				leaf_vec.push_back(node);
			}
			else{
				if(node->get_type() == 2){
					collect_leaf(node->children()[0], level + 1);
					collect_leaf(node->children()[1], level + 1);
				}
				else if(node->get_type() == 1){
					collect_leaf(node->children()[0], level + 1);
					collect_leaf(node->children()[1], level + 1);
					collect_leaf(node->children()[2], level + 1);
					collect_leaf(node->children()[3], level + 1);
					collect_leaf(node->children()[4], level + 1);
					collect_leaf(node->children()[5], level + 1);
					collect_leaf(node->children()[6], level + 1);
					collect_leaf(node->children()[7], level + 1);
				}
				else{
					print("WARNING");
				}

			}
		}


		void generate_tree(Eigen::Matrix4f center_rt, float resolution, int node_sample_interval){
			int h, w;
			float* depth_map;
			float d;
			float mask;
			vector_type ipoint;
			Eigen::Matrix4f frame_to_center_rt;

			float * mask_d = (float*)mask_depth.data;

			for(int frame_num = 0; frame_num < depth_map_vec.size(); frame_num++){

				depth_map = (float*)depth_map_vec[frame_num].data;
				h = depth_map_vec[frame_num].size[0];
				w = depth_map_vec[frame_num].size[1];

				// we need rt to change 3D points from each view to center view
				frame_to_center_rt = center_rt * frame_rt_vec[frame_num].inverse();
				printl("Generating tree -- frame_num", frame_num);

				for(int i = 0; i < h; i++){
					for(int j = 0; j < w; j++){
						if(i % node_sample_interval == 0 && j % node_sample_interval == 0){

							d = 1.0f / depth_map[i * w + j];
							mask = mask_d[i * w + j];
							if(mask < 1) continue;
							if(d < 0) continue;

							ipoint = to_phi_theta_invd(i, j, d, h, w, frame_to_center_rt);
							
							insert_point(&root_, ipoint, d, 0, resolution);
						}
					}
				}
			}
		}


		bool check_node_is_balanced(ioctree_node_type* node){

			if(1.4f * (node->max_phi - node->min_phi) * ((1.0f/node->min_idepth + 1.0f/node->max_idepth)/2.0f) <
				(1.0f/node->min_idepth - 1.0f/node->max_idepth)){
				return false;
//			if(1.4f * (node->max_phi - node->min_phi) * (1.0f/sqrtf(node->min_idepth * node->max_idepth)) <
//			   (1.0f/node->min_idepth - 1.0f/node->max_idepth)){
//				return false;
			}
			else{
				return true;
			}
		}

		void insert_point(ioctree_node_type* node, vector_type ipoint, float d_resolution_cue, int level, float resolution)
		{
			int insert_child = -1;

			if(check_point_is_inside_node(node, ipoint)){
				if(node->get_type() == 0){
					float mid_phi = (node->min_phi + node->max_phi) / 2;
					float mid_theta = (node->min_theta + node->max_theta) / 2;
					
					float mid_idepth = 1.0f/sqrtf((1.0f/node->min_idepth) * (1.0f/node->max_idepth));
//					float mid_idepth = sqrtf(node->min_idepth * node->max_idepth);
//					float mid_idepth = 1.0f/((1.0f/node->min_idepth + 1.0f/node->max_idepth)/2);
//					float mid_idepth = ((node->min_idepth + node->max_idepth)/2);

					float q1_phi = (node->min_phi + mid_phi) / 2;
					float q3_phi = (mid_phi + node->max_phi) / 2;

					node->set_is_leaf(false);
					node->set_type(1);
					make_type1_children(node, mid_phi, mid_theta, mid_idepth, level + 1);
					insert_child = get_insert_type1_child(ipoint, mid_phi, mid_theta, mid_idepth, q1_phi, q3_phi, level + 1);
					insert_point(node->children()[insert_child], ipoint, d_resolution_cue, level + 1, resolution);
				}
				else{
					if(check_node_is_balanced(node)){
						if(check_node_is_small_enough(node, d_resolution_cue, resolution)){
							// do nothing
						}
						else{ // balanced, not small enough
							float mid_phi = (node->min_phi + node->max_phi) / 2.0f;
							float mid_theta = (node->min_theta + node->max_theta) / 2.0f;
							
							float mid_idepth = 1.0f/sqrtf((1.0f/node->min_idepth) * (1.0f/node->max_idepth));
//							float mid_idepth = sqrtf(node->min_idepth * node->max_idepth);
//							float mid_idepth = ((node->min_idepth + node->max_idepth)/2);
//							float mid_idepth = 1.0f/((1.0f/node->min_idepth + 1.0f/node->max_idepth)/2);

							float q1_phi = (node->min_phi + mid_phi) / 2.0f;
							float q3_phi = (mid_phi + node->max_phi) / 2.0f;
							if (node->get_is_leaf()){
								node->set_is_leaf(false);
								node->set_type(1);
								make_type1_children(node, mid_phi, mid_theta, mid_idepth, level + 1);
							}
							insert_child = get_insert_type1_child(ipoint, mid_phi, mid_theta, mid_idepth, q1_phi, q3_phi, level + 1);
							insert_point(node->children()[insert_child], ipoint, d_resolution_cue, level + 1, resolution);
						}
					}
					else{

						if(check_node_is_small_enough(node, d_resolution_cue, resolution)){
							// do nothing
						}
						else{ // not balanced, not small enough
							float mid_phi = (node->min_phi + node->max_phi) / 2;
							float mid_theta = (node->min_theta + node->max_theta) / 2;

							float mid_idepth = 1.0f/sqrtf((1.0f/node->min_idepth) * (1.0f/node->max_idepth));
//							float mid_idepth = sqrtf(node->min_idepth * node->max_idepth);
//							float mid_idepth = ((node->min_idepth + node->max_idepth)/2);
//							float mid_idepth = 1.0f/((1.0f/node->min_idepth + 1.0f/node->max_idepth)/2);

							float q1_phi = (node->min_phi + mid_phi) / 2;
							float q3_phi = (mid_phi + node->max_phi) / 2;
							float rr = (1.0f/node->min_idepth - 1.0f/node->max_idepth)/2;

							if (node->get_is_leaf()){
								node->set_is_leaf(false);
								node->set_type(2);
								make_type2_children(node, mid_phi, mid_theta, mid_idepth, level + 1);
							}
							insert_child = get_insert_type2_child(ipoint, mid_phi, mid_theta, mid_idepth, q1_phi, q3_phi, level + 1);
							insert_point(node->children()[insert_child], ipoint, d_resolution_cue, level + 1, resolution);
						}
					}
				}

			}

		}

		void make_type2_children(ioctree_node_type* node, float mid_phi, float mid_theta, float mid_idepth, int level){
			ioctree_node_type* child0;
			ioctree_node_type* child1;
			ioctree_node_type* child2;
			ioctree_node_type* child3;
			ioctree_node_type* child4;
			ioctree_node_type* child5;
			ioctree_node_type* child6;
			ioctree_node_type* child7;
			if(level == 1){
				child0 = new ioctree_node_type(vertex_type(vector_type(-1, -1, -1), 1), true, (mid_phi + node->max_phi)/2, node->max_phi, mid_theta, node->max_theta, node->min_idepth, node->max_idepth);
				child1 = new ioctree_node_type(vertex_type(vector_type(-1, -1, -1), 1), true, mid_phi, (mid_phi + node->max_phi)/2, mid_theta, node->max_theta, node->min_idepth, node->max_idepth);
				child2 = new ioctree_node_type(vertex_type(vector_type(-1, -1, -1), 1), true, (node->min_phi + mid_phi)/2, mid_phi, mid_theta, node->max_theta, node->min_idepth, node->max_idepth);
				child3 = new ioctree_node_type(vertex_type(vector_type(-1, -1, -1), 1), true, node->min_phi, (node->min_phi + mid_phi)/2, mid_theta, node->max_theta, node->min_idepth, node->max_idepth);
				child4 = new ioctree_node_type(vertex_type(vector_type(-1, -1, -1), 1), true, (mid_phi + node->max_phi)/2, node->max_phi, node->min_theta, mid_theta, node->min_idepth, node->max_idepth);
				child5 = new ioctree_node_type(vertex_type(vector_type(-1, -1, -1), 1), true, mid_phi, (mid_phi + node->max_phi)/2, node->min_theta, mid_theta, node->min_idepth, node->max_idepth);
				child6 = new ioctree_node_type(vertex_type(vector_type(-1, -1, -1), 1), true, (node->min_phi + mid_phi)/2, mid_phi, node->min_theta, mid_theta, node->min_idepth, node->max_idepth);
				child7 = new ioctree_node_type(vertex_type(vector_type(-1, -1, -1), 1), true, node->min_phi, (node->min_phi + mid_phi)/2, node->min_theta, mid_theta, node->min_idepth, node->max_idepth);
				node->children()[0] = child0;
				node->children()[1] = child1;
				node->children()[2] = child2;
				node->children()[3] = child3;
				node->children()[4] = child4;
				node->children()[5] = child5;
				node->children()[6] = child6;
				node->children()[7] = child7;
			}
			else{
				ioctree_node_type* child0 = new ioctree_node_type(vertex_type(vector_type(-1, -1, -1), 1), true, node->min_phi, node->max_phi, node->min_theta, node->max_theta, mid_idepth, node->max_idepth);
				ioctree_node_type* child1 = new ioctree_node_type(vertex_type(vector_type(-1, -1, -1), 1), true, node->min_phi, node->max_phi, node->min_theta, node->max_theta, node->min_idepth, mid_idepth);
				node->children()[0] = child0;
				node->children()[1] = child1;
			}
		}

		int get_insert_type2_child(vector_type point , float mid_phi, float mid_theta, float mid_idepth, float q1_phi, float q3_phi, int level){
			float phi = point[0];
			float theta = point[1];
			float idepth = point[2];

			if(level == 1){
				if(theta > mid_theta){
					if(phi < q1_phi){
						return 3;
					}
					else if(phi < mid_phi){
						return 2;
					}
					else if(phi < q3_phi){
						return 1;
					}
					else{
						return 0;
					}
				}
				else{
					if(phi < q1_phi){
						return 7;
					}
					else if(phi < mid_phi){
						return 6;
					}
					else if(phi < q3_phi){
						return 5;
					}
					else{
						return 4;
					}
				}
			}
			else{
				if(idepth < mid_idepth){
					return 1;
				}
				else{
					return 0;
				}
			}

		}

		int get_insert_type1_child(vector_type point , float mid_phi, float mid_theta, float mid_idepth, float q1_phi, float q3_phi, int level){
			float phi = point[0];
			float theta = point[1];
			float idepth = point[2];
			if(level == 1){
				if(theta > mid_theta){
					if(phi < q1_phi){
						return 3;
					}
					else if(phi < mid_phi){
						return 2;
					}
					else if(phi < q3_phi){
						return 1;
					}
					else{
						return 0;
					}
				}
				else{
					if(phi < q1_phi){
						return 7;
					}
					else if(phi < mid_phi){
						return 6;
					}
					else if(phi < q3_phi){
						return 5;
					}
					else{
						return 4;
					}
				}
			}
			else{
				if(phi < mid_phi){
					if(theta > mid_theta){
						if(idepth < mid_idepth){
							return 5;
						}
						else{
							return 1;
						}
					}
					else{
						if(idepth < mid_idepth){
							return 7;
						}
						else{
							return 3;
						}
					}
				}
				else{
					if(theta > mid_theta){
						if(idepth < mid_idepth){
							return 4;
						}
						else{
							return 0;
						}
					}
					else{
						if(idepth < mid_idepth){
							return 6;
						}
						else{
							return 2;
						}
					}
				}

			}

		}

		void make_type1_children(ioctree_node_type* node, float mid_phi, float mid_theta, float mid_idepth, int level){

			ioctree_node_type* child0;
			ioctree_node_type* child1;
			ioctree_node_type* child2;
			ioctree_node_type* child3;
			ioctree_node_type* child4;
			ioctree_node_type* child5;
			ioctree_node_type* child6;
			ioctree_node_type* child7;
			if(level == 1){
				child0 = new ioctree_node_type(vertex_type(vector_type(-1, -1, -1), 1), true, (mid_phi + node->max_phi)/2, node->max_phi, mid_theta, node->max_theta, node->min_idepth, node->max_idepth);
				child1 = new ioctree_node_type(vertex_type(vector_type(-1, -1, -1), 1), true, mid_phi, (mid_phi + node->max_phi)/2, mid_theta, node->max_theta, node->min_idepth, node->max_idepth);
				child2 = new ioctree_node_type(vertex_type(vector_type(-1, -1, -1), 1), true, (node->min_phi + mid_phi)/2, mid_phi, mid_theta, node->max_theta, node->min_idepth, node->max_idepth);
				child3 = new ioctree_node_type(vertex_type(vector_type(-1, -1, -1), 1), true, node->min_phi, (node->min_phi + mid_phi)/2, mid_theta, node->max_theta, node->min_idepth, node->max_idepth);
				child4 = new ioctree_node_type(vertex_type(vector_type(-1, -1, -1), 1), true, (mid_phi + node->max_phi)/2, node->max_phi, node->min_theta, mid_theta, node->min_idepth, node->max_idepth);
				child5 = new ioctree_node_type(vertex_type(vector_type(-1, -1, -1), 1), true, mid_phi, (mid_phi + node->max_phi)/2, node->min_theta, mid_theta, node->min_idepth, node->max_idepth);
				child6 = new ioctree_node_type(vertex_type(vector_type(-1, -1, -1), 1), true, (node->min_phi + mid_phi)/2, mid_phi, node->min_theta, mid_theta, node->min_idepth, node->max_idepth);
				child7 = new ioctree_node_type(vertex_type(vector_type(-1, -1, -1), 1), true, node->min_phi, (node->min_phi + mid_phi)/2, node->min_theta, mid_theta, node->min_idepth, node->max_idepth);
			}
			else{
				child0 = new ioctree_node_type(vertex_type(vector_type(-1, -1, -1), 1), true, mid_phi, node->max_phi, mid_theta, node->max_theta, mid_idepth, node->max_idepth);
				child1 = new ioctree_node_type(vertex_type(vector_type(-1, -1, -1), 1), true, node->min_phi, mid_phi, mid_theta, node->max_theta, mid_idepth, node->max_idepth);
				child2 = new ioctree_node_type(vertex_type(vector_type(-1, -1, -1), 1), true, mid_phi, node->max_phi, node->min_theta, mid_theta, mid_idepth, node->max_idepth);
				child3 = new ioctree_node_type(vertex_type(vector_type(-1, -1, -1), 1), true, node->min_phi, mid_phi, node->min_theta, mid_theta, mid_idepth, node->max_idepth);
				child4 = new ioctree_node_type(vertex_type(vector_type(-1, -1, -1), 1), true, mid_phi, node->max_phi, mid_theta, node->max_theta, node->min_idepth, mid_idepth);
				child5 = new ioctree_node_type(vertex_type(vector_type(-1, -1, -1), 1), true, node->min_phi, mid_phi, mid_theta, node->max_theta, node->min_idepth, mid_idepth);
				child6 = new ioctree_node_type(vertex_type(vector_type(-1, -1, -1), 1), true, mid_phi, node->max_phi, node->min_theta, mid_theta, node->min_idepth, mid_idepth);
				child7 = new ioctree_node_type(vertex_type(vector_type(-1, -1, -1), 1), true, node->min_phi, mid_phi, node->min_theta, mid_theta, node->min_idepth, mid_idepth);
			}

			node->children()[0] = child0;
			node->children()[1] = child1;
			node->children()[2] = child2;
			node->children()[3] = child3;
			node->children()[4] = child4;
			node->children()[5] = child5;
			node->children()[6] = child6;
			node->children()[7] = child7;
		}

		// ipoint is phi theta idepth
		bool check_point_is_inside_node(ioctree_node_type* node, vector_type ipoint){
			if((node->min_phi <= ipoint[0]) && (ipoint[0] < node->max_phi) && (node->min_theta <= ipoint[1])
				&& (ipoint[1] < node->max_theta) && (node->min_idepth <= ipoint[2]) && (ipoint[2] < node->max_idepth)){
				return true;
			}
			else{
				return false;
			}
		}

		bool check_node_is_small_enough(ioctree_node_type* node, float d_resolution_cue, float resolution){
			float node_size = node->get_node_size();
			float radius = powf(node_size * 3.0f / (4.0f * PI), 1.0f/3.0f);

			if(radius >= d_resolution_cue)
			{
				return false;
			}

			float angle = asinf(radius / d_resolution_cue);
			float solid_angle = 4 * PI * sinf(angle / 2.0f) * sinf(angle / 2.0f);

			if(solid_angle <= resolution){
				return true;
			}
			else{
				return false;
			}


		}


		// return phi, theta, invd from center point of view
		vector_type to_phi_theta_invd(int i , int j, float d, int h, int w, Eigen::Matrix4f frame_to_center_rt){
			
			float theta = (float)PI * (float)(i+0.5f) / h;
			float phi = (float)(3 * PI) / 2 - 2 * PI * (float)(j+0.5f) / w;
			float new_theta;
			float new_phi;
			float new_depth;

			Eigen::Vector4f xyzw = Eigen::Vector4f(d * sinf(theta) * cosf(phi), -d * cosf(theta), d * sinf(theta) * sinf(phi), 1);
			Eigen::Vector4f temp1 = Eigen::Vector4f(0.0f, 0.0f, 0.0f, 1.0f);
			Eigen::Vector4f temp2 = Eigen::Vector4f(0.0f, 0.0f, 0.0f, 1.0f);

			xyzw = frame_to_center_rt * xyzw;

			temp1(0,0) = xyzw(2,0);
			temp1(1,0) = -xyzw(1,0);
			temp1(2,0) = xyzw(0,0);
			temp2(0,0) = -temp1(0,0);
			temp2(1,0) = temp1(1,0);
			temp2(2,0) = -temp1(2,0);
			Eigen::Vector3f xyz = Eigen::Vector3f(temp2(0,0), temp2(1,0), temp2(2,0));

			new_depth = xyz.norm();
			xyz = xyz / new_depth;
			new_phi = (float)((3 * PI) / 2.0f - atan2f(xyz(2,0), xyz(0,0)));
			new_theta = acosf(xyz(1,0));

			if(new_phi < 0) new_phi = new_phi + 2 * PI;
			if(new_phi > 2 * PI) new_phi = new_phi - 2 * PI;
			if(new_theta < 0) new_theta = new_theta + 2 * PI;
			if(new_theta > 2 * PI) new_theta = new_theta - 2 * PI;

			return vector_type(new_phi, new_theta, 1.0f / new_depth);
		}

		std::vector<triangle_type> enumerate()
		{
			print("Enumerating...")
			std::vector<triangle_type> triangles;

			auto receiver = [&](const triangle_type& t) {
			  triangles.emplace_back(t);
			};

            enumerate_impl_c(root_.children()[0], receiver);
            enumerate_impl_c(root_.children()[1], receiver);
            enumerate_impl_c(root_.children()[2], receiver);
            enumerate_impl_c(root_.children()[3], receiver);
            enumerate_impl_c(root_.children()[4], receiver);
            enumerate_impl_c(root_.children()[5], receiver);
            enumerate_impl_c(root_.children()[6], receiver);
            enumerate_impl_c(root_.children()[7], receiver);

            enumerate_impl_f_x(root_.children()[0], root_.children()[1], receiver);
            enumerate_impl_f_x(root_.children()[1], root_.children()[2], receiver);
            enumerate_impl_f_x(root_.children()[2], root_.children()[3], receiver);
            enumerate_impl_f_x(root_.children()[3], root_.children()[0], receiver);
            enumerate_impl_f_x(root_.children()[4], root_.children()[5], receiver);
            enumerate_impl_f_x(root_.children()[5], root_.children()[6], receiver);
            enumerate_impl_f_x(root_.children()[6], root_.children()[7], receiver);
            enumerate_impl_f_x(root_.children()[7], root_.children()[4], receiver);

            enumerate_impl_f_y(root_.children()[0], root_.children()[4], receiver);
            enumerate_impl_f_y(root_.children()[1], root_.children()[5], receiver);
            enumerate_impl_f_y(root_.children()[2], root_.children()[6], receiver);
            enumerate_impl_f_y(root_.children()[3], root_.children()[7], receiver);



            enumerate_impl_e_xy(root_.children()[0], root_.children()[1], root_.children()[4], root_.children()[5], receiver);
            enumerate_impl_e_xy(root_.children()[1], root_.children()[2], root_.children()[5], root_.children()[6], receiver);
            enumerate_impl_e_xy(root_.children()[2], root_.children()[3], root_.children()[6], root_.children()[7], receiver);
            enumerate_impl_e_xy(root_.children()[3], root_.children()[0], root_.children()[7], root_.children()[4], receiver);

			return triangles;
		}


		template <class Receiver>
		void enumerate_impl_c(ioctree_node_type* n, Receiver receiver)
		{
			if (n->get_is_branch())
			{
				if(n->get_type() == 1){
					enumerate_impl_c(n->children()[0], receiver);
					enumerate_impl_c(n->children()[1], receiver);
					enumerate_impl_c(n->children()[2], receiver);
					enumerate_impl_c(n->children()[3], receiver);
					enumerate_impl_c(n->children()[4], receiver);
					enumerate_impl_c(n->children()[5], receiver);
					enumerate_impl_c(n->children()[6], receiver);
					enumerate_impl_c(n->children()[7], receiver);

					enumerate_impl_f_x(n->children()[0], n->children()[1], receiver);
					enumerate_impl_f_x(n->children()[2], n->children()[3], receiver);
					enumerate_impl_f_x(n->children()[4], n->children()[5], receiver);
					enumerate_impl_f_x(n->children()[6], n->children()[7], receiver);

					enumerate_impl_f_y(n->children()[0], n->children()[2], receiver);
					enumerate_impl_f_y(n->children()[1], n->children()[3], receiver);
					enumerate_impl_f_y(n->children()[4], n->children()[6], receiver);
					enumerate_impl_f_y(n->children()[5], n->children()[7], receiver);

					enumerate_impl_f_z(n->children()[0], n->children()[4], receiver);
					enumerate_impl_f_z(n->children()[1], n->children()[5], receiver);
					enumerate_impl_f_z(n->children()[2], n->children()[6], receiver);
					enumerate_impl_f_z(n->children()[3], n->children()[7], receiver);

					enumerate_impl_e_xy(n->children()[0], n->children()[1], n->children()[2], n->children()[3], receiver);
					enumerate_impl_e_xy(n->children()[4], n->children()[5], n->children()[6], n->children()[7], receiver);

					enumerate_impl_e_yz(n->children()[0], n->children()[2], n->children()[4], n->children()[6], receiver);
					enumerate_impl_e_yz(n->children()[1], n->children()[3], n->children()[5], n->children()[7], receiver);

					enumerate_impl_e_xz(n->children()[0], n->children()[1], n->children()[4], n->children()[5], receiver);
					enumerate_impl_e_xz(n->children()[2], n->children()[3], n->children()[6], n->children()[7], receiver);

					enumerate_impl_v(
						n->children()[0],
						n->children()[1],
						n->children()[2],
						n->children()[3],
						n->children()[4],
						n->children()[5],
						n->children()[6],
						n->children()[7],
						receiver);
				}
				else if(n->get_type() == 2){
					enumerate_impl_c(n->children()[0], receiver);
					enumerate_impl_c(n->children()[1], receiver);
					enumerate_impl_f_z(n->children()[0], n->children()[1], receiver);
				}
				else{
					print("hello");
				}

			}
		}

		template <class Receiver>
		void enumerate_impl_f_x(ioctree_node_type* n1, ioctree_node_type* n2, Receiver receiver)
		{
			if (n1->get_is_branch() || n2->get_is_branch())
			{
				if(n1->get_type() != 2 && n2->get_type() != 2){
					enumerate_impl_f_x(n1->get_is_branch() ? n1->children()[1] : n1, n2->get_is_branch() ? n2->children()[0] : n2, receiver);
					enumerate_impl_f_x(n1->get_is_branch() ? n1->children()[3] : n1, n2->get_is_branch() ? n2->children()[2] : n2, receiver);
					enumerate_impl_f_x(n1->get_is_branch() ? n1->children()[5] : n1, n2->get_is_branch() ? n2->children()[4] : n2, receiver);
					enumerate_impl_f_x(n1->get_is_branch() ? n1->children()[7] : n1, n2->get_is_branch() ? n2->children()[6] : n2, receiver);

					enumerate_impl_e_xy(
						n1->get_is_branch() ? n1->children()[1] : n1,
						n2->get_is_branch() ? n2->children()[0] : n2,
						n1->get_is_branch() ? n1->children()[3] : n1,
						n2->get_is_branch() ? n2->children()[2] : n2,
						receiver);

					enumerate_impl_e_xy(
						n1->get_is_branch() ? n1->children()[5] : n1,
						n2->get_is_branch() ? n2->children()[4] : n2,
						n1->get_is_branch() ? n1->children()[7] : n1,
						n2->get_is_branch() ? n2->children()[6] : n2,
						receiver);

					enumerate_impl_e_xz(
						n1->get_is_branch() ? n1->children()[1] : n1,
						n2->get_is_branch() ? n2->children()[0] : n2,
						n1->get_is_branch() ? n1->children()[5] : n1,
						n2->get_is_branch() ? n2->children()[4] : n2,
						receiver);

					enumerate_impl_e_xz(
						n1->get_is_branch() ? n1->children()[3] : n1,
						n2->get_is_branch() ? n2->children()[2] : n2,
						n1->get_is_branch() ? n1->children()[7] : n1,
						n2->get_is_branch() ? n2->children()[6] : n2,
						receiver);

					enumerate_impl_v(
						n1->get_is_branch() ? n1->children()[1] : n1,
						n2->get_is_branch() ? n2->children()[0] : n2,
						n1->get_is_branch() ? n1->children()[3] : n1,
						n2->get_is_branch() ? n2->children()[2] : n2,
						n1->get_is_branch() ? n1->children()[5] : n1,
						n2->get_is_branch() ? n2->children()[4] : n2,
						n1->get_is_branch() ? n1->children()[7] : n1,
						n2->get_is_branch() ? n2->children()[6] : n2,
						receiver);
				}
				else if(n1->get_type() == 2 && n2->get_type() == 2){
					enumerate_impl_f_x(n1->children()[0], n2->children()[0], receiver);
					enumerate_impl_f_x(n1->children()[1], n2->children()[1], receiver);
					enumerate_impl_e_xz(
						n1->children()[0],
						n2->children()[0],
						n1->children()[1],
						n2->children()[1],
						receiver);

				}
				else if(n1->get_type() == 2 && n2->get_type() == -1){
					enumerate_impl_f_x(n1->children()[0], n2, receiver);
					enumerate_impl_f_x(n1->children()[1], n2, receiver);
					enumerate_impl_e_xz(
						n1->children()[0],
						n2,
						n1->children()[1],
						n2,
						receiver);
				}
				else if(n1->get_type() == -1 && n2->get_type() == 2){
					enumerate_impl_f_x(n1, n2->children()[0], receiver);
					enumerate_impl_f_x(n1, n2->children()[1], receiver);
					enumerate_impl_e_xz(
						n1,
						n2->children()[0],
						n1,
						n2->children()[1],
						receiver);
				}
				else{
					print("fx");
					printl(n1->get_type(), n2->get_type());
				}
			}
		}

		template <class Receiver>
		void enumerate_impl_f_y(ioctree_node_type* n1, ioctree_node_type* n2, Receiver receiver)
		{
			if (n1->get_is_branch() || n2->get_is_branch())
			{
				if(n1->get_type() != 2 && n2->get_type() != 2){
					enumerate_impl_f_y(n1->get_is_branch() ? n1->children()[2] : n1, n2->get_is_branch() ? n2->children()[0] : n2, receiver);
					enumerate_impl_f_y(n1->get_is_branch() ? n1->children()[3] : n1, n2->get_is_branch() ? n2->children()[1] : n2, receiver);
					enumerate_impl_f_y(n1->get_is_branch() ? n1->children()[6] : n1, n2->get_is_branch() ? n2->children()[4] : n2, receiver);
					enumerate_impl_f_y(n1->get_is_branch() ? n1->children()[7] : n1, n2->get_is_branch() ? n2->children()[5] : n2, receiver);

					enumerate_impl_e_xy(
						n1->get_is_branch() ? n1->children()[2] : n1,
						n1->get_is_branch() ? n1->children()[3] : n1,
						n2->get_is_branch() ? n2->children()[0] : n2,
						n2->get_is_branch() ? n2->children()[1] : n2,
						receiver);

					enumerate_impl_e_xy(
						n1->get_is_branch() ? n1->children()[6] : n1,
						n1->get_is_branch() ? n1->children()[7] : n1,
						n2->get_is_branch() ? n2->children()[4] : n2,
						n2->get_is_branch() ? n2->children()[5] : n2,
						receiver);

					enumerate_impl_e_yz(
						n1->get_is_branch() ? n1->children()[3] : n1,
						n2->get_is_branch() ? n2->children()[1] : n2,
						n1->get_is_branch() ? n1->children()[7] : n1,
						n2->get_is_branch() ? n2->children()[5] : n2,
						receiver);

					enumerate_impl_e_yz(
						n1->get_is_branch() ? n1->children()[2] : n1,
						n2->get_is_branch() ? n2->children()[0] : n2,
						n1->get_is_branch() ? n1->children()[6] : n1,
						n2->get_is_branch() ? n2->children()[4] : n2,
						receiver);

					enumerate_impl_v(
						n1->get_is_branch() ? n1->children()[2] : n1,
						n1->get_is_branch() ? n1->children()[3] : n1,
						n2->get_is_branch() ? n2->children()[0] : n2,
						n2->get_is_branch() ? n2->children()[1] : n2,
						n1->get_is_branch() ? n1->children()[6] : n1,
						n1->get_is_branch() ? n1->children()[7] : n1,
						n2->get_is_branch() ? n2->children()[4] : n2,
						n2->get_is_branch() ? n2->children()[5] : n2,
						receiver);
				}
				else if(n1->get_type() == 2 && n2->get_type() == 2){

					enumerate_impl_f_y(n1->children()[0], n2->children()[0], receiver);
					enumerate_impl_f_y(n1->children()[1], n2->children()[1], receiver);

					enumerate_impl_e_yz(
						n1->children()[0],
						n2->children()[0],
						n1->children()[1],
						n2->children()[1],
						receiver);
				}
				else if(n1->get_type() == 2 && n2->get_type() == -1){
					enumerate_impl_f_y(n1->children()[0], n2, receiver);
					enumerate_impl_f_y(n1->children()[1], n2, receiver);

					enumerate_impl_e_yz(
						n1->children()[0],
						n2,
						n1->children()[1],
						n2,
						receiver);
				}
				else if (n1->get_type() == -1 && n2->get_type() == 2){
					enumerate_impl_f_y(n1, n2->children()[0], receiver);
					enumerate_impl_f_y(n1, n2->children()[1], receiver);

					enumerate_impl_e_yz(
						n1,
						n2->children()[0],
						n1,
						n2->children()[1],
						receiver);
				}
				else if(n1->get_type() == 1 && n2->get_type() == 2){
					enumerate_impl_f_y(n1->children()[2], n2->children()[0], receiver);
					enumerate_impl_f_y(n1->children()[3], n2->children()[0], receiver);
					enumerate_impl_f_y(n1->children()[6], n2->children()[1], receiver);
					enumerate_impl_f_y(n1->children()[7], n2->children()[1], receiver);

					enumerate_impl_e_yz(
						n1->children()[3],
						n2->children()[0],
						n1->children()[7],
						n2->children()[1],
						receiver);

					enumerate_impl_e_yz(
						n1->children()[2],
						n2->children()[0],
						n1->children()[6],
						n2->children()[1],
						receiver);

				}
				else if(n1->get_type() == 2 && n2->get_type() == 1){
					enumerate_impl_f_y(n1->children()[0], n2->children()[0], receiver);
					enumerate_impl_f_y(n1->children()[0], n2->children()[1], receiver);
					enumerate_impl_f_y(n1->children()[1], n2->children()[4], receiver);
					enumerate_impl_f_y(n1->children()[1], n2->children()[5], receiver);

					enumerate_impl_e_yz(
						n1->children()[0],
						n2->children()[1],
						n1->children()[1],
						n2->children()[5],
						receiver);

					enumerate_impl_e_yz(
						n1->children()[0],
						n2->children()[0],
						n1->children()[1],
						n2->children()[4],
						receiver);

				}
				else{
					print("fy");
					printl(n1->get_type(), n2->get_type());
				}
			}
		}

		template <class Receiver>
		void enumerate_impl_f_z(ioctree_node_type* n1, ioctree_node_type* n2, Receiver receiver)
		{
			if (n1->get_is_branch() || n2->get_is_branch())
			{
				if (n1->get_type() != 2 && n2->get_type() != 2)
				{
					enumerate_impl_f_z(n1->get_is_branch() ? n1->children()[4] : n1, n2->get_is_branch() ? n2->children()[0] : n2, receiver);
					enumerate_impl_f_z(n1->get_is_branch() ? n1->children()[5] : n1, n2->get_is_branch() ? n2->children()[1] : n2, receiver);
					enumerate_impl_f_z(n1->get_is_branch() ? n1->children()[6] : n1, n2->get_is_branch() ? n2->children()[2] : n2, receiver);
					enumerate_impl_f_z(n1->get_is_branch() ? n1->children()[7] : n1, n2->get_is_branch() ? n2->children()[3] : n2, receiver);

					enumerate_impl_e_xz(
						n1->get_is_branch() ? n1->children()[4] : n1,
						n1->get_is_branch() ? n1->children()[5] : n1,
						n2->get_is_branch() ? n2->children()[0] : n2,
						n2->get_is_branch() ? n2->children()[1] : n2,
						receiver);

					enumerate_impl_e_xz(
						n1->get_is_branch() ? n1->children()[6] : n1,
						n1->get_is_branch() ? n1->children()[7] : n1,
						n2->get_is_branch() ? n2->children()[2] : n2,
						n2->get_is_branch() ? n2->children()[3] : n2,
						receiver);

					enumerate_impl_e_yz(
						n1->get_is_branch() ? n1->children()[4] : n1,
						n1->get_is_branch() ? n1->children()[6] : n1,
						n2->get_is_branch() ? n2->children()[0] : n2,
						n2->get_is_branch() ? n2->children()[2] : n2,
						receiver);

					enumerate_impl_e_yz(
						n1->get_is_branch() ? n1->children()[5] : n1,
						n1->get_is_branch() ? n1->children()[7] : n1,
						n2->get_is_branch() ? n2->children()[1] : n2,
						n2->get_is_branch() ? n2->children()[3] : n2,
						receiver);

					enumerate_impl_v(
						n1->get_is_branch() ? n1->children()[4] : n1,
						n1->get_is_branch() ? n1->children()[5] : n1,
						n1->get_is_branch() ? n1->children()[6] : n1,
						n1->get_is_branch() ? n1->children()[7] : n1,
						n2->get_is_branch() ? n2->children()[0] : n2,
						n2->get_is_branch() ? n2->children()[1] : n2,
						n2->get_is_branch() ? n2->children()[2] : n2,
						n2->get_is_branch() ? n2->children()[3] : n2,
						receiver);
				}
				else if(n1->get_type() == 2 && n2->get_type() == 2){
					enumerate_impl_f_z(n1->children()[1],n2->children()[0], receiver);
				}
				else if(n1->get_type() == 2 && n2->get_type() == -1){
					enumerate_impl_f_z(n1->children()[1], n2, receiver);
				}
				else if(n1->get_type() == -1 && n2->get_type() == 2){
					enumerate_impl_f_z(n1, n2->children()[0], receiver);
				}
				else if(n1->get_type() == 2 && n2->get_type() == 1){
					enumerate_impl_f_z(n1->children()[1], n2->children()[0], receiver);
					enumerate_impl_f_z(n1->children()[1], n2->children()[1], receiver);
					enumerate_impl_f_z(n1->children()[1], n2->children()[2], receiver);
					enumerate_impl_f_z(n1->children()[1], n2->children()[3], receiver);

					enumerate_impl_e_yz(
						n1->children()[1],
						n1->children()[1],
						n2->children()[0],
						n2->children()[2],
						receiver);

					enumerate_impl_e_yz(
						n1->children()[1],
						n1->children()[1],
						n2->children()[1],
						n2->children()[3],
						receiver);

					enumerate_impl_e_xz(
						n1->children()[1],
						n1->children()[1],
						n2->children()[0],
						n2->children()[1],
						receiver);
					enumerate_impl_e_xz(
						n1->children()[1],
						n1->children()[1],
						n2->children()[2],
						n2->children()[3],
						receiver);
					enumerate_impl_v(
						n1->children()[1],
						n1->children()[1],
						n1->children()[1],
						n1->children()[1],
						n2->children()[0],
						n2->children()[1],
						n2->children()[2],
						n2->children()[3],
						receiver);
				}
				else if(n1->get_type() == 1 && n2->get_type() == 2){
					enumerate_impl_f_z(n1->children()[4], n2->children()[0], receiver);
					enumerate_impl_f_z(n1->children()[5], n2->children()[0], receiver);
					enumerate_impl_f_z(n1->children()[6], n2->children()[0], receiver);
					enumerate_impl_f_z(n1->children()[7], n2->children()[0], receiver);

					enumerate_impl_e_yz(
						n1->children()[4],
						n1->children()[6],
						n2->children()[0],
						n2->children()[0],
						receiver);

					enumerate_impl_e_yz(
						n1->children()[5],
						n1->children()[7],
						n2->children()[0],
						n2->children()[0],
						receiver);

					enumerate_impl_e_xz(
						n1->children()[4],
						n1->children()[5],
						n2->children()[0],
						n2->children()[0],
						receiver);
					enumerate_impl_e_xz(
						n1->children()[6],
						n1->children()[7],
						n2->children()[0],
						n2->children()[0],
						receiver);
					enumerate_impl_v(
						n1->children()[4],
						n1->children()[5],
						n1->children()[6],
						n1->children()[7],
						n2->children()[0],
						n2->children()[0],
						n2->children()[0],
						n2->children()[0],
						receiver);
				}
				else{
					print("fz");
					printl(n1->get_type(), n2->get_type());
				}
			}

		}

		template <class Receiver>
		void enumerate_impl_e_xy(
			ioctree_node_type* n1,
			ioctree_node_type* n2,
			ioctree_node_type* n3,
			ioctree_node_type* n4,
			Receiver receiver)
		{
			if (n1->get_is_branch() || n2->get_is_branch() || n3->get_is_branch() || n4->get_is_branch())
			{

				if(n1->get_type() == 2 && n2->get_type() == 2 && n3->get_type() == 2 && n4->get_type() == 2){
					enumerate_impl_e_xy(
						n1->children()[0],
						n2->children()[0],
						n3->children()[0],
						n4->children()[0],
						receiver);
					enumerate_impl_e_xy(
						n1->children()[1],
						n2->children()[1],
						n3->children()[1],
						n4->children()[1],
					receiver);
					enumerate_impl_v(
						n1->children()[0],
						n2->children()[0],
						n3->children()[0],
						n4->children()[0],
						n1->children()[1],
						n2->children()[1],
						n3->children()[1],
						n4->children()[1],
						receiver);
				}

				else if(n1->get_type() != 2 && n2->get_type() != 2 && n3->get_type() != 2 && n4->get_type() != 2)
				{
					enumerate_impl_e_xy(
						n1->get_is_branch() ? n1->children()[3] : n1,
						n2->get_is_branch() ? n2->children()[2] : n2,
						n3->get_is_branch() ? n3->children()[1] : n3,
						n4->get_is_branch() ? n4->children()[0] : n4,
						receiver);

					enumerate_impl_e_xy(
						n1->get_is_branch() ? n1->children()[7] : n1,
						n2->get_is_branch() ? n2->children()[6] : n2,
						n3->get_is_branch() ? n3->children()[5] : n3,
						n4->get_is_branch() ? n4->children()[4] : n4,
						receiver);
					enumerate_impl_v(
						n1->get_is_branch() ? n1->children()[3] : n1,
						n2->get_is_branch() ? n2->children()[2] : n2,
						n3->get_is_branch() ? n3->children()[1] : n3,
						n4->get_is_branch() ? n4->children()[0] : n4,
						n1->get_is_branch() ? n1->children()[7] : n1,
						n2->get_is_branch() ? n2->children()[6] : n2,
						n3->get_is_branch() ? n3->children()[5] : n3,
						n4->get_is_branch() ? n4->children()[4] : n4,
						receiver);
				}

				else{
					enumerate_impl_e_xy(
						n1->get_is_branch() ? (n1->get_type() == 2 ? n1->children()[0] : n1->children()[3]) : n1,
						n2->get_is_branch() ? (n2->get_type() == 2 ? n2->children()[0] : n2->children()[2]) : n2,
						n3->get_is_branch() ? (n3->get_type() == 2 ? n3->children()[0] : n3->children()[1]) : n3,
						n4->get_is_branch() ? (n4->get_type() == 2 ? n4->children()[0] : n4->children()[0]) : n4,
						receiver);
					enumerate_impl_e_xy(
						n1->get_is_branch() ? (n1->get_type() == 2 ? n1->children()[1] : n1->children()[7]) : n1,
						n2->get_is_branch() ? (n2->get_type() == 2 ? n2->children()[1] : n2->children()[6]) : n2,
						n3->get_is_branch() ? (n3->get_type() == 2 ? n3->children()[1] : n3->children()[5]) : n3,
						n4->get_is_branch() ? (n4->get_type() == 2 ? n4->children()[1] : n4->children()[4]) : n4,
						receiver);
					enumerate_impl_v(
						n1->get_is_branch() ? (n1->get_type() == 2 ? n1->children()[0] : n1->children()[3]) : n1,
						n2->get_is_branch() ? (n2->get_type() == 2 ? n2->children()[0] : n2->children()[2]) : n2,
						n3->get_is_branch() ? (n3->get_type() == 2 ? n3->children()[0] : n3->children()[1]) : n3,
						n4->get_is_branch() ? (n4->get_type() == 2 ? n4->children()[0] : n4->children()[0]) : n4,
						n1->get_is_branch() ? (n1->get_type() == 2 ? n1->children()[1] : n1->children()[7]) : n1,
						n2->get_is_branch() ? (n2->get_type() == 2 ? n2->children()[1] : n2->children()[6]) : n2,
						n3->get_is_branch() ? (n3->get_type() == 2 ? n3->children()[1] : n3->children()[5]) : n3,
						n4->get_is_branch() ? (n4->get_type() == 2 ? n4->children()[1] : n4->children()[4]) : n4,
						receiver);
				}
			}
		}

		template <class Receiver>
		void enumerate_impl_e_yz(
			ioctree_node_type* n1,
			ioctree_node_type* n2,
			ioctree_node_type* n3,
			ioctree_node_type* n4,
			Receiver receiver)
		{
			if (n1->get_is_branch() || n2->get_is_branch() || n3->get_is_branch() || n4->get_is_branch())
			{
				if(n1->get_type() == 2 && n2->get_type() == 2 && n3->get_type() == 2 && n4->get_type() == 2){
					enumerate_impl_e_yz(
						n1->children()[1],
						n2->children()[1],
						n3->children()[0],
						n4->children()[0],
						receiver);
				}
				else if(n1->get_type() != 2 && n2->get_type() != 2 && n3->get_type() != 2 && n4->get_type() != 2){
					enumerate_impl_e_yz(
						n1->get_is_branch() ? n1->children()[6] : n1,
						n2->get_is_branch() ? n2->children()[4] : n2,
						n3->get_is_branch() ? n3->children()[2] : n3,
						n4->get_is_branch() ? n4->children()[0] : n4,
						receiver);

					enumerate_impl_e_yz(
						n1->get_is_branch() ? n1->children()[7] : n1,
						n2->get_is_branch() ? n2->children()[5] : n2,
						n3->get_is_branch() ? n3->children()[3] : n3,
						n4->get_is_branch() ? n4->children()[1] : n4,
						receiver);

					enumerate_impl_v(
						n1->get_is_branch() ? n1->children()[6] : n1,
						n1->get_is_branch() ? n1->children()[7] : n1,
						n2->get_is_branch() ? n2->children()[4] : n2,
						n2->get_is_branch() ? n2->children()[5] : n2,
						n3->get_is_branch() ? n3->children()[2] : n3,
						n3->get_is_branch() ? n3->children()[3] : n3,
						n4->get_is_branch() ? n4->children()[0] : n4,
						n4->get_is_branch() ? n4->children()[1] : n4,
						receiver);
				}
				else if(n1->get_type() != 1 && n2->get_type() != 1 && n3->get_type() != 1 && n4->get_type() != 1){
					enumerate_impl_e_yz(
						n1->get_is_branch() ? n1->children()[1] : n1,
						n2->get_is_branch() ? n2->children()[1] : n2,
						n3->get_is_branch() ? n3->children()[0] : n3,
						n4->get_is_branch() ? n4->children()[0] : n4,
						receiver);
				}
				else if(n1->get_type() == 1 || n2->get_type() == 1 || n3->get_type() == 1 || n4->get_type() == 1){
					enumerate_impl_e_yz(
						n1->get_is_branch() ? (n1->get_type() == 2 ? n1->children()[1] : n1->children()[7]) : n1,
						n2->get_is_branch() ? (n2->get_type() == 2 ? n2->children()[1] : n2->children()[5]) : n2,
						n3->get_is_branch() ? (n3->get_type() == 2 ? n3->children()[0] : n3->children()[3]) : n3,
						n4->get_is_branch() ? (n4->get_type() == 2 ? n4->children()[0] : n4->children()[1]) : n4,
						receiver);
					enumerate_impl_e_yz(
						n1->get_is_branch() ? (n1->get_type() == 2 ? n1->children()[1] : n1->children()[6]) : n1,
						n2->get_is_branch() ? (n2->get_type() == 2 ? n2->children()[1] : n2->children()[4]) : n2,
						n3->get_is_branch() ? (n3->get_type() == 2 ? n3->children()[0] : n3->children()[2]) : n3,
						n4->get_is_branch() ? (n4->get_type() == 2 ? n4->children()[0] : n4->children()[0]) : n4,
						receiver);
					enumerate_impl_v(
						n1->get_is_branch() ? (n1->get_type() == 2 ? n1->children()[1] : n1->children()[6]) : n1,
						n2->get_is_branch() ? (n2->get_type() == 2 ? n2->children()[1] : n2->children()[7]) : n2,
						n1->get_is_branch() ? (n1->get_type() == 2 ? n1->children()[1] : n1->children()[4]) : n1,
						n2->get_is_branch() ? (n2->get_type() == 2 ? n2->children()[1] : n2->children()[5]) : n2,
						n3->get_is_branch() ? (n3->get_type() == 2 ? n3->children()[0] : n3->children()[2]) : n3,
						n4->get_is_branch() ? (n4->get_type() == 2 ? n4->children()[0] : n4->children()[3]) : n4,
						n3->get_is_branch() ? (n3->get_type() == 2 ? n3->children()[0] : n3->children()[0]) : n3,
						n4->get_is_branch() ? (n4->get_type() == 2 ? n4->children()[0] : n4->children()[1]) : n4,
						receiver);
				}
				else{
					print("eyz");
					printl(n1->get_type(), n2->get_type());
					printl(n3->get_type(), n4->get_type());
				}
			}
		}

		template <class Receiver>
		void enumerate_impl_e_xz(
			ioctree_node_type* n1,
			ioctree_node_type* n2,
			ioctree_node_type* n3,
			ioctree_node_type* n4,
			Receiver receiver)
		{
			if (n1->get_is_branch() || n2->get_is_branch() || n3->get_is_branch() || n4->get_is_branch())
			{

				if(n1->get_type() == 2 && n2->get_type() == 2 && n3->get_type() == 2 && n4->get_type() == 2){
					enumerate_impl_e_xz(
						n1->children()[1],
						n2->children()[1],
						n3->children()[0],
						n4->children()[0],
						receiver);
				}
				else if(n1->get_type() != 2 && n2->get_type() != 2 && n3->get_type() != 2 && n4->get_type() != 2){
					enumerate_impl_e_xz(
						n1->get_is_branch() ? n1->children()[5] : n1,
						n2->get_is_branch() ? n2->children()[4] : n2,
						n3->get_is_branch() ? n3->children()[1] : n3,
						n4->get_is_branch() ? n4->children()[0] : n4,
						receiver);

					enumerate_impl_e_xz(
						n1->get_is_branch() ? n1->children()[7] : n1,
						n2->get_is_branch() ? n2->children()[6] : n2,
						n3->get_is_branch() ? n3->children()[3] : n3,
						n4->get_is_branch() ? n4->children()[2] : n4,
						receiver);

					enumerate_impl_v(
						n1->get_is_branch() ? n1->children()[5] : n1,
						n2->get_is_branch() ? n2->children()[4] : n2,
						n1->get_is_branch() ? n1->children()[7] : n1,
						n2->get_is_branch() ? n2->children()[6] : n2,
						n3->get_is_branch() ? n3->children()[1] : n3,
						n4->get_is_branch() ? n4->children()[0] : n4,
						n3->get_is_branch() ? n3->children()[3] : n3,
						n4->get_is_branch() ? n4->children()[2] : n4,
						receiver);
				}
				else if(n1->get_type() != 1 && n2->get_type() != 1 && n3->get_type() != 1 && n4->get_type() != 1){
					enumerate_impl_e_xz(
						n1->get_is_branch() ? n1->children()[1] : n1,
						n2->get_is_branch() ? n2->children()[1] : n2,
						n3->get_is_branch() ? n3->children()[0] : n3,
						n4->get_is_branch() ? n4->children()[0] : n4,
						receiver);
				}
				else if(n1->get_type() == 1 || n2->get_type() == 1 || n3->get_type() == 1 || n4->get_type() == 1){
					enumerate_impl_e_xz(
						n1->get_is_branch() ? (n1->get_type() == 2 ? n1->children()[1] : n1->children()[7]) : n1,
						n2->get_is_branch() ? (n2->get_type() == 2 ? n2->children()[1] : n2->children()[6]) : n2,
						n3->get_is_branch() ? (n3->get_type() == 2 ? n3->children()[0] : n3->children()[3]) : n3,
						n4->get_is_branch() ? (n4->get_type() == 2 ? n4->children()[0] : n4->children()[2]) : n4,
						receiver);
					enumerate_impl_e_xz(
						n1->get_is_branch() ? (n1->get_type() == 2 ? n1->children()[1] : n1->children()[5]) : n1,
						n2->get_is_branch() ? (n2->get_type() == 2 ? n2->children()[1] : n2->children()[4]) : n2,
						n3->get_is_branch() ? (n3->get_type() == 2 ? n3->children()[0] : n3->children()[1]) : n3,
						n4->get_is_branch() ? (n4->get_type() == 2 ? n4->children()[0] : n4->children()[0]) : n4,
						receiver);
					enumerate_impl_v(
						n1->get_is_branch() ? (n1->get_type() == 2 ? n1->children()[1] : n1->children()[5]) : n1,
						n2->get_is_branch() ? (n2->get_type() == 2 ? n2->children()[1] : n2->children()[4]) : n2,
						n1->get_is_branch() ? (n1->get_type() == 2 ? n1->children()[1] : n1->children()[7]) : n1,
						n2->get_is_branch() ? (n2->get_type() == 2 ? n2->children()[1] : n2->children()[6]) : n2,
						n3->get_is_branch() ? (n3->get_type() == 2 ? n3->children()[0] : n3->children()[1]) : n3,
						n4->get_is_branch() ? (n4->get_type() == 2 ? n4->children()[0] : n4->children()[0]) : n4,
						n3->get_is_branch() ? (n3->get_type() == 2 ? n3->children()[0] : n3->children()[3]) : n3,
						n4->get_is_branch() ? (n4->get_type() == 2 ? n4->children()[0] : n4->children()[2]) : n4,
						receiver);
				}
				else{
					print("exz");
					printl(n1->get_type(), n2->get_type());
					printl(n3->get_type(), n4->get_type());
				}
			}
		}

		template <class Receiver>
		void enumerate_impl_v(
			ioctree_node_type* n1,
			ioctree_node_type* n2,
			ioctree_node_type* n3,
			ioctree_node_type* n4,
			ioctree_node_type* n5,
			ioctree_node_type* n6,
			ioctree_node_type* n7,
			ioctree_node_type* n8,
			Receiver receiver)
		{
			if (n1->get_is_branch() || n2->get_is_branch() || n3->get_is_branch() || n4->get_is_branch() || n5->get_is_branch() || n6->get_is_branch() || n7->get_is_branch() || n8->get_is_branch())
			{
				enumerate_impl_v(
					n1->get_is_branch() ? (n1->get_type() == 2 ? n1->children()[1] : n1->children()[7]) : n1,
					n2->get_is_branch() ? (n2->get_type() == 2 ? n2->children()[1] : n2->children()[6]) : n2,
					n3->get_is_branch() ? (n3->get_type() == 2 ? n3->children()[1] : n3->children()[5]) : n3,
					n4->get_is_branch() ? (n4->get_type() == 2 ? n4->children()[1] : n4->children()[4]) : n4,
					n5->get_is_branch() ? (n5->get_type() == 2 ? n5->children()[0] : n5->children()[3]) : n5,
					n6->get_is_branch() ? (n6->get_type() == 2 ? n6->children()[0] : n6->children()[2]) : n6,
					n7->get_is_branch() ? (n7->get_type() == 2 ? n7->children()[0] : n7->children()[1]) : n7,
					n8->get_is_branch() ? (n8->get_type() == 2 ? n8->children()[0] : n8->children()[0]) : n8,
					receiver);
			}
			else
			{
			    if(n1->sdf_cnt > min_sdf_cnt &&
					n2->sdf_cnt > min_sdf_cnt &&
					n3->sdf_cnt > min_sdf_cnt &&
					n4->sdf_cnt > min_sdf_cnt &&
					n5->sdf_cnt > min_sdf_cnt &&
					n6->sdf_cnt > min_sdf_cnt &&
					n7->sdf_cnt > min_sdf_cnt &&
					n8->sdf_cnt > min_sdf_cnt){
                    std::array<const vertex_type *, 8> vertices = {{
                                                                           &n1->vertex(),
                                                                           &n2->vertex(),
                                                                           &n3->vertex(),
                                                                           &n4->vertex(),
                                                                           &n5->vertex(),
                                                                           &n6->vertex(),
                                                                           &n7->vertex(),
                                                                           &n8->vertex(),
                                                                   }};
                    marching_cubes(vertices, receiver);
                }
			}
		}
	};
}
