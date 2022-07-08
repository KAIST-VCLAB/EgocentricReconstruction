#pragma once

#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>



void load_frame_rt(std::vector<Eigen::Matrix4f>* frame_rt, std::string rt_path){
	std::ifstream rt_stream(rt_path);

	std::string line;
	std::vector<float> tokens;

	char delimiter = ' ';
	std::string tok;
	int elem_cnt = 0;

	while(std::getline(rt_stream, line)){
		std::stringstream string_stream(line);
		tokens.clear();
		elem_cnt = 0;
		while(std::getline(string_stream, tok, delimiter)){
			if(elem_cnt != 0){
				tokens.push_back(std::stof(tok));
			}
			else{

			}
			elem_cnt++;
		}
		Eigen::Matrix4f rt = Eigen::Map<Eigen::Matrix4f>(tokens.data()).transpose();

		frame_rt->push_back(rt);

	}

}

void show_depth_maps(std::string depth_path, int total_depth_num){
	for(int i = 0; i < total_depth_num; i++){
		std::string path = depth_path + std::to_string(i) + ".exr";
		cv::Mat depth = cv::imread(path, -1);
		cv::resize(depth, depth, depth.size() / 4);
		cv::Mat depth_vis;
		cv::convertScaleAbs(depth, depth_vis, 200);
		cv::imshow("depth", depth_vis);
		cv::waitKey(3);
	}
}

Eigen::Matrix4f find_center(std::vector<Eigen::Matrix4f>* frame_rt, float depth_scale){
	typedef dmc::vector<float, 3> vector_type;


	Eigen::Matrix4f rt_first = (*frame_rt)[0];
	Eigen::Matrix4f rt_next;
	Eigen::Matrix4f rt_temp;
	Eigen::Matrix4f rt_first_to_next;

	vector_type t_from_first_sum(0,0,0);
	vector_type t_from_first;
	vector_type t_from_first_mean;

	vector_type baseline_temp;
	float cnt = 0;
	Eigen::Matrix4f center = Eigen::Matrix4f::Identity();

	for(int i = 0; i < frame_rt->size(); i++){

		rt_next = (*frame_rt)[i];
		rt_first_to_next = rt_next * rt_first.inverse();
		rt_temp = rt_first_to_next.inverse();
		t_from_first = vector_type(rt_temp(0,3),rt_temp(1,3),rt_temp(2,3));
		t_from_first_sum = t_from_first_sum + t_from_first;
		cnt++;
	}
	t_from_first_mean = -t_from_first_sum / cnt;

	// first to center
	center(0,3) = t_from_first_mean[0];
	center(1,3) = t_from_first_mean[1];
	center(2,3) = t_from_first_mean[2];
	// center: change point from first to center

	// first to center an
	center = center * rt_first;

	// scale
	center(0,3) = depth_scale * center(0,3);
	center(1,3) = depth_scale * center(1,3);
	center(2,3) = depth_scale * center(2,3);


	std::cout << center.inverse() << std::endl;
	return center;
}
