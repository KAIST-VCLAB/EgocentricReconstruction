#include <dmc/dmc.hpp>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <string>
#include <opencv2/opencv.hpp>
#include <time.h>


#define PI 3.14159265358979323846
#define print(x) std::cout << x << std::endl;
#define printl(x, y) std::cout << x << " " << y << std::endl;

template <class Scalar>

struct test_object : dmc::dual_object<Scalar, test_object<Scalar>>
{
private:
	typedef dmc::object<Scalar> base_type;

public:
	using typename base_type::scalar_type;
	using typename base_type::vector_type;

	template <class T>
	T templated_value(const dmc::vector<T, 3>& p) const
	{
		auto cube1 = 1.0 - p.norm_l_inf();
		auto cube2 = 1.0 - (p - dmc::vector<T, 3>(0.5, 0.5, 0.5)).norm_l_inf();

		return std::min(cube1, -cube2);
	}
};


int main(int argc, char* argv[])
{
	std::string video_name = argv[1];
	float resolution = atof(argv[2]);
	int interval = atoi(argv[3]);
	float min_depth = atof(argv[4]);
	float max_depth = atof(argv[5]);
	float depth_scale = 1.0f; // change depth scale (depth domain)
	float voxel_scale = 1.0f; // high value will decrease resolution
	float trunc = atof(argv[6]);
	std::string data_root_path = argv[7];
	std::string video_root_path = argv[8];
	std::string depth_root_path = argv[9];
	std::string mesh_root_path = argv[10];
	int min_sdf_cnt = atoi(argv[11]);
	int node_sample_interval = atoi(argv[12]);
	float sky_depth_threshold = atof(argv[13]);
	float weight_trunc_scale = atof(argv[14]);
	float truncation_offset = atof(argv[15]);
	float truncation_change = atof(argv[16]);

	float closest = 0.8f; // inverse depth value of the closest point

	std::string depth_path = data_root_path + "/" + depth_root_path + "/" + video_name + "_" + "depth";
	std::string weight_path = data_root_path + "/" + depth_root_path + "/" + video_name + "_" + "weight";
	std::string rt_path = data_root_path + "/" + video_root_path + "/" + video_name + "/" + "traj.csv";
	std::string stl_path = data_root_path + "/" + mesh_root_path + "/" + video_name + "/" + "mesh.stl";
	std::string mask_path = data_root_path + "/" + video_root_path + "/" + video_name + "/" + "mask_depth.png";
	std::string voxel_path = data_root_path + "/" + mesh_root_path + "/" + video_name + "/" + "voxel_center.obj";
	std::string node_distribution_path = data_root_path + "/" + mesh_root_path + "/" + video_name + "/" + "node_distribution.txt";

	std::vector<Eigen::Matrix4f> frame_rt;

	print("loading rt start");
	load_frame_rt(&frame_rt, rt_path);
	print("loading rt done");


	if(frame_rt.size() == 0){
		print(rt_path);
		print("No traj.csv read");
	}


	Eigen::Matrix4f center_rt = find_center(&frame_rt, depth_scale);

	dmc::ioctree_config iot_config;
	iot_config.min_phi = 0;
	iot_config.max_phi = 2 * PI;
	iot_config.min_theta = 0;
	iot_config.max_theta = PI;
	iot_config.min_idepth = 1.0f / max_depth;
	iot_config.max_idepth = 1.0f / min_depth;

	iot_config.user_max_idepth = closest;
	iot_config.voxel_scale = (float)voxel_scale;
	iot_config.depth_scale = depth_scale;
	iot_config.mask_path = mask_path;
	iot_config.voxel_path = voxel_path;
	iot_config.node_distribution_path = node_distribution_path;
	iot_config.min_sdf_cnt = min_sdf_cnt;
	dmc::ioctree iot(iot_config);


	clock_t load_start, load_end;
	load_start = clock();
	print("loading depth map start");
	print(depth_path);
	print(weight_path)
	iot.load_depth_map(&frame_rt, depth_path, weight_path, interval, sky_depth_threshold);
	print("loading depth map done");
	load_end = clock();

	clock_t generate_start, generate_end;
	generate_start = clock();
	iot.generate_tree(center_rt, resolution, node_sample_interval);
	generate_end = clock();

	clock_t update_start, update_end;
	update_start = clock();
	iot.update_tree(center_rt, trunc, weight_trunc_scale, truncation_offset, truncation_change);
	update_end = clock();

	clock_t enumerate_start, enumerate_end;
	enumerate_start = clock();
	auto triangles = iot.enumerate();
	printl("Number of triangles",triangles.size());
	enumerate_end = clock();
	auto enumerated_time = std::chrono::high_resolution_clock::now();

	clock_t stl_start, stl_end;
	stl_start = clock();
	std::ofstream os(stl_path, std::ios::binary);
	write_stl(os, triangles);
	stl_end = clock();

	std::cout << "loading     " << (load_end -     load_start)/     CLOCKS_PER_SEC << std::endl;
	std::cout << "generating  " << (generate_end - generate_start)/ CLOCKS_PER_SEC << std::endl;
	std::cout << "updating    " << (update_end -   update_start)/   CLOCKS_PER_SEC << std::endl;
	std::cout << "enumerating " << (enumerate_end -enumerate_start)/CLOCKS_PER_SEC << std::endl;
	std::cout << "stl save    " << (stl_end -      stl_start)/      CLOCKS_PER_SEC << std::endl;

	return 0;

}
