#pragma once
#include <cstddef>

namespace dmc
{
	template <class Scalar>
	struct tree_config
	{
		typedef Scalar scalar_type;

		scalar_type grid_width = static_cast<scalar_type>(1.0);
		scalar_type tolerance = static_cast<scalar_type>(0.1);
		std::size_t maximum_depth = static_cast<std::size_t>(-1);
		scalar_type nominal_weight = static_cast<scalar_type>(0.1);
	};


	struct ioctree_config
	{

		float min_phi;
		float max_phi;
		float min_theta;
		float max_theta;
		float min_idepth;
		float max_idepth;
		float user_min_idepth;
		float user_max_idepth;
		float voxel_scale;
		float depth_scale;
		std::string mask_path;
		std::string voxel_path;
		std::string node_distribution_path;
		int min_sdf_cnt;
	};

	struct cubic_tree_config
	{

		float min_x;
		float max_x;
		float min_y;
		float max_y;
		float min_z;
		float max_z;
		float user_min_idepth;
		float user_max_idepth;
		float voxel_scale;
		float depth_scale;
		std::string mask_path;
		std::string voxel_path;
		std::string node_distribution_path;
		int min_sdf_cnt;
	};

}
