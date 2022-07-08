#pragma once
#include "vector.hpp"
#include "vertex.hpp"
#include <boost/noncopyable.hpp>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

namespace dmc
{
	template <class Scalar>
	class ioctree_node
	{
	public:
		typedef Scalar scalar_type;
		typedef vector<scalar_type, 3> vector_type;
		typedef dmc::vertex<scalar_type> vertex_type;
		typedef ioctree_node<float> ioctree_node_type;

		float min_idepth;
		float max_idepth;
		float min_phi;
		float max_phi;
		float min_theta;
		float max_theta;

		float sdf_cnt;
		float sdf_value;
		float d_cue;
		int type;




		ioctree_node(const vertex_type& vertex, bool is_leaf, float min_phi_, float max_phi_, float min_theta_, float max_theta_, float min_idepth_, float max_idepth_)
		: vertex_(vertex),
			is_leaf_(is_leaf), min_phi(min_phi_), max_phi(max_phi_), min_theta(min_theta_), max_theta(max_theta_), min_idepth(min_idepth_), max_idepth(max_idepth_)
			,sdf_cnt(0),sdf_value(-0.000001f)
		{
			type = -1;
			d_cue = 9999999.0f;
		}
		vertex_type& vertex()
		{
			return vertex_;
		}
		std::array<ioctree_node_type*, 8>& children()
		{
			return children_;
		}
		void set_is_leaf(bool is_leaf){
			is_leaf_ = is_leaf;
		}

		bool get_is_leaf(){
			return is_leaf_;
		}

		bool get_is_branch(){
			return !is_leaf_;
		}

		void set_type(int t){
			type = t;
		}

		int get_type(){
			return type;
		}

		float get_node_size(){
			return (1.0f / (min_idepth * min_idepth * min_idepth) - 1.0f / (max_idepth * max_idepth * max_idepth)) *
				   (cosf(min_theta) - cosf(max_theta)) * (max_phi - min_phi) / 3;
		}



		vector_type get_center(){
			return vector_type((min_phi + max_phi)/2, (min_theta + max_theta)/2, (1.0f / min_idepth + 1.0f / max_idepth)/2);
		}

		 Eigen::Vector4f get_center_point_cart(){
			float mid_phi = (min_phi + max_phi)/2;
			float mid_theta = (min_theta + max_theta)/2;
			float mid_depth = (1.0f / min_idepth + 1.0f / max_idepth)/2;


			Eigen::Vector4f xyzw;
			xyzw(0,0) = mid_depth * sinf(mid_theta) * cosf(mid_phi);
			xyzw(1,0) = -mid_depth * cosf(mid_theta);
			xyzw(2,0) = mid_depth * sinf(mid_theta) * sinf(mid_phi);
			xyzw(3,0) = 1;
			return xyzw;
		}



		float get_center_depth(){
			return (1.0f / min_idepth + 1.0f / max_idepth)/2;
		}

		void set_vertex(vector_type pos, float offs){
			vertex_.set_position(pos);
			vertex_.set_offset_sign(offs);
		}



	private:
		vertex_type vertex_;
		bool is_leaf_;
		std::array<ioctree_node_type*, 8> children_;

	};
}
