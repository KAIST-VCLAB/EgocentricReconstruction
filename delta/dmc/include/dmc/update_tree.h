#pragma once

struct Holder{

	float* sdf_value;
	float* sdf_weight;
	float* sdf_cnt;
	float* center_xyz;
	float* center_inv;
	float* depth_map;
	float* weight_map;
	float* frame_rt;
	float* unit;
	float* cell_depth;
	float* mask_depth;
	float* weight_trunc_scale;
	float* trunc_offset;
	float* trunc_change;
};


class cuda{
public:
	cuda();
	~cuda();
	void update_cuda(float* a, int len);


	void update_cuda_alloc(unsigned int len, int depth_size, int depth_num);
	void update_cuda_load(int mode, int i, float* src, int size);
	void update_cuda_sdf(unsigned int len, int depth_num, int h, int w, float* mask_depth);
	void update_cuda_unload(int mode, float* dst, int i, int size);
	Holder holder;
};
