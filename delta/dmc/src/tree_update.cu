#define PI 3.14159265358979323846f
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "../include/dmc/update_tree.h"

__global__ void update_kernel(float* a_gpu){
	int idx = threadIdx.x;
	a_gpu[idx] = 1;

}

cuda::cuda(){

}
cuda::~cuda(){

}

void cuda::update_cuda(float* a, int len){
	dim3 grid(1);
	dim3 block(len);
	float* a_gpu;
	cudaMalloc((void**)&a_gpu, sizeof(float) * len);
	cudaMemcpy(a_gpu, a, sizeof(float) * len, cudaMemcpyHostToDevice);
	update_kernel<<<grid, block>>>(a_gpu);
	cudaMemcpy(a, a_gpu, sizeof(float) * len, cudaMemcpyDeviceToHost);
	cudaFree(a_gpu);
	return;
}


// len: number of leaves
// depth_num: number of depth maps
// h: height
// w: width
__global__ void update_cuda_sdf_kernel(Holder holder, unsigned int len, int depth_num, int h, int w){

    unsigned int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    unsigned int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

	if(threadId >= len) return;

    float phi = holder.center_xyz[3 * threadId + 0];
    float theta = holder.center_xyz[3 * threadId + 1];
    float center_depth = holder.center_xyz[3 * threadId + 2];

	float x = center_depth * sinf(theta) * cosf(phi);
    float y = -center_depth * cosf(theta);
    float z = center_depth * sinf(theta) * sinf(phi);


    float new_phi, new_theta;
    float new_x, new_y, new_z;
	float frame_depth;
	float mask;
	float depth;
	float weight = 0.0f;
	float distance_weight = 0.0f;
	float final_weight = 0.0f;

	int u, v;

	holder.sdf_value[threadId] = 0;
	holder.sdf_cnt[threadId] = 0;
	holder.sdf_weight[threadId] = 0;
	float trunc = holder.unit[0];
	float weight_trunc_scale = holder.weight_trunc_scale[0];
	float trunc_offset = holder.trunc_offset[0];
	float trunc_change = holder.trunc_change[0];
	float truncation_threshold = 0.0;

    for(int i = 0; i < depth_num; i++)
	{
		if (0 <= i)
		{
			new_x = holder.frame_rt[16 * i + 4 * 0 + 0] * x + holder.frame_rt[16 * i + 4 * 0 + 1] * y + holder.frame_rt[16 * i + 4 * 0 + 2] * z + holder.frame_rt[16 * i + 4 * 0 + 3];
			new_y = holder.frame_rt[16 * i + 4 * 1 + 0] * x + holder.frame_rt[16 * i + 4 * 1 + 1] * y + holder.frame_rt[16 * i + 4 * 1 + 2] * z + holder.frame_rt[16 * i + 4 * 1 + 3];
			new_z = holder.frame_rt[16 * i + 4 * 2 + 0] * x + holder.frame_rt[16 * i + 4 * 2 + 1] * y + holder.frame_rt[16 * i + 4 * 2 + 2] * z + holder.frame_rt[16 * i + 4 * 2 + 3];
			frame_depth = sqrtf(new_x * new_x + new_y * new_y + new_z * new_z);
			new_x = new_x / frame_depth;
			new_y = new_y / frame_depth;
			new_z = new_z / frame_depth;

			new_phi = (float)(3 * PI) / 2 - atan2f(new_z, new_x);
			new_theta = acosf(-new_y);
			u = round(new_theta * (float)(h) / PI - 0.5f);
			v = round(new_phi * (float)(w) / (2 * PI) - 0.5f);
			u = u < 0 ? (h + u) : u;
			u = u >= h ? (u - h) : u;
			v = v < 0 ? (w + v) : v;
			v = v >= w ? (v - w) : v;

			depth = 1.0f / holder.depth_map[h * w * i + w * u + v];
			weight = holder.weight_map[h * w * i + w * u + v];

			mask = holder.mask_depth[w * u + v];
			if (depth <= 0)
				continue;
			if (weight <= 0)
				continue;
			if (mask < 1)
				continue;

			if (depth < trunc_change){
				truncation_threshold = trunc_offset;
			}
			else{
				truncation_threshold = trunc * depth + (trunc_offset - trunc * trunc_change);
			}

			if (frame_depth > depth + truncation_threshold)
				continue;

			distance_weight = expf(-(depth * depth) / (weight_trunc_scale)) < 0.0001f ? 0.0001f : expf(-(depth * depth) / (weight_trunc_scale));
			weight = weight < 0.0001f ? 0.0001f : weight;
			final_weight = distance_weight * weight;

			holder.sdf_value[threadId] = (holder.sdf_value[threadId] * holder.sdf_weight[threadId] + (depth - frame_depth) * final_weight) / (holder.sdf_weight[threadId] + final_weight);
			holder.sdf_weight[threadId] = holder.sdf_weight[threadId] + final_weight;
			holder.sdf_cnt[threadId] = holder.sdf_cnt[threadId] + 1;
		}
	}

	holder.center_xyz[3 * threadId + 0] = x;
	holder.center_xyz[3 * threadId + 1] = y;
	holder.center_xyz[3 * threadId + 2] = z;

}

void cuda::update_cuda_alloc(unsigned int len, int depth_size, int depth_num){
	cudaMalloc((void**)&(holder.center_xyz), sizeof(float) * len * 3);
	cudaMalloc((void**)&(holder.sdf_value), sizeof(float) * len);
	cudaMalloc((void**)&(holder.cell_depth), sizeof(float) * len);
	cudaMalloc((void**)&(holder.sdf_cnt), sizeof(float) * len);
	cudaMalloc((void**)&(holder.sdf_weight), sizeof(float) * len);
	cudaMalloc((void**)&(holder.depth_map), sizeof(float) * depth_size * depth_num);
	cudaMalloc((void**)&(holder.weight_map), sizeof(float) * depth_size * depth_num);
	cudaMalloc((void**)&(holder.frame_rt), sizeof(float) * 16 * depth_num);
	cudaMalloc((void**)&(holder.unit), sizeof(float));
	cudaMalloc((void**)&(holder.weight_trunc_scale), sizeof(float));
	cudaMalloc((void**)&(holder.trunc_offset), sizeof(float));
	cudaMalloc((void**)&(holder.mask_depth), sizeof(float) * depth_size);
	cudaMalloc((void**)&(holder.trunc_change), sizeof(float));
}


void cuda::update_cuda_load(int mode, int i, float* src, int size){
	switch(mode){
	case 0:
        cudaMemcpy(holder.sdf_value + i, src, size, cudaMemcpyHostToDevice);
		break;
	case 1:
        cudaMemcpy(holder.sdf_cnt + i, src, size, cudaMemcpyHostToDevice);
		break;
	case 2:
        cudaMemcpy(holder.center_xyz + i, src, size, cudaMemcpyHostToDevice);
		break;
	case 3:
        cudaMemcpy(holder.center_inv + i, src, size, cudaMemcpyHostToDevice);
        break;
	case 4:
        cudaMemcpy(holder.depth_map + i, src, size, cudaMemcpyHostToDevice);
		break;
	case 5:
        cudaMemcpy(holder.frame_rt + i, src, size, cudaMemcpyHostToDevice);
		break;
	case 6:
		cudaMemcpy(holder.unit + i, src, size, cudaMemcpyHostToDevice);
		break;
	case 7:
		cudaMemcpy(holder.cell_depth + i, src, size, cudaMemcpyHostToDevice);
		break;
    case 8:
        cudaMemcpy(holder.sdf_value + i, src, size, cudaMemcpyHostToDevice);
        break;
    case 9:
        cudaMemcpy(holder.sdf_cnt + i, src, size, cudaMemcpyHostToDevice);
        break;
    case 10:
        cudaMemcpy(holder.center_xyz + i, src, size, cudaMemcpyHostToDevice);
        break;
	case 11:
		cudaMemcpy(holder.mask_depth + i, src, size, cudaMemcpyHostToDevice);
		break;
	case 12:
		cudaMemcpy(holder.sdf_weight + i, src, size, cudaMemcpyHostToDevice);
		break;
	case 13:
		cudaMemcpy(holder.weight_trunc_scale + i, src, size, cudaMemcpyHostToDevice);
		break;
	case 14:
		cudaMemcpy(holder.trunc_offset + i, src, size, cudaMemcpyHostToDevice);
		break;
	case 15:
		cudaMemcpy(holder.weight_map + i, src, size, cudaMemcpyHostToDevice);
		break;
	case 16:
		cudaMemcpy(holder.trunc_change + i, src, size, cudaMemcpyHostToDevice);
		break;
	}
}

void cuda::update_cuda_sdf(unsigned int len, int depth_num, int h, int w, float* mask_depth){

    if(len < 32 * 32){
        dim3 block(1, 1, 1);
        dim3 grid(len + 1, 1, 1);
        update_cuda_sdf_kernel<<<grid, block>>>(holder, len, depth_num, h, w);
	}
	else if(len < 32 * 32 * 512){
        dim3 block(32, 32, 1);
        dim3 grid((int)(len / 32 / 32) + 1, 1, 1);
        update_cuda_sdf_kernel<<<grid, block>>>(holder, len, depth_num, h, w);
	}
    else if(len < 32 * 32 * 512 * 512){
        dim3 block(32, 32, 1);
        dim3 grid((int)(len / 32 / 32 / 512) + 1, 512, 1);
        update_cuda_sdf_kernel<<<grid, block>>>(holder, len, depth_num, h, w);
    }
	else{
        dim3 block(32, 32, 1);
        dim3 grid((int)(len / 32 / 32 / 512 / 512) + 1, 512, 512);
        update_cuda_sdf_kernel<<<grid, block>>>(holder, len, depth_num, h, w);
	}
}

void cuda::update_cuda_unload(int mode, float* dst, int i, int size){

    switch(mode) {
        case 0:
            cudaMemcpy(dst, holder.sdf_value + i, size, cudaMemcpyDeviceToHost);
            break;
        case 1:
            cudaMemcpy(dst, holder.sdf_cnt + i, size, cudaMemcpyDeviceToHost);
            break;
        case 2:
            cudaMemcpy(dst, holder.center_xyz + 3 * i, size, cudaMemcpyDeviceToHost);
            break;
        case 3:
            cudaMemcpy(dst, holder.sdf_value + i, size, cudaMemcpyDeviceToHost);
            break;
        case 4:
            cudaMemcpy(dst, holder.sdf_cnt + i, size, cudaMemcpyDeviceToHost);
            break;
        case 5:
            cudaMemcpy(dst, holder.center_xyz + i, size, cudaMemcpyDeviceToHost);
            break;
    }
}






































