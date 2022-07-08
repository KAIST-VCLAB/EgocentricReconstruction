#define PI 3.1415926535897932384626433832795028841971693993751058209749445923078164062f
#define K_SIZE 2

extern "C"{

__device__ float3 matMul33(const float* rt, float3 vect)
{
    float3 result;
    result.x = rt[0] * vect.x + rt[1] * vect.y + rt[2] * vect.z;
    result.y = rt[3] * vect.x + rt[4] * vect.y + rt[5] * vect.z;
    result.z = rt[6] * vect.x + rt[7] * vect.y + rt[8] * vect.z;


    return result;
}

__global__ void rectifyLL(const unsigned char* refImg, const unsigned char* warpImg, const float* ref_r,
const float* warp_r, unsigned char* ref_ll, unsigned char* neighbor_ll, const long long cols, const long long rows)
{
    long long x = blockDim.x * blockIdx.x + threadIdx.x;
    long long y = blockDim.y * blockIdx.y + threadIdx.y;

    long long indexImg = y * cols + x;

    float theta = (1 - 2 * (float)(y+0.5f) / (rows)) * PI;
    float phi =  PI * (1 - (float)(x+0.5f) / (cols));

    float3 xyz_;
    float3 xyz;

    long long eq_rows = cols;
    long long eq_cols = rows;


    xyz_.x = __cosf(phi);
    xyz_.y = - __sinf(phi) * __sinf(theta);
    xyz_.z = __sinf(phi) * __cosf(theta);

    xyz = matMul33(ref_r, xyz_);

    if(xyz.y >= 1) xyz.y = 1;
    if(xyz.y <= -1) xyz.y = -1;
    if(xyz.x >= 1) xyz.x = 1;
    if(xyz.x <= -1) xyz.x = -1;
    if(xyz.z >= 1) xyz.z = 1;
    if(xyz.z <= -1) xyz.z = -1;

    float theta_new = acosf(-xyz.y);
    float phi_new = 3 * PI / 2 - atan2f(xyz.z, xyz.x);

    float u = theta_new * (float)(eq_rows) / PI - 0.5f;
    float v = phi_new * (float)(eq_cols) / (2 * PI) - 0.5f;

    long long u1, u2, v1, v2;
    u1 = (long long)u;
    v1 = (long long)v;
    u2 = u1 + 1;
    v2 = v1 + 1;

    float w1, w2, w3, w4;
    w1 = ((float)u2 - u) * ((float)v2 - v);
    w2 = ((float)u2 - u) * ((float)v - v1);
    w3 = ((float)u - u1) * ((float)v2 - v);
    w4 = ((float)u - u1) * ((float)v - v1);


    u1 = u1 < 0? (eq_rows + u1) : u1;  u1 = u1 >= eq_rows? (u1 - eq_rows): u1;
    v1 = v1 < 0? (eq_cols + v1) : v1;  v1 = v1 >= eq_cols? (v1 - eq_cols): v1;
    u2 = u2 < 0? (eq_rows + u2) : u2;  u2 = u2 >= eq_rows? (u2 - eq_rows): u2;
    v2 = v2 < 0? (eq_cols + v2) : v2;  v2 = v2 >= eq_cols? (v2 - eq_cols): v2;

    float3 p1, p2, p3, p4;
    p1.x = (float)refImg[(u1 * eq_cols + v1) * 3 + 0];
    p1.y = (float)refImg[(u1 * eq_cols + v1) * 3 + 1];
    p1.z = (float)refImg[(u1 * eq_cols + v1) * 3 + 2];

    p2.x = (float)refImg[(u1 * eq_cols + v2) * 3 + 0];
    p2.y = (float)refImg[(u1 * eq_cols + v2) * 3 + 1];
    p2.z = (float)refImg[(u1 * eq_cols + v2) * 3 + 2];

    p3.x = (float)refImg[(u2 * eq_cols + v1) * 3 + 0];
    p3.y = (float)refImg[(u2 * eq_cols + v1) * 3 + 1];
    p3.z = (float)refImg[(u2 * eq_cols + v1) * 3 + 2];

    p4.x = (float)refImg[(u2 * eq_cols + v2) * 3 + 0];
    p4.y = (float)refImg[(u2 * eq_cols + v2) * 3 + 1];
    p4.z = (float)refImg[(u2 * eq_cols + v2) * 3 + 2];

    float3 return_val;

    return_val.x = p1.x * w1 + p2.x * w2 +  p3.x * w3 + p4.x * w4;
    return_val.y = p1.y * w1 + p2.y * w2 +  p3.y * w3 + p4.y * w4;
    return_val.z = p1.z * w1 + p2.z * w2 +  p3.z * w3 + p4.z * w4;
    ref_ll[indexImg * 3] = (unsigned char)return_val.x;
    ref_ll[indexImg * 3 + 1] = (unsigned char)return_val.y;
    ref_ll[indexImg * 3 + 2] = (unsigned char)return_val.z;

    xyz = matMul33(warp_r, xyz_);

    if(xyz.y >= 1) xyz.y = 1;
    if(xyz.y <= -1) xyz.y = -1;
    if(xyz.x >= 1) xyz.x = 1;
    if(xyz.x <= -1) xyz.x = -1;
    if(xyz.z >= 1) xyz.z = 1;
    if(xyz.z <= -1) xyz.z = -1;

    theta_new = acosf(-xyz.y);
    phi_new = 3 * PI / 2 - atan2f(xyz.z, xyz.x);

    u = theta_new * (float)(eq_rows) / PI - 0.5f;
    v = phi_new * (float)(eq_cols) / (2 * PI) - 0.5f;

    u1 = (long long)u;
    v1 = (long long)v;
    u2 = u1 + 1;
    v2 = v1 + 1;

    w1 = ((float)u2 - u) * ((float)v2 - v);
    w2 = ((float)u2 - u) * ((float)v - v1);
    w3 = ((float)u - u1) * ((float)v2 - v);
    w4 = ((float)u - u1) * ((float)v - v1);

    u1 = u1 < 0? (eq_rows + u1) : u1;  u1 = u1 >= eq_rows? (u1 - eq_rows): u1;
    v1 = v1 < 0? (eq_cols + v1) : v1;  v1 = v1 >= eq_cols? (v1 - eq_cols): v1;
    u2 = u2 < 0? (eq_rows + u2) : u2;  u2 = u2 >= eq_rows? (u2 - eq_rows): u2;
    v2 = v2 < 0? (eq_cols + v2) : v2;  v2 = v2 >= eq_cols? (v2 - eq_cols): v2;


    p1.x = (float)warpImg[(u1 * eq_cols + v1) * 3 + 0];
    p1.y = (float)warpImg[(u1 * eq_cols + v1) * 3 + 1];
    p1.z = (float)warpImg[(u1 * eq_cols + v1) * 3 + 2];

    p2.x = (float)warpImg[(u1 * eq_cols + v2) * 3 + 0];
    p2.y = (float)warpImg[(u1 * eq_cols + v2) * 3 + 1];
    p2.z = (float)warpImg[(u1 * eq_cols + v2) * 3 + 2];

    p3.x = (float)warpImg[(u2 * eq_cols + v1) * 3 + 0];
    p3.y = (float)warpImg[(u2 * eq_cols + v1) * 3 + 1];
    p3.z = (float)warpImg[(u2 * eq_cols + v1) * 3 + 2];

    p4.x = (float)warpImg[(u2 * eq_cols + v2) * 3 + 0];
    p4.y = (float)warpImg[(u2 * eq_cols + v2) * 3 + 1];
    p4.z = (float)warpImg[(u2 * eq_cols + v2) * 3 + 2];

    return_val.x = p1.x * w1 + p2.x * w2 +  p3.x * w3 + p4.x * w4;
    return_val.y = p1.y * w1 + p2.y * w2 +  p3.y * w3 + p4.y * w4;
    return_val.z = p1.z * w1 + p2.z * w2 +  p3.z * w3 + p4.z * w4;
    neighbor_ll[indexImg * 3] = (unsigned char)return_val.x;
    neighbor_ll[indexImg * 3 + 1] = (unsigned char)return_val.y;
    neighbor_ll[indexImg * 3 + 2] = (unsigned char)return_val.z;
}

__global__ void rectifyLLdepth(const float* refImg, const float* warpImg, const float* ref_r,
const float* warp_r, float* ref_ll, float* neighbor_ll, const long long cols, const long long rows)
{
    long long x = blockDim.x * blockIdx.x + threadIdx.x;
    long long y = blockDim.y * blockIdx.y + threadIdx.y;

    long long indexImg = y * cols + x;

    float theta = (1 - 2 * (float)(y+0.5f) / (rows)) * PI;
    float phi =  PI * (1 - (float)(x+0.5f) / (cols));

    float3 xyz_;
    float3 xyz;

    long long eq_rows = cols;
    long long eq_cols = rows;


    xyz_.x = __cosf(phi);
    xyz_.y = - __sinf(phi) * __sinf(theta);
    xyz_.z = __sinf(phi) * __cosf(theta);

    xyz = matMul33(ref_r, xyz_);

    if(xyz.y >= 1) xyz.y = 1;
    if(xyz.y <= -1) xyz.y = -1;
    if(xyz.x >= 1) xyz.x = 1;
    if(xyz.x <= -1) xyz.x = -1;
    if(xyz.z >= 1) xyz.z = 1;
    if(xyz.z <= -1) xyz.z = -1;

    float theta_new = acosf(-xyz.y);
    float phi_new = 3 * PI / 2 - atan2f(xyz.z, xyz.x);

    float u = theta_new * (float)(eq_rows) / PI - 0.5f;
    float v = phi_new * (float)(eq_cols) / (2 * PI) - 0.5f;

    long long u1, v1;
    u1 = (long long)(u);
    v1 = (long long)(v);

    u1 = u1 < 0? (eq_rows + u1) : u1;  u1 = u1 >= eq_rows? (u1 - eq_rows): u1;
    v1 = v1 < 0? (eq_cols + v1) : v1;  v1 = v1 >= eq_cols? (v1 - eq_cols): v1;

    ref_ll[indexImg] = refImg[u1 * eq_cols + v1];


    xyz = matMul33(warp_r, xyz_);

    if(xyz.y >= 1) xyz.y = 1;
    if(xyz.y <= -1) xyz.y = -1;
    if(xyz.x >= 1) xyz.x = 1;
    if(xyz.x <= -1) xyz.x = -1;
    if(xyz.z >= 1) xyz.z = 1;
    if(xyz.z <= -1) xyz.z = -1;

    theta_new = acosf(-xyz.y);
    phi_new = 3 * PI / 2 - atan2f(xyz.z, xyz.x);

    u = theta_new * (float)(eq_rows) / PI - 0.5f;
    v = phi_new * (float)(eq_cols) / (2 * PI) - 0.5f;

    u1 = (long long)(u);
    v1 = (long long)(v);

    u1 = u1 < 0? (eq_rows + u1) : u1;  u1 = u1 >= eq_rows? (u1 - eq_rows): u1;
    v1 = v1 < 0? (eq_cols + v1) : v1;  v1 = v1 >= eq_cols? (v1 - eq_cols): v1;

    neighbor_ll[indexImg] = warpImg[u1 * eq_cols + v1];

}

__global__ void warpToRefDepth(const float* distance, float* ref_distance, const float* ref_r, const long long cols, const long long rows)
{
    long long x = blockDim.x * blockIdx.x + threadIdx.x;
    long long y = blockDim.y * blockIdx.y + threadIdx.y;

    long long indexImg = y * cols + x;


    float theta = PI * (float)(y+0.5f) / (rows);
    float phi =  3 * PI / 2 - 2 * PI * (float)(x+0.5f) / (cols);

    float3 xyz_;
    float3 xyz;

    xyz_.x = __sinf(theta) * __cosf(phi);
    xyz_.y = -__cosf(theta);
    xyz_.z = __sinf(theta) * __sinf(phi);


    xyz = matMul33(ref_r, xyz_);

    if(xyz.y >= 1) xyz.y = 1;
    if(xyz.y <= -1) xyz.y = -1;
    if(xyz.x >= 1) xyz.x = 1;
    if(xyz.x <= -1) xyz.x = -1;
    if(xyz.z >= 1) xyz.z = 1;
    if(xyz.z <= -1) xyz.z = -1;

    float phi_new = acosf(xyz.x);
    float theta_new = atan2f(-xyz.y, xyz.z);

    long long rows_ll = cols;
    long long cols_ll = rows;

    long long u = (long long)((rows_ll) * (1 - (theta_new / PI)) / 2 - 0.5f);
    long long v = (long long)((cols_ll) * (1 - (phi_new / PI)) - 0.5f);

    u = u < 0? (rows_ll + u) : u;  u = u >= rows_ll? (u - rows_ll): u;
    v = v < 0? (cols_ll + v) : v;  v = v >= cols_ll? (v - cols_ll): v;


    ref_distance[indexImg] = distance[u * cols_ll + v];


}

__global__ void depthFromFlow(const float* flow, float* distance, float* weight, const float* baseline,
const long long cols, const long long rows)
{
    long long x = blockDim.x * blockIdx.x + threadIdx.x;
    long long y = blockDim.y * blockIdx.y + threadIdx.y;

    long long indexImg = y * cols + x;

    float u = flow[indexImg] < 0.001f ? 0.001f : flow[indexImg];


    float phi_left = PI * ((float)cols - (float)x - 1 + 0.5)/(cols);
    float delta = (u * PI) / (cols);
    float d = baseline[0] * abs((__sinf(phi_left + delta)) / (__sinf(delta)));

    distance[indexImg] = d;


    if(u < 0){
        distance[indexImg] = -1;
        return;
    }

    if(delta < 2e-07f){
     distance[indexImg] = -1;
     return;
    }

}

__global__ void warpToRef(const float* distance, float* ref_distance, const float* ref_weight_horizontal, float* ref_weight, const float* ref_r,
const long long cols, const long long rows, const float* r)
{
    long long x = blockDim.x * blockIdx.x + threadIdx.x;
    long long y = blockDim.y * blockIdx.y + threadIdx.y;

    long long indexImg = y * cols + x;


    float theta = PI * (float)(y+0.5f) / (rows);
    float phi =  3 * PI / 2 - 2 * PI * (float)(x+0.5f) / (cols);

    float3 xyz_;
    float3 xyz;

    xyz_.x = __sinf(theta) * __cosf(phi);
    xyz_.y = -__cosf(theta);
    xyz_.z = __sinf(theta) * __sinf(phi);


    xyz = matMul33(ref_r, xyz_);

    if(xyz.y >= 1) xyz.y = 1;
    if(xyz.y <= -1) xyz.y = -1;
    if(xyz.x >= 1) xyz.x = 1;
    if(xyz.x <= -1) xyz.x = -1;
    if(xyz.z >= 1) xyz.z = 1;
    if(xyz.z <= -1) xyz.z = -1;

    float phi_new = acosf(xyz.x);
    float theta_new = atan2f(-xyz.y, xyz.z);

    long long rows_ll = cols;
    long long cols_ll = rows;

    long long u = (long long)((rows_ll) * (1 - (theta_new / PI)) / 2 - 0.5f);
    long long v = (long long)((cols_ll) * (1 - (phi_new / PI)) - 0.5f);

    u = u < 0? (rows_ll + u) : u;  u = u >= rows_ll? (u - rows_ll): u;
    v = v < 0? (cols_ll + v) : v;  v = v >= cols_ll? (v - cols_ll): v;


    ref_distance[indexImg] = distance[u * cols_ll + v];

    // step weight
    ref_weight[indexImg] = 0.0;

    if (r[0] < (v / (float)cols_ll) && (v / (float)cols_ll) < 1 - r[0]){
         ref_weight[indexImg] = 1.0;
    }
    if(distance[u * cols_ll + v] < 0){
        ref_weight[indexImg] = 0.0;
    }
}


__global__ void weightSum(const float* distance, const float* weight, float* second_depth_list, float* second_weight_list, const long long second_num, const long long second_idx,
const long long cols, const long long rows)
{
    long long x = blockDim.x * blockIdx.x + threadIdx.x;
    long long y = blockDim.y * blockIdx.y + threadIdx.y;

    long long indexImg = y * cols + x;
    long long indexVol = second_num * indexImg + second_idx;


    second_depth_list[indexVol] = distance[indexImg];
    second_weight_list[indexVol] = weight[indexImg];

}


// end of extern C
}
