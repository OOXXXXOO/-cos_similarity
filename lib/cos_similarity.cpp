#include "lib/cos_similarity.h"

using namespace std;

bool cos_similarity(const std::vector<float>& a_data,
                        const std::vector<float>& b_data,
                        float & score)
{
    /* -------------------------------------------------------------------------- */
    /*                 cpp vector version cos_similarity function                 */
    /* -------------------------------------------------------------------------- */
    const int data_length = a_data.size();
    double ab_mult_add = 0.0f;
    double a_len = 0.0f;
    double b_len = 0.0f;    

    if((0 != data_length) and (b_data.size() != data_length))
    {
        return false;
    }

	//ab_mult_add:分母
	//sqrt(a_len * b_len) :分子
    for (std::size_t i = 0; i < data_length; i++) 
    {
        ab_mult_add += (a_data[i] * b_data[i]);
        a_len += a_data[i] * a_data[i];
        b_len += b_data[i] * b_data[i];
    }

    score = (float) (ab_mult_add / sqrt(a_len * b_len));
    return true;
}


double neon_cos_similarity(const std::vector<float>& a_data,
                         const std::vector<float>& b_data, 
                         float & score)
{
    /* -------------------------------------------------------------------------- */
    /*                  cpp neon version cos_similarity function                  */
    /* -------------------------------------------------------------------------- */
    const int data_length = a_data.size();
    double ab_mult_add = 0.0f;
    double a_len = 0.0f;
    double b_len = 0.0f;    
    // float32x4_t 由4个32位的float组成的数据类型，对它做一次操作，4个float都被用到
    float32x4_t ab_mult_add_vec = vdupq_n_f32(0);// 存储的四个 float32 都初始化为 0，寄存器ab_mult_add_vec
    float32x4_t a_qua_sum_vec = vdupq_n_f32(0);
    float32x4_t b_qua_sum_vec = vdupq_n_f32(0);

    if((0 != data_length) and (b_data.size() != data_length))
    {
        return false;
    }

    float* a_data_ptr = (float*)a_data.data();
    float* b_data_ptr = (float*)b_data.data();
    for (int i = 0; i < data_length / 4; ++i) //四个数据为一组.或有剩余数据,下文处理
    {
        float32x4_t a_data_vec = vld1q_f32(a_data_ptr + 4*i);// 加载 data + 4*i 地址起始的 4 个 float 数据到寄存器tmp_vec
        float32x4_t b_data_vec = vld1q_f32(b_data_ptr + 4*i);
        ab_mult_add_vec += vmulq_f32(a_data_vec, b_data_vec);//点乘 [a0*b0, a1*b1, a2*b2, a3*b3],并累加
        a_qua_sum_vec += vmulq_f32(a_data_vec, a_data_vec);//点乘 [a0*a0, a1*a1, a2*a2, a3*a3],并累加,就是平方和
        b_qua_sum_vec += vmulq_f32(b_data_vec, b_data_vec);
    }
    //将累加结果寄存器中的所有元素相加得到最终累加值
    ab_mult_add += vgetq_lane_f32(ab_mult_add_vec, 0);
    ab_mult_add += vgetq_lane_f32(ab_mult_add_vec, 1);
    ab_mult_add += vgetq_lane_f32(ab_mult_add_vec, 2);
    ab_mult_add += vgetq_lane_f32(ab_mult_add_vec, 3);
    // std::cout << "neon ab_mult_add = " << ab_mult_add  << std::endl;

    a_len += vgetq_lane_f32(a_qua_sum_vec, 0);
    a_len += vgetq_lane_f32(a_qua_sum_vec, 1);
    a_len += vgetq_lane_f32(a_qua_sum_vec, 2);
    a_len += vgetq_lane_f32(a_qua_sum_vec, 3);
    // std::cout << "neon a_len = " << a_len  << std::endl;

    b_len += vgetq_lane_f32(b_qua_sum_vec, 0);
    b_len += vgetq_lane_f32(b_qua_sum_vec, 1);
    b_len += vgetq_lane_f32(b_qua_sum_vec, 2);
    b_len += vgetq_lane_f32(b_qua_sum_vec, 3);
    // std::cout << "neon b_len = " << b_len  << std::endl;

    int odd = data_length & 3;//数组长度除有4余数
    if(0 < odd) 
    {
        //处理剩余数据
        // std::cout << "data_length = " << data_length << ", odd = " << odd << std::endl;
        for(int i = data_length - odd; i < data_length; i++) 
        {
            ab_mult_add += (a_data[i] * b_data[i]);
            a_len += a_data[i] * a_data[i];
            b_len += b_data[i] * b_data[i];
        }
    }

    score = (float) (ab_mult_add / sqrt(a_len * b_len));

    return true;
}


float cv_cos_similarity(const cv::Mat& first,const cv::Mat& second)
{
    /* -------------------------------------------------------------------------- */
    /*               cpp opencv::mat version cos_similarity function              */
    /* -------------------------------------------------------------------------- */
    double dotSum=first.dot(second);//内积
    double normFirst=cv::norm(first);//取模
    double normSecond=cv::norm(second); 
    if(normFirst!=0 && normSecond!=0){
        return dotSum/(normFirst*normSecond);
    }
}

bool eigen_cos_similarity(const Eigen::VectorXf& first,
                             const Eigen::VectorXf& second, float* result) {

    /* -------------------------------------------------------------------------- */
    /*                  cpp eigen version cos_similarity function                 */
    /* -------------------------------------------------------------------------- */
  if (result == nullptr) {
    return false;
  }
  if (first.size() != second.size() || first.size() == 0) {
    return false;
  }
  const float norm = first.norm() * second.norm();
  if (std::abs(norm) < 1e-6) {
    return false;
  }
  *result = first.dot(second) / norm;
  return true;
}