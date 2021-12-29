#include <arm_neon.h>
#include <math.h>
#include <stdio.h>

#include <Eigen/Core>
#include <iostream>
#include <vector>

#include "opencv2/opencv.hpp"

using namespace std;

bool cos_similarity(const std::vector<float>& a_data,
                    const std::vector<float>& b_data, float& score);

double neon_cos_similarity(const std::vector<float>& a_data,
                           const std::vector<float>& b_data, float& score);

float cv_cos_similarity(const cv::Mat& first, const cv::Mat& second);

bool eigen_cos_similarity(const Eigen::VectorXf& first,
                          const Eigen::VectorXf& second, float* result);