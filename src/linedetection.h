#pragma once
#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>

std::vector<cv::Vec4f> lineDetect(cv::Mat in);

void eraseLinesInBox(std::vector<cv::Vec4f>& lines, const cv::Rect& rect);
