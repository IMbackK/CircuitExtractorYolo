#pragma once
#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>
#include "circut.h"

std::vector<cv::Vec4f> lineDetect(cv::Mat in);

std::pair<cv::Point2i, cv::Point2i> lineToPoints(const cv::Vec4f& line);

bool lineCrossesOrtho(const cv::Vec4f& lineA, const cv::Vec4f& lineB, double tollerance);

void clipLinesAgainstRect(std::vector<cv::Vec4f>& lines, const cv::Rect& rect);
