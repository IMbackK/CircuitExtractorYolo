#pragma once

#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>

bool pointInRect(const cv::Point2i& point, const cv::Rect& rect);

//taken fom opencv line detector src
void drawLineSegments(cv::Mat& in, cv::InputArray lines);

double lineDotProd(const cv::Vec4f lineA, const cv::Vec4f lineB);

double closestLineEndpoint(const cv::Vec4f lineA, const cv::Vec4f lineB, bool* firstInA = nullptr, bool* firstInB = nullptr);
