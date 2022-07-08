#pragma once

#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>
#include <utility>

bool pointInRect(const cv::Point2i& point, const cv::Rect& rect);

std::pair<cv::Point2i, cv::Point2i> furthestPoints(std::vector<cv::Point2i> points);

void deduplicatePoints(std::vector<cv::Point2i>& points, double tollerance);

//taken fom opencv line detector src
void drawLineSegments(cv::Mat& in, cv::InputArray lines, cv::Scalar color = cv::Scalar(0, 0, 255));

double lineDotProd(const cv::Vec4f lineA, const cv::Vec4f lineB);

double closestLineEndpoint(const cv::Vec4f lineA, const cv::Vec4f lineB);

bool pointIsOnLine(const cv::Point2i& point, const cv::Vec4f& line, double tollerance);
