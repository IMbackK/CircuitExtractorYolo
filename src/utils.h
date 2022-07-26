#pragma once

#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>
#include <utility>
#include "log.h"

typedef enum {
	C_DIRECTION_HORIZ,
	C_DIRECTION_VERT,
	C_DIRECTION_UNKOWN
} DirectionHint;

bool pointInRect(const cv::Point2i& point, const cv::Rect& rect);

bool pointIsOnLine(const cv::Point2i& point, const cv::Vec4f& line, double tollerance);

std::pair<cv::Point2i, cv::Point2i> furthestPoints(std::vector<cv::Point2i> points);

void deduplicatePoints(std::vector<cv::Point2i>& points, double tollerance);

double pointDist(const cv::Point2i& pointA, const cv::Point2i& pointB);


void drawLineSegments(cv::Mat& in, cv::InputArray lines, cv::Scalar color = cv::Scalar(0, 0, 255)); //taken fom opencv line detector src

double lineDotProd(const cv::Vec4f lineA, const cv::Vec4f lineB);

double closestLineEndpoint(const cv::Vec4f lineA, const cv::Vec4f lineB);


std::string getMatType(const cv::Mat& mat);

void printMatInfo(const cv::Mat& mat, const std::string& prefix = "", const Log::Level lvl = Log::INFO);

void printMat(const cv::Mat& mat, const Log::Level lvl = Log::INFO);


bool rectsIntersect(const cv::Rect& a, const cv::Rect& b);

cv::Rect rectFromPoints(const std::vector<cv::Point>& points);

cv::Rect padRect(const cv::Rect& rect, double xPadPercent, double yPadPercent, int minimumPad = 1);

cv::Mat getMatPlane(cv::Mat& in, int plane);

cv::Mat getMatPlane4d(cv::Mat& in, int plane);

std::pair<double, double> getRectXYPaddingPercents(DirectionHint hint, double tolleranceFactor);
