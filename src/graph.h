#pragma once

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <string>
#include <utility>
#include <vector>

class Graph
{
private:
	float prob;
	cv::Mat image;
	cv::Rect rect;
	std::vector<std::pair<double, double>> points;

public:
	Graph() = default;
	Graph(const cv::Mat& imageI, float probI, cv::Rect rectI);
	void setImage(cv::Mat& imageI) {image = imageI;}
	cv::Mat getImage() {return image;}
	void setRect(const cv::Rect& rectI) {rect = rectI;}
	cv::Rect getRect() {return rect;}
	void setProb(float in) {prob = in;}
	float getProb() {return prob;}
	void dropImage();
	std::vector<std::pair<double, double>>& getPoints();
};
