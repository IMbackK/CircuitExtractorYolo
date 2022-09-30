#include "graph.h"

std::vector<std::pair<double, double>>& Graph::getPoints()
{
	//TODO
	return points;
}

Graph::Graph(const cv::Mat& imageI, float probI, cv::Rect rectI): image(imageI), prob(probI), rect(rectI)
{

}

void Graph::dropImage()
{
	image.release();
}
