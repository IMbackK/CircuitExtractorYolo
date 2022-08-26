#include "linedetection.h"

#include <limits>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <cmath>
#include <algorithm>

#include "thinning.h"
#include "utils.h"
#include "log.h"

static constexpr double SCALE_FACTOR = 2.0;

static constexpr double ORTHO_TRESH = 0.08;

std::pair<cv::Point2i, cv::Point2i> lineToPoints(const cv::Vec4f& line)
{
	cv::Point2i pointA;
	pointA.x = line[0];
	pointA.y = line[1];

	cv::Point2i pointB;
	pointB.x = line[2];
	pointB.y = line[3];

	return std::pair<cv::Point2i, cv::Point2i>(pointA, pointB);
}

static void eraseLinesInBox(std::vector<cv::Vec4f>& lines, const cv::Rect& rect)
{
	for(size_t i = 0; i < lines.size(); ++i)
	{
		std::pair<cv::Point2i, cv::Point2i> points = lineToPoints(lines[i]);

		if(pointInRect(points.first, rect) && pointInRect(points.second, rect))
		{
			lines.erase(lines.begin()+i);
			--i;
		}
	}
}

void clipLinesAgainstRect(std::vector<cv::Vec4f>& lines, const cv::Rect& rect)
{
	eraseLinesInBox(lines, rect);

	for(size_t i = 0; i < lines.size(); ++i)
	{
		std::pair<cv::Point2i, cv::Point2i> points = lineToPoints(lines[i]);
		std::pair<cv::Point2i, cv::Point2i> pointsClipped = points;
		if(cv::clipLine(rect, pointsClipped.first, pointsClipped.second))
		{
			if(points.first == pointsClipped.first || points.first == pointsClipped.second)
			{
				if(points.first == pointsClipped.first)
				{
					lines[i][0] = pointsClipped.second.x;
					lines[i][1] = pointsClipped.second.y;
				}
				else
				{
					lines[i][0] = pointsClipped.first.x;
					lines[i][1] = pointsClipped.first.y;
				}
			}
			else if(points.second == pointsClipped.first || points.second == pointsClipped.second)
			{
				if(points.second == pointsClipped.first)
				{
					lines[i][2] = pointsClipped.second.x;
					lines[i][3] = pointsClipped.second.y;
				}
				else
				{
					lines[i][2] = pointsClipped.first.x;
					lines[i][3] = pointsClipped.first.y;
				}
			}
			else
			{
				if(pointDist(points.first, pointsClipped.first) < pointDist(points.first, pointsClipped.second))
				{
					lines[i][2] = pointsClipped.first.x;
					lines[i][3] = pointsClipped.first.y;

					cv::Vec4f newline;
					newline[0] = pointsClipped.second.x;
					newline[1] = pointsClipped.second.y;
					newline[2] = points.second.x;
					newline[3] = points.second.y;
					lines.push_back(newline);
				}
				else
				{
					lines[i][2] = pointsClipped.second.x;
					lines[i][3] = pointsClipped.second.y;

					cv::Vec4f newline;
					newline[0] = pointsClipped.first.x;
					newline[1] = pointsClipped.first.y;
					newline[2] = points.second.x;
					newline[3] = points.second.y;
					lines.push_back(newline);
				}
			}
		}
	}
}

static void removeShort(std::vector<cv::Vec4f>& lines, double lengthThresh)
{
	for(size_t i = 0; i < lines.size(); ++i)
	{
		std::pair<cv::Point2i, cv::Point2i> points = lineToPoints(lines[i]);

		double norm = cv::norm(points.first-points.second);

		if(norm < lengthThresh)
		{
			lines.erase(lines.begin()+i);
			--i;
		}
	}
}

static double lineFurthestEndDistance(const cv::Vec4f lineA, const cv::Vec4f lineB)
{
	std::pair<cv::Point2i, cv::Point2i> pointsA = lineToPoints(lineA);
	std::pair<cv::Point2i, cv::Point2i> pointsB = lineToPoints(lineB);

	float closestA = std::numeric_limits<float>::max();
	float closestB = std::numeric_limits<float>::max();

	cv::LineIterator masterLineIt(pointsA.first, pointsA.second, 8);

	for(int k = 0; k < masterLineIt.count; ++k, ++masterLineIt)
	{
		float normA = cv::norm(pointsB.first - masterLineIt.pos());
		float normB = cv::norm(pointsB.second - masterLineIt.pos());

		if(normA < closestA)
			closestA = normA;
		if(normB < closestB)
			closestB = normB;
	}
	return std::max(closestA, closestB);
}

static double lineDuplicationScore(const cv::Vec4f lineA, const cv::Vec4f lineB, double distanceThresh)
{
	cv::Vec2f vectorA;
	vectorA[0] = std::abs(lineA[0] - lineA[1]);
	vectorA[1] = std::abs(lineA[2] - lineA[3]);
	vectorA = vectorA/cv::norm(vectorA);

	cv::Vec2f vectorB;
	vectorB[0] = std::abs(lineB[0] - lineB[1]);
	vectorB[1] = std::abs(lineB[2] - lineB[3]);
	vectorB = vectorB/cv::norm(vectorA);

	double dprod = lineDotProd(lineA, lineB);

	double normA = cv::norm(vectorA);
	double normB = cv::norm(vectorB);
	double lengthRatio = normA > normB ? normB/normA : normA/normB;

	double linedistance = lineFurthestEndDistance(lineA, lineB);
	double distanceScore = -0.5*((linedistance-distanceThresh)/distanceThresh)+1;

	if(distanceScore > 1)
		distanceScore = 1;
	else if(distanceScore < 0)
		distanceScore = 0;

	double score = std::max(dprod*distanceScore, lengthRatio*dprod*(distanceScore+0.5));
	return std::min(score, 1.0);
}

static void deduplicateLines(std::vector<cv::Vec4f>& lines, double distanceThresh)
{
	for(size_t i = 0; i < lines.size(); ++i)
	{
		for(size_t j = 0; j < lines.size(); ++j)
		{
			if(j == i)
				continue;

			if(lineDuplicationScore(lines[i], lines[j], distanceThresh) > 0.8)
			{
				lines.erase(lines.begin()+j);
				--j;
				--i;
				break;
			}
		}
	}
}

static void mergeCloseInlineLines(std::vector<cv::Vec4f>& lines, double tollerance)
{
	for(size_t i = 0; i < lines.size(); ++i)
	{
		for(size_t j = 0; j < lines.size(); ++j)
		{
			if(i == j)
				continue;

			double dprod = lineDotProd(lines[i], lines[j]);
			if(dprod < 1-ORTHO_TRESH)
				continue;

			double closestEndDist = closestLineEndpoint(lines[i], lines[j]);

			if(closestEndDist < tollerance)
			{
				std::vector<cv::Point2i> points;
				points.push_back(cv::Point2i(lines[i][0], lines[i][1]));
				points.push_back(cv::Point2i(lines[i][2], lines[i][3]));
				points.push_back(cv::Point2i(lines[j][0], lines[j][1]));
				points.push_back(cv::Point2i(lines[j][2], lines[j][3]));
				std::pair<cv::Point2i, cv::Point2i> furthest = furthestPoints(points);
				cv::Vec4f newline(furthest.first.x, furthest.first.y, furthest.second.x, furthest.second.y);
				cv::Vec4f xAxis(0, 0, 1, 0);

				double dprod = lineDotProd(newline, xAxis);
				if(dprod < 1-ORTHO_TRESH && dprod > ORTHO_TRESH)
					continue;

				lines[i] = newline;
				lines.erase(lines.begin()+j);

				--i;
				--j;
				break;
			}
		}
	}
}

//TODO make faster by checking bounding boxes first
bool lineCrossesOrtho(const cv::Vec4f& lineA, const cv::Vec4f& lineB, double tollerance)
{
	double dprod = lineDotProd(lineA, lineB);
	if(dprod > ORTHO_TRESH)
		return false;

	cv::LineIterator lineAIt(cv::Point2i(lineA[0], lineA[1]), cv::Point2i(lineA[2], lineA[3]), 8);

	for(int i = 0; i < lineAIt.count; ++i, ++lineAIt)
	{
		cv::LineIterator lineBIt(cv::Point2i(lineB[0], lineB[1]), cv::Point2i(lineB[2], lineB[3]), 8);
		for(int j = 0; j < lineBIt.count; ++j, ++lineBIt)
		{
			if(cv::norm(lineBIt.pos()-lineAIt.pos()) < tollerance)
				return true;
		}
	}

	return false;
}

std::vector<cv::Vec4f> lineDetect(cv::Mat in)
{
	cv::Mat work;
	cv::Mat vizualization;
	std::vector<cv::Vec4f> lines;

	cv::cvtColor(in, work, cv::COLOR_BGR2GRAY);
	cv::resize(work, work, cv::Size(), SCALE_FACTOR, SCALE_FACTOR, cv::INTER_LINEAR);
	work.convertTo(work, CV_8U, 1);
	cv::threshold(work, work, 2*std::numeric_limits<uint8_t>::max()/3, std::numeric_limits<uint8_t>::max(), cv::THRESH_BINARY);
	cv::bitwise_not(work, work);
	thinning(work, work, THINNING_ZHANGSUEN);

	std::shared_ptr<cv::LineSegmentDetector> detector = cv::createLineSegmentDetector(cv::LSD_REFINE_STD, 1, 2, 2.0, 50.5);

	detector->detect(work, lines);

	removeShort(lines, std::max(work.rows/50.0, 4.0));

	Log(Log::WARN)<<"thresh "<<std::max(work.rows/100.0, 5.0);
	deduplicateLines(lines, std::max(work.rows/100.0, 5.0));

	mergeCloseInlineLines(lines, std::max(work.rows/50.0, 10.0));

	if(Log::level == Log::SUPERDEBUG)
	{
		work.copyTo(vizualization);
		drawLineSegments(vizualization, lines);
		cv::imshow("Viewer", vizualization);
		cv::waitKey(0);
	}

	for(cv::Vec4f& line : lines)
	{
		line[0] /= SCALE_FACTOR;
		line[1] /= SCALE_FACTOR;
		line[2] /= SCALE_FACTOR;
		line[3] /= SCALE_FACTOR;
	}

	return lines;
}
