#include "linedetection.h"

#include <limits>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <cmath>
#include <algorithm>
#include <opencv2/ximgproc.hpp>

#include "utils.h"
#include "log.h"

void eraseLinesInBox(std::vector<cv::Vec4f>& lines, const cv::Rect& rect)
{
	for(size_t i = 0; i < lines.size(); ++i)
	{
		cv::Point2i pointA;
		pointA.x = lines[i][0];
		pointA.y = lines[i][1];

		cv::Point2i pointB;
		pointB.x = lines[i][2];
		pointB.y = lines[i][3];

		if(pointInRect(pointA, rect) && pointInRect(pointB, rect))
		{
			lines.erase(lines.begin()+i);
			--i;
		}
	}
}

static void removeShort(std::vector<cv::Vec4f>& lines, double lengthThresh)
{
	for(size_t i = 0; i < lines.size(); ++i)
	{
		cv::Vec2f pointA;
		pointA[0] = lines[i][0];
		pointA[1] = lines[i][1];

		cv::Vec2f pointB;
		pointB[0] = lines[i][2];
		pointB[1] = lines[i][3];

		double norm = cv::norm(pointA-pointB);

		if(norm < lengthThresh)
		{
			lines.erase(lines.begin()+i);
			--i;
		}
	}
}

static bool endsOnLineOrtho(const cv::Vec4f lineA, const cv::Vec4f lineB)
{
	return true;
}

static double lineFurthestEndDistance(const cv::Vec4f lineA, const cv::Vec4f lineB)
{
	cv::Point2i pointA;
	pointA.x = lineA[0];
	pointA.y = lineA[1];

	cv::Point2i pointB;
	pointB.x = lineA[2];
	pointB.y = lineA[3];

	cv::Point2i testPointA;
	testPointA.x = lineB[0];
	testPointA.y = lineB[1];

	cv::Point2i testPointB;
	testPointB.x = lineB[2];
	testPointB.y = lineB[3];

	float closestA = std::numeric_limits<float>::max();
	float closestB = std::numeric_limits<float>::max();

	cv::LineIterator masterLineIt(pointA, pointB, 8);

	for(int k = 0; k < masterLineIt.count; ++k, ++masterLineIt)
	{
		float normA = cv::norm(testPointA - masterLineIt.pos());
		float normB = cv::norm(testPointB - masterLineIt.pos());

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
		cv::Point2i iPointA;
		iPointA.x = lines[i][0];
		iPointA.y = lines[i][1];

		cv::Point2i iPointB;
		iPointB.x = lines[i][2];
		iPointB.y = lines[i][3];

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
			if(dprod < 0.90)
				continue;

			bool firstInI;
			bool firstInJ;
			double closestEndDist = closestLineEndpoint(lines[i], lines[j], &firstInI, &firstInJ);

			if(closestEndDist < tollerance)
			{
				cv::Vec4f newline(firstInI ? lines[i][2] : lines[i][0], firstInI ? lines[i][3] : lines[i][1],
				                  firstInJ ? lines[j][2] : lines[j][0], firstInJ ? lines[j][3] : lines[j][1]);

				lines[i] = newline;
				lines.erase(lines.begin()+j);

				--i;
				--j;
				break;
			}
		}
	}
}

std::vector<cv::Vec4f> lineDetect(cv::Mat in)
{
	cv::Mat work;
	cv::Mat vizualization;
	std::vector<cv::Vec4f> lines;

	cv::cvtColor(in, work, cv::COLOR_BGR2GRAY);
	cv::resize(work, work, cv::Size(), 2, 2, cv::INTER_LINEAR);
	work.convertTo(work, CV_8U, 1);
	cv::threshold(work, work, std::numeric_limits<uint8_t>::max()/2, std::numeric_limits<uint8_t>::max(), cv::THRESH_BINARY);
	cv::bitwise_not(work, work);
	cv::ximgproc::thinning(work, work, cv::ximgproc::THINNING_ZHANGSUEN);
	cv::imshow("Viewer", work);
	cv::waitKey(0);

	std::shared_ptr<cv::LineSegmentDetector> detector = cv::createLineSegmentDetector(cv::LSD_REFINE_STD, 1, 0.6, 2.0, 12.5);

	detector->detect(work, lines);

	removeShort(lines, std::max(work.rows/100.0, 4.0));
	{
		work.copyTo(vizualization);
		drawLineSegments(vizualization, lines );
		cv::imshow("Viewer", vizualization);
		cv::waitKey(0);
	}
	Log(Log::WARN)<<"thresh "<<std::max(work.rows/100.0, 5.0);
	deduplicateLines(lines, std::max(work.rows/100.0, 5.0));
	{
		work.copyTo(vizualization);
		drawLineSegments(vizualization, lines );
		cv::imshow("Viewer", vizualization);
		cv::waitKey(0);
	}

	mergeCloseInlineLines(lines, std::max(work.rows/100.0, 5.0));

	//if(Log::level == Log::SUPERDEBUG)
	{
		work.copyTo(vizualization);
		drawLineSegments(vizualization, lines );
		cv::imshow("Viewer", vizualization);
		cv::waitKey(0);
	}

	return lines;
}
