#include "linedetection.h"

#include <limits>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <cmath>
#include <algorithm>

#include "utils.h"

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

		if(norm < lengthThresh || norm < 4)
		{
			lines.erase(lines.begin()+i);
			--i;
		}
	}
}

static void deduplicateLines(std::vector<cv::Vec4f>& lines, double distanceThresh)
{
	for(size_t i = 0; i < lines.size(); ++i)
	{
		cv::Vec2f pointA;
		pointA[0] = lines[i][0];
		pointA[1] = lines[i][1];

		cv::Vec2f pointB;
		pointB[0] = lines[i][2];
		pointB[1] = lines[i][3];

		for(size_t j = 0; j < lines.size(); ++j)
		{
			if(j == i)
				continue;

			double distSum = 0;

			cv::Vec2f testPointA;
			testPointA[0] = lines[j][0];
			testPointA[1] = lines[j][1];

			cv::Vec2f testPointB;
			testPointB[0] = lines[j][2];
			testPointB[1] = lines[j][3];

			float normAA = cv::norm(pointA-testPointA);
			float normAB = cv::norm(pointA-testPointB);

			distSum += std::min(normAA, normAB);
			distSum += std::min(cv::norm(pointB-testPointA), cv::norm(pointB-testPointB));
			if(distSum/2 < distanceThresh)
			{
				if(normAA < normAB)
				{
					pointA = (pointA + testPointA)/2;
					pointB = (pointB + testPointB)/2;
				}
				else
				{
					pointA = (pointA + testPointB)/2;
					pointB = (pointB + testPointA)/2;
				}
				lines[i][0] = pointA[0];
				lines[i][1] = pointA[1];
				lines[i][2] = pointB[0];
 				lines[i][3] = pointB[1];
				lines.erase(lines.begin()+j);
				--j;
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
	cv::threshold(work, work, 2*std::numeric_limits<uint8_t>::max()/3, std::numeric_limits<uint8_t>::max(), cv::THRESH_BINARY);
	cv::imshow("Viewer", work*std::numeric_limits<uint8_t>::max());
	cv::waitKey(0);
	work.convertTo(work, CV_8U, 1);

	std::shared_ptr<cv::LineSegmentDetector> detector = cv::createLineSegmentDetector(cv::LSD_REFINE_NONE, 1, 0.8, 2.0, 12.5);

	detector->detect(work, lines);

	removeShort(lines, in.rows/50.0);
	deduplicateLines(lines, in.rows/40.0);

	//if(Log::level == Log::SUPERDEBUG)
	{
		in.copyTo(vizualization);
		detector->drawSegments(vizualization, lines );
		cv::imshow("Viewer", vizualization);
		cv::waitKey(0);
	}

	return lines;
}
