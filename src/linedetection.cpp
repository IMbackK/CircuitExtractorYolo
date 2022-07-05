#include "linedetection.h"

#include <limits>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <cmath>
#include <algorithm>

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
		cv::Point2i pointA;
		pointA.x = lines[i][0];
		pointA.y = lines[i][1];

		cv::Point2i pointB;
		pointB.x = lines[i][2];
		pointB.y = lines[i][3];

		for(size_t j = 0; j < lines.size(); ++j)
		{
			if(j == i)
				continue;

			cv::Point2i testPointA;
			testPointA.x = lines[j][0];
			testPointA.y = lines[j][1];

			cv::Point2i testPointB;
			testPointB.x = lines[j][2];
			testPointB.y = lines[j][3];

			cv::LineIterator masterLineIt(pointA, pointB, 8);

			float closestA = std::numeric_limits<float>::max();
			float closestB = std::numeric_limits<float>::max();
			int closestAIndex = 0;
			int closestBIndex = 0;

			for(int k = 0; k < masterLineIt.count; ++k, ++masterLineIt)
			{
				float normA = cv::norm(testPointA - masterLineIt.pos());
				float normB = cv::norm(testPointB - masterLineIt.pos());

				if(normA < distanceThresh)
					Log(Log::WARN)<<normA;

				if(normA < closestA)
				{
					closestA = normA;
					closestAIndex = k;
				}
				if(normB < closestB)
				{
					closestB = normB;
					closestBIndex = k;
				}
			}

			if((closestA+closestB)/2 < distanceThresh || (closestA+closestB)/2 < 3)
			{
				Log(Log::WARN)<<"removing "<<i;
				if(closestAIndex == 0)
					pointA = (pointA + testPointA)/2;
				if(closestBIndex == 0)
					pointA = (pointA + testPointB)/2;
				if(closestAIndex > masterLineIt.count-2)
					pointB = (pointB + testPointA)/2;
				if(closestBIndex > masterLineIt.count-2)
					pointB = (pointB + testPointB)/2;

				lines[i][0] = pointA.x;
				lines[i][1] = pointA.y;
				lines[i][2] = pointB.x;
				lines[i][3] = pointB.y;

				lines.erase(lines.begin()+j);
				--j;
				--i;
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
	cv::threshold(work, work, 2*std::numeric_limits<uint8_t>::max()/3, std::numeric_limits<uint8_t>::max(), cv::THRESH_BINARY);
	cv::imshow("Viewer", work*std::numeric_limits<uint8_t>::max());
	cv::waitKey(0);
	work.convertTo(work, CV_8U, 1);

	std::shared_ptr<cv::LineSegmentDetector> detector = cv::createLineSegmentDetector(cv::LSD_REFINE_NONE, 1, 0.6, 2.0, 12.5);

	detector->detect(work, lines);

	removeShort(lines, in.rows/100.0);
		{
		in.copyTo(vizualization);
		detector->drawSegments(vizualization, lines );
		cv::imshow("Viewer", vizualization);
		cv::waitKey(0);
	}
	deduplicateLines(lines, in.rows/40.0);
	{
		in.copyTo(vizualization);
		detector->drawSegments(vizualization, lines );
		cv::imshow("Viewer", vizualization);
		cv::waitKey(0);
	}

	//if(Log::level == Log::SUPERDEBUG)
	{
		in.copyTo(vizualization);
		detector->drawSegments(vizualization, lines );
		cv::imshow("Viewer", vizualization);
		cv::waitKey(0);
	}

	return lines;
}
