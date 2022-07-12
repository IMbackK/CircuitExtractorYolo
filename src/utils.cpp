#include "utils.h"
#include <opencv2/imgproc.hpp>
#include "log.h"

bool pointInRect(const cv::Point2i& point, const cv::Rect& rect)
{
	return point.x >= rect.x && point.x <= rect.x+rect.width &&
		   point.y >= rect.y && point.y <= rect.y+rect.height;
}

void deduplicatePoints(std::vector<cv::Point2i>& points, double tollerance)
{
	for(size_t i = 0; i < points.size(); ++i)
	{
		const cv::Point2i& point = points[i];
		for(size_t j = i+1; j < points.size(); ++j)
		{
			if(cv::norm(point - points[j]) < tollerance)
			{
				points[j].x = (point.x + points[j].x)/2;
				points[j].y = (point.y + points[j].y)/2;
				points.erase(points.begin()+i);
				--i;
				break;
			}
		}
	}
}

//taken fom opencv line detector src
void drawLineSegments(cv::Mat& in, cv::InputArray lines, cv::Scalar color)
{
	CV_Assert(!in.empty() && (in.channels() == 1 || in.channels() == 3));

	if (in.channels() == 1)
		cv::cvtColor(in, in, cv::COLOR_GRAY2BGR);

	cv::Mat _lines = lines.getMat();
	const int N = _lines.checkVector(4);

	CV_Assert(_lines.depth() == CV_32F || _lines.depth() == CV_32S);

	// Draw segments
	if (_lines.depth() == CV_32F)
	{
		for (int i = 0; i < N; ++i)
		{
			const cv::Vec4f& v = _lines.at<cv::Vec4f>(i);
			const cv::Point2f b(v[0], v[1]);
			const cv::Point2f e(v[2], v[3]);
			cv::line(in, b, e, color, 2);
		}
	}
	else
	{
		for (int i = 0; i < N; ++i)
		{
			const cv::Vec4i& v = _lines.at<cv::Vec4i>(i);
			const cv::Point2i b(v[0], v[1]);
			const cv::Point2i e(v[2], v[3]);
			cv::line(in, b, e, color, 2);
		}
	}
}

double lineDotProd(const cv::Vec4f lineA, const cv::Vec4f lineB)
{
	cv::Vec2f vectorA;
	vectorA[0] = lineA[0] - lineA[2];
	vectorA[1] = lineA[1] - lineA[3];
	vectorA = vectorA/cv::norm(vectorA);

	cv::Vec2f vectorB;
	vectorB[0] = lineB[0] - lineB[2];
	vectorB[1] = lineB[1] - lineB[3];
	vectorB = vectorB/cv::norm(vectorB);

	return std::abs(vectorA.dot(vectorB));
}

std::pair<cv::Point2i, cv::Point2i> furthestPoints(std::vector<cv::Point2i> points)
{
	double maxDist = 0;
	size_t a = 0;
	size_t b = 0;
	for(size_t i = 0; i < points.size(); ++i)
	{
		for(size_t j = i+1; j < points.size(); ++j)
		{
			double norm = cv::norm(points[i] - points[j]);
			if(norm > maxDist)
			{
				maxDist = norm;
				a = i;
				b = j;
			}
		}
	}
	return std::pair<cv::Point2i, cv::Point2i>(points[a], points[b]);
}

double closestLineEndpoint(const cv::Vec4f lineA, const cv::Vec4f lineB)
{
	const cv::Vec2f fistAVec(lineA[0], lineA[1]);
	const cv::Vec2f lastAVec(lineA[2], lineA[3]);

	const cv::Vec2f fistBVec(lineB[0], lineB[1]);
	const cv::Vec2f lastBVec(lineB[2], lineB[3]);

	double distFF = cv::norm(fistAVec-fistBVec);
	double distFL = cv::norm(fistAVec-lastBVec);
	double distLF = cv::norm(lastAVec-fistBVec);
	double distLL = cv::norm(lastAVec-lastBVec);

	double minDist = std::min({distFF, distFL, distLF, distLL});

	return minDist;
}

bool pointIsOnLine(const cv::Point2i& point, const cv::Vec4f& line, double tollerance)
{
	cv::LineIterator lineIt(cv::Point2i(line[0], line[1]), cv::Point2i(line[2], line[3]), 8);

	for(int i = 0; i < lineIt.count; ++i, ++lineIt)
	{
		if(cv::norm(point-lineIt.pos()) < tollerance)
			return true;
	}

	return false;
}