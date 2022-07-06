#pragma once

bool pointInRect(const cv::Point2i& point, const cv::Rect& rect)
{
	return point.x >= rect.x && point.x <= rect.x+rect.width &&
		   point.y >= rect.y && point.y <= rect.y+rect.height;
}

//taken fom opencv line detector src
void drawLineSegments(cv::Mat& in, cv::InputArray lines)
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
			cv::line(in, b, e, cv::Scalar(0, 0, 255), 5);
		}
	}
	else
	{
		for (int i = 0; i < N; ++i)
		{
			const cv::Vec4i& v = _lines.at<cv::Vec4i>(i);
			const cv::Point2i b(v[0], v[1]);
			const cv::Point2i e(v[2], v[3]);
			cv::line(in, b, e, cv::Scalar(0, 0, 255), 5);
		}
	}
}
