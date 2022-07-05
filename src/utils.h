#pragma once

bool pointInRect(const cv::Point2i& point, const cv::Rect& rect)
{
	return point.x >= rect.x && point.x <= rect.x+rect.width &&
		   point.y >= rect.y && point.y <= rect.y+rect.height;
}
