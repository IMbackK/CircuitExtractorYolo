#include "utils.h"
#include <opencv2/imgproc.hpp>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <sstream>

#include "log.h"

bool CompString::operator()(const std::string& a, const std::string& b) const
{
	for(size_t i = 0; i < a.size(); ++i)
	{
		if(i > b.size())
			return false;
		if(a[i] < b[i])
			return true;
		else if(a[i] > b[i])
			return false;
	}
	return false;
}

bool isInvalid(const char in)
{
	if(in < 32 || in > 126)
		return true;
	else
		return false;
}

const char* getDirectionString(DirectionHint hint)
{
	switch(hint)
	{
		case C_DIRECTION_HORIZ:
			return "Horizontal";
		case C_DIRECTION_VERT:
			return "Vertical";
		case C_DIRECTION_UNKOWN:
		default:
			return "Unkown";
	}
}

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
	//CV_Assert(!in.empty() && (in.channels() == 1 || in.channels() == 3));
	if(in.empty() || (in.channels() != 1 && in.channels() != 3))
	{
		Log(Log::WARN)<<"Could not draw lines "<<(in.empty() ? "due to empty image" : "due to image channel depth");
		return;
	}

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
			cv::line(in, b, e, color, 1);
		}
	}
	else
	{
		for (int i = 0; i < N; ++i)
		{
			const cv::Vec4i& v = _lines.at<cv::Vec4i>(i);
			const cv::Point2i b(v[0], v[1]);
			const cv::Point2i e(v[2], v[3]);
			cv::line(in, b, e, color, 1);
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

bool rectsIntersect(const cv::Rect& a, const cv::Rect& b)
{
	cv::Rect rect = a & b;
	return rect.width*rect.height > 0;
}

bool rectsFullyOverlap(const cv::Rect& a, const cv::Rect& b)
{
	int area = std::min(a.width*a.height, b.width*b.height);
	cv::Rect rect = a & b;
	return rect.width*rect.height >= area;
}

cv::Rect rectFromPoints(const std::vector<cv::Point>& points)
{
	int left = std::numeric_limits<int>::max();
	int right = std::numeric_limits<int>::min();
	int top = std::numeric_limits<int>::max();
	int bottom = std::numeric_limits<int>::min();

	for(const cv::Point point : points)
	{
		left = point.x < left ? point.x : left;
		right = point.x > right ? point.x : right;

		top = point.y < top ? point.y : top;
		bottom = point.y > bottom ? point.y : bottom;
	}

	return cv::Rect(left, top, right-left, bottom-top);
}

cv::Rect& ofsetRect(cv::Rect& rect, int dx, int dy)
{
	rect.x+=dx;
	rect.y+=dy;
	return rect;
}

cv::Rect padRect(const cv::Rect& rect, double xPadPercent, double yPadPercent, int minimumPad)
{
	cv::Rect out;
	if(xPadPercent > 0)
	{
		out.x = rect.x - std::max(static_cast<int>(rect.width*xPadPercent), minimumPad);
		out.width = std::max(static_cast<int>(rect.width*(1+xPadPercent*2)), rect.width+minimumPad*2);
	}
	else
	{
		out.x = rect.x;
		out.width = rect.width;
	}

	if(yPadPercent > 0)
	{
		out.y = rect.y - std::max(static_cast<int>(rect.height*yPadPercent), minimumPad);
		out.height = std::max(static_cast<int>(rect.height*(1+xPadPercent*2)), rect.height+minimumPad*2);
	}
	else
	{
		out.y = rect.y;
		out.height = rect.height;
	}
	return out;
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

cv::Mat getMatPlane4d(cv::Mat& in, int plane)
{
	assert(in.size.dims() == 4);
	assert(in.isContinuous());
	assert(in.type() == CV_32F);
	const int dims[3] = {in.size[1], in.size[2], in.size[3]};
	cv::Mat crushed(3, dims, in.type());
	for(int z = 0; z < crushed.size[0]; ++z)
	{
		for(int row = 0; row < crushed.size[1]; ++row)
		{
			for(int col = 0; col < crushed.size[2]; ++col)
			{
				int indexIn = plane*in.size[1]*in.size[2]*in.size[3]+z*in.size[2]*in.size[3]+row*in.size[3]+col;
				int indexOut = z*crushed.size[1]*crushed.size[2]+row*crushed.size[2]+col;
				reinterpret_cast<float*>(crushed.data)[indexOut] = reinterpret_cast<float*>(in.data)[indexIn];
			}
		}
	}
	return crushed;
}

double pointDist(const cv::Point2i& pointA, const cv::Point2i& pointB)
{
	cv::Vec2i a(pointA.x, pointA.y);
	cv::Vec2i b(pointB.x, pointB.y);
	return cv::norm(a-b);
}

cv::Mat getMatPlane(cv::Mat& in, int plane)
{
	assert(in.size.dims() == 3);
	assert(in.type() == CV_32F);
	cv::Mat crushed(in.size[1], in.size[2], in.type(), cv::Scalar(0,0));

	for(int row = 0; row < crushed.rows; ++row)
	{
		float *rowPtr = crushed.ptr<float>(row);
		for(int col = 0; col < crushed.cols; ++col)
			rowPtr[col] = in.at<float>(plane, row, col);
	}
	return crushed;
}

std::string getMatType(const cv::Mat& mat)
{
	std::string out;

	unsigned int depth = mat.type() & CV_MAT_DEPTH_MASK;
	unsigned int chans = 1 + (mat.type() >> CV_CN_SHIFT);

	switch(depth)
	{
		case CV_8U:
			out = "8U";
			break;
		case CV_8S:
			out = "8S";
			break;
		case CV_16U:
			out = "16U";
			break;
		case CV_16S:
			out = "16S";
			break;
		case CV_32S:
			out = "32S";
			break;
		case CV_32F:
			out = "32F";
			break;
		case CV_64F:
			out = "64F";
			break;
		default:
			out = "User";
			break;
	}

	out += "C";
	out += std::to_string(chans);

	return out;
}

cv::Mat extendBorder(cv::Mat& in, int borderSize)
{
	cv::Mat out(in.rows+borderSize*2, in.cols+borderSize*2, in.type(), cv::Scalar(255,255,255));

	in.copyTo(out(cv::Rect(borderSize, borderSize, in.cols, in.rows)));

	return out;
}

void printMatInfo(const cv::Mat& mat, const std::string& prefix, const Log::Level lvl)
{
	Log(lvl, false)<<prefix<<' ';
	const cv::MatSize size = mat.size;
	Log(lvl, false)<<"Mat of type "<<getMatType(mat)<<" Dimentions: "<<size.dims()<<' ';
	for(int i = 0; i < size.dims(); ++i)
	{
		Log(lvl, false)<<size[i];
		if(i+1 < size.dims())
			Log(lvl, false)<<'x';
	}

	Log(lvl)<<" Channels: "<<mat.channels();
}

void printMat(const cv::Mat& mat, const Log::Level lvl)
{
	assert(mat.size.dims() < 5 && mat.size.dims() > 1);
	assert(mat.size.dims() != 4 || mat.isContinuous());
	assert(mat.type() == CV_32F);
	Log(lvl)<<"Mat type:"<<getMatType(mat)<<" dim: "<<mat.size.dims()<<" data: ";

	std::cout<<std::fixed;
	if(mat.size.dims() == 4)
	{
		for(int plane = 0; plane < mat.size[0]; ++plane)
		{
			Log(lvl)<<"-- plane "<<plane<<" --";
			for(int z = 0; z < mat.size[1]; ++z)
			{
				Log(lvl)<<"-- Z "<<z<<" --";
				for(int row = 0; row < mat.size[2]; ++row)
				{
					for(int col = 0; col < mat.size[3]; ++col)
					{
						int index = plane*mat.size[1]*mat.size[2]*mat.size[3]+z*mat.size[2]*mat.size[3]+row*mat.size[3]+col;
						Log(lvl, false)<<reinterpret_cast<float*>(mat.data)[index]<<",\t";
					}
					Log(lvl, false)<<'\n';
				}
			}
		}
	}
	else if(mat.size.dims() == 3)
	{
		for(int plane = 0; plane < mat.size[0]; ++plane)
		{
			Log(lvl)<<"-- plane "<<plane<<" --";
			for(int row = 0; row < mat.size[1]; ++row)
			{
				for(int col = 0; col < mat.size[2]; ++col)
				{
					Log(lvl, false)<<mat.at<float>(plane, row, col)<<",\t";
				}
				Log(lvl, false)<<'\n';
			}
		}
	}
	else
	{
		for(int row = 0; row < mat.size[0]; ++row)
		{
			for(int col = 0; col < mat.size[1]; ++col)
			{
				Log(lvl, false)<<mat.at<float>(row, col)<<",\t";
			}
			Log(lvl, false)<<'\n';
		}
	}

	std::cout<<std::defaultfloat;
}

std::pair<double, double> getRectXYPaddingPercents(DirectionHint hint, double tolleranceFactor)
{
	double padX;
	double padY;
	switch(hint)
	{
		case C_DIRECTION_HORIZ:
			padX = 0.15*tolleranceFactor;
			padY = 0;
			break;
		case C_DIRECTION_VERT:
			padX = 0;
			padY = 0.15*tolleranceFactor;
			break;
		case C_DIRECTION_UNKOWN:
		default:
			padX = 0.15*tolleranceFactor;
			padY = 0.075*tolleranceFactor;
			break;
	}
	return std::pair<double, double>(padX, padY);
}

std::vector<std::string> loadFileLines(const std::filesystem::path& path)
{
	std::vector<std::string> lines;

	std::fstream file(path, std::fstream::in);
	if(!file.is_open())
	{
		Log(Log::WARN)<<"Could not open file at "<<path;
		return lines;
	}

	while(file.good())
	{
		std::string str;
		std::getline(file, str);
		lines.push_back(str);
	}
	return lines;
}

std::string yoloLabelsFromRect(const cv::Rect& rect, const cv::Mat& image, int label)
{
	std::stringstream ss;
	ss<<label;
	ss<<' '<<(static_cast<double>(rect.x+rect.width/2)/image.cols)<<' '<<(static_cast<double>(rect.y+rect.height/2)/image.rows);
	ss<<' '<<(static_cast<double>(rect.width)/image.cols)<<' '<<(static_cast<double>(rect.height)/image.rows)<<'\n';
	return ss.str();
}
