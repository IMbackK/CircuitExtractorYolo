#include "popplertocv.h"
#include <opencv2/imgproc.hpp>
#include <poppler-document.h>
#include <poppler-page.h>
#include <poppler-page-renderer.h>

#include "log.h"

int popplerEnumToCvFormat(int format)
{
	int cvFormat;
	switch(format)
	{
		case poppler::image::format_mono:
			cvFormat = CV_8UC1;
			break;
		case poppler::image::format_rgb24:
			cvFormat = CV_8UC3;
			break;
		case poppler::image::format_argb32:
			cvFormat = CV_8UC4;
			break;
		case poppler::image::format_gray8:
			cvFormat = CV_8UC1;
			break;
		case poppler::image::format_bgr24:
			cvFormat = CV_8UC3;
			break;
		case poppler::image::format_invalid:
		default:
			cvFormat = -1;
	}
	return cvFormat;
}

std::vector<cv::Mat> getMatsFromDocument(poppler::document* document)
{
	poppler::page_renderer renderer;

	int pagesCount = document->pages();

	std::vector<cv::Mat> output;

	for(int i = 0; i < pagesCount; ++i)
	{
		poppler::page* page = document->create_page(i);
		poppler::image image = renderer.render_page(page, 512, 512);
		Log(Log::INFO)<<image.height()<<" format "<<image.format();

		cv::Mat cvBuffer(image.height(), image.width(), popplerEnumToCvFormat(image.format()), image.data(), image.bytes_per_row());
		if(image.format() == poppler::image::format_rgb24 || image.format() == poppler::image::format_argb32)
			cvtColor(cvBuffer, cvBuffer, cv::COLOR_RGB2BGR);
		output.push_back(cvBuffer);
	}
	return output;
}
