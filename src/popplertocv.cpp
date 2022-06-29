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

std::vector<cv::Mat> getMatsFromDocument(poppler::document* document, const cv::Size& size)
{
	poppler::page_renderer renderer;
	renderer.set_render_hint(poppler::page_renderer::antialiasing, true);

	int pagesCount = document->pages();

	std::vector<cv::Mat> output;

	for(int i = 0; i < pagesCount; ++i)
	{
		poppler::page* page = document->create_page(i);
		poppler::rectf pagesize = page->page_rect();
		Log(Log::DEBUG)<<"Pagesize: "<<pagesize.width()<<'x'<<pagesize.height();
		poppler::image image = renderer.render_page(page, size.width/4, size.height/4);
		cv::Mat cvBuffer(image.height(), image.width(), popplerEnumToCvFormat(image.format()), image.data(), image.bytes_per_row());
		if(image.format() == poppler::image::format_rgb24 || image.format() == poppler::image::format_argb32)
			cvtColor(cvBuffer, cvBuffer, cv::COLOR_RGB2BGR);
		cv::resize(cvBuffer, cvBuffer, size, 0, 0, cv::INTER_LINEAR);
		output.push_back(cvBuffer);
	}
	return output;
}
