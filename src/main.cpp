#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <poppler-document.h>
#include <poppler-image.h>

#include "log.h"
#include "popplertocv.h"

int main(int argc, char** argv)
{
	Log::level = Log::DEBUG;
	if(argc < 2)
	{
		Log(Log::ERROR)<<"A file name must be provided";
		return 1;
	}

	cv::namedWindow( "Viewer", cv::WINDOW_NORMAL );
	cv::resizeWindow("Viewer", 960, 500);

	poppler::document* document = poppler::document::load_from_file(argv[1]);

	if(document->is_encrypted())
	{
		Log(Log::ERROR)<<"Only unencrypted files are supported";
		return 1;
	}

	std::string keywords = document->get_keywords().to_latin1();
	Log(Log::INFO)<<"Got PDF with "<<document->pages()<<" pages";
	if(!keywords.empty())
		Log(Log::INFO)<<"With keywords: "<<keywords;

	std::vector<cv::Mat> images = getMatsFromDocument(document);

	for(cv::Mat& image : images)
	{
		cv::imshow("Viewer", image);
		cv::waitKey(0);
	}

	delete document;

	return 0;
}
