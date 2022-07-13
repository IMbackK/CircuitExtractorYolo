#include "document.h"

#include <poppler-document.h>
#include <poppler-image.h>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "popplertocv.h"
#include "linedetection.h"

std::vector<cv::Mat> getCircutImages(std::vector<cv::Mat> images, Yolo5* yolo, std::vector<float>* probs)
{
	std::vector<cv::Mat> circuts;

	for(cv::Mat& image : images)
	{
		std::vector<Yolo5::DetectedClass> detections = yolo->detect(image);
		cv::Mat visulization;

		if(Log::level == Log::SUPERDEBUG)
			image.copyTo(visulization);

		for(const Yolo5::DetectedClass& detection : detections)
		{
			try
			{
				circuts.push_back(cv::Mat(image, detection.rect));
				if(probs)
					probs->push_back(detection.prob);
			}
			catch(const cv::Exception& ex)
			{
				Log(Log::WARN)<<"Failed to process rect "<<ex.what();
			}

			if(Log::level == Log::SUPERDEBUG)
				Yolo5::drawDetection(visulization, detection);
		}

		if(detections.size() > 0 && Log::level == Log::SUPERDEBUG)
		{
			cv::imshow("Viewer", visulization);
			cv::waitKey(0);
		}
	}

	return circuts;
}

void Document::print(Log::Level level) const
{
	Log(level)<<"Document: "<<basename<<" \""<<title<<"\" by \""<<author<<'\"';
	if(keywords.size() > 0)
	Log(level)<<"keywords: "<<keywords;
}

bool Document::saveCircutImages(const std::filesystem::path folder) const
{
	if(!std::filesystem::is_directory(folder))
	{
		if(!std::filesystem::create_directory(folder))
		{
			Log(Log::ERROR)<<folder<<" is not a valid directory and no directory could be created at this location";
			return false;
		}
	}
	for(size_t i = 0; i < circuts.size(); ++i)
	{
		const Circut& circut = circuts[i];
		if(circut.elements.empty())
			continue;
		std::filesystem::path path = folder /
			std::filesystem::path(basename + "_" +
			std::to_string(i) + "P" +
			std::to_string(circut.prob) + ".png");
		try
		{
			cv::imwrite(path, circut.ciructImage());
		}
		catch(const cv::Exception& ex)
		{
			Log(Log::ERROR)<<"Cant write "<<path<<' '<<ex.what();
			return false;
		}
		Log(Log::INFO)<<"Wrote image to "<<path;
	}
	return true;
}

bool Document::process(Yolo5* circutYolo, Yolo5* elementYolo)
{
	std::vector<float> probs;
	if(pages.empty())
		return false;
	std::vector<cv::Mat> circutImages = getCircutImages(pages, circutYolo, &probs);

	for(size_t i = 0; i < circutImages.size(); ++i)
	{
		Circut circut;
		circut.image = circutImages[i];
		circut.prob = probs[i];
		cv::resize(circut.image, circut.image, cv::Size(640,640), 0, 0, cv::INTER_LINEAR);
		circut.getElements(elementYolo);
		circuts.push_back(circut);
	}

	return true;
}

std::shared_ptr<Document> Document::load(const std::string& fileName)
{
	poppler::document* popdocument = poppler::document::load_from_file(fileName);

	if(!popdocument)
	{
		Log(Log::ERROR)<<"Could not load pdf file from "<<fileName;
		return std::shared_ptr<Document>();
	}

	if(popdocument->is_encrypted())
	{
		Log(Log::ERROR)<<"Only unencrypted files are supported";
		return std::shared_ptr<Document>();
	}

	std::shared_ptr<Document> document = std::make_shared<Document>();
	document->keywords = popdocument->get_keywords().to_latin1();
	document->title = popdocument->get_title().to_latin1();
	document->author = popdocument->get_creator().to_latin1();
	document->basename = std::filesystem::path(fileName).filename();
	document->print(Log::EXTRA);
	document->pages = getMatsFromDocument(popdocument, cv::Size(1280, 1280));

	delete popdocument;

	return document;
}

void Document::removeEmptyCircuts()
{
	for(size_t i = 0; i < circuts.size(); ++i)
	{
		if(circuts[i].elements.size() < 2)
		{
			circuts.erase(circuts.begin()+i);
			--i;
		}
	}
}
