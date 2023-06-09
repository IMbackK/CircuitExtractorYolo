#include "document.h"

#include <fstream>

#include <poppler-document.h>
#include <poppler-page.h>
#include <poppler-image.h>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <map>

#include "popplertocv.h"
#include "linedetection.h"
#include "tokenize.h"

std::vector<cv::Mat> getYoloImages(std::vector<cv::Mat> images, Yolo5* yolo,
								   std::vector<float>* probs, std::vector<cv::Rect>* rects,
								   std::vector<size_t>* imageNums)
{
	std::vector<cv::Mat> circuts;

	for(size_t i = 0; i < images.size(); ++i)
	{
		cv::Mat& image = images[i];
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
				if(rects)
					rects->push_back(detection.rect);
				if(imageNums)
					imageNums->push_back(i);
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
	Log(level)<<"Document: "<<basename<<" \""<<metadata.title<<"\" by \""<<metadata.author<<'\"';
	if(metadata.keywords.size() > 0)
		Log(level)<<"keywords: "<<metadata.keywords;
}

bool Document::saveElementLabels(const std::filesystem::path& folder) const
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
		if(circut.getElements().empty())
			continue;
		std::filesystem::path path = folder /
			std::filesystem::path(basename + "_" +
			std::to_string(i) + "P" +
			std::to_string(circut.prob) + ".png");
		try
		{
			cv::imwrite(path, circut.plainCircutImage());

			std::fstream file;
			std::filesystem::path labelPath = path;
			labelPath.replace_extension(".txt");
			file.open(labelPath, std::ios_base::out);
			if(!file.is_open())
			{
				Log(Log::ERROR)<<"Could not open file "<<labelPath<<" for writeing";
				return false;
			}
			file<<circut.getYoloElementLabels();
			file.close();
			Log(Log::INFO)<<"Wrote labels to "<<labelPath;
		}
		catch(const cv::Exception& ex)
		{
			Log(Log::ERROR)<<"Cant write "<<path<<' '<<ex.what();
			return false;
		}
	}
	return true;
}

bool Document::saveCircutLabels(const std::filesystem::path& folder) const
{
	if(!std::filesystem::is_directory(folder))
	{
		if(!std::filesystem::create_directory(folder))
		{
			Log(Log::ERROR)<<folder<<" is not a valid directory and no directory could be created at this location";
			return false;
		}
	}

	for(size_t i = 0; i < pages.size(); ++i)
	{
		if(circutsOnPage(i) > 0)
		{
			std::filesystem::path path = folder /
				std::filesystem::path(basename + "_" +
				std::to_string(i) + ".png");

			try
			{
				cv::imwrite(path, pages[i]);

				std::fstream file;
				std::filesystem::path labelPath = path;
				labelPath.replace_extension(".txt");
				file.open(labelPath, std::ios_base::out);
				if(!file.is_open())
				{
					Log(Log::ERROR)<<"Could not open file "<<labelPath<<" for writeing";
					return false;
				}
				file<<getYoloCircutLabels(i);
				file.close();
				Log(Log::INFO)<<"Wrote Circut labels to "<<labelPath;
			}
			catch(const cv::Exception& ex)
			{
				Log(Log::ERROR)<<"Cant write "<<path<<' '<<ex.what();
				return false;
			}
		}
	}
	return true;
}

bool Document::saveCircutImages(const std::filesystem::path& folder) const
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
		if(circut.getElements().empty())
			continue;
		std::filesystem::path path = folder /
			std::filesystem::path(basename + "_" +
			std::to_string(i) + "P" +
			std::to_string(circut.prob) + ".png");
		try
		{
			cv::imwrite(path, circut.ciructImage());
			Log(Log::INFO)<<"Wrote image to "<<path;
		}
		catch(const cv::Exception& ex)
		{
			Log(Log::ERROR)<<"Cant write "<<path<<' '<<ex.what();
			return false;
		}
	}
	return true;
}

bool Document::saveDatafile(const std::filesystem::path& folder)
{
	if(!std::filesystem::is_directory(folder))
	{
		if(!std::filesystem::create_directory(folder))
		{
			Log(Log::ERROR)<<folder<<" is not a valid directory and no directory could be created at this location";
			return false;
		}
	}
	std::fstream file;
	std::filesystem::path path = folder/std::filesystem::path(basename + ".txt");
	file.open(path, std::ios_base::out);
	if(!file.is_open())
	{
		Log(Log::ERROR)<<"Could not open file "<<path<<" for writeing";
		return false;
	}

	file<<"title = "<<metadata.title<<'\n';
	file<<"author = "<<metadata.author<<'\n';
	file<<"keywords = "<<metadata.keywords<<'\n';
	file<<"circuts = "<<circuts.size()<<'\n';

	for(size_t i = 0; i < circuts.size(); ++i)
	{
		file<<"circut "<<i<<'\n';
		file<<circuts[i].getSummary()<<'\n';
	}
	return true;
}

bool Document::process(Yolo5* circutYolo, Yolo5* elementYolo, Yolo5* graphYolo)
{
	std::vector<float> probs;
	std::vector<cv::Rect> rects;
	std::vector<size_t> pageNums;
	if(pages.empty())
		return false;
	std::vector<cv::Mat> circutImages = getYoloImages(pages, circutYolo, &probs, &rects, &pageNums);

	for(size_t i = 0; i < circutImages.size(); ++i)
	{
		Circut circut(extendBorder(circutImages[i], 10), probs[i], rects[i], pageNums[i]);
		circut.detectElements(elementYolo);
		circut.detectNets();
		DirectionHint hint = circut.estimateDirection();
		circut.setDirectionHint(hint);
		circut.parseCircut();
		std::string model = circut.getString();
		if(model.size() > 2)
			circuts.push_back(circut);
	}

	probs.clear();
	rects.clear();

	if(graphYolo)
	{
		std::vector<cv::Mat> graphImages = getYoloImages(pages, graphYolo, &probs, &rects);
		for(size_t i = 0; i < graphImages.size(); ++i)
		{
			Graph graph(extendBorder(graphImages[i], 10), probs[i], rects[i]);
			graph.getPoints();
			graphs.push_back(graph);
		}
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
	document->metadata.keywords = popdocument->get_keywords().to_latin1();
	document->metadata.title = popdocument->get_title().to_latin1();
	document->metadata.author = popdocument->get_creator().to_latin1();
	document->basename = std::filesystem::path(fileName).filename();
	document->print(Log::EXTRA);
	document->pages = getMatsFromDocument(popdocument, cv::Size(1280, 1280));

	for(size_t i = 0; i < document->pages.size(); ++i)
		document->text.push_back(popdocument->create_page(i)->text().to_latin1());

	delete popdocument;

	return document;
}

void Document::dropImages()
{
	pages.clear();
	for(Circut& circut : circuts)
		circut.dropImage();
	for(Graph& graph : graphs)
		graph.dropImage();
}

void Document::removeEmptyCircuts()
{
	for(size_t i = 0; i < circuts.size(); ++i)
	{
		if(circuts[i].getElements().size() < 2)
		{
			circuts.erase(circuts.begin()+i);
			--i;
		}
	}
}

std::string Document::getField() const
{
	return field;
}

std::vector<std::string> Document::getText()
{
	return text;
}

std::vector<size_t> Document::getWordOccurances(const std::vector<std::string>& words)
{
	//In the beginning where words and the words made the world.
	std::map<std::string, size_t, CompString> wordMap;
	//I am the words, the words are everything.
	std::vector<std::string> textPages = getText();
	std::string allText;
	for(const std::string& pageText : textPages)
	{
		allText.append(pageText);
		allText.push_back('\n');
	}
	std::vector<std::string> tokens = tokenize(allText, " ,.?!(){}[]\n\t\"'~-|+=@");


	for(const std::string& word : words)
		wordMap.insert({word, 0});

	//Where the words end, the world ends.
	//You cannot move forward in an absence of space
	for(std::string &token : tokens)
	{
		token.resize(std::remove_if(token.begin(), token.end(), &isInvalid) - token.begin());
		for(char& ch : token)
		{
			if(ch >= 65 && ch <= 90)
				ch = ch + 32;
		}

		if(wordMap.find(token) == wordMap.end())
			continue;
		++wordMap.at(token);
		//Repeat.
	}

	std::vector<size_t> occurances;
	occurances.reserve(words.size());
	for(size_t i = 0; i < words.size(); ++i)
		occurances.push_back(wordMap.at(words[i]));

	return occurances;
}

const Document::Metadata Document::getMetadata() const
{
	return metadata;
}

std::string Document::getBasename() const
{
	return basename;
}

size_t Document::circutsOnPage(size_t page) const
{
	size_t ret = 0;
	for(const Circut& circut : circuts)
	{
		if(circut.getPagenum() == page)
			++ret;
	}
	return ret;
}

std::string Document::getYoloCircutLabels(size_t page) const
{
	std::stringstream ss;
	for(const Circut& circut : circuts)
	{
		if(circut.getPagenum() == page)
			ss<<yoloLabelsFromRect(circut.getRect(), pages[page], 0);
	}
	return ss.str();
}
