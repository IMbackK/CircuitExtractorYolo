#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <poppler-document.h>
#include <poppler-image.h>
#include <poppler-global.h>
#include <filesystem>
#include <vector>
#include <future>
#include <memory>

#include "log.h"
#include "popplertocv.h"
#include "yolo.h"
#include "document.h"
#include "circut.h"
#include "randomgen.h"

static void printUsage(int argc, char** argv)
{
	Log(Log::INFO)<<"Usage: "<<argv[0]<<"[CIRCUTNETWORKFILENAME] [ELEMENTNETWOKRFILENAME] [GRAPHNETWORFILENAME] [PDFFILENAMES]";
}

/*
static void cleanDocuments(std::vector<std::shared_ptr<Document>> documents)
{
	for(size_t i = 0; i < documents.size(); ++i)
	{
		documents[i]->removeEmptyCircuts();
		if(documents[i]->circuts.empty())
		{
			documents.erase(documents.begin()+i);
			--i;
		}
	}
}
*/

static bool save(std::shared_ptr<Document> document)
{
	bool retImage = document->saveCircutImages("./circuts");
	bool retDatafile = document->saveDatafile("./documents");
	bool ret = retImage && retDatafile;
	if(!ret)
		Log(Log::WARN)<<"Error saving files for "<<document->basename;
	return ret;
}

static bool process(std::shared_ptr<Document> document, Yolo5* circutYolo, Yolo5* elementYolo, Yolo5* graphYolo)
{
	document->process(circutYolo, elementYolo, graphYolo);
	if(!document->circuts.empty())
		std::thread(save, document).detach();
	return true;
}

static std::string getNextString(int argc, char** argv, int& argvCounter)
{
	while(argvCounter < argc)
	{
		std::string str(argv[argvCounter]);
		++argvCounter;
		if(str != "-v")
			return str;
		else
			Log::level = Log::level == Log::DEBUG ? Log::SUPERDEBUG : Log::DEBUG;
	}
	return "";
}

static void dropMessage(const std::string& message, void* userdata)
{
	(void)message;
	(void)userdata;
}

int main(int argc, char** argv)
{
	rd::init();
	Log::level = Log::INFO;
	if(argc < 5)
	{
		printUsage(argc, argv);
		return 1;
	}

	poppler::set_debug_error_function(dropMessage, nullptr);

	Yolo5* circutYolo;
	Yolo5* elementYolo;
	Yolo5* graphYolo;

	int argvCounter = 1;
	std::string circutNetworkFileName = getNextString(argc, argv, argvCounter);
	if(circutNetworkFileName.empty())
	{
		printUsage(argc, argv);
		return 1;
	}
	try
	{
		circutYolo = new Yolo5(circutNetworkFileName, 1);
		Log(Log::INFO)<<"Red circut network from "<<circutNetworkFileName;
	}
	catch(const cv::Exception& ex)
	{
		Log(Log::ERROR)<<"Can not read network from "<<circutNetworkFileName;
		return 1;
	}

	std::string elementNetworkFileName = getNextString(argc, argv, argvCounter);
	if(elementNetworkFileName.empty())
	{
		printUsage(argc, argv);
		return 1;
	}
	try
	{
		elementYolo = new Yolo5(elementNetworkFileName, 7);
		Log(Log::INFO)<<"Red element network from "<<elementNetworkFileName;
	}
	catch(const cv::Exception& ex)
	{
		Log(Log::ERROR)<<"Can not read network from "<<elementNetworkFileName;
		return 1;
	}

	std::string graphNetworkFileName = getNextString(argc, argv, argvCounter);
	if(graphNetworkFileName.empty())
	{
		printUsage(argc, argv);
		return 1;
	}
	try
	{
		graphYolo = new Yolo5(graphNetworkFileName, 1);
		Log(Log::INFO)<<"Red element network from "<<graphNetworkFileName;
	}
	catch(const cv::Exception& ex)
	{
		Log(Log::ERROR)<<"Can not read network from "<<graphNetworkFileName;
		return 1;
	}

	std::vector<std::string> fileNames;
	fileNames.reserve(argc - argvCounter);
	for(int i = argvCounter; i < argc; ++i)
	{
		if(std::string(argv[i]) == "-v")
			Log::level = Log::level == Log::DEBUG ? Log::SUPERDEBUG : Log::DEBUG;
		fileNames.push_back(argv[i]);
	}

	if(Log::level == Log::SUPERDEBUG)
	{
		cv::namedWindow( "Viewer", cv::WINDOW_NORMAL );
		cv::resizeWindow("Viewer", 960, 500);
	}

	std::vector<std::shared_future<std::shared_ptr<Document>>> futures;
	futures.reserve(8);

	for(size_t i = 0; i < fileNames.size();)
	{
		while(i < fileNames.size() && futures.size() < 8)
		{
			futures.push_back(std::async(std::launch::async, Document::load, fileNames[i]));
			Log(Log::INFO)<<"Loading document "<<i<<" of "<<fileNames.size();
			++i;
		}

		for(size_t j = 0; j < futures.size(); ++j)
		{
			if(futures[j].wait_for(std::chrono::microseconds(0)) == std::future_status::ready)
			{
				std::shared_ptr<Document> document = futures[j].get();
				if(document)
				{
					process(document, circutYolo, elementYolo, graphYolo);
					Log(Log::INFO)<<"Finished document. documents in qeue: "<<futures.size();
				}
				else
				{
					Log(Log::WARN)<<"Failed to load document. documents in qeue: "<<futures.size();
				}
				futures.erase(futures.begin()+j);
				break;
			}
		}
	}

	for(size_t j = 0; j < futures.size(); ++j)
	{
		std::shared_ptr<Document> document = futures[j].get();
		if(document)
			process(document, circutYolo, elementYolo, graphYolo);
		Log(Log::INFO)<<"Finished document";
	}

	delete circutYolo;
	delete elementYolo;
	delete graphYolo;

	return 0;
}
