#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <poppler-document.h>
#include <poppler-image.h>
#include <poppler-global.h>
#include <filesystem>
#include <fstream>
#include <vector>
#include <future>
#include <memory>
#include <set>

#include "log.h"
#include "popplertocv.h"
#include "yolo.h"
#include "document.h"
#include "circut.h"
#include "randomgen.h"
#include "options.h"
#include "resources.h"

#define THREADS 16

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

static bool save(std::shared_ptr<Document> document, const Config config)
{
	int ret = 0;
	if(!document->circuts.empty())
	{
		if(config.outputCircut)
			ret += document->saveCircutImages(config.outDir/"circuts");
		if(config.outputSummaries)
			ret += document->saveDatafile(config.outDir/"summaries");
		if(ret != config.outputCircut + config.outputSummaries)
			Log(Log::WARN)<<"Error saveing files for "<<document->getBasename();
	}

	document->dropImages();
	return ret != 2;
}

static bool process(std::shared_ptr<Document> document,
					Yolo5* circutYolo, Yolo5* elementYolo, Yolo5* graphYolo,
					const Config& config)
{
	document->process(circutYolo, elementYolo, graphYolo);
	std::thread(save, document, config).detach();
	return true;
}

static void dropMessage(const std::string& message, void* userdata)
{
	(void)message;
	(void)userdata;
}

std::vector<std::filesystem::path> toFilePaths(const std::vector<std::filesystem::path>& paths)
{
	std::vector<std::filesystem::path> filePaths;

	for(const std::filesystem::path& pathC : paths)
	{
		std::filesystem::path path = pathC;
		if(std::filesystem::is_symlink(path))
			path = std::filesystem::read_symlink(path);
		if(std::filesystem::is_regular_file(path))
		{
			filePaths.push_back(path);
		}
		else if(std::filesystem::is_directory(path))
		{
			for(const std::filesystem::directory_entry& dirent : std::filesystem::directory_iterator(path))
			{
				std::filesystem::path filePath = dirent.path();
				if(std::filesystem::is_symlink(filePath))
					filePath = std::filesystem::read_symlink(filePath);
				if(std::filesystem::is_regular_file(filePath))
					filePaths.push_back(filePath);
			}
		}
	}

	return filePaths;
}

static bool checkParams(Config& config)
{
	if(config.circutNetworkFileName.empty())
		Log(Log::INFO)<<"Internal circut network will be used";
	if(config.elementNetworkFileName.empty())
		Log(Log::INFO)<<"Internal element network will be used";
	if(config.graphNetworkFileName.empty())
		Log(Log::WARN)<<"a graph network file name is not provided, wont be able to extract graphs";

	if(config.outDir.empty())
	{
		Log(Log::INFO)<<"output directory not specified, using ./out";
		config.outDir.assign("./out");
	}

	if(!std::filesystem::exists(config.outDir) || !std::filesystem::is_directory(config.outDir))
	{
		if(!std::filesystem::create_directory(config.outDir))
		{
			Log(Log::ERROR)<<config.outDir<<" is not a directory and a directory can not be created at this location";
			return false;
		}
	}

	if(config.paths.empty())
	{
		Log(Log::ERROR)<<"path(s) to pdf a file(s) or a directory with pdf files must be provided";
		return false;
	}

	if(config.baysenFileName.empty() && !config.wordFileName.empty())
	{
		Log(Log::ERROR)<<"For document classification both a parameter file must be provided";
		return false;
	}
	if(config.baysenFileName.empty())
		Log(Log::WARN)<<"Document classification disabled";


	if(!config.baysenFileName.empty() && config.wordFileName.empty())
		Log(Log::INFO)<<"Internal word dictionary will be used";

	return true;
}

size_t removeLessThanN(std::map<std::string, size_t, CompString>& map, size_t n)
{
	size_t otherCount = 0;
	for(const std::pair<std::string, size_t> circut : map)
	{
		if(circut.second < n)
			otherCount += circut.second;
	}
	std::erase_if(map, [n](const std::pair<std::string, size_t>& circut)->bool{return circut.second < n;});

	return otherCount;
}

bool outputStatistics(const std::vector<std::shared_ptr<Document>>& documents, const Config& config)
{
	std::map<std::string, size_t, CompString> allCircutMap;
	std::map<std::string, std::map<std::string, size_t, CompString>, CompString> fields;

	for(const std::shared_ptr<Document>& document : documents)
		fields.insert({document->getField(), std::map<std::string, size_t, CompString>()});

	Log(Log::INFO)<<"Found "<<fields.size()<<" fields:";
	for(const std::pair<std::string, std::map<std::string, size_t, CompString>> field : fields)
		Log(Log::INFO)<<field.first;

	Log(Log::INFO)<<"Calculating statistics";
	for(const std::shared_ptr<Document>& document : documents)
	{
		for(Circut& circut : document->circuts)
		{
			std::string circutStr = circut.getString();

			if(circutStr.find("s") != std::string::npos || circutStr.find("x") != std::string::npos)
				continue;

			std::map<std::string, size_t, CompString>& fieldMap = fields.at(document->getField());

			auto fieldMapIterator = fieldMap;
			if(fieldMap.find(circutStr) == fieldMap.end())
				fieldMap.insert({circutStr, 1});
			else
				++fieldMap.at(circutStr);

			auto iterator = allCircutMap.find(circutStr);
			if(iterator == allCircutMap.end())
				allCircutMap.insert({circutStr, 1});
			else
				++allCircutMap.at(circutStr);
		}
	}

	Log(Log::INFO)<<"Saveing statistics to "<<config.outDir/"statistics.txt";
	std::fstream file;
	file.open(config.outDir/"statistics.txt", std::ios_base::out);
	if(!file.is_open())
	{
		Log(Log::ERROR)<<"Could not open "<<config.outDir/"statistics.txt"<<" for writeing";
		return false;
	}
	file<<"All circuts:\n";
	for(const std::pair<std::string, size_t> circut : allCircutMap)
		file<<circut.second<<",\t"<<circut.first<<'\n';
	file<<'\n';

	for(const std::pair<std::string, std::map<std::string, size_t, CompString>> field : fields)
	{
		file<<field.first<<":\n";
		for(const std::pair<std::string, size_t> circut : field.second)
			file<<circut.second<<", \t"<<circut.first<<'\n';
	}
	file<<'\n';
	file.close();

	return true;
}

int main(int argc, char** argv)
{
	rd::init();
	Log::level = Log::INFO;

	Config config;
	argp_parse(&argp, argc, argv, 0, 0, &config);

	if(!checkParams(config))
		return 1;

	poppler::set_debug_error_function(dropMessage, nullptr);

	Yolo5* circutYolo;
	Yolo5* elementYolo;
	Yolo5* graphYolo = nullptr;

	try
	{
		if(config.circutNetworkFileName.empty())
		{
			size_t length;
			const char* data = res::circutNetwork(length);
			circutYolo = new Yolo5(length, data, 1);
		}
		else
		{
			Log(Log::DEBUG)<<"Reading circut network from "<<config.circutNetworkFileName;
			circutYolo = new Yolo5(config.circutNetworkFileName, 1);
		}

		if(config.elementNetworkFileName.empty())
		{
			size_t length;
			const char* data = res::elementNetwork(length);
			elementYolo = new Yolo5(length, data, 7);
		}
		else
		{
			Log(Log::DEBUG)<<"Reading circut network from "<<config.circutNetworkFileName;
			elementYolo = new Yolo5(config.elementNetworkFileName, 7);
		}

		if(!config.graphNetworkFileName.empty())
		{
			graphYolo = new Yolo5(config.graphNetworkFileName, 1);
			Log(Log::DEBUG)<<"Red element network from "<<config.graphNetworkFileName;
		}
	}
	catch(const cv::Exception& ex)
	{
		Log(Log::ERROR)<<ex.what();
		return 1;
	}

	if(config.outputCircut && !std::filesystem::is_directory(config.outDir/"circuts"))
	{
		if(!std::filesystem::create_directory(config.outDir/"circuts"))
		{
			Log(Log::ERROR)<<config.outDir/"circuts"<<" is not a directory and a directory could not be created at this location";
			return 4;
		}
	}
	if(config.outputSummaries && !std::filesystem::is_directory(config.outDir/"summaries"))
	{
		if(!std::filesystem::create_directory(config.outDir/"summaries"))
		{
			Log(Log::ERROR)<<config.outDir/"summaries"<<" is not a directory and a directory could not be created at this location";
			return 4;
		}
	}

	if(Log::level == Log::SUPERDEBUG)
	{
		cv::namedWindow( "Viewer", cv::WINDOW_NORMAL );
		cv::resizeWindow("Viewer", 960, 500);
	}

	std::vector<std::shared_future<std::shared_ptr<Document>>> futures;
	futures.reserve(THREADS);

	const std::vector<std::filesystem::path> files = toFilePaths(config.paths);

	std::vector<std::shared_ptr<Document>> documents;

	for(size_t i = 0; i < files.size();)
	{
		while(i < files.size() && futures.size() < THREADS)
		{
			futures.push_back(std::async(std::launch::async, Document::load, files[i]));
			Log(Log::INFO)<<"Loading document "<<i<<" of "<< files.size();
			++i;
		}

		while(futures.size() >= THREADS)
		{
			for(size_t j = 0; j < futures.size(); ++j)
			{
				if(futures[j].wait_for(std::chrono::microseconds(0)) == std::future_status::ready)
				{
					std::shared_ptr<Document> document = futures[j].get();
					if(document)
					{
						process(document, circutYolo, elementYolo, graphYolo, config);
						if(config.outputStatistics)
							documents.push_back(document);
						Log(Log::INFO)<<"Finished document. documents in queue: "<<futures.size();
					}
					else
					{
						Log(Log::WARN)<<"Failed to load document. documents in qeue: "<<futures.size();
					}
					futures.erase(futures.begin()+j);
				}
			}
		}
	}

	Log(Log::INFO)<<"Working on final documents";

	for(size_t j = 0; j < futures.size(); ++j)
	{
		std::shared_ptr<Document> document = futures[j].get();
		if(document)
		{
			process(document, circutYolo, elementYolo, graphYolo, config);
			if(config.outputStatistics)
				documents.push_back(document);
		}
		Log(Log::INFO)<<"Finished document";
	}

	delete circutYolo;
	delete elementYolo;
	if(graphYolo)
		delete graphYolo;

	if(config.outputStatistics && !outputStatistics(documents, config))
		return 3;

	return 0;
}
