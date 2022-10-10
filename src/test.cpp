#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <poppler-document.h>
#include <poppler-image.h>
#include <poppler-global.h>
#include <poppler-page.h>
#include <poppler-page-renderer.h>
#include <filesystem>
#include <vector>
#include <future>
#include <memory>
#include <string>
#include <filesystem>
#include <thread>

#include "log.h"
#include "document.h"
#include "circut.h"
#include "randomgen.h"
#include "linedetection.h"
#include "yolo.h"
#include "document.h"
#include "resources.h"

#define THREADS 16

typedef enum
{
	ALGO_INVALID = -1,
	ALGO_CIRCUT = 0,
	ALGO_ELEMENT,
	ALGO_NET,
	ALGO_GRAPH,
	ALGO_COUNT,
	ALGO_POPPLER
} Algo;

void printUsage(int argc, char** argv)
{
	Log(Log::INFO)<<"Usage: "<<argv[0]<<" [ALGO] [IMAGEFILENAME]";
	Log(Log::INFO)<<"Valid algos: circut, element, net, graph";
}

Algo parseAlgo(const std::string& in)
{
	Algo out = ALGO_INVALID;
	try
	{
		int tmp = std::stoi(in);
		if(tmp >= ALGO_COUNT || tmp < 0)
		{
			const char msg[] = "Algo enum out of range";
			Log(Log::ERROR)<<msg;
			throw std::invalid_argument(msg);
		}
		out = static_cast<Algo>(tmp);
	}
	catch(const std::invalid_argument& ex)
	{
		if(in == "circut")
			out = ALGO_CIRCUT;
		else if(in == "element")
			out = ALGO_ELEMENT;
		else if(in == "net")
			out = ALGO_NET;
		else if(in == "graph")
			out = ALGO_GRAPH;
		else if(in == "poppler")
			out = ALGO_POPPLER;
		else
			out = ALGO_INVALID;
	}
	return out;
}

void algoCircut(cv::Mat& image)
{
	size_t length;
	const char* data = res::circutNetwork(length);
	Yolo5* yolo = new Yolo5(length, data, 1, 640, 640);

	std::vector<cv::Mat> images({image});
	std::vector<cv::Mat> detections = getYoloImages(images, yolo);

	delete yolo;
}

void algoElement(cv::Mat& image)
{
	size_t length;
	const char* data = res::elementNetwork(length);
	Yolo5* yolo = new Yolo5(length, data, 7, 640, 640);

	Circut circut;
	circut.image = extendBorder(image, 15);

	if(Log::level == Log::SUPERDEBUG)
	{
		cv::imshow("Viewer", circut.image);
		cv::waitKey(0);
	}

	circut.detectElements(yolo);

	if(Log::level == Log::SUPERDEBUG)
	{
		cv::imshow("Viewer", circut.ciructImage());
		cv::waitKey(0);
	}

	delete yolo;
}

void algoLine(cv::Mat& image)
{
	size_t length;
	const char* data = res::elementNetwork(length);
	Yolo5* yolo = new Yolo5(length, data, 7, 640, 640);

	Circut circut;
	circut.image = extendBorder(image, 15);

	circut.detectElements(yolo);

	circut.detectNets();

	circut.setDirectionHint(circut.estimateDirection());

	circut.parseCircut();

	std::string modelString = circut.getString();

	if(Log::level == Log::SUPERDEBUG)
	{
		cv::imshow("Viewer", circut.ciructImage());
		cv::waitKey(0);
	}

	Log(Log::INFO)<<"Parsed string: "<<modelString;
}

void algoGraph(cv::Mat& image)
{
	size_t length;
	const char* data = res::graphNetwork(length);
	Yolo5* yolo = new Yolo5(length, data, 1, 640, 640);

	std::vector<cv::Mat> images({image});
	std::vector<cv::Mat> detections = getYoloImages(images, yolo);

	delete yolo;
}

static std::vector<std::filesystem::path> toFilePaths(const std::vector<std::filesystem::path>& paths)
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

static int popplerEnumToCvFormat(int format)
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

void documentPipeline(const std::vector<std::filesystem::path>& files, size_t stride, size_t offset)
{
	poppler::page_renderer renderer;
	renderer.set_render_hint(poppler::page_renderer::antialiasing, true);

	for(size_t i = offset; i < files.size(); i+=stride)
	{
		/*std::shared_ptr<Document> document = Document::load(files[i]);*/
		poppler::document* document = poppler::document::load_from_file(files[i]);
		if(!document)
		{
			Log(Log::ERROR)<<"Could not load pdf file from "<<files[i];
			continue;
		}

		if(document->is_encrypted())
		{
			Log(Log::ERROR)<<"Only unencrypted files are supported";
			continue;
		}

		int pagesCount = document->pages();
		if(pagesCount > 10)
		{
			Log(Log::WARN)<<"only loading first 2 pages down from "<<pagesCount;
			pagesCount = 2;
		}
		std::vector<cv::Mat> output;
		for(int j = 0; j < pagesCount; ++j)
		{
			poppler::page* page = document->create_page(j);
			poppler::image image = renderer.render_page(page, 300, 300);

			const char* data = image.const_data();
			if(data)
				Log(Log::DEBUG)<<data[0];
			else
				Log(Log::DEBUG)<<"No data";
			delete page;
			cv::Mat cvBufferConst(image.height(), image.width(), popplerEnumToCvFormat(image.format()), const_cast<char*>(image.const_data()), image.bytes_per_row());
			cv::Mat cvBuffer(cvBufferConst.clone());
			output.push_back(cvBuffer);

			cv::imwrite("out.png", cvBuffer);
		}

		for(const cv::Mat& mat : output)
		{
			cv::imwrite("out.png", mat);
		}

		delete document;
		if(!document)
		{
			Log(Log::INFO)<<i<<"/"<<files.size()<<" skipped";
			continue;
		}
		else
		{
			Log(Log::INFO)<<i<<"/"<<files.size();
		}
	}
}

static void altAlgoPoppler(const std::filesystem::path& path)
{
	std::vector<std::filesystem::path> files = toFilePaths({path});
	std::vector<std::thread> threads(THREADS);
	for(size_t i = 0; i < threads.size(); ++i)
	{
		threads[i] = std::thread(documentPipeline, files, THREADS, i);
	}

	for(size_t i = 0; i < threads.size(); ++i)
	{
		threads[i].join();
	}
}

static void algoPoppler(const std::filesystem::path& path)
{
	std::vector<std::filesystem::path> files = toFilePaths({path});
	/*for(size_t i = 0; i < files.size(); ++i)
	{
		std::shared_ptr<Document> document = Document::load(files[i]);
		if(!document)
		{
			Log(Log::INFO)<<i<<"/"<<files.size()<<" skipped";
			continue;
		}
		else
		{
			Log(Log::INFO)<<i<<"/"<<files.size();
		}

		(void)document->getText();
	}*/

	std::vector<std::future<std::shared_ptr<Document>>> futures;
	futures.reserve(THREADS);

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
					futures.erase(futures.begin()+j);
					if(document)
					{
						for(const cv::Mat& mat : document->pages)
						{
							if(!mat.u)
								assert(false);
							else
								Log(Log::DEBUG)<<"refcount: "<<mat.u->refcount;
						}
					}
				}
			}
		}
	}
}

int main(int argc, char** argv)
{
	rd::init();
	Log::level = Log::SUPERDEBUG;
	if(argc != 3)
	{
		printUsage(argc, argv);
		return 1;
	}

	Algo algo = parseAlgo(argv[1]);

	cv::Mat image;

	if(algo != ALGO_POPPLER)
	{
		image = cv::imread(argv[2]);
		if(!image.data)
		{
			Log(Log::ERROR)<<argv[2]<<" is not a valid image file";
			return 2;
		}
		if(Log::level == Log::SUPERDEBUG)
		{
			cv::namedWindow( "Viewer", cv::WINDOW_NORMAL );
			cv::imshow("Viewer", image);
			cv::waitKey(0);
		}
	}

	switch(algo)
	{
		case ALGO_CIRCUT:
			algoCircut(image);
			break;
		case ALGO_ELEMENT:
			algoElement(image);
			break;
		case ALGO_NET:
			algoLine(image);
			break;
		case ALGO_GRAPH:
			algoGraph(image);
			break;
		case ALGO_POPPLER:
			altAlgoPoppler(argv[2]);
			break;
		case ALGO_INVALID:
		default:
			Log(Log::ERROR)<<'\"'<<argv[1]<<"\" is not a valid algorithm";
			return 3;
	}

	return 0;
}
