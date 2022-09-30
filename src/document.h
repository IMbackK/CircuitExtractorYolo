#pragma once
#include <memory>
#include <string>
#include <filesystem>
#include <opencv2/core/mat.hpp>

#include "circut.h"
#include "graph.h"
#include "yolo.h"
#include "log.h"

class Document
{
public:
	std::string title;
	std::string keywords;
	std::string basename;
	std::string author;
	std::string field = "Unkown";
	std::vector<cv::Mat> pages;
	std::vector<Circut> circuts;
	std::vector<Graph> graphs;

	void print(Log::Level level) const;

	bool saveCircutImages(const std::filesystem::path& folder) const;
	bool saveDatafile(const std::filesystem::path& folder);
	void dropImages();

	static std::shared_ptr<Document> load(const std::string& fileName);

	bool process(Yolo5* circutYolo, Yolo5* elementYolo, Yolo5* graphYolo);

	void removeEmptyCircuts();
};

std::vector<cv::Mat> getYoloImages(std::vector<cv::Mat> images, Yolo5* yolo,
										std::vector<float>* probs = nullptr, std::vector<cv::Rect>* rects = nullptr);
