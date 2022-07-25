#pragma once
#include <memory>
#include <string>
#include <filesystem>
#include <opencv2/core/mat.hpp>

#include "circut.h"
#include "yolo.h"
#include "log.h"

class Document
{
public:
	std::string title;
	std::string keywords;
	std::string basename;
	std::string author;
	std::vector<cv::Mat> pages;
	std::vector<Circut> circuts;

	void print(Log::Level level) const;

	bool saveCircutImages(const std::filesystem::path folder) const;
	bool saveDatafiles();

	static std::shared_ptr<Document> load(const std::string& fileName);

	bool process(Yolo5* circutYolo, Yolo5* elementYolo);

	void removeEmptyCircuts();
};

std::vector<cv::Mat> getCircutImages(std::vector<cv::Mat> images, Yolo5* yolo,
										std::vector<float>* probs = nullptr, std::vector<cv::Rect>* rects = nullptr);
