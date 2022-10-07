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

	struct Metadata
	{
		std::string title;
		std::string keywords;
		std::string author;
	};

private:
	std::vector<std::string> text;
	std::string field = "Unkown";
	Metadata metadata;
	std::string basename;

public:

	std::vector<cv::Mat> pages;
	std::vector<Circut> circuts;
	std::vector<Graph> graphs;

	explicit Document() = default;
	static std::shared_ptr<Document> load(const std::string& fileName);

	void dropImages();
	void removeEmptyCircuts();

	bool process(Yolo5* circutYolo, Yolo5* elementYolo, Yolo5* graphYolo);
	bool saveCircutImages(const std::filesystem::path& folder) const;
	bool saveDatafile(const std::filesystem::path& folder);
	void print(Log::Level level) const;
	std::vector<size_t> getWordOccurances(const std::vector<std::string>& words);

	std::string getBasename() const;
	std::string getField() const;
	const Metadata getMetadata() const;
	std::vector<std::string> getText();
};

std::vector<cv::Mat> getYoloImages(std::vector<cv::Mat> images, Yolo5* yolo,
										std::vector<float>* probs = nullptr, std::vector<cv::Rect>* rects = nullptr);
