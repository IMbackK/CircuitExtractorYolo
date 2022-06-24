#pragma once

#include <poppler-document.h>
#include <opencv2/core.hpp>
#include <vector>

int popplerEnumToCvFormat(int format);

std::vector<cv::Mat> getMatsFromDocument(poppler::document* document);
