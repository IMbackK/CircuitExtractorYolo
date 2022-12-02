#include "resources.h"

#define INCBIN_PREFIX r
#include "incbin.h"
#include <sstream>

INCBIN(CircutNetwork, "../CircutExtractorYoloData/networks/circut/640/best.onnx");
INCBIN(ElementNetwork, "../CircutExtractorYoloData/networks/element/640/best.onnx");
INCBIN(GraphNetwork, "../CircutExtractorYoloData/networks/graph/1280/best.onnx");
INCTXT(Dictionary, "../CircutExtractorYoloData/top1000words.txt");

const char* res::circutNetwork(size_t& size)
{
	size = rCircutNetworkSize;
	return reinterpret_cast<const char*>(rCircutNetworkData);
}

const char* res::elementNetwork(size_t& size)
{
	size = rElementNetworkSize;
	return reinterpret_cast<const char*>(rElementNetworkData);
}

const char* res::graphNetwork(size_t& size)
{
	size = rGraphNetworkSize;
	return reinterpret_cast<const char*>(rGraphNetworkData);
}

std::vector<std::string> res::dictionary()
{
	std::vector<std::string> lines;

	std::stringstream file(rDictionaryData);

	while(file.good())
	{
		std::string str;
		std::getline(file, str);
		lines.push_back(str);
	}
	return lines;
}
