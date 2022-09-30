#pragma once
#include <string>
#include <vector>
#include <argp.h>
#include <iostream>
#include <filesystem>
#include "log.h"

const char *argp_program_version = "1.0";
const char *argp_program_bug_address = "<carl@uvos.xyz>";
static char doc[] = "Application detects EIS circuts and EIS graphs in pdf files";
static char args_doc[] = "";

static struct argp_option options[] =
{
  {"verbose",			'v', 0,				0,	"Show debug messages" },
  {"quiet", 			'q', 0,				0,	"only output data" },
  {"circut-network",	'c', "[FILE]",		0,	"Circut detection neural network onnx file" },
  {"element-network",	'e', "[FILE]",		0,	"Element detection neural network onnx file" },
  {"graph-network",		'g', "[FILE]",		0,	"Graph network file name"},
  {"out-dir",			'o', "[DIRECTORY]",	0,	"Place to save output" },
  {"circut-images",		'i', 0,				0,	"Save annotated images of the found circuts"},
  {"document-summaries", 's', 0,			0,	"Save document summaries"},
  {"statistics", 		 't', 0,			0,	"Save statistics"},
  { 0 }
};

struct Config
{
	std::filesystem::path circutNetworkFileName;
	std::filesystem::path elementNetworkFileName;
	std::filesystem::path graphNetworkFileName;
	std::filesystem::path outDir;
	std::vector<std::filesystem::path> paths;
	bool outputCircut = false;
	bool outputSummaries = false;
	bool outputStatistics = false;
};

static error_t parse_opt (int key, char *arg, struct argp_state *state)
{
	Config *config = reinterpret_cast<Config*>(state->input);

	switch (key)
	{
	case 'q':
		Log::level = Log::ERROR;
		break;
	case 'v':
		Log::level = Log::DEBUG;
		break;
	case 'c':
		config->circutNetworkFileName.assign(arg);
		break;
	case 'e':
		config->elementNetworkFileName.assign(arg);
	break;
	case 'g':
		config->elementNetworkFileName.assign(arg);
	break;
	case 'o':
		config->outDir.assign(arg);
	break;
	case 'i':
		config->outputCircut = true;
		break;
	case 's':
		config->outputSummaries = true;
		break;
	case 't':
		config->outputStatistics = true;
		break;
	case ARGP_KEY_ARG:
		config->paths.push_back(std::filesystem::path(arg));
		break;
	default:
		return ARGP_ERR_UNKNOWN;
	}
	return 0;
}

static struct argp argp = {options, parse_opt, args_doc, doc};
