#pragma once
#include <fstream>
#include <windows.h>
#include <iostream>
#include <vector>
#include <string>

void read(std::vector<std::vector<double>>* data, std::vector<std::vector<double>>* expected)
{
	std::ifstream file;
	#if defined(NDEBUG)
		file.open("../../train.csv");
	#else
		file.open("../train.csv");
	#endif
	if (!file.is_open())
	{
		std::cout << "Error opening file\n";
	}
	std::string buffer;
	std::getline(file, buffer); //skiping title
	std::string delimiter = ",";
	while (std::getline(file,buffer))
	{
		data->push_back({});
		size_t pos = 0;
		std::string token;
		while ((pos = buffer.find(delimiter)) != std::string::npos) {
			token = buffer.substr(0, pos);
			data->at(data->size() - 1).push_back(atoi(token.c_str()));
			buffer.erase(0, pos + delimiter.length());
		}
		int dsize = data->size() - 1;
		data->at(dsize).push_back(atoi(buffer.c_str()));
		expected->push_back(std::vector<double>(10,-1));
		expected->at(expected->size() - 1)[int(data->at(dsize)[0])] = 1;
		data->at(dsize).erase(data->at(dsize).begin());
	}
	std::cout << "Loaded Training Data\n";
}