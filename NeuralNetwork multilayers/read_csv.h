#pragma once
#include <fstream>
#include <windows.h>
#include <iostream>
#include <mutex>
#include <vector>
#include <string>
#include <chrono>

const char delimiter = ',';
std::mutex mtx;

struct params
{
	std::string data;
	std::vector<std::vector<std::vector<double>>>* input;
};

DWORD WINAPI loadVectorsfromString(void* para) {
	params par = *(params*)para;
	size_t pos = 0;
	std::string str = par.data;
	std::string token;
	pos = str.find(delimiter);
	token = str.substr(0, pos);
	std::vector<double> expected(10, -1);
	std::vector<double> input(0);
	expected[atoi(token.c_str())] = 1;
	str.erase(0, pos + 1);
	{
		while ((pos = str.find(delimiter)) != std::string::npos) {
			token = str.substr(0, pos);
			input.push_back(atoi(token.c_str()));
			str.erase(0, pos + 1);
		}
	}
	input.push_back(atoi(str.c_str()));
	mtx.lock();
	par.input->push_back({ input,expected });
	mtx.unlock();
	delete para;
	return 0;
};


void read(std::vector<std::vector<std::vector<double>>>* input)
{
	std::chrono::time_point<std::chrono::system_clock> start;
	start = std::chrono::system_clock::now();

	std::ifstream file;
	#if defined _WIN64 
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

	while (std::getline(file,buffer))
	{
		params* para = new params;
		para->data = buffer;
		para->input = input;
		//ThreadFunc(para);
		CreateThread(NULL, 0,loadVectorsfromString ,para, 0, NULL);
	}

	std::chrono::duration<double> elapsed_seconds = std::chrono::system_clock::now() - start;
	std::cout << "Loaded Training Data in " << elapsed_seconds.count() << "s\n";
}