#pragma once
#include <fstream>
#include <windows.h>
#include <iostream>
#include <mutex>

#include <vector>
#include <string>
#include <chrono>

const char delimiter = ',';	// csv delimiter
std::mutex mtx;	//mutex to ensure loadVectorsfromString thread saftey

//parameter struct for loadVectorsfromString
struct params
{
	std::string data;
	std::vector<std::vector<std::vector<double>>>* input;
};

//thread save loading function for csv file 
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

//reads ppm file
std::vector<double> read_ppm(std::string path)
{
	std::basic_ifstream<unsigned char> file;
	file.open(path, std::ios::binary);
	std::basic_string<unsigned char> buffer;
	
	//header
	std::getline(file, buffer);
	std::getline(file, buffer);
	std::getline(file, buffer);
	std::getline(file, buffer);
	std::vector<double> colorArray;

	for (int i = 0; i < buffer.size(); i+=3)
	{
		//inbuild grayscale
		colorArray.push_back(abs((0.3*buffer[i]+0.59*buffer[i + 1]+0.11*buffer[i + 2])-255));
	}
	return colorArray;
}

/// non encoded bitmap only
std::vector<double> read_bmp(std::string path)
{
	std::basic_ifstream<unsigned char> file;
	file.open(path, std::ios::binary);
	if (!file.is_open())
	{
		std::cout << "File not found\n";
		return std::vector<double>();
	}

	//header
	std::basic_string<unsigned char> buffer;
	std::getline(file, buffer);
	buffer.erase(buffer.begin(), buffer.begin() + 54);
	std::vector<double> colorArray;
	for (int i = 0; i < buffer.size(); i += 3)
	{
		//inbuild grayscale
		//r = 0.3 g = 0.59 b = 0.11
		colorArray.push_back(abs(int(0.3 * buffer[i] + 0.59 * buffer[i + 1] + 0.11 * buffer[i + 2]) - 255));
	}
	//rotate img
	std::reverse(colorArray.begin(),colorArray.end());
	for (int i = 0; i < 784; i+= 28)
	{
		std::reverse(colorArray.begin()+i, colorArray.begin()+28+i);
	}
	return colorArray;
}

//draw in paint blocks until enter is pressed
//file needs to be saved before continueing
void draw()
{
	CopyFile(L"../copy.bmp", L"temp.bmp", false);
	ShellExecute(nullptr, L"open", L"mspaint.exe", L"temp.bmp", nullptr, SW_SHOWDEFAULT);
	std::cout << "Press enter if you drew the number ...";
	std::cin.get();
}

//returns path of exe (not used)
std::string ExePath() {
	char buffer[MAX_PATH];
	GetModuleFileNameA(NULL, buffer, MAX_PATH);
	return std::string(buffer);
}

//reads train.csv file, loads it in input vector needs to be 784 pixels
void read_csv(std::vector<std::vector<std::vector<double>>>* input)
{
	std::chrono::time_point<std::chrono::system_clock> start;
	start = std::chrono::system_clock::now(); // starting timer

	std::ifstream file;
	file.open("../train.csv"); // needs to be location of training csv

	if (!file.is_open())
	{
		std::cout << "Error opening file train.csv\nLook at read_csv for more information\n";
	}

	std::string buffer;
	std::getline(file, buffer); //skiping title

	while (std::getline(file,buffer))
	{
		params* para = new params;
		para->data = buffer;
		para->input = input;
		loadVectorsfromString(para);
		//CreateThread(NULL, 0,loadVectorsfromString ,para, 0, NULL); //thread instead of blocking -> less performent
	}

	std::chrono::duration<double> elapsed_seconds = std::chrono::system_clock::now() - start; // measuring time
	std::cout << "Loaded Training Data in " << elapsed_seconds.count() << "s\n";
}