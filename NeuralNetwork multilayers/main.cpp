#include <iostream>
#include <string>
#include <algorithm>
#include <random>
#include <windows.h>
#include <thread>
#include "NeuralNet.h"
#include "Activations.h"
#include "read_csv.h"

std::vector<std::vector<double>> train_and_test_samples = {};

std::string CHARMAP = ".'`^_,:;-~+*?!i><][}{1)(|/IltfjrxnuvczXYUJCLQ0OZmwqpdbkhao#MW&8%B$@";

template<class T>
void show(std::vector<T> toPrint)
{
	for (T item: toPrint)
	{
		std::cout << item << " ";
	}
	std::cout << "\n";
}

template<class T>
int argmax(std::vector<T> findMax)
{
	int index = 0;
	double high = findMax[0];
	for (int i = 0; i < findMax.size(); i++)
	{
		if (high < findMax[i])
		{
			index = i;
			high = findMax[i];
		}
	}
	return index;
}

std::vector<std::vector<double>> average(std::vector < std::vector<std::vector<double>>>& inp)
{
	std::vector<std::vector<double>> ret;
	for (int i = 0; i < inp[0].size(); i++)
	{
		std::vector<double> temp;
		for (int j = 0; j < inp[0][0].size(); j++)
		{
			double sum = 0;
			for (int k = 0; k < inp.size(); k++)
			{
				sum += inp[k][i][j];
			}
			temp.push_back(sum / inp.size());
		}
		ret.push_back(temp);
	}
	return ret;
}

void createDataSet(unsigned int number, float precision=0.05)
{
	switch (number)
	{
	case 0:
	{
		for (float x = 0; x < 1; x += precision)
		{
			for (float y = 0; y < 1; y += precision)
			{
				if ((x > 0.5 && y <= 0.5) || (x <= 0.5 && y > 0.5))
				{
					train_and_test_samples.push_back({ x,y,0 });
				}
				else
				{
					train_and_test_samples.push_back({ x,y,1 });
				}
			}
		}
		return;
	}
	case 1:
	{
		for (float x = 0; x < 1; x += precision)
		{
			for (float y = 0; y < 1; y += precision)
			{
				if (x > 0.3 && x <= 0.7 && y > 0.3 && y <= 0.7)
				{
					train_and_test_samples.push_back({ x,y,1 });
				}
				else
				{
					train_and_test_samples.push_back({ x,y,0 });
				}
			}
		}
		return;
	}
	case 2:
		for (double i = 0; i < 10; i++)
		{
			for (double j = 0; j < 10; j++)
			{
				train_and_test_samples.push_back({ i,j,i * j });
			}
		}
	}
}

std::vector<unsigned int> create_range(int to)
{
	std::vector<unsigned int> range;
	for (int i = 0; i < to; i++)
	{
		range.push_back(i);
	}
	auto rng = std::default_random_engine{};
	std::shuffle(std::begin(range), std::end(range), rng);
	return range;
}

void predict(Net* network, std::vector<std::vector<double>> input)
{
	for (int i = 0; i < 784; i++)
	{
		std::cout << input[0][i];
		if (input[0][i] / 10 <= 1)
		{
			std::cout << 0;
		}
		if (input[0][i] / 10 <= 10)
		{
			std::cout << 0;
		}
		if (i % 28 == 27)
		{
			std::cout << "\n";
		}
	}
	std::chrono::time_point<std::chrono::system_clock> start;
	start = std::chrono::system_clock::now();

	int a =  argmax(network->forawrd_prop(input[0]));

	std::chrono::duration<double> elapsed_seconds = std::chrono::system_clock::now() - start;
	std::cout << "predicted: " << a << "\n";
	std::cout << "Predicted img in " << elapsed_seconds.count() << "s\n";
	
}

int main()
{
	srand(time(NULL));
	read_ppm("C:/Users/TH/source/repos/Shieyteo/NeuralNetwork-multilayers/test.ppm");
	std::vector<std::vector<std::vector<double>>> data;
	Net network({ {784,"non"},{100,"tanh"},{100,"tanh"},{10,"tanh"}}, 0.003);

	read(&data);
	int sampleSize = data.size();

	for (int epochs = 0; epochs < 5; epochs++)
	{	
		double error = 0;
		for (int index: create_range(sampleSize))
		{
			std::vector<double> output = network.forawrd_prop(data[index][0]);	// forward prop
			if (argmax(output) != argmax(data[index][1]))						// check results
				error++;														// calc display error

			network.back_prop(data[index][1]);									// train
		}
		std::cout<<"Epoch: " << epochs << "\n";
		std::cout<<"Error: " << error / sampleSize << "\n";
	}

	network.save("../../here");
	Net newNet = Net::load("../../here");
	while (std::cin.get())
	{

		predict(&newNet, data[rand()]);
	}
	return 0;
}