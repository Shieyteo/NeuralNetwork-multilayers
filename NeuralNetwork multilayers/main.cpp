#include <iostream>
#include <string>
#include <windows.h>
//#include "NerualNetwork.h"
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

int main()
{
	
	srand(time(NULL));
	/*
	createDataSet(0,0.1);
	//train_and_test_samples = { {1,0,0},{1,1,1},{0,1,0},{0,0,1} };
	Net network({ {2,NON},{10,SIGMOID},{10,SIGMOID},{1,SIGMOID}}, 0.1);
	for (int epoch = 0; epoch < 1000000; epoch++)
	{
		double err = 0;
		for (std::vector<double> sample : train_and_test_samples)
		{
			std::vector<double> output = network.forawrd_prop({ sample[0],sample[1] });
			err += abs(sample[2] - output[0]);
			network.back_prop({ sample[2] });
		}
		if (epoch % 50 == 0)
		{
			std::string map;
			int size = 40;
			for (float i = size - 1; i > -1; i--)
			{
				for (float j = 0; j < size; j++)
				{
					std::vector<double> val_input = { i / size,j / size };
					std::vector<double> o = network.forawrd_prop(val_input);
					int index = (m_min(66, m_max(0, round(o[0] * 67))));
					map += CHARMAP[index];
					map += ' ';
				}
				map += "\n";
			}
			Sleep(100);
			system("cls");
			std::cout << "Epoch: " << epoch << "\nError: " << (err / train_and_test_samples.size()) << "\n" << map;
		}
	}
	*/
	std::vector<std::vector<double>> data;
	std::vector< std::vector<double>> expected;
	read(&data, &expected);
	Net network({ {784,NON},{100,TANH},{10,TANH} }, 0.002);
	//Network NW({ {784,NON},{200,TANH},{10,TANH}}, 0.002);
	system("cls");
	for (int epochs = 0; epochs < 1000; epochs++)
	{	
		for (int sample_index = 0; sample_index < data.size();sample_index++)
		{
			network.forawrd_prop(data[sample_index]);
			network.back_prop(expected[sample_index]);
		}
		//test
		double error = 0;
		for (int test = 0; test < 20000; test++)
		{
			std::vector<double> output = network.forawrd_prop(data[test]);
			if (argmax(output) != argmax(expected[test]))
			{
				error++;
			}
		}
		std::cout<<"Epoch: "<<epochs<<"\n";
		std::cout<<"Error: " << error / 20000 << "\n";
	}
	
	std::cin.get();
	return 0;
}