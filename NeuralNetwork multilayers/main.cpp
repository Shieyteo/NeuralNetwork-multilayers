#include <iostream>
#include <string>
#include <windows.h>
#include "NeuralNet.h"
#include "Activations.h"
std::vector<std::vector<double>> train_and_test_samples = {}; //XOR Problem (Non linear) 

std::string CHARMAP = ".'`^_,:;-~+*?!i><][}{1)(|/IltfjrxnuvczXYUJCLQ0OZmwqpdbkhao#MW&8%B$@";

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
	}
}

int main()
{
	srand(time(NULL));
	createDataSet(0,0.2);
	Net network({ 2,100,70,60,10,1 }, 0.1, Sigmoid, derivative_Sigmoid);
	for (int epoch = 0; epoch < 1000000; epoch++)
	{
		double err = 0;
		for (std::vector<double> sample: train_and_test_samples)
		{
			std::vector<double> output = network.forwars_prop({ sample[0],sample[1] });
			err += abs(sample[2] - output[0]);
			network.back_prop({ sample[2] });
		}
		if (epoch % 100 == 0)
		{
			std::string map;
			int size = 50;
			for (float i = size - 1; i > -1; i--)
			{
				for (float j = 0; j < size; j++)
				{
					std::vector<double> val_input = { i / size,j / size };
					std::vector<double> o = network.forwars_prop(val_input);
					int index = (m_min(66, m_max(0, round(o[0] * 67))));
					map += CHARMAP[index];
					map += ' ';
				}
				map += "\n";
			}
			Sleep(300);
			system("cls");
			std::cout << "Epoch: " << epoch << "\nError: " << (err / train_and_test_samples.size()) << "\n" << map;
		}
	}
	return 0;
}