#include <iostream>
#include <string>
#include <random>

#include "NeuralNet.h"
#include "Activations.h"
#include "read_data.h"

std::vector<std::vector<double>> train_and_test_samples = {};

std::string CHARMAP = ".'`^_,:;-~+*?!i><][}{1)(|/IltfjrxnuvczXYUJCLQ0OZmwqpdbkhao#MW&8%B$@";

//prints a 1d vector
template<class T>
void show(std::vector<T> toPrint)
{
	for (T item: toPrint)
	{
		std::cout << item << " ";
	}
	std::cout << "\n";
}

//returns index of highest number in vector
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

//create random generated range
std::vector<unsigned int> create_range(int to)
{
	std::vector<unsigned int> range;
	for (int i = 0; i < to; i++)
	{
		range.push_back(i);
	}
	std::shuffle(std::begin(range), std::end(range), std::default_random_engine{});
	return range;
}

//predict with network on input
void predict(Net* network, std::vector<double> input)
{
	
	for (int i = 0; i < 784; i++)
	{
		std::cout << input[i];
		if (input[i] / 10 <= 1) //padding for printing
		{
			std::cout << 0;
		}
		if (input[i] / 10 <= 10) //padding for printing
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

	int predicted_value =  argmax(network->forward_prop(input));

	std::chrono::duration<double> elapsed_seconds = std::chrono::system_clock::now() - start;
	std::cout << "predicted: " << predicted_value << "\n";
	std::cout << "Predicted img in " << elapsed_seconds.count() << "s\n";
}

int main()
{
	//intializing random for weight intialisation
	srand(time(NULL));
	std::vector<std::vector<std::vector<double>>> data;
	read_csv(&data); // requires 400 MB of RAM

	//when using tanh as output layer activation function, changing base values for one hot encoded must be set to -1 instead of 0
	//see loadVectorsfromString -> expected vector second argument
	Net network({ {784,"non"},{200,"tanh"},{100,"tanh"},{10,"tanh"}}, 0.001);
	int sampleSize = data.size();

	for (int epochs = 0; epochs < 5; epochs++)
	{	
		double error = 0;
		for (int index: create_range(sampleSize))
		{
			std::vector<double> output = network.forward_prop(data[index][0]);	// forward prop
			if (argmax(output) != argmax(data[index][1]))						// check results
				error++;														// calc display error (just check if predicted values == actual values)

			network.back_prop(data[index][1]);									// train
		}
		std::cout<<"Epoch: " << epochs << "\n";
		std::cout<<"Error: " << error / sampleSize << "\n";
		network.save("../here"); //saving network after each epoch
	}
	Net newNet = Net::load("../here");

	//testing image 68 from traindata set
	show(data[67][1]);
	predict(&newNet, data[67][0]);

	//draw your own number and let it be predicted by the nn
	while (true)
	{
		draw();
		std::vector<double> input1 = read_bmp("temp.bmp");
		predict(&newNet, input1);
		std::cin.get();
	}
	
	return 0;
}