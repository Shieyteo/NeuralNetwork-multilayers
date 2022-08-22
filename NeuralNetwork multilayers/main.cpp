#include <iostream>
#include <vector>
#include <string>
#include <windows.h>
#include "_matrix.h"
#include "NeuralNet.h"

const static double LRELUC = 0.25;

double m_max(double a, double b)
{
	return a > b ? a : b;
}

double m_min(double a, double b)
{
	return a < b ? a : b;
}

double Sigmoid(double inp)
{
	return 1 / (1 + exp(-inp));
}

double ReLU(double in)
{
	return m_max(in, 0);
}

double LeakyReLU(double in)
{
	return in >= 0 ? in : in * LRELUC;
}

double derivative_LeakyReLU(double in)
{
	return in >= 0 ? 0 : LRELUC;
}

double derivative_ReLU(double in)
{
	return in <= 0 ? 0 : 1;
}

double derivative_Sigmoid(double b)
{
	return b * (1 - b);
}


double derivative_tanh(double b)
{
	double e = tanh(b);
	return 1 - e * e;
}

double derivative_Error(double a, double b)
{
	return 2 * (b - a); //derivative of (a-b)^2
}

static double (*activation)(double) { &ReLU};

static double (*derivative_activation)(double) { &derivative_ReLU};

double getRand(char c = 'w')
{
	switch (c)
	{
	case 'w':
		return double(rand()) / RAND_MAX * 2 - 1;
	case 'b':
		return double(rand()) / RAND_MAX;
	}
}

std::vector<double> forwardProp(std::vector<std::vector<double>>* weights, std::vector<double>* values, std::vector<double>* bias)
{
	std::vector<double> ret;
	for (int output = 0; output < weights->at(0).size(); output++)
	{
		double sum = bias->at(output);
		for (int input = 0; input < weights->size(); input++)
		{
			sum += weights->at(input)[output] * values->at(input);
		}
		//ret.push_back(Sigmoid(sum));
		ret.push_back(activation(sum));
	}
	return ret;	// takes in weights and values and multiplies them with eachother + adds the biases
}

// input at index 0 and 1 expected output at index 2
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
	
	srand(time(NULL)); // intialize rand
	const double lr = 0.01;	//learning rate
	createDataSet(1,0.2);
	Net network({ 2,3,3,1 }, lr, Sigmoid, derivative_Sigmoid);
	for (int epoch = 0; epoch < 100000; epoch++)
	{
		for (std::vector<double> sample: train_and_test_samples)
		{
			std::vector<double> output = network.forwars_prop({ sample[0],sample[1] });
			network.back_prop({ sample[2] });
			system("cls");
			std::cout << "Output: " << output[0] << " Expected: " << sample[2];
		}

	}

	return 0;
	//train_and_test_samples = { {1,1,1},{0,1,0},{1,0,0},{0,0,1} };
	//initiate weights and biases
	std::vector<std::vector<double>> weights0 = { { getRand(),getRand(),getRand() },{ getRand(),getRand(),getRand()} };
	std::vector<std::vector<double>> weights1 = { { getRand(),getRand(),getRand() },{ getRand(),getRand(),getRand()},{ getRand(),getRand(),getRand() } }; // from input layer to hidden layer 
	std::vector<std::vector<double>> weights2 = { { getRand()},{getRand() },{ getRand()} }; // from hidden layer to output layer
	std::vector<double> bias0 = { getRand('b'),getRand('b'),getRand('b')};
	std::vector<double> bias1 = { getRand('b'),getRand('b'),getRand('b') }; // biases for  the hidden layer
	std::vector<double> bias2 = { getRand('b')}; // bias for  the output layer
	
	
	for (int epoch = 0; epoch < 1000000000; epoch++)
	{
		double error = 0;
		for (std::vector<double> sample : train_and_test_samples)
		{
			//initalize train data
			std::vector<double> input = { sample[0],sample[1] };
			double expected = sample[2];

			//forward propagation
			std::vector<double> hidden_layer_values0 = forwardProp(&weights0, &input, &bias0);
			std::vector<double> hidden_layer_values1 = forwardProp(&weights1, &hidden_layer_values0, &bias1);
			std::vector<double> output_layer_values = forwardProp(&weights2, &hidden_layer_values1, &bias2); // in this senario just one value

			error += abs(expected - output_layer_values[0]);
			//BACKPROP

			//Outputlayer error calculation	
			double d_err = derivative_Error(expected, output_layer_values[0]);
			double d_sig = derivative_activation(output_layer_values[0]);

			double complete_error20 = d_err * d_sig;

			//Hiddenlayer error calculation										
			double d_err_hidden10 = complete_error20 * weights2[0][0];
			double d_err_hidden11 = complete_error20 * weights2[1][0];
			double d_err_hidden12 = complete_error20 * weights2[2][0];


			double d_sig_hidden10 = derivative_activation(hidden_layer_values1[0]);
			double d_sig_hidden11 = derivative_activation(hidden_layer_values1[1]);
			double d_sig_hidden12 = derivative_activation(hidden_layer_values1[2]);

			double complete_error10 = d_err_hidden10 * d_sig_hidden10;
			double complete_error11 = d_err_hidden11 * d_sig_hidden11;
			double complete_error12 = d_err_hidden12 * d_sig_hidden12;

			double d_err_hidden00 = complete_error10 * weights1[0][0] + complete_error11 * weights1[0][1] + complete_error12 * weights1[0][2];
			double d_err_hidden01 = complete_error10 * weights1[1][0] + complete_error11 * weights1[1][1] + complete_error12 * weights1[1][2];
			double d_err_hidden02 = complete_error10 * weights1[2][0] + complete_error11 * weights1[2][1] + complete_error12 * weights1[2][2];

			double d_sig_hidden00 = derivative_activation(hidden_layer_values0[0]);
			double d_sig_hidden01 = derivative_activation(hidden_layer_values0[1]);
			double d_sig_hidden02 = derivative_activation(hidden_layer_values0[2]);

			double complete_error00 = d_err_hidden00 * d_sig_hidden00;
			double complete_error01 = d_err_hidden01 * d_sig_hidden01;
			double complete_error02 = d_err_hidden02 * d_sig_hidden02;

			//bias adjustment
			bias2[0] -= lr * complete_error20;

			//weight adjustment
			weights2[0][0] -= lr * complete_error20 * hidden_layer_values1[0];
			weights2[1][0] -= lr * complete_error20 * hidden_layer_values1[1];

			//bias adjustment
			bias1[0] -= lr * complete_error10;
			bias1[1] -= lr * complete_error11;
			bias1[2] -= lr * complete_error12;

			//weight adjustment
			weights1[0][0] -= lr * complete_error10 * hidden_layer_values0[0];
			weights1[1][0] -= lr * complete_error10 * hidden_layer_values0[1];
			weights1[2][0] -= lr * complete_error10 * hidden_layer_values0[2];
			weights1[0][1] -= lr * complete_error11 * hidden_layer_values0[0];
			weights1[1][1] -= lr * complete_error11 * hidden_layer_values0[1];
			weights1[2][1] -= lr * complete_error11 * hidden_layer_values0[2];
			weights1[0][2] -= lr * complete_error12 * hidden_layer_values0[0];
			weights1[1][2] -= lr * complete_error12 * hidden_layer_values0[1];
			weights1[2][2] -= lr * complete_error12 * hidden_layer_values0[2];

			//bias adjustment
			bias0[0] -= lr * complete_error00;
			bias0[1] -= lr * complete_error01;
			bias0[2] -= lr * complete_error02;

			//weight adjustment
			weights0[0][0] -= lr * complete_error00 * input[0];
			weights0[1][0] -= lr * complete_error00 * input[1];
			weights0[0][1] -= lr * complete_error01 * input[0];
			weights0[1][1] -= lr * complete_error01 * input[1];
			weights0[0][2] -= lr * complete_error02 * input[0];
			weights0[1][2] -= lr * complete_error02 * input[1];

		}
		if (epoch % 100 == 0)
		{
			std::string map;
			int size = 50;
			for (float i = size-1; i > -1; i--)
			{
				for (float j = 0; j < size; j++)
				{
					std::vector<double> val_input = { i / size,j / size };
					std::vector<double> h0 = forwardProp(&weights0, &val_input, &bias0);
					std::vector<double> h1 = forwardProp(&weights1, &h0, &bias1);
					std::vector<double> o = forwardProp(&weights2, &h1, &bias2); // in this senario just one value
					int index = (m_min(66,m_max(0, round(o[0] * 67))));
					map += CHARMAP[index];
					map += ' ';
				}
				map += "\n";
			}
			Sleep(300);
			system("cls");
			std::cout << "Epoch: "<<epoch<<"\nError: "<<(error/train_and_test_samples.size()) << "\n" << map ;
			for (int i = 0; i < weights2.size(); i++)
			{
				for (int j = 0; j < weights2[0].size(); j++)
				{
					std::cout << weights2[i][j] << "\n";
				}
			}
		}
	}
}