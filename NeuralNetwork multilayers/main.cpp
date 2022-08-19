#include <iostream>
#include <vector>
#include <string>
#include <windows.h>

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

static double (*activation)(double) { &Sigmoid};

static double (*derivative_activation)(double) { &derivative_Sigmoid};

double getRand()
{
	return double(rand()) / RAND_MAX * 2 - 1; // for weight initialization value between -1 and 1
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

void createDataSet()
{
	//unten links 1
	train_and_test_samples.push_back({ 0.4,0.0,1 });
	train_and_test_samples.push_back({ 0.3,0.0,1 });
	train_and_test_samples.push_back({ 0.2,0.0,1 });
	train_and_test_samples.push_back({ 0.1,0.0,1 });
	train_and_test_samples.push_back({ 0.0,0.0,1 });

	train_and_test_samples.push_back({ 0.0,0.1,1 });
	train_and_test_samples.push_back({ 0.1,0.1,1 });
	train_and_test_samples.push_back({ 0.2,0.1,1 });
	train_and_test_samples.push_back({ 0.3,0.1,1 });
	train_and_test_samples.push_back({ 0.4,0.1,1 });

	train_and_test_samples.push_back({ 0.0,0.2,1 });
	train_and_test_samples.push_back({ 0.1,0.2,1 });
	train_and_test_samples.push_back({ 0.2,0.2,1 });
	train_and_test_samples.push_back({ 0.3,0.2,1 });
	train_and_test_samples.push_back({ 0.4,0.2,1 });

	train_and_test_samples.push_back({ 0.0,0.3,1 });
	train_and_test_samples.push_back({ 0.1,0.3,1 });
	train_and_test_samples.push_back({ 0.2,0.3,1 });
	train_and_test_samples.push_back({ 0.3,0.3,1 });
	train_and_test_samples.push_back({ 0.4,0.3,1 });

	train_and_test_samples.push_back({ 0.0,0.4,1 });
	train_and_test_samples.push_back({ 0.1,0.4,1 });
	train_and_test_samples.push_back({ 0.2,0.4,1 });
	train_and_test_samples.push_back({ 0.3,0.4,1 });
	train_and_test_samples.push_back({ 0.4,0.4,1 });

	//unten rechts 0
	train_and_test_samples.push_back({ 0.1,0.6,0 });
	train_and_test_samples.push_back({ 0.2,0.6,0 });
	train_and_test_samples.push_back({ 0.3,0.6,0 });
	train_and_test_samples.push_back({ 0.4,0.6,0 });
	train_and_test_samples.push_back({ 0.0,0.6,0 });

	train_and_test_samples.push_back({ 0.1,0.7,0 });
	train_and_test_samples.push_back({ 0.2,0.7,0 });
	train_and_test_samples.push_back({ 0.3,0.7,0 });
	train_and_test_samples.push_back({ 0.4,0.7,0 });
	train_and_test_samples.push_back({ 0.0,0.7,0 });

	train_and_test_samples.push_back({ 0.1,0.8,0 });
	train_and_test_samples.push_back({ 0.2,0.8,0 });
	train_and_test_samples.push_back({ 0.3,0.8,0 });
	train_and_test_samples.push_back({ 0.4,0.8,0 });
	train_and_test_samples.push_back({ 0.0,0.8,0 });

	train_and_test_samples.push_back({ 0.1,0.9,0 });
	train_and_test_samples.push_back({ 0.2,0.9,0 });
	train_and_test_samples.push_back({ 0.3,0.9,0 });
	train_and_test_samples.push_back({ 0.4,0.9,0 });
	train_and_test_samples.push_back({ 0.0,0.9,0 });

	train_and_test_samples.push_back({ 0.1,1.0,0 });
	train_and_test_samples.push_back({ 0.2,1.0,0 });
	train_and_test_samples.push_back({ 0.3,1.0,0 });
	train_and_test_samples.push_back({ 0.4,1.0,0 });
	train_and_test_samples.push_back({ 0.0,1.0,0 });

	//oben rechts 1
	train_and_test_samples.push_back({ 0.6,0.6,1 });
	train_and_test_samples.push_back({ 0.7,0.6,1 });
	train_and_test_samples.push_back({ 0.8,0.6,1 });
	train_and_test_samples.push_back({ 0.9,0.6,1 });
	train_and_test_samples.push_back({ 1.0,0.6,1 });

	train_and_test_samples.push_back({ 0.6,0.7,1 });
	train_and_test_samples.push_back({ 0.7,0.7,1 });
	train_and_test_samples.push_back({ 0.8,0.7,1 });
	train_and_test_samples.push_back({ 0.9,0.7,1 });
	train_and_test_samples.push_back({ 1.0,0.7,1 });

	train_and_test_samples.push_back({ 0.6,0.8,1 });
	train_and_test_samples.push_back({ 0.7,0.8,1 });
	train_and_test_samples.push_back({ 0.8,0.8,1 });
	train_and_test_samples.push_back({ 0.9,0.8,1 });
	train_and_test_samples.push_back({ 1.0,0.8,1 });

	train_and_test_samples.push_back({ 0.6,0.9,1 });
	train_and_test_samples.push_back({ 0.7,0.9,1 });
	train_and_test_samples.push_back({ 0.8,0.9,1 });
	train_and_test_samples.push_back({ 0.9,0.9,1 });
	train_and_test_samples.push_back({ 1.0,0.9,1 });

	train_and_test_samples.push_back({ 0.6,1.0,1 });
	train_and_test_samples.push_back({ 0.7,1.0,1 });
	train_and_test_samples.push_back({ 0.8,1.0,1 });
	train_and_test_samples.push_back({ 0.9,1.0,1 });
	train_and_test_samples.push_back({ 1.0,1.0,1 });

	//oben links 0
	train_and_test_samples.push_back({ 0.6,0.0,0 });
	train_and_test_samples.push_back({ 0.7,0.0,0 });
	train_and_test_samples.push_back({ 0.8,0.0,0 });
	train_and_test_samples.push_back({ 0.9,0.0,0 });
	train_and_test_samples.push_back({ 1.0,0.0,0 });

	train_and_test_samples.push_back({ 0.6,0.1,0 });
	train_and_test_samples.push_back({ 0.7,0.1,0 });
	train_and_test_samples.push_back({ 0.8,0.1,0 });
	train_and_test_samples.push_back({ 0.9,0.1,0 });
	train_and_test_samples.push_back({ 1.0,0.1,0 });

	train_and_test_samples.push_back({ 0.6,0.2,0 });
	train_and_test_samples.push_back({ 0.7,0.2,0 });
	train_and_test_samples.push_back({ 0.8,0.2,0 });
	train_and_test_samples.push_back({ 0.9,0.2,0 });
	train_and_test_samples.push_back({ 1.0,0.2,0 });

	train_and_test_samples.push_back({ 0.6,0.3,0 });
	train_and_test_samples.push_back({ 0.7,0.3,0 });
	train_and_test_samples.push_back({ 0.8,0.3,0 });
	train_and_test_samples.push_back({ 0.9,0.3,0 });
	train_and_test_samples.push_back({ 1.0,0.3,0 });

	train_and_test_samples.push_back({ 0.6,0.4,0 });
	train_and_test_samples.push_back({ 0.7,0.4,0 });
	train_and_test_samples.push_back({ 0.8,0.4,0 });
	train_and_test_samples.push_back({ 0.9,0.4,0 });
	train_and_test_samples.push_back({ 1.0,0.4,0 });
}

int main()
{
	srand(time(NULL)); // intialize rand
	const double lr = 0.05;	//learning rate
	createDataSet();
	//train_and_test_samples = { {1,1,1},{0,1,0},{1,0,0},{0,0,1} };
	//initiate weights and biases
	std::vector<std::vector<double>> weights0 = { { getRand(),getRand(),getRand() },{ getRand(),getRand(),getRand()} };
	std::vector<std::vector<double>> weights1 = { { getRand(),getRand(),getRand() },{ getRand(),getRand(),getRand()},{ getRand(),getRand(),getRand() } }; // from input layer to hidden layer 
	std::vector<std::vector<double>> weights2 = { { getRand()},{getRand() },{ getRand()} }; // from hidden layer to output layer
	std::vector<double> bias0 = { 1,1,1 };
	std::vector<double> bias1 = { 1,1,1 }; // biases for  the hidden layer
	std::vector<double> bias2 = {1 }; // bias for  the output layer
	
	
	for (int epoch = 0; epoch < 100000000; epoch++)
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
		if (epoch % 1000 == 0)
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
					std::vector<double> hidden_output0 = forwardProp(&weights0, &val_input, &bias0);
					std::vector<double> hidden_output1 = forwardProp(&weights1, &hidden_output0, &bias1);
					std::vector<double> output = forwardProp(&weights2, &hidden_output1, &bias2);
					int index = m_max(0,o[0] * 67);
					map += CHARMAP[index];
					map += ' ';
				}
				map += "\n";
			}
			//Sleep(300);
			system("cls");
			std::cout << "Epoch: "<<epoch<<"\nError: "<<(error/train_and_test_samples.size()) << "\n" << map ;
		}
	}
}