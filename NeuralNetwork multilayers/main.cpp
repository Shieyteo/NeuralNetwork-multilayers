#include <iostream>
#include <vector>
#include <string>

double Sigmoid(double inp)
{
	return 1 / (1 + exp(-inp));
}


double derivative_Error(double a, double b)
{
	return 2 * (b - a); //derivative of (a-b)^2
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
		ret.push_back(tanh(sum));
	}
	return ret;	// takes in weights and values and multiplies them with eachother + adds the biases
}

// input at index 0 and 1 expected output at index 2
const std::vector<std::vector<double>> train_and_test_samples = { {1,0, 1},{1,1,0} ,{0,1,1} ,{0,0,0}}; //XOR Problem (Non linear)

const char* CHARMAP = "@$B%8&WM#oahkbdpqwmZO0QLCJUYXzcvunxrjftlI/|()1{}[]<>i!?*+~-;:,_^`'.";

int main()
{
	srand(time(NULL)); // intialize rand
	const double lr = 0.2;	//learning rate

	//initiate weights and biases
	std::vector<std::vector<double>> weights1 = { { getRand(),getRand() },{ getRand(),getRand()} }; // from input layer to hidden layer ,{ getRand(),getRand()}
	std::vector<std::vector<double>> weights2 = { { getRand()},{getRand() } }; // from hidden layer to output layer ,{ getRand()}
	std::vector<double> bias1 = { 1,1,1 }; // biases for  the hidden layer
	std::vector<double> bias2 = { 1 }; // bias for  the output layer
	

	for (int epoch = 0; epoch < 100000; epoch++)
	{
		double error = 0;
		for (std::vector<double> sample : train_and_test_samples)
		{
			//initalize train data
			std::vector<double> input = { sample[0],sample[1] };
			double expected = sample[2];

			//forward propagation
			std::vector<double> hidden_layer_values = forwardProp(&weights1, &input, &bias1);
			std::vector<double> output_layer_values = forwardProp(&weights2, &hidden_layer_values, &bias2); // in this senario just one value

			//BACKPROP

			//Outputlayer error calculation	
			double d_err = derivative_Error(expected, output_layer_values[0]);
			//double d_sig = derivative_Sigmoid(output_layer_values[0]);
			double d_sig = derivative_tanh(output_layer_values[0]);
			double complete_error = d_err * d_sig;


			//Hiddenlayer error calculation										
			double d_err_hidden0 = complete_error * weights2[0][0];
			double d_err_hidden1 = complete_error * weights2[1][0];
			//double d_err_hidden2 = complete_error * weights2[2][0];


			//double d_sig_hidden0 = derivative_Sigmoid(hidden_layer_values[0]);
			//double d_sig_hidden1 = derivative_Sigmoid(hidden_layer_values[1]);
			double d_sig_hidden0 = derivative_tanh(hidden_layer_values[0]);
			double d_sig_hidden1 = derivative_tanh(hidden_layer_values[1]);
			//double d_sig_hidden2 = derivative_tanh(hidden_layer_values[2]);

			//bias adjustment
			bias2[0] -= lr * complete_error;

			//weight adjustment
			weights2[0][0] -= lr * complete_error * input[0];
			weights2[1][0] -= lr * complete_error * input[1];

			//bias adjustment
			bias1[0] -= lr * d_err_hidden0 * d_sig_hidden0;
			bias1[1] -= lr * d_err_hidden1 * d_sig_hidden1;

			//weight adjustment
			weights1[0][0] -= lr * d_err_hidden0 * d_sig_hidden0 * input[0];
			weights1[1][0] -= lr * d_err_hidden0 * d_sig_hidden0 * input[1];
			weights1[0][1] -= lr * d_err_hidden1 * d_sig_hidden1 * input[0];
			weights1[1][1] -= lr * d_err_hidden1 * d_sig_hidden1 * input[1];
			error += abs(expected - output_layer_values[0]);
		}
		std::string map;
		int size = 30;
		
		if (epoch % 100 == 0)
		{
			for (float i = 0; i < size; i++)
			{
				for (float j = 0; j < size; j++)
				{
					std::vector<double> val_input = { i / size,j / size };
					std::vector<double> hidden_output = forwardProp(&weights1, &val_input, &bias1);
					std::vector<double> output = forwardProp(&weights2, &hidden_output, &bias2);
					map += CHARMAP[int(output[0] * 67)];
					map += ' ';
				}
				map += "\n";
			}
			system("cls");
			std::cout << "Epoch: "<<epoch<<"\nError: "<<(error / 4) << "\n" << map;
		}
	}
}