#include <iostream>
#include <vector>

double Sigmoid(double inp)
{
	return 1 / (1 + exp(-inp)); // returns sigmoid function applied to inp
}

double derivative_Error(double a, double b)
{
	return 2 * (b - a); //derivative of (a-b)^2
}

double derivative_Sigmoid(double b)
{
	return b * (1 - b); // b value already sigmoid applied
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
		ret.push_back(Sigmoid(sum)); // applies sigmoid function to summ of weights * values
	}
	return ret;	// takes in weights and values and multiplies them with eachother + adds the biases
}

// input at index 0 and 1 expected output at index 2
const std::vector<std::vector<double>> train_and_test_samples = { {1,0, 1},{1,1,0} ,{0,1,1} ,{0,0,0},}; //XOR Problem (Non linear)

int main()
{
	srand(time(NULL)); // intialize rand
	const double lr = 0.1;	//learning rate

	//initiate weights and biases
	std::vector<std::vector<double>> weights1 = { { getRand(),getRand() },{ getRand(),getRand()} }; // from input layer to hidden layer
	std::vector<std::vector<double>> weights2 = { { getRand()},{getRand() } }; // from hidden layer to output layer
	std::vector<double> bias1 = { 1,1 }; // biases for  the hidden layer
	std::vector<double> bias2 = { 1 }; // bias for  the output layer
	

	for (int epochs = 0; epochs < 100000; epochs++)
	{
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
			double d_sig = derivative_Sigmoid(output_layer_values[0]);
			double complete_error = d_err * d_sig;


			//Hiddenlayer error calculation										
			double d_err_hidden0 = complete_error * weights2[0][0];
			double d_err_hidden1 = complete_error * weights2[1][0];


			double d_sig_hidden0 = derivative_Sigmoid(hidden_layer_values[0]);
			double d_sig_hidden1 = derivative_Sigmoid(hidden_layer_values[1]);


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
		}
	}
	std::vector<std::vector<char>> map;
	int size = 20;
	for (float i = 0; i < size; i++)
	{
		map.push_back({});
		for (float j = 0; j < size; j++)
		{
			std::vector<double> val_input = { i/size,j/size};
			std::vector<double> hidden_output = forwardProp(&weights1, &val_input, &bias1);
			std::vector<double> output = forwardProp(&weights2, &hidden_output, &bias2);
			if (output[0]>=0.7)
			{
				map[i].push_back('$');
			}
			else if (output[0] <= 0.3)
			{
				map[i].push_back('.');
			}
			else
			{
				map[i].push_back('i');
			}
		}
	}
	std::cout << map.size() << " " << map[0].size() << "\n";
	for (std::vector<char> line: map)
	{
		for (char c : line)
		{
			std::cout << c<< " ";
		}
		std::cout<<"\n";
	}
}