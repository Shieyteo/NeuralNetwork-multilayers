#include "_matrix.h"
#include <iostream>

double derr(double a, double b)
{
	return 2 * (b - a); //derivative of (a-b)^2
}

class Net
{
	std::vector<std::vector<std::vector<double>>> weights;
	std::vector<std::vector<double>> bias;
	std::vector<std::vector<double>> values;
	std::vector<std::vector<double>> errors;
	std::vector<unsigned int> topo;
	double lr;
	double (*activation)(double);
	double (*derivative_activation)(double);

public:
	Net(std::vector<unsigned int> topology, double learnrate, double (*activationFunc)(double), double (*derivative_activationFunc)(double))
		:
		lr(learnrate),
		topo(topology),
		activation(activationFunc),
		derivative_activation(derivative_activationFunc)
	{
		for (int i = 0; i < topology.size(); i++)
		{
			values.push_back({});
			bias.push_back({});
			errors.push_back({});
			for (int _ = 0; _ < topology[i]; _++)
			{
				values[i].push_back(0);
				if (i != 0)
				{
					errors[i].push_back(0);
					bias[i].push_back(_getRand());
				}
			}
		}
		for (int i = 0; i < topology.size()-1; i++)
		{	
			weights.push_back({});
			for (int h = 0; h < topology[i]; h++)
			{
				weights[i].push_back({});
				for (int g = 0; g < topology[i+1]; g++)
				{
					weights[i][h].push_back(_getRand());
				}
			}
		}
		std::cout << "Created Net sucessfuly\n";
	}
	std::vector<double> forwars_prop(std::vector<double> input)
	{
		if (input.size()!= topo[0])
		{
			std::cout << "Non valid input got " << input.size() << "  expected " << values[0].size() << "\n";
		}
		for (int layer = 0; layer < topo.size()-1; layer++)
		{
			for (int output = 0; output < values[layer+1].size(); output++)
			{
				double sum = bias[layer + 1][output];
				for (int input = 0; input < topo[layer]; input++)
				{
					sum += weights[layer][input][output] * values[layer][input];
				}
				//ret.push_back(Sigmoid(sum));
				values[layer+1][output] = activation(sum);
			}
		}
		return values[values.size() - 1];
	}

	/// Expects already forward proped network
	void back_prop(std::vector<double> expected)
	{
		for (int i = 0; i < topo[topo.size()-1]; i++)
		{
			double d_err = derr(expected[i], values[values.size()-1][i]);
			double d_act = derivative_activation(values[values.size() - 1][i]);
			errors[topo.size()-1][i] = d_err * d_act;
		}
		for (int layer = values.size() -2; layer > 0; layer--)
		{
			for (int ErrorIndex = 0; ErrorIndex < topo[layer]; ErrorIndex++)
			{
				double sum_basic_error = 0;
				for (int prevErrorIndex = 0; prevErrorIndex < topo[layer+1]; prevErrorIndex++)
				{
					sum_basic_error += errors[layer + 1][prevErrorIndex]*weights[layer][ErrorIndex][prevErrorIndex];
				}
				double d_activation = derivative_activation(values[layer][ErrorIndex]);
				errors[layer][ErrorIndex] = sum_basic_error * d_activation;
			}
		}
		for (int layer = 1; layer < topo.size()-1; layer++)
		{
			for (int to = 0; to < topo[layer + 1]; to++)
			{
				bias[layer][to] -= lr + errors[layer][to];
				for (int from = 0; from < topo[layer]; from++)
				{
					weights[layer-1][from][to] -= lr * errors[layer][to] * values[layer][to];
				}
			}
		}
		//Weight adjustment
		//bias[]
		//bias2[0] -= lr * complete_error20;

		////weight adjustment
		//weights2[0][0] -= lr * complete_error20 * hidden_layer_values1[0];
		//weights2[1][0] -= lr * complete_error20 * hidden_layer_values1[1];

		////bias adjustment
		//bias1[0] -= lr * complete_error10;
		//bias1[1] -= lr * complete_error11;
		//bias1[2] -= lr * complete_error12;

		////weight adjustment
		//weights1[0][0] -= lr * complete_error10 * hidden_layer_values0[0];
		//weights1[1][0] -= lr * complete_error10 * hidden_layer_values0[1];
		//weights1[2][0] -= lr * complete_error10 * hidden_layer_values0[2];
		//weights1[0][1] -= lr * complete_error11 * hidden_layer_values0[0];
		//weights1[1][1] -= lr * complete_error11 * hidden_layer_values0[1];
		//weights1[2][1] -= lr * complete_error11 * hidden_layer_values0[2];
		//weights1[0][2] -= lr * complete_error12 * hidden_layer_values0[0];
		//weights1[1][2] -= lr * complete_error12 * hidden_layer_values0[1];
		//weights1[2][2] -= lr * complete_error12 * hidden_layer_values0[2];

		////bias adjustment
		//bias0[0] -= lr * complete_error00;
		//bias0[1] -= lr * complete_error01;
		//bias0[2] -= lr * complete_error02;

		////weight adjustment
		//weights0[0][0] -= lr * complete_error00 * input[0];
		//weights0[1][0] -= lr * complete_error00 * input[1];
		//weights0[0][1] -= lr * complete_error01 * input[0];
		//weights0[1][1] -= lr * complete_error01 * input[1];
		//weights0[0][2] -= lr * complete_error02 * input[0];
		//weights0[1][2] -= lr * complete_error02 * input[1];
	}
};