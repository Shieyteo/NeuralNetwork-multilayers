#include "Activations.h"
#include <vector>
#include <tuple>
#include <iostream>

double nderr(double a, double b)
{
	return 2 * (b - a); //derivative of (a-b)^2
}

class Network
{
public:
	double*** weights;
	double** bias;
	double** values;
	double** errors;
	std::vector<unsigned int> topo;
	double lr;
	std::vector<double (*)(double)> activations;
	std::vector<double (*)(double)> derivative_activations;
public:
	Network(std::vector<std::tuple<unsigned int, Activation>> input, double learnrate)
		:
		lr(learnrate)
	{
		for (int i = 0; i < input.size(); i++)
		{
			topo.push_back(std::get<0>(input[i]));
			activations.push_back(std::get<1>(input[i]).act);
			derivative_activations.push_back(std::get<1>(input[i]).dact);
		}
		values = new double* [topo.size()];
		bias = new double* [topo.size()];
		errors = new double* [topo.size()];
		for (int i = 0; i < topo.size(); i++)
		{
			values[i] = new double[topo[i]];
			bias[i] = new double[topo[i]];
			errors[i] = new double[topo[i]];
			for (int j = 0; j < topo[i]; j++)
			{
				bias[i][j] = getRand();
			}
		}
		weights = new double** [topo.size()];
		for (int i = 0; i < topo.size()-1; i++)
		{
			weights[i] = new double* [topo[i]];
			for (int j = 0; j < topo[i]; j++)
			{
				weights[i][j] = new double[topo[i + 1]];
				for (int k = 0; k < topo[i+1]; k++)
				{
					weights[i][j][k] = getRand();
				}
			}
		}
		std::cout << "Created Net sucessfuly\n";
	}
	double* forawrd_prop(double* input)
	{
		values[0] = input;
		for (int layer = 0; layer < topo.size() - 1; layer++)
		{
			for (int output = 0; output < topo[layer+1]; output++)
			{
				double sum = bias[layer][output];
				for (int input = 0; input < topo[layer]; input++)
				{
					sum += weights[layer][input][output] * values[layer][input];
				}
				values[layer + 1][output] = activations[layer + 1](sum);
			}
		}
		return values[topo.size() - 1];
	}

	/// Expects already forward proped network
	void back_prop(std::vector<double> expected)
	{
		for (int i = 0; i < topo[topo.size() - 1]; i++)
		{
			double d_err = derr(expected[i], values[values.size() - 1][i]);
			double d_act = derivative_activations[topo.size() - 1](values[values.size() - 1][i]);
			errors[topo.size() - 1][i] = d_err * d_act;
		}
		for (int layer = values.size() - 2; layer > 0; layer--)
		{
			for (int ErrorIndex = 0; ErrorIndex < topo[layer]; ErrorIndex++)
			{
				double sum_basic_error = 0;
				for (int prevErrorIndex = 0; prevErrorIndex < topo[layer + 1]; prevErrorIndex++)
				{
					sum_basic_error += errors[layer + 1][prevErrorIndex] * weights[layer][ErrorIndex][prevErrorIndex];
				}
				double d_activation = derivative_activations[layer + 1](values[layer][ErrorIndex]);
				errors[layer][ErrorIndex] = sum_basic_error * d_activation;
			}
		}
		for (int layer = 0; layer < topo.size() - 1; layer++)
		{
			for (int to = 0; to < topo[layer + 1]; to++)
			{
				bias[layer][to] -= lr * errors[layer + 1][to];
				for (int from = 0; from < topo[layer]; from++)
				{
					weights[layer][from][to] -= lr * errors[layer + 1][to] * values[layer][from];
				}
			}
		}
	}
	*/
};