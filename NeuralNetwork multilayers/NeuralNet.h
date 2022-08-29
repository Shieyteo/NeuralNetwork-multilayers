#include "Activations.h"
#include <vector>
#include <tuple>
#include <iostream>

double derr(double a, double b)
{
	return 2 * (b - a); //derivative of (a-b)^2
}

class Net
{
public:
	std::vector<std::vector<std::vector<double>>> weights;
	std::vector<std::vector<double>> bias;
	std::vector<std::vector<double>> values;
	std::vector<std::vector<double>> errors;
	std::vector<unsigned int> topo;
	int layerNum;
	double lr;
	std::vector<void (*)(std::vector<double>&)> activations;
	std::vector<std::vector<double> (*)(std::vector<double>&)> derivative_activations;
public:
	Net(std::vector<std::tuple<unsigned int, Activation>> input, double learnrate)
		:
		layerNum(input.size()),
		lr(learnrate)
	{
		for (int i = 0; i < layerNum; i++)
		{
			topo.push_back(std::get<0>(input[i]));
			activations.push_back(std::get<1>(input[i]).act);
			derivative_activations.push_back(std::get<1>(input[i]).dact);
		}
		for (int i = 0; i < layerNum; i++)
		{
			values.push_back({});
			bias.push_back({});
			errors.push_back({});
			for (int _ = 0; _ < topo[i]; _++)
			{
				values[i].push_back(0);
				if (i != 0)
				{
					errors[i].push_back(0);
					bias[i].push_back(getRand());
				}
			}
		}
		for (int i = 0; i < layerNum -1; i++)
		{	
			weights.push_back({});
			for (int h = 0; h < topo[i]; h++)
			{
				weights[i].push_back({});
				for (int g = 0; g < topo[i+1]; g++)
				{
					weights[i][h].push_back(getRand());
				}
			}
		}
		std::cout << "Created Net sucessfuly\n";
	}
	std::vector<double> forawrd_prop(std::vector<double> input)
	{
		values[0]=input;
		for (int layer = 0; layer < layerNum -1; layer++)
		{
			for (int output = 0; output < values[layer+1].size(); output++)
			{
				double sum = bias[layer+1][output];
				for (int input = 0; input < topo[layer]; input++)
				{
					sum += weights[layer][input][output] * values[layer][input];
				}
				values[layer+1][output] = sum;
			}
			activations[layer + 1](values[layer + 1]);
		}
		return values[values.size() - 1];
	}

	void weightsupdate()
	{
		for (int layer = 0; layer < layerNum - 1; layer++)
		{
			for (int to = 0; to < topo[layer + 1]; to++)
			{
				bias[layer + 1][to] -= lr * errors[layer + 1][to];
				for (int from = 0; from < topo[layer]; from++)
				{
					weights[layer][from][to] -= lr * errors[layer + 1][to] * values[layer][from];
				}
			}
		}
	}

	/// Expects already forward proped network
	std::vector<std::vector<double>> back_prop(std::vector<double> expected,bool updateWeights = true)
	{
		std::vector<double> d_act = derivative_activations[layerNum - 1](values[layerNum - 1]);
		for (int i = 0; i < topo[layerNum -1]; i++)
		{
			double d_err = derr(expected[i], values[layerNum -1][i]);
			errors[layerNum -1][i] = d_err * d_act[i];
		}
		for (int layer = layerNum -2; layer > 0; layer--)
		{
			std::vector<double> d_activation = derivative_activations[layer + 1](values[layer]);
			for (int ErrorIndex = 0; ErrorIndex < topo[layer]; ErrorIndex++)
			{
				double sum_basic_error = 0;
				for (int prevErrorIndex = 0; prevErrorIndex < topo[layer+1]; prevErrorIndex++)
				{
					sum_basic_error += errors[layer + 1][prevErrorIndex]*weights[layer][ErrorIndex][prevErrorIndex];
				}
				errors[layer][ErrorIndex] = sum_basic_error * d_activation[ErrorIndex];
			}
		}
		if (updateWeights)
		{
			weightsupdate();
		}
		return errors;
	}
};