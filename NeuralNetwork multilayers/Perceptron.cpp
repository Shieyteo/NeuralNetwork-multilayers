#include <iostream>
#include <vector>

double ReLU_P(double in)
{
	if (in > 0) return in;
	return 0;
}

double Sigmoid_P(double in)
{
	return 1 / (1 + exp(-in));
}

double wti_P(std::vector<double> weights, std::vector<double> input, double bias)
{
	double sum = bias;
	for (int i = 0; i < weights.size(); i++)
	{
		sum += weights[i] * input[i];
	}
	return sum;
}

//b = output
//a = wanted

double sqError_P(double a, double b)
{
	return (a - b) * (a - b);
}

double derivative_Error_P(double a, double b)
{
	return 2 * (b - a);
}

double derivative_Sigmoid_P(double b)
{
	return b * (1 - b);
}

int main_P()
{
	srand(time(NULL));
	rand(); rand(); rand();
	const double lr = 0.1;
	std::vector<double> weights = { double(rand()) / RAND_MAX - 0.5,double(rand()) / RAND_MAX - 0.5 };
	double bias = 1;

	std::vector<std::vector<double>> samples = { {1,0, 0},{1,1,1} ,{0,1,0} ,{0,0,1} };

	for (int j = 0; j < 100000; j++)
	{
		for (std::vector<double> sample : samples)
		{
			double expected = sample[2];
			std::vector<double> input = { sample[0],sample[1] };
			double b_output = wti_P(weights, input, bias);
			double output = Sigmoid_P(b_output);
			for (int i = 0; i < input.size(); i++)
			{
				double a = derivative_Error_P(expected, output);
				double b = derivative_Sigmoid_P(output);
				weights[i] -= lr * a * b * input[i];
			}
			double a = derivative_Error_P(expected, output);
			double b = derivative_Sigmoid_P(output);
			bias -= lr * a * b;
		}

	}
	std::cout << Sigmoid_P(wti_P(weights, { 1,1 }, bias)) << "\n";
	std::cout << Sigmoid_P(wti_P(weights, { 1,0 }, bias)) << "\n";
	std::cout << Sigmoid_P(wti_P(weights, { 0,0 }, bias)) << "\n";
	std::cout << Sigmoid_P(wti_P(weights, { 0,1 }, bias)) << "\n";
	return 0;
}

