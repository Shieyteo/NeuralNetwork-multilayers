#include <iostream>
#include <vector>

double ReLU(double in)
{
	if (in > 0) return in;
	return 0;
}

double Sigmoid(double in)
{
	return 1 / (1 + exp(-in));
}

double wti(std::vector<double> weights, std::vector<double> input, double bias)
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

double sqError(double a, double b)
{
	return (a - b) * (a - b);
}

double derivative_Error(double a, double b)
{
	return 2 * (b - a);
}

double derivative_Sigmoid(double b)
{
	return b * (1 - b);
}

int main()
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
			double b_output = wti(weights, input, bias);
			double output = Sigmoid(b_output);
			for (int i = 0; i < input.size(); i++)
			{
				double a = derivative_Error(expected, output);
				double b = derivative_Sigmoid(output);
				weights[i] -= lr * a * b * input[i];
			}
			double a = derivative_Error(expected, output);
			double b = derivative_Sigmoid(output);
			bias -= lr * a * b;
		}

	}
	std::cout << Sigmoid(wti(weights, { 1,1 }, bias)) << "\n";
	std::cout << Sigmoid(wti(weights, { 1,0 }, bias)) << "\n";
	std::cout << Sigmoid(wti(weights, { 0,0 }, bias)) << "\n";
	std::cout << Sigmoid(wti(weights, { 0,1 }, bias)) << "\n";
	return 0;
}

