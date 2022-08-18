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
	
	return 0;
}

