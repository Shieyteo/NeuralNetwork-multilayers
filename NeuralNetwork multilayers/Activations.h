#pragma once
#include <math.h>
#include <iostream>

static const double LRELUC = 0.25;

double non(double a) { std::cout << "Illegal \n"; return a; }

double getRand()
{
	return ((double(rand()) / RAND_MAX) * 2) - 1;
}

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

double elu(double inp)
{
	if (inp>=0)
	{
		return inp;
	}
	return exp(inp) - 1;
}

double derivative_elu(double in)
{
	if (in>=0)
	{
		return 0;
	}
	return exp(in);
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


class Activation
{
public:
	double (*act)(double);
	double (*dact)(double);
	Activation(double (*actFunc)(double), double (*dactFunc)(double))
		:
		act(actFunc),
		dact(dactFunc)
	{}
};

Activation TANH(tanh, derivative_tanh);
Activation SIGMOID(Sigmoid, derivative_Sigmoid);
Activation RELU(ReLU, derivative_ReLU);
Activation LEAKYRELU(LeakyReLU, derivative_LeakyReLU);
Activation NON(non, non);
Activation ELU(elu, derivative_elu);