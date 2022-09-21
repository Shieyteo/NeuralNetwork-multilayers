#pragma once
#include <math.h>
#include <vector>
#include <string>

static const double LRELUC = 0.25;

// see some activation functions at https://www.geogebra.org/m/mszfwq6a

// random number between -1 and 1 for weights and biases
double getRand()
{
	return ((double(rand()) / RAND_MAX) * 2) - 1;
}

// returns bigger number
double m_max(double a, double b)
{
	return a > b ? a : b;
}

// returns smaller number
double m_min(double a, double b)
{
	return a < b ? a : b;
}

// non function for activation
void nona(std::vector<double>& a) {}

// non function for derivative
std::vector<double> nond(std::vector<double>& a) 
{ 
	return std::vector<double>(a.size(),1);  // derivative for standart linear function is 1
}

// applies sigmoid to inp vector
void Sigmoid(std::vector<double>& inp)
{
	for (int i = 0; i < inp.size(); i++)
	{
		inp[i] = 1 / (1 + exp(-inp[i]));
	}
}

// applies derivative of sigmoid to inp vector
std::vector<double> dSigmoid(std::vector<double>& inp)
{
	std::vector<double> ret = std::vector<double>(inp.size());
	for (int i = 0; i < inp.size(); i++)
	{
		ret[i] = inp[i] * (1 - inp[i]);
	}
	return ret;
}

// applies tanh to vector
void Tanh(std::vector<double>& inp)
{
	for (int i = 0; i < inp.size(); i++)
	{
		inp[i] = tanh(inp[i]);
	}
}

// applies derivative of tanh to vector
std::vector<double> dTanh(std::vector<double>& inp)
{
	std::vector<double> ret = std::vector<double>(inp.size());
	for (int i = 0; i < inp.size(); i++)
	{
		double e = tanh(inp[i]);
		ret[i] = 1 - e*e;
	}
	return ret;
}

// applies relu activation to vector
void ReLU(std::vector<double>& inp)
{
	for (int i = 0; i < inp.size(); i++)
	{
		inp[i] = m_max(0, inp[i]);
	}
}

// applies relu derivative to vector
std::vector<double> dReLU(std::vector<double>& inp)
{
	std::vector<double> ret;
	for (int i = 0; i < inp.size(); i++)
	{
		ret.push_back(inp[i] <= 0 ? 0 : 1);
	}
	return ret;
}

// applies softmax to inp (expirimental)
void Softmax(std::vector<double>& inp)
{
	double sum = 0;
	for (int i = 0; i < inp.size(); i++)
	{
		sum += exp(inp[i]);
	}
	for (int i = 0; i < inp.size(); i++)
	{
		inp[i] = exp(inp[i])/sum;
	}
}

// applies elu
double elu(double inp)
{
	if (inp>=0)
	{
		return inp;
	}
	return exp(inp) - 1;
}

//applies d_elu
double derivative_elu(double in)
{
	if (in>=0)
	{
		return 0;
	}
	return exp(in);
}

// applies leakyReLU
double LeakyReLU(double in)
{
	return in >= 0 ? in : in * LRELUC;
}

// applies leakydReLU
double derivative_LeakyReLU(double in)
{
	return in >= 0 ? 0 : LRELUC;
}

class Activation
{
public:
	static std::vector<Activation*> allActivations;
	void (*act)(std::vector<double>&);
	std::vector<double>(*dact)(std::vector<double>&);
	std::string name;
	Activation(void (*actFunc)(std::vector<double>&), std::vector<double>(*dactFunc)(std::vector<double>&),std::string name)
		:
		act(actFunc),
		dact(dactFunc),
		name(name)
	{
		allActivations.push_back(this);
	}
	static Activation* search(std::string searching)
	{
		for (int i = 0; i < Activation::allActivations.size(); i++)
		{
			if (Activation::allActivations[i]->name == searching)
			{
				return Activation::allActivations[i];
			}
		}
		std::cout << "Activation not found see Activation.h for more\n";
		return nullptr;
	}
};
std::vector<Activation*> Activation::allActivations = {};

//available activation functions
Activation TANH(Tanh, dTanh,"tanh");
Activation SIGMOID(Sigmoid, dSigmoid,"sigmoid");
Activation RELU(ReLU, dReLU,"relu");
Activation NON(nona, nond,"non");
//Activation SOFTMAX(Softmax, dSoftmax);
/*
Activation LEAKYRELU(LeakyReLU, derivative_LeakyReLU);
Activation NON(non, non);
Activation ELU(elu, derivative_elu);
*/
