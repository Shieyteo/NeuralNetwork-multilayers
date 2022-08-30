#pragma once
#include <math.h>
#include <vector>
#include <string>

static const double LRELUC = 0.25;

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

void nona(std::vector<double>& a) {}

std::vector<double> nond(std::vector<double>& a) 
{ 
	return std::vector<double>(a.size(),1); 
}

void Sigmoid(std::vector<double>& inp)
{
	for (int i = 0; i < inp.size(); i++)
	{
		inp[i] = 1 / (1 + exp(-inp[i]));
	}
}

std::vector<double> dSigmoid(std::vector<double>& inp)
{
	std::vector<double> ret = std::vector<double>(inp.size());
	for (int i = 0; i < inp.size(); i++)
	{
		ret[i] = inp[i] * (1 - inp[i]);
	}
	return ret;
}

void Tanh(std::vector<double>& inp)
{
	for (int i = 0; i < inp.size(); i++)
	{
		inp[i] = tanh(inp[i]);
	}
}

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

void ReLU(std::vector<double>& inp)
{
	for (int i = 0; i < inp.size(); i++)
	{
		inp[i] = m_max(0, inp[i]);
	}
}

std::vector<double> dReLU(std::vector<double>& inp)
{
	std::vector<double> ret;
	for (int i = 0; i < inp.size(); i++)
	{
		ret.push_back(inp[i] <= 0 ? 0 : 1);
	}
	return ret;
}

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

//double Sigmoid(double inp)
//{
//	return 1 / (1 + exp(-inp));
//}

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
		return nullptr;
	}
};
std::vector<Activation*> Activation::allActivations = {};


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
