#pragma once
#include <vector>

double _getRand()
{
	return ((double(rand()) / RAND_MAX) * 2) - 1;
}

class _Matrix
{
private:
	std::vector<std::vector<double>> data;

public:
	_Matrix(unsigned int height, unsigned int length, bool fillrandom = true)
	{
		data = std::vector<std::vector<double>>(height, std::vector<double>(length, 0));
		if (fillrandom)
		{
			for (int rows = 0; rows < data.size(); rows++)
			{
				for (int collumns = 0; collumns < data[rows].size(); collumns++)
				{
					data[rows][collumns] = _getRand();
				}
			}
		}
	}
	std::vector<double>* operator[](int num)
	{
		return &data.at(num);
	}
	_Matrix operator*(double scalar)
	{
		_Matrix copy = *this;
		for (int rows = 0; rows < data.size(); rows++)
		{
			for (int collumns = 0; collumns < data[rows].size(); collumns++)
			{
				copy.data[rows][collumns] *= scalar;
			}
		}
		return copy;
	}
	_Matrix operator+(double add)
	{
		_Matrix copy = *this;
		for (int rows = 0; rows < data.size(); rows++)
		{
			for (int collumns = 0; collumns < data[rows].size(); collumns++)
			{
				copy.data[rows][collumns] += add;
			}
		}
		return copy;
	}
	_Matrix operator+(_Matrix  add)
	{
		_Matrix copy = *this;
		if (data.size() != add.data.size() == data.size())
		{
			std::cout << "Not same size\n";
		}
		for (int rows = 0; rows < data.size(); rows++)
		{
			for (int collumns = 0; collumns < data[rows].size(); collumns++)
			{
				if (data[rows].size() != add.data[rows].size())
				{
					std::cout << "Not same size\n";
				}
				copy.data[rows][collumns] =  data[rows][collumns] + add.data[rows][collumns];
			}
		}
		return copy;
	}
	_Matrix operator<<(_Matrix multiplicator)
	{
		_Matrix firstsum(data.size(),multiplicator.data[0].size());
		for (int i = 0; i < data.size(); i++)
		{
			for (int j = 0; j < multiplicator.data[0].size(); j++)
			{
				double sum = 0;
				for (int k = 0; k < multiplicator.data.size(); k++)
				{
					sum += data[i][k] * multiplicator.data[k][j];
				}
				firstsum.data[i][j] = sum;
			}
		}
		return firstsum;
	}
	///applies function to matrix
	void apply(double (*func)(double))
	{
		for (int i = 0; i < data.size(); i++)
		{
			for (int j = 0; j < data[i].size(); j++)
			{
				data[i][j] = func(data[i][j]);
			}
		}
	}
	void print()
	{
		for (std::vector<double> row: data)
		{
			for (double item: row)
			{
				std::cout << item << ' ';
			}
			std::cout << '\n';
		}
	}
	_Matrix operator !()
	{
		_Matrix temp(data[0].size(), data.size());
		for (size_t i = 0; i < data.size(); i++)
		{
			for (size_t j = 0; j < data[i].size(); j++)
			{
				temp.data[j][i] = data[i][j];
			}
		}
		return temp;
	}
};