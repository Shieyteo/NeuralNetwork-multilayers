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
	_Matrix(unsigned int length, unsigned int height, bool fillrandom = true)
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
				add[rows][collumns] +=  data[rows][collumns] + add.data[rows][collumns];
			}
		}
	}
};