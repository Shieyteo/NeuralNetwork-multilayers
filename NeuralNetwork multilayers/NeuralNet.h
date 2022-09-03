#include "Activations.h"
#include <vector>
#include <fstream>
#include <tuple>
#include <iostream>

double derr(double a, double b)
{
	return 2 * (b - a); //derivative of (a-b)^2
}

std::vector<double> split(std::string str)
{
	char deli = ' ';
	size_t pos = 0;
	std::string token;
	std::vector<double> out;
	while ((pos = str.find(deli)) != std::string::npos) {
		token = str.substr(0, pos);
		out.push_back(atof(token.c_str()));
		str.erase(0, pos + 1);
	}
	//out.push_back(atof(str.c_str()));
	return out;
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
	std::vector<Activation*> activations;
public:
	Net(std::vector<std::tuple<unsigned int, std::string>> input, double learnrate)
		:
		layerNum(input.size()),
		lr(learnrate)
	{
		for (int i = 0; i < layerNum; i++)
		{
			topo.push_back(std::get<0>(input[i]));
			activations.push_back(Activation::search(std::get<1>(input[i])));
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
			activations[layer + 1]->act(values[layer + 1]);
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
		std::vector<double> d_act = activations[layerNum - 1]->dact(values[layerNum - 1]); // calculatin error for outpu layer
		for (int i = 0; i < topo[layerNum -1]; i++)
		{
			double d_err = derr(expected[i], values[layerNum -1][i]);
			errors[layerNum -1][i] = d_err * d_act[i];
		}
		for (int layer = layerNum -2; layer > 0; layer--)	// calaculating error for the rest
		{
			std::vector<double> d_activation = activations[layer + 1]->dact(values[layer]);
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
			weightsupdate(); //adjust weights
		}
		return errors;
	}
	void save(const std::string path)
	{
		std::ofstream file;
		file.open(path);
		file << "topo & activation\n";
		for (int i = 0; i < topo.size(); i++)
		{
			file << topo[i] << "|"<<activations[i]->name<<"\n";
		}
		file << "lr\n" << lr << '\n';
		file << "weights\n";
		for (int i = 0; i < layerNum - 1; i++)
		{
			for (int h = 0; h < topo[i]; h++)
			{
				for (int g = 0; g < topo[i + 1]; g++)
				{
					file << weights[i][h][g] << " ";
				}
				file << "\n";
			}
		}
		file << "bias\n";
		for (int i = 1; i < layerNum; i++)
		{
			for (int j = 0; j < topo[i]; j++)
			{
				file << bias[i][j] << " ";
			}
			file << "\n";
		}
		file.close();
	}
	static Net load(std::string path)
	{
		std::ifstream file;
		file.open(path);
		std::string buffer;

		//Topologz + Activaitions
		std::getline(file,buffer);
		if (buffer != "topo & activation")
		{
			std::cout << "Error loading file\n";
			return Net({}, 0);
		}
		std::vector<std::tuple<unsigned int, std::string>> passin;
		while (std::getline(file, buffer))
		{
			if (buffer == "lr")
				break;
			
			int in = buffer.find('|');
			std::string newStr;
			newStr = buffer.substr(0, in);
			buffer = buffer.substr(in+1);
			passin.push_back({ atoi(newStr.c_str()),buffer});
		}
		//learning rate
		std::getline(file, buffer);
		double lr = atof(buffer.c_str());
		Net retNet(passin,lr);
		//weights
		std::getline(file, buffer);
		if (buffer!="weights")
		{
			std::cout << "Error loading file\n";
			return retNet;
		}
		
		for (int i = 0; i < retNet.layerNum - 1; i++)
		{
			for (int h = 0; h < retNet.topo[i]; h++)
			{
				std::getline(file, buffer);
				retNet.weights[i][h] = split(buffer);
			}
		}
		//bias
		std::getline(file, buffer);
		if (buffer != "bias") {
			std::cout << "Error opening the file\n";
			return retNet;
		}
		for (int layer = 0; layer < retNet.layerNum - 1; layer++)
		{
			std::getline(file, buffer);
			retNet.bias[layer + 1] = split(buffer);
		}
		return retNet;
	}
};