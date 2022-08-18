#ifndef MATRIX_H
#define MATRIX_H
#pragma warning(disable : 4996)
#define MAXCHAR 10000
#include <vector>
#include <iostream>

class Matrix {
public:
	int label;
	std::vector<std::vector<double>> data;
	Matrix(std::vector<std::vector<double>> d_data = {});
	void create(int rows, int cols);
	Matrix apply(double (*func)(double));
	void print();
	void printl();
	void fill(double num);
	void save(char* file_string);
	void load(char* file_string);
	void randomize(int n);
	int argmax();
	Matrix flatten(int axis);
	Matrix operator *(const Matrix& mul);
	Matrix operator *(const double mul);
	Matrix operator <<(const Matrix& sec);
	Matrix operator +(const Matrix& add);
	Matrix operator +(const double add);
	Matrix operator -(const Matrix& sub);
	double operator %(const Matrix& dp);
	Matrix operator !();
};

double uniform_distribution(double low, double high);
int check_dims(Matrix m1, Matrix m2);
std::vector<Matrix> csv_to_img(char* file_string, int number_of_imgs);
double sigmoid(double input);
Matrix sigmoidPrime(Matrix m);
Matrix softmax(Matrix m);


#endif