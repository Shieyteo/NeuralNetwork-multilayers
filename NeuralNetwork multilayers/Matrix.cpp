#include <C:\Users\Matteo\source\repos\NN my self\NN my self\matrix.h>
#pragma warning(disable : 4996)

Matrix::Matrix(std::vector<std::vector<double>> d_data)
{
    data = d_data;
}

Matrix Matrix::apply(double (*func)(double))
{
    Matrix temp;
    for (int i = 0; i < data.size(); i++)
    {
        temp.data.push_back({});
        for (int j = 0; j < data[i].size(); j++)
        {
            temp.data[i].push_back((*func)(data[i][j]));
        }
    }
    return temp;
}

void Matrix::print()
{
    for (size_t i = 0; i < data.size(); i++)
    {
        for (size_t ji = 0; ji < data[i].size(); ji++)
        {
            printf("%.3f ", data[i][ji]);
        }
        std::cout << std::endl;
    }
    return;
}

void Matrix::printl()
{
    print();
    std::cout << "Img Label: " << label << "\n";
}

void Matrix::fill(double num)
{
    for (int i = 0; i < data.size(); i++)
    {
        for (int j = 0; j < data[i].size(); j++)
        {
            data[i][j] = num;
        }
    }
}

void Matrix::save(char* file_string)
{
    this;
    FILE* file = fopen(file_string, "w");
    fprintf(file, "%d\n", data.size());
    fprintf(file, "%d\n", data[0].size());
    for (int i = 0; i < data.size(); i++)
    {
        for (int j = 0; j < data[0].size(); j++)
        {
            fprintf(file, "%.6f\n", data[i][j]);
        }
    }
    std::cout << "Succesfully saved matrix to " << file_string << "\n";
    fclose(file);
}

void Matrix::load(char* file_string)
{
    FILE* file = fopen(file_string, "r");
    char entry[MAXCHAR];
    fgets(entry, MAXCHAR, file);
    int rows = atoi(entry);
    fgets(entry, MAXCHAR, file);
    int cols = atoi(entry);
    Matrix s;
    s.create(rows, cols);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            fgets(entry, MAXCHAR, file);
            s.data[i][j] = strtod(entry, NULL);
        }
    }
    std::cout << "Succesfully loaded matrix from " << file_string << "\n";
    this->data = s.data;
    fclose(file);
}

void Matrix::create(int rows, int cols)
{
    data = {};
    for (int i = 0; i < rows; i++)
    {
        data.push_back({});
        for (int j = 0; j < cols; j++)
        {
            data[i].push_back({});
        }
    }
}

void Matrix::randomize(int n)
{
    double min = -1.0 / sqrt(n);
    double max = 1.0 / sqrt(n);
    for (int i = 0; i < data.size(); i++)
    {
        for (int j = 0; j < data[i].size(); j++)
        {
            data[i][j] = uniform_distribution(min, max);
        }
    }
}

int Matrix::argmax()
{
    double max_score = 0;
    int max_idx = 0;
    for (int i = 0; i < data.size(); i++)
    {
        if (data[i][0] > max_score)
        {
            max_score = data[i][0];
            max_idx = i;
        }
    }
    return max_idx;
}

Matrix Matrix::flatten(int axis)
{
    Matrix temp;
    if (axis == 0)
    {
        temp.create(data.size() * data[0].size(), 1);
    }
    else if (axis == 1)
    {
        temp.create(1, data.size() * data[0].size());
    }
    else
    {
        std::cout << "MATRIX ERROR Argumet between 0 or 1";
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < data.size(); i++)
    {
        if (i % 5 == 0);
        for (int j = 0; j < data[0].size(); j++)
        {

            if (axis == 0) temp.data[i * data[0].size() + j][0] = data[i][j];
            else if (axis == 1) temp.data[0][i * data[0].size() + j] = data[i][j];
        }
    }
    return temp;
}

Matrix Matrix::operator *(const Matrix& mul)
{
    Matrix temp;
    for (int i = 0; i < data.size(); i++)
    {
        temp.data.push_back({});
        for (int j = 0; j < data[i].size(); j++)
        {
            temp.data[i].push_back(data[i][j] * mul.data[i][j]);
        }
    }
    return temp;
}

Matrix Matrix::operator *(const double mul)
{
    Matrix temp;
    for (int i = 0; i < data.size(); i++)
    {
        temp.data.push_back({});
        for (int j = 0; j < data[i].size(); j++)
        {
            temp.data[i].push_back(data[i][j] * mul);
        }
    }
    return temp;
}

double Matrix::operator %(const Matrix& dp)
{
    double sum = 0;
    for (size_t i = 0; i < data[0].size(); i++)
    {
        sum += data[0][i] * dp.data[0][i];
    }
    return sum;
}

Matrix Matrix::operator <<(const Matrix& sec)
{
    Matrix firstsum;
    firstsum.create(data.size(), sec.data[0].size());
    for (int i = 0; i < data.size(); i++)
    {
        for (int j = 0; j < sec.data[0].size(); j++)
        {
            double sum = 0;
            for (int k = 0; k < sec.data.size(); k++)
            {
                sum += data[i][k] * sec.data[k][j];
            }
            firstsum.data[i][j] = sum;
        }
    }
    return firstsum;
}

Matrix Matrix::operator +(const Matrix& add)
{
    Matrix NEW;
    if (data.size() != add.data.size())
        throw "Not same size";
    for (int i = 0; i < data.size(); i++)
    {
        NEW.data.push_back({});
        if (data[i].size() != add.data[i].size())
            throw "Not same size";
        for (int x = 0; x < data[i].size(); x++)
        {
            NEW.data[i].push_back(data[i][x] + add.data[i][x]);
        }
    }
    return NEW;
}

Matrix Matrix::operator +(const double add)
{
    Matrix NEW;
    for (int i = 0; i < data.size(); i++)
    {
        NEW.data.push_back({});
        for (int x = 0; x < data[i].size(); x++)
        {
            NEW.data[i].push_back(data[i][x] + add);
        }
    }
    return NEW;
}

Matrix Matrix::operator -(const Matrix& sub)
{
    Matrix NEW;
    if (data.size() != sub.data.size())
        std::cout << "Not same size";
    for (int i = 0; i < data.size(); i++)
    {
        NEW.data.push_back({});
        if (data[i].size() != sub.data[i].size())
            std::cout << "Not same size";
        for (int x = 0; x < data[i].size(); x++)
        {
            NEW.data[i].push_back(data[i][x] - sub.data[i][x]);
        }
    }
    return NEW;
}

Matrix Matrix::operator !()
{
    Matrix temp;
    temp.create(data[0].size(), data.size());
    for (size_t i = 0; i < data.size(); i++)
    {
        for (size_t j = 0; j < data[i].size(); j++)
        {
            temp.data[j][i] = data[i][j];
        }
    }
    return temp;
}


double uniform_distribution(double low, double high)
{
    double diffrence = high - low;
    int scale = 10000;
    int scaled_diffrence = (int)(diffrence * scale);
    return low + (1.0 * (rand() % scaled_diffrence) / scale);
}
int check_dims(Matrix m1, Matrix m2)
{
    if (m1.data.size() == m2.data.size() && m1.data[0].size() == m2.data[0].size()) return 1;
    return 0;
}
std::vector<Matrix> csv_to_img(char* file_string, int number_of_imgs)
{
    FILE* file = fopen(file_string, "r");
    std::vector<Matrix> returnVal;
    char row[MAXCHAR];
    int i = 0;
    while (feof(file) != 1 && i < number_of_imgs)
    {
        if (i % 100 == 99)std::cout << "Reading Files ... No " << i + 1 << "\n";
        Matrix temp;
        temp.create(28, 28);
        returnVal.push_back(temp);
        int j = 0;
        fgets(row, MAXCHAR, file);
        char* token = strtok(row, ",");


        while (token != NULL)
        {
            if (j == 0)returnVal[i].label = atoi(token);
            else returnVal[i].data[(j - 1) / 28][(j - 1) % 28] = atoi(token) / 255.0;
            token = strtok(NULL, ",");
            j++;
        }
        i++;
    }
    fclose(file);
    return returnVal;
}
double sigmoid(double input)
{
    return  1 / (1 + exp(-input));
}
Matrix sigmoidPrime(Matrix m) {
    Matrix ones;
    ones.create(m.data.size(), m.data[0].size());
    ones.fill(1);
    Matrix subtracted = ones - m;
    Matrix multiplied = m * subtracted;
    return multiplied;
}
Matrix softmax(Matrix m)
{
    double total = 0;
    for (int i = 0; i < m.data.size(); i++)
    {
        for (int j = 0; j < m.data[0].size(); j++)
        {
            total += exp(m.data[i][j]);
        }
    }
    Matrix mat;
    mat.create(m.data.size(), m.data[0].size());
    for (int i = 0; i < mat.data.size(); i++)
    {
        for (int j = 0; j < mat.data[0].size(); j++)
        {
            mat.data[i][j] = exp(m.data[i][j]) / total;
        }
    }
    return mat;
}