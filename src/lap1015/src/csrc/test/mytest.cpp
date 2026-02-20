

// enable OpenMP support
//#ifdef _OPENMP
//#  define LAP_OPENMP
//#endif
//// quiet mode
//#define LAP_QUIET
// increase numerical stability for non-integer costs
#define LAP_MINIMIZE_V
//In case you would like to use GPU support, use the following as a starting point:
//// enable CUDA support
//#define LAP_CUDA
//// OpenMP required for multiple devices
//#define LAP_CUDA_OPENMP
//// quiet mode
//#define LAP_QUIET
//// increase numerical stability for non-integer costs
//#define LAP_MINIMIZE_V

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cassert>
#include <random>
#include <iostream>
#include <random>
#include <iomanip>
#include <limits>
#include <numbers>

#include "../lap.h"


template <class SC, class TC, class CF, class TP>
void solveTable(TP &start_time, int N1, int N2, CF &get_cost, int *rowsol, bool epsilon)
{
	lap::SimpleCostFunction<TC, CF> costFunction(get_cost);
	lap::TableCost<TC> costMatrix(N1, N2, costFunction);
	lap::DirectIterator<SC, TC, lap::TableCost<TC>> iterator(N1, N2, costMatrix);

	lap::displayTime(start_time, "setup complete", std::cout);

	// estimating epsilon
	lap::solve<SC>(N1, N2, costMatrix, iterator, rowsol, epsilon);

	std::stringstream ss;
	ss << "cost = " << std::setprecision(std::numeric_limits<SC>::max_digits10) << lap::cost<SC>(N1, N2, costMatrix, rowsol);
	lap::displayTime(start_time, ss.str().c_str(), std::cout);
}

int main(int argc, char* argv[])
{
    // define the type of the cost
    typedef float C;

    // params
    int RANDOM_SEED = 42;
    const int Nx = 1000;
    const int Ny = 100;

    // generate random numbers
    std::uniform_real_distribution<C> distribution(0.0, 1.0);
    std::mt19937_64 generator(RANDOM_SEED);
    std::vector<std::vector<C>> cost_matrix(Nx, std::vector<C>(Ny));
    for (long long i = 0; i < Nx; i++) {
        for (long long j = 0; j < Ny; j++) {
            cost_matrix[i][j] = distribution(generator);
        }
    }

    // define cost function using random numbers
    auto get_cost = [&cost_matrix, &Nx](int x, int y) -> C
    {
        C r = cost_matrix[x][y];
        if (x == y) return r;
        else return r + C(0.1);
    };
    int *rowsol = new int[Nx];

    // solve the problem
    bool epsilon = true;
    auto start_time = std::chrono::high_resolution_clock::now();
    solveTable<C, C>(start_time, Nx, Ny, get_cost, rowsol, epsilon);

    return 0;
}
