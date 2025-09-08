// based on https://github.com/pybind/scikit_build_example

#define LAP_MINIMIZE_V
#define LAP_QUIET

// enable OpenMP support
#ifdef _OPENMP
#define LAP_OPENMP
#endif

// the LFU cache is very slow due to the heap being used for storing the priority queue
#define NO_LFU

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "csrc/lap.h"
#include <iostream>

namespace py = pybind11;


#ifdef LAP_OPENMP
template <class SC, class TC, class CF, class TP>
void solveTableOMP(TP &start_time, int N1, int N2, CF &get_cost, int *rowsol, bool eps, bool sequential = false)
{
	lap::omp::SimpleCostFunction<TC, CF> costFunction(get_cost, sequential);
	lap::omp::Worksharing ws(N2, 8);
	lap::omp::TableCost<TC> costMatrix(N1, N2, costFunction, ws);
	lap::omp::DirectIterator<SC, TC, lap::omp::TableCost<TC>> iterator(N1, N2, costMatrix, ws);
	lap::omp::solve<SC>(N1, N2, costMatrix, iterator, rowsol, eps);
}
#endif


template <class SC, class TC, class CF, class TP>
void solveTable(TP &start_time, int N1, int N2, CF &get_cost, int *rowsol, bool eps)
{
	lap::SimpleCostFunction<TC, CF> costFunction(get_cost);
	lap::TableCost<TC> costMatrix(N1, N2, costFunction);
	lap::DirectIterator<SC, TC, lap::TableCost<TC>> iterator(N1, N2, costMatrix);
	lap::solve<SC>(N1, N2, costMatrix, iterator, rowsol, eps);
}


py::array_t<int> linear_sum_assignment(py::array_t<float> cost_matrix, bool omp = false, bool eps = false) {
    auto start_time = std::chrono::high_resolution_clock::now();

    // create a view of the array with dimension 2 that does not check index bounds, for speed
    auto C = cost_matrix.mutable_unchecked<2>();

    // check input shape
    int ndim = C.ndim();
    if (ndim != 2) {
        throw std::runtime_error("The cost matrix must be 2-dimensional");
    }
    int Nx = C.shape(0);
    int Ny = C.shape(1);
    if (Nx > Ny) {
        throw std::runtime_error("The cost matrix must be larger along the second dimension than the first");
    }

    // create a function that returns the cost of a given assignment
    auto get_cost = [&C](int x, int y) -> float { return C(x, y); };

    // create the output array
    int *rowsol = new int[Ny];

    // solve
    if (omp) {
#ifdef LAP_OPENMP
        solveTableOMP<float, float>(start_time, Nx, Ny, get_cost, rowsol, eps);
#else
        throw std::runtime_error("OpenMP not enabled");
#endif
    }
    else {
        solveTable<float, float>(start_time, Nx, Ny, get_cost, rowsol, eps);
    }
    // convert the output to a numpy array
    py::array_t<int, py::array::c_style> result(Ny);
    auto r = result.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < result.shape(0); i++) {
        r(i) = rowsol[i];
    }

    delete[] rowsol;
    return result;
}

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.def(
        "linear_sum_assignment",
        &linear_sum_assignment,
        py::arg("cost_matrix"), py::arg("omp")=true, py::arg("eps")=true,
        "Solve the linear sum assignment problem"
    );
}
