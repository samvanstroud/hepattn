#ifdef _OPENMP
#  define LAP_OPENMP
#endif
#define LAP_QUIET
// #define LAP_DISPLAY_EVALUATED
// #define LAP_DEBUG
//#define LAP_NO_MEM_DEBUG
//#define LAP_ROWS_SCANNED
//#define LAP_VERIFY_RESULT
#define LAP_MINIMIZE_V

//#define RANDOM_SEED 1234
#ifndef RANDOM_SEED
#ifdef LAP_OPENMP
#define RANDOM_PARALLEL
#endif
#endif

// the LFU cache is very slow due to the heap being used for storing the priority queue
#define NO_LFU

#include "../lap.h"

#include <random>
#include <string>
#include "test_options.h"
#include "image.h"
#include <iomanip>

template <class C> void testRandom(long long min_tab, long long max_tab, int runs, bool omp, bool epsilon, std::string name_C);
template <class C> void testSanity(long long min_tab, long long max_tab, int runs, bool omp, bool epsilon, std::string name_C);
template <class C> void testSanityCached(long long min_cached, long long max_cached, long long max_memory, int runs, bool omp, bool epsilon, std::string name_C);
template <class C> void testGeometric(long long min_tab, long long max_tab, int runs, bool omp, bool epsilon, bool disjoint, std::string name_C);
template <class C> void testGeometricCached(long long min_cached, long long max_cached, long long max_memory, int runs, bool omp, bool epsilon, bool disjoint, std::string name_C);
template <class C> void testRandomLowRank(long long min_tab, long long max_tab, long long min_rank, long long max_rank, int runs, bool omp, bool epsilon, std::string name_C);
template <class C> void testRandomLowRankCached(long long min_cached, long long max_cached, long long max_memory, long long min_rank, long long max_rank, int runs, bool omp, bool epsilon, std::string name_C);
template <class C> void testImages(std::vector<std::string> &images, long long max_memory, int runs, bool omp, bool epsilon, std::string name_C);
template <class C> void testInteger(long long min_tab, long long max_tab, int runs, bool omp, bool epsilon, std::string name_C);

int main(int argc, char* argv[])
{
	Options opt;
	int r = opt.parseOptions(argc, argv);
	if (r != 0) return r;

	if (opt.use_omp)
	{
		// omp "warmup"
		int *tmp = new int[1024];
#pragma omp parallel for
		for (int i = 0; i < 1024; i++)
		{
			tmp[i] = -1;
		}
		delete[] tmp;
	}

	if (opt.use_double)
	{
		if (opt.use_single)
		{
			if (opt.run_sanity) testSanity<double>(opt.lap_min_tab, opt.lap_max_tab, opt.runs, opt.use_omp, false, std::string("double"));
			if (opt.run_sanity_cached) testSanityCached<double>(opt.lap_min_cached, opt.lap_max_cached, opt.lap_max_memory, opt.runs, opt.use_omp, false, std::string("double"));
			if (opt.run_random) testRandom<double>(opt.lap_min_tab, opt.lap_max_tab, opt.runs, opt.use_omp, false, std::string("double"));
			if (opt.run_random_low_rank) testRandomLowRank<double>(opt.lap_min_tab, opt.lap_max_tab, opt.lap_min_rank, opt.lap_max_rank, opt.runs, opt.use_omp, false, std::string("double"));
			if (opt.run_random_low_rank_cached) testRandomLowRankCached<double>(opt.lap_min_cached, opt.lap_max_cached, opt.lap_max_memory, opt.lap_min_rank, opt.lap_max_rank, opt.runs, opt.use_omp, false, std::string("double"));
			if (opt.run_geometric) testGeometric<double>(opt.lap_min_tab, opt.lap_max_tab, opt.runs, opt.use_omp, false, false, std::string("double"));
			if (opt.run_geometric_disjoint) testGeometric<double>(opt.lap_min_tab, opt.lap_max_tab, opt.runs, opt.use_omp, false, true, std::string("double"));
			if (opt.run_geometric_cached) testGeometricCached<double>(opt.lap_min_cached, opt.lap_max_cached, opt.lap_max_memory, opt.runs, opt.use_omp, false, false, std::string("double"));
			if (opt.run_geometric_disjoint_cached) testGeometricCached<double>(opt.lap_min_cached, opt.lap_max_cached, opt.lap_max_memory, opt.runs, opt.use_omp, false, true, std::string("double"));
			if (opt.images.size() > 1) testImages<double>(opt.images, opt.lap_max_memory, opt.runs, opt.use_omp, false, std::string("double"));
		}
		if (opt.use_epsilon)
		{
			if (opt.run_sanity) testSanity<double>(opt.lap_min_tab, opt.lap_max_tab, opt.runs, opt.use_omp, true, std::string("double"));
			if (opt.run_sanity_cached) testSanityCached<double>(opt.lap_min_cached, opt.lap_max_cached, opt.lap_max_memory, opt.runs, opt.use_omp, true, std::string("double"));
			if (opt.run_random) testRandom<double>(opt.lap_min_tab, opt.lap_max_tab, opt.runs, opt.use_omp, true, std::string("double"));
			if (opt.run_random_low_rank) testRandomLowRank<double>(opt.lap_min_tab, opt.lap_max_tab, opt.lap_min_rank, opt.lap_max_rank, opt.runs, opt.use_omp, true, std::string("double"));
			if (opt.run_random_low_rank_cached) testRandomLowRankCached<double>(opt.lap_min_cached, opt.lap_max_cached, opt.lap_max_memory, opt.lap_min_rank, opt.lap_max_rank, opt.runs, opt.use_omp, true, std::string("double"));
			if (opt.run_geometric) testGeometric<double>(opt.lap_min_tab, opt.lap_max_tab, opt.runs, opt.use_omp, true, false, std::string("double"));
			if (opt.run_geometric_disjoint) testGeometric<double>(opt.lap_min_tab, opt.lap_max_tab, opt.runs, opt.use_omp, true, true, std::string("double"));
			if (opt.run_geometric_cached) testGeometricCached<double>(opt.lap_min_cached, opt.lap_max_cached, opt.lap_max_memory, opt.runs, opt.use_omp, true, false, std::string("double"));
			if (opt.run_geometric_disjoint_cached) testGeometricCached<double>(opt.lap_min_cached, opt.lap_max_cached, opt.lap_max_memory, opt.runs, opt.use_omp, true, true, std::string("double"));
			if (opt.images.size() > 1) testImages<double>(opt.images, opt.lap_max_memory, opt.runs, opt.use_omp, true, std::string("double"));
		}
	}
	if (opt.use_float)
	{
		if (opt.use_single)
		{
			if (opt.run_sanity) testSanity<float>(opt.lap_min_tab, opt.lap_max_tab, opt.runs, opt.use_omp, false, std::string("float"));
			if (opt.run_sanity_cached) testSanityCached<float>(opt.lap_min_cached, opt.lap_max_cached, opt.lap_max_memory, opt.runs, opt.use_omp, false, std::string("float"));
			if (opt.run_random) testRandom<float>(opt.lap_min_tab, opt.lap_max_tab, opt.runs, opt.use_omp, false, std::string("float"));
			if (opt.run_random_low_rank) testRandomLowRank<float>(opt.lap_min_tab, opt.lap_max_tab, opt.lap_min_rank, opt.lap_max_rank, opt.runs, opt.use_omp, false, std::string("float"));
			if (opt.run_random_low_rank_cached) testRandomLowRankCached<float>(opt.lap_min_cached, opt.lap_max_cached, opt.lap_max_memory, opt.lap_min_rank, opt.lap_max_rank, opt.runs, opt.use_omp, false, std::string("float"));
			if (opt.run_geometric) testGeometric<float>(opt.lap_min_tab, opt.lap_max_tab, opt.runs, opt.use_omp, false, false, std::string("float"));
			if (opt.run_geometric_disjoint) testGeometric<float>(opt.lap_min_tab, opt.lap_max_tab, opt.runs, opt.use_omp, false, true, std::string("float"));
			if (opt.run_geometric_cached) testGeometricCached<float>(opt.lap_min_cached, opt.lap_max_cached, opt.lap_max_memory, opt.runs, opt.use_omp, false, false, std::string("float"));
			if (opt.run_geometric_disjoint_cached) testGeometricCached<float>(opt.lap_min_cached, opt.lap_max_cached, opt.lap_max_memory, opt.runs, opt.use_omp, false, true, std::string("float"));
			if (opt.images.size() > 1) testImages<float>(opt.images, opt.lap_max_memory, opt.runs, opt.use_omp, false, std::string("float"));
		}
		if (opt.use_epsilon)
		{
			if (opt.run_sanity) testSanity<float>(opt.lap_min_tab, opt.lap_max_tab, opt.runs, opt.use_omp, true, std::string("float"));
			if (opt.run_sanity_cached) testSanityCached<float>(opt.lap_min_cached, opt.lap_max_cached, opt.lap_max_memory, opt.runs, opt.use_omp, true, std::string("float"));
			if (opt.run_random) testRandom<float>(opt.lap_min_tab, opt.lap_max_tab, opt.runs, opt.use_omp, true, std::string("float"));
			if (opt.run_random_low_rank) testRandomLowRank<float>(opt.lap_min_tab, opt.lap_max_tab, opt.lap_min_rank, opt.lap_max_rank, opt.runs, opt.use_omp, true, std::string("float"));
			if (opt.run_random_low_rank_cached) testRandomLowRankCached<float>(opt.lap_min_cached, opt.lap_max_cached, opt.lap_max_memory, opt.lap_min_rank, opt.lap_max_rank, opt.runs, opt.use_omp, true, std::string("float"));
			if (opt.run_geometric) testGeometric<float>(opt.lap_min_tab, opt.lap_max_tab, opt.runs, opt.use_omp, true, false, std::string("float"));
			if (opt.run_geometric_disjoint) testGeometric<float>(opt.lap_min_tab, opt.lap_max_tab, opt.runs, opt.use_omp, true, true, std::string("float"));
			if (opt.run_geometric_cached) testGeometricCached<float>(opt.lap_min_cached, opt.lap_max_cached, opt.lap_max_memory, opt.runs, opt.use_omp, true, false, std::string("float"));
			if (opt.run_geometric_disjoint_cached) testGeometricCached<float>(opt.lap_min_cached, opt.lap_max_cached, opt.lap_max_memory, opt.runs, opt.use_omp, true, true, std::string("float"));
			if (opt.images.size() > 1) testImages<float>(opt.images, opt.lap_max_memory, opt.runs, opt.use_omp, true, std::string("float"));
		}
	}
	if (opt.run_integer)
	{
		if (opt.use_double)
		{
			if (opt.use_single) testInteger<double>(opt.lap_min_tab, opt.lap_max_tab, opt.runs, opt.use_omp, false, std::string("double"));
			if (opt.use_epsilon) testInteger<double>(opt.lap_min_tab, opt.lap_max_tab, opt.runs, opt.use_omp, true, std::string("double"));
		}
		if (opt.use_float)
		{
			if (opt.use_single) testInteger<float>(opt.lap_min_tab, opt.lap_max_tab, opt.runs, opt.use_omp, false, std::string("float"));
			if (opt.use_epsilon) testInteger<float>(opt.lap_min_tab, opt.lap_max_tab, opt.runs, opt.use_omp, true, std::string("float"));
		}
		if (opt.use_int)
		{
			if (opt.use_single) testInteger<int>(opt.lap_min_tab, opt.lap_max_tab, opt.runs, opt.use_omp, false, std::string("int"));
			if (opt.use_epsilon) testInteger<int>(opt.lap_min_tab, opt.lap_max_tab, opt.runs, opt.use_omp, true, std::string("int"));
		}
		if (opt.use_long)
		{
			if (opt.use_single) testInteger<long long>(opt.lap_min_tab, opt.lap_max_tab, opt.runs, opt.use_omp, false, std::string("long long"));
			if (opt.use_epsilon) testInteger<long long>(opt.lap_min_tab, opt.lap_max_tab, opt.runs, opt.use_omp, true, std::string("long long"));
		}
	}

#ifndef LAP_QUIET
	lap::allocationLogger.destroy();
#endif

	return 0;
}

#ifdef LAP_OPENMP
template <class SC, class TC, class CF, class TP>
void solveTableOMP(TP &start_time, int N1, int N2, CF &get_cost, int *rowsol, bool epsilon, bool sequential = false)
{
	lap::omp::SimpleCostFunction<TC, CF> costFunction(get_cost, sequential);
	lap::omp::Worksharing ws(N2, 8);
	lap::omp::TableCost<TC> costMatrix(N1, N2, costFunction, ws);
	lap::omp::DirectIterator<SC, TC, lap::omp::TableCost<TC>> iterator(N1, N2, costMatrix, ws);

	lap::displayTime(start_time, "setup complete", std::cout);

	lap::omp::solve<SC>(N1, N2, costMatrix, iterator, rowsol, epsilon);

	std::stringstream ss;
	ss << "cost = " << std::setprecision(std::numeric_limits<SC>::max_digits10) << lap::omp::cost<SC>(N1, N2, costMatrix, rowsol);
	lap::displayTime(start_time, ss.str().c_str(), std::cout);
}
#endif

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

#ifdef LAP_OPENMP
template <class SC, class TC, class CF, class TP>
void solveCachingOMP(TP &start_time, int N1, int N2, CF &get_cost, int *rowsol, int entries, bool epsilon)
{
	lap::omp::SimpleCostFunction<TC, CF> costFunction(get_cost);
	lap::omp::Worksharing ws(N2, costFunction.getMultiple());

#ifndef NO_LFU
	if (4 * entries < N1)
#endif
	{
		lap::omp::CachingIterator<SC, TC, lap::omp::SimpleCostFunction<TC, CF>, lap::CacheSLRU> iterator(N1, N2, entries, costFunction, ws);

		lap::displayTime(start_time, "setup complete", std::cout);

		// estimating epsilon
		lap::omp::solve<SC>(N1, N2, costFunction, iterator, rowsol, epsilon);
	}
#ifndef NO_LFU
	else
	{
		lap::omp::CachingIterator<SC, TC, lap::omp::SimpleCostFunction<TC, CF>, lap::CacheLFU> iterator(N1, N2, entries, costFunction, ws);

		lap::displayTime(start_time, "setup complete", std::cout);

		// estimating epsilon
		lap::omp::solve<SC>(N1, N2, costFunction, iterator, rowsol, epsilon);
	}
#endif

	std::stringstream ss;
	ss << "cost = " << std::setprecision(std::numeric_limits<SC>::max_digits10) << lap::omp::cost<SC>(N1, N2, costFunction, rowsol);
	lap::displayTime(start_time, ss.str().c_str(), std::cout);
}
#endif

template <class SC, class TC, class CF, class TP>
void solveCaching(TP &start_time, int N1, int N2, CF &get_cost, int *rowsol, int entries, bool epsilon)
{
	lap::SimpleCostFunction<TC, CF> costFunction(get_cost);

#ifndef NO_LFU
	if (4 * entries < N1)
#endif
	{
		lap::CachingIterator<SC, TC, lap::SimpleCostFunction<TC, CF>, lap::CacheSLRU> iterator(N1, N2, entries, costFunction);

		lap::displayTime(start_time, "setup complete", std::cout);

		lap::solve<SC>(N1, N2, costFunction, iterator, rowsol, epsilon);
	}
#ifndef NO_LFU
	else
	{
		lap::CachingIterator<SC, TC, lap::SimpleCostFunction<TC, CF>, lap::CacheLFU> iterator(N1, N2, entries, costFunction);

		lap::displayTime(start_time, "setup complete", std::cout);

		lap::solve<SC>(N1, N2, costFunction, iterator, rowsol, epsilon);
	}
#endif

	std::stringstream ss;
	ss << "cost = " << std::setprecision(std::numeric_limits<SC>::max_digits10) << lap::cost<SC>(N1, N2, costFunction, rowsol);
	lap::displayTime(start_time, ss.str().c_str(), std::cout);
}

#ifdef LAP_OPENMP
template <class SC, class TC, class CF, class TP>
void solveAdaptiveOMP(TP &start_time, int N1, int N2, CF &get_cost, int *rowsol, int entries, bool epsilon)
{
	if (N1 <= entries)
	{
		std::cout << "using multithreaded table with " << N1 << " rows." << std::endl;
		solveTableOMP<SC, TC>(start_time, N1, N2, get_cost, rowsol, epsilon);
	}
	else
	{
		std::cout << "using multithreaded caching with " << entries << "/" << N1 << " entries." << std::endl;
		solveCachingOMP<SC, TC>(start_time, N1, N2, get_cost, rowsol, entries, epsilon);
	}
}
#endif

template <class SC, class TC, class CF, class TP>
void solveAdaptive(TP &start_time, int N1, int N2, CF &get_cost, int *rowsol, int entries, bool epsilon)
{
	if (N1 <= entries)
	{
		std::cout << "using table with " << N1 << " rows." << std::endl;
		solveTable<SC, TC>(start_time, N1, N2, get_cost, rowsol, epsilon);
	}
	else
	{
		std::cout << "using caching with " << entries << "/" << N1 << " entries." << std::endl;
		solveCaching<SC, TC>(start_time, N1, N2, get_cost, rowsol, entries, epsilon);
	}
}

template <class C>
void testRandom(long long min_tab, long long max_tab, int runs, bool omp, bool epsilon, std::string name_C)
{
	// random costs (directly supply cost matrix)
	for (long long NN = min_tab * min_tab; NN <= max_tab * max_tab; NN <<= 1)
	{
		for (int r = 0; r < runs; r++)
		{
			int N = (int)floor(sqrt((double)NN));

			std::cout << "Random";
			std::cout << "<" << name_C << "> " << N << "x" << N << " table";
			if (omp) std::cout << " multithreaded";
			if (epsilon) std::cout << " with epsilon scaling";
			std::cout << std::endl;

			std::uniform_real_distribution<C> distribution(0.0, 1.0);
#ifdef RANDOM_SEED
			std::mt19937_64 generator(RANDOM_SEED);
#else
			std::random_device rd;
#ifdef RANDOM_PARALLEL
			int threads = omp_get_max_threads();
			std::mt19937_64* generator = new std::mt19937_64[threads];
			for (int t = 0; t < threads; t++) generator[t].seed(rd());
#else
			std::mt19937_64 generator(rd());
#endif
#endif

			auto start_time = std::chrono::high_resolution_clock::now();

			int *rowsol = new int[N];

			auto get_cost = [&distribution, &generator](int x, int y) -> C
			{
#ifdef RANDOM_PARALLEL
				return distribution(generator[omp_get_thread_num()]);
#else
				return distribution(generator);
#endif
			};

			if (omp)
			{
#ifdef LAP_OPENMP
				// initialization needs to be serial
#ifdef RANDOM_PARALLEL
				solveTableOMP<C, C>(start_time, N, N, get_cost, rowsol, epsilon);
#else
				solveTableOMP<C, C>(start_time, N, N, get_cost, rowsol, epsilon, true);
#endif
#endif
			}
			else
			{
				solveTable<C, C>(start_time, N, N, get_cost, rowsol, epsilon);
			}

			delete[] rowsol;
		}
	}
}

template <class C>
void testSanity(long long min_tab, long long max_tab, int runs, bool omp, bool epsilon, std::string name_C)
{
	// random costs (directly supply cost matrix)
	for (long long NN = min_tab * min_tab; NN <= max_tab * max_tab; NN <<= 1)
	{
		for (int r = 0; r < runs; r++)
		{
			int N = (int)floor(sqrt((double)NN));

			std::cout << "Sanity";
			std::cout << "<" << name_C << "> " << N << "x" << N << " table";
			if (omp) std::cout << " multithreaded";
			if (epsilon) std::cout << " with epsilon scaling";
			std::cout << std::endl;

			std::uniform_real_distribution<C> distribution(0.0, 1.0);
#ifdef RANDOM_SEED
			std::mt19937_64 generator(RANDOM_SEED);
#else
			std::random_device rd;
			std::mt19937_64 generator(rd());
#endif

			auto start_time = std::chrono::high_resolution_clock::now();

			int *rowsol = new int[N];

			C *vec = new C[N << 1];

			for (long long i = 0; i < N << 1; i++) vec[i] = distribution(generator);

			// cost functions
			auto get_cost = [&vec, &N](int x, int y) -> C
			{
				C r = vec[x] + vec[y + N];
				if (x == y) return r;
				else return r + C(0.1);
			};


			if (omp)
			{
#ifdef LAP_OPENMP
				solveTableOMP<C, C>(start_time, N, N, get_cost, rowsol, epsilon);
#endif
			}
			else
			{
				solveTable<C, C>(start_time, N, N, get_cost, rowsol, epsilon);
			}

			bool passed = true;
			for (long long i = 0; (passed) && (i < N); i++)
			{
				passed &= (rowsol[i] == i);
			}
			std::stringstream ss;
			// for (int i = 0; i < N; i++) ss << rowsol[i] << " ";
			// ss << '\n';
			if (passed) ss << "test passed: ";
			else ss << "test failed: ";
			C real_cost(0);
			for (int i = 0; i < N; i++) real_cost += get_cost(i, i);
			ss << "ground truth cost = " << std::setprecision(std::numeric_limits<C>::max_digits10) << real_cost;
			lap::displayTime(start_time, ss.str().c_str(), std::cout);

			delete[] vec;
			delete[] rowsol;
		}
	}
}

template <class C>
void testSanityCached(long long min_cached, long long max_cached, long long max_memory, int runs, bool omp, bool epsilon, std::string name_C)
{
	for (long long NN = min_cached * min_cached; NN <= max_cached * max_cached; NN <<= 1)
	{
		for (int r = 0; r < runs; r++)
		{
			int N = (int)floor(sqrt((double)NN));
			int entries = (int)std::min((long long)N, (long long)(max_memory / (sizeof(C) * N)));

			std::cout << "Sanity";
			std::cout << "<" << name_C << "> " << N << "x" << N << " (" << entries << ")";
			if (omp) std::cout << " multithreaded";
			if (epsilon) std::cout << " with epsilon scaling";
			std::cout << std::endl;

			auto start_time = std::chrono::high_resolution_clock::now();

			std::uniform_real_distribution<C> distribution(0.0, 1.0);
#ifdef RANDOM_SEED
			std::mt19937_64 generator(RANDOM_SEED);
#else
			std::random_device rd;
			std::mt19937_64 generator(rd());
#endif

			C *vec = new C[N << 1];

			for (long long i = 0; i < N << 1; i++) vec[i] = distribution(generator);

			// cost function
			auto get_cost = [&vec, &N](int x, int y) -> C
			{
				C r = vec[x] + vec[y + N];
				if (x == y) return r;
				else return r + C(0.1);
			};

			int *rowsol = new int[N];

			if (omp)
			{
#ifdef LAP_OPENMP
				solveAdaptiveOMP<C, C>(start_time, N, N, get_cost, rowsol, entries, epsilon);
#endif
			}
			else
			{
				solveAdaptive<C, C>(start_time, N, N, get_cost, rowsol, entries, epsilon);
			}

			bool passed = true;
			for (long long i = 0; (passed) && (i < N); i++)
			{
				passed &= (rowsol[i] == i);
			}
			std::stringstream ss;
			if (passed) ss << "test passed: ";
			else ss << "test failed: ";
			C real_cost(0);
			for (int i = 0; i < N; i++) real_cost += get_cost(i, i);
			ss << "ground truth cost = " << std::setprecision(std::numeric_limits<C>::max_digits10) << real_cost;
			lap::displayTime(start_time, ss.str().c_str(), std::cout);

			delete[] rowsol;
			delete[] vec;
		}
	}
}

template <class C>
void testRandomLowRank(long long min_tab, long long max_tab, long long min_rank, long long max_rank, int runs, bool omp, bool epsilon, std::string name_C)
{
	// random costs (directly supply cost matrix)
	for (long long rank = min_rank; rank <= max_rank; rank <<= 1)
	{
		for (long long NN = min_tab * min_tab; NN <= max_tab * max_tab; NN <<= 1)
		{
			for (int r = 0; r < runs; r++)
			{
				int N = (int)floor(sqrt((double)NN));

				std::cout << "RandomLowRank<" << name_C << "> " << N << "x" << N << " table rank = " << rank;
				if (omp) std::cout << " multithreaded";
				if (epsilon) std::cout << " with epsilon scaling";
				std::cout << std::endl;

				auto start_time = std::chrono::high_resolution_clock::now();

				std::uniform_real_distribution<C> distribution(0.0, 1.0);
#ifdef RANDOM_SEED
				std::mt19937_64 generator(RANDOM_SEED);
#else
				std::random_device rd;
				std::mt19937_64 generator(rd());
#endif

				// The following matrix will have at most the seletcted rank.
				C *vec = new C[N * rank];
				C max_vec;
				C min_vec;
				for (long long i = 0; i < rank; i++)
				{
					for (long long j = 0; j < N; j++) vec[i * N + j] = distribution(generator);
					max_vec = vec[i * N];
					for (long long j = 1; j < N; j++) max_vec = std::max(max_vec, vec[i * N + j]);
					min_vec = vec[i * N];
					for (long long j = 1; j < N; j++) min_vec = std::min(min_vec, vec[i * N + j]);
				}

				// cost function
				auto get_cost = [&vec, &N, &rank, &max_vec](int x, int y) -> C
				{
					C sum(0);
					for (long long k = 0; k < rank; k++)
					{
						sum += vec[k * N + x] * vec[k * N + y];
					}
					return sum / C(rank);
				};

				int *rowsol = new int[N];

				if (omp)
				{
#ifdef LAP_OPENMP
					solveTableOMP<C, C>(start_time, N, N, get_cost, rowsol, epsilon);
#endif
				}
				else
				{
					solveTable<C, C>(start_time, N, N, get_cost, rowsol, epsilon);
				}

				delete[] vec;
				delete[] rowsol;
			}
		}
	}
}

template <class C>
void testRandomLowRankCached(long long min_cached, long long max_cached, long long max_memory, long long min_rank, long long max_rank, int runs, bool omp, bool epsilon, std::string name_C)
{
	for (long long rank = min_rank; rank <= max_rank; rank <<= 1)
	{
		for (long long NN = min_cached * min_cached; NN <= max_cached * max_cached; NN <<= 1)
		{
			for (int r = 0; r < runs; r++)
			{
				int N = (int)floor(sqrt((double)NN));
				int entries = (int)std::min((long long)N, (long long)(max_memory / (sizeof(C) * N)));

				std::cout << "RandomLowRank<" << name_C << "> " << N << "x" << N << " (" << entries << ") rank = " << rank;
				if (omp) std::cout << " multithreaded";
				if (epsilon) std::cout << " with epsilon scaling";
				std::cout << std::endl;

				auto start_time = std::chrono::high_resolution_clock::now();

				std::uniform_real_distribution<C> distribution(0.0, 1.0);
#ifdef RANDOM_SEED
				std::mt19937_64 generator(RANDOM_SEED);
#else
				std::random_device rd;
				std::mt19937_64 generator(rd());
#endif

				// The following matrix will have at most the seletcted rank.
				C *vec = new C[N * rank];
				for (long long i = 0; i < rank; i++)
				{
					for (long long j = 0; j < N; j++) vec[i * N + j] = distribution(generator);
				}

				// cost function
				auto get_cost = [&vec, &N, &rank](int x, int y) -> C
				{
					C sum(0);
					for (long long k = 0; k < rank; k++)
					{
						sum += vec[k * N + x] * vec[k * N + y];
					}
					return sum / C(rank);
				};

				int *rowsol = new int[N];

				if (omp)
				{
#ifdef LAP_OPENMP
					solveAdaptiveOMP<C, C>(start_time, N, N, get_cost, rowsol, entries, epsilon);
#endif
				}
				else
				{
					solveAdaptive<C, C>(start_time, N, N, get_cost, rowsol, entries, epsilon);
				}

				delete[] rowsol;
				delete[] vec;
			}
		}
	}
}

template <class C>
void testGeometric(long long min_tab, long long max_tab, int runs, bool omp, bool epsilon, bool disjoint, std::string name_C)
{
	// geometric costs in R^2 (supply function for calculating cost matrix)
	for (long long NN = min_tab * min_tab; NN <= max_tab * max_tab; NN <<= 1)
	{
		for (int r = 0; r < runs; r++)
		{
			int N = (int)floor(sqrt((double)NN));

			std::cout << "Geometric";
			if (disjoint) std::cout << " Disjoint";
			std::cout << " R^2<" << name_C << "> " << N << "x" << N << " table";
			if (omp) std::cout << " multithreaded";
			if (epsilon) std::cout << " with epsilon scaling";
			std::cout << std::endl;

			auto start_time = std::chrono::high_resolution_clock::now();

			std::uniform_real_distribution<C> distribution(0.0, 1.0);
#ifdef RANDOM_SEED
			std::mt19937_64 generator(RANDOM_SEED);
#else
			std::random_device rd;
			std::mt19937_64 generator(rd());
#endif

			C *tab_s = new C[2 * N];
			C *tab_t = new C[2 * N];

			for (int i = 0; i < 2 * N; i++)
			{
				tab_s[i] = distribution(generator);
				tab_t[i] = distribution(generator);
			}

			if (disjoint)
			{
				for (int i = 0; i < 2 * N; i += 2)
				{
					if (i < N)
					{
						tab_t[i] += C(1);
					}
					else
					{
						tab_s[i] += C(1);
						tab_s[i + 1] += C(1);
						tab_t[i + 1] += C(1);
					}
				}
			}

			// cost function
			auto get_cost = [&tab_s, &tab_t](int x, int y) -> C
			{
				int xx = x + x;
				int yy = y + y;
				C d0 = tab_s[xx] - tab_t[yy];
				C d1 = tab_s[xx + 1] - tab_t[yy + 1];
				return d0 * d0 + d1 * d1;
			};

			int *rowsol = new int[N];

			if (omp)
			{
#ifdef LAP_OPENMP
				solveTableOMP<C, C>(start_time, N, N, get_cost, rowsol, epsilon);
#endif
			}
			else
			{
				solveTable<C, C>(start_time, N, N, get_cost, rowsol, epsilon);
			}

			delete[] tab_s;
			delete[] tab_t;
			delete[] rowsol;
		}
	}
}

template <class C>
void testGeometricCached(long long min_cached, long long max_cached, long long max_memory, int runs, bool omp, bool epsilon, bool disjoint, std::string name_C)
{
	for (long long NN = min_cached * min_cached; NN <= max_cached * max_cached; NN <<= 1)
	{
		for (int r = 0; r < runs; r++)
		{
			int N = (int)floor(sqrt((double)NN));
			int entries = (int)std::min((long long)N, (long long)(max_memory / (sizeof(C) * N)));

			std::cout << "Geometric";
			if (disjoint) std::cout << " Disjoint";
			std::cout << " R^2<" << name_C << "> " << N << "x" << N << " (" << entries << ")";
			if (omp) std::cout << " multithreaded";
			if (epsilon) std::cout << " with epsilon scaling";
			std::cout << std::endl;

			auto start_time = std::chrono::high_resolution_clock::now();

			std::uniform_real_distribution<C> distribution(0.0, 1.0);
#ifdef RANDOM_SEED
			std::mt19937_64 generator(RANDOM_SEED);
#else
			std::random_device rd;
			std::mt19937_64 generator(rd());
#endif

			C *tab_s = new C[2 * N];
			C *tab_t = new C[2 * N];
			for (int i = 0; i < 2 * N; i++)
			{
				tab_s[i] = distribution(generator);
				tab_t[i] = distribution(generator);
			}

			if (disjoint)
			{
				for (int i = 0; i < 2 * N; i += 2)
				{
					if (i < N)
					{
						tab_t[i] += C(1);
					}
					else
					{
						tab_s[i] += C(1);
						tab_s[i + 1] += C(1);
						tab_t[i + 1] += C(1);
					}
				}
			}

			// cost function
			auto get_cost = [&tab_s, &tab_t](int x, int y) -> C
			{
				int xx = x + x;
				int yy = y + y;
				C d0 = tab_s[xx] - tab_t[yy];
				C d1 = tab_s[xx + 1] - tab_t[yy + 1];
				return d0 * d0 + d1 * d1;
			};

			int *rowsol = new int[N];

			if (omp)
			{
#ifdef LAP_OPENMP
				solveAdaptiveOMP<C, C>(start_time, N, N, get_cost, rowsol, entries, epsilon);
#endif
			}
			else
			{
				solveAdaptive<C, C>(start_time, N, N, get_cost, rowsol, entries, epsilon);
			}

			delete[] rowsol;
			delete[] tab_s;
			delete[] tab_t;
		}
	}
}

template <class C>
void testInteger(long long min_tab, long long max_tab, int runs, bool omp, bool epsilon, std::string name_C)
{
	// random costs (directly supply cost matrix)
	//for (int range = 0; range < 3; range++)
	int range = 2;
	{
		for (long long NN = min_tab * min_tab; NN <= max_tab * max_tab; NN <<= 1)
		{
			for (int r = 0; r < runs; r++)
			{
				int N = (int)floor(sqrt((double)NN));

				std::cout << "Integer";
				std::cout << "<" << name_C << " ";
				if (range == 0) std::cout << "1/10n";
				else if (range == 1) std::cout << "n";
				else std::cout << "10n";
				std::cout << "> " << N << "x" << N << " table";
				if (omp) std::cout << " multithreaded";
				if (epsilon) std::cout << " with epsilon scaling";
				std::cout << std::endl;

				int n;
				if (range == 0) n = N / 10;
				else if (range == 1) n = N;
				else n = 10 * N;
				std::uniform_int_distribution<int> distribution(0, n);
#ifdef RANDOM_SEED
				std::mt19937_64 generator(RANDOM_SEED);
#else
				std::random_device rd;
#ifdef RANDOM_PARALLEL
				int threads = omp_get_max_threads();
				std::mt19937_64* generator = new std::mt19937_64[threads];
				for (int t = 0; t < threads; t++) generator[t].seed(rd());
#else
				std::mt19937_64 generator(rd());
#endif
#endif

				auto start_time = std::chrono::high_resolution_clock::now();

				auto get_cost = [&distribution, &generator](int x, int y) -> int
				{
#ifdef RANDOM_PARALLEL
					return distribution(generator[omp_get_thread_num()]);
#else
					return distribution(generator);
#endif
				};

				int *rowsol = new int[N];

				if (omp)
				{
#ifdef LAP_OPENMP
					// initialization needs to be serial
#ifdef RANDOM_PARALLEL
					solveTableOMP<C, int>(start_time, N, N, get_cost, rowsol, epsilon);
#else
					solveTableOMP<C, int>(start_time, N, N, get_cost, rowsol, epsilon, true);
#endif
#endif
				}
				else
				{
					solveTable<C, int>(start_time, N, N, get_cost, rowsol, epsilon);
				}

				delete[] rowsol;
			}
		}
	}
}

template <class C> void testImages(std::vector<std::string> &images, long long max_memory, int runs, bool omp, bool epsilon, std::string name_C)
{
	std::cout << "Comparing images ";
	if (omp) std::cout << " multithreaded";
	if (epsilon) std::cout << " with epsilon scaling";
	std::cout << std::endl;
	for (unsigned int a = 0; a < images.size() - 1; a++)
	{
		for (unsigned int b = a + 1; b < images.size(); b++)
		{
			PPMImage img_a(images[a]);
			PPMImage img_b(images[b]);
			std::cout << "Comparing image \"" << images[a] << "\" (" << img_a.width << "x" << img_a.height << ") with image \"" << images[b] << "\" (" << img_b.width << "x" << img_b.height << ")." << std::endl;
			for (int r = 0; r < runs; r++)
			{
				auto start_time = std::chrono::high_resolution_clock::now();

				// make sure img[0] is at most as large as img[1]
				PPMImage* img[2];
				if (img_a.width * img_a.height < img_b.width * img_b.height)
				{
					img[0] = &img_a;
					img[1] = &img_b;
				}
				else
				{
					img[0] = &img_b;
					img[1] = &img_a;
				}
				int N1 = img[0]->width * img[0]->height;
				int N2 = img[1]->width * img[1]->height;

				// cost function
				auto get_cost = [&img](int x, int y) -> C
				{
					C r = C(img[0]->raw[3 * x]) / C(img[0]->max_val) - C(img[1]->raw[3 * y]) / C(img[1]->max_val);
					C g = C(img[0]->raw[3 * x + 1]) / C(img[0]->max_val) - C(img[1]->raw[3 * y + 1]) / C(img[1]->max_val);
					C b = C(img[0]->raw[3 * x + 2]) / C(img[0]->max_val) - C(img[1]->raw[3 * y + 2]) / C(img[1]->max_val);
					C u = C(x % img[0]->width) / C(img[0]->width - 1) - C(y % img[1]->width) / C(img[1]->width - 1);
					C v = C(x / img[0]->width) / C(img[0]->height - 1) - C(y / img[1]->width) / C(img[1]->height - 1);
					return r * r + g * g + b * b + u * u + v * v;
				};

				int* rowsol = new int[N2];
				int entries = (int)std::min((long long)N1, (long long)(max_memory / (sizeof(C) * N2)));

				if (omp)
				{
#ifdef LAP_OPENMP
					solveAdaptiveOMP<C, C>(start_time, N1, N2, get_cost, rowsol, entries, epsilon);
#endif
				}
				else
				{
					solveAdaptive<C, C>(start_time, N1, N2, get_cost, rowsol, entries, epsilon);
				}
				delete[] rowsol;
			}
		}
	}
}
