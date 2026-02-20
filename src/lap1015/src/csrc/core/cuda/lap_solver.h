#pragma once

#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#ifdef LAP_CUDA_OPENMP
#include <omp.h>
#endif

#include "lap_kernel.cuh"

#include <fstream>

namespace lap
{
	namespace cuda
	{
		template <typename T>
		void allocPinned(T * &ptr, unsigned long long width, const char *file, const int line)
		{
			__checkCudaErrors(cudaMallocHost(&ptr, sizeof(T) * width), file, line);
#ifndef LAP_QUIET
			allocationLogger.alloc(1, ptr, width, file, line);
#endif
		}

		template <typename T>
		void freePinned(T *&ptr)
		{
			if (ptr == (T *)NULL) return;
#ifndef LAP_QUIET
			allocationLogger.free(1, ptr);
#endif
			checkCudaErrors(cudaFreeHost(ptr));
			ptr = (T *)NULL;
		}

		template <typename T>
		void allocDevice(T * &ptr, unsigned long long width, const char *file, const int line)
		{
			__checkCudaErrors(cudaMalloc(&ptr, sizeof(T) * width), file, line);
#ifndef LAP_QUIET
			allocationLogger.alloc(2, ptr, width, file, line);
#endif
		}

		template <typename T>
		void freeDevice(T *&ptr)
		{
			if (ptr == (T *)NULL) return;
#ifndef LAP_QUIET
			allocationLogger.free(2, ptr);
#endif
			checkCudaErrors(cudaFree(ptr));
			ptr = (T *)NULL;
		}

		template <class SC>
		class estimateEpsilon_struct
		{
		public:
			SC min;
			SC max;
			SC picked;
			SC v_jmin;
			int jmin;
		};

		template <class SC>
		class min_struct
		{
		public:
			SC min;
			SC max;
			int jmin;
			int colsol;
			int data_valid;
		};

		template <class SC, class MS>
		void findMaximum(SC *v_private, MS *max_struct, cudaStream_t &stream, int min_count)
		{
			if (min_count <= 32) findMaxSmall_kernel<<<1, 32, 0, stream>>>(max_struct, v_private, std::numeric_limits<SC>::lowest(), min_count);
			else if (min_count <= 256) findMaxMedium_kernel<<<1, 256, 0, stream>>>(max_struct, v_private, std::numeric_limits<SC>::lowest(), min_count);
			else findMaxLarge_kernel<<<1, 1024, 0, stream>>>(max_struct, v_private, std::numeric_limits<SC>::lowest(), min_count);
		}

		template <class SC, class MS>
		void findMaximum(SC *v_private, int *colsol_private, MS *max_struct, cudaStream_t &stream, int min_count)
		{
			if (min_count <= 32) findMaxSmall_kernel<<<1, 32, 0, stream>>>(max_struct, v_private, colsol_private, std::numeric_limits<SC>::lowest(), min_count);
			else if (min_count <= 256) findMaxMedium_kernel<<<1, 256, 0, stream>>>(max_struct, v_private, colsol_private, std::numeric_limits<SC>::lowest(), min_count);
			else findMaxLarge_kernel<<<1, 1024, 0, stream>>>(max_struct, v_private, colsol_private, std::numeric_limits<SC>::lowest(), min_count);
		}

		template <class SC, class MS>
		SC mergeMaximum(MS *max_struct, int devices)
		{
			SC max_cost = max_struct[0].max;
			for (int tx = 1; tx < devices; tx++)
			{
				max_cost = std::max(max_cost, max_struct[tx].max);
			}
			return max_cost;
		}

		int getMinSize(int num_items)
		{
			if (num_items <= 1024) return (num_items + 31) >> 5;
			else if (num_items <= 65536) return (num_items + 255) >> 8;
			else return (num_items + 1023) >> 10;
		}

		int getBlockSize(int num_items)
		{
			if (num_items <= 1024) return 32;
			else if (num_items <= 65536) return 256;
			else return 1024;
		}

		template <class I>
		void selectDevice(int &start, int &num_items, cudaStream_t &stream, int &bs, int &gs, int t, I &iterator)
		{
			checkCudaErrors(cudaSetDevice(iterator.ws.device[t]));
			start = iterator.ws.part[t].first;
			num_items = iterator.ws.part[t].second - start;
			stream = iterator.ws.stream[t];

			bs = getBlockSize(num_items);
			gs = getMinSize(num_items);
		}

		template <class SC, class TC, class I>
		std::pair<SC, SC> estimateEpsilon(int dim, int dim2, I& iterator, SC **v_private, int *perm)
		{
#ifdef LAP_DEBUG
			auto start_time = std::chrono::high_resolution_clock::now();
#endif
			SC *mod_v;
			SC **mod_v_private;
			SC **min_cost_private;
			SC **max_cost_private;
			SC **picked_cost_private;
			int **jmin_private;
			int **start_private;
			int **picked_private;
			int *picked;
			int *data_valid;
			estimateEpsilon_struct<SC> *host_struct_private;
			estimateEpsilon_struct<SC> **gpu_struct_private;
			unsigned int **semaphore_private;


#ifdef LAP_CUDA_OPENMP
			int devices = (int)iterator.ws.device.size();
			bool peerEnabled = iterator.ws.peerAccess();

			int max_threads = omp_get_max_threads();
			if (max_threads < devices) omp_set_num_threads(devices);
#else
			int devices = 1;
#endif

			decltype(iterator.getRow(0, 0, false)) *tt;
			lapAlloc(tt, devices, __FILE__, __LINE__);

			lapAllocPinned(mod_v, dim2, __FILE__, __LINE__);
			lapAlloc(mod_v_private, devices, __FILE__, __LINE__);
			lapAlloc(min_cost_private, devices, __FILE__, __LINE__);
			lapAlloc(max_cost_private, devices, __FILE__, __LINE__);
			lapAlloc(picked_cost_private, devices, __FILE__, __LINE__);
			lapAlloc(jmin_private, devices, __FILE__, __LINE__);
			lapAlloc(picked_private, devices, __FILE__, __LINE__);
			lapAlloc(picked, dim2, __FILE__, __LINE__);
			lapAlloc(semaphore_private, devices, __FILE__, __LINE__);
			lapAlloc(start_private, devices, __FILE__, __LINE__);
			lapAlloc(gpu_struct_private, devices, __FILE__, __LINE__);
			lapAllocPinned(host_struct_private, dim2 * devices, __FILE__, __LINE__);
			lapAllocPinned(data_valid, dim2 * devices, __FILE__, __LINE__);

			{
				int *host_start;
				lapAllocPinned(host_start, devices, __FILE__, __LINE__);
				for (int t = 0; t < devices; t++)
				{
					host_start[t] = iterator.ws.part[t].first;

					checkCudaErrors(cudaSetDevice(iterator.ws.device[t]));
					int num_items = iterator.ws.part[t].second - iterator.ws.part[t].first;
					int count = getMinSize(num_items);

					lapAllocDevice(mod_v_private[t], num_items, __FILE__, __LINE__);
					lapAllocDevice(picked_private[t], num_items, __FILE__, __LINE__);
					lapAllocDevice(semaphore_private[t], 2, __FILE__, __LINE__);
					lapAllocDevice(min_cost_private[t], count, __FILE__, __LINE__);
					lapAllocDevice(max_cost_private[t], count, __FILE__, __LINE__);
					lapAllocDevice(picked_cost_private[t], count, __FILE__, __LINE__);
					lapAllocDevice(jmin_private[t], count, __FILE__, __LINE__);
					lapAllocDevice(gpu_struct_private[t], 1, __FILE__, __LINE__);
					lapAllocDevice(start_private[t], devices, __FILE__, __LINE__);
				}
				for (int t = 0; t < devices; t++)
				{
					checkCudaErrors(cudaSetDevice(iterator.ws.device[t]));
					int num_items = iterator.ws.part[t].second - iterator.ws.part[t].first;
					cudaStream_t stream = iterator.ws.stream[t];

					checkCudaErrors(cudaMemsetAsync(picked_private[t], 0, num_items * sizeof(int), stream));
					checkCudaErrors(cudaMemsetAsync(semaphore_private[t], 0, 2 * sizeof(unsigned int), stream));
					checkCudaErrors(cudaMemcpyAsync(start_private[t], host_start, devices * sizeof(int), cudaMemcpyHostToDevice, stream));
				}
				lapFreePinned(host_start);
			}

			double lower_bound = 0.0;
			double greedy_bound = 0.0;
			double upper_bound = 0.0;

			if (devices == 1)
			{
				int start, num_items, bs, gs;
				cudaStream_t stream;
				selectDevice(start, num_items, stream, bs, gs, 0, iterator);


				for (int i = 0; i < dim; i++)
				{
					tt[0] = iterator.getRow(0, i, true);
					if (bs == 32) getMinMaxBestSingleSmall_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], min_cost_private[0], max_cost_private[0], picked_cost_private[0], jmin_private[0], tt[0], picked_private[0], std::numeric_limits<SC>::lowest(), std::numeric_limits<SC>::max(), i, num_items, dim2);
					else if (bs == 256) getMinMaxBestSingleMedium_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], min_cost_private[0], max_cost_private[0], picked_cost_private[0], jmin_private[0], tt[0], picked_private[0], std::numeric_limits<SC>::lowest(), std::numeric_limits<SC>::max(), i, num_items, dim2);
					else getMinMaxBestSingleLarge_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], min_cost_private[0], max_cost_private[0], picked_cost_private[0], jmin_private[0], tt[0], picked_private[0], std::numeric_limits<SC>::lowest(), std::numeric_limits<SC>::max(), i, num_items, dim2);

					updateEstimatedV_kernel<<<gs, bs, 0, stream>>>(i, v_private[0], mod_v_private[0], tt[0], picked_private[0], min_cost_private[0], jmin_private[0], dim2);
				}

				for (int i = dim; i < dim2; i++)
				{
					if (bs == 32) getMinMaxBestSingleSmallVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], min_cost_private[0], max_cost_private[0], picked_cost_private[0], jmin_private[0], picked_private[0], std::numeric_limits<SC>::lowest(), std::numeric_limits<SC>::max(), i, num_items, dim2);
					else if (bs == 256) getMinMaxBestSingleMediumVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], min_cost_private[0], max_cost_private[0], picked_cost_private[0], jmin_private[0], picked_private[0], std::numeric_limits<SC>::lowest(), std::numeric_limits<SC>::max(), i, num_items, dim2);
					else getMinMaxBestSingleLargeVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], min_cost_private[0], max_cost_private[0], picked_cost_private[0], jmin_private[0], picked_private[0], std::numeric_limits<SC>::lowest(), std::numeric_limits<SC>::max(), i, num_items, dim2);

					updateEstimatedVVirtual_kernel<<<gs, bs, 0, stream>>>(i, v_private[0], mod_v_private[0], picked_private[0], min_cost_private[0], jmin_private[0], dim2);
				}

				// no perf issue here
				checkCudaErrors(cudaStreamSynchronize(stream));
				for (int i = 0; i < dim2; i++)
				{
					lower_bound += host_struct_private[i].min;
					upper_bound += host_struct_private[i].max;
					greedy_bound += host_struct_private[i].picked;
				}
				findMaximum(v_private[0], gpu_struct_private[0], stream, dim2);
				subtractMaximum_kernel<<<gs, bs, 0, stream>>>(v_private[0], gpu_struct_private[0], dim2);
			}
#ifdef LAP_CUDA_OPENMP
			else
			{
				SC max_v;
				memset(data_valid, 0, dim * devices * sizeof(int));
#pragma omp parallel num_threads(devices) shared(max_v)
				{
					int t = omp_get_thread_num();
					int start, num_items, bs, gs;
					cudaStream_t stream;
					selectDevice(start, num_items, stream, bs, gs, t, iterator);


					for (int i = 0; i < dim; i++)
					{
						tt[t] = iterator.getRow(t, i, true);

						if (bs == 32) getMinMaxBestSmall_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[t + i * devices]), gpu_struct_private[t], semaphore_private[t], &(data_valid[t + i * devices]), min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], tt[t], picked_private[t], std::numeric_limits<SC>::lowest(), std::numeric_limits<SC>::max(), i, start, num_items, dim2);
						else if (bs == 256) getMinMaxBestMedium_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[t + i * devices]), gpu_struct_private[t], semaphore_private[t], &(data_valid[t + i * devices]), min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], tt[t], picked_private[t], std::numeric_limits<SC>::lowest(), std::numeric_limits<SC>::max(), i, start, num_items, dim2);
						else getMinMaxBestLarge_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[t + i * devices]), gpu_struct_private[t], semaphore_private[t], &(data_valid[t + i * devices]), min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], tt[t], picked_private[t], std::numeric_limits<SC>::lowest(), std::numeric_limits<SC>::max(), i, start, num_items, dim2);

						if (devices > 32) updateEstimatedVLarge_kernel<<<gs, bs, 0, stream>>>(i, v_private[t], mod_v_private[t], gpu_struct_private[t], semaphore_private[t], tt[t], picked_private[t], &(data_valid[i * devices]), &(host_struct_private[i * devices]), start, num_items, dim2, std::numeric_limits<SC>::max(), devices);
						else updateEstimatedVSmall_kernel<<<gs, bs, 0, stream>>>(i, v_private[t], mod_v_private[t], gpu_struct_private[t], semaphore_private[t], tt[t], picked_private[t], &(data_valid[i * devices]), &(host_struct_private[i * devices]), start, num_items, dim2, std::numeric_limits<SC>::max(), devices);
					}

					for (int i = dim; i < dim2; i++)
					{
						if (bs == 32) getMinMaxBestSmallVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[t + i * devices]), gpu_struct_private[t], semaphore_private[t], &(data_valid[t + i * devices]), min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], picked_private[t], std::numeric_limits<SC>::lowest(), std::numeric_limits<SC>::max(), i, start, num_items, dim2);
						else if (bs == 256) getMinMaxBestMediumVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[t + i * devices]), gpu_struct_private[t], semaphore_private[t], &(data_valid[t + i * devices]), min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], picked_private[t], std::numeric_limits<SC>::lowest(), std::numeric_limits<SC>::max(), i, start, num_items, dim2);
						else getMinMaxBestLargeVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[t + i * devices]), gpu_struct_private[t], semaphore_private[t], &(data_valid[t + i * devices]), min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], picked_private[t], std::numeric_limits<SC>::lowest(), std::numeric_limits<SC>::max(), i, start, num_items, dim2);

						if (devices > 32) updateEstimatedVLargeVirtual_kernel<<<gs, bs, 0, stream>>>(i, v_private[t], mod_v_private[t], gpu_struct_private[t], semaphore_private[t], picked_private[t], &(data_valid[i * devices]), &(host_struct_private[i * devices]), start, num_items, dim2, std::numeric_limits<SC>::max(), devices);
						else updateEstimatedVSmallVirtual_kernel<<<gs, bs, 0, stream>>>(i, v_private[t], mod_v_private[t], gpu_struct_private[t], semaphore_private[t], picked_private[t], &(data_valid[i * devices]), &(host_struct_private[i * devices]), start, num_items, dim2, std::numeric_limits<SC>::max(), devices);
					}
					// no perf issue here
					checkCudaErrors(cudaStreamSynchronize(stream));
#pragma omp barrier
					if (t == 0)
					{
						for (int i = 0; i < dim2; i++)
						{
							SC t_min_cost = host_struct_private[i * devices].min;
							SC t_max_cost = host_struct_private[i * devices].max;
							SC t_picked_cost = host_struct_private[i * devices].picked;
							int t_jmin = host_struct_private[i * devices].jmin;

							// read additional values
							for (int ti = 1; ti < devices; ti++)
							{
								SC c_min_cost = host_struct_private[ti + i * devices].min;
								SC c_max_cost = host_struct_private[ti + i * devices].max;
								SC c_picked_cost = host_struct_private[ti + i * devices].picked;
								int c_jmin = host_struct_private[ti + i * devices].jmin;
								if (c_min_cost < t_min_cost) t_min_cost = c_min_cost;
								if (c_max_cost > t_max_cost) t_max_cost = c_max_cost;
								if ((c_picked_cost < t_picked_cost) || ((c_picked_cost == t_picked_cost) && (c_jmin < t_jmin)))
								{
									t_jmin = c_jmin;
									t_picked_cost = c_picked_cost;
								}
							}

							lower_bound += t_min_cost;
							upper_bound += t_max_cost;
							greedy_bound += t_picked_cost;
						}
					}
#pragma omp barrier
					findMaximum(v_private[t], &(host_struct_private[t]), stream, num_items);
					// no perf issue here
					checkCudaErrors(cudaStreamSynchronize(stream));
#pragma omp barrier
					max_v = mergeMaximum<SC>(host_struct_private, devices);
					subtractMaximum_kernel<<<gs, bs, 0, stream>>>(v_private[t], max_v, num_items);
				}
			}
#endif

			greedy_bound = std::min(greedy_bound, upper_bound);

			double initial_gap = upper_bound - lower_bound;
			double greedy_gap = greedy_bound - lower_bound;
			double initial_greedy_gap = greedy_gap;

#ifdef LAP_DEBUG
			{
				std::stringstream ss;
				ss << "upper_bound = " << upper_bound << " lower_bound = " << lower_bound << " initial_gap = " << initial_gap;
				lap::displayTime(start_time, ss.str().c_str(), lapDebug);
			}
			{
				std::stringstream ss;
				ss << "upper_bound = " << greedy_bound << " lower_bound = " << lower_bound << " greedy_gap = " << greedy_gap << " ratio = " << greedy_gap / initial_gap;
				lap::displayTime(start_time, ss.str().c_str(), lapDebug);
			}
#endif

			lower_bound = 0.0;
			upper_bound = 0.0;

			if (devices == 1)
			{
				int start, num_items, bs, gs;
				cudaStream_t stream;
				selectDevice(start, num_items, stream, bs, gs, 0, iterator);

				checkCudaErrors(cudaMemsetAsync(picked_private[0], 0, num_items * sizeof(int), stream));

				for (int i = dim2 - 1; i >= dim; --i)
				{
					if (bs == 32) getMinSecondBestSingleSmallVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], min_cost_private[0], max_cost_private[0], picked_cost_private[0], jmin_private[0], v_private[0], picked_private[0], std::numeric_limits<SC>::max(), i, num_items, dim2);
					else if (bs == 256) getMinSecondBestSingleMediumVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], min_cost_private[0], max_cost_private[0], picked_cost_private[0], jmin_private[0], v_private[0], picked_private[0], std::numeric_limits<SC>::max(), i, num_items, dim2);
					else getMinSecondBestSingleLargeVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], min_cost_private[0], max_cost_private[0], picked_cost_private[0], jmin_private[0], v_private[0], picked_private[0], std::numeric_limits<SC>::max(), i, num_items, dim2);
				}
				for (int i = dim - 1; i >= 0; --i)
				{
					tt[0] = iterator.getRow(0, i, true);

					if (bs == 32) getMinSecondBestSingleSmall_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], min_cost_private[0], max_cost_private[0], picked_cost_private[0], jmin_private[0], tt[0], v_private[0], picked_private[0], std::numeric_limits<SC>::max(), i, num_items, dim2);
					else if (bs == 256) getMinSecondBestSingleMedium_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], min_cost_private[0], max_cost_private[0], picked_cost_private[0], jmin_private[0], tt[0], v_private[0], picked_private[0], std::numeric_limits<SC>::max(), i, num_items, dim2);
					else getMinSecondBestSingleLarge_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], min_cost_private[0], max_cost_private[0], picked_cost_private[0], jmin_private[0], tt[0], v_private[0], picked_private[0], std::numeric_limits<SC>::max(), i, num_items, dim2);
				}
				// no perf issue here
				checkCudaErrors(cudaStreamSynchronize(stream));

				for (int i = 0; i < dim2; i++)
				{
					SC min_cost = host_struct_private[i].min;
					SC second_cost = host_struct_private[i].max;
					SC picked_cost = host_struct_private[i].picked;
					SC v_jmin = host_struct_private[i].v_jmin;

					perm[i] = i;
					mod_v[i] = second_cost - min_cost;
					// need to use the same v values in total
					lower_bound += min_cost + v_jmin;
					upper_bound += picked_cost + v_jmin;
				}
			}
#ifdef LAP_CUDA_OPENMP
			else
			{
#pragma omp parallel num_threads(devices)
				{
					int t = omp_get_thread_num();
					int start, num_items, bs, gs;
					cudaStream_t stream;
					selectDevice(start, num_items, stream, bs, gs, t, iterator);

					checkCudaErrors(cudaMemsetAsync(picked_private[t], 0, num_items * sizeof(int), stream));

					for (int i = 0; i < dim2; i++) host_struct_private[i * devices + t].jmin = -1;
#pragma omp barrier
					for (int i = dim2 - 1; i >= dim; --i)
					{
						if (i == dim2 - 1)
						{
							if (bs == 32) getMinSecondBestSmallVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], v_private[t], picked_private[t], std::numeric_limits<SC>::max(), i, start, num_items, dim2);
							else if (bs == 256) getMinSecondBestMediumVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], v_private[t], picked_private[t], std::numeric_limits<SC>::max(), i, start, num_items, dim2);
							else getMinSecondBestLargeVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], v_private[t], picked_private[t], std::numeric_limits<SC>::max(), i, start, num_items, dim2);
						}
						else
						{
							if (devices > 32)
							{
								if (bs == 32) getMinSecondBestLargeSmallVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], v_private[t], picked_private[t], &(host_struct_private[(i + 1) * devices]), std::numeric_limits<SC>::max(), i, start, num_items, dim2, devices);
								else if (bs == 256) getMinSecondBestLargeMediumVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], v_private[t], picked_private[t], &(host_struct_private[(i + 1) * devices]), std::numeric_limits<SC>::max(), i, start, num_items, dim2, devices);
								else getMinSecondBestLargeLargeVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], v_private[t], picked_private[t], &(host_struct_private[(i + 1) * devices]), std::numeric_limits<SC>::max(), i, start, num_items, dim2, devices);
							}
							else
							{
								if (bs == 32) getMinSecondBestSmallSmallVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], v_private[t], picked_private[t], &(host_struct_private[(i + 1) * devices]), std::numeric_limits<SC>::max(), i, start, num_items, dim2, devices);
								else if (bs == 256) getMinSecondBestSmallMediumVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], v_private[t], picked_private[t], &(host_struct_private[(i + 1) * devices]), std::numeric_limits<SC>::max(), i, start, num_items, dim2, devices);
								else getMinSecondBestSmallLargeVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], v_private[t], picked_private[t], &(host_struct_private[(i + 1) * devices]), std::numeric_limits<SC>::max(), i, start, num_items, dim2, devices);
							}
						}
					}
					for (int i = dim - 1; i >= 0; --i)
					{
						tt[t] = iterator.getRow(t, i, true);

						if (i == dim2 - 1)
						{
							if (bs == 32) getMinSecondBestSmall_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], tt[t], v_private[t], picked_private[t], std::numeric_limits<SC>::max(), i, start, num_items, dim2);
							else if (bs == 256) getMinSecondBestMedium_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], tt[t], v_private[t], picked_private[t], std::numeric_limits<SC>::max(), i, start, num_items, dim2);
							else getMinSecondBestLarge_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], tt[t], v_private[t], picked_private[t], std::numeric_limits<SC>::max(), i, start, num_items, dim2);
						}
						else
						{
							if (devices > 32)
							{
								if (bs == 32) getMinSecondBestLargeSmall_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], tt[t], v_private[t], picked_private[t], &(host_struct_private[(i + 1) * devices]), std::numeric_limits<SC>::max(), i, start, num_items, dim2, devices);
								else if (bs == 256) getMinSecondBestLargeMedium_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], tt[t], v_private[t], picked_private[t], &(host_struct_private[(i + 1) * devices]), std::numeric_limits<SC>::max(), i, start, num_items, dim2, devices);
								else getMinSecondBestLargeLarge_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], tt[t], v_private[t], picked_private[t], &(host_struct_private[(i + 1) * devices]), std::numeric_limits<SC>::max(), i, start, num_items, dim2, devices);
							}
							else
							{
								if (bs == 32) getMinSecondBestSmallSmall_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], tt[t], v_private[t], picked_private[t], &(host_struct_private[(i + 1) * devices]), std::numeric_limits<SC>::max(), i, start, num_items, dim2, devices);
								else if (bs == 256) getMinSecondBestSmallMedium_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], tt[t], v_private[t], picked_private[t], &(host_struct_private[(i + 1) * devices]), std::numeric_limits<SC>::max(), i, start, num_items, dim2, devices);
								else getMinSecondBestSmallLarge_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], tt[t], v_private[t], picked_private[t], &(host_struct_private[(i + 1) * devices]), std::numeric_limits<SC>::max(), i, start, num_items, dim2, devices);
							}
						}
					}
					// no perf issue here
					checkCudaErrors(cudaStreamSynchronize(stream));
				}
				for (int i = 0; i < dim2; i++)
				{
					SC min_cost = host_struct_private[i * devices].min;
					SC second_cost = host_struct_private[i * devices].max;
					SC picked_cost = host_struct_private[i * devices].picked;
					SC v_jmin = host_struct_private[i * devices].v_jmin;
					int jmin = host_struct_private[i * devices].jmin;

					// read additional values
					for (int ti = 1; ti < devices; ti++)
					{
						SC c_min_cost = host_struct_private[i * devices + ti].min;
						SC c_second_cost = host_struct_private[i * devices + ti].max;
						SC c_picked_cost = host_struct_private[i * devices + ti].picked;
						SC c_vjmin = host_struct_private[i * devices + ti].v_jmin;
						int c_jmin = host_struct_private[i * devices + ti].jmin;
						if (c_min_cost < min_cost)
						{
							if (min_cost < c_second_cost) second_cost = min_cost;
							else second_cost = c_second_cost;
							min_cost = c_min_cost;
						}
						else if (c_min_cost < second_cost) second_cost = c_min_cost;
						if ((c_picked_cost < picked_cost) || ((c_picked_cost == picked_cost) && (c_jmin < jmin)))
						{
							jmin = c_jmin;
							picked_cost = c_picked_cost;
							v_jmin = c_vjmin;
						}
					}

					perm[i] = i;
					mod_v[i] = second_cost - min_cost;
					// need to use the same v values in total
					lower_bound += min_cost + v_jmin;
					upper_bound += picked_cost + v_jmin;
				}
			}
#endif
			upper_bound = greedy_bound = std::min(upper_bound, greedy_bound);

			greedy_gap = upper_bound - lower_bound;

#ifdef LAP_DEBUG
			{
				std::stringstream ss;
				ss << "upper_bound = " << upper_bound << " lower_bound = " << lower_bound << " greedy_gap = " << greedy_gap << " ratio = " << greedy_gap / initial_gap;
				lap::displayTime(start_time, ss.str().c_str(), lapDebug);
			}
#endif

			if (initial_gap < 4.0 * greedy_gap)
			{
				// sort permutation by keys
				std::sort(perm, perm + dim, [&mod_v](int a, int b) { return (mod_v[a] > mod_v[b]) || ((mod_v[a] == mod_v[b]) && (a > b)); });

				lower_bound = 0.0;
				upper_bound = 0.0;
				// greedy search
				if (devices == 1)
				{
					int start, num_items, bs, gs;
					cudaStream_t stream;
					selectDevice(start, num_items, stream, bs, gs, 0, iterator);

					checkCudaErrors(cudaMemcpyAsync(mod_v_private[0], v_private[0], dim2 * sizeof(SC), cudaMemcpyDeviceToDevice, stream));
					checkCudaErrors(cudaMemsetAsync(picked_private[0], 0, dim2 * sizeof(int), stream));

					for (int i = 0; i < dim2; i++)
					{
						if (perm[i] < dim)
						{
							tt[0] = iterator.getRow(0, perm[i], true);

							if (bs == 32) getMinimalCostSingleSmall_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], picked_cost_private[0], jmin_private[0], min_cost_private[0], tt[0], v_private[0], picked_private[0], std::numeric_limits<SC>::max(), num_items, dim2);
							else if (bs == 256) getMinimalCostSingleMedium_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], picked_cost_private[0], jmin_private[0], min_cost_private[0], tt[0], v_private[0], picked_private[0], std::numeric_limits<SC>::max(), num_items, dim2);
							else getMinimalCostSingleLarge_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], picked_cost_private[0], jmin_private[0], min_cost_private[0], tt[0], v_private[0], picked_private[0], std::numeric_limits<SC>::max(), num_items, dim2);
						}
						else
						{
							if (bs == 32) getMinimalCostSingleSmallVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], picked_cost_private[0], jmin_private[0], min_cost_private[0], v_private[0], picked_private[0], std::numeric_limits<SC>::max(), num_items, dim2);
							else if (bs == 256) getMinimalCostSingleMediumVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], picked_cost_private[0], jmin_private[0], min_cost_private[0], v_private[0], picked_private[0], std::numeric_limits<SC>::max(), num_items, dim2);
							else getMinimalCostSingleLargeVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], picked_cost_private[0], jmin_private[0], min_cost_private[0], v_private[0], picked_private[0], std::numeric_limits<SC>::max(), num_items, dim2);
						}
					}

					// no perf issue here
					checkCudaErrors(cudaStreamSynchronize(stream));

					for (int i = 0; i < dim2; i++)
					{
						SC min_cost = host_struct_private[i].picked;
						SC min_cost_real = host_struct_private[i].min;
						int jmin = host_struct_private[i].jmin;
						SC v_jmin = host_struct_private[i].v_jmin;

						upper_bound += min_cost + v_jmin;
						// need to use the same v values in total
						lower_bound += min_cost_real + v_jmin;

						picked[i] = jmin;
					}
				}
#ifdef LAP_CUDA_OPENMP
				else
				{
#pragma omp parallel num_threads(devices)
					{
						int t = omp_get_thread_num();

						int start, num_items, bs, gs;
						cudaStream_t stream;
						selectDevice(start, num_items, stream, bs, gs, t, iterator);

						checkCudaErrors(cudaMemcpyAsync(mod_v_private[t], v_private[t], num_items * sizeof(SC), cudaMemcpyDeviceToDevice, stream));
						checkCudaErrors(cudaMemsetAsync(picked_private[t], 0, num_items * sizeof(int), stream));
						for (int i = 0; i < dim2; i++) host_struct_private[i * devices + t].jmin = -1;
#pragma omp barrier
						for (int i = 0; i < dim2; i++)
						{
							if (perm[i] < dim)
							{
								tt[t] = iterator.getRow(t, perm[i], true);

								if (i == 0)
								{
									if (bs == 32) getMinimalCostSmall_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], picked_cost_private[t], jmin_private[t], min_cost_private[t], tt[t], v_private[t], picked_private[t], std::numeric_limits<SC>::max(), start, num_items, dim2);
									else if (bs == 256) getMinimalCostMedium_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], picked_cost_private[t], jmin_private[t], min_cost_private[t], tt[t], v_private[t], picked_private[t], std::numeric_limits<SC>::max(), start, num_items, dim2);
									else getMinimalCostLarge_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], picked_cost_private[t], jmin_private[t], min_cost_private[t], tt[t], v_private[t], picked_private[t], std::numeric_limits<SC>::max(), start, num_items, dim2);
								}
								else
								{
									if (devices > 32)
									{
										if (bs == 32) getMinimalCostLargeSmall_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], picked_cost_private[t], jmin_private[t], min_cost_private[t], tt[t], v_private[t], picked_private[t], &(host_struct_private[(i - 1) * devices]), std::numeric_limits<SC>::max(), start, num_items, dim2, devices);
										else if (bs == 256) getMinimalCostLargeMedium_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], picked_cost_private[t], jmin_private[t], min_cost_private[t], tt[t], v_private[t], picked_private[t], &(host_struct_private[(i - 1) * devices]), std::numeric_limits<SC>::max(), start, num_items, dim2, devices);
										else getMinimalCostLargeLarge_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], picked_cost_private[t], jmin_private[t], min_cost_private[t], tt[t], v_private[t], picked_private[t], &(host_struct_private[(i - 1) * devices]), std::numeric_limits<SC>::max(), start, num_items, dim2, devices);
									}
									else
									{
										if (bs == 32) getMinimalCostSmallSmall_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], picked_cost_private[t], jmin_private[t], min_cost_private[t], tt[t], v_private[t], picked_private[t], &(host_struct_private[(i - 1) * devices]), std::numeric_limits<SC>::max(), start, num_items, dim2, devices);
										else if (bs == 256) getMinimalCostSmallMedium_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], picked_cost_private[t], jmin_private[t], min_cost_private[t], tt[t], v_private[t], picked_private[t], &(host_struct_private[(i - 1) * devices]), std::numeric_limits<SC>::max(), start, num_items, dim2, devices);
										else getMinimalCostSmallLarge_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], picked_cost_private[t], jmin_private[t], min_cost_private[t], tt[t], v_private[t], picked_private[t], &(host_struct_private[(i - 1) * devices]), std::numeric_limits<SC>::max(), start, num_items, dim2, devices);
									}
								}
							}
							else
							{
								if (i == 0)
								{
									if (bs == 32) getMinimalCostSmallVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], picked_cost_private[t], jmin_private[t], min_cost_private[t], v_private[t], picked_private[t], std::numeric_limits<SC>::max(), start, num_items, dim2);
									else if (bs == 256) getMinimalCostMediumVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], picked_cost_private[t], jmin_private[t], min_cost_private[t], v_private[t], picked_private[t], std::numeric_limits<SC>::max(), start, num_items, dim2);
									else getMinimalCostLargeVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], picked_cost_private[t], jmin_private[t], min_cost_private[t], v_private[t], picked_private[t], std::numeric_limits<SC>::max(), start, num_items, dim2);
								}
								else
								{
									if (devices > 32)
									{
										if (bs == 32) getMinimalCostLargeSmallVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], picked_cost_private[t], jmin_private[t], min_cost_private[t], v_private[t], picked_private[t], &(host_struct_private[(i - 1) * devices]), std::numeric_limits<SC>::max(), start, num_items, dim2, devices);
										else if (bs == 256) getMinimalCostLargeMediumVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], picked_cost_private[t], jmin_private[t], min_cost_private[t], v_private[t], picked_private[t], &(host_struct_private[(i - 1) * devices]), std::numeric_limits<SC>::max(), start, num_items, dim2, devices);
										else getMinimalCostLargeLargeVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], picked_cost_private[t], jmin_private[t], min_cost_private[t], v_private[t], picked_private[t], &(host_struct_private[(i - 1) * devices]), std::numeric_limits<SC>::max(), start, num_items, dim2, devices);
									}
									else
									{
										if (bs == 32) getMinimalCostSmallSmallVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], picked_cost_private[t], jmin_private[t], min_cost_private[t], v_private[t], picked_private[t], &(host_struct_private[(i - 1) * devices]), std::numeric_limits<SC>::max(), start, num_items, dim2, devices);
										else if (bs == 256) getMinimalCostSmallMediumVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], picked_cost_private[t], jmin_private[t], min_cost_private[t], v_private[t], picked_private[t], &(host_struct_private[(i - 1) * devices]), std::numeric_limits<SC>::max(), start, num_items, dim2, devices);
										else getMinimalCostSmallLargeVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], picked_cost_private[t], jmin_private[t], min_cost_private[t], v_private[t], picked_private[t], &(host_struct_private[(i - 1) * devices]), std::numeric_limits<SC>::max(), start, num_items, dim2, devices);
									}
								}
							}
						}
						// no perf issue here
						checkCudaErrors(cudaStreamSynchronize(stream));
					}

					for (int i = 0; i < dim2; i++)
					{
						SC t_min_cost = host_struct_private[i * devices].picked;
						SC t_min_cost_real = host_struct_private[i * devices].min;
						int t_jmin = host_struct_private[i * devices].jmin;
						SC t_vjmin = host_struct_private[i * devices].v_jmin;

						// read additional values
						for (int ti = 1; ti < devices; ti++)
						{
							SC c_min_cost = host_struct_private[i * devices + ti].picked;
							SC c_min_cost_real = host_struct_private[i * devices + ti].min;
							int c_jmin = host_struct_private[i * devices + ti].jmin;
							SC c_vjmin = host_struct_private[i * devices + ti].v_jmin;

							if ((c_min_cost < t_min_cost) || ((c_min_cost == t_min_cost) && (c_jmin < t_jmin)))
							{
								t_jmin = c_jmin;
								t_min_cost = c_min_cost;
								t_vjmin = c_vjmin;
							}
							if (c_min_cost_real < t_min_cost_real) t_min_cost_real = c_min_cost_real;
						}
						upper_bound += t_min_cost + t_vjmin;
						// need to use the same v values in total
						lower_bound += t_min_cost_real + t_vjmin;

						picked[i] = t_jmin;
					}
				}
#endif
				greedy_gap = upper_bound - lower_bound;

#ifdef LAP_DEBUG
				{
					std::stringstream ss;
					ss << "upper_bound = " << upper_bound << " lower_bound = " << lower_bound << " greedy_gap = " << greedy_gap << " ratio = " << greedy_gap / initial_gap;
					lap::displayTime(start_time, ss.str().c_str(), lapDebug);
				}
#endif

				if (devices == 1)
				{
					int start, num_items, bs, gs;
					cudaStream_t stream;
					selectDevice(start, num_items, stream, bs, gs, 0, iterator);

					for (int i = dim2 - 1; i >= 0; --i)
					{
						if (perm[i] < dim)
						{
							tt[0] = iterator.getRow(0, perm[i], true);

							if (bs == 32) updateVSingleSmall_kernel<<<gs, bs, 0, stream>>>(tt[0], v_private[0], picked_private[0], picked[i], dim2);
							else updateVSingle_kernel<<<gs, bs, 0, stream>>>(tt[0], v_private[0], picked_private[0], picked[i], dim2);
						}
						else
						{
							if (bs == 32) updateVSingleSmallVirtual_kernel<<<gs, bs, 0, stream>>>(v_private[0], picked_private[0], picked[i], dim2);
							else updateVSingleVirtual_kernel<<<gs, bs, 0, stream>>>(v_private[0], picked_private[0], picked[i], dim2);
						}
					}
					findMaximum(v_private[0], &(host_struct_private[0]), stream, dim2);
					subtractMaximum_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(v_private[0], &(host_struct_private[0]), dim2);
				}
#ifdef LAP_CUDA_OPENMP
				else
				{
					for (int i = 0; i < dim2; i++) host_struct_private[i].jmin = 0;
#pragma omp parallel num_threads(devices)
					{
						int t = omp_get_thread_num();

						int start, num_items, bs, gs;
						cudaStream_t stream;
						selectDevice(start, num_items, stream, bs, gs, t, iterator);

						checkCudaErrors(cudaMemsetAsync(&(gpu_struct_private[t]->jmin), 0, sizeof(int), stream));
						for (int i = dim2 - 1; i >= 0; --i)
						{
							if (perm[i] < dim)
							{
								tt[t] = iterator.getRow(t, perm[i], true);
								if (bs == 32) updateVMultiSmall_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), gpu_struct_private[t], semaphore_private[t], tt[t], v_private[t], picked_private[t], picked[i] - start, num_items);
								else updateVMulti_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), gpu_struct_private[t], semaphore_private[t], tt[t], v_private[t], picked_private[t], picked[i] - start, num_items);
							}
							else
							{
								if (bs == 32) updateVMultiSmallVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), gpu_struct_private[t], semaphore_private[t], v_private[t], picked_private[t], picked[i] - start, num_items);
								else updateVMultiVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), gpu_struct_private[t], semaphore_private[t], v_private[t], picked_private[t], picked[i] - start, num_items);
							}
						}
						// no perf issue here
						checkCudaErrors(cudaStreamSynchronize(stream));
#pragma omp barrier
						findMaximum(v_private[t], &(host_struct_private[t]), stream, num_items);
						// no perf issue here
						checkCudaErrors(cudaStreamSynchronize(stream));
#pragma omp barrier
						SC max_v = mergeMaximum<SC>(host_struct_private, devices);
						subtractMaximum_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(v_private[t], max_v, num_items);
					}
				}
#endif

				double old_upper_bound = upper_bound;
				double old_lower_bound = lower_bound;
				upper_bound = 0.0;
				lower_bound = 0.0;
				if (devices == 1)
				{
					int start, num_items, bs, gs;
					cudaStream_t stream;
					selectDevice(start, num_items, stream, bs, gs, 0, iterator);

					for (int i = 0; i < dim2; i++)
					{
						if (perm[i] < dim)
						{
							tt[0] = iterator.getRow(0, perm[i], true);

							if (bs == 32) getFinalCostSmall_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], min_cost_private[0], picked_cost_private[0], max_cost_private[0], tt[0], v_private[0], std::numeric_limits<SC>::max(), picked[i], num_items);
							else if (bs == 256) getFinalCostMedium_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], min_cost_private[0], picked_cost_private[0], max_cost_private[0], tt[0], v_private[0], std::numeric_limits<SC>::max(), picked[i], num_items);
							else getFinalCostLarge_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], min_cost_private[0], picked_cost_private[0], max_cost_private[0], tt[0], v_private[0], std::numeric_limits<SC>::max(), picked[i], num_items);
						}
						else
						{
							if (bs == 32) getFinalCostSmallVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], min_cost_private[0], picked_cost_private[0], max_cost_private[0], v_private[0], std::numeric_limits<SC>::max(), picked[i], num_items);
							else if (bs == 256) getFinalCostMediumVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], min_cost_private[0], picked_cost_private[0], max_cost_private[0], v_private[0], std::numeric_limits<SC>::max(), picked[i], num_items);
							else getFinalCostLargeVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], min_cost_private[0], picked_cost_private[0], max_cost_private[0], v_private[0], std::numeric_limits<SC>::max(), picked[i], num_items);
						}
					}

					// no perf issue here
					checkCudaErrors(cudaStreamSynchronize(stream));

					for (int i = 0; i < dim2; i++)
					{
						SC picked_cost = host_struct_private[i].picked;
						SC v_picked = host_struct_private[i].v_jmin;
						SC min_cost_real = host_struct_private[i].min;

						// need to use all picked v for the lower bound as well
						upper_bound += picked_cost;
						lower_bound += min_cost_real + v_picked;
					}
				}
#ifdef LAP_CUDA_OPENMP
				else
				{
#pragma omp parallel num_threads(devices)
					{
						int t = omp_get_thread_num();

						int start, num_items, bs, gs;
						cudaStream_t stream;
						selectDevice(start, num_items, stream, bs, gs, t, iterator);

						for (int i = 0; i < dim2; i++)
						{
							if (perm[i] < dim)
							{
								tt[t] = iterator.getRow(t, perm[i], true);

								if (bs == 32) getFinalCostSmall_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[t + i * devices]), semaphore_private[t], min_cost_private[t], picked_cost_private[t], max_cost_private[t], tt[t], v_private[t], std::numeric_limits<SC>::max(), picked[i] - iterator.ws.part[t].first, num_items);
								else if (bs == 256) getFinalCostMedium_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[t + i * devices]), semaphore_private[t], min_cost_private[t], picked_cost_private[t], max_cost_private[t], tt[t], v_private[t], std::numeric_limits<SC>::max(), picked[i] - iterator.ws.part[t].first, num_items);
								else getFinalCostLarge_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[t + i * devices]), semaphore_private[t], min_cost_private[t], picked_cost_private[t], max_cost_private[t], tt[t], v_private[t], std::numeric_limits<SC>::max(), picked[i] - iterator.ws.part[t].first, num_items);
							}
							else
							{
								if (bs == 32) getFinalCostSmallVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[t + i * devices]), semaphore_private[t], min_cost_private[t], picked_cost_private[t], max_cost_private[t], v_private[t], std::numeric_limits<SC>::max(), picked[i] - iterator.ws.part[t].first, num_items);
								else if (bs == 256) getFinalCostMediumVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[t + i * devices]), semaphore_private[t], min_cost_private[t], picked_cost_private[t], max_cost_private[t], v_private[t], std::numeric_limits<SC>::max(), picked[i] - iterator.ws.part[t].first, num_items);
								else getFinalCostLargeVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[t + i * devices]), semaphore_private[t], min_cost_private[t], picked_cost_private[t], max_cost_private[t], v_private[t], std::numeric_limits<SC>::max(), picked[i] - iterator.ws.part[t].first, num_items);
							}
						}
						// no perf issue here
						checkCudaErrors(cudaStreamSynchronize(stream));
					}

					for (int i = 0; i < dim2; i++)
					{
						SC picked_cost = host_struct_private[i * devices].picked;
						SC v_picked = host_struct_private[i * devices].v_jmin;
						SC min_cost_real = host_struct_private[i * devices].min;
						// read additional values
						for (int ti = 1; ti < devices; ti++)
						{
							picked_cost = std::min(picked_cost, host_struct_private[i * devices + ti].picked);
							v_picked = std::min(v_picked, host_struct_private[i * devices + ti].v_jmin);
							min_cost_real = std::min(min_cost_real, host_struct_private[i * devices + ti].min);
						}

						// need to use all picked v for the lower bound as well
						upper_bound += picked_cost;
						lower_bound += min_cost_real + v_picked;
					}
				}
#endif
				upper_bound = std::min(upper_bound, old_upper_bound);
				lower_bound = std::max(lower_bound, old_lower_bound);
				greedy_gap = upper_bound - lower_bound;

#ifdef LAP_DEBUG
				double ratio = greedy_gap / initial_gap;
				{
					std::stringstream ss;
					ss << "upper_bound = " << upper_bound << " lower_bound = " << lower_bound << " greedy_gap = " << greedy_gap << " ratio = " << ratio;
					lap::displayTime(start_time, ss.str().c_str(), lapDebug);
				}
#endif
				double ratio2 = greedy_gap / initial_greedy_gap;
				if (ratio2 > 1.0e-09)
				{
					if (devices == 1)
					{
						int start, num_items, bs, gs;
						cudaStream_t stream;
						selectDevice(start, num_items, stream, bs, gs, 0, iterator);

						interpolateV_kernel<<<gs, bs, 0, stream >>>(v_private[0], mod_v_private[0], ratio2, dim2);
					}
#ifdef LAP_CUDA_OPENMP
					else
					{
#pragma omp parallel num_threads(devices)
						{
							int t = omp_get_thread_num();

							int start, num_items, bs, gs;
							cudaStream_t stream;
							selectDevice(start, num_items, stream, bs, gs, t, iterator);

							interpolateV_kernel<<<gs, bs, 0, stream >>>(v_private[t], mod_v_private[t], ratio2, num_items);
						}
					}
#endif
				}
			}

			SC upper, lower;
			getUpperLower(upper, lower, greedy_gap, initial_gap, dim, dim2);

			for (int t = 0; t < devices; t++)
			{
				checkCudaErrors(cudaSetDevice(iterator.ws.device[t]));
				lapFreeDevice(mod_v_private[t]);
				lapFreeDevice(picked_private[t]);
				lapFreeDevice(semaphore_private[t]);
				lapFreeDevice(min_cost_private[t]);
				lapFreeDevice(max_cost_private[t]);
				lapFreeDevice(picked_cost_private[t]);
				lapFreeDevice(jmin_private[t]);
				lapFreeDevice(gpu_struct_private[t]);
				lapFreeDevice(start_private[t]);
			}

			lapFreePinned(mod_v);
			lapFree(mod_v_private);
			lapFree(min_cost_private);
			lapFree(max_cost_private);
			lapFree(picked_cost_private);
			lapFree(jmin_private);
			lapFree(picked_private);
			lapFree(picked);
			lapFree(semaphore_private);
			lapFreePinned(host_struct_private);
			lapFree(tt);
			lapFree(start_private);
			lapFree(gpu_struct_private);
			lapFreePinned(data_valid);

#ifdef LAP_CUDA_OPENMP
			if (max_threads < devices) omp_set_num_threads(max_threads);
#endif
			return std::pair<SC, SC>((SC)upper, (SC)lower);
		}

		template <class SC, class TC, class CF, class I>
		void solve(int dim, int dim2, CF &costfunc, I &iterator, int *rowsol, bool use_epsilon)

			// input:
			// dim        - problem size
			// costfunc - cost matrix
			// findcost   - searching cost matrix

			// output:
			// rowsol     - column assigned to row in solution
			// colsol     - row assigned to column in solution
			// u          - dual variables, row reduction numbers
			// v          - dual variables, column reduction numbers

		{
#ifndef LAP_QUIET
			auto start_time = std::chrono::high_resolution_clock::now();

			long long total_hit = 0LL;
			long long total_miss = 0LL;

			long long total_rows = 0LL;
			long long total_virtual = 0LL;

			int elapsed = -1;
#else
#ifdef LAP_DISPLAY_EVALUATED
			long long total_hit = 0LL;
			long long total_miss = 0LL;

			long long total_rows = 0LL;
			long long total_virtual = 0LL;
#endif
#endif

#ifdef LAP_DEBUG
			SC *v;
#endif
			SC *h_total_d;
			SC *h_total_eps;
			// for calculating h2
			SC *tt_jmin;
			SC *v_jmin;

			int devices = (int)iterator.ws.device.size();

#ifdef LAP_CUDA_OPENMP
			int old_max_threads = omp_get_max_threads();
			omp_set_num_threads(devices);
#endif

			const TC **tt;
			lapAlloc(tt, devices, __FILE__, __LINE__);

#ifdef LAP_DEBUG
			std::vector<SC *> v_list;
			std::vector<TC> eps_list;
#endif


			// used for copying
			min_struct<SC> *host_min_private;
			min_struct<SC> **gpu_min_private;
#ifdef LAP_DEBUG
			lapAllocPinned(v, dim2, __FILE__, __LINE__);
#endif
			lapAllocPinned(h_total_d, dim2, __FILE__, __LINE__);
			lapAllocPinned(h_total_eps, dim2, __FILE__, __LINE__);
			lapAllocPinned(host_min_private, devices, __FILE__, __LINE__);
			lapAllocPinned(tt_jmin, 1, __FILE__, __LINE__);
			lapAllocPinned(v_jmin, 1, __FILE__, __LINE__);

			SC **min_private;
			int **jmin_private;
			int **csol_private;
			char **colactive_private;
			int **pred_private;
			SC **d_private;
			int *pred;
			int *colsol;
			int **colsol_private;
			SC **v_private;
			SC **total_eps_private;
			SC **total_d_private;
			int **start_private;
			unsigned int **semaphore_private;
			int *perm;
			// on device
			lapAlloc(min_private, devices, __FILE__, __LINE__);
			lapAlloc(jmin_private, devices, __FILE__, __LINE__);
			lapAlloc(csol_private, devices, __FILE__, __LINE__);
			lapAlloc(colactive_private, devices, __FILE__, __LINE__);
			lapAlloc(pred_private, devices, __FILE__, __LINE__);
			lapAlloc(d_private, devices, __FILE__, __LINE__);
			lapAlloc(colsol_private, devices, __FILE__, __LINE__);
			lapAlloc(v_private, devices, __FILE__, __LINE__);
			lapAlloc(total_d_private, devices, __FILE__, __LINE__);
			lapAlloc(total_eps_private, devices, __FILE__, __LINE__);
			lapAlloc(semaphore_private, devices, __FILE__, __LINE__);
			lapAlloc(perm, dim2, __FILE__, __LINE__);
			lapAlloc(start_private, devices, __FILE__, __LINE__);
			lapAlloc(gpu_min_private, devices, __FILE__, __LINE__);
			lapAllocPinned(colsol, dim2, __FILE__, __LINE__);
			lapAllocPinned(pred, dim2, __FILE__, __LINE__);

			int *host_start;
			lapAllocPinned(host_start, devices, __FILE__, __LINE__);
			for (int t = 0; t < devices; t++)
			{
				checkCudaErrors(cudaSetDevice(iterator.ws.device[t]));
				int num_items = iterator.ws.part[t].second - iterator.ws.part[t].first;
				int count = getMinSize(num_items);

				host_start[t] = iterator.ws.part[t].first;
				lapAllocDevice(min_private[t], count, __FILE__, __LINE__);
				lapAllocDevice(jmin_private[t], count, __FILE__, __LINE__);
				lapAllocDevice(csol_private[t], count, __FILE__, __LINE__);
				lapAllocDevice(colactive_private[t], num_items, __FILE__, __LINE__);
				lapAllocDevice(d_private[t], num_items, __FILE__, __LINE__);
				lapAllocDevice(v_private[t], num_items, __FILE__, __LINE__);
				lapAllocDevice(total_d_private[t], num_items, __FILE__, __LINE__);
				lapAllocDevice(total_eps_private[t], num_items, __FILE__, __LINE__);
				lapAllocDevice(colsol_private[t], num_items, __FILE__, __LINE__);
				lapAllocDevice(pred_private[t], num_items, __FILE__, __LINE__);
				lapAllocDevice(semaphore_private[t], 2, __FILE__, __LINE__);
				lapAllocDevice(start_private[t], devices, __FILE__, __LINE__);
				lapAllocDevice(gpu_min_private[t], 1, __FILE__, __LINE__);
			}
			for (int t = 0; t < devices; t++)
			{
				checkCudaErrors(cudaSetDevice(iterator.ws.device[t]));
				int num_items = iterator.ws.part[t].second - iterator.ws.part[t].first;
				cudaStream_t stream = iterator.ws.stream[t];

				if (!use_epsilon) checkCudaErrors(cudaMemsetAsync(v_private[t], 0, sizeof(SC) * num_items, stream));
				checkCudaErrors(cudaMemsetAsync(semaphore_private[t], 0, 2 * sizeof(unsigned int), stream));
				checkCudaErrors(cudaMemcpyAsync(start_private[t], host_start, devices * sizeof(int), cudaMemcpyHostToDevice, stream));
			}

			TC epsilon_upper, epsilon_lower;

			if (use_epsilon)
			{
				std::pair<SC, SC> eps = estimateEpsilon<SC, TC, I>(dim, dim2, iterator, v_private, perm);
				epsilon_upper = (TC)eps.first;
				epsilon_lower = (TC)eps.second;
			}
			else
			{
				epsilon_upper = TC(0);
				epsilon_lower = TC(0);
			}


#ifdef LAP_ROWS_SCANNED
			unsigned long long *scancount;
			unsigned long long *pathlength;
			lapAlloc(scancount, dim2, __FILE__, __LINE__);
			lapAlloc(pathlength, dim2, __FILE__, __LINE__);
			memset(scancount, 0, dim2 * sizeof(unsigned long long));
			memset(pathlength, 0, dim2 * sizeof(unsigned long long));
#endif

			TC epsilon = epsilon_upper;

			bool first = true;
			bool second = false;
			bool reverse = true;
			bool peerEnabled = iterator.ws.peerAccess();

			if ((!use_epsilon) || (epsilon > SC(0)))
			{
				for (int i = 0; i < dim2; i++) perm[i] = i;
				reverse = false;
			}

			SC total_d = SC(0);
			SC total_eps = SC(0);
			while (epsilon >= TC(0))
			{
#ifdef LAP_DEBUG
				if (first)
				{
					for (int t = 0; t < devices; t++)
					{
						checkCudaErrors(cudaSetDevice(iterator.ws.device[t]));
						int start = iterator.ws.part[t].first;
						int end = iterator.ws.part[t].second;
						int num_items = end - start;
						cudaStream_t stream = iterator.ws.stream[t];
						checkCudaErrors(cudaMemcpyAsync(&(v[start]), v_private[t], sizeof(SC) * num_items, cudaMemcpyDeviceToHost, stream));
					}
					for (int t = 0; t < devices; t++) checkCudaErrors(cudaStreamSynchronize(iterator.ws.stream[t]));
					SC* vv;
					lapAlloc(vv, dim2, __FILE__, __LINE__);
					v_list.push_back(vv);
					eps_list.push_back(epsilon);
					memcpy(v_list.back(), v, sizeof(SC) * dim2);
				}
#endif
				getNextEpsilon(epsilon, epsilon_lower, total_d, total_eps, first, second, dim2);
				total_d = SC(0);
				total_eps = SC(0);
#ifndef LAP_QUIET
				{
					std::stringstream ss;
					ss << "eps = " << epsilon;
					const std::string tmp = ss.str();
					displayTime(start_time, tmp.c_str(), lapInfo);
				}
#endif
				// this is to ensure termination of the while statement
				if (epsilon == TC(0)) epsilon = TC(-1.0);
				memset(colsol, -1, dim2 * sizeof(int));

				for (int t = 0; t < devices; t++)
				{
					// upload v to devices
					checkCudaErrors(cudaSetDevice(iterator.ws.device[t]));
					int num_items = iterator.ws.part[t].second - iterator.ws.part[t].first;
					cudaStream_t stream = iterator.ws.stream[t];
					checkCudaErrors(cudaMemsetAsync(total_d_private[t], 0, sizeof(SC) * num_items, stream));
					checkCudaErrors(cudaMemsetAsync(total_eps_private[t], 0, sizeof(SC) * num_items, stream));
					checkCudaErrors(cudaMemsetAsync(colsol_private[t], -1, num_items * sizeof(int), stream));
				}

#ifndef LAP_QUIET
				int old_complete = 0;
#endif

#ifdef LAP_MINIMIZE_V
				//				int dim_limit = ((reverse) || (epsilon < TC(0))) ? dim2 : dim;
				int dim_limit = dim2;
#else
				int dim_limit = dim2;
#endif

				// AUGMENT SOLUTION for each free row.
#ifndef LAP_QUIET
				displayProgress(start_time, elapsed, 0, dim_limit, " rows");
#endif
				long long count = 0ll;

				if (devices == 1)
				{
					int jmin, colsol_old;
					SC min, min_n;
					bool unassignedfound;

					bool require_colsol_copy = false;
					int  endofpath = -1;

					int t = 0;
					int start, num_items, bs, gs;
					cudaStream_t stream;
					selectDevice(start, num_items, stream, bs, gs, t, iterator);

					for (int fc = 0; fc < dim_limit; fc++)
					{
						// mark as incomplete
						host_min_private[t].min = std::numeric_limits<SC>::infinity();
						int f = perm[((reverse) && (fc < dim)) ? (dim - 1 - fc) : fc];
						// start search and find minimum value
						if (require_colsol_copy)
						{
							if (f < dim)
							{
								tt[t] = iterator.getRow(t, f, false);

								if (bs == 32) initializeSearchMinSmallCopy_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt[t], colactive_private[t], colsol_private[t], colsol, pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
								else if (bs == 256) initializeSearchMinMediumCopy_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt[t], colactive_private[t], colsol_private[t], colsol, pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
								else initializeSearchMinLargeCopy_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt[t], colactive_private[t], colsol_private[t], colsol, pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
							}
							else
							{
								if (bs == 32) initializeSearchMinSmallVirtualCopy_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], colsol, pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
								else if (bs == 256) initializeSearchMinMediumVirtualCopy_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], colsol, pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
								else initializeSearchMinLargeVirtualCopy_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], colsol, pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
							}
							require_colsol_copy = false;
						}
						else
						{
							if (f < dim)
							{
								tt[t] = iterator.getRow(t, f, false);
								if (bs == 32) initializeSearchMinSmall_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt[t], colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
								else if (bs == 256) initializeSearchMinMedium_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt[t], colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
								else initializeSearchMinLarge_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt[t], colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
							}
							else
							{
								if (bs == 32) initializeSearchMinSmallVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
								else if (bs == 256) initializeSearchMinMediumVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
								else initializeSearchMinLargeVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
							}
						}
						// min is now set so we need to find the correspoding minima for free and taken columns
						checkCudaErrors(cudaStreamSynchronize(stream));
#ifndef LAP_QUIET
						if (f < dim) total_rows++; else total_virtual++;
#else
#ifdef LAP_DISPLAY_EVALUATED
						if (f < dim) total_rows++; else total_virtual++;
#endif
#endif
#ifdef LAP_ROWS_SCANNED
						scancount[f]++;
#endif
						count++;

						unassignedfound = false;

						// Dijkstra search
						min = host_min_private[t].min;
						jmin = host_min_private[t].jmin;
						colsol_old = host_min_private[t].colsol;

						// dijkstraCheck
						if (colsol_old < 0)
						{
							endofpath = jmin;
							unassignedfound = true;
						}
						else
						{
							unassignedfound = false;
						}

						bool fast = unassignedfound;

						while (!unassignedfound)
						{
							int i = colsol_old;
							if (i < dim)
							{
								// get row
								tt[t] = iterator.getRow(t, i, false);
								// continue search
								if (bs == 32) continueSearchJMinMinSmall_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt[t], colactive_private[t], colsol_private[t], pred_private[t], i, jmin, min, std::numeric_limits<SC>::max(), num_items, dim2);
								else if (bs == 256) continueSearchJMinMinMedium_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt[t], colactive_private[t], colsol_private[t], pred_private[t], i, jmin, min, std::numeric_limits<SC>::max(), num_items, dim2);
								else continueSearchJMinMinLarge_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt[t], colactive_private[t], colsol_private[t], pred_private[t], i, jmin, min, std::numeric_limits<SC>::max(), num_items, dim2);
							}
							else
							{
								// continue search
								if (bs == 32) continueSearchJMinMinSmallVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], i, jmin, min, std::numeric_limits<SC>::max(), num_items, dim, dim2);
								else if (bs == 256) continueSearchJMinMinMediumVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], i, jmin, min, std::numeric_limits<SC>::max(), num_items, dim, dim2);
								else continueSearchJMinMinLargeVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], i, jmin, min, std::numeric_limits<SC>::max(), num_items, dim, dim2);
							}
							// min is now set so we need to find the correspoding minima for free and taken columns
							checkCudaErrors(cudaStreamSynchronize(stream));
#ifndef LAP_QUIET
							if (i < dim) total_rows++; else total_virtual++;
#else
#ifdef LAP_DISPLAY_EVALUATED
							if (i < dim) total_rows++; else total_virtual++;
#endif
#endif
#ifdef LAP_ROWS_SCANNED
							scancount[i]++;
#endif
							count++;

							min_n = host_min_private[t].min;
							jmin = host_min_private[t].jmin;
							colsol_old = host_min_private[t].colsol;

							min = std::max(min, min_n);

							// dijkstraCheck
							if (colsol_old < 0)
							{
								endofpath = jmin;
								unassignedfound = true;
							}
							else
							{
								unassignedfound = false;
							}
						}

						if (fast)
						{
							colsol[endofpath] = f;
							rowsol[f] = endofpath;
							if (epsilon > TC(0))
							{
								updateColumnPricesEpsilonFast_kernel<<<gs, bs, 0, stream>>>(colactive_private[t], jmin, min, v_private[t], d_private[t], total_d_private[t], total_eps_private[t], epsilon, num_items, &(colsol_private[t][endofpath]), colsol[endofpath]);
							}
							else
							{
								updateColumnPricesFast_kernel<<<gs, bs, 0, stream>>>(colactive_private[t], jmin, min, v_private[t], d_private[t], num_items, &(colsol_private[t][endofpath]), colsol[endofpath]);
							}
						}
						else
						{
							// update column prices. can increase or decrease
							if (epsilon > TC(0))
							{
								updateColumnPricesEpsilonCopy_kernel<<<gs, bs, 0, stream>>>(colactive_private[t], jmin, min, v_private[t], d_private[t], total_d_private[t], total_eps_private[t], epsilon, pred, pred_private[t], num_items);
							}
							else
							{
								updateColumnPricesCopy_kernel<<<gs, bs, 0, stream>>>(colactive_private[t], jmin, min, v_private[t], d_private[t], pred, pred_private[t], num_items);
							}
							// reset row and column assignments along the alternating path.
							checkCudaErrors(cudaStreamSynchronize(stream));
#ifdef LAP_ROWS_SCANNED
							{
								int i;
								int eop = endofpath;
								do
								{
									i = pred[eop];
									eop = rowsol[i];
									if (i != f) pathlength[f]++;
								} while (i != f);
							}
#endif
							resetRowColumnAssignment(endofpath, f, pred, rowsol, colsol);
							require_colsol_copy = true;
						}
#ifndef LAP_QUIET
						{
							int level;
							if ((level = displayProgress(start_time, elapsed, fc + 1, dim_limit, " rows")) != 0)
							{
								long long hit, miss;
								iterator.getHitMiss(hit, miss);
								total_hit += hit;
								total_miss += miss;
								if ((hit != 0) || (miss != 0))
								{
									if (level == 1) lapInfo << "  hit: " << hit << " miss: " << miss << " (" << miss - (fc + 1 - old_complete) << " + " << fc + 1 - old_complete << ")" << std::endl;
									else lapDebug << "  hit: " << hit << " miss: " << miss << " (" << miss - (fc + 1 - old_complete) << " + " << fc + 1 - old_complete << ")" << std::endl;
								}
								old_complete = fc + 1;
							}
						}
#endif
					}

#ifdef LAP_MINIMIZE_V
					if (epsilon > TC(0))
					{
						if (dim_limit < dim2)
						{
							if (require_colsol_copy)
							{
								checkCudaErrors(cudaMemcpyAsync(colsol_private[t], &(colsol[start]), num_items * sizeof(int), cudaMemcpyHostToDevice, stream));
							}
							findMaximum(v_private[0], colsol_private[0], &(host_min_private[0]), stream, dim2);
							subtractMaximumLimited_kernel<<<gs, bs, 0, stream>>>(v_private[0], &(host_min_private[0]), dim2);
						}
						else
						{
							findMaximum(v_private[0], &(host_min_private[0]), stream, dim2);
							subtractMaximum_kernel<<<gs, bs, 0, stream>>>(v_private[0], &(host_min_private[0]), dim2);
						}
					}
#endif

					// download updated v
					checkCudaErrors(cudaMemcpyAsync(&(h_total_d[start]), total_d_private[t], sizeof(SC) * num_items, cudaMemcpyDeviceToHost, stream));
					checkCudaErrors(cudaMemcpyAsync(&(h_total_eps[start]), total_eps_private[t], sizeof(SC) * num_items, cudaMemcpyDeviceToHost, stream));
#ifdef LAP_DEBUG
					checkCudaErrors(cudaMemcpyAsync(&(v[start]), v_private[t], sizeof(SC) * num_items, cudaMemcpyDeviceToHost, stream));
#endif
					checkCudaErrors(cudaStreamSynchronize(stream));
				}
#ifdef LAP_CUDA_OPENMP
				else /* devices > 1*/
				{
					int triggered = -1;
					int start_t = -1;
#pragma omp parallel num_threads(devices) shared(triggered, start_t)
					{
						int jmin_local, colsol_old_local;
						SC min_local, min_n_local;
						bool unassignedfound_local;

						bool require_colsol_copy_local = false;
						int  endofpath_local = -1;

						int t = omp_get_thread_num();
						int start, num_items, bs, gs;
						cudaStream_t stream;
						selectDevice(start, num_items, stream, bs, gs, t, iterator);

						for (int fc = 0; fc < dim_limit; fc++)
						{
							int f = perm[((reverse) && (fc < dim)) ? (dim - 1 - fc) : fc];
#pragma omp barrier
							// start search and find minimum value, no difference if peer access is enabled
							if (require_colsol_copy_local)
							{
								if (f < dim)
								{
									tt[t] = iterator.getRow(t, f, false);
									if (bs == 32) initializeSearchMinSmallCopyRemote_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt[t], colactive_private[t], colsol_private[t], colsol + start, pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
									else if (bs == 256) initializeSearchMinMediumCopyRemote_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt[t], colactive_private[t], colsol_private[t], colsol + start, pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
									else initializeSearchMinLargeCopyRemote_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt[t], colactive_private[t], colsol_private[t], colsol + start, pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
								}
								else
								{
									if (bs == 32) initializeSearchMinSmallVirtualCopyRemote_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], colsol + start, pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
									else if (bs == 256) initializeSearchMinMediumVirtualCopyRemote_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], colsol + start, pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
									else initializeSearchMinLargeVirtualCopyRemote_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], colsol + start, pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
								}
							}
							else
							{
								if (f < dim)
								{
									tt[t] = iterator.getRow(t, f, false);
									if (bs == 32) initializeSearchMinSmallRemote_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt[t], colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
									else if (bs == 256) initializeSearchMinMediumRemote_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt[t], colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
									else initializeSearchMinLargeRemote_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt[t], colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
								}
								else
								{
									if (bs == 32) initializeSearchMinSmallVirtualRemote_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
									else if (bs == 256) initializeSearchMinMediumVirtualRemote_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
									else initializeSearchMinLargeVirtualRemote_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
								}
							}
							checkCudaErrors(cudaStreamSynchronize(stream));
#pragma omp barrier
							// Dijkstra search
							min_local = host_min_private[0].min;
							jmin_local = host_min_private[0].jmin;
							colsol_old_local = host_min_private[0].colsol;

							// read additional values
							for (int ti = 1; ti < devices; ti++)
							{
								SC c_min = host_min_private[ti].min;
								int c_jmin = host_min_private[ti].jmin + iterator.ws.part[ti].first;
								int c_colsol = host_min_private[ti].colsol;
								if ((c_min < min_local) || ((c_min == min_local) && (colsol_old_local >= 0) && (c_colsol < 0)))
								{
									min_local = c_min;
									jmin_local = c_jmin;
									colsol_old_local = c_colsol;
								}
							}

							require_colsol_copy_local = false;
							if (t == 0)
							{
#ifndef LAP_QUIET
								if (f < dim) total_rows++; else total_virtual++;
#else
#ifdef LAP_DISPLAY_EVALUATED
								if (f < dim) total_rows++; else total_virtual++;
#endif
#endif
#ifdef LAP_ROWS_SCANNED
								scancount[f]++;
#endif
								count++;
							}

							unassignedfound_local = false;

							// dijkstraCheck
							if (colsol_old_local < 0)
							{
								endofpath_local = jmin_local;
								unassignedfound_local = true;
							}
							else
							{
								unassignedfound_local = false;
							}

							bool fast = unassignedfound_local;
							while (!unassignedfound_local)
							{
								// update 'distances' between freerow and all unscanned columns, via next scanned column.
								int i = colsol_old_local;

								if ((jmin_local >= start) && (jmin_local < start + num_items))
								{
									triggered = t;
									start_t = start;
									host_min_private[triggered].data_valid = 0;
								}

								if (i < dim)
								{
									// continue search
									if (peerEnabled)
									{
										// get row
										tt[t] = iterator.getRow(t, i, false);
#pragma omp barrier
										if (bs == 32) continueSearchMinPeerSmall_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], &(host_min_private[triggered].data_valid), min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt[t], colactive_private[t], colsol_private[t], pred_private[t], i, &(tt[triggered][jmin_local - start_t]), &(v_private[triggered][jmin_local - start_t]), jmin_local - start, min_local, std::numeric_limits<SC>::max(), num_items, dim2);
										else if (bs == 256) continueSearchMinPeerMedium_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], &(host_min_private[triggered].data_valid), min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt[t], colactive_private[t], colsol_private[t], pred_private[t], i, &(tt[triggered][jmin_local - start_t]), &(v_private[triggered][jmin_local - start_t]), jmin_local - start, min_local, std::numeric_limits<SC>::max(), num_items, dim2);
										else continueSearchMinPeerLarge_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], &(host_min_private[triggered].data_valid), min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt[t], colactive_private[t], colsol_private[t], pred_private[t], i, &(tt[triggered][jmin_local - start_t]), &(v_private[triggered][jmin_local - start_t]), jmin_local - start, min_local, std::numeric_limits<SC>::max(), num_items, dim2);
									}
									else
									{
#pragma omp barrier
										// get row
										tt[t] = iterator.getRow(t, i, false);
										if (bs == 32) continueSearchMinSmall_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], &(host_min_private[triggered].data_valid), min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt[t], colactive_private[t], colsol_private[t], pred_private[t], i, tt_jmin, v_jmin, jmin_local - start, min_local, std::numeric_limits<SC>::max(), num_items, dim2);
										else if (bs == 256) continueSearchMinMedium_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], &(host_min_private[triggered].data_valid), min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt[t], colactive_private[t], colsol_private[t], pred_private[t], i, tt_jmin, v_jmin, jmin_local - start, min_local, std::numeric_limits<SC>::max(), num_items, dim2);
										else continueSearchMinLarge_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], &(host_min_private[triggered].data_valid), min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt[t], colactive_private[t], colsol_private[t], pred_private[t], i, tt_jmin, v_jmin, jmin_local - start, min_local, std::numeric_limits<SC>::max(), num_items, dim2);
									}
								}
								else
								{
#pragma omp barrier
									// continue search
									if (peerEnabled)
									{
										if (bs == 32) continueSearchMinPeerSmall_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], i, &(v_private[triggered][jmin_local - start_t]), jmin_local - start, min_local, std::numeric_limits<SC>::max(), num_items, dim, dim2);
										else if (bs == 256) continueSearchMinPeerMedium_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], i, &(v_private[triggered][jmin_local - start_t]), jmin_local - start, min_local, std::numeric_limits<SC>::max(), num_items, dim, dim2);
										else continueSearchMinPeerLarge_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], i, &(v_private[triggered][jmin_local - start_t]), jmin_local - start, min_local, std::numeric_limits<SC>::max(), num_items, dim, dim2);
									}
									else
									{
										if (bs == 32) continueSearchMinSmall_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], &(host_min_private[triggered].data_valid), min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], i, v_jmin, jmin_local - start, min_local, std::numeric_limits<SC>::max(), num_items, dim, dim2);
										else if (bs == 256) continueSearchMinMedium_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], &(host_min_private[triggered].data_valid), min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], i, v_jmin, jmin_local - start, min_local, std::numeric_limits<SC>::max(), num_items, dim, dim2);
										else continueSearchMinLarge_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], &(host_min_private[triggered].data_valid), min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], i, v_jmin, jmin_local - start, min_local, std::numeric_limits<SC>::max(), num_items, dim, dim2);
									}
								}
								checkCudaErrors(cudaStreamSynchronize(stream));
#pragma omp barrier
								min_n_local = host_min_private[0].min;
								jmin_local = host_min_private[0].jmin;
								colsol_old_local = host_min_private[0].colsol;

								// read additional values
								for (int ti = 1; ti < devices; ti++)
								{
									SC c_min = host_min_private[ti].min;
									int c_jmin = host_min_private[ti].jmin + iterator.ws.part[ti].first;
									int c_colsol = host_min_private[ti].colsol;
									if ((c_min < min_n_local) || ((c_min == min_n_local) && (colsol_old_local >= 0) && (c_colsol < 0)))
									{
										min_n_local = c_min;
										jmin_local = c_jmin;
										colsol_old_local = c_colsol;
									}
								}
								if (t == 0)
								{
#ifndef LAP_QUIET
									if (i < dim) total_rows++; else total_virtual++;
#else
#ifdef LAP_DISPLAY_EVALUATED
									if (i < dim) total_rows++; else total_virtual++;
#endif
#endif
#ifdef LAP_ROWS_SCANNED
									scancount[i]++;
#endif
									count++;
								}

								min_local = std::max(min_local, min_n_local);

								// dijkstraCheck
								if (colsol_old_local < 0)
								{
									endofpath_local = jmin_local;
									unassignedfound_local = true;
								}
								else
								{
									unassignedfound_local = false;
								}
							}

							// update column prices. can increase or decrease
							if (fast)
							{
								if ((endofpath_local >= start) && (endofpath_local < start + num_items))
								{
									colsol[endofpath_local] = f;
									rowsol[f] = endofpath_local;
									if (epsilon > TC(0))
									{
										updateColumnPricesEpsilonFast_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(colactive_private[t], jmin_local - start, min_local, v_private[t], d_private[t], total_d_private[t], total_eps_private[t], epsilon, num_items, &(colsol_private[t][endofpath_local - start]), f);
									}
									else
									{
										updateColumnPricesFast_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(colactive_private[t], jmin_local - start, min_local, v_private[t], d_private[t], num_items, &(colsol_private[t][endofpath_local - start]), f);
									}
								}
								else
								{
									if (epsilon > TC(0))
									{
										updateColumnPricesEpsilon_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(colactive_private[t], jmin_local - start, min_local, v_private[t], d_private[t], total_d_private[t], total_eps_private[t], epsilon, num_items);
									}
									else
									{
										updateColumnPrices_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(colactive_private[t], jmin_local - start, min_local, v_private[t], d_private[t], num_items);
									}
								}
							}
							else
							{
								if (epsilon > TC(0))
								{
									updateColumnPricesEpsilonCopy_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(colactive_private[t], jmin_local - start, min_local, v_private[t], d_private[t], total_d_private[t], total_eps_private[t], epsilon, pred + start, pred_private[t], num_items);
								}
								else
								{
									updateColumnPricesCopy_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(colactive_private[t], jmin_local - start, min_local, v_private[t], d_private[t], pred + start, pred_private[t], num_items);
								}
								checkCudaErrors(cudaStreamSynchronize(iterator.ws.stream[t]));
#pragma omp barrier
								if (t == 0)
								{
#ifdef LAP_ROWS_SCANNED
									{
										int i;
										int eop = endofpath_local;
										do
										{
											i = pred[eop];
											eop = rowsol[i];
											if (i != f) pathlength[f]++;
										} while (i != f);
									}
#endif
									resetRowColumnAssignment(endofpath_local, f, pred, rowsol, colsol);
								}
								require_colsol_copy_local = true;
							}
#ifndef LAP_QUIET
							if (t == 0)
							{
								int level;
								if ((level = displayProgress(start_time, elapsed, fc + 1, dim_limit, " rows")) != 0)
								{
									long long hit, miss;
									iterator.getHitMiss(hit, miss);
									total_hit += hit;
									total_miss += miss;
									if ((hit != 0) || (miss != 0))
									{
										if (level == 1) lapInfo << "  hit: " << hit << " miss: " << miss << " (" << miss - (fc + 1 - old_complete) << " + " << fc + 1 - old_complete << ")" << std::endl;
										else lapDebug << "  hit: " << hit << " miss: " << miss << " (" << miss - (fc + 1 - old_complete) << " + " << fc + 1 - old_complete << ")" << std::endl;
									}
									old_complete = fc + 1;
								}
							}
#endif
						}

#ifdef LAP_MINIMIZE_V
						if (epsilon > TC(0))
						{
							if (dim_limit < dim2)
							{
								if (require_colsol_copy_local)
								{
									checkCudaErrors(cudaMemcpyAsync(colsol_private[t], &(colsol[start]), num_items * sizeof(int), cudaMemcpyHostToDevice, stream));
								}
								findMaximum(v_private[t], colsol_private[t], &(host_min_private[t]), stream, num_items);
							}
							else
							{
								findMaximum(v_private[t], &(host_min_private[t]), stream, num_items);
							}
							checkCudaErrors(cudaStreamSynchronize(stream));
#pragma omp barrier
							SC max_v = mergeMaximum<SC>(host_min_private, devices);

							if (dim_limit < dim2)
							{
								subtractMaximumLimited_kernel<<<gs, bs, 0, stream>>>(v_private[t], max_v, num_items);
							}
							else
							{
								subtractMaximum_kernel<<<gs, bs, 0, stream>>>(v_private[t], max_v, num_items);
							}
						}
#endif

						// download updated v
						checkCudaErrors(cudaMemcpyAsync(&(h_total_d[start]), total_d_private[t], sizeof(SC) * num_items, cudaMemcpyDeviceToHost, stream));
						checkCudaErrors(cudaMemcpyAsync(&(h_total_eps[start]), total_eps_private[t], sizeof(SC) * num_items, cudaMemcpyDeviceToHost, stream));
#ifdef LAP_DEBUG
						checkCudaErrors(cudaMemcpyAsync(&(v[start]), v_private[t], sizeof(SC) * num_items, cudaMemcpyDeviceToHost, stream));
#endif
						checkCudaErrors(cudaStreamSynchronize(iterator.ws.stream[t]));
					}
				}
#endif
				// get total_d and total_eps
				for (int i = 0; i < dim2; i++)
				{
					total_d += h_total_d[i];
					total_eps += h_total_eps[i];
				}

#ifdef LAP_DEBUG
				if (epsilon > TC(0))
				{
					SC *vv;
					lapAlloc(vv, dim2, __FILE__, __LINE__);
					v_list.push_back(vv);
					eps_list.push_back(epsilon);
					memcpy(v_list.back(), v, sizeof(SC) * dim2);
				}
				else
				{
					int count = (int)v_list.size();
					if (count > 0)
					{
						for (int l = 0; l < count; l++)
						{
							SC dlt(0), dlt2(0);
							for (int i = 0; i < dim2; i++)
							{
								SC diff = v_list[l][i] - v[i];
								dlt += diff;
								dlt2 += diff * diff;
							}
							dlt /= SC(dim2);
							dlt2 /= SC(dim2);
							lapDebug << "iteration = " << l << " eps/mse = " << eps_list[l] << " " << dlt2 - dlt * dlt << " eps/rmse = " << eps_list[l] << " " << sqrt(dlt2 - dlt * dlt) << std::endl;
							lapFree(v_list[l]);
						}
					}
				}
#endif
				second = first;
				first = false;
				reverse = !reverse;
#ifndef LAP_QUIET
				lapInfo << "  rows evaluated: " << total_rows;
				if (total_virtual > 0) lapInfo << " virtual rows evaluated: " << total_virtual;
				lapInfo << std::endl;
				if ((total_hit != 0) || (total_miss != 0)) lapInfo << "  hit: " << total_hit << " miss: " << total_miss << std::endl;
#endif
			}

#ifdef LAP_QUIET
#ifdef LAP_DISPLAY_EVALUATED
			iterator.getHitMiss(total_hit, total_miss);
			lapInfo << "  rows evaluated: " << total_rows;
			if (total_virtual > 0) lapInfo << " virtual rows evaluated: " << total_virtual;
			lapInfo << std::endl;
			if ((total_hit != 0) || (total_miss != 0)) lapInfo << "  hit: " << total_hit << " miss: " << total_miss << std::endl;
#endif
#endif

#ifdef LAP_ROWS_SCANNED
			lapInfo << "row\tscanned\tlength" << std::endl;
			for (int f = 0; f < dim2; f++)
			{
				lapInfo << f << "\t" << scancount[f] << "\t" << pathlength[f] << std::endl;
			}

			lapFree(scancount);
#endif

			// free CUDA memory
			for (int t = 0; t < devices; t++)
			{
				checkCudaErrors(cudaSetDevice(iterator.ws.device[t]));
				lapFreeDevice(min_private[t]);
				lapFreeDevice(jmin_private[t]);
				lapFreeDevice(csol_private[t]);
				lapFreeDevice(colactive_private[t]);
				lapFreeDevice(d_private[t]);
				lapFreeDevice(v_private[t]);
				lapFreeDevice(total_d_private[t]);
				lapFreeDevice(total_eps_private[t]);
				lapFreeDevice(pred_private[t]);
				lapFreeDevice(colsol_private[t]);
				lapFreeDevice(semaphore_private[t]);
				lapFreeDevice(start_private[t]);
				lapFreeDevice(gpu_min_private[t]);
			}

			// free reserved memory.
#ifdef LAP_DEBUG
			lapFreePinned(v);
#endif
			lapFreePinned(colsol);
			lapFreePinned(pred);
			lapFreePinned(h_total_d);
			lapFreePinned(h_total_eps);
			lapFreePinned(tt_jmin);
			lapFreePinned(v_jmin);
			lapFree(min_private);
			lapFree(jmin_private);
			lapFree(csol_private);
			lapFreePinned(host_min_private);
			lapFree(colactive_private);
			lapFree(pred_private);
			lapFree(d_private);
			lapFree(colsol_private);
			lapFree(v_private);
			lapFree(total_d_private);
			lapFree(total_eps_private);
			lapFree(semaphore_private);
			lapFree(tt);
			lapFree(perm);
			lapFree(start_private);
			lapFree(gpu_min_private);
			lapFreePinned(host_start);

			// set device back to first one
			checkCudaErrors(cudaSetDevice(iterator.ws.device[0]));

#ifdef LAP_CUDA_OPENMP
			omp_set_num_threads(old_max_threads);
#endif
		}

		template <class SC, class TC, class CF>
		SC cost(int dim, int dim2, CF &costfunc, int *rowsol, cudaStream_t stream)
		{
			SC my_cost(0);
			TC *row = new TC[dim];
			int *d_rowsol;
			TC *d_row;
			lapAllocDevice(d_rowsol, dim, __FILE__, __LINE__);
			lapAllocDevice(d_row, dim, __FILE__, __LINE__);
			checkCudaErrors(cudaMemcpyAsync(d_rowsol, rowsol, dim * sizeof(int), cudaMemcpyHostToDevice, stream));
			costfunc.getCost(d_row, stream, d_rowsol, dim);
			checkCudaErrors(cudaMemcpyAsync(row, d_row, dim * sizeof(TC), cudaMemcpyDeviceToHost, stream));
			checkCudaErrors(cudaStreamSynchronize(stream));
			lapFreeDevice(d_row);
			lapFreeDevice(d_rowsol);
			for (int i = 0; i < dim; i++) my_cost += row[i];
			delete[] row;
			return my_cost;
		}

		// shortcut for square problems
		template <class SC, class TC, class CF, class I>
		void solve(int dim, CF &costfunc, I &iterator, int *rowsol, bool use_epsilon)
		{
			lap::cuda::solve<SC, TC>(dim, dim, costfunc, iterator, rowsol, use_epsilon);
		}

		// shortcut for square problems
		template <class SC, class TC, class CF>
		SC cost(int dim, CF &costfunc, int *rowsol, cudaStream_t stream)
		{
			return lap::cuda::cost<SC, TC, CF>(dim, dim, costfunc, rowsol, stream);
		}
	}
}
