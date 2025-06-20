#pragma once



#include "lap_worksharing.h"

namespace lap
{
namespace omp
{
    template <class TC, class CF>
    class NoIterator
    {
    public:
    CF &costfunc;
    Worksharing &ws;
    public:
    NoIterator(CF &costfunc, Worksharing &ws) : costfunc(costfunc), ws(ws) {}
    ~NoIterator() {}

    void getHitMiss(long long &hit, long long &miss) { hit = miss = 0; }

    __forceinline auto getRow(int t, int i) { return lap::NoIteratorLine<TC, CF>(costfunc, i); }
    };
}
}