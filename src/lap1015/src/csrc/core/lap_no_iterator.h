#pragma once



 namespace lap
 {
   template <class TC, class CF>
   class NoIteratorLine
   {
   public:
     CF &costfunc;
     int i;
     NoIteratorLine(CF &costfunc, int i) : costfunc(costfunc), i(i) {}
     ~NoIteratorLine() {}

     __forceinline const TC operator[] (int j) const { return costfunc.getCost(i,j); }
   };

   template <class TC, class CF>
   class NoIterator
   {
   public:
     CF &costfunc;
   public:
     NoIterator(CF &costfunc) : costfunc(costfunc) {}
     ~NoIterator() {}

     void getHitMiss(long long &hit, long long &miss) { hit = miss = 0; }

     __forceinline auto getRow(int i) { return NoIteratorLine<TC, CF>(costfunc, i); }
   };
 }