#ifndef _COMBINATION_UTIL_H_
#define _COMBINATION_UTIL_H_

#include <stdint.h>

uint64_t calcNumComb(int n, int c);
uint64_t calcCombNum(int num_cols, int start, int end);


//Non recursive template function
template <class BidIt>
inline bool next_combination(BidIt n_begin, BidIt n_end,
   BidIt r_begin, BidIt r_end)
{
     bool boolmarked=false;
     BidIt r_marked;

     BidIt n_it1=n_end;
     --n_it1;

     BidIt tmp_r_end=r_end;
     --tmp_r_end;

     for(BidIt r_it1=tmp_r_end; r_it1!=r_begin || r_it1==r_begin; --r_it1,--n_it1)
     {
       if(*r_it1==*n_it1 )
       {
         if(r_it1!=r_begin) //to ensure not at the start of r sequence
         {
           boolmarked=true;
           r_marked=(--r_it1);
           ++r_it1;//add it back again
          continue;
         }
         else // it means it is at the start the sequence, so return false
           return false;
       }
       else //if(*r_it1!=*n_it1 )
       {
         //marked code
         if(boolmarked==true)
         {
           //for loop to find which marked is in the first sequence
           BidIt n_marked;//mark in first sequence
           for (BidIt n_it2=n_begin;n_it2!=n_end;++n_it2)
             if(*r_marked==*n_it2) {n_marked=n_it2;break;}
           BidIt n_it3=++n_marked;
           for  (BidIt r_it2=r_marked;r_it2!=r_end;++r_it2,++n_it3)
           {
             *r_it2=*n_it3;
           }
           return true;
         }
         for(BidIt n_it4=n_begin; n_it4!=n_end; ++n_it4)
           if(*r_it1==*n_it4)
           {
             *r_it1=*(++n_it4);
             return true;
           }
       }
     }

	return true;//will never reach here
 }

#endif // _COMBINATION_UTIL_H_