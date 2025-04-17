/******************************************************************************/
/*                                                                            */
/*  COMP_IND - Compute the indicator being tested                             */
/*                                                                            */
/*  This example uses the PRICE INTENSITY indicator described in my book      */
/*  "Statistically Sound Indicators for Financial Market Prediction"          */
/*                                                                            */
/*  Feel free to substitute the indicator of your choice.                     */
/*                                                                            */
/******************************************************************************/

#include "funcdefs.h"


void comp_ind (
   int n ,         // Number of market price bars
   double *open ,  // They are here
   double *high ,
   double *low ,
   double *close ,
   int iparam ,    // Competition parameter, n to smooth here
   double *ind     // The indicator values are output here
   )
{
   int ibar ;
   double denom, alpha, rawval, smoothed ;

   if (iparam < 1)  // This is just insurance against a careless caller
      iparam = 1 ;

   alpha = 2.0 / (iparam + 1.0) ;   // Exponential smoothing constant

   // Initialize the first indicator value

   denom = high[0] - low[0] ;
   if (denom < 1.e-60)
      denom = 1.e-60 ;
   ind[0] = (close[0] - open[0]) / denom ;
   smoothed = ind[0] ;   // Initialize for smoothing

   // Here we go

   for (ibar=1 ; ibar<n ; ibar++) {

      // Find the max range, the max of these three differences
      denom = high[ibar] - low[ibar] ;
      if (high[ibar] - close[ibar-1] > denom)
         denom = high[ibar] - close[ibar-1] ;
      if (close[ibar-1] - low[ibar] > denom)
         denom = close[ibar-1] - low[ibar] ;
      if (denom < 1.e-60)
         denom = 1.e-60 ;

      rawval = (close[ibar] - open[ibar]) / denom ;

      // Smooth it
      smoothed = alpha * rawval + (1.0 - alpha) * smoothed ;
      ind[ibar] = 100.0 * smoothed ;
      }
}