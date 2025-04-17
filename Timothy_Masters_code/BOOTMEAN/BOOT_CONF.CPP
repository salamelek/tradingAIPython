/******************************************************************************/
/*                                                                            */
/*  BOOT_CONF - Bootstrap confidence interval with BCa method                 */
/*                                                                            */
/******************************************************************************/

#include <math.h>
#include <stdlib.h>
#include "funcdefs.h"


/*
--------------------------------------------------------------------------------

   boot_conf_BCa - Compute confidence intervals using improved quantile method

--------------------------------------------------------------------------------
*/

void boot_conf_BCa (
   int n ,              // Number of cases in sample
   double *x ,          // Variable in sample
   double (*user_t) (int , double * ) , // Compute parameter
   int nboot ,          // Number of bootstrap replications
   double low_conf ,    // Probability (<=0.5) of being below
   double high_conf ,   // Probability (>=0.5) of being below
   double *low_bound ,  // Output of lower 5% bound
   double *high_bound , // Output of upper 5% bound
   double *xwork ,      // Work area n long
   double *work2        // Work area nboot long
   )
{
   int i, rep, k, z0_count ;
   double param, theta_hat, theta_dot, z0, zlo, zhi, alo, ahi ;
   double xtemp, xlast, diff, numer, denom, accel ;

   theta_hat = user_t ( n , x ) ;       // Parameter for full set

   z0_count = 0 ;                       // Will count for computing z0 later

   for (rep=0 ; rep<nboot ; rep++) {    // Do all bootstrap reps (b from 1 to B)

      for (i=0 ; i<n ; i++) {           // Generate the bootstrap sample
         k = (int) (unifrand() * n) ;   // Select a case from the sample
         if (k >= n)                    // Should never happen, but be prepared
            k = n - 1 ;
         xwork[i] = x[k] ;              // Put bootstrap sample in xwork
         }

      param = user_t ( n , xwork ) ;    // Param for this bootstrap rep
      work2[rep] = param ;              // Save it for CDF later
      if (param < theta_hat)            // Count how many < full set param
         ++z0_count ;                   // For computing z0 later
      }

   if (z0_count >= nboot)               // Prevent nastiness
      z0_count = nboot - 1 ;
   if (z0_count <= 0)
      z0_count = 1 ;

   z0 = inverse_normal_cdf ( (double) z0_count / (double) nboot ) ;

/*
   Do the jackknife for computing accel.
   Borrow xwork for storing jackknifed parameter values.
*/

   xlast = x[n-1] ;
   theta_dot = 0.0 ;
   for (i=0 ; i<n ; i++) {          // Jackknife
      xtemp = x[i] ;                // Preserve case being temporarily removed
      x[i] = xlast ;                // Swap in last case
      param = user_t ( n-1 , x ) ;  // Param for this jackknife
      theta_dot += param ;          // Cumulate mean across jackknife
      xwork[i] = param ;            // Save for computing accel later
      x[i] = xtemp ;                // Restore original case
      }

/*
   Compute accel
*/

   theta_dot /= n ;
   numer = denom = 0.0 ;
   for (i=0 ; i<n ; i++) {
      diff = theta_dot - xwork[i] ;
      xtemp = diff * diff ;
      denom += xtemp ;
      numer += xtemp * diff ;
      }

   denom = sqrt ( denom ) ;
   denom = denom * denom * denom ;
   accel = numer / (6.0 * denom + 1.e-60) ;

/*
   Compute the outputs
*/

   qsortd ( 0 , nboot-1 , work2 ) ;     // Sort ascending

   zlo = inverse_normal_cdf ( low_conf ) ;
   zhi = inverse_normal_cdf ( high_conf ) ;
   alo = normal_cdf ( z0 + (z0 + zlo) / (1.0 - accel * (z0 + zlo)) ) ;
   ahi = normal_cdf ( z0 + (z0 + zhi) / (1.0 - accel * (z0 + zhi)) ) ;

   k = (int) (alo * (nboot + 1)) - 1 ; // Unbiased fractile estimator
   if (k < 0)
      k = 0 ;
   *low_bound = work2[k] ;

   k = (int) ((1.0-ahi) * (nboot + 1)) - 1 ;
   if (k < 0)
      k = 0 ;
   *high_bound = work2[nboot-1-k] ;
}