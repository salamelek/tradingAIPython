/******************************************************************************/
/*                                                                            */
/*  SUP3 - Demonstrate superiority algorithm for optimal lookahead in         */
/*         PRICE INTENSITY                                                    */
/*                                                                            */
/******************************************************************************/

// Note for Software Verify under MS VCPP2019:  Settings Link/Debug/FULL

#include <stdio.h>
#include <malloc.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <conio.h>
#include <assert.h>
#include <conio.h>
#include "funcdefs.h"

#define MKTBUF 2048   /* Alloc for market info in chunks of this many records */


/*
--------------------------------------------------------------------------------

   Main routine

--------------------------------------------------------------------------------
*/

int main (
   int argc ,    // Number of command line arguments (includes prog name)
   char *argv[]  // Arguments (prog name is argv[0])
   )
{
   int i, j, k, prior_k, icomp, irep, nreps, n_bars, bufcnt, n_smooth ;
   int ibar, look_low, look_high, n_looks, *index, *sort_key, trend_elim ;
   double *open, *high, *low, *close, *returns, *indicators, *mean_rets ;
   double xmean, xmin, xmax, *work1, *work2, *high_thresh, *low_thresh ;
   double *rho, pf_all, pf_low, pf_high, *criters, *counts;
   double long_win, long_lose, short_win, short_lose, median, factor ;
   double *xwork, *ret_work, *ret_ptr, ret_val ;
   char line[256], filename[4096], *cptr ;
   FILE *fp ;

/*
   Process command line parameters
*/

#if 1
   if (argc != 7) {
      printf ( "\nUsage: SUP3  n_smooth  look_low  look_high  trend_elim  nreps  filename" ) ;
      printf ( "\n  n_smooth - Smoothing lookback for indicator" ) ;
      printf ( "\n  look_low - Lowest lookahead" ) ;
      printf ( "\n  look_high - Highest lookahead" ) ;
      printf ( "\n  trend_elim - Eliminate trend effect" ) ;
      printf ( "\n  nreps - Number of MCPT replications (hundreds or thousands)" ) ;
      printf ( "\n  filename - name of market file (YYYYMMDD Open High Low Close)" ) ;
      exit ( 1 ) ;
      }

   n_smooth = atoi ( argv[1] ) ;
   look_low = atoi ( argv[2] ) ;
   look_high = atoi ( argv[3] ) ;
   trend_elim = atoi ( argv[4] ) ;
   nreps = atoi ( argv[5] ) ;
   strcpy_s ( filename , argv[6] ) ;
#else
   n_smooth = 3 ;
   look_low = 1 ;
   look_high = 20 ;
   trend_elim = 1 ;
   nreps = 500 ;
   strcpy_s ( filename , "E:\\INDEXES\\OEX.TXT" ) ;
#endif


/*
   Prevent careless user errors
*/

   if (look_low < 1) {
      look_low = 1 ;
      printf ( "\nWARNING... look_low cannot be less than 1.  Changed to 1." ) ;
      }

   if (look_high < look_low+1) {
      look_high = look_low + 1 ;
      printf ( "\nWARNING... look_high cannot be less than look_low+1.  Reset to that minimum." ) ;
      }

   if (nreps < 1) {
      nreps = 1 ;
      printf ( "\nWARNING... nreps cannot be less than 1.  Changed to 1." ) ;
      }

   printf ( "SUP3  n_smooth=%d  look_low=%d  look_high=%d  trend_elim=%d  nreps=%d",
            n_smooth, look_low, look_high, trend_elim, nreps ) ;

   n_looks = look_high - look_low + 1 ;


/*
   Allocate memory that we can now
*/

   indicators = NULL ;     // We'll allocate this when we know how many bars
   work1 = work2 = NULL ;  // Ditto
   returns = NULL ;        // Ditto
   index = NULL ;          // Ditto
   xwork = NULL ;          // Ditto
   ret_work = NULL ;       // Ditto

   open = (double *) malloc ( MKTBUF * sizeof(double) ) ;
   high = (double *) malloc ( MKTBUF * sizeof(double) ) ;
   low = (double *) malloc ( MKTBUF * sizeof(double) ) ;
   close = (double *) malloc ( MKTBUF * sizeof(double) ) ;
   mean_rets = (double *) malloc ( n_looks * sizeof(double) ) ;
   low_thresh = (double *) malloc ( n_looks * sizeof(double) ) ;
   high_thresh = (double *) malloc ( n_looks * sizeof(double) ) ;
   counts = (double *) malloc ( n_looks * sizeof(double) ) ;
   criters = (double *) malloc ( n_looks * sizeof(double) ) ;
   sort_key = (int *) malloc ( n_looks * sizeof(int) ) ;
   rho = (double *) malloc ( n_looks * sizeof(double) ) ;
   if (open == NULL  ||  high == NULL  ||  low == NULL  ||  close == NULL
    || mean_rets == NULL  || low_thresh == NULL  ||  high_thresh == NULL
    || counts == NULL  ||  criters == NULL  ||  sort_key == NULL  ||  rho == NULL) {
      printf ( "\n\nInsufficient memory reading market history file %s", filename ) ;
      goto FINISH ;
      }

/*
--------------------------------------------------------------------------------

   Read market prices

--------------------------------------------------------------------------------
*/

   if (fopen_s ( &fp, filename , "rt" )) {
      printf ( "\n\nCannot open market history file %s", filename ) ;
      goto FINISH ;
      }

   bufcnt = MKTBUF ;  // There are this many record slots available now

   printf ( "\nReading market file..." ) ;

   n_bars = 0 ;    // Counts lines (prices) read

   for (;;) {

      if (feof ( fp )                          // If end of file
       || (fgets ( line , 256 , fp ) == NULL)  // Or unable to read line
       || (strlen ( line ) < 2))               // Or empty line
         break ;                               // We are done reading price history

      if (ferror ( fp )) {                     // If an error reading file
         fclose ( fp ) ;                       // Quit immediately
         printf ( "\nError reading line %d of file %s", n_bars+1, filename ) ;
         goto FINISH ;
         }

      if (! bufcnt) {  // Allocate a new memory block if needed
         open = (double *) realloc ( open , (n_bars+MKTBUF) * sizeof(double) ) ;
         high = (double *) realloc ( high , (n_bars+MKTBUF) * sizeof(double) ) ;
         low = (double *) realloc ( low , (n_bars+MKTBUF) * sizeof(double) ) ;
         close = (double *) realloc ( close , (n_bars+MKTBUF) * sizeof(double) ) ;
         if (open == NULL  ||  high == NULL  ||  low == NULL  ||  close == NULL) {
            fclose ( fp ) ;
            printf ( "\n\nInsufficient memory reading market history file %s", filename ) ;
            goto FINISH ;
            } // If insufficient memory
         bufcnt = MKTBUF ;  // There are this many new record slots available now
         } // If allocating new block

      // Parse the date and do a crude sanity check

      for (i=0 ; i<8 ; i++) {
         if ((line[i] < '0')  ||  (line[i] > '9')) {
            fclose ( fp ) ;
            printf ( "\nInvalid date reading line %d of file %s", n_bars+1, filename ) ;
            goto FINISH ;
            }
         }

      // Parse the open

      cptr = line + 9 ;  // Price is in this column or beyond
                         // (Next loop allows price to start past this)

      while (*cptr == ' '  ||  *cptr == '\t'  ||  *cptr == ',')  // Delimiters
         ++cptr ;  // Move up to the price

      open[n_bars] = atof ( cptr ) ;

      while (*cptr  &&  *cptr != ' '  &&  *cptr != ','  &&  *cptr != '\t')
         ++cptr ;  // Pass the price; stop at delimiter

      // Parse the high

      while (*cptr == ' '  ||  *cptr == '\t'  ||  *cptr == ',')  // Delimiters
         ++cptr ;  // Move up to the price

      high[n_bars] = atof ( cptr ) ;

      while (*cptr  &&  *cptr != ' '  &&  *cptr != ','  &&  *cptr != '\t')
         ++cptr ;  // Pass the price; stop at delimiter

      // Parse the low

      while (*cptr == ' '  ||  *cptr == '\t'  ||  *cptr == ',')  // Delimiters
         ++cptr ;  // Move up to the price

      low[n_bars] = atof ( cptr ) ;

      while (*cptr  &&  *cptr != ' '  &&  *cptr != ','  &&  *cptr != '\t')
         ++cptr ;  // Pass the price; stop at delimiter

      // Parse the close

      while (*cptr == ' '  ||  *cptr == '\t'  ||  *cptr == ',')  // Delimiters
         ++cptr ;  // Move up to the price

      close[n_bars] = atof ( cptr ) ;

      if (low[n_bars] > open[n_bars]  ||  low[n_bars] > close[n_bars]  ||
          high[n_bars] < open[n_bars]  ||  high[n_bars] < close[n_bars]) {
         fclose ( fp ) ;
         printf ( "\nInvalid open/high/low/close reading line %d of file %s", n_bars+1, filename ) ;
         goto FINISH ;
         }

      ++n_bars  ;
      --bufcnt ;           // One less slot remains

      } // For all lines

   fclose ( fp ) ;

   printf ( "\n\n%d bars were read, %d replications beginning now", n_bars, nreps ) ;


/*
   The market data is read.
   Allocate memory that we can now that we know n_bars.
*/

   if (n_bars-look_high < 20) {  // Really too small, but do something!
      printf ( "\nERROR... Number of prices must be at least look_high+20" ) ;
      goto FINISH ;
      }

   indicators = (double *) malloc ( n_bars * sizeof(double) ) ;
   returns = (double *) malloc ( n_looks * n_bars * sizeof(double) ) ;
   work1 = (double *) malloc ( n_bars * sizeof(double) ) ;
   work2 = (double *) malloc ( n_bars * sizeof(double) ) ;
   xwork = (double *) malloc ( n_bars * sizeof(double) ) ;
   ret_work = (double *) malloc ( n_bars * sizeof(double) ) ;
   index = (int *) malloc ( n_bars * sizeof(index) ) ;
   if (indicators == NULL  ||  work1 == NULL  ||  work2 == NULL
    || xwork == NULL  ||  ret_work == NULL  ||  index == NULL) {
      printf ( "\n\nInsufficient memory processing market history file %s", filename ) ;
      goto FINISH ;
      }


/*
   Compute the returns as log close-to-close.
   We compute a different return series for each holding period (lookahead)
   Also compute the mean of each return series so we can remove trend when
   calculating performance.
*/

   for (icomp=look_low ; icomp<=look_high ; icomp++) {
      k = icomp - look_low ;
      ret_ptr = returns + k * n_bars ;
      mean_rets[k] = 0.0 ;
      for (i=0 ; i<n_bars-look_high ; i++) {
         ret_ptr[i] = log ( close[i+icomp] / close[i] ) ;
         mean_rets[k] += ret_ptr[i] ;
         }
      mean_rets[k] /= n_bars-look_high ;
      }


/*
   Fill in the indicator vector, which is the same for all competitors.
   For a sanity check, I also compute the mean, minimum, and maximum.
   We have the same indicator for all competitors, PRICE INTENSITY with lookback=2.
*/

   comp_ind ( n_bars , open , high , low , close , n_smooth , indicators ) ;

   xmean = xmin = xmax = indicators[0] ;
   for (i=1 ; i<n_bars ; i++) {
      xmean += indicators[i] ;
      if (indicators[i] < xmin)
         xmin = indicators[i] ;
      if (indicators[i] > xmax)
         xmax = indicators[i] ;
      }
   xmean /= n_bars ;

   printf ( "\nIndicator Mean=%12.5lf   Min=%12.5lf   Max=%12.5lf", xmean, xmin, xmax ) ;
               

/*
--------------------------------------------------------------------------------

   Do the replications

--------------------------------------------------------------------------------
*/

   for (icomp=0 ; icomp<n_looks ; icomp++)
      counts[icomp] = 0.0 ;  // Will count failures to exceed median for each ranked competitor

   for (ibar=0 ; ibar<n_bars-look_high ; ibar++)
      index[ibar] = ibar ;   // Will index bars

   for (irep=0 ; irep<nreps ; irep++) {

      // Shuffle the bar indices.  First half will be training, second half test.

      i = n_bars-look_high ;     // Number remaining to be shuffled; we lose last prices due to lookahead
      while (i > 1) {  // While at least 2 left to shuffle
         j = (int) (unifrand() * i) ;
         if (j >= i)  // Cheap insurance against disaster if unifrand() returns 1.0
            j = i - 1 ;
         k = index[--i] ;
         index[i] = index[j] ;
         index[j] = k ;
         }

/*
   Find the training-set criterion for each competitor
   Then sort them ascending, simultaneously moving a sort key.
   The first step processing the training set is collecting the training set data.
   The first step for each competitor is to find indicator's Spearman correlation
   with the returns.  If it is negatively correlated flip its sign.
   I arbitrarily demand that at least 10 percent (0.1) of bars have a trade.
   Feel free to change this or make it a user parameter, but 0.1 generally works well.
*/

      for (icomp=0 ; icomp<n_looks ; icomp++) {     // Process each competitor (look-ahead)
         sort_key[icomp] = icomp ;

         for (i=0 ; i<(n_bars-look_high)/2 ; i++) { // Training set
            k = index[i] ;
            xwork[i] = indicators[k] ;              // Collect indicator for training set
            ret_work[i] = returns[icomp*n_bars+k] ; // Collect returns for training set
            if (trend_elim)
               ret_work[i] -= mean_rets[icomp] ;
            }

         rho[icomp] = spearman ( (n_bars-look_high)/2 , xwork , ret_work , work1 , work2 ) ;
         if (rho[icomp] < 0.0) {  // For simplicity I demand non-negative correlation
            for (i=0 ; i<(n_bars-look_high)/2 ; i++)
               xwork[i] = -xwork[i] ;
            }
         opt_thresh ( (n_bars-look_high)/2 , (int) (0.1 * (n_bars-look_high)/2) , 0 , xwork , ret_work ,
                      &pf_all , high_thresh+icomp , &pf_high , low_thresh+icomp , &pf_low ,
                      work1 , work2 ) ;
         long_win = long_lose = short_win = short_lose = 1.e-60 ;
         for (i=0 ; i<(n_bars-look_high)/2 ; i++) {      // Training set
            ret_val = ret_work[i] ;
            if (xwork[i] >= high_thresh[icomp]) { // Take a long position?
               if (ret_val > 0.0)
                  long_win += ret_val ;
               else
                  long_lose -= ret_val ;
               }
            if (xwork[i] < low_thresh[icomp]) { // Take a short position?
               if (ret_val < 0.0)
                  short_win -= ret_val ;
               else
                  short_lose += ret_val ;
               }
            } // For training set
         criters[icomp] = (long_win + short_win) / (long_lose + short_lose) ;
         } // For all competitors

      qsortdsi ( 0 , n_looks-1 , criters , sort_key ) ;


/*
   Find the test-set criterion for each competitor
   The first step processing the test set is collecting the test set data.
   Then find the median test-set return.
*/

      for (icomp=0 ; icomp<n_looks ; icomp++) {

         for (i=(n_bars-look_high)/2 ; i<n_bars-look_high ; i++) { // Test set
            k = index[i] ;
            xwork[i] = indicators[k] ;              // Collect indicator for test set
            ret_work[i] = returns[icomp*n_bars+k] ; // Collect returns for test set
            if (trend_elim)
               ret_work[i] -= mean_rets[icomp] ;
            }

         if (rho[icomp] < 0.0) {  // If we flipped sign in training, must also in test
            for (i=(n_bars-look_high)/2 ; i<n_bars-look_high ; i++) // Test set
               xwork[i] = -xwork[i] ;
            }
         long_win = long_lose = short_win = short_lose = 1.e-60 ;
         for (i=(n_bars-look_high)/2 ; i<n_bars-look_high ; i++) { // Test set
            ret_val = ret_work[i] ;
            if (xwork[i] >= high_thresh[icomp]) { // Take a long position?
               if (ret_val > 0.0)
                  long_win += ret_val ;
               else
                  long_lose -= ret_val ;
               }
            if (xwork[i] < low_thresh[icomp]) { // Take a short position?
               if (ret_val < 0.0)
                  short_win -= ret_val ;
               else
                  short_lose += ret_val ;
               }
            } // For test set
         criters[icomp] = (long_win + short_win) / (long_lose + short_lose) ;
         work1[icomp] = criters[icomp] ;    // We cannot disturb test-set crits, so sort this for median
         } // For all competitors

      qsortd ( 0 , n_looks-1 , work1 ) ;
      if (n_looks % 2)
         median = work1[n_looks/2] ;
      else
         median = 0.5 * (work1[n_looks/2-1] + work1[n_looks/2]) ;

      // For each training-set rank, see if its test-set criter fails to exceed the median
      for (icomp=0 ; icomp<n_looks ; icomp++) {
         if (criters[sort_key[icomp]] <= median)
            ++counts[icomp] ;
         }
      } // For irep

// Divide counts by number of reps to make them probabilities

   for (icomp=0 ; icomp<n_looks ; icomp++)
      counts[icomp] /= nreps ;

/*
--------------------------------------------------------------------------------

   The essential work is complete.
   At this time, counts[0] is the probability that the worst training-set
   competitor will have a test-set performance that fails to exceed the median.
   Similarly,  counts[n_looks-1] is the probability that the best training-set
   competitor will have a test-set performance that fails to exceed the median.
   The interior ranks are similarly ordered.

   Thus, we can consider these quantities to reveal the likelihood that each
   ranked in-sample performer could have done as well as it did by pure luck.

   This is especially useful information for the high-ranked competitors,
   those that did well in the training set.  If we find that this probability
   is nicely small (0.05 or even 0.01 are common thresholds) then we can
   conclude that this high-ranked competitor is truly superior.

   As a useful final step we find the performance of each competitor using the
   entire dataset.  We print these, sorted from best to worst, and also print
   the associated probabilities.

--------------------------------------------------------------------------------
*/

   printf ( "\n\nThe following analyses are for the entire dataset..." ) ;
   printf ( "\n\nParameter   Correl   Long thresh   Long pf   Short thresh   Short pf" ) ;

   for (icomp=0 ; icomp<n_looks ; icomp++) {
      for (i=0 ; i<n_bars-look_high ; i++) {
         xwork[i] = indicators[i] ;
         ret_work[i] = returns[icomp*n_bars+i] ;
         if (trend_elim)
            ret_work[i] -= mean_rets[icomp] ;
         }
      rho[icomp] = spearman ( n_bars-look_high , indicators , ret_work , work1 , work2 ) ;
      if (rho[icomp] < 0.0) {  // For simplicity I demand non-negative correlation
         for (i=0 ; i<n_bars-look_high ; i++)
            xwork[i] = -xwork[i] ;
         }
      opt_thresh ( n_bars-look_high , (int) (0.1 * (n_bars-look_high)) , 0 , xwork , ret_work ,
                   &pf_all , high_thresh+icomp , &pf_high , low_thresh+icomp , &pf_low ,
                   work1 , work2 ) ;
      printf ( "\n%6d %11.4lf %12.5lf %9.3lf %14.5lf %10.3lf",
               look_low+icomp, rho[icomp], high_thresh[icomp], pf_high, low_thresh[icomp], pf_low ) ;
      }

   for (icomp=0 ; icomp<n_looks ; icomp++) {
      sort_key[icomp] = icomp ;
      for (i=0 ; i<n_bars-look_high ; i++) {
         xwork[i] = indicators[i] ;
         ret_work[i] = returns[icomp*n_bars+i] ;
         if (trend_elim)
            ret_work[i] -= mean_rets[icomp] ;
         }
      if (rho[icomp] < 0.0) {
         for (i=0 ; i<n_bars-look_high ; i++)
            xwork[i] = -xwork[i] ;
         }
      ret_ptr = returns + icomp * n_bars ;
      long_win = long_lose = short_win = short_lose = 1.e-60 ;
      for (i=0 ; i<n_bars-look_high ; i++) {
         ret_val = ret_ptr[i] ;
         if (xwork[i] >= high_thresh[icomp]) { // Take a long position?
            if (ret_val > 0.0)
               long_win += ret_val ;
            else
               long_lose -= ret_val ;
            }
         if (xwork[i] < low_thresh[icomp]) { // Take a short position?
            if (ret_val < 0.0)
               short_win -= ret_val ;
            else
               short_lose += ret_val ;
            }
         } // For training set
      criters[icomp] = (long_win + short_win) / (long_lose + short_lose) ;
      } // For all competitors

/*
   Print a crude graph of results before sorting
*/

   xmin = xmax = criters[0] ;
   for (icomp=1 ; icomp<n_looks ; icomp++) {
      if (criters[icomp] < xmin)
         xmin = criters[icomp] ;
      if (criters[icomp] > xmax)
         xmax = criters[icomp] ;
      }

   factor = 75.0 / (xmax - xmin) ;

   printf ( "\n\n" ) ;
   for (icomp=0 ; icomp<n_looks ; icomp++) {
      k = (int) (factor * (criters[icomp] - xmin) + 0.5) ;
      if (icomp) {
         j = (int) (0.666667 * prior_k + 0.333333 * k + 0.5) ;
         printf ( "\n     |" ) ;
         for (i=0 ; i<j ; i++)
            printf ( " " ) ;
         printf ( "*" ) ;
         j = (int) (0.333333 * prior_k + 0.666667 * k + 0.5) ;
         printf ( "\n     |" ) ;
         for (i=0 ; i<j ; i++)
            printf ( " " ) ;
         printf ( "*" ) ;
         }
      printf ( "\n%4d |", icomp+look_low ) ;
      for (i=0 ; i<k ; i++)
         printf ( " " ) ;
      printf ( "*" ) ;
      prior_k = k ;
      }


/*
   Print the final results with failure probabilities
*/

   qsortdsi ( 0 , n_looks-1 , criters , sort_key ) ;

   printf ( "\n\nCompetitor performance ranked best to worst, with superiority 'p-values' (failure rates)" ) ;
   printf ( "\nProfit factors are optimistically biased because the threshold was optimized" ) ;
   printf ( "\nParameter  Profit factor   p-value" ) ;

   for (icomp=n_looks-1 ; icomp>=0 ; icomp--) {  // Best to worst
      k = sort_key[icomp] ;
      printf ( "\n%7d %12.3lf %12.3lf", look_low + k, criters[icomp], counts[icomp] ) ;
      }

FINISH:
   printf ( "\n\nPress any key..." ) ;
   _getch () ;  // Wait for user to press a key

   if (open != NULL)
      free ( open ) ;
   if (high != NULL)
      free ( high ) ;
   if (low != NULL)
      free ( low ) ;
   if (close != NULL)
      free ( close ) ;
   if (mean_rets != NULL)
      free ( mean_rets ) ;
   if (indicators != NULL)
      free ( indicators ) ;
   if (returns != NULL)
      free ( returns ) ;
   if (high_thresh != NULL)
      free ( high_thresh ) ;
   if (low_thresh != NULL)
      free ( low_thresh ) ;
   if (work1 != NULL)
      free ( work1 ) ;
   if (work2 != NULL)
      free ( work2 ) ;
   if (counts != NULL)
      free ( counts ) ;
   if (criters != NULL)
      free ( criters ) ;
   if (index != NULL)
      free ( index ) ;
   if (xwork != NULL)
      free ( xwork ) ;
   if (ret_work != NULL)
      free ( ret_work ) ;
   if (rho != NULL)
      free ( rho ) ;

   exit ( 0 ) ;
}
