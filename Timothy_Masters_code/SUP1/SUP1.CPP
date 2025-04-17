/******************************************************************************/
/*                                                                            */
/*  SUP1 - Demonstrate superiority algorithm for smoothing constant in        */
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
   int i, j, k, icomp, irep, nreps, nprices, bufcnt ;
   int ibar, s_low, s_high, n_smooth, *index, *sort_key ;
   double *open, *high, *low, *close, *returns, *indicators, param ;
   double *xptr, xmean, xmin, xmax, *work1, *work2, *high_thresh, *low_thresh ;
   double *rho, pf_all, pf_low, pf_high, *criters, *counts;
   double long_win, long_lose, short_win, short_lose, median ;
   double *xwork, *ret_work ;
   char line[256], filename[4096], *cptr ;
   FILE *fp ;

/*
   Process command line parameters
*/

#if 1
   if (argc != 6) {
      printf ( "\nUsage: SUP1  s_low  s_high  n_smooth  nreps  filename" ) ;
      printf ( "\n  s_low - Lowest smoothing" ) ;
      printf ( "\n  s_high - Highest smoothing" ) ;
      printf ( "\n  n_smooth - Number of smoothing lookbacks competing" ) ;
      printf ( "\n  nreps - Number of MCPT replications (hundreds or thousands)" ) ;
      printf ( "\n  filename - name of market file (YYYYMMDD Open High Low Close)" ) ;
      exit ( 1 ) ;
      }

   s_low = atoi ( argv[1] ) ;
   s_high = atoi ( argv[2] ) ;
   n_smooth = atoi ( argv[3] ) ;
   nreps = atoi ( argv[4] ) ;
   strcpy_s ( filename , argv[5] ) ;
#else
   s_low = 1 ;
   s_high = 500 ;
   n_smooth = 14 ;
   nreps = 100 ;
   strcpy_s ( filename , "E:\\INDEXES\\OEX.TXT" ) ;
#endif


/*
   Prevent careless user errors
*/

   if (s_low < 1) {
      s_low = 1 ;
      printf ( "\nWARNING... s_low cannot be less than 1.  Changed to 1." ) ;
      }

   if (s_high < s_low+1) {
      s_high = s_low + 1 ;
      printf ( "\nWARNING... s_high cannot be less than s_low+1.  Rest to that minimum." ) ;
      }

   if (n_smooth < 2) {
      n_smooth = 2 ;
      printf ( "\nWARNING... n_smooth cannot be less than 2.  Changed to 2." ) ;
      }

   if (nreps < 1) {
      nreps = 1 ;
      printf ( "\nWARNING... nreps cannot be less than 1.  Changed to 1." ) ;
      }

   printf ( "\nSUP1  s_low=%d  s_high=%d  n_smooth=%d  nreps=%d",
            s_low, s_high, n_smooth, nreps ) ;


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
   low_thresh = (double *) malloc ( n_smooth * sizeof(double) ) ;
   high_thresh = (double *) malloc ( n_smooth * sizeof(double) ) ;
   counts = (double *) malloc ( n_smooth * sizeof(double) ) ;
   criters = (double *) malloc ( n_smooth * sizeof(double) ) ;
   sort_key = (int *) malloc ( n_smooth * sizeof(int) ) ;
   rho = (double *) malloc ( n_smooth * sizeof(double) ) ;
   if (open == NULL  ||  high == NULL  ||  low == NULL  ||  close == NULL
    || low_thresh == NULL  ||  high_thresh == NULL  ||  counts == NULL
    || criters == NULL  ||  sort_key == NULL  ||  rho == NULL) {
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

   nprices = 0 ;    // Counts lines (prices) read

   for (;;) {

      if (feof ( fp )                          // If end of file
       || (fgets ( line , 256 , fp ) == NULL)  // Or unable to read line
       || (strlen ( line ) < 2))               // Or empty line
         break ;                               // We are done reading price history

      if (ferror ( fp )) {                     // If an error reading file
         fclose ( fp ) ;                       // Quit immediately
         printf ( "\nError reading line %d of file %s", nprices+1, filename ) ;
         goto FINISH ;
         }

      if (! bufcnt) {  // Allocate a new memory block if needed
         open = (double *) realloc ( open , (nprices+MKTBUF) * sizeof(double) ) ;
         high = (double *) realloc ( high , (nprices+MKTBUF) * sizeof(double) ) ;
         low = (double *) realloc ( low , (nprices+MKTBUF) * sizeof(double) ) ;
         close = (double *) realloc ( close , (nprices+MKTBUF) * sizeof(double) ) ;
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
            printf ( "\nInvalid date reading line %d of file %s", nprices+1, filename ) ;
            goto FINISH ;
            }
         }

      // Parse the open

      cptr = line + 9 ;  // Price is in this column or beyond
                         // (Next loop allows price to start past this)

      while (*cptr == ' '  ||  *cptr == '\t'  ||  *cptr == ',')  // Delimiters
         ++cptr ;  // Move up to the price

      open[nprices] = atof ( cptr ) ;

      while (*cptr  &&  *cptr != ' '  &&  *cptr != ','  &&  *cptr != '\t')
         ++cptr ;  // Pass the price; stop at delimiter

      // Parse the high

      while (*cptr == ' '  ||  *cptr == '\t'  ||  *cptr == ',')  // Delimiters
         ++cptr ;  // Move up to the price

      high[nprices] = atof ( cptr ) ;

      while (*cptr  &&  *cptr != ' '  &&  *cptr != ','  &&  *cptr != '\t')
         ++cptr ;  // Pass the price; stop at delimiter

      // Parse the low

      while (*cptr == ' '  ||  *cptr == '\t'  ||  *cptr == ',')  // Delimiters
         ++cptr ;  // Move up to the price

      low[nprices] = atof ( cptr ) ;

      while (*cptr  &&  *cptr != ' '  &&  *cptr != ','  &&  *cptr != '\t')
         ++cptr ;  // Pass the price; stop at delimiter

      // Parse the close

      while (*cptr == ' '  ||  *cptr == '\t'  ||  *cptr == ',')  // Delimiters
         ++cptr ;  // Move up to the price

      close[nprices] = atof ( cptr ) ;

      if (low[nprices] > open[nprices]  ||  low[nprices] > close[nprices]  ||
          high[nprices] < open[nprices]  ||  high[nprices] < close[nprices]) {
         fclose ( fp ) ;
         printf ( "\nInvalid open/high/low/close reading line %d of file %s", nprices+1, filename ) ;
         goto FINISH ;
         }

      ++nprices  ;
      --bufcnt ;           // One less slot remains

      } // For all lines

   fclose ( fp ) ;

   printf ( "\n\n%d prices were read, %d replications beginning now", nprices, nreps ) ;


/*
   The market data is read.
   Allocate memory that we can now that we know nprices.
*/

   if (nprices < 20) {  // Really too small, but do something!
      printf ( "\nERROR... Number of prices must be at least 20" ) ;
      goto FINISH ;
      }

   indicators = (double *) malloc ( n_smooth * nprices * sizeof(double) ) ;
   returns = (double *) malloc ( nprices * sizeof(double) ) ;
   work1 = (double *) malloc ( nprices * sizeof(double) ) ;
   work2 = (double *) malloc ( nprices * sizeof(double) ) ;
   xwork = (double *) malloc ( nprices * sizeof(double) ) ;
   ret_work = (double *) malloc ( nprices * sizeof(double) ) ;
   index = (int *) malloc ( nprices * sizeof(index) ) ;
   if (indicators == NULL  ||  work1 == NULL  ||  work2 == NULL
    || xwork == NULL  ||  ret_work == NULL  ||  index == NULL) {
      printf ( "\n\nInsufficient memory processing market history file %s", filename ) ;
      goto FINISH ;
      }


/*
   Compute the returns as log close-to-close
*/

   for (i=0 ; i<nprices-1 ; i++)
      returns[i] = log ( close[i+1] / close[i] ) ;


/*
   Fill in the indicator matrix.
   I store the values with the observation changing fastest, as that makes for fastest processing.
   My illustration in the book shows the observations going down, while traditional C++
   array storage has the columns changing fastest.  Thus, my actual storage here is the
   transpose of the book illustration.  However, I wanted to use the same illustration as
   was used in the Bailey et al paper in case anyone refers back to that paper.

   For a sanity check, I also compute the mean, minimum, and maximum for each competitor.
*/


   printf ( "\n\nParameter    Mean       Minimum      Maximum" ) ;

   for (icomp=0 ; icomp<n_smooth ; icomp++) {
      xptr = indicators + icomp * nprices ;    // This competitor's observations
      param = s_low * exp ( icomp / (n_smooth - 1.0) * log ( (double) s_high / (double) s_low ) ) ;
      comp_ind ( nprices , open , high , low , close , (int) (param + 0.5) , xptr ) ;

      xmean = xmin = xmax = xptr[0] ;
      for (i=1 ; i<nprices ; i++) {
         xmean += xptr[i] ;
         if (xptr[i] < xmin)
            xmin = xptr[i] ;
         if (xptr[i] > xmax)
            xmax = xptr[i] ;
         }
      xmean /= nprices ;

      printf ( "\n%6d %12.5lf %12.5lf %12.5lf", (int) (param + 0.5), xmean, xmin, xmax ) ;
      }
               

/*
--------------------------------------------------------------------------------

   Do the replications

--------------------------------------------------------------------------------
*/

   for (icomp=0 ; icomp<n_smooth ; icomp++)
      counts[icomp] = 0.0 ;  // Will count failures to exceed median for each ranked competitor

   for (ibar=0 ; ibar<nprices-1 ; ibar++)
      index[ibar] = ibar ;   // Will index bars

   for (irep=0 ; irep<nreps ; irep++) {

      // Shuffle the bar indices.  First half will be training, second half test.

      i = nprices-1 ;     // Number remaining to be shuffled; we lose last price due to lookahead
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
   We do so for the returns first, outside the competitor loop.
   The first step for each competitor is to find its Spearman correlation with
   the returns.  If it is negatively correlated flip its sign.
   I arbitrarily demand that at least 10 percent (0.1) of bars have a trade.
   Feel free to change this or make it a user parameter, but 0.1 generally works well.
*/

      for (i=0 ; i<(nprices-1)/2 ; i++) {    // Training set
         k = index[i] ;
         ret_work[i] = returns[k] ;          // Collect returns for training set
         }

      for (icomp=0 ; icomp<n_smooth ; icomp++) { // Process each competitor
         sort_key[icomp] = icomp ;
         xptr = indicators + icomp * nprices ;  // This competitor's observations
         for (i=0 ; i<(nprices-1)/2 ; i++) {    // Training set
            k = index[i] ;
            xwork[i] = xptr[k] ;                // Collect this competitor for training set
            }
         rho[icomp] = spearman ( (nprices-1)/2 , xwork , ret_work , work1 , work2 ) ;
         if (rho[icomp] < 0.0) {  // For simplicity I demand non-negative correlation
            for (i=0 ; i<(nprices-1)/2 ; i++)
               xwork[i] = -xwork[i] ;
            }
         opt_thresh ( (nprices-1)/2 , (int) (0.1 * (nprices-1)/2) , 0 , xwork , ret_work ,
                      &pf_all , high_thresh+icomp , &pf_high , low_thresh+icomp , &pf_low ,
                      work1 , work2 ) ;
         long_win = long_lose = short_win = short_lose = 1.e-60 ;
         for (i=0 ; i<(nprices-1)/2 ; i++) {      // Training set
            if (xwork[i] >= high_thresh[icomp]) { // Take a long position?
               if (ret_work[i] > 0.0)
                  long_win += ret_work[i] ;
               else
                  long_lose -= ret_work[i] ;
               }
            if (xwork[i] < low_thresh[icomp]) { // Take a short position?
               if (ret_work[i] < 0.0)
                  short_win -= ret_work[i] ;
               else
                  short_lose += ret_work[i] ;
               }
            } // For training set
         criters[icomp] = (long_win + short_win) / (long_lose + short_lose) ;
         } // For all competitors

      qsortdsi ( 0 , n_smooth-1 , criters , sort_key ) ;


/*
   Find the test-set criterion for each competitor
   The first step processing the test set is collecting the test set data.
   We do so for the returns first, outside the competitor loop.
   Then find the median test-set return.
*/

      for (i=(nprices-1)/2 ; i<nprices-1 ; i++) { // Test set
         k = index[i] ;
         ret_work[i] = returns[k] ;          // Collect returns for test set
         }
      for (icomp=0 ; icomp<n_smooth ; icomp++) {
         xptr = indicators + icomp * nprices ;       // This competitor's observations
         for (i=(nprices-1)/2 ; i<nprices-1 ; i++) { // Test set
            k = index[i] ;
            xwork[i] = xptr[k] ;                     // Collect this competitor for test set
            }
         if (rho[icomp] < 0.0) {  // If we flipped sign in training, must also in test
            for (i=(nprices-1)/2 ; i<nprices-1 ; i++) // Test set
               xwork[i] = -xwork[i] ;
            }
         long_win = long_lose = short_win = short_lose = 1.e-60 ;
         for (i=(nprices-1)/2 ; i<nprices-1 ; i++) { // Test set
            if (xwork[i] >= high_thresh[icomp]) { // Take a long position?
               if (ret_work[i] > 0.0)
                  long_win += ret_work[i] ;
               else
                  long_lose -= ret_work[i] ;
               }
            if (xwork[i] < low_thresh[icomp]) { // Take a short position?
               if (ret_work[i] < 0.0)
                  short_win -= ret_work[i] ;
               else
                  short_lose += ret_work[i] ;
               }
            } // For test set
         criters[icomp] = (long_win + short_win) / (long_lose + short_lose) ;
         work1[icomp] = criters[icomp] ;    // We cannot disturb test-set crits, so sort this for median
         } // For all competitors

      qsortd ( 0 , n_smooth-1 , work1 ) ;
      if (n_smooth % 2)
         median = work1[n_smooth/2] ;
      else
         median = 0.5 * (work1[n_smooth/2-1] + work1[n_smooth/2]) ;

      // For each training-set rank, see if its test-set criter fails to exceed the median
      for (icomp=0 ; icomp<n_smooth ; icomp++) {
         if (criters[sort_key[icomp]] <= median)
            ++counts[icomp] ;
         }
      } // For irep

// Divide counts by number of reps to make them probabilities

   for (icomp=0 ; icomp<n_smooth ; icomp++)
      counts[icomp] /= nreps ;

/*
--------------------------------------------------------------------------------

   The essential work is complete.
   At this time, counts[0] is the probability that the worst training-set
   competitor will have a test-set performance that fails to exceed the median.
   Similarly,  counts[n_smooth-1] is the probability that the best training-set
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

   for (icomp=0 ; icomp<n_smooth ; icomp++) {
      xptr = indicators + icomp * nprices ;    // This competitor's observations
      param = s_low * exp ( icomp / (n_smooth - 1.0) * log ( (double) s_high / (double) s_low ) ) ;
      rho[icomp] = spearman ( nprices-1 , xptr , returns , work1 , work2 ) ;
      if (rho[icomp] < 0.0) {  // For simplicity I demand non-negative correlation
         for (i=0 ; i<nprices-1 ; i++)
            xptr[i] = -xptr[i] ;
         }
      opt_thresh ( nprices-1 , (int) (0.1 * nprices) , 0 , xptr , returns , &pf_all ,
                   high_thresh+icomp , &pf_high , low_thresh+icomp , &pf_low ,
                   work1 , work2 ) ;
      printf ( "\n%6d %11.4lf %12.5lf %9.3lf %14.5lf %10.3lf",
               (int) (param + 0.5), rho[icomp], high_thresh[icomp], pf_high, low_thresh[icomp], pf_low ) ;
      }

   for (icomp=0 ; icomp<n_smooth ; icomp++) {
      sort_key[icomp] = icomp ;
      xptr = indicators + icomp * nprices ;  // This competitor's observations
      long_win = long_lose = short_win = short_lose = 1.e-60 ;
      for (i=0 ; i<nprices-1 ; i++) {
         if (xptr[i] >= high_thresh[icomp]) { // Take a long position?
            if (returns[i] > 0.0)
               long_win += returns[i] ;
            else
               long_lose -= returns[i] ;
            }
         if (xptr[i] < low_thresh[icomp]) { // Take a short position?
            if (returns[i] < 0.0)
               short_win -= returns[i] ;
            else
               short_lose += returns[i] ;
            }
         } // For training set
      criters[icomp] = (long_win + short_win) / (long_lose + short_lose) ;
      } // For all competitors

   qsortdsi ( 0 , n_smooth-1 , criters , sort_key ) ;

   printf ( "\n\nCompetitor performance ranked best to worst, with superiority 'p-values' (failure rates)" ) ;
   printf ( "\nProfit factors are optimistically biased because the threshold was optimized" ) ;
   printf ( "\nParameter  Profit factor   p-value" ) ;

   for (icomp=n_smooth-1 ; icomp>=0 ; icomp--) {  // Best to worst
      k = sort_key[icomp] ;
      param = s_low * exp ( k / (n_smooth - 1.0) * log ( (double) s_high / (double) s_low ) ) ;
      printf ( "\n%7d %12.3lf %12.3lf", (int) (param + 0.5), criters[icomp], counts[icomp] ) ;
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
