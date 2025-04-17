/******************************************************************************/
/*                                                                            */
/* BOOTMEAN - Demonstrate bootstrap confidence interval for mean              */
/*                                                                            */
/******************************************************************************/

#include <stdio.h>
#include <malloc.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <conio.h>
#include <assert.h>
#include "funcdefs.h"

#define MKTBUF 2048   /* Alloc for market info in chunks of this many records */


/*
--------------------------------------------------------------------------------

   Local routine computes mean return, the parameter being bootstrapped

--------------------------------------------------------------------------------
*/

double param_mean (  // Returns mean return
   int n ,           // Number of returns
   double *returns   // Log prices
   )
{
   int i ;
   double sum ;

   sum = 0.0 ;
   for (i=0 ; i<n ; i++)
      sum += returns[i] ;
   return sum / n ;
}

/*
--------------------------------------------------------------------------------

   Local routine computes optimal short-term and long-term lookbacks
   for a primitive moving-average crossover system
   We optimize profit factor based on log prices.
   (Performance criteria based on raw prices overly weights recent higher prices.)
   This is where you would substitute your own trading algorithm and
   optimization method.

--------------------------------------------------------------------------------
*/

double opt_params (  // Returns optimal profit factor
   int nprices ,     // Number of log prices in 'prices'
   double *prices ,  // Log prices
   int *short_term , // Returns optimal short-term lookback
   int *long_term    // Returns optimal long-term lookback
   )
{
   int i, j, ishort, ilong, ibestshort, ibestlong ;
   double short_sum, long_sum, short_mean, long_mean, best_perf ;
   double ret, win_sum, lose_sum ;

   best_perf = -1.e60 ;                             // Will be best performance across all trials
   *short_term = *long_term = 0 ;                   // Will be a problem flag if too few prices

   for (ilong=2 ; ilong<252 ; ilong++) {            // Trial long-term lookback
      if (ilong > nprices-1)                        // Never happens if careful user
         break ;                                    // We need ilong lookback plus 1 future
      for (ishort=1 ; ishort<=ilong/2 ; ishort++) { // Trial short-term lookback

         // We have a pair of lookbacks to try.
         // Cumulate performance for all valid cases

         win_sum = lose_sum = 1.e-60 ;           // Cumulates for profit factor

         for (i=ilong-1 ; i<nprices-1 ; i++) {    // Compute performance across history

            if (i == ilong-1) { // Find the short-term and long-term moving averages for the first valid case.
               short_sum = 0.0 ;                 // Cumulates short-term lookback sum
               for (j=i ; j>i-ishort ; j--)
                  short_sum += prices[j] ;
               long_sum = short_sum ;            // Cumulates long-term lookback sum
               while (j>i-ilong)
                  long_sum += prices[j--] ;
               }

            else {                               // Update the moving averages
               short_sum += prices[i] - prices[i-ishort] ;
               long_sum += prices[i] - prices[i-ilong] ;
               }

            short_mean = short_sum / ishort ;
            long_mean = long_sum / ilong ;

            // We now have the short-term and long-term moving averages ending at day i
            // Take our position and cumulate performance

            if (short_mean > long_mean)       // Long position
               ret = prices[i+1] - prices[i] ;
            else if (short_mean < long_mean)  // Short position
               ret = prices[i] - prices[i+1] ;
            else
               ret = 0.0 ;

            if (ret > 0.0)      // A positive return is a win
               win_sum += ret ;
            else
               lose_sum -= ret ;

            } // For i, summing performance for this trial

         // We now have the performance figures across the history
         // Keep track of the best lookbacks

         if (win_sum / lose_sum > best_perf) {
            best_perf = win_sum / lose_sum ;
            ibestshort = ishort ;
            ibestlong = ilong ;
            }
         } // For ishort, all short-term lookbacks
      } // For ilong, all long-term lookbacks

   *short_term = ibestshort ;
   *long_term = ibestlong ;

   return best_perf ;
}


/*
----------------------------------------------------------------------------------

   Local routine executes the trading system whose parameters have been optimized.
   This is where you would substitute your own trading algorithm

----------------------------------------------------------------------------------
*/

double execute (     // Returns profit factor
   int nprices ,     // Number of log prices in 'prices'
   int istart ,      // First trade decision is made on this price index
   double *prices ,  // Log prices
   int short_term ,  // Short-term lookback
   int long_term ,   // Long-term lookback
   int *nret ,       // Returns number of returns
   double *returns   // Returns returns
   )
{
   int i, j, n ;
   double short_sum, long_sum, short_mean, long_mean ;
   double ret, win_sum, lose_sum ;

   assert ( istart >= long_term - 1 ) ; // Must have enough lookback
   assert ( istart < nprices - 1 ) ;    // Must have one future for return

   n = 0 ;
   win_sum = lose_sum = 1.e-60 ;           // Cumulates for profit factor

   for (i=istart ; i<nprices-1 ; i++) {    // Compute performance across history

      if (i == istart) { // Find the short-term and long-term moving averages for the first valid case.
         short_sum = 0.0 ;                 // Cumulates short-term lookback sum
         for (j=i ; j>i-short_term ; j--)
            short_sum += prices[j] ;
         long_sum = short_sum ;            // Cumulates long-term lookback sum
         while (j>i-long_term)
            long_sum += prices[j--] ;
         }

      else {                               // Update the moving averages
         short_sum += prices[i] - prices[i-short_term] ;
         long_sum += prices[i] - prices[i-long_term] ;
         }

      short_mean = short_sum / short_term ;
      long_mean = long_sum / long_term ;

      // We now have the short-term and long-term moving averages ending at day i
      // Take our position and cumulate performance

      if (short_mean > long_mean)       // Long position
         ret = prices[i+1] - prices[i] ;
      else if (short_mean < long_mean)  // Short position
         ret = prices[i] - prices[i+1] ;
      else
         ret = 0.0 ;

      if (ret > 0.0)      // A positive return is a win
         win_sum += ret ;
      else
         lose_sum -= ret ;

//      returns[n++] = ret ; // I prefer this, but it's not standard
      returns[n++] = (exp(prices[i+1]) - exp(prices[i])) / exp(prices[i]) ; // Fractional return
      } // For i, summing performance for this trial

   *nret = n ;
   return win_sum / lose_sum ;
}


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
   int i, ntrain, nboot, nprices, bufcnt, opt_long, opt_short, nret ;
   double *open, *high, *low, *close, *returns, opt_return ;
   double *xwork, *work2, low_conf, high_conf, low_bound, high_bound ;
   char line[256], filename[4096], *cptr ;
   FILE *fp ;

/*
   Process command line parameters
*/

#if 1
   if (argc != 6) {
      printf ( "\nUsage: BOOTMEAN  ntrain  nboot  low_conf  high_conf  filename" ) ;
      printf ( "\n  ntrain - Number of training prices (includes lookback)" ) ;
      printf ( "\n  nboot - Number of bootstrap draws (hundreds or thousands)" ) ;
      printf ( "\n  low_conf - Lower confidence probability, 0-0.5" ) ;
      printf ( "\n  high_conf - Upper confidence probability, 0.5-1.0" ) ;
      printf ( "\n  filename - name of market file (YYYYMMDD Open High Low Close)" ) ;
      exit ( 1 ) ;
      }

   ntrain = atoi ( argv[1] ) ;
   nboot = atoi ( argv[2] ) ;
   low_conf = atof ( argv[3] ) ;
   high_conf = atof ( argv[4] ) ;
   strcpy_s ( filename , argv[5] ) ;
#else
   ntrain = 7936 ;
   nboot = 10000 ;
   low_conf = 0.1 ;
   high_conf = 0.9 ;
   strcpy_s ( filename , "E:\\MarketDataAssorted\\ETF\\QQQ\\QQQ.TXT" ) ;
   strcpy_s ( filename , "E:\\MarketDataAssorted\\INDEXES\\OEX.TXT" ) ;
#endif


/*
   Read market prices
*/

   if (fopen_s ( &fp, filename , "rt" )) {
      printf ( "\n\nCannot open market history file %s", filename ) ;
      exit ( 1 ) ;
      }

   returns = NULL ;
   xwork = NULL ;
   work2 = NULL ;

   open = (double *) malloc ( MKTBUF * sizeof(double) ) ;
   high = (double *) malloc ( MKTBUF * sizeof(double) ) ;
   low = (double *) malloc ( MKTBUF * sizeof(double) ) ;
   close = (double *) malloc ( MKTBUF * sizeof(double) ) ;
   if (open == NULL  ||  high == NULL  ||  low == NULL  ||  close == NULL) {
      if (open != NULL)
         free ( open ) ;
      if (high != NULL)
         free ( high ) ;
      if (low != NULL)
         free ( low ) ;
      if (close != NULL)
         free ( close ) ;
      printf ( "\n\nInsufficient memory reading market history file %s  Press any key...", filename ) ;
      _getch () ;  // Wait for user to press a key
      fclose ( fp ) ;
      exit ( 1 ) ;
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
         free ( open ) ;
         free ( high ) ;
         free ( low ) ;
         free ( close ) ;
         printf ( "\nError reading line %d of file %s", nprices+1, filename ) ;
         exit ( 1 ) ;
         }

      if (! bufcnt) {  // Allocate a new memory block if needed
         open = (double *) realloc ( open , (nprices+MKTBUF) * sizeof(double) ) ;
         high = (double *) realloc ( high , (nprices+MKTBUF) * sizeof(double) ) ;
         low = (double *) realloc ( low , (nprices+MKTBUF) * sizeof(double) ) ;
         close = (double *) realloc ( close , (nprices+MKTBUF) * sizeof(double) ) ;
         if (open == NULL  ||  high == NULL  ||  low == NULL  ||  close == NULL) {
            if (open != NULL)
               free ( open ) ;
            if (high != NULL)
               free ( high ) ;
            if (low != NULL)
               free ( low ) ;
            if (close != NULL)
               free ( close ) ;
            fclose ( fp ) ;
            printf ( "\n\nInsufficient memory reading market history file %s  Press any key...", filename ) ;
            _getch () ;  // Wait for user to press a key
            exit ( 1 ) ;
            } // If insufficient memory
         bufcnt = MKTBUF ;  // There are this many new record slots available now
         } // If allocating new block

      // Parse the date and do a crude sanity check

      for (i=0 ; i<8 ; i++) {
         if ((line[i] < '0')  ||  (line[i] > '9')) {
            fclose ( fp ) ;
            free ( open ) ;
            free ( high ) ;
            free ( low ) ;
            free ( close ) ;
            printf ( "\nInvalid date reading line %d of file %s", nprices+1, filename ) ;
            exit ( 1 ) ;
            }
         }

      // Parse the open

      cptr = line + 9 ;  // Price is in this column or beyond
                         // (Next loop allows price to start past this)

      while (*cptr == ' '  ||  *cptr == '\t'  ||  *cptr == ',')  // Delimiters
         ++cptr ;  // Move up to the price

      open[nprices] = atof ( cptr ) ;
      if (open[nprices] > 0.0)                     // Always true, but avoid disaster
         open[nprices] = log ( open[nprices] ) ;

      while (*cptr  &&  *cptr != ' '  &&  *cptr != ','  &&  *cptr != '\t')
         ++cptr ;  // Pass the price; stop at delimiter

      // Parse the high

      while (*cptr == ' '  ||  *cptr == '\t'  ||  *cptr == ',')  // Delimiters
         ++cptr ;  // Move up to the price

      high[nprices] = atof ( cptr ) ;
      if (high[nprices] > 0.0)                     // Always true, but avoid disaster
         high[nprices] = log ( high[nprices] ) ;

      while (*cptr  &&  *cptr != ' '  &&  *cptr != ','  &&  *cptr != '\t')
         ++cptr ;  // Pass the price; stop at delimiter

      // Parse the low

      while (*cptr == ' '  ||  *cptr == '\t'  ||  *cptr == ',')  // Delimiters
         ++cptr ;  // Move up to the price

      low[nprices] = atof ( cptr ) ;
      if (low[nprices] > 0.0)                     // Always true, but avoid disaster
         low[nprices] = log ( low[nprices] ) ;

      while (*cptr  &&  *cptr != ' '  &&  *cptr != ','  &&  *cptr != '\t')
         ++cptr ;  // Pass the price; stop at delimiter

      // Parse the close

      while (*cptr == ' '  ||  *cptr == '\t'  ||  *cptr == ',')  // Delimiters
         ++cptr ;  // Move up to the price

      close[nprices] = atof ( cptr ) ;
      if (close[nprices] > 0.0)                     // Always true, but avoid disaster
         close[nprices] = log ( close[nprices] ) ;

      if (low[nprices] > open[nprices]  ||  low[nprices] > close[nprices]  ||
          high[nprices] < open[nprices]  ||  high[nprices] < close[nprices]) {
         fclose ( fp ) ;
         free ( open ) ;
         free ( high ) ;
         free ( low ) ;
         free ( close ) ;
         printf ( "\nInvalid open/high/low/close reading line %d of file %s", nprices+1, filename ) ;
         exit ( 1 ) ;
         }

      ++nprices  ;
      --bufcnt ;           // One less slot remains

      } // For all lines

   fclose ( fp ) ;

   printf ( "\n\n%d prices were read.  Training with the first %d...", nprices, ntrain ) ;

   returns = (double *) malloc ( nprices * sizeof(double) ) ;
   xwork = (double *) malloc ( nprices * sizeof(double) ) ;
   work2 = (double *) malloc ( nboot * sizeof(double) ) ;
   if (returns == NULL  ||  xwork == NULL  ||  work2 == NULL) {
      printf ( "\n\nInsufficient memory" ) ;
      goto FINISH ;
      }

/*
   The market data is read.  Train, execute to get returns, and run bootstrap test.
*/

   if (nprices < ntrain+1) {
      printf ( "\nERROR... Number of prices must be at least number of training cases plus one" ) ;
      exit ( 1 ) ;
      }

   opt_return = opt_params ( ntrain , close , &opt_short , &opt_long ) ; // Train

   printf ( "\nProfit factor based on log prices = %.5lf  optimal long lookback = %d  short lookback = %d",
            opt_return, opt_long, opt_short ) ;

   opt_return = execute ( nprices , ntrain-1 , close , opt_short , opt_long , &nret , returns ) ;

   printf ( "\nOOS profit factor = %.5lf with %d returns\n", opt_return, nret ) ;

   boot_conf_BCa ( nret , returns , param_mean , nboot , low_conf , high_conf , &low_bound , &high_bound , xwork , work2 ) ;

   printf ( "\nUser-specified confidence limits = %.4lf %.4lf", low_conf , high_conf ) ;
   printf ( "\nAnnualized daily percent confidence bounds = %.4lf %.4lf", 100 * 252 * low_bound, 100 * 252 * high_bound ) ;

FINISH:
   printf ( "\n\nPress any key..." ) ;
   _getch () ;  // Wait for user to press a key

   if (returns != NULL)
      free ( returns ) ;
   if (xwork != NULL)
      free ( xwork ) ;
   if (work2 != NULL)
      free ( work2 ) ;
   free ( open ) ;
   free ( high ) ;
   free ( low ) ;
   free ( close ) ;

   exit ( 0 ) ;
}
