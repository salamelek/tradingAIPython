/******************************************************************************/
/*                                                                            */
/*  SUP2 - Test PRICE INTENSITY(2) with multiple markets competing            */
/*                                                                            */
/******************************************************************************/

#include <windows.h>
#include <stdio.h>
#include <malloc.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <conio.h>
#include <assert.h>
#include "funcdefs.h"

#define MAX_MARKETS 256     /* Max number of markets that we can have */
#define MAX_NAME_LENGTH 16  /* Max length of a market name */
#define MKTBUF 2048         /* Alloc for market info in chunks of this many records */
                            /* This is not critical and can be any reasonable value */


/*
-------------------------------------------------------------------------------

   Local routine to read market prices and volume if present

-------------------------------------------------------------------------------
*/

static int read_market (
   char *MarketName ,
   int **date ,
   double **open ,
   double **high ,
   double **low ,
   double **close ,
   double **volume ,
   int *nprices
   )
{
   int i, bufcnt ;
   char line[256], *cptr ;
   FILE *fp ;

   if (fopen_s ( &fp, MarketName , "rt" )) {
      printf ( "\n\nCannot open market history file %s", MarketName ) ;
      return 1 ;
      }

   *date = (int *) malloc ( MKTBUF * sizeof(int) ) ;
   *open = (double *) malloc ( MKTBUF * sizeof(double) ) ;
   *high = (double *) malloc ( MKTBUF * sizeof(double) ) ;
   *low = (double *) malloc ( MKTBUF * sizeof(double) ) ;
   *close = (double *) malloc ( MKTBUF * sizeof(double) ) ;
   *volume = (double *) malloc ( MKTBUF * sizeof(double) ) ;
   if (*date == NULL  ||  *open == NULL  ||  *high == NULL  ||  *low == NULL  ||  *close == NULL  ||  *volume == NULL) {
      printf ( "\n\nInsufficient memory reading market history file %s", MarketName ) ;
      fclose ( fp ) ;
      if (*date != NULL)
         free ( *date ) ;
      if (*open != NULL)
         free ( *open ) ;
      if (*high != NULL)
         free ( *high ) ;
      if (*low != NULL)
         free ( *low ) ;
      if (*close != NULL)
         free ( *close ) ;
      if (*volume != NULL)
         free ( *volume ) ;
      return 1 ;
      }

   bufcnt = MKTBUF ;  // There are this many record slots available now

   printf ( "\nReading market file %s", MarketName ) ;

   *nprices = 0 ;    // Counts lines (prices) read

   for (;;) {

      if (feof ( fp )                          // If end of file
       || (fgets ( line , 256 , fp ) == NULL)  // Or unable to read line
       || (strlen ( line ) < 2))               // Or empty line
         break ;                               // We are done reading price history

      if (ferror ( fp )) {                     // If an error reading file
         fclose ( fp ) ;                       // Quit immediately
         printf ( "\nError reading line %d of file %s", *nprices+1, MarketName ) ;
         free ( *date ) ;
         free ( *open ) ;
         free ( *high ) ;
         free ( *low ) ;
         free ( *close ) ;
         free ( *volume ) ;
         return 1 ;
         }

      if (! bufcnt) {  // Allocate a new memory block if needed
         *date = (int *) realloc ( *date , (*nprices+MKTBUF) * sizeof(int) ) ;
         *open = (double *) realloc ( *open , (*nprices+MKTBUF) * sizeof(double) ) ;
         *high = (double *) realloc ( *high , (*nprices+MKTBUF) * sizeof(double) ) ;
         *low = (double *) realloc ( *low , (*nprices+MKTBUF) * sizeof(double) ) ;
         *close = (double *) realloc ( *close , (*nprices+MKTBUF) * sizeof(double) ) ;
         *volume = (double *) realloc ( *volume , (*nprices+MKTBUF) * sizeof(double) ) ;
         if (*date == NULL  ||  *open == NULL  ||  *high == NULL  ||  *low == NULL  ||  *close == NULL  ||  *volume == NULL) {
            fclose ( fp ) ;
            printf ( "\n\nInsufficient memory reading market history file %s  Press any key...", MarketName ) ;
            if (*date != NULL)
               free ( *date ) ;
            if (*open != NULL)
               free ( *open ) ;
            if (*high != NULL)
               free ( *high ) ;
            if (*low != NULL)
               free ( *low ) ;
            if (*close != NULL)
               free ( *close ) ;
            if (*volume != NULL)
               free ( *volume ) ;
            return 1 ;
            } // If insufficient memory
         bufcnt = MKTBUF ;  // There are this many new record slots available now
         } // If allocating new block

      // Parse the date and do a crude sanity check

      for (i=0 ; i<8 ; i++) {
         if ((line[i] < '0')  ||  (line[i] > '9')) {
            fclose ( fp ) ;
            printf ( "\nInvalid date reading line %d of file %s", *nprices+1, MarketName ) ;
            free ( *date ) ;
            free ( *open ) ;
            free ( *high ) ;
            free ( *low ) ;
            free ( *close ) ;
            free ( *volume ) ;
            return 1 ;
            }
         }
      (*date)[*nprices] = atoi ( line ) ;

      // Parse the open

      cptr = line + 9 ;  // Price is in this column or beyond
                         // (Next loop allows price to start past this)

      while (*cptr == ' '  ||  *cptr == '\t'  ||  *cptr == ',')  // Delimiters
         ++cptr ;  // Move up to the price

      (*open)[*nprices] = atof ( cptr ) ;

      while (*cptr  &&  *cptr != ' '  &&  *cptr != ','  &&  *cptr != '\t')
         ++cptr ;  // Pass the price; stop at delimiter

      // Parse the high

      while (*cptr == ' '  ||  *cptr == '\t'  ||  *cptr == ',')  // Delimiters
         ++cptr ;  // Move up to the price

      (*high)[*nprices] = atof ( cptr ) ;

      while (*cptr  &&  *cptr != ' '  &&  *cptr != ','  &&  *cptr != '\t')
         ++cptr ;  // Pass the price; stop at delimiter

      // Parse the low

      while (*cptr == ' '  ||  *cptr == '\t'  ||  *cptr == ',')  // Delimiters
         ++cptr ;  // Move up to the price

      (*low)[*nprices] = atof ( cptr ) ;

      while (*cptr  &&  *cptr != ' '  &&  *cptr != ','  &&  *cptr != '\t')
         ++cptr ;  // Pass the price; stop at delimiter

      // Parse the close

      while (*cptr == ' '  ||  *cptr == '\t'  ||  *cptr == ',')  // Delimiters
         ++cptr ;  // Move up to the price

      (*close)[*nprices] = atof ( cptr ) ;

      while (*cptr  &&  *cptr != ' '  &&  *cptr != ','  &&  *cptr != '\t')
         ++cptr ;  // Pass the price; stop at delimiter

      // Parse the volume

      while (*cptr == ' '  ||  *cptr == '\t'  ||  *cptr == ',')  // Delimiters
         ++cptr ;  // Move up

      (*volume)[*nprices] = atof ( cptr ) ;

      if ((*low)[*nprices] > (*open)[*nprices]  ||  (*low)[*nprices] > (*close)[*nprices]  ||
          (*high)[*nprices] < (*open)[*nprices]  ||  (*high)[*nprices] < (*close)[*nprices]) {
         fclose ( fp ) ;
         printf ( "\nInvalid open/high/low/close reading line %d of file %s", *nprices+1, MarketName ) ;
         free ( *date ) ;
         free ( *open ) ;
         free ( *high ) ;
         free ( *low ) ;
         free ( *close ) ;
         free ( *volume ) ;
         return 1 ;
         }

      ++*nprices  ;
      --bufcnt ;           // One less slot remains

      } // For all lines

   fclose ( fp ) ;

   printf ( "\nMarket price history read; %d cases from %d to %d", *nprices, (*date)[0], (*date)[*nprices-1] ) ;
   return 0 ;
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
   int i, j, k, imarket, ibar, irep, nreps, n_markets, lookback ;
   int date, max_date, all_same_date, n_bars ;
   int **market_date, *market_index, *market_n, grand_index, *index, *sort_key ;
   double **market_open, **market_high, **market_low, **market_close, **market_volume ;
   double *indicators, *returns, *work1, *work2, *xwork, *ret_work ;
   double *ret_ptr, *low_thresh, *high_thresh, *counts, *criters, *rho ;
   double xmean, xmin, xmax, *xptr ;
   double long_win, long_lose, short_win, short_lose, pf_all, pf_low, pf_high, median ;
   char *market_names ;
   char MarketListName[1024], MarketFileName[1024], line[512], msg[256], *lptr ;
   FILE *fp ;

   indicators = NULL ;     // We'll allocate this when we know how many bars
   work1 = work2 = NULL ;  // Ditto
   returns = NULL ;        // Ditto
   index = NULL ;          // Ditto
   xwork = NULL ;          // Ditto
   ret_work = NULL ;       // Ditto


/*
   Process command line parameters
*/

#if 1
   if (argc != 4) {
      printf ( "\nUsage: SUP2  Lookback  Nreps  MarketList" ) ;
      printf ( "\n  Lookback - Smoothing lookback" ) ;
      printf ( "\n  Nreps - Number of replications" ) ;
      printf ( "\n  MarketList - List of all markets (complete file names)" ) ;
      exit ( 1 ) ;
      }

   lookback = atoi ( argv[1] ) ;
   nreps = atoi ( argv[2] ) ;
   strcpy_s ( MarketListName , argv[3] ) ;
#else
   lookback = 2 ;
   nreps = 10 ;
   strcpy_s ( MarketListName , "D:\\MCPT\\SUP2\\MULT_MKTS.TXT" ) ;
#endif


   market_names = NULL ;
   market_date = NULL ;
   market_index = NULL ;
   market_n = NULL ;
   market_open = NULL ;
   market_high = NULL ;
   market_low = NULL ;
   market_close = NULL ;
   market_volume = NULL ;
   fp = NULL ;
   n_markets = 0 ;


/*
-------------------------------------------------------------------------------

   Allocate memory that we can allocate now
   It's trivially inefficient to allocate MAX_MARKETS when we could wait
   and allocate just n_markets later, when we know it.
   But the waste is tiny, and this is simpler.

-------------------------------------------------------------------------------
*/

   indicators = NULL ;     // We'll allocate this when we know how many bars
   work1 = work2 = NULL ;  // Ditto
   returns = NULL ;        // Ditto
   index = NULL ;          // Ditto
   xwork = NULL ;          // Ditto
   ret_work = NULL ;       // Ditto

   if (fopen_s ( &fp , MarketListName , "rt" )) {
      printf ( "\nCannot open file list %s... aborting", MarketListName ) ;
      goto FINISH ;
      }

   low_thresh = (double *) malloc ( MAX_MARKETS * sizeof(double) ) ;
   assert (low_thresh != NULL ) ;

   high_thresh = (double *) malloc ( MAX_MARKETS * sizeof(double) ) ;
   assert (high_thresh != NULL ) ;

   counts = (double *) malloc ( MAX_MARKETS * sizeof(double) ) ;
   assert (counts != NULL ) ;

   criters = (double *) malloc ( MAX_MARKETS * sizeof(double) ) ;
   assert (criters != NULL ) ;

   sort_key = (int *) malloc ( MAX_MARKETS * sizeof(int) ) ;
   assert (sort_key != NULL ) ;

   rho = (double *) malloc ( MAX_MARKETS * sizeof(double) ) ;
   assert (rho != NULL ) ;

   market_names = (char *) malloc ( MAX_MARKETS * MAX_NAME_LENGTH * sizeof(char) ) ;
   assert ( market_names != NULL ) ;

   market_date = (int **) malloc ( MAX_MARKETS * sizeof(int *) ) ;
   assert ( market_date != NULL ) ;

   market_index = (int *) malloc ( MAX_MARKETS * sizeof(int) ) ;
   assert ( market_index != NULL ) ;

   market_n = (int *) malloc ( MAX_MARKETS * sizeof(int) ) ;
   assert ( market_n != NULL ) ;

   market_open = (double **) malloc ( MAX_MARKETS * sizeof(double *) ) ;
   assert ( market_open != NULL ) ;

   market_high = (double **) malloc ( MAX_MARKETS * sizeof(double *) ) ;
   assert ( market_high != NULL ) ;

   market_low = (double **) malloc ( MAX_MARKETS * sizeof(double *) ) ;
   assert ( market_low != NULL ) ;

   market_close = (double **) malloc ( MAX_MARKETS * sizeof(double *) ) ;
   assert ( market_close != NULL ) ;

   market_volume = (double **) malloc ( MAX_MARKETS * sizeof(double *) ) ;
   assert ( market_volume != NULL ) ;

   for (i=0 ; i<MAX_MARKETS ; i++) {
      market_date[i] = NULL ;
      market_open[i] = NULL ;
      market_high[i] = NULL ;
      market_low[i] = NULL ;
      market_close[i] = NULL ;
      market_volume[i] = NULL ;
      }

/*
-------------------------------------------------------------------------------

   Read the market list file and market files

-------------------------------------------------------------------------------
*/

   for (;;) {

      // Get the name of a market file
      if ((fgets ( line , 256 , fp ) == NULL) || (strlen ( line ) < 2)) {
         if (ferror ( fp )  ||  ! n_markets) {
            printf ( "\nERROR... Cannot read market list file %s", MarketListName ) ;
            goto FINISH ;
            }
         else
            break ;       // Normal end of list file
         }

      // Copy this market file name

      lptr = &line[0] ;
      k = 0 ;
      while (isalnum(*lptr)  ||  *lptr == '_'  ||  *lptr == '\\'  ||  *lptr == ':'  ||  *lptr == '.')
         MarketFileName[k++] = *lptr++ ;
      MarketFileName[k] = 0 ;  // This is now the exact file name

      // Get and save the name of the market from the file name
      // We assume it is just before the last period.

      strcpy_s ( msg , MarketFileName ) ;
      lptr = &msg[k-1] ;  // Last character in file name
      while (lptr > &msg[0]  &&  *lptr != '.')
         --lptr ;
      if (*lptr != '.') {   // We require an extension, not unreasonable
         printf ( "\nERROR... Market file name (%s) is not legal", MarketFileName ) ;
         goto FINISH ;
         }
      *lptr = 0 ;   // This removes extension
      while (lptr > &msg[0]  &&  *lptr != '.'  &&  *lptr != '\\'  &&  *lptr != ':')
         --lptr ;   // Back up until we get path stuff
      if (*lptr == '.'  ||  *lptr == '\\'  ||  *lptr == ':')  // If a path character caused loop exit, pass it
         ++lptr ;
      if (strlen ( lptr ) > MAX_NAME_LENGTH-1) {
         printf ( "\nERROR... Market name (%s) is too long", lptr ) ;
         goto FINISH ;
         }
      strcpy_s ( market_names+n_markets*MAX_NAME_LENGTH , MAX_NAME_LENGTH , lptr ) ;

      // We now have the name of a market history file.  Read this file.

      if (read_market ( MarketFileName , market_date+n_markets , market_open+n_markets , market_high+n_markets ,
                        market_low+n_markets , market_close+n_markets , market_volume+n_markets , market_n+n_markets )) {
         printf ( "\nError reading market %s... aborting", lptr ) ;
         goto FINISH ;
         }
      ++n_markets ;
      } // Read all markets

   fclose ( fp ) ;
   fp = NULL ;


/*
-----------------------------------------------------------------------------------------

   We have completely finished reading all markets, and their data is stored.
   However, this test requires that all data be date aligned, and
   there is no guaranty that we have this.
   So now we remove all records that do not have data for all markets.
   The algorithm given here may not be the most efficient way to do this,
   but compared with what's coming soon this is a drop in the bucket of CPU time.

   The algorithm is as follows:
      Initialize each market's current index to 0
      Initialize the grand (compressed) index to 0
      Loop
         Find the latest (largest) date at each market's current index across all markets
         Advance all markets' current index until the date reaches or passes this date
         If all markets have the same current date:
            keep this date by copying this market record to the grand index spot
            Advance each market's current index as well as the grand index

-----------------------------------------------------------------------------------------
*/

   for (i=0 ; i<n_markets ; i++)
      market_index[i] = 0 ;
   grand_index = 0 ;

   for (;;) {

      // Find max date at current index of each market

      max_date = 0 ;
      for (i=0 ; i<n_markets ; i++) {
         date = market_date[i][market_index[i]] ;
         if (date > max_date)
            max_date = date ;
         }

      // Advance all markets until they reach or pass max date
      // Keep track of whether they all equal max_date

      all_same_date = 1 ;
      for (i=0 ; i<n_markets ; i++) {
         while (market_index[i] < market_n[i]) {    // Must not over-run a market!
            date = market_date[i][market_index[i]] ;
            if (date >= max_date)
               break ;
            ++market_index[i] ;
            }
         if (date != max_date)                // Did some market jump over max?
            all_same_date = 0 ;

         if (market_index[i] >= market_n[i])  // If even one market runs out
            break ;                           // We are done
         }

      if (i < n_markets)  // If even one market runs out
         break ;          // We are done

      // If we have a complete set for this date, grab it

      if (all_same_date) {
         for (i=0 ; i<n_markets ; i++) {
            assert ( market_date[i][market_index[i]] = max_date ) ;
            market_date[i][grand_index] = max_date ;  // Redundant, but clear
            market_open[i][grand_index] = market_open[i][market_index[i]] ;
            market_high[i][grand_index] = market_high[i][market_index[i]] ;
            market_low[i][grand_index] = market_low[i][market_index[i]] ;
            market_close[i][grand_index] = market_close[i][market_index[i]] ;
            market_volume[i][grand_index] = market_volume[i][market_index[i]] ;
            ++market_index[i] ;
            }
         ++grand_index ;
         }
      } // Endless compression loop

   n_bars = grand_index ;

   printf ( "\n\nMerged database has %d records from date %d to %d",
            n_bars, market_date[0][0], market_date[0][n_bars-1] ) ;

/*
   Free memory that we no longer need
*/

   free ( market_n ) ;
   market_n = NULL ;
   free ( market_index ) ;
   market_index = NULL ;

   if (n_bars == 0) {
      printf ( "\nAborting because there are no common dates" ) ;
      goto FINISH ;
      }


/*
-----------------------------------------------------------------------------------------

   The markets are all read.  We're ready to go.
   First, allocate memory now that we know how many bars there are.

-----------------------------------------------------------------------------------------
*/


   if (n_bars < 20) {  // Really too small, but do something!
      printf ( "\nERROR... Number of prices must be at least 20" ) ;
      goto FINISH ;
      }

   indicators = (double *) malloc ( n_markets * n_bars * sizeof(double) ) ;
   assert ( indicators != NULL ) ;

   returns = (double *) malloc ( n_markets * n_bars * sizeof(double) ) ;
   assert ( returns != NULL ) ;

   work1 = (double *) malloc ( n_bars * sizeof(double) ) ;
   assert ( work1 != NULL ) ;

   work2 = (double *) malloc ( n_bars * sizeof(double) ) ;
   assert ( work2 != NULL ) ;

   xwork = (double *) malloc ( n_bars * sizeof(double) ) ;
   assert ( xwork != NULL ) ;

   ret_work = (double *) malloc ( n_bars * sizeof(double) ) ;
   assert ( ret_work != NULL ) ;

   index = (int *) malloc ( n_bars * sizeof(int) ) ;
   assert ( index != NULL ) ;


/*
   Compute the returns for each as log close-to-close
   We have no return for the last bar because we have to look ahead one bar
*/

   for (imarket=0 ; imarket<n_markets ; imarket++) {
      for (ibar=0 ; ibar<n_bars-1 ; ibar++)
         returns[imarket*n_bars+ibar] = log ( market_close[imarket][ibar+1] / market_close[imarket][ibar] ) ;
      }


/*
   Fill in the indicator matrix.
   I store the values with the observation changing fastest, as that makes for fastest processing.
   My illustration in the book shows the observations going down, while traditional C++
   array storage has the columns changing fastest.  Thus, my actual storage here is the
   transpose of the book illustration.  However, I wanted to use the same illustration as
   was used in the Bailey et al paper in case anyone refers back to that paper.

   For a sanity check, I also compute the mean, minimum, and maximum for each competitor (market).
*/


   printf ( "\n\n         Market        Mean       Minimum      Maximum" ) ;

   for (imarket=0 ; imarket<n_markets ; imarket++) {
      xptr = indicators + imarket * n_bars ;    // This competitor's observations
      comp_ind ( n_bars , market_open[imarket] , market_high[imarket] ,
                 market_low[imarket] , market_close[imarket] , lookback , xptr ) ;

      xmean = xmin = xmax = xptr[0] ;
      for (i=1 ; i<n_bars ; i++) {
         xmean += xptr[i] ;
         if (xptr[i] < xmin)
            xmin = xptr[i] ;
         if (xptr[i] > xmax)
            xmax = xptr[i] ;
         }
      xmean /= n_bars ;

      printf ( "\n%15s %12.5lf %12.5lf %12.5lf", market_names+imarket*MAX_NAME_LENGTH, xmean, xmin, xmax ) ;
      }
               

/*
--------------------------------------------------------------------------------

   Do the replications

--------------------------------------------------------------------------------
*/

   for (imarket=0 ; imarket<n_markets ; imarket++)
      counts[imarket] = 0.0 ;  // Will count failures to exceed median for each ranked competitor

   for (ibar=0 ; ibar<n_bars-1 ; ibar++)
      index[ibar] = ibar ;   // Will index bars

   for (irep=0 ; irep<nreps ; irep++) {

      // Shuffle the bar indices.  First half will be training, second half test.

      i = n_bars-1 ;     // Number remaining to be shuffled; we lose last price due to lookahead
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
   We must fetch the returns inside the competitor loop, because markets are competing.
   The first step for each competitor is to find its Spearman correlation with
   the returns.  If it is negatively correlated flip its sign.
   I arbitrarily demand that at least 10 percent (0.1) of bars have a trade.
   Feel free to change this or make it a user parameter, but 0.1 generally works well.
*/

      for (imarket=0 ; imarket<n_markets ; imarket++) { // Process each competitor
         sort_key[imarket] = imarket ;
         for (i=0 ; i<(n_bars-1)/2 ; i++) {      // Training set
            k = index[i] ;
            ret_work[i] = returns[imarket*n_bars+k] ; // Collect returns for training set
            }
         xptr = indicators + imarket * n_bars ;  // This competitor's observations
         for (i=0 ; i<(n_bars-1)/2 ; i++) {      // Training set
            k = index[i] ;
            xwork[i] = xptr[k] ;                 // Collect this competitor for training set
            }
         rho[imarket] = spearman ( (n_bars-1)/2 , xwork , ret_work , work1 , work2 ) ;
         if (rho[imarket] < 0.0) {  // For simplicity I demand non-negative correlation
            for (i=0 ; i<(n_bars-1)/2 ; i++)
               xwork[i] = -xwork[i] ;
            }
         opt_thresh ( (n_bars-1)/2 , (int) (0.1 * (n_bars-1)/2) , 0 , xwork , ret_work ,
                      &pf_all , high_thresh+imarket , &pf_high , low_thresh+imarket , &pf_low ,
                      work1 , work2 ) ;
         long_win = long_lose = short_win = short_lose = 1.e-60 ;
         for (i=0 ; i<(n_bars-1)/2 ; i++) {      // Training set
            if (xwork[i] >= high_thresh[imarket]) { // Take a long position?
               if (ret_work[i] > 0.0)
                  long_win += ret_work[i] ;
               else
                  long_lose -= ret_work[i] ;
               }
            if (xwork[i] < low_thresh[imarket]) { // Take a short position?
               if (ret_work[i] < 0.0)
                  short_win -= ret_work[i] ;
               else
                  short_lose += ret_work[i] ;
               }
            } // For training set
         criters[imarket] = (long_win + short_win) / (long_lose + short_lose) ;
         } // For all competitors

      qsortdsi ( 0 , n_markets-1 , criters , sort_key ) ;


/*
   Find the test-set criterion for each competitor
   The first step processing the test set is collecting the test set data.
   Then find the median test-set return.
*/

      for (imarket=0 ; imarket<n_markets ; imarket++) {
         for (i=(n_bars-1)/2 ; i<n_bars-1 ; i++) {    // Test set
            k = index[i] ;
            ret_work[i] = returns[imarket*n_bars+k] ; // Collect returns for test set
            }
         xptr = indicators + imarket * n_bars ;       // This competitor's observations
         for (i=(n_bars-1)/2 ; i<n_bars-1 ; i++) {    // Test set
            k = index[i] ;
            xwork[i] = xptr[k] ;                      // Collect this competitor for test set
            }
         if (rho[imarket] < 0.0) {  // If we flipped sign in training, must also in test
            for (i=(n_bars-1)/2 ; i<n_bars-1 ; i++)   // Test set
               xwork[i] = -xwork[i] ;
            }
         long_win = long_lose = short_win = short_lose = 1.e-60 ;
         for (i=(n_bars-1)/2 ; i<n_bars-1 ; i++) { // Test set
            if (xwork[i] >= high_thresh[imarket]) { // Take a long position?
               if (ret_work[i] > 0.0)
                  long_win += ret_work[i] ;
               else
                  long_lose -= ret_work[i] ;
               }
            if (xwork[i] < low_thresh[imarket]) { // Take a short position?
               if (ret_work[i] < 0.0)
                  short_win -= ret_work[i] ;
               else
                  short_lose += ret_work[i] ;
               }
            } // For test set
         criters[imarket] = (long_win + short_win) / (long_lose + short_lose) ;
         work1[imarket] = criters[imarket] ;    // We cannot disturb test-set crits, so sort this for median
         } // For all competitors

      qsortd ( 0 , n_markets-1 , work1 ) ;
      if (n_markets % 2)
         median = work1[n_markets/2] ;
      else
         median = 0.5 * (work1[n_markets/2-1] + work1[n_markets/2]) ;

      // For each training-set rank, see if its test-set criter fails to exceed the median
      for (imarket=0 ; imarket<n_markets ; imarket++) {
         if (criters[sort_key[imarket]] <= median)
            ++counts[imarket] ;
         }
      } // For irep

// Divide counts by number of reps to make them probabilities

   for (imarket=0 ; imarket<n_markets ; imarket++)
      counts[imarket] /= nreps ;


/*
--------------------------------------------------------------------------------

   The essential work is complete.
   At this time, counts[0] is the probability that the worst training-set
   competitor will have a test-set performance that fails to exceed the median.
   Similarly,  counts[n_markets-1] is the probability that the best training-set
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
   printf ( "\n\n         Market      Correl   Long thresh   Long pf   Short thresh   Short pf" ) ;

   for (imarket=0 ; imarket<n_markets ; imarket++) {
      xptr = indicators + imarket * n_bars ;    // This competitor's observations
      rho[imarket] = spearman ( n_bars-1 , xptr , returns+imarket*n_bars , work1 , work2 ) ;
      if (rho[imarket] < 0.0) {  // For simplicity I demand non-negative correlation
         for (i=0 ; i<n_bars-1 ; i++)
            xptr[i] = -xptr[i] ;
         }
      opt_thresh ( n_bars-1 , (int) (0.1 * n_bars) , 0 , xptr , returns+imarket*n_bars ,
                   &pf_all , high_thresh+imarket , &pf_high , low_thresh+imarket , &pf_low ,
                   work1 , work2 ) ;
      printf ( "\n%15s %11.4lf %12.5lf %9.3lf %14.5lf %10.3lf",
               market_names+imarket*MAX_NAME_LENGTH, rho[imarket], high_thresh[imarket], pf_high, low_thresh[imarket], pf_low ) ;
      }

   for (imarket=0 ; imarket<n_markets ; imarket++) {
      sort_key[imarket] = imarket ;
      xptr = indicators + imarket * n_bars ;  // This competitor's observations
      ret_ptr = returns + imarket * n_bars ;  // This competitor's market returns
      long_win = long_lose = short_win = short_lose = 1.e-60 ;
      for (i=0 ; i<n_bars-1 ; i++) {
         if (xptr[i] >= high_thresh[imarket]) { // Take a long position?
            if (ret_ptr[i] > 0.0)
               long_win += ret_ptr[i] ;
            else
               long_lose -= ret_ptr[i] ;
            }
         if (xptr[i] < low_thresh[imarket]) { // Take a short position?
            if (ret_ptr[i] < 0.0)
               short_win -= ret_ptr[i] ;
            else
               short_lose += ret_ptr[i] ;
            }
         } // For training set
      criters[imarket] = (long_win + short_win) / (long_lose + short_lose) ;
      } // For all competitors

   qsortdsi ( 0 , n_markets-1 , criters , sort_key ) ;

   printf ( "\n\nCompetitor performance ranked best to worst, with superiority 'p-values'" ) ;
   printf ( "\nProfit factors are optimistically biased because the threshold was optimized" ) ;
   printf ( "\n\n         Market   Profit factor   p-value" ) ;

   for (imarket=n_markets-1 ; imarket>=0 ; imarket--) {  // Best to worst
      k = sort_key[imarket] ;
      printf ( "\n%15s %12.3lf %12.3lf", market_names+k*MAX_NAME_LENGTH, criters[imarket], counts[imarket] ) ;
      }

FINISH:

   if (fp != NULL)
      fclose ( fp ) ;

   printf ( "\n\nPress any key..." ) ;
   _getch () ;  // Wait for user to press a key

   for (i=0 ; i<n_markets ; i++) {
      if (market_date != NULL  &&  market_date[i] != NULL)
         free ( market_date[i] ) ;
      if (market_open != NULL  &&  market_close[i] != NULL)
         free ( market_open[i] ) ;
      if (market_high != NULL  &&  market_close[i] != NULL)
         free ( market_high[i] ) ;
      if (market_low != NULL  &&  market_close[i] != NULL)
         free ( market_low[i] ) ;
      if (market_close != NULL  &&  market_close[i] != NULL)
         free ( market_close[i] ) ;
      if (market_volume != NULL  &&  market_close[i] != NULL)
         free ( market_volume[i] ) ;
      }

   if (market_names != NULL)
      free ( market_names ) ;
   if (market_date != NULL)
      free ( market_date ) ;
   if (market_n != NULL)
      free ( market_n ) ;
   if (market_index != NULL)
      free ( market_index ) ;
   if (market_open != NULL)
      free ( market_open ) ;
   if (market_high != NULL)
      free ( market_high ) ;
   if (market_low != NULL)
      free ( market_low ) ;
   if (market_close != NULL)
      free ( market_close ) ;
   if (market_volume != NULL)
      free ( market_volume ) ;
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
