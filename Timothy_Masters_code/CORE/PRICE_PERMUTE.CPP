/***********************************************************************************/
/*                                                                                 */
/* Permutation routines (assumes prices in 'data' are actually log prices)         */
/*                                                                                 */
/*   To use these routines, first invoke the constructor.                          */
/*   Then call the do_permute routine as many times as desired.                    */
/*                                                                                 */
/*   We have nc cases from index 0 through nc-1.                                   */
/*     Case 0 is the 'basis' case and remains unchanged (single prices or bars)    */
/*     Case 1 is the first case that will change.                                  */
/*     The last case examined is at nc-1, but it, too, will remain unchanged       */
/*     for single-price data.  For bars, the close of the last case is unchanged,  */
/*     although the open, high, and low will generally change.                     */
/*     Thus, the shuffled array starts and ends at their original prices.          */
/*     Only the interior elements change.                                          */
/*                                                                                 */
/*   We do not have to start with Case 0 being the basis case.                     */
/*   The 'index' parameter is the index of our desired basis case, and prior       */
/*   cases (prices or bars) are ignored.                                           */
/*                                                                                 */
/*   If a dataset is permuted in separate sections, the sections must not overlap. */
/*   The 'basis' case is included in the region that must not overlap.             */
/*                                                                                 */
/*   For example, we could permute by calling this with 'index' pointing to        */
/*   the first price (0) in the market history, and nc=5.                          */
/*   Cases 1 through 3 would then change.                                          */
/*   (The first (0) and last (4) cases remain the same.)                           */
/*                                                                                 */
/*   A subsequent permute must then begin at history index 4 or more.              */
/*   Case 4 is not changed by either permute operation.                            */
/*                                                                                 */
/*   Looking at it another way, it is vital that a price change in one section     */
/*   never find itself permuted into another section.  In the example just cited,  */
/*   the first section permutes the following changes: 0-1, 1-2, 2-3, 3-4.         */
/*   The second section permutes the following changes: 4-5, 5-6, ...              */
/*   The changes in each section are restricted to that section.                   */
/*   By isolating changes to within each section, we guarantee that some wild      */
/*   change, which would appear only once in the unpermuted walkforward, will      */
/*   also appear only once in the permuted walkforwards.                           */
/*                                                                                 */
/*   In a walkforward situation, each section should begin on the LAST price/bar   */
/*   of the prior section.  When this is done, we are independently testing the    */
/*   CHANGES in the current section.  Thus, nc will be one more than the number    */
/*   of prices/bars in the walkforward fold, since nc includes that extra price at */
/*   the beginning.  This is even true for the first walkforward fold, where       */
/*   we begin at the last training-set price.  In this way we begin permutation    */
/*   with the first CHANGE in the out-of-sample period.                            */
/*                                                                                 */
/*   Note that because permutation preserves the net change from the first price   */
/*   to the last in the section, starting each new section at the last price       */
/*   of the prior section preserves the net change across the entire walkforward.  */
/*                                                                                 */
/*   About prices in these parameter lists...                                      */
/*   Each market has its own price array independently allocated.                  */
/*   Thus, price[imarket] is a pointer to an array of prices.                      */
/*                                                                                 */
/***********************************************************************************/

#include <stdlib.h>

extern double unifrand () ;

class PricePermute {

public:
   PricePermute (
      int np ,           // Number of prices
      int nmkt ,         // Number of markets
      int index ,        // Index of basis price, one prior to first permuted price
      double **prices    // Input of nmkt by nc price matrix; Output permuted
      ) ;
   ~PricePermute () ;
   void do_permute () ;

private:
   int ok ;               // Was memory allocation successful?
   int n_prices ;         // Number of prices
   int n_markets ;        // Number of markets
   int permute_index ;    // Index of first permuted price
   double *basis_prices ; // Work area for saving basis prices
   double **prices_ptr ;  // Saves pointer to user's price input/output
   double **changes ;     // Work area preserves changes
} ;

PricePermute::PricePermute (
      int np ,           // Number of prices
      int nmkt ,         // Number of markets
      int index ,        // Index of basis price, one prior to first permuted price
      double **prices    // Input of nmkt by nc price matrix; Output permuted
      )
{
   int i, iprice, imarket ;

   n_prices = np ;             // Copy parameters to private areas
   n_markets = nmkt ;
   prices_ptr = prices ;
   permute_index = index + 1 ; // Point to first permuted price

   ok = 1 ;           // Start out optimistic

/*
   Allocate memory
*/

   basis_prices = (double *) malloc ( nmkt * sizeof(double) ) ;
   changes = (double **) malloc ( nmkt * sizeof(double *) ) ;
   if (basis_prices == NULL  ||  changes == NULL) {
      if (basis_prices != NULL) {
         free ( basis_prices ) ;
         basis_prices = NULL ;
         }
      if (changes != NULL) {
         free ( changes ) ;
         changes = NULL ;
         }
      ok = 0 ;
      return ;
      }

   for (imarket=0 ; imarket<n_markets ; imarket++) {
      changes[imarket] = (double *) malloc ( n_prices * sizeof(double) ) ;
      if (changes[imarket] == NULL) {
         for (i=0 ; i<imarket ; i++)
            free ( changes[i] ) ;
         free ( changes ) ;
         free ( basis_prices ) ;
         ok = 0 ;
         return ;
         }
      }

/*
   Compute and save the changes.
   Also save the basis price in case the user modifies (ie exponentiates) the permuted prices
*/

   for (imarket=0 ; imarket<nmkt ; imarket++) {
      basis_prices[imarket] = prices[imarket][index] ;
      for (iprice=permute_index ; iprice<n_prices ; iprice++)
         changes[imarket][iprice] = prices[imarket][iprice] - prices[imarket][iprice-1] ;
      }
}

PricePermute::~PricePermute ()
{
   int imarket ;

   if (basis_prices != NULL)
      free ( basis_prices ) ;

   if (changes != NULL) {
      for (imarket=0 ; imarket<n_markets ; imarket++)
         free ( changes[imarket] ) ;
      free ( changes ) ;
      }
}

void PricePermute::do_permute ()
{
   int i, j, iprice, imarket ;
   double dtemp ;

   // Shuffle the changes, permuting each market the same to preserve correlations
   // We do not include the first case in the shuffling, as it is undefined.

   i = n_prices - permute_index ; // Number remaining to be shuffled
   while (i > 1) {  // While at least 2 left to shuffle
      j = (int) (unifrand() * i) ;
      if (j >= i)   // Should never happen, but be safe
         j = i - 1 ;
      --i ;
      for (imarket=0 ; imarket<n_markets ; imarket++) {
         dtemp = changes[imarket][i+permute_index] ;
         changes[imarket][i+permute_index] = changes[imarket][j+permute_index] ;
         changes[imarket][j+permute_index] = dtemp ;
         }
      } // Shuffle the changes

   // Now rebuild the prices, using the shuffled changes

   for (imarket=0 ; imarket<n_markets ; imarket++) {
      prices_ptr[imarket][permute_index-1] = basis_prices[imarket] ; // Recover basis price in case user changed it
      for (iprice=permute_index ; iprice<n_prices ; iprice++)
         prices_ptr[imarket][iprice] = prices_ptr[imarket][iprice-1] + changes[imarket][iprice] ;
      }
}
   
//--------------------------------> Bars <------------------------------------

class BarPermute {

public:
   BarPermute (
      int np ,             // Number of prices
      int nmkt ,           // Number of markets
      int index ,          // Index of basis bar, one prior to first permuted price
      double **open ,      // Input of nmkt by nc opens
      double **high ,      // Input of nmkt by nc highs
      double **low ,       // Input of nmkt by nc lows
      double **close       // Input of nmkt by nc closes
      ) ;
   ~BarPermute () ;
   void do_permute () ;

private:
   int ok ;               // Was memory allocation successful?
   int n_prices ;         // Number of prices (bars)
   int n_markets ;        // Number of markets
   int permute_index ;    // Index of first permuted price
   double *basis_open ;   // Work area for saving basis prices (bars)
   double *basis_high ;   // Ditto
   double *basis_low ;    // Ditto
   double *basis_close ;  // Ditto
   double **open_ptr ;    // Saves pointer to user's price input/output
   double **high_ptr ;    // Ditto
   double **low_ptr ;     // Ditto
   double **close_ptr ;   // Ditto
   double **rel_open ;    // Work area of n_markets arrays np long
   double **rel_high ;    // Ditto
   double **rel_low ;     // Ditto
   double **rel_close ;   // Ditto
} ;

BarPermute::BarPermute (
      int np ,           // Number of prices
      int nmkt ,         // Number of markets
      int index ,        // Index of basis price, one prior to first permuted price
      double **open ,    // Input of nmkt by nc opens
      double **high ,    // Input of nmkt by nc highs
      double **low ,     // Input of nmkt by nc lows
      double **close     // Input of nmkt by nc closes
      )
{
   int i, iprice, imarket ;

   n_prices = np ;             // Copy parameters to private areas
   n_markets = nmkt ;
   open_ptr = open ;
   high_ptr = high ;
   low_ptr = low ;
   close_ptr = close ;
   permute_index = index + 1 ; // Point to first permuted price

   ok = 1 ;           // Start out optimistic

/*
   Allocate memory
*/

   basis_open = (double *) malloc ( 4 * n_markets * sizeof(double) ) ;
   rel_open = (double **) malloc ( 4 * n_markets * sizeof(double *) ) ;
   if (basis_open == NULL  ||  rel_open == NULL) {
      if (basis_open != NULL) {
         free ( basis_open ) ;
         basis_open = NULL ;
         }
      if (rel_open != NULL) {
         free ( rel_open ) ;
         rel_open = NULL ;
         }
      ok = 0 ;
      return ;
      }

   basis_high = basis_open + n_markets ;
   basis_low = basis_high + n_markets ;
   basis_close = basis_low + n_markets ;

   rel_high = rel_open + n_markets ;
   rel_low = rel_high + n_markets ;
   rel_close = rel_low + n_markets ;

   for (imarket=0 ; imarket<n_markets ; imarket++) {
      rel_open[imarket] = (double *) malloc ( 4 * n_prices * sizeof(double) ) ;
      if (rel_open[imarket] == NULL) {
         for (i=0 ; i<imarket ; i++)
            free ( rel_open[i] ) ;
         free ( rel_open ) ;
         free ( basis_open ) ;
         ok = 0 ;
         return ;
         }
      rel_high[imarket] = rel_open[imarket] + n_prices ;
      rel_low[imarket] = rel_high[imarket] + n_prices ;
      rel_close[imarket] = rel_low[imarket] + n_prices ;
      }

/*
   Compute and save the changes.
   Also save the basis price in case the user modifies (ie exponentiates) the permuted prices
*/

   for (imarket=0 ; imarket<n_markets ; imarket++) {
      basis_open[imarket] = open[imarket][index] ;
      basis_high[imarket] = high[imarket][index] ;
      basis_low[imarket] = low[imarket][index] ;
      basis_close[imarket] = close[imarket][index] ;
      for (iprice=permute_index ; iprice<n_prices ; iprice++) {
         rel_open[imarket][iprice] = open[imarket][iprice] - close[imarket][iprice-1] ;
         rel_high[imarket][iprice] = high[imarket][iprice] - open[imarket][iprice] ;
         rel_low[imarket][iprice] = low[imarket][iprice] - open[imarket][iprice] ;
         rel_close[imarket][iprice] = close[imarket][iprice] - open[imarket][iprice] ;
         }
      }
}

BarPermute::~BarPermute ()
{
   int imarket ;

   if (basis_open != NULL)
      free ( basis_open ) ;

   if (rel_open != NULL) {
      for (imarket=0 ; imarket<n_markets ; imarket++)
         free ( rel_open[imarket] ) ;
      free ( rel_open ) ;
      }
}

void BarPermute::do_permute ()
{
   int i, j, iprice, imarket ;
   double dtemp ;

   // Shuffle the changes, permuting each market the same to preserve correlations
   // We do not include the first case in the shuffling, as it is the starting price, not a change

   i = n_prices - permute_index ; // Number remaining to be shuffled
   while (i > 1) {  // While at least 2 left to shuffle
      j = (int) (unifrand() * i) ;
      if (j >= i)   // Should never happen, but be safe
         j = i - 1 ;
      --i ;
      for (imarket=0 ; imarket<n_markets ; imarket++) {
         dtemp = rel_high[imarket][i+permute_index] ;
         rel_high[imarket][i+permute_index] = rel_high[imarket][j+permute_index] ;
         rel_high[imarket][j+permute_index] = dtemp ;
         dtemp = rel_low[imarket][i+permute_index] ;
         rel_low[imarket][i+permute_index] = rel_low[imarket][j+permute_index] ;
         rel_low[imarket][j+permute_index] = dtemp ;
         dtemp = rel_close[imarket][i+permute_index] ;
         rel_close[imarket][i+permute_index] = rel_close[imarket][j+permute_index] ;
         rel_close[imarket][j+permute_index] = dtemp ;
         }
      } // Shuffle the changes

   // Separately shuffle the close-to-open changes,
   // permuting each market the same to preserve correlations.

   i = n_prices - permute_index ; // Number remaining to be shuffled
   while (i > 1) {  // While at least 2 left to shuffle
      j = (int) (unifrand() * i) ;
      if (j >= i)   // Should never happen, but be safe
         j = i - 1 ;
      --i ;
      for (imarket=0 ; imarket<n_markets ; imarket++) {
         dtemp = rel_open[imarket][i+permute_index] ;
         rel_open[imarket][i+permute_index] = rel_open[imarket][j+permute_index] ;
         rel_open[imarket][j+permute_index] = dtemp ;
         }
      } // Shuffle the close-to-open changes

   // Now rebuild the prices, using the shuffled changes

   for (imarket=0 ; imarket<n_markets ; imarket++) {
      open_ptr[imarket][permute_index-1] = basis_open[imarket] ; // Recover basis price in case user changed it
      high_ptr[imarket][permute_index-1] = basis_high[imarket] ;
      low_ptr[imarket][permute_index-1] = basis_low[imarket] ;
      close_ptr[imarket][permute_index-1] = basis_close[imarket] ;
      for (iprice=permute_index ; iprice<n_prices ; iprice++) {
         open_ptr[imarket][iprice] = close_ptr[imarket][iprice-1] + rel_open[imarket][iprice] ;
         high_ptr[imarket][iprice] = open_ptr[imarket][iprice] + rel_high[imarket][iprice] ;
         low_ptr[imarket][iprice] = open_ptr[imarket][iprice] + rel_low[imarket][iprice] ;
         close_ptr[imarket][iprice] = open_ptr[imarket][iprice] + rel_close[imarket][iprice] ;
         }
      }
}