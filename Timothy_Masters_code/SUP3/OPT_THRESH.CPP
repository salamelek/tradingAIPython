/******************************************************************************/
/*                                                                            */
/*  OPT_THRESH - Find optimal threshold to maximize long/short profit factors */
/*                                                                            */
/******************************************************************************/

#include "funcdefs.h"

void opt_thresh (
   int n ,                // Number of indicator/return pairs
   int min_kept ,         // Must keep (trade) at least this many cases
   int flip_sign ,        // If nonzero, flip sign of indicator
   double *signal_vals ,  // Indicators
   double *returns ,      // Associated returns
   double *pf_all ,       // Profit factor of entire dataset
   double *high_thresh ,  // Upper threshold, for long trades
   double *pf_high ,      // Profit factor >= threshold
   double *low_thresh ,   // Lower threshold, for short trades
   double *pf_low ,       // Profit factor < threshold
   double *work_signal ,  // Work area n long
   double *work_return    // Work area n long
   )
{
   int i, best_low_index, best_high_index ;
   double win_above, win_below, lose_above, lose_below, best_high_pf, best_low_pf ;

   if (min_kept < 1)
      min_kept = 1 ;

/*
   Copy signals and returns to work areas and sort signal,
   simultaneously moving returns.
*/

   for (i=0 ; i<n ; i++) {
      work_signal[i] = flip_sign ? (-signal_vals[i]) : signal_vals[i] ;
      work_return[i] = returns[i] ;
      }

   qsortds ( 0 , n-1 , work_signal , work_return ) ;

/*
   Initialize by computing 'above' win/loss for complete set
*/

   win_above = win_below = lose_above = lose_below = 0.0 ;

   for (i=0 ; i<n ; i++) {
      if (work_return[i] > 0.0)  // We are long so this is a win
         win_above += work_return[i] ;
      else
         lose_above -= work_return[i] ;
      }

   *pf_all = best_high_pf = win_above / (lose_above + 1.e-30) ;
   best_high_index = 0 ;   // Complete set is implied by threshold at smallest value

/*
   Search all possible thresholds
*/

   best_low_pf = -1.0 ;
   best_low_index = n-1 ;     // This should never be necessary
                              // but it avoids a crash if pathological condition

   for (i=0 ; i<n-1 ; i++) {  // Indicator[i+1] is a candidate threshold

      // Remove this case from high (long) set
      if (work_return[i] > 0.0)
         win_above -= work_return[i] ;
      else
         lose_above += work_return[i] ;

      // Add this case to low (short) set
      if (work_return[i] > 0.0)
         lose_below += work_return[i] ;
      else
         win_below -= work_return[i] ;

      // This trial threshold at i+1 is legitimate only if it is a new value.
      // Subsequent values in a tied block are not new, unique thresholds.
      if (work_signal[i+1] == work_signal[i])
         continue ;

      // If we have enough in the 'above' set, see about updating the best high threshold
      if (n-i-1 >= min_kept) {
         if (win_above / (lose_above + 1.e-30) > best_high_pf) {
            best_high_pf = win_above / (lose_above + 1.e-30) ;
            best_high_index = i+1 ;
            }
         }

      // If we have enough in the 'below' set, see about updating the best low threshold
      if (i+1 >= min_kept) {
         if (win_below / (lose_below + 1.e-30) > best_low_pf) {
            best_low_pf = win_below / (lose_below + 1.e-30) ;
            best_low_index = i+1 ;
            }
         }
#if 0
      char msg[256] ;
      sprintf_s ( msg , "Thr=%7.3lf  WA=%6.2lf  LA=%6.2lf  WB=%6.2lf  LB=%6.2lf  PFA=%6.2lf  PFB=%6.2lf",
                  work_signal[i+1], win_above, lose_above, win_below, lose_below, win_above / (lose_above + 1.e-30), win_below / (lose_below + 1.e-30) ) ;
      MEMTEXT ( msg ) ;
#endif
      } // For all trial thresholds

   *high_thresh = work_signal[best_high_index] ;
   *low_thresh = work_signal[best_low_index] ;
   *pf_high = best_high_pf ;
   *pf_low = best_low_pf ;
}
