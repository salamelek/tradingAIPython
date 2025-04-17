/******************************************************************************/
/*                                                                            */
/* IND_FAM - Indicator analysis with permutation tests                        */
/*    This is the new version that controls familywise error rate.            */
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

#include "CONST.H"
#include "FUNCDEFS.H"

#define MAX_LINE 128  /* Max length of a control file line; extremely conservative */

/*
--------------------------------------------------------------------------------

   Local subroutine finds long and short thresholds that maximize profit factor

--------------------------------------------------------------------------------
*/

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
      } // For all trial thresholds

   *high_thresh = work_signal[best_high_index] ;
   *low_thresh = work_signal[best_low_index] ;
   *pf_high = best_high_pf ;
   *pf_low = best_low_pf ;
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
   int i, j, k, ivar, ret_val, irep, nreps, nvars, ncases, n_indicators, target_index, *ind_index ;
   int *count, *sort_indices ;
   double alpha, floor, *data, diff, mean, stddev, *target_work, *ind_work, *dwork1, *dwork2, rho, dummy ;
   double *original_crits, long_thresh, short_thresh, long_pf, short_pf, best_pf, best_crit, pval, prior ;
   char *lptr, data_filename[4096], control_filename[4096], **names, line[MAX_LINE] ;
   FILE *fp, *fp_log ;

/*
   Process command line parameters
*/

   if (argc != 6) {
      printf ( "\nUsage: IND_FAM  floor  alpha  nreps  data_filename  control_filename" ) ;
      printf ( "\n  floor - Minimum fraction (0-0.5) of cases that trade" ) ;
      printf ( "\n  alpha - Familywise alpha level (0-1, generally small)" ) ;
      printf ( "\n  nreps - Number of MCPT replications (hundreds or thousands)" ) ;
      printf ( "\n  data_filename - Name of database file" ) ;
      printf ( "\n  control_filename - Name of file that specifies target and indicators" ) ;
      exit ( 1 ) ;
      }

   floor = atof ( argv[1] ) ;
   alpha = atof ( argv[2] ) ;
   nreps = atoi ( argv[3] ) ;
   strcpy_s ( data_filename , argv[4] ) ;
   strcpy_s ( control_filename , argv[5] ) ;


/*
   These initializations let us do a clean exit.
   Open the log file for writing.
*/

   nvars = 0 ;
   ind_index = NULL ;
   original_crits = NULL ;
   dwork1 = NULL ;
   dwork2 = NULL ;
   target_work = NULL ;
   ind_work = NULL ;
   count = NULL ;
   sort_indices = NULL ;

   if (fopen_s ( &fp_log , "IND_FAM.LOG" , "wt" )) {
      printf ( "\nERROR... Unable to open log file IND_FAM.LOG for writing." ) ;
      goto FINISH ;
      }


/*
   Read dataset and allocate some memory.
   The readfile() routine allocates all required memory (names and data).
   This memory will be freed later by a call to free_data().
*/

   ret_val = readfile ( data_filename , &nvars , &names , &ncases , &data ) ;
   if (ret_val) {
      printf ( "\nAborting due to error reading file" ) ;
      goto FINISH ;
      }

   fprintf ( fp_log , "Read %d records of %d variables each from %s",
             ncases, nvars, data_filename ) ;

   ind_index = (int *) malloc ( nvars * sizeof(int) ) ;
   if (ind_index == NULL) {
      printf ( "\nERROR... Insufficient memory" ) ;
      goto FINISH ;
      }


/*
   Read the control file.
   The first line is the target, and subsequent lines are the indicators.
*/

   if (fopen_s ( &fp , control_filename , "rt" )) {
      printf ( "\nERROR... Unable to open control file %s", control_filename ) ;
      free_data ( nvars , names , data ) ;  // Frees memory allocated by readfile() (names and data)
      goto FINISH ;
      }

   for (n_indicators=-1 ; ; n_indicators++) {

      if ((fgets ( line , MAX_LINE , fp ) == NULL) || (strlen ( line ) < 2)) {
         if (ferror ( fp )  ||  n_indicators < 1) {
            fclose ( fp ) ;
            printf ( "\nERROR... Problem reading control file %s", control_filename ) ;
            free_data ( nvars , names , data ) ;  // Frees memory allocated by readfile() (names and data)
            goto FINISH ;
            }
         else
            break ;       // Normal end of file
         } // If error or normal end of file
      line[MAX_LINE-1] = 0 ;   // Insurance: terminate the line as a string

      // Terminate the string that names a variable
      lptr = line ;
      while (isalnum(*lptr) || *lptr == '_') // Variable names can be alphanumeric or underscore
         ++lptr ;
      *lptr = 0 ;       // Terminate this name
      _strupr_s ( line ) ;

      // Find this name in dataset variable name array
      for (ivar=0 ; ivar<nvars ; ivar++) {
         if (! strcmp ( line , names[ivar] ))
            break ;
         }
      if (ivar >= nvars) {
         printf ( "\nERROR... Variable '%s' is not in dataset", line ) ;
         free_data ( nvars , names , data ) ;  // Frees memory allocated by readfile() (names and data)
         goto FINISH ;
         }
      if (n_indicators == -1) {
         target_index = ivar ;
         fprintf ( fp_log , "\n\nTarget is %s\nIndicators...", line ) ;
         }
      else {
         ind_index[n_indicators] = ivar ;
         fprintf ( fp_log , "\n  %s", line ) ;
         }
         
      } // For reading all control file lines

   fclose ( fp ) ;


/*
   Compute and print means and standard deviations for user's edification.
   This formatting assumes that MAX_NAME_LENGTH is 15.
*/

   fprintf ( fp_log , "\n\nMeans and standard deviations...\n" ) ;
   fprintf ( fp_log , "       Variable       Mean        StdDev" ) ;

   for (ivar=-1 ; ivar<n_indicators ; ivar++) {

      if (ivar == -1)
         k = target_index ;
      else
         k = ind_index[ivar] ;

      mean = 0.0 ;
      for (i=0 ; i<ncases ; i++)
         mean += data[i*nvars+k] ;
      mean /= ncases ;

      stddev = 0.0 ;
      for (i=0 ; i<ncases ; i++) {
         diff = data[i*nvars+k] - mean ;
         stddev += diff * diff ;
         }
      stddev = sqrt ( stddev / ncases ) ;

      fprintf ( fp_log , "\n%15s %12.5lf %12.5lf", names[k], mean, stddev ) ;
      }

   fprintf ( fp_log , "\n" ) ;
 
/*
   Allocate all remaining memory that we will need.
*/

   k = (ncases > n_indicators) ? ncases : n_indicators ;

   original_crits = (double *) malloc ( n_indicators * sizeof(double) ) ;
   target_work = (double *) malloc ( ncases * sizeof(double) ) ;
   ind_work = (double *) malloc ( ncases * sizeof(double) ) ;
   count = (int *) malloc ( n_indicators * sizeof(int) ) ;
   sort_indices = (int *) malloc ( n_indicators * sizeof(int) ) ;
   dwork1 = (double *) malloc ( k * sizeof(double) ) ;
   dwork2 = (double *) malloc ( k * sizeof(double) ) ;
   if (original_crits == NULL  ||  count == NULL  || sort_indices == NULL
    || dwork1 == NULL  ||  dwork2 == NULL  ||  target_work == NULL  ||  ind_work == NULL) {
      printf ( "\nERROR... Insufficient memory" ) ;
      free_data ( nvars , names , data ) ;  // Frees memory allocated by readfile() (names and data)
      goto FINISH ;
      }


/*
   Compute and print long and short profit factors and thresholds
   The first step for each indicator is to compute the Spearman correlation
   between the indicator and the target.  If negative, flip the sign of the indicator.
   This lets us assume that large values of the indicator signal long positions
   and vice versa.
*/

   for (i=0 ; i<ncases ; i++)
      target_work[i] = data[i*nvars+target_index] ;

   fprintf ( fp_log , "\n\nOptimization results...\n" ) ;
   fprintf ( fp_log , "       Variable       Rho   Long thresh  Long pf   Short thresh  Short pf" ) ;

   for (ivar=0 ; ivar<n_indicators ; ivar++) {
      k = ind_index[ivar] ;   // Column in database of this competitor
      for (i=0 ; i<ncases ; i++)
         ind_work[i] = data[i*nvars+k] ;
      rho = spearman ( ncases , ind_work , target_work , dwork1 , dwork2 ) ;

      if (rho < 0.0) {  // Make sure that indicator and target are positively correlated
         for (i=0 ; i<ncases ; i++)       // Flip sign of this indicator
            ind_work[i] = -ind_work[i] ;
         }

      if (rho >= 0.0)  // For user, print sign of correlation (redundant but clear)
         fprintf ( fp_log , "\n%15s (+) %6.3lf", names[k], rho ) ;
      else
         fprintf ( fp_log , "\n%15s (-) %6.3lf", names[k], rho ) ;

      opt_thresh ( ncases , (int) (floor * ncases + 0.5) , 0 , ind_work , target_work ,
                   &dummy , &long_thresh , &long_pf ,
                   &short_thresh , &short_pf , dwork1 , dwork2 ) ;
      fprintf ( fp_log , " %11.4lf %8.3lf  %12.4lf %10.3lf",
                long_thresh , long_pf , short_thresh , short_pf ) ;
      original_crits[ivar] = (long_pf > short_pf) ? long_pf : short_pf ;
      }

   fprintf ( fp_log , "\n" ) ;

/*
   Later we need them sorted.  Save the sort indices.
   But we must not disturb the original values, so use a work vector.
*/

   for (i=0 ; i<n_indicators ; i++) {
      sort_indices[i] = i ;
      dwork1[i] = original_crits[i] ;
      count[i] = -999999 ;   // Not needed; here to aid debugging
      }
   qsortdsi ( 0 , n_indicators-1 , dwork1 , sort_indices ) ; // Sort ascending, moving indices


/*
-----------------------------------------------------------------------------------------------
   All of the preliminary work is done.
   We have in ind_index the indices into the complete database of all indicator candidates,
   such that ind_index[i] is the column in 'data' of the i'th indicator.
   Similarly, target_index is the column in 'data' of the returns target,
   typically next-day return, possibly normalized by a measure of volatility such as ATR.
   It is VITAL that these targets have negligible serial correlation!
   Thus, multiple-day-ahead returns CANNOT be used.  That would completely invalidate results.

   The original (unpermuted) criteria (best of long/short pf) are in original_crits.
   Their indices, sorted from worst to best, are in sort_indices.

   The 'traditional form of the algorithm performs nreps replications, with the first
   being unpermuted.  But we did that above, so we can jump right into the permutations.

   for ivar from 0 through n-1 {We just did this, above}
      compute and print criterion[ivar]

   For i from 0 through n-1
      sort_indices[i] = i ;
      original[i] = relationship of competitor i with Y
      work[i] = original[i] ;

   Sort work ascending, moving sort_indices simultaneously

   Step 1 of 2: do the random replications and count right tail

   For i from 0 through n-1
      count[i] = 1 ;  Counts right-tail probability

   For irep from 1 through m
      Shuffle Y

      max_f = number smaller than smallest possible relationship
      For i from 0 through n-1    Work from worst to best
         this_i = sort_indices[i]
         this_f = relationship between this_i and Y
         if (this_f > max_f) Keep track of maximum
            max_f = this_f
   
         If (max_f >= original[this_i])
            ++count[this_i] ;  Count right-tail probability
         } // For irep

   Step 2 of 2: Do the stepwise inclusion

      For i from n-1 through 0    Work from best to worst
         this_i = sort_indices[i] Index of best remaining competitor
         If count[this_i] / (m+1) <= alpha
            Accept this competitor
         Else
            Break out of step loop; we are done

   Note that in the code below, nreps corresponds to m+1 in the algorithm above,
   because nreps includes the original, unpermuted run.

-----------------------------------------------------------------------------------------------
*/


   for (ivar=0 ; ivar<n_indicators ; ivar++)
      count[ivar] = 1 ;

/*
   Step 1 of 2: Do the replications
*/

   for (irep=0 ; irep<nreps-1 ; irep++) {

      // Shuffle target
      i = ncases ;     // Number remaining to be shuffled
      while (i > 1) {  // While at least 2 left to shuffle
         j = (int) (unifrand() * i) ;
         if (j >= i)  // Cheap insurance against disaster if unifrand() returns 1.0
            j = i - 1 ;
         dummy = target_work[--i] ;
         target_work[i] = target_work[j] ;
         target_work[j] = dummy ;
         }

      // This loop processes competitors in order from poorest to best
      best_crit = -1.e60 ;
      for (ivar=0 ; ivar<n_indicators ; ivar++) {
         k = ind_index[sort_indices[ivar]] ;     // Column of ivar'th poorest indicator in database
         for (i=0 ; i<ncases ; i++)
            ind_work[i] = data[i*nvars+k] ;  // Fetch it from database
         rho = spearman ( ncases , ind_work , target_work , dwork1 , dwork2 ) ;
         if (rho < 0.0) {  // Make sure that indicator and target are positively correlated
            for (i=0 ; i<ncases ; i++)       // Flip sign of this indicator
               ind_work[i] = -ind_work[i] ;
            }
         opt_thresh ( ncases , (int) (floor * ncases + 0.5) , 0 , ind_work , target_work ,
                      &dummy , &dummy , &long_pf , &dummy , &short_pf , dwork1 , dwork2 ) ;
         best_pf = (long_pf > short_pf) ? long_pf : short_pf ;  // This is our performance criterion

         if (best_pf > best_crit)
            best_crit = best_pf ;

         k = sort_indices[ivar] ;     // Index of ivar'th poorest indicator
         if (best_crit >= original_crits[k])
            ++count[k] ;
         } // For ivar, poorest to best
      } // For irep

/*
   Step 2 of 2: Stepwise rejection of null hypothesis, with printing
   Then free memory allocated in readfile().
*/

   prior = 0.0 ;
   fprintf ( fp_log , "\n\nBest of long/short profit factors and p-values..." ) ;
   for (i=n_indicators-1 ; i>=0 ; i--) {
      k = sort_indices[i] ;
      pval = (double) count[k] / (double) nreps ;
      if (pval < prior)
         pval = prior ;
      prior = pval ;
      if (pval <= alpha) {
         fprintf ( fp_log , "\n%15s %10.3lf %12.3lf",
                   names[ind_index[k]], original_crits[k], pval ) ;
         }
      else {
         fprintf ( fp_log, "\n\nBest remaining p-value=%.4lf while user's alpha=%.4lf, so quitting", pval, alpha ) ;
         break ;
         }
      } // For all competitors, working from best to worst


   free_data ( nvars , names , data ) ;  // Frees memory allocated by readfile() (names and data)

FINISH:

   if (fp_log != NULL)
      fclose ( fp_log ) ;

   if (ind_index != NULL)
      free ( ind_index ) ;

   if (target_work != NULL)
      free ( target_work ) ;

   if (ind_work != NULL)
      free ( ind_work ) ;

   if (count != NULL)
      free ( count ) ;

   if (sort_indices != NULL)
      free ( sort_indices ) ;

   if (dwork1 != NULL)
      free ( dwork1 ) ;

   if (dwork2 != NULL)
      free ( dwork2 ) ;

   if (original_crits != NULL)
      free ( original_crits ) ;

   printf ( "\n\nPress any key..." ) ;
   _getch () ;  // Wait for user to press a key

   exit ( 0 ) ;
}
