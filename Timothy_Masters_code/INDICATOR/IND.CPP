/******************************************************************************/
/*                                                                            */
/* IND - Indicator analysis with permutation tests                            */
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
   int *long_solo_count, *short_solo_count, *best_solo_count, *iwork ;
   int *long_unbiased_count, *short_unbiased_count, *best_unbiased_count ;
   double floor, *data, diff, mean, stddev, *target_work, *ind_work, *dwork1, *dwork2, rho, dummy ;
   double *original_long_pf, *original_short_pf, *original_best_pf, *long_thresh, *short_thresh ;
   double long_pf, short_pf, best_pf, max_long_pf, max_short_pf, max_best_pf ;
   char *lptr, data_filename[4096], control_filename[4096], **names, line[MAX_LINE] ;
   FILE *fp, *fp_log ;

/*
   Process command line parameters
*/

#if 1
   if (argc != 5) {
      printf ( "\nUsage: IND  floor  nreps  data_filename  control_filename" ) ;
      printf ( "\n  floor - Minimum fraction (0-0.5) of cases that trade" ) ;
      printf ( "\n  nreps - Number of MCPT replications (hundreds or thousands)" ) ;
      printf ( "\n  data_filename - Name of database file" ) ;
      printf ( "\n  control_filename - Name of file that specifies target and indicators" ) ;
      exit ( 1 ) ;
      }

   floor = atof ( argv[1] ) ;
   nreps = atoi ( argv[2] ) ;
   strcpy_s ( data_filename , argv[3] ) ;
   strcpy_s ( control_filename , argv[4] ) ;
#else
   floor = 0.1 ;
   nreps = 100 ;
   strcpy_s ( data_filename , "D:\\Varscreen\\Test\\VARS.TXT" ) ;
   strcpy_s ( data_filename , "D:\\Varscreen\\Test\\SYNTH.TXT" ) ;
   strcpy_s ( control_filename , "IND_CONTROL.TXT" ) ;
   strcpy_s ( control_filename , "IND_CONTROL2.TXT" ) ;
#endif


/*
   These initializations let us do a clean exit.
   Open the log file for writing.
*/

   nvars = 0 ;
   ind_index = NULL ;
   original_long_pf = NULL ;
   original_short_pf = NULL ;
   original_best_pf = NULL ;
   long_thresh = NULL ;
   short_thresh = NULL ;
   dwork1 = NULL ;
   dwork2 = NULL ;
   target_work = NULL ;
   ind_work = NULL ;
   long_solo_count = NULL ;
   short_solo_count = NULL ;
   best_solo_count = NULL ;
   long_unbiased_count = NULL ;
   short_unbiased_count = NULL ;
   best_unbiased_count = NULL ;
   iwork = NULL ;

   if (fopen_s ( &fp_log , "IND.LOG" , "wt" )) {
      printf ( "\nERROR... Unable to open log file IND.LOG for writing." ) ;
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
      _strupr ( line ) ;

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

   original_long_pf = (double *) malloc ( n_indicators * sizeof(double) ) ;
   original_short_pf = (double *) malloc ( n_indicators * sizeof(double) ) ;
   original_best_pf = (double *) malloc ( n_indicators * sizeof(double) ) ;
   long_thresh = (double *) malloc ( n_indicators * sizeof(double) ) ;
   short_thresh = (double *) malloc ( n_indicators * sizeof(double) ) ;
   target_work = (double *) malloc ( ncases * sizeof(double) ) ;
   ind_work = (double *) malloc ( ncases * sizeof(double) ) ;
   long_solo_count = (int *) malloc ( n_indicators * sizeof(int) ) ;
   short_solo_count = (int *) malloc ( n_indicators * sizeof(int) ) ;
   best_solo_count = (int *) malloc ( n_indicators * sizeof(int) ) ;
   long_unbiased_count = (int *) malloc ( n_indicators * sizeof(int) ) ;
   short_unbiased_count = (int *) malloc ( n_indicators * sizeof(int) ) ;
   best_unbiased_count = (int *) malloc ( n_indicators * sizeof(int) ) ;
   iwork = (int *) malloc ( n_indicators * sizeof(int) ) ;
   dwork1 = (double *) malloc ( k * sizeof(double) ) ;
   dwork2 = (double *) malloc ( k * sizeof(double) ) ;
   if (original_long_pf == NULL  ||  original_short_pf == NULL  ||  original_best_pf == NULL
    || long_thresh == NULL  ||  short_thresh == NULL
    || long_solo_count == NULL  ||  short_solo_count == NULL  ||  best_solo_count == NULL
    || long_unbiased_count == NULL  ||  short_unbiased_count == NULL  ||  best_unbiased_count == NULL
    || iwork == NULL  ||  dwork1 == NULL  ||  dwork2 == NULL  ||  target_work == NULL  ||  ind_work == NULL) {
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
         for (i=0 ; i<ncases ; i++) {  // Flip sign of this indicator in database
            data[i*nvars+k] = -data[i*nvars+k] ;
            ind_work[i] = data[i*nvars+k] ;  // Get flipped version from database
            }
         }

      if (rho >= 0.0)  // For user, print sign of correlation (redundant but clear)
         fprintf ( fp_log , "\n%15s (+) %6.3lf", names[k], rho ) ;
      else
         fprintf ( fp_log , "\n%15s (-) %6.3lf", names[k], rho ) ;

      opt_thresh ( ncases , (int) (floor * ncases + 0.5) , 0 , ind_work , target_work ,
                   &dummy , &long_thresh[ivar] , &original_long_pf[ivar] ,
                   &short_thresh[ivar] , &original_short_pf[ivar] , dwork1 , dwork2 ) ;
      fprintf ( fp_log , " %11.4lf %8.3lf  %12.4lf %10.3lf",
                long_thresh[ivar] , original_long_pf[ivar] ,
                short_thresh[ivar] , original_short_pf[ivar] ) ;
      original_best_pf[ivar] = (original_long_pf[ivar] > original_short_pf[ivar]) ? original_long_pf[ivar] : original_short_pf[ivar] ;
      }

   fprintf ( fp_log , "\n" ) ;


/*
   All of the preliminary work is done.
   We have in ind_index the indices into the complete database of all indicator candidates,
   such that ind_index[i] is the column in 'data' of the i'th indicator.
   Similarly, target_index is the column in 'data' of the returns target,
   typically next-day return, possibly normalized by a measure of volatility such as ATR.
   It is VITAL that these targets have negligible serial correlation!
   Thus, multiple-day-ahead returns CANNOT be used.  That would completely invalidate results.

   The original (unpermuted) profit factors are in original_long_pf and original_short_pf.

   The 'traditional form of the algorithm performs nreps replications, with the first
   being unpermuted.  But we did that above, so now we jump right into the permutations.

   The algorithm is as follows:

   for ivar from 0 through n_indicators-1 {We just did this, above}
      compute and print criterion[ivar]

   for ivar from 0 through n_indicators-1
      solo_count[ivar] = unbiased_count[ivar] = 1

   for nreps-1 replications
      permute target
      for ivar from 0 through n_indicators-1
         compute criterion[ivar]  {This is the criterion for this permutation}
         if criterion[ivar] >= original_criterion[ivar]
            solo_count[ivar] = solo_count[ivar] + 1
      max_criterion = MAX (criterion[ivar], ivar=0,...,n_indicators-1)
      for ivar from 0 through n_indicators-1
         if max_criterion >= original_criterion[ivar]
            unbiased_count[ivar] = unbiased_count[ivar] + 1

   for ivar from 0 through n_indicators-1
      solo_pval[ivar] = solo_count[ivar] / nreps
      unbiased_pval[ivar] = unbiased_count[ivar] / nreps
*/

   for (ivar=0 ; ivar<n_indicators ; ivar++) {
      long_solo_count[ivar] = long_unbiased_count[ivar] = 1 ;
      short_solo_count[ivar] = short_unbiased_count[ivar] = 1 ;
      best_solo_count[ivar] = best_unbiased_count[ivar] = 1 ;
      }

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
      for (ivar=0 ; ivar<n_indicators ; ivar++) {
         k = ind_index[ivar] ;               // Column of this indicator in database
         for (i=0 ; i<ncases ; i++)
            ind_work[i] = data[i*nvars+k] ;  // Fetch it from database
         rho = spearman ( ncases , ind_work , target_work , dwork1 , dwork2 ) ;
         if (rho < 0.0) {  // Make sure that indicator and target are positively correlated
            for (i=0 ; i<ncases ; i++)       // Flip sign of this indicator
               ind_work[i] = -ind_work[i] ;
            }
         opt_thresh ( ncases , (int) (floor * ncases + 0.5) , 0 , ind_work , target_work ,
                      &dummy , &dummy , &long_pf , &dummy , &short_pf , dwork1 , dwork2 ) ;
         best_pf = (long_pf > short_pf) ? long_pf : short_pf ;
         if (long_pf >= original_long_pf[ivar])
            ++long_solo_count[ivar] ;
         if (short_pf >= original_short_pf[ivar])
            ++short_solo_count[ivar] ;
         if (best_pf >= original_best_pf[ivar])
            ++best_solo_count[ivar] ;
         if (ivar == 0  ||  long_pf > max_long_pf)  // Keep track of MAX for next step
            max_long_pf = long_pf ;
         if (ivar == 0  ||  short_pf > max_short_pf)
            max_short_pf = short_pf ;
         if (ivar == 0  ||  best_pf > max_best_pf)
            max_best_pf = best_pf ;
         }

      for (ivar=0 ; ivar<n_indicators ; ivar++) {
         if (max_long_pf >= original_long_pf[ivar])
            ++long_unbiased_count[ivar] ;
         if (max_short_pf >= original_short_pf[ivar])
            ++short_unbiased_count[ivar] ;
         if (max_best_pf >= original_best_pf[ivar])
            ++best_unbiased_count[ivar] ;
         }
      } // For irep


/*
   All of the hard work is done.
   Print results.
   Free memory allocated in readfile().
*/

   fprintf ( fp_log , "\n\nLong-only profit factors and p-values..." ) ;
   for (i=0 ; i<n_indicators ; i++)
      iwork[i] = i ;
   qsortdsi ( 0 , n_indicators-1 , original_long_pf , iwork ) ; // Sort ascending, moving indices
   fprintf ( fp_log , "\n\n       Variable  profit factor  solo pval  unbiased pval" ) ;
   for (i=0 ; i<n_indicators ; i++) {
      k = iwork[n_indicators-i-1] ;
      fprintf ( fp_log , "\n%15s %10.3lf %12.3lf %12.3lf",
                names[ind_index[k]], original_long_pf[n_indicators-i-1],
                (double) long_solo_count[k] / (double) nreps,
                (double) long_unbiased_count[k] / (double) nreps ) ;
      }

   fprintf ( fp_log , "\n\nShort-only profit factors and p-values..." ) ;
   for (i=0 ; i<n_indicators ; i++)
      iwork[i] = i ;
   qsortdsi ( 0 , n_indicators-1 , original_short_pf , iwork ) ; // Sort ascending, moving indices
   fprintf ( fp_log , "\n\n       Variable  profit factor  solo pval  unbiased pval" ) ;
   for (i=0 ; i<n_indicators ; i++) {
      k = iwork[n_indicators-i-1] ;
      fprintf ( fp_log , "\n%15s %10.3lf %12.3lf %12.3lf",
                names[ind_index[k]], original_short_pf[n_indicators-i-1],
                (double) short_solo_count[k] / (double) nreps,
                (double) short_unbiased_count[k] / (double) nreps ) ;
      }

   fprintf ( fp_log , "\n\nBest of long/short profit factors and p-values..." ) ;
   for (i=0 ; i<n_indicators ; i++)
      iwork[i] = i ;
   qsortdsi ( 0 , n_indicators-1 , original_best_pf , iwork ) ; // Sort ascending, moving indices
   fprintf ( fp_log , "\n\n       Variable  profit factor  solo pval  unbiased pval" ) ;
   for (i=0 ; i<n_indicators ; i++) {
      k = iwork[n_indicators-i-1] ;
      fprintf ( fp_log , "\n%15s %10.3lf %12.3lf %12.3lf",
                names[ind_index[k]], original_best_pf[n_indicators-i-1],
                (double) best_solo_count[k] / (double) nreps,
                (double) best_unbiased_count[k] / (double) nreps ) ;
      }

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

   if (long_solo_count != NULL)
      free ( long_solo_count ) ;

   if (short_solo_count != NULL)
      free ( short_solo_count ) ;

   if (best_solo_count != NULL)
      free ( best_solo_count ) ;

   if (long_unbiased_count != NULL)
      free ( long_unbiased_count ) ;

   if (short_unbiased_count != NULL)
      free ( short_unbiased_count ) ;

   if (best_unbiased_count != NULL)
      free ( best_unbiased_count ) ;

   if (iwork != NULL)
      free ( iwork ) ;

   if (dwork1 != NULL)
      free ( dwork1 ) ;

   if (dwork2 != NULL)
      free ( dwork2 ) ;

   if (original_long_pf != NULL)
      free ( original_long_pf ) ;

   if (original_short_pf != NULL)
      free ( original_short_pf ) ;

   if (original_best_pf != NULL)
      free ( original_best_pf ) ;

   if (long_thresh != NULL)
      free ( long_thresh ) ;

   if (short_thresh != NULL)
      free ( short_thresh ) ;

   printf ( "\n\nPress any key..." ) ;
   _getch () ;  // Wait for user to press a key

   exit ( 0 ) ;
}
