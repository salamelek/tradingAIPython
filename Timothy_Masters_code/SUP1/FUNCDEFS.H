void comp_ind (
   int n ,         // Number of market price bars
   double *open ,  // They are here
   double *high ,
   double *low ,
   double *close ,
   int iparam ,    // Competition parameter, n to smooth here
   double *ind     // The indicator values are output here
   ) ;

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
   ) ;

double spearman (  // Returns rho in range -1 to 1
   int n ,         // Input: Number of cases
   double *var1 ,  // Input: One variable
   double *var2 ,  // Input: Other variable
   double *x ,     // Work vector n long
   double *y       // Work vector n long
   ) ;

void qsortd ( int first , int last , double *data ) ;
void qsortds ( int first , int last , double *data , double *slave ) ;
void qsortdsi ( int first , int last , double *data , int *slave ) ;
double unifrand () ;
