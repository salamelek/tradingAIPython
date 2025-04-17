extern void boot_conf_BCa (
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
   ) ;
extern double inverse_normal_cdf ( double p ) ;
extern double normal_cdf ( double z ) ;
extern void qsortd ( int first , int last , double *data ) ;
extern double unifrand () ;
