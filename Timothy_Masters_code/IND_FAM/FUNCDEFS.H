extern void free_data ( int nvars , char **names , double *data ) ;
extern void qsortd ( int first , int last , double *data ) ;
extern void qsortds ( int first , int last , double *data , double *slave ) ;
extern void qsortdsi ( int first , int last , double *data , int *slave ) ;
extern void qsorti ( int first , int last , int *data ) ;
extern void qsortisd ( int first , int last , int *data , double *slave ) ;
extern int readfile ( char *name , int *nvars , char ***names , int *ncases , double **data ) ;
double spearman ( int n , double *var1 , double *var2 , double *x , double *y ) ;
extern double unifrand () ;
