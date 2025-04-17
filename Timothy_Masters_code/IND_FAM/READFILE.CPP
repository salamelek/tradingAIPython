/******************************************************************************/
/*                                                                            */
/*  READFILE - Read the dataset from a text file                              */
/*                                                                            */
/*  The first line must contain the names of the variables.                   */
/*  Subsequent lines are the values, one case (all variables) per line        */
/*  Names (in the first line) and values (in subsequent lines) are delimited  */
/*  by space, tab, or comma.                                                  */
/*  This routine will allocate names and data, but they must be freed by      */
/*  calling free_data() (defined at the end of this file).                    */
/*  This returns 0 if no error, 1 if error.                                   */
/*                                                                            */
/*  Note that this routine does not do good cleanup in case of insufficient   */
/*  memory, but that would be rare to the point of being unknown.             */
/*  However, in case of user errors it does good cleanup.                     */
/*                                                                            */
/******************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "CONST.H"
#include "FUNCDEFS.H"

#define BUFSIZE 8192        /* This many allocated at a time; not critical */

static inline int digit ( int character )
{
   character &= 255 ;
   return (character >= '0'  &&  character <= '9') ;
}

double parse_double ( char **str )
{
   double number = 0.0 ;

   while (**str  &&  ! ( digit ( **str ) || (**str == '-') || (**str == '.')))
      ++(*str) ;  // Move up to the number

   sscanf_s ( *str , "%lf" , &number ) ; // Get the number

   while (digit ( **str )  ||  (**str == '-')  ||  (**str == '.'))
      ++(*str) ;  // Pass the number

   return number ;
}

int readfile (
   char *name ,    // Name of the data file to read
   int *nvars ,    // Output: Number of variables (as defined by first line)
   char ***names , // Output: Array of pointers to names
   int *ncases ,   // Output: The number of cases in the file
   double **data ) // Output: ncases by nvars data matrix, vars changing fastest
{
   int i, j, k, linelen, bufcnt ;
   char *line, *lptr, var_name[MAX_NAME_LENGTH+1] ;
   double *dptr, *temp ;
   FILE *fp ;

   if (fopen_s ( &fp , name , "rt" )) {
      printf ( "\nERROR... Cannot open file %s", name ) ;
      return 1 ;
      }

   linelen = MAX_VARS * (MAX_NAME_LENGTH+1) ;
   line = (char *) malloc ( linelen ) ;
   assert (line != NULL) ;

   *names = (char **) malloc ( MAX_VARS * sizeof(char *) ) ;
   assert ( names != NULL ) ;
   bufcnt = BUFSIZE ;

/*
   Read the variable names from the first line
*/

   if ((fgets ( line , linelen , fp ) == NULL) || (strlen ( line ) < 2)) {
      fclose ( fp ) ;
      free ( line ) ;
      free ( *names ) ;
      printf ( "\nERROR... problem reading file %s", name ) ;
      return 1 ;
      }

   // In the highly unusual case that the line in the file is longer
   // than we can handle, we must sadly abort
   if (strchr(line,'\n') == NULL  &&  strchr(line,'\r') == NULL) {
      fclose ( fp ) ;
      free ( line ) ;
      free ( *names ) ;
      printf ( "\nERROR... First line in file too long" ) ;
      return 1 ;
      }

   *nvars = 0 ;                 // Will count variables
   lptr = line ;
   for (;;) {                  // For all variables (will count them now)

      if (*nvars >= MAX_VARS) {
         fclose ( fp ) ;
         free ( line ) ;
         free ( *names ) ;
         printf ( "\nERROR... More than %d variables in file", MAX_VARS ) ;
         return 1 ;
         }

      // Parse a single variable name
      k = 0 ;                  // Will index character in name
      while (*lptr  &&  *lptr != ','  &&  *lptr != '\t'  &&  *lptr != ' '
         &&  *lptr != '\n'  &&  *lptr != '\r') {
         if (k < MAX_NAME_LENGTH-1) // Ensure that we do not overrun name array
            var_name[k++] = *lptr++ ;  // Copy the name to EOL or delimiter
         else {               // Should never happen: user's name is too long
            fclose ( fp ) ;
            free ( line ) ;
            printf ( "\nERROR... Variable name longer than %d characters",
                     MAX_NAME_LENGTH ) ;
            return 1 ;
            }
         }
      var_name[k] = 0 ;       // Terminate this name
      _strupr ( var_name ) ;

      // We have just completed parsing a single variable name
      for (j=0 ; j<*nvars ; j++) {  // Is this name already present?
         if (! strcmp ( var_name , (*names)[j] )) { // Check names so far
            fclose ( fp ) ;
            free ( line ) ;
            free ( *names ) ;
            printf ( "\nERROR... name '%s' is duplicated", var_name ) ;
            return 1 ;
            }
         }

      k = (unsigned int) (strlen ( var_name )) + 1 ;
      (*names)[*nvars] = (char *) malloc ( k ) ;
      assert ( (*names)[*nvars] != NULL ) ;
      strcpy_s ( (*names)[*nvars] , k , var_name ) ;
      ++*nvars ;   // Count the number of variables in this file

      // If we have a delimiter, another name follows (probably; see below)
      if (*lptr != ','  &&  *lptr != '\t'  &&  *lptr != ' ') // Not a delimiter?
         break ;      // We have reached the end of the line
      ++lptr ;        // Pass the delimiter

      // A careless user may have multiple blanks or tabs after a variable name
      // This would cause problems, so pass them
      while (*lptr == ' '  ||  *lptr == '\t')
         ++lptr ;
      if (! *lptr  ||  *lptr == '\n'  ||  *lptr == '\r') // Reached end of line?
         break ;                                         // Done if so
      } // For parsing all variables from the header line

   *names = (char **) realloc ( *names , *nvars * sizeof(char *) ) ;
   printf ( "\nFile %s contained %d variables", name, *nvars ) ;

/*
   Read the file.
*/

   *data = (double *) malloc ( BUFSIZE * *nvars * sizeof(double) ) ;
   assert (*data != NULL) ;

   *ncases = 0 ;
   for (;;) {

      if ((fgets ( line , linelen , fp ) == NULL) || (strlen ( line ) < 2)) {
         if (ferror ( fp )  ||  ! ncases) {
            fclose ( fp ) ;
            free ( line ) ;
            free ( *data ) ;
            for (i=0 ; i<*nvars ; i++)
               free ( (*names)[i] ) ;
            free ( *names ) ;
            printf ( "\nERROR... Problem reading file %s", name ) ;
            return 1 ;
            }
         else
            break ;       // Normal end of file
         }

      // Normally, every line will be terminated by a newline character,
      // which is included in the string read by fgets.  If the line is
      // not terminated by a newline, one of two things has happened,
      // either of which means that we are done.  It may be that the program
      // that created the file failed to terminate the last line of the file
      // with a newline.  If so, we just process the line and proceed normally,
      // though perhaps grumbling a bit and perhaps flogging whoever wrote the
      // offending program.  The other possibility, hopefully rare to the point
      // of never happening, is that the line in the file is much longer than
      // we anticipated (linelen).  In this case all we can do is quit in shame.

      if (strchr(line,'\n') == NULL  &&  strchr(line,'\r') == NULL) { // No newline?
         fgetc ( fp ) ;        // Try reading another character to see if EOF
         if (! feof ( fp )) {  // If not at EOF, the line was longer than linelen
            fclose ( fp ) ;    // So all we can do is quit
            free ( line ) ;
            free ( *data ) ;
            for (i=0 ; i<*nvars ; i++)
               free ( (*names)[i] ) ;
            free ( *names ) ;
            printf ( "\nERROR... A line in the file is too long" ) ;
            return 1 ;
            }
         }

      if (! bufcnt--) {  // Allocate a new memory block if needed

         temp = (double *) realloc ( *data ,
                           (*ncases + BUFSIZE) * *nvars * sizeof(double) ) ;
         assert (temp != NULL) ;
         *data = temp ;
         bufcnt = BUFSIZE - 1 ;
         }

      lptr = line ;                        // Parse the data from this line
      dptr = *data + *ncases * *nvars ;    // This case will go here

      for (i=0 ; i<*nvars ; i++)
         *dptr++ = parse_double ( &lptr ) ;

      ++*ncases ;                           // Count cases read

      if (feof ( fp ))
         break ;

      }  /* Endless loop until file runs out */

   fclose ( fp ) ;
   free ( line ) ;

   if (! *ncases) {
      free ( *data ) ;
      for (i=0 ; i<*nvars ; i++)
         free ( (*names)[i] ) ;
      free ( *names ) ;
      return 1 ;
      }

   *data = (double *) realloc ( *data , *ncases * *nvars * sizeof(double) ) ;

   printf ( " and %d cases", *ncases ) ;

   return 0 ;
}

void free_data ( int nvars , char **names , double *data )
{
   int i ;

   assert ( names != NULL ) ;
   for (i=0 ; i<nvars ; i++) {
      assert ( names[i] != NULL ) ;
      free ( names[i] ) ;
      }
   free ( names ) ;

   assert ( data != NULL ) ;
   free ( data ) ;
}
