/******************************************************************************/
/*                                                                            */
/* CONST.H - System and program limitation constants                          */
/*                                                                            */
/******************************************************************************/

#if !defined(CONST_H)
#define CONST_H
#endif

#define MAXPOSNUM 2147483647

#if ! defined ( PI )
#define PI 3.141592653589793
#define TWO_PI (2 * 3.141592653589793)
#endif


/*
   These are universal error codes.  The first two have mandatory values!
   All values should be nonnegative, so that negatives can have special use.
*/

#define ERROR_OK 0
#define ERROR_ESCAPE 1
#define ERROR_ABORT 2
#define ERROR_INSUFFICIENT_MEMORY 3
#define ERROR_SYNTAX 4
#define ERROR_COMMAND_TOO_LONG 5
#define ERROR_CANNOT_OPEN_FILE 6
#define ERROR_CANNOT_READ_FILE 7
#define ERROR_CANNOT_WRITE_FILE 8
#define ERROR_INVALID_DATAFIELD 9
#define ERROR_VARIABLE_NOT_FOUND 10
#define ERROR_INVALID_FILE 11
#define ERROR_NO_CASES 12

#define MAX_VARS 8192       /* Maximum number of variables in the file */
#define MAX_NAME_LENGTH 15  /* Maximum number of characters in name */
