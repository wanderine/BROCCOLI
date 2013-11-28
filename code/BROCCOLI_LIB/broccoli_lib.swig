%module broccoli
%{
  /* Includes the header in the wrapper code */
  #include "broccoli_lib.h"
%}
 
/* Parse the header file to generate wrappers */
%include "broccoli_lib.h"