%module broccoli
%{
  /* Includes the header in the wrapper code */
  #include "broccoli_lib.h"
%}
 
/* Parse the header file to generate wrappers */
%include "broccoli_lib.h"

%extend Coords3D
{
    int __getitem__(int i) const
    {
        return (*$self)[i];
    }
    
    void __setitem(int i, int v)
    {
        (*$self)[i] = v;
    }
}