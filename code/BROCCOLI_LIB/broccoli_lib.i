%module broccoli
%{
  /* Includes the header in the wrapper code */
  #include "broccoli_lib.h"
%}
 
/* Parse the header file to generate wrappers */

%include exception.i

%typemap(in) float *
{
    /* Check if is a list */
    if (PyList_Check($input)) {
        int size = PyList_Size($input);
        int i = 0;
        $1 = (float*) malloc((size)*sizeof(float *));
        for (i = 0; i < size; i++)
        {
            PyObject *o = PyList_GetItem($input,i);
            if (PyFloat_Check(o))
            {
                $1[i] = (float)PyFloat_AsDouble(PyList_GetItem($input,i));
            }
            else if (PyInt_Check(o))
            {
                $1[i] = (float)PyInt_AsLong(PyList_GetItem($input,i));
            }
            else
            {
                PyErr_SetString(PyExc_TypeError,"list must contain only numbers");
                free($1);
                return NULL;
            }
        }
        $1[i] = 0;
    }
    else 
    {
        PyErr_SetString(PyExc_TypeError,"not a list");
        return NULL;
    }
}

%typemap(out) int *
{
  $result = PyList_New(200);
  for (int i = 0; i < 200; ++i) {
      PyList_SetItem($result, i, PyInt_FromLong($1[i]));
  }
}

%ignore Coords3D::operator[];
%include "broccoli_lib.h"

typedef unsigned int cl_uint;

%extend Coords3D
{
    int __getitem__(int i) const
    {
        if (i < 0 || i > 2)
        {
            SWIG_Error(SWIG_IndexError, "index must be between 0 and 2");
            return 0;
        }
        return (*$self)[i];
    }
    
    void __setitem__(int i, int v)
    {
        if (i < 0 || i > 2)
        {
            SWIG_Error(SWIG_IndexError, "index must be between 0 and 2");
            return;
        }
        (*$self)[i] = v;
    }
    
    const char* __repr__() const
    {
        char* buf = (char*)malloc(64);
        sprintf(buf, "Coords3D(%d, %d, %d)", (*$self)[0], (*$self)[1], (*$self)[2]);
        return buf;
    }
}
