from nipype.interfaces.base import BaseInterfaceInputSpec, traits

class BroccoliInputSpec(BaseInterfaceInputSpec):
    opencl_platform = traits.Int(0, usedefault=True)
    opencl_device = traits.Int(0, usedefault=True)
    show_results = traits.Bool(False)
    
    base_name = traits.Str(value='output', desc='base name that all output files will start with')
