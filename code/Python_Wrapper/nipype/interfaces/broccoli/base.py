from nipype.interfaces.base import BaseInterfaceInputSpec
import traits

class BroccoliInputSpec(BaseInterfaceInputSpec):
    opencl_platform = traits.Int(0)
    opencl_device = traits.Int(0)
    show_results = traits.Bool(False)
    
    base_name = traits.Str(value='output', desc='base name that all output files will start with')
