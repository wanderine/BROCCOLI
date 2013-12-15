from nipype.interfaces.base import BaseInterfaceInputSpec, BaseInterface, traits, isdefined

class BroccoliInputSpec(BaseInterfaceInputSpec):
    opencl_platform = traits.Int(0, usedefault=True)
    opencl_device = traits.Int(0, usedefault=True)
    show_results = traits.Bool(False)
    
    base_name = traits.Str(value='output', desc='base name that all output files will start with')

class BroccoliInterface(BaseInterface):
    def _get_output_filename(self, name):
        if isdefined(self.inputs.base_name):
            return self.inputs.base_name + '_' + name
        else:
            return name