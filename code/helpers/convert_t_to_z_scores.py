#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author JensNRAD


import os, sys
import argparse
import nipype.interfaces.fsl as fsl


parser = argparse.ArgumentParser(description='convert BROCCOLI tscores to zstats using fsl tools')
parser.add_argument('-t','--tscores', 
                    help='input tscores',
                    type=str,
                    required=True)

parser.add_argument('-d','--dof', 
                    help='Degree of freedoms',
                    type=int,
                    required=True)

args = vars(parser.parse_args())
input_file_name = args['tscores']
dof = args['dof']

ones = fsl.BinaryMaths(in_file=input_file_name,
                       operand_file=input_file_name,
                       out_file=os.path.join(os.getcwd(),
                                            'ones.nii.gz'),
                        operation='div',
                        nan2zeros=True)
result = ones.run()

'''
Unfortunatley, ttoz is not wrapped by NiPype yet.
'''

out_file_name = os.path.join(os.getcwd(),
            os.path.basename(input_file_name).replace('_tscores_',
                                                      '_zstasts_'))

cmd = 'ttoz -zout {} {} {} {}'.format(out_file_name,
                                      result.outputs.out_file,
                                      input_file_name,
                                      dof)
os.system(cmd)
os.remove(result.outputs.out_file)
