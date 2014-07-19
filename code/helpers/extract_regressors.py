#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author JensNRAD


import os, sys
import argparse

import numpy as np

def read_file(fname):
    lines = []
    number_block = False
    with open(fname,'r') as fp:
        for line in fp:
            if line.startswith('/Matrix'):
                number_block = True
                continue
            if number_block:
                vals = line.rstrip().split('\t')
                # print vals
                lines.append(vals)
    return np.array(lines)


def write_files(values):
    for i in range(values.shape[1]):
        fname = 'task{}.txt'.format(i+1)
        print fname
        with open(fname,'w') as fp:
            for j in range(values.shape[0]):
                fp.write('{:.8f}\n'.format(float(values[j,i])))

parser = argparse.ArgumentParser(description='convert fsl design.mat to BROCCOLI regressor file')
parser.add_argument('-i','--input', 
                    help='input file',
                    type=str,
                    required=True)

args = vars(parser.parse_args())
input_file = args['input']

v = read_file(input_file)
print "Found {} regressors".format(v.shape[1])
write_files(v)
