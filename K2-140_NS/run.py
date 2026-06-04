import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import sys
sys.path.append('../')
import allesfitter
import matplotlib as mpl

mpl.use("Agg")
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

def run_allesfitter(path):
   allesfitter.show_initial_guess(path,do_logprint=False)
   allesfitter.ns_fit(path)
   allesfitter.ns_output(path)

path = '.'

run_allesfitter(path)

