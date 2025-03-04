import sys
sys.path.append('../')
import allesfitter, os
import matplotlib as mpl

mpl.use("Agg")
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

def run_allesfitter(path):
   allesfitter.show_initial_guess(path,do_logprint=False)
   allesfitter.mcmc_fit(path)
   allesfitter.mcmc_output(path)

path = '.'

run_allesfitter(path)

