import numpy as np 
from pymbar import MBAR, timeseries
import os,sys
import time
import glob

#load the simfile with numpy
#len_lambdas = int(sys.argv[2])
#niterations = int(sys.argv[3])
#collect all the simfiles
print("Collecting simfiles...")
inptfiles = glob.glob("lambda-*/simfile.csv")
inptfiles.sort()
print(inptfiles)
len_lambdas =len(inptfiles)
print("total number of lambda windows %d" % len_lambdas)
columns = [] # these are the columns we need
for i in range(len_lambdas):
    columns.append(i+1)

counter = 0 
for ifile in inptfiles:
    reader = np.loadtxt(ifile,delimiter=",",usecols=columns)
    #now chec if everything is correctly sorted
    #the number of iterations comes from the length of each columnn
    if counter==0:
        #initialize the u_kln matrix
        n_iterations = len(reader[:,0])
        u_kln = np.zeros([len_lambdas,len_lambdas,n_iterations],np.float64) 
    test_iterations = len(reader[:,0])
    print(test_iterations)
    #now fill the array
    #counter is associated with the lambda
    for n in range(n_iterations):
        for l in range(len_lambdas):
            #len lambdas is basically the number of states
            #counter the actual lambda
            u_kln[counter,l,n] = float(reader[n,l])
    counter+=1

print("**************************************************")
print("Estimation of free energy with MBAR ...")

# Subsample data to extract uncorrelated equilibrium timeseries
N_k = np.zeros([len_lambdas], np.int32) # number of uncorrelated samples
for k in range(len_lambdas):
    [nequil, g, Neff_max] = timeseries.detectEquilibration(u_kln[k,k,:])
    indices = timeseries.subsampleCorrelatedData(u_kln[k,k,:], g=g)
    N_k[k] = len(indices)
    u_kln[k,:,0:N_k[k]] = u_kln[k,:,indices].T
# Compute free energy differences and statistical uncertainties
mbar = MBAR(u_kln, N_k,verbose = True, method="adaptive", relative_tolerance=1e-10)#, initialize="BAR")
[DeltaF_ij, dDeltaF_ij, Theta_ij] = mbar.getFreeEnergyDifferences(uncertainty_method='svd-ew')



print('DeltaF_ij (kcal/mol):')
print(DeltaF_ij[0,len_lambdas-1]*298.0*0.001987204)

mbar_fe = DeltaF_ij[0,len_lambdas-1]*298.0*0.001987204
dmbar_fe = dDeltaF_ij[0,len_lambdas-1]*298.0*0.001987204
#write the free energy
mbar_file = open("freenrg-MBAR.dat","w")
mbar_file.write("\n")
mbar_file.write("Free energy differences matrix from MBAR in reduced units:")
mbar_file.write(DeltaF_ij)
mbar_file.write("\n")
mbar_file.write("Free energy MBAR:  %.4f +/- %.4f\n" %(mbar_fe, dmbar_fe))
mbar_file.close()
