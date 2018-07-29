from simtk import openmm, unit
from simtk.openmm import app
from parmed.amber import * 
import time,sys, collections,copy
import numpy as np 
from pymbar import MBAR, timeseries
import os

#DEFINE ugly constants
pressure = 1*unit.atmospheres
temperature = 298.0*unit.kelvin
collision_rate = 10.0/unit.picoseconds
timestep = 2.0*unit.femtosecond 


def set_lambda_electrostatics(reference_system,alchemical_atoms, lambda_sterics):
    #set to (1-lambda_electrostatics)*charge the ion charge
    #copy_system = copy.deepcopy(reference_system)
    #take the non bonded forces
    copy_system = copy.deepcopy(reference_system)
    for force_index, reference_force in enumerate(reference_system.getForces()):
        reference_force_name = reference_force.__class__.__name__ 
        if (reference_force_name == "NonbondedForce"):
            #now we have the nonbonded forces 
            nonbonded_force =copy.deepcopy(reference_force)#copy the force 

            break
            

    #work on the non bonded force
    for particle_index in range(nonbonded_force.getNumParticles()):
        [charge,sigma,epsilon] = nonbonded_force.getParticleParameters(particle_index)
        if particle_index in alchemical_atoms:
            new_epsilon = (1-lambda_sterics)*epsilon
            new_sigma = (1-lambda_sterics)*sigma
            nonbonded_force.setParticleParameters(particle_index,abs(0*charge),new_sigma,new_epsilon)
           
    #do we need to add the exception?
    #no since they deal only with water mols
    #for exception_index in range(nonbonded_force.getNumExceptions()):
    #    [iatom,jatom,chargeprod,sigma,epsilon] = nonbonded_force.getExceptionParameters(exception_index)
    #    print(iatom,jatom,chargeprod,sigma,epsilon)
    #THIS IS ABSOLUTELY IMPORTANT! remove the force to avoid duplicates and OpenMM errors
    copy_system.removeForce(force_index)  
    copy_system.addForce(nonbonded_force)
    return copy_system


################
###MAIN#########
################

#Load Top and Crd
base = AmberParm("../input/SYSTEM.top","../input/SYSTEM.crd") 
#create an openmm System from Amber parameters
reference_system = base.createSystem(constraints=app.AllBonds,rigidWater=True,nonbondedMethod=app.PME,nonbondedCutoff=1.0*unit.nanometers)
platform = openmm.Platform.getPlatformByName('CUDA')
properties = {'CudaDeviceIndex': '0','CudaPrecision': 'mixed'}
#reference values for a simulation:
nsteps =1000
niterations = 50
#we want to save 2000 frames for end states
freq_endstates = (nsteps*niterations)/2000 #this will slow down the simulation!
#for al lthe others it's ok to save 5 snapshots
freq_midstates = (nsteps*niterations)/5

lambdas = np.linspace(0.0,1.0,11)
nstates = len(lambdas)
#create an array to store the potential energy for all the states --> MBAR
u_kln = np.zeros([nstates,nstates,niterations],np.float64)
kT =unit.AVOGADRO_CONSTANT_NA*unit.BOLTZMANN_CONSTANT_kB*temperature

#now run the simulations
for k in range(nstates):
    print("*******************Lambda %.2f*******************" % lambdas[k])
    #create the lambda folderse if it does not exist
    if not os.path.exists("lambda-%.2f" % lambdas[k]):
            os.makedirs("lambda-%.2f" % lambdas[k])
    simfile = open("lambda-%.2f/simfile.csv" % lambdas[k],"w")

    if (k==0):
        #at lambda 0.0 initialize the system 
        #when we are at the very first iteration of lambda 0, create all the initial system
        #create an alchemical system
        alchemical_system = set_lambda_electrostatics(reference_system,[0],lambdas[k])
        #add barostat
        alchemical_barostat = openmm.MonteCarloBarostat(pressure, temperature, 25)
        alchemical_system.addForce(alchemical_barostat)
        #add the thermostat
        alchemical_thermostat = openmm.AndersenThermostat(temperature, collision_rate)
        alchemical_system.addForce(alchemical_thermostat)
        #define the Verlet integrator
        alchemical_integrator = openmm.VerletIntegrator(timestep)
        #set the simulation
        alchemical_simulation = app.Simulation(base.topology, alchemical_system,alchemical_integrator,platform,properties)
        #set the positions based on the coordinate files
        alchemical_simulation.context.setPositions(base.positions)
        alchemical_simulation.context.setVelocitiesToTemperature(temperature)
    else:
        #if we are doing the other iterations or lambdas, recrate the system, based on the temporary/previous
        #alchemical system
        alchemical_system = set_lambda_electrostatics(reference_system,[0],lambdas[k])
        alchemical_barostat = openmm.MonteCarloBarostat(pressure, temperature, 25)
        alchemical_system.addForce(alchemical_barostat)
        alchemical_thermostat = openmm.AndersenThermostat(temperature, collision_rate)
        alchemical_system.addForce(alchemical_thermostat)
        alchemical_integrator = openmm.VerletIntegrator(timestep)
        alchemical_simulation = app.Simulation(base.topology, alchemical_system,alchemical_integrator,platform)
        #take the positions from the previous simulation calling the temporary system
        alchemical_simulation.context.setPositions(tmp.context.getState(getPositions = True).getPositions())
        alchemical_simulation.context.setVelocitiesToTemperature(temperature)

    #save a traj with a minimal option 
    if k==0 or  k==(nstates)-1:
        alchemical_simulation.reporters.append(app.DCDReporter('lambda-%.2f/trajectory.dcd'%lambdas[k], freq_endstates))
    else:
        alchemical_simulation.reporters.append(app.DCDReporter('lambda-%.2f/trajectory.dcd'%lambdas[k], freq_midstates))

    for iteration in range(niterations):
        #print the potential energy
        #alchemical_simulation.reporters.append(app.StateDataReporter(sys.stdout, 500, step=True, potentialEnergy=True, temperature=True,separator='\t'))
        #run the md steps
        alchemical_simulation.step(nsteps)
        #store a temporary alchemical system
        tmp_system = alchemical_system
        tmp = alchemical_simulation
        #compute the alchemical potentials
        outline = "\n"
        #now deal with the thermodynamic states to compute the potential energy
        for l in range(nstates):
            #save the actual coordinates 
            current_pos = alchemical_simulation.context.getState(getPositions = True).getPositions()
            #set the lambda parameters for all the l states 
            thermo_state = set_lambda_electrostatics(reference_system,[0],lambdas[l])
            #set the positions
            thermo_integrator = openmm.VerletIntegrator(timestep)
            thermo_simulation = app.Simulation(base.topology,thermo_state,thermo_integrator,platform)
            #set the position 
            thermo_simulation.context.setPositions(current_pos)
            #retrieve the reduced unit potential energy
            potential_energy = thermo_simulation.context.getState(getEnergy=True).getPotentialEnergy()/(kT)
            print(potential_energy)
            u_kln[k,l,iteration] = potential_energy
            outline+=",%.4f"  % potential_energy
        simfile.write(outline)

simfile.close()        

print("**************************************************")
print("Estimation of free energy with MBAR ...")
#try an on the fly mbar estimation
# Subsample data to extract uncorrelated equilibrium timeseries
N_k = np.zeros([nstates], np.int32) # number of uncorrelated samples
for k in range(nstates):
    [nequil, g, Neff_max] = timeseries.detectEquilibration(u_kln[k,k,:])
    indices = timeseries.subsampleCorrelatedData(u_kln[k,k,:], g=g)
    N_k[k] = len(indices)
    u_kln[k,:,0:N_k[k]] = u_kln[k,:,indices].T
# Compute free energy differences and statistical uncertainties
mbar = MBAR(u_kln, N_k,verbose = True, method="adaptive", relative_tolerance=1e-10)#, initialize="BAR")
[DeltaF_ij, dDeltaF_ij, Theta_ij] = mbar.getFreeEnergyDifferences(uncertainty_method='svd-ew')



print('DeltaF_ij (kcal/mol):')
print(DeltaF_ij[0,nstates-1]*298.0*0.001987204)

mbar_fe = DeltaF_ij[0,nstates-1]*298.0*0.001987204
dmbar_fe = dDeltaF_ij[0,nstates-1]*298.0*0.001987204
#write the free energy
mbar_file = open("freenrg-MBAR.dat","w")
mbar_file.write("\n")
mbar_file.write("Free energy differences matrix from MBAR in reduced units:")
mbar_file.write(DeltaF_ij)
mbar_file.write("\n")
mbar_file.write("Free energy MBAR:  %.4f +/- %.4f\n" (mbar_fe, dmbar_fe))
mbar_file.close()
