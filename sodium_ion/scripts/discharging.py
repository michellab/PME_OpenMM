""" Alchemical free energy calculations with OpenMM PME

 Scripts discharging.py and vanishing.py allow the computation of discharging and 
 vanishing free energy for a sodium ion. They work only for a single ion solvation free energy
 and I did not tested them for polyatomic species.

The script was tested on a Ubuntu 14.04, CUDA 7.5, OpenMM 7.0. 

Initially, the necessary packages are imported: """

from simtk import openmm, unit  #--> OpenMM 7.0 
from simtk.openmm import app    #--> OpenMM 
from parmed.amber import *      # This is parmed. It can be cloned from here: https://github.com/ParmEd/ParmEd
                                # all the versions are fine for the purpose of this script
import time,sys, copy # time was used to benchmark the code, while copy is necessary for a deep copy of an OpenMM system
import numpy as np 
import os


""" Initially I defined the constants""":
#TODO: add a parser/reader of a sim file
#DEFINE ugly constants
pressure = 1*unit.atmospheres             # pressure  Monte Carlo barostat
temperature = 298*unit.kelvin             # temperature
collision_rate = 10.0/unit.picoseconds    # collision rate for the Andersen Thermostat
timestep = 2.0*unit.femtosecond           # Time step of integration
kT =unit.AVOGADRO_CONSTANT_NA*unit.BOLTZMANN_CONSTANT_kB*temperature # Boltzmann


"""set_lambda_electrostatics is the main function of the code as it allows to alchemically perturbed
the solutes"""
def set_lambda_electrostatics(reference_system,alchemical_atoms, lambda_electrostatics):
    r"""This function allows to alchemically perturb atoms partial charges of a solute. 
    Parameters
    ----------
    reference_system:       OpenMM system
                            This is a reference system created in OpenMM at the beginning of the code
    
    alchemical_atoms:       list
                            list of atoms'indexes to be perturbed
                        
    lambda_electrostatics:  double
                            lambda value, used to scale the atomic charges

    Returns
    -------
    copy_system:            OpenMM system
                            This is a copy of the reference system, whose partial charges are scaled by lambda
    """

    #If we are going to work directly onto the reference system OpenMM will raise an error
    #so the strategy is to make a deep copy of the original system and work on the copy
    copy_system = copy.deepcopy(reference_system) # copy the reference system
    #now we are looking for the nonbonded forces of the OpenMM system
    for force_index, reference_force in enumerate(reference_system.getForces()):
        reference_force_name = reference_force.__class__.__name__ 
        if (reference_force_name == "NonbondedForce"):
            #now we have the nonbonded forces 
            nonbonded_force =copy.deepcopy(reference_force)#copy the non bonded force specifications to the copied system
            break
            

    #here there's the core of the alchemical perturbation
    #cycle through all the particle of the system
    for particle_index in range(nonbonded_force.getNumParticles()):
        #take the values of charge, sigma and epsilon -- this is a common procedure in OpenMM scirpts
        [charge,sigma,epsilon] = nonbonded_force.getParticleParameters(particle_index)
        #if the particle index is an alchemical particle, scale its charges
        if particle_index in alchemical_atoms:
            #charges are scaled linearly
            new_charge = (1-lambda_electrostatics)*charge
            #and then added to the nonbonded_force of the copied system
            nonbonded_force.setParticleParameters(particle_index,new_charge,sigma,epsilon)
           

    #TODO:
    # do we need to add the exception?
    # we should carefully understand if we needed to add exceptions, namely add 
    # the calculation of the 1-4 intramolecular terms. For the moment 
    # everything was tested without them
    #for exception_index in range(nonbonded_force.getNumExceptions()):
    #    [iatom,jatom,chargeprod,sigma,epsilon] = nonbonded_force.getExceptionParameters(exception_index)
    #    print(iatom,jatom,chargeprod,sigma,epsilon)

    #NOW THIS IS ABSOLUTELY IMPORTANT TO MAKE EVERYTHING WORK
    # Since we are working with a copy of the real system we MUST  remove 
    # the nonbonded forces since they are exactly the same of the reference system
    # So to remove these force we can call .removeForce(INDEX), where index is force_index
    # and then we must add the nonbonded_force variable, which is the new non bonded forces
    # modified with the alchemical parameter
    copy_system.removeForce(force_index)    #remove
    copy_system.addForce(nonbonded_force)   #add
    return copy_system


################
###MAIN#########
################

#Parmed allows an easy loading of the topology and coordinate files  
# and it easily handle them with OpenMM
base = AmberParm("../input/SYSTEM.top","../input/SYSTEM.crd") # load top and crd
#create an openmm System from Amber parameters
#the reference system can be create through parmed, which supports OpenMM 
reference_system = base.createSystem(constraints=app.AllBonds,rigidWater=True,nonbondedMethod=app.PME,nonbondedCutoff=1.0*unit.nanometers)
# here we have specified all bonds, shake on waters, PME as electrostatic and a cutoff of 10 angstrom 

#create a platform object, work on CUDA
platform = openmm.Platform.getPlatformByName('CUDA')
properties = {'CudaDeviceIndex': '0','CudaPrecision': 'mixed'}
#reference values for a simulation:
nsteps =10000
niterations = 100
#we want to save 2000 frames for end states
freq_endstates = (nsteps*niterations)/2000 #this will slow down the simulation!
#for al lthe others it's ok to save 5 snapshots
freq_midstates = (nsteps*niterations)/5
#set the lambda spacing
lambdas = np.linspace(0.0,1.0,11)
nstates = len(lambdas)
#create a 3D array to store all the potential energy for each thermodynamic state
#this will be useful late for MBAR
u_kln = np.zeros([nstates,nstates,niterations],np.float64)

#now run the simulations
for k in range(nstates):
    print("*******************Lambda %.2f*******************" % lambdas[k])
    #create the lambda folderse if it does not exist
    if not os.path.exists("lambda-%.2f" % lambdas[k]):
            os.makedirs("lambda-%.2f" % lambdas[k])
    #in each lambda folder save a simfile.csv
    simfile = open("lambda-%.2f/simfile.csv" % lambdas[k],"w")

    #if we are doing lambda =0.0
    if (k==0):
        #at lambda 0.0 initialize the system 
        #when we are at the very first iteration of lambda 0, create all the initial system
        #create an alchemical system
        #to avoid possibel misunderstandings in OpenMM we willcall the copied system from 
        # set_lambda_electrostatics with another name
        alchemical_system = set_lambda_electrostatics(reference_system,[0],lambdas[k])
        #add MC barostat
        alchemical_barostat = openmm.MonteCarloBarostat(pressure, temperature, 25)
        alchemical_system.addForce(alchemical_barostat)
        #add Andersen Thermostat
        alchemical_thermostat = openmm.AndersenThermostat(temperature, collision_rate)
        alchemical_system.addForce(alchemical_thermostat)
        #define the Verlet integrator
        alchemical_integrator = openmm.VerletIntegrator(timestep)
        #set the simulation for the copied system
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
        #save  2000 frames for end states
        alchemical_simulation.reporters.append(app.DCDReporter('lambda-%.2f/trajectory.dcd'%lambdas[k], freq_endstates))
    else:
        #save 3 snapshot for the other states
        alchemical_simulation.reporters.append(app.DCDReporter('lambda-%.2f/trajectory.dcd'%lambdas[k], freq_midstates))


    #now we can proceed with the iteration for each step
    #For each iteration we are running nsteps of MD 
    #then we extract the potential energy of each thermodynamic state

    for iteration in range(niterations):
        #print the potential energy
        #alchemical_simulation.reporters.append(app.StateDataReporter(sys.stdout, 500, step=True, potentialEnergy=True, temperature=True,separator='\t'))
        #run the md steps
        alchemical_simulation.step(nsteps)
        #store a temporary alchemical system
        tmp_system = alchemical_system  #this is the system
        tmp = alchemical_simulation     #there are the simulation info
        #compute the alchemical potentials
        outline = "\n" #create a string to store all the pot energies and save them to a file
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
            #print(potential_energy)
            u_kln[k,l,iteration] = potential_energy
            outline+=",%.4f"  % potential_energy

        #write the potential energy of each thermodynamic state on the simfile.csv
        simfile.write(outline)


#once all the simulations are finished closed the simfile.csv
simfile.close()        



#ONLY FOR SODIUM ION SIMULATIONS
#COMPUTE THE discharging free energy with MBAR
print("**************************************************")
print("Estimation of free energy with MBAR ...")
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
#extract the free energy change
print('DeltaF_ij (kcal/mol):')
print(DeltaF_ij[0,nstates-1]*298.0*0.001987204)

#write a freenrg-MBAR.dat file
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
