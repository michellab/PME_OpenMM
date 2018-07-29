from simtk import openmm, unit
from simtk.openmm import app
from parmed.amber import *
import time,sys, collections,copy
import numpy as np
from pymbar import MBAR, timeseries
import os,sys
import time
#from openmm_ext import call_openmm_external
#TODO: add a parser/reader of a sim file
#DEFINE ugly constants
pressure = 1*unit.atmospheres
temperature = 298*unit.kelvin
collision_rate = 10.0/unit.picoseconds
timestep = 2.0*unit.femtosecond


def set_lambda_electrostatics(reference_system,alchemical_atoms, lambda_electrostatics):
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
            new_charge = (1-lambda_electrostatics)*charge
            nonbonded_force.setParticleParameters(particle_index,new_charge,sigma,epsilon)

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

#take as input the lambda value
lambda_val = float(sys.argv[1])
#Load Top and Crd
base = AmberParm("../input/SYSTEM.top","../input/SYSTEM.crd")
#collect the atoms to switch off
alchemical_atoms = []
for at in base.residues[0]:
    alchemical_atoms.append(at.idx)

#create an openmm System from Amber parameters
reference_system = base.createSystem(constraints=app.AllBonds,rigidWater=True,nonbondedMethod=app.PME,nonbondedCutoff=1.0*unit.nanometers)
platform = openmm.Platform.getPlatformByName('CUDA')
#properties = {"CudaDeviceIndex":"1"}#{'OpenCLDeviceIndex': '0'}#, 'CudaDeviceIndex': '1'}
properties = {"CudaPrecision":"mixed"}#{'CudaDeviceIndex': '0','CudaPrecision': 'mixed'}
#platform.setPropertyDefaultValue('OpenCLPrecision', 'mixed')
#reference values for a simulation:
#uncomment
nsteps =80000
niterations = 100
#nsteps = 4000
#niterations = 5

#we want to save 2000 frames for end states
freq_endstates = (nsteps*niterations)/2000 #this will slow down the simulation!
#for al lthe others it's ok to save 5 snapshots
freq_midstates = (nsteps*niterations)/5
lambdas = np.linspace(0.0,1.0,11) # this is the lambda array
nstates = len(lambdas)

#create an array to store the potential energy for all the states --> MBAR
#u_kln = np.zeros([nstates,nstates,niterations],np.float64)
kT =unit.AVOGADRO_CONSTANT_NA*unit.BOLTZMANN_CONSTANT_kB*temperature


#create the lambda folder

print("*******************Lambda %.2f*******************" % lambda_val)
if not os.path.exists("lambda-%.2f" % lambda_val):
        os.makedirs("lambda-%.2f" % lambda_val)
#create a position file folders
lambda_position_folder = "lambda-%.2f/positions/" % lambda_val
os.makedirs(lambda_position_folder)
#initialize the system
alchemical_system = set_lambda_electrostatics(reference_system,alchemical_atoms,lambda_val)
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
#check the lambda val
if (lambda_val==0.0) or (lambda_val==1.0):
    alchemical_simulation.reporters.append(app.DCDReporter('lambda-%.2f/trajectory.dcd'%lambda_val, freq_endstates))
else:
    alchemical_simulation.reporters.append(app.DCDReporter('lambda-%.2f/trajectory.dcd'%lambda_val, freq_midstates))

print("Running simulation...")
for iteration in range(niterations):
    print("Iteration %d/%d" % (iteration,niterations))
    #save the position file
    alchemical_simulation.step(int(nsteps))
    ofile_name= lambda_position_folder + "%d.pos" % iteration
    #create this file and use the
    alchemical_simulation.saveState(ofile_name)

#once all the interactions are done copy the openmm_ext.py  script into the lambda folder
cmd = "cp openmm_ext.py  lambda-%.2f/." % lambda_val
os.system(cmd)
#change directory and go into the lambda folder
os.chdir("lambda-%.2f" %lambda_val)
#now write the submission file
submit_file_name = "create_simfile.sh"
submit_file = open(submit_file_name,"w")
submit_file.write("""#!/bin/bash
#SBATCH -o simfile_creation-%A.out
#SBATCH -p serial
#SBATCH -n 1
#SBATCH --time 48:00:00

module load cuda/7.5
export OPENMM_PLUGIN_DIR=/home/steboss/sire.app/lib/plugins/

~/sire.app/bin/python openmm_ext.py
wait

#eliminate all the positions folder
rm -r positions
""")
submit_file.close()
#submit the script
cmd = "sbatch create_simfile.sh"
os.system(cmd)

#change dir
os.chdir("../")
#done
