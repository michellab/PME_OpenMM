from simtk import openmm, unit
from simtk.openmm import app
from parmed.amber import *
import time,sys, collections,copy
import numpy as np
from pymbar import MBAR, timeseries
import os,sys
import time
import glob

pressure = 1*unit.atmospheres
temperature = 298*unit.kelvin
collision_rate = 10.0/unit.picoseconds
timestep = 2.0*unit.femtosecond
kT =unit.AVOGADRO_CONSTANT_NA*unit.BOLTZMANN_CONSTANT_kB*temperature

def set_lambda_electrostatics(reference_system,alchemical_atoms, lambda_electrostatics):
    #reference_system: the OpenMM system
    #alchemical_atoms: ligand atoms to be switched off -- G2
    #alchemical_ions :  adidtional ions to be switched on
    #lambda_electrostatics: lambda parameter

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


    for particle_index in range(nonbonded_force.getNumParticles()):
        [charge,sigma,epsilon] = nonbonded_force.getParticleParameters(particle_index)
        #check if the particle index is in the alchemical atoms, so switch off the ligand
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


#MAIN
#load all the save states of OpenMM
#Load Top and Crd
base = AmberParm("../../input/SYSTEM.top","../../input/SYSTEM.crd")
#collect the atoms to switch off
alchemical_atoms = []
for at in base.residues[0]:
    alchemical_atoms.append(at.idx)

reference_system = base.createSystem(constraints=app.AllBonds,rigidWater=True,nonbondedMethod=app.PME,nonbondedCutoff=1.2*unit.nanometers)
platform = openmm.Platform.getPlatformByName('Reference')
lambdas = np.linspace(0.0,1.0,11) # this is the lambda array

pos_files = glob.glob("positions/*.pos")
pos_files.sort()
simfile = open("simfile.csv","w")

print("Computing the potential energy...")
#print(pos_files)
for pos_file in pos_files:
    #cycle through all the position files
    #compute the potentiial energy for every thermodyamic state
    outline = "\n"
    #print("File %s"% pos_file)
    for l in lambdas:
        #print("State %.2f" % l)
        #set the thermodynamic state l-th
        thermo_state = set_lambda_electrostatics(reference_system,alchemical_atoms, l)
        #add barostat -- basically create a system as it were a simulation
        alchemical_barostat = openmm.MonteCarloBarostat(pressure, temperature, 25)
        thermo_state.addForce(alchemical_barostat)
        #add the thermostat
        alchemical_thermostat = openmm.AndersenThermostat(temperature, collision_rate)
        thermo_state.addForce(alchemical_thermostat)
        #define the Verlet integrator
        thermo_integrator = openmm.VerletIntegrator(timestep)
        #create a context simulation object
        thermo_simulation = app.Simulation(base.topology,thermo_state,thermo_integrator,platform)
        thermo_simulation.loadState(pos_file)
        potential_energy = thermo_simulation.context.getState(getEnergy=True).getPotentialEnergy()/(kT)
        #print("potential energy %.4f" % potential_energy)
        outline+=",%.4f"  % potential_energy

        del thermo_state #delete the context so you can create a new one
    #write the potential energies for all the lambda and go one with the new positions
    #print("writing potential..")
    simfile.write(outline)


simfile.close()
#done
