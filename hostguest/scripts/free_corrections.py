#Script to estimate CBC-G3 correction terms with rocklin approach
import parmed
from parmed.amber import * 
import mdtraj as md
import os, math
import numpy as np
import sys

#ugly constant
ofile = open("cor_terms.csv","w")
PoissonNPSolverBin= "/home/steboss/local/apbs/bin/apbs"
psi_ls = -2.837297
xi_CB = 2.38008 #see eq. (21) of Rocklin paper
pi = math.pi
coulomb_factor = 138.93545585 # (4*pi*eps_0)**-1 in (kJ nm)/(e**2 mol)
kB = 0.0083144621 # kJ/(mol K)
T = 298
kBT = kB * T
eps_solv = 82.0 # switch this to 97 to compare with Rocklin value

###################### PQR CONVERSION #########################################
def general_pqr(pqrfile):
    #recreate the pqr according to Rocklin standard,
    #so start with ATOM and no END
    ifile = open(pqrfile,"r")
    ifilereader = ifile.readlines()
    ofile = open("free_tmp.pqr","w")

    for line in ifilereader:
        splitter = line.split()
        if splitter[0]=="END":
            continue
        else:
            copy = "ATOM"+ line[6:]
            ofile.write(copy)

    ofile.close()
    ifile.close()
    #rename the file
    cmd = "mv free_tmp.pqr free.pqr"
    os.system(cmd)



def ligand_only(pqrfile,lig_name):
    #here we want the ligand to have charges
    #while the protein 0.0          
    ifile = open(pqrfile,"r").readlines()
    ofile = open("ligand_only.pqr","w")

    for line in ifile:
        res_name = line.split()[3]
        if res_name==lig_name:
            lig_atnumb = int(line.split()[1])
            lig_atom = line.split()[2]
            lig_chain = int(line.split()[4])
            lig_xx = float(line.split()[5])
            lig_yy = float(line.split()[6])
            lig_zz = float(line.split()[7])
            lig_charge = float(line.split()[8])
            lig_radius = float(line.split()[-1])
            #print(std_string)
            ofile.write("ATOM  %5d %-3s  %-3s   %3d    %7.3f %7.3f %7.3f %8.4f %8.4f\n" % (lig_atnumb,lig_atom,lig_name,lig_chain,\
                                                                                            lig_xx,lig_yy,lig_zz,lig_charge,lig_radius))
        else:
            res_atnumb = int(line.split()[1])
            res_atom = line.split()[2]
            res_name = line.split()[3]
            res_chain = int(line.split()[4])
            res_xx = float(line.split()[5])
            res_yy = float(line.split()[6])
            res_zz = float(line.split()[7])
            res_charge =0.0000
            res_radius = float(line.split()[-1])
            #print(std_string)
            ofile.write("ATOM  %5d %-3s  %-3s   %3d    %7.3f %7.3f %7.3f %8.4f %8.4f\n" % (res_atnumb,res_atom,res_name,res_chain,\
                                                                                            res_xx,res_yy,res_zz,res_charge,res_radius))

    ofile.close()

#################CORRECTION TERMS##########################################

#NET
def compute_net(qprot,qlig,box_edge):

    #coulomb_factor =(4*pi*eps_0)**-1
    # pi*coulomb_factor = (4*eps_0)**-1
    # (1/2)*coulomb above

    coulomb_8pi = (pi/2)*(coulomb_factor)
    DGnet = -(psi_ls/coulomb_8pi)*((qprot+qlig)**2 - qprot**2)*(1/box_edge)

    #DG net units are kJ/mol
    return DGnet

#USV
def compute_usv(qprot,qlig,box_edge):

    coulomb_8pi = (pi/2)*(coulomb_factor)
    DGusv = (psi_ls/coulomb_8pi)*(1-(1/eps_solv))*((qprot+qlig)**2 - qprot**2)*(1/box_edge)

    #units in kJ/mol
    return DGusv



######RIP CALCULATION######
def compute_rip(qprot,qlig,bound_edge):

    #firs tcompute the integrals:
    #1) apbs calculation of protein and ligand
    print("Performin the APBS calculation...")
    apbs_rip(eps_solv)
    print("Running APBS for RIP calculations ...")
    cmd = "%s  apbs.in 1>std.out 2>std.err"  % PoissonNPSolverBin
    os.system(cmd)
    #once apbs is run we need to analyse the RIP
    print("Reading the protein_RIP.dx and ligand_RIP.dx")
    #dxfile for the protein and ligand:
    dx_ligand  = "ligand_RIP.dx"
    #read the APBS potentialmaps for ligand
    apbs_ligand = open(dx_ligand,"r")
    lines_ligand = apbs_ligand.readlines()
    apbs_ligand.close()
    #due to the nature of the calculation we assume that the spacing 
    #on the x axis is the same for y and z 
    spacing_ligand = float(lines_ligand[6].split()[1])
    #eliminate all the other non-data lines:
    lines_ligand  = filter(lambda x: x[0] in "0123456789-.", lines_ligand)
    tot_pot_lig = 0.0 # ligand total potential
    numpts_lig = 0 # number of points for the ligand grid

    print("Integration...")
    for line in lines_ligand:
        tot_pot_lig+=sum([float(x) for x in line.split()])
        numpts_lig+= len(line.split())
    #compute the volume in nm^3:
    V_lig = (((spacing_ligand**3)/1000.0)*numpts_lig)
    #B_HET[L,apbs_box]
    B_lig = tot_pot_lig*kBT*V_lig/numpts_lig
    #and B_HET[Qlig,apbs_box]
    B_Q_lig = ((qlig * xi_CB * coulomb_factor / eps_solv * (V_lig ** (2.0 / 3.0))))
    print("Carrying out all the numbers...")
    #this are the integrals from the apbs cubic box size
    RIP_lig  = B_lig - B_Q_lig     #to be converted to kcal by /4.184
    #print(RIP_prot, RIP_lig)
    #compute the real rip term for bound and free phase
    #while here we have the real ddg rip for passing from np to a box
    #RIP_bound = ((RIP_prot + RIP_lig)*(qprot+qlig) - (RIP_prot*qprot))*(1/bound_edge**3)
    RIP_free  = ((RIP_lig*qlig))*(1/free_edge**3)
    return(RIP_free,RIP_lig)  #these will be in kJ
    

def apbs_rip(epsilon):

    ifile = open("apbs.in","w")

    ifile.write("""# READ IN MOLECULES
read 
    mol pqr %s    
end

# CALCULATE POTENTIAL WITH ONLY THE LIGAND CHARGES
elec name free
    mg-manual
    dime 257 257 257
    glen 120 120 120
    gcent mol 1
    mol 1
    lpbe
    bcfl mdh
    pdie 1.0
    sdie %.1f
    chgm spl4
    srfm smol
    srad 1.4
    swin 0.3
    sdens 40.0
    temp 298.0
    calcenergy no
    calcforce no
    write pot dx ligand_RIP
end

# SO LONG
quit""" %("free.pqr",epsilon) )

    ifile.close()
    #run apbs
###################################################

#EMP
def compute_emp(qprot,qlig,res_int_lig,bound_edge):
    #res_int_lig is the ligand residual integral
    #############################################################
    #first perform the calculation of I_L,SLV:

    #compute the homogeneous ligand integral
    print("Computation of the ligand vacuum potential...")
    apbs_rip(1.0)
    #dxfile for the ligand:
    dx_ligand  = "ligand_RIP.dx"
    #next charge forligand qlig
    #read the APBS potential map
    apbs_ligand = open(dx_ligand,"r")
    lines_ligand = apbs_ligand.readlines()
    apbs_ligand.close()
    #due to the nature of the calculation we assume that the spacing 
    #on the x axis is the same for y and z 
    spacing_ligand = float(lines_ligand[6].split()[1])
    #eliminate all the other non-data lines:
    lines_ligand  = filter(lambda x: x[0] in "0123456789-.", lines_ligand)

    tot_pot_lig = 0.0 # ligand total potential
    numpts_lig = 0 # number of points for the ligand grid

    print("Integration...")

    for line in lines_ligand:
        tot_pot_lig+=sum([float(x) for x in line.split()])
        numpts_lig+= len(line.split())
    #compute the volume in nm^3:
    V_lig = (((spacing_ligand**3)/1000.0)*numpts_lig)

    #B_HOM[L,apbs_box]
    B_hom_lig = tot_pot_lig*kBT*V_lig/numpts_lig
    #and B_HOM[Q_lig,apbs_box]
    B_Q_hom_lig = ((qlig * xi_CB * coulomb_factor / 1.0 * (V_lig ** (2.0 / 3.0))))
    print("Carrying out all the numbers...")
    #this are the integrals from the apbs cubic box size #units kJ nm^3/(mol e)
    RIP_hom_lig  = B_hom_lig - B_Q_hom_lig     #to be converted to kcal by /4.184
    #now compute I_L,SLV:
    I_L_SLV = res_int_lig - (B_hom_lig - B_Q_hom_lig)
    print("Solvation contribution to the residual integrated potential I_L, I_L_SLV = %.4f  kJ nm^3/(mol e)" % (I_L_SLV))
    ####################################################################################

    #now compute the protein-ligand  ligand radius:
    R_L_first = (coulomb_factor/2)*(4*pi/3.0)*(1- (1/eps_solv))*(qlig)  #first part
    #then cmopute the inverse of R_L_first and multiply by I_L_SLV aand take the sqrt
    R_L = ((R_L_first)**-1) * (I_L_SLV)
    R_L = math.sqrt(R_L) #nm length
    print("Ligand radius in the bound phase R_L = %.4f nm" % (R_L))
    #TODO we should tae into consideration the case where q_prot is 0, 
    #namely the sodium ion 
    DGemp = (-coulomb_factor/2)*( (16/45)*pi**2)*(1-(1/eps_solv))*((qprot + qlig)**2 - (qprot**2))*((R_L**5)/(bound_edge**6))
    #print("Empirical correction DGemp %.4f kJ" % (DGemp))
    return DGemp

#DSC
def compute_dsc(nwater,qlig,box_edge):
    #the quadrupole trace for tip3p water is 0.007 e nm^2
    gamma = 0.007 # e nm^2
    eps0 = (1/6.0)*coulomb_factor*4*pi  #6*epsilon_0 permittivity
    DGdsc = - (gamma*qlig)*(eps0)*(nwater/box_edge**3)
    return DGdsc

####################################################################
########MAIN#####################

#read the trajectory
top = sys.argv[1]
crd = sys.argv[2]
dcd = sys.argv[3]
step = int(sys.argv[4])#100
base = AmberParm(top,crd)

md_dcd = md.load_dcd(dcd,top=top)


dg_cors = []
dg_nets = []
dg_usvs = []
dg_rips = []
dg_emps = []
dg_dscs = []
ofile.write("NET, USV, RIP, EMP, DSC, COR\n")
#now cycle through all the framse
for i in range(0,len(md_dcd),step ):
    print("Analysing frame %d" % i)
    frame = md_dcd[i]
    #create a folder
    folder = "analysis-%d" % i
    os.mkdir(folder)
    os.chdir(folder)
    #start with the analysis
    #convert the frame  into amberrst7
    frame.save_amberrst7("SYSTEM.crd")
    
    base = AmberParm("../SYSTEM.prmtop","SYSTEM.crd")

    #these operations can be done once:
    if i==0 :
        qlig = 0.0
        #assuming th eligand is the first residue
        for at in base.residues[0].atoms:
            qlig+=at.charge
        print("Total ligand charge %.2f" % (qlig))

        wat = 0 
        for res in base.residues:
            if res.name=="WAT":
                wat+=1
    
    
    #now do all the analysis
    #take the edge of the box in nm
    free_edge = base.box[0]/10  #nm
    #strip out water and ions for the moment
    base.strip(":WAT")
    #create a PQR file
    print("Creating PQR file...")
    parmed.formats.PQRFile.write(base,"free.pqr")
    #generalise the pqr
    general_pqr("free.pqr")
    #compute the correction terms
    #NET
    dg_net_free = compute_net(0.0,qlig,free_edge)
    print("DG_net %.4f kJ  %.4f kcal" % (dg_net_free,dg_net_free/4.184))
    dg_nets.append(dg_net_free/4.184)
    #USV
    dg_usv_free = compute_usv(0.0,qlig,free_edge)
    print("DG_usv %.4f kJ  %.4f kcal" % (dg_usv_free,dg_usv_free/4.184))
    dg_usvs.append(dg_usv_free/4.184)
    #now compute the RIP term for the free phase
    dg_rip_free, res_int_lig= compute_rip(0.0,qlig,free_edge)
    print("DG_rip %.4f kJ  %.4f kcal" % (dg_rip_free,dg_rip_free/4.184))
    dg_rips.append(dg_rip_free/4.184)
    #empirical term
    dg_emp = compute_emp(0.0,qlig,res_int_lig,free_edge)
    dg_emps.append(dg_emp/4.184)
    print("DG_emp %.4f kJ  %.4f kcal" % (dg_emp,dg_emp/4.184))
    #finally we need to compute the psum or dsc correction
    dg_dsc = compute_dsc(wat,qlig,free_edge)
    dg_dscs.append(dg_dsc/4.184)
    print("DG_dsc %.4f kJ  %.4f kcal" % (dg_dsc,dg_dsc/4.184))
    #so the final analytical correction is: NET + USV + RIP + EMP
    ddg_cor = dg_net_free + dg_usv_free + dg_rip_free + dg_emp + dg_dsc
    print("The final correction term for frame %d is DGcor %.4f kJ/mol  or  %.4f kcal/mol" % (i,ddg_cor,ddg_cor/4.184)) 
    #store the correction term
    dg_cors.append(ddg_cor)
    ofile.write("%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n" % (dg_net_free/4.184, dg_usv_free/4.184, dg_rip_free/4.184,\
                                                    dg_emp/4.184,dg_dsc/4.184, ddg_cor/4.184))
   
    os.chdir("../")#return to previous directory

#extract the final value
mean = np.mean(dg_cors)
stddev = np.std(dg_cors)
if np.isnan(mean):
    mean,stddev = 0.0, 0.0

dg_net_mean = np.mean(dg_nets)
dg_net_std  = np.std(dg_nets)
if np.isnan(dg_net_mean):
    dg_net_mean,dg_net_std = 0.0, 0.0

dg_usv_mean = np.mean(dg_usvs)
dg_usv_std  = np.std(dg_usvs)
if np.isnan(dg_usv_mean):
    dg_usv_mean,dg_usv_std = 0.0, 0.0

dg_rip_mean = np.mean(dg_rips)
dg_rip_std  = np.std(dg_rips)
if np.isnan(dg_rip_mean):
    dg_rip_mean,dg_rip_std = 0.0, 0.0

dg_emp_mean = np.mean(dg_emps)
dg_emp_std  = np.std(dg_emps)
if np.isnan(dg_emp_mean):
    dg_emp_mean,dg_emp_std = 0.0, 0.0

dg_dsc_mean = np.mean(dg_dscs)
dg_dsc_std = np.std(dg_dscs)
if np.isnan(dg_dsc_mean):
    dg_dsc_mean,dg_dsc_std = 0.0, 0.0

print("Mean values:")
average_file = open("cor_average.csv","w")
average_file.write("<NET>, std, <USV>, std, <RIP>, std, <EMP>, std, <DSC>, std, <COR>, std\n")
average_file.write("%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n" %\
                    (dg_net_mean, dg_net_std, dg_usv_mean, dg_usv_std, dg_rip_mean, dg_rip_std,\
                    dg_emp_mean, dg_emp_std, dg_dsc_mean, dg_dsc_std, mean/4.184, stddev/4.184))


print("Final free DG cor %.4f +/- %.4f kJ/mol  or  %.4f +/- %.4f kcal/mol" % (mean,stddev,mean/4.184,stddev/4.184))

cmd = "rm -rf analysis-*"
os.system(cmd)