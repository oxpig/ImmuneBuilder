import pdbfixer
import os
import numpy as np
from openmm import app, LangevinIntegrator, CustomExternalForce, CustomTorsionForce, OpenMMException, Platform, unit
from scipy import spatial
import logging
logging.disable()

ENERGY = unit.kilocalories_per_mole
LENGTH = unit.angstroms
spring_unit = ENERGY / (LENGTH ** 2)

CLASH_CUTOFF = 0.63

# Atomic radii for various atom types.
atom_radii = {"C": 1.70, "N": 1.55, 'O': 1.52, 'S': 1.80}

# Sum of van-der-waals radii
radii_sums = dict(
    [(i + j, (atom_radii[i] + atom_radii[j])) for i in list(atom_radii.keys()) for j in list(atom_radii.keys())])
# Clash_cutoff-based radii values
cutoffs = dict(
    [(i + j, CLASH_CUTOFF * (radii_sums[i + j])) for i in list(atom_radii.keys()) for j in list(atom_radii.keys())])

# Using amber14 recommended protein force field
forcefield = app.ForceField("amber14/protein.ff14SB.xml")


def refine(input_file, output_file, check_for_strained_bonds=True, tries=3, n=6, n_threads=-1):
    for i in range(tries):
        if refine_once(input_file, output_file, check_for_strained_bonds=check_for_strained_bonds, n=n, n_threads=n_threads):
            return True
    return False


def refine_once(input_file, output_file, check_for_strained_bonds=True, n=6, n_threads=-1):
    k1s = [2.5,1,0.5,0.25,0.1,0.001]
    k2s = [2.5,5,7.5,15,25,50]
    success = False

    fixer = pdbfixer.PDBFixer(input_file)

    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    
    k1 = k1s[0]
    k2 = -1 if cis_check(fixer.topology, fixer.positions) else k2s[0]

    topology, positions = fixer.topology, fixer.positions

    for i in range(n):
        try:
            simulation = minimize_energy(topology, positions, k1=k1, k2 = k2, n_threads=n_threads)
            topology, positions = simulation.topology, simulation.context.getState(getPositions=True).getPositions()
            acceptable_bonds, trans_peptide_bonds = bond_check(topology, positions), cis_check(topology, positions)
        except OpenMMException as e:
            if (i == n-1) and ("positions" not in locals()):
                print("OpenMM failed to refine {}".format(input_file), flush=True)
                raise e
            else:
                topology, positions = fixer.topology, fixer.positions
                continue

        # If peptide bonds are the wrong length, decrease the strength of the positional restraint
        if not acceptable_bonds:
            k1 = k1s[min(i, len(k1s)-1)]

        # If there are still cis isomers in the model, increase the force to fix these
        if not trans_peptide_bonds:
            k2 = k2s[min(i, len(k2s)-1)]
        else:
            k2 = -1
        
        if acceptable_bonds and trans_peptide_bonds:
            # If peptide bond lengths and torsions are okay, check and fix the chirality.
            try:
                simulation = chirality_fixer(simulation)
                topology, positions = simulation.topology, simulation.context.getState(getPositions=True).getPositions()
            except OpenMMException as e:
                topology, positions = fixer.topology, fixer.positions
                continue

            if check_for_strained_bonds:
                # If all other checks pass, check and fix strained sidechain bonds:
                try:
                    strained_bonds = strained_sidechain_bonds_check(topology, positions)
                    if len(strained_bonds) > 0:
                        needs_recheck = True
                        topology, positions = strained_sidechain_bonds_fixer(strained_bonds, topology, positions, n_threads=n_threads)
                    else:
                        needs_recheck = False
                except OpenMMException as e:
                    topology, positions = fixer.topology, fixer.positions
                    continue
            else:
                needs_recheck = False

            # If it passes all the tests, we are done
            tests = bond_check(topology, positions) and cis_check(topology, positions)
            if needs_recheck:
                tests = tests and strained_sidechain_bonds_check(topology, positions)
            if tests and stereo_check(topology, positions) and clash_check(topology, positions):
                success = True
                break

    with open(output_file, "w") as out_handle:
        app.PDBFile.writeFile(topology, positions, out_handle, keepIds=True)

    return success


def minimize_energy(topology, positions, k1=2.5, k2=2.5, n_threads=-1):
    # Fill in the gaps with OpenMM Modeller
    modeller = app.Modeller(topology, positions)
    modeller.addHydrogens(forcefield)

    # Set up force field
    system = forcefield.createSystem(modeller.topology)

    # Keep atoms close to initial prediction
    force = CustomExternalForce("k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    force.addGlobalParameter("k", k1 * spring_unit)
    for p in ["x0", "y0", "z0"]:
        force.addPerParticleParameter(p)

    for residue in modeller.topology.residues():
        for atom in residue.atoms():
            if atom.name in ["CA", "CB", "N", "C"]:
                force.addParticle(atom.index, modeller.positions[atom.index])
    
    system.addForce(force)

    if k2 > 0.0:
        cis_force = CustomTorsionForce("10*k2*(1+cos(theta))^2")
        cis_force.addGlobalParameter("k2", k2 * ENERGY)

        for chain in modeller.topology.chains():
            residues = [res for res in chain.residues()]
            relevant_atoms = [{atom.name:atom.index for atom in res.atoms() if atom.name in ["N", "CA", "C"]} for res in residues]
            for i in range(1,len(residues)):
                if residues[i].name == "PRO":
                    continue

                resi = relevant_atoms[i-1]
                n_resi = relevant_atoms[i]
                cis_force.addTorsion(resi["CA"], resi["C"], n_resi["N"], n_resi["CA"])
        
        system.addForce(cis_force)

    # Set up integrator
    integrator = LangevinIntegrator(0, 0.01, 0.0)

    # Set up the simulation
    if n_threads > 0:
        # Set number of threads used by OpenMM
        platform = Platform.getPlatformByName('CPU')
        simulation = app.Simulation(modeller.topology, system, integrator, platform, {'Threads': str(n_threads)})
    else:
        simulation = app.Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)

    # Minimize the energy
    simulation.minimizeEnergy()

    return simulation


def chirality_fixer(simulation):
    topology = simulation.topology
    positions = simulation.context.getState(getPositions=True).getPositions()
    
    d_stereoisomers = []
    for residue in topology.residues():
        if residue.name == "GLY":
            continue

        atom_indices = {atom.name:atom.index for atom in residue.atoms() if atom.name in ["N", "CA", "C", "CB"]}
        vectors = [positions[atom_indices[i]] - positions[atom_indices["CA"]] for i in ["N", "C", "CB"]]

        if np.dot(np.cross(vectors[0], vectors[1]), vectors[2]) < .0*LENGTH**3:
            # If it is a D-stereoisomer then flip its H atom
            indices = {x.name:x.index for x in residue.atoms() if x.name in ["HA", "CA"]}
            positions[indices["HA"]] = 2*positions[indices["CA"]] - positions[indices["HA"]]
            
            # Fix the H atom in place
            particle_mass = simulation.system.getParticleMass(indices["HA"])
            simulation.system.setParticleMass(indices["HA"], 0.0)
            d_stereoisomers.append((indices["HA"], particle_mass))
            
    if len(d_stereoisomers) > 0:
        simulation.context.setPositions(positions)

        # Minimize the energy with the evil hydrogens fixed
        simulation.minimizeEnergy()

        # Minimize the energy letting the hydrogens move
        for atom in d_stereoisomers:
            simulation.system.setParticleMass(*atom)
        simulation.minimizeEnergy()
    
    return simulation


def bond_check(topology, positions):
    for chain in topology.chains():
        residues = [{atom.name:atom.index for atom in res.atoms() if atom.name in ["N", "C"]} for res in chain.residues()]
        for i in range(len(residues)-1):
            # For simplicity we only check the peptide bond length as the rest should be correct as they are hard coded 
            v = np.linalg.norm(positions[residues[i]["C"]] -  positions[residues[i+1]["N"]])
            if abs(v - 1.329*LENGTH) > 0.1*LENGTH:
                return False
    return True


def cis_bond(p0,p1,p2,p3):
    ab = p1-p0
    cd = p2-p1
    db = p3-p2
    
    u = np.cross(-ab, cd)
    v = np.cross(db, cd)
    return np.dot(u,v) > 0
            

def cis_check(topology, positions):
    pos = np.array(positions.value_in_unit(LENGTH))
    for chain in topology.chains():
        residues = [res for res in chain.residues()]
        relevant_atoms = [{atom.name:atom.index for atom in res.atoms() if atom.name in ["N", "CA", "C"]} for res in residues]
        for i in range(1,len(residues)):
            if residues[i].name == "PRO":
                continue

            resi = relevant_atoms[i-1]
            n_resi = relevant_atoms[i]
            p0,p1,p2,p3 = pos[resi["CA"]],pos[resi["C"]],pos[n_resi["N"]],pos[n_resi["CA"]]
            if cis_bond(p0,p1,p2,p3):
                return False
    return True


def stereo_check(topology, positions):
    pos = np.array(positions.value_in_unit(LENGTH))
    for residue in topology.residues():
        if residue.name == "GLY":
            continue

        atom_indices = {atom.name:atom.index for atom in residue.atoms() if atom.name in ["N", "CA", "C", "CB"]}
        vectors = pos[[atom_indices[i] for i in ["N", "C", "CB"]]] - pos[atom_indices["CA"]]

        if np.linalg.det(vectors) < 0:
            return False
    return True


def clash_check(topology, positions):
    heavies = [x for x in topology.atoms() if x.element.symbol != "H"]
    pos = np.array(positions.value_in_unit(LENGTH))[[x.index for x in heavies]]

    tree = spatial.KDTree(pos)
    pairs = tree.query_pairs(r=max(cutoffs.values()))

    for pair in pairs:
        atom_i, atom_j = heavies[pair[0]], heavies[pair[1]]

        if atom_i.residue.index == atom_j.residue.index:
            continue
        elif (atom_i.name == "C" and atom_j.name == "N") or (atom_i.name == "N" and atom_j.name == "C"):
            continue

        atom_distance = np.linalg.norm(pos[pair[0]] - pos[pair[1]])
            
        if (atom_i.name == "SG" and atom_j.name == "SG") and atom_distance > 1.88:
            continue

        elif atom_distance < (cutoffs[atom_i.element.symbol + atom_j.element.symbol]):
            return False
    return True


def strained_sidechain_bonds_check(topology, positions):
    atoms = list(topology.atoms())
    pos = np.array(positions.value_in_unit(LENGTH))
    
    system = forcefield.createSystem(topology)
    bonds = [x for x in system.getForces() if type(x).__name__ == "HarmonicBondForce"][0]
    
    # Initialise arrays for bond details
    n_bonds = bonds.getNumBonds()
    i = np.empty(n_bonds, dtype=int)
    j = np.empty(n_bonds, dtype=int)
    k = np.empty(n_bonds)
    x0 = np.empty(n_bonds)
    
    # Extract bond details to arrays
    for n in range(n_bonds):
        i[n],j[n],_x0,_k = bonds.getBondParameters(n)
        k[n] = _k.value_in_unit(spring_unit)
        x0[n] = _x0.value_in_unit(LENGTH)
        
    # Check if there are any abnormally strained bond
    distance = np.linalg.norm(pos[i] - pos[j], axis=-1)
    check = k*(distance - x0)**2 > 100
    
    # Return residues with strained bonds if any
    return [atoms[x].residue for x in i[check]]


def strained_sidechain_bonds_fixer(strained_residues, topology, positions, n_threads=-1):
    # Delete all atoms except the main chain for badly refined residues.
    bb_atoms = ["N","CA","C"]
    bad_side_chains = sum([[atom for atom in residue.atoms() if atom.name not in bb_atoms] for residue in strained_residues],[])
    modeller = app.Modeller(topology, positions)
    modeller.delete(bad_side_chains)
    
    # Save model with deleted side chains to temporary file.
    random_number = str(int(np.random.rand()*10**8))
    tmp_file = f"side_chain_fix_tmp_{random_number}.pdb"
    with open(tmp_file,"w") as handle:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, handle, keepIds=True)
        
    # Load model into pdbfixer
    fixer = pdbfixer.PDBFixer(tmp_file)
    os.remove(tmp_file)
    
    # Repair deleted side chains 
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    # Fill in the gaps with OpenMM Modeller
    modeller = app.Modeller(fixer.topology, fixer.positions)
    modeller.addHydrogens(forcefield)

    # Set up force field
    system = forcefield.createSystem(modeller.topology)

    # Set up integrator
    integrator = LangevinIntegrator(0, 0.01, 0.0)

    # Set up the simulation
    if n_threads > 0:
        # Set number of threads used by OpenMM
        platform = Platform.getPlatformByName('CPU')
        simulation = app.Simulation(modeller.topology, system, integrator, platform, {'Threads': str(n_threads)})
    else:
        simulation = app.Simulation(modeller.topology, system, integrator)    
    simulation.context.setPositions(modeller.positions)

    # Minimize the energy
    simulation.minimizeEnergy()
    
    return simulation.topology, simulation.context.getState(getPositions=True).getPositions()
