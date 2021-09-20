# -*- coding: utf-8 -*-
# pylint: disable-msg=invalid-name   #because I use snake_case
"""
Provide a class for a P 1 crystalline material with associated methods:
  initialize from raw data
  initialize from CIF
  initialize from XYZ plus unit cell information
  write to simplified CIF
  write to XYZ
  replicate unit cell
  recenter unit cell
  add a Lennard-Jones forcefield
  assign Lennard-Jones parameters to atoms
  neutralize the net charge

  WARNING: This code is currently assuming the Hermann–Mauguin symmetry space group is "P 1"
           No checks are made to confirm the space group type

  Fractional <-> Real Conversions:
    s = np.dot(H_inv,r)
    r = np.dot(H,s)
"""

import numpy as np
from gemmi import cif  #pylint: disable-msg=no-name-in-module
from standard_forcefields import atomic_mass

DEG_TO_RAD = np.pi / 180.


class MOF_crystal:
    """ Class for P 1 crystalline material """

    #pylint: disable-msg=too-many-instance-attributes

    # pylint: disable-msg=too-many-arguments
    def __init__(self,
                 box=None,
                 angles=None,
                 atom_symbols=None,
                 ratoms=None,
                 charges=None,
                 atom_labels=None):
        """ Default Constructor: pass arguments for unit cell, atoms, positions,
            and charges directly to class object"""
        self.box = box
        self.angles = angles
        self.atom_symbols = atom_symbols  # _atom_site_type_symbol from CIF standard
        self.atom_labels = atom_labels  # _atom_site_label from CIF standard
        self.ratoms = ratoms
        self.charges = charges
        self.forcefield = []
        self.LJ_params = []

        #Compute the lattice matrix
        self.H = np.array([
            [
                box[0], box[1] * np.cos(angles[2] * DEG_TO_RAD),
                box[2] * np.cos(angles[1] * DEG_TO_RAD)
            ],
            [
                0.e0, box[1] * np.sin(angles[2] * DEG_TO_RAD),
                box[2] * (np.cos(angles[0] * DEG_TO_RAD) - np.cos(
                    angles[1] * DEG_TO_RAD) * np.cos(angles[2] * DEG_TO_RAD)) /
                np.sin(angles[2] * DEG_TO_RAD)
            ],
            [
                0.e0, 0.e0,
                box[2] * np.sqrt((np.sin(angles[1] * DEG_TO_RAD))**2 -
                                 ((np.cos(angles[0] * DEG_TO_RAD) -
                                   np.cos(angles[1] * DEG_TO_RAD) *
                                   np.cos(angles[2] * DEG_TO_RAD))**2) /
                                 (np.sin(angles[2] * DEG_TO_RAD))**2)
            ]
        ])
        self.Hinv = np.linalg.inv(self.H)

        # Ensure that lists are nparrays
        if not isinstance(self.box, np.ndarray):
            self.box = np.array(self.box)
        if not isinstance(self.ratoms, np.ndarray):
            self.ratoms = np.array(self.ratoms)

    @classmethod
    def from_CIF(cls, file):
        #pylint: disable-msg=unnecessary-comprehension
        #pylint: disable-msg=too-many-locals
        """ Construct object from CIF file  """
        cif_data = cif.read(file).sole_block()
        # Unit Cell Parameters
        box = np.array([
            float(cif_data.find_values('_cell_length_a')[0]),
            float(cif_data.find_values('_cell_length_b')[0]),
            float(cif_data.find_values('_cell_length_c')[0])
        ])
        angles = np.array([
            float(cif_data.find_values('_cell_angle_alpha')[0]),
            float(cif_data.find_values('_cell_angle_beta')[0]),
            float(cif_data.find_values('_cell_angle_gamma')[0])
        ])
        # Atom Symbols [e.g., Carbon=C, Oxygen=O, etc.]
        atom_symbols = [
            x for x in cif_data.find_loop('_atom_site_type_symbol')
        ]
        # Atom Labels [i.e., the specific type of the atom]
        atom_labels = [x for x in cif_data.find_loop('_atom_site_label')]
        # Atom Positions
        x = np.array(cif_data.find_loop('_atom_site_fract_x'), dtype=float)
        y = np.array(cif_data.find_loop('_atom_site_fract_y'), dtype=float)
        z = np.array(cif_data.find_loop('_atom_site_fract_z'), dtype=float)
        # Stored as fractional coordinates
        ratoms = np.array([[xi, yi, zi] for xi, yi, zi in zip(x, y, z)])
        # Basic validation
        Nsymbols = len(atom_symbols)
        Nlabels = len(atom_labels)
        if Nsymbols == Nlabels:
            Natoms = Nsymbols  #easy
        elif Nsymbols == 0 or Nlabels == 0:
            Natoms = max(Nsymbols, Nlabels)
        else:
            raise Exception(
                'ERROR: Some mismatch in _atom_site_label and _atom_site_type_symbol columns'
            )
        if len(ratoms) != Natoms:
            raise Exception(
                'ERROR: mismatch in length of atom and position vectors')
        # Point Charges
        charges = np.array(cif_data.find_loop('_atom_site_charge'),
                           dtype=float)
        if len(charges) == Natoms:
            # Check Charge
            qsum = sum(charges)
            if np.abs(qsum) > 1.e-4:
                print('Warning: Net Charge is ' + str(qsum) +
                      ' before neutralization')
            # Neutralize the structure
            charges = charges - qsum / len(charges)
        else:
            charges = []
        return cls(box=box,
                   angles=angles,
                   atom_symbols=atom_symbols,
                   atom_labels=atom_labels,
                   ratoms=ratoms,
                   charges=charges)

    @classmethod
    def from_XYZ(cls, file, box, angles):
        """ Construct object from XYZ file plus (orthorhombic) unit cell parameters """
        with open(file, mode='r') as f:
            lines = f.read().splitlines()
        atom_symbols = []
        ratoms = []
        charges = []
        # What to do: build the lattice matrix? or do something fancy later?
        for line in lines[2:]:
            entries = line.split()
            atom_symbols.append(entries[0])
            real_pos = [float(x) for x in entries[1:4]]
            #frac_pos = np.dot(H_inv,real_pos)
            ratoms.append(real_pos)
        ratoms = np.array(ratoms)
        box = np.array(box)
        return cls(box=box,
                   angles=angles,
                   atom_symbols=atom_symbols,
                   ratoms=ratoms,
                   charges=charges)

    def to_XYZ(self, outfile):
        """ Write the MOF as a cartesian XYZ file """
        with open(outfile, mode='w') as f:
            f.write(str(len(self.ratoms)) + '\n')
            f.write('\n')
            for atom, frac_pos in zip(self.atom_symbols, self.ratoms):
                pos = np.dot(self.H, frac_pos)
                f.write(atom)
                for idim in range(3):
                    f.write(' ' + str(pos[idim]))
                f.write('\n')

    def to_CIF(self, outfile, write_charges=True):
        """ Write the MOF as a simplified, minimal CIF file """
        d = cif.Document()
        d.add_new_block(outfile.replace('.cif', ''))
        block = d.sole_block()
        block.set_pair('_symmetry_space_group_name_H-M', "'P 1'")
        block.set_pair('_symmetry_Int_Tables_number', str(1))
        block.set_pair('_cell_length_a', str(self.box[0]))
        block.set_pair('_cell_length_b', str(self.box[1]))
        block.set_pair('_cell_length_c', str(self.box[2]))
        block.set_pair('_cell_angle_alpha', str(self.angles[0]))
        block.set_pair('_cell_angle_beta', str(self.angles[1]))
        block.set_pair('_cell_angle_gamma', str(self.angles[2]))
        volume = self.box[0] * self.box[1] * self.box[2]
        block.set_pair('_cell_volume', str(volume))
        # Build the _atom_site loop
        Natoms = len(self.ratoms)
        columns = []
        loop_data = []
        # Atom Labels (specific atom types)
        if len(self.atom_labels) == Natoms:
            columns.append('label')
            loop_data.append(np.array(self.atom_labels).astype(str))
        # Atom Symbol
        if len(self.atom_symbols) == Natoms:
            columns.append('type_symbol')
            loop_data.append(np.array(self.atom_symbols).astype(str))
        else:
            columns.append('type_symbol')
            loop_data.append(np.array(self.atom_labels).astype(str))
        # Atom Position
        columns += ['fract_x', 'fract_y', 'fract_z']
        loop_data += [
            list(np.array([x[0] for x in self.ratoms]).astype(str)),
            list(np.array([x[1] for x in self.ratoms]).astype(str)),
            list(np.array([x[2] for x in self.ratoms]).astype(str))
        ]
        # Charges?
        if len(self.charges) == Natoms and write_charges:
            columns.append('charge')
            loop_data.append(list(np.array(self.charges).astype(str)))
        loop_pos = block.init_loop('_atom_site_', columns)
        loop_pos.set_all_values(loop_data)
        d.write_file(outfile)

    def neutralize(self):
        """ Adjust charges to neutralize the structure """
        qsum = sum(self.charges)
        if np.abs(qsum) > 1.e-4:
            print('Warning: Net Charge is ' + str(qsum) +
                  ' before neutralization')
        # Neutralize the structure
        self.charges = self.charges - qsum / len(self.charges)

    def recenter(self):
        """ Recenter the atom positions """
        for idim in range(3):
            rmin = min([x[idim] for x in self.ratoms])
            for x in self.ratoms:
                x[idim] += -rmin - 1. / 2.

    def add_LJ_forcefield(self, input_dict):
        """ Import a dictionary of atom types with associated Lennard-Jones parameters """
        if len(self.forcefield) == 0:
            self.forcefield = input_dict
        else:
            self.forcefield += input_dict

    def assign_LJ_params(self, mapping):
        """
           Assign LJ parameters to each atom based on forcefield dictionary

           mapping: string, either 'atom_symbol' or 'atom_label'
                    identifies the class property for mapping
                    to LJ types
        """
        if mapping == 'atom_symbol':
            atom_loop = self.atom_symbols
        elif mapping == 'atom_label':
            atom_loop = self.atom_labels
        else:
            raise Exception('Unknown mapping:', mapping)
        LJ_params = []
        params_fixed = True
        for atom in atom_loop:
            param_fixed = False
            for site_type in self.forcefield:
                if site_type[mapping] == atom:
                    LJ_params.append({
                        'sigma': site_type['sigma'],
                        'epsilon': site_type['epsilon']
                    })
                    param_fixed = True
            if param_fixed is False:
                params_fixed = False
                print('Warning: no match for', atom, 'by', mapping)
        if params_fixed:
            print('LJ Parameters set successfully')
            self.LJ_params = LJ_params
        else:
            raise Exception('ERROR: unable to match some atoms')

    def replicate(self, reps):
        """
           Replicate the unit cell based on reps specification

           reps: list of 3 integers
        """
        #pylint: disable-msg=too-many-branches
        #pylint: disable-msg=too-many-locals
        # Check types
        for irep in reps:
            if not isinstance(irep, int):
                raise Exception('ERROR in replicate list', reps)
        # Lazily return original object if reps = [1,1,1]
        if reps == [1, 1, 1]:
            return self
        # To continue, we need box dimensions
        if len(self.box) == 3:
            pass
        elif self.box is None or self.box == []:
            raise Exception(
                'ERROR: Cannot replicate cell without box specification')
        # Replicate the atoms and positions
        box = [self.box[idim] * float(reps[idim]) for idim in range(3)]
        angles = self.angles
        atom_symbols = []
        atom_labels = []
        ratoms = []
        charges = []
        Natoms = len(self.ratoms)
        # Replicate charges?
        if len(self.charges) == Natoms:
            copy_charges = True
            print('Copying Charges: ', True)
        elif len(self.charges) == 0:
            copy_charges = False
            print('Copying Charges: ', False)
        else:
            raise Exception('ERROR: list of charges does not match atom count')
        copy_symbols = bool(len(self.atom_symbols) > 0)
        copy_labels = bool(len(self.atom_labels) > 0)
        # Replicate in positive octants [can re-center later]
        for xrep in range(reps[0]):
            for yrep in range(reps[1]):
                for zrep in range(reps[2]):
                    irep = [xrep, yrep, zrep]
                    for iatom in range(Natoms):
                        if copy_symbols:
                            atom_symbols.append(self.atom_symbols[iatom])
                        if copy_labels:
                            atom_labels.append(self.atom_labels[iatom])
                        position = [
                            self.ratoms[iatom][idim] +
                            float(irep[idim] * self.box[idim])
                            for idim in range(3)
                        ]
                        ratoms.append(position)
                        if copy_charges:
                            charges.append(self.charges[iatom])
        return MOF_crystal(box=box,
                           angles=angles,
                           atom_symbols=atom_symbols,
                           atom_labels=atom_labels,
                           ratoms=ratoms,
                           charges=charges)

    def cell_weight(self):
        """
        Compute the molecular weight of the MOF cell
        Units are atomic mass units
        """

        total_mass = 0.
        for atom in self.atom_symbols:
            try:
                total_mass += atomic_mass[atom]
            except Exception as e:
                raise Exception('Unknown atom symbol', atom) from e
        return total_mass
