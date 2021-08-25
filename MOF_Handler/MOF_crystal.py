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
"""

import numpy as np
from gemmi import cif  #pylint: disable-msg=no-name-in-module


class MOF_crystal:
    """ Class for P 1 crystalline material """

    # pylint: disable-msg=too-many-arguments
    def __init__(self,
                 box=None,
                 atom_symbols=None,
                 ratoms=None,
                 charges=None,
                 atom_labels=None):
        """ Default Constructor: pass arguments for unit cell, atoms, positions,
            and charges directly to class object"""
        self.box = box
        self.atom_symbols = atom_symbols  # _atom_site_type_symbol from CIF standard
        self.atom_labels = atom_labels  # _atom_site_label from CIF standard
        self.ratoms = ratoms
        self.charges = charges
        self.forcefield = []
        self.LJ_params = []

        # Ensure that lists are nparrays
        if not isinstance(self.box, np.ndarray):
            self.box = np.array(self.box)
        if not isinstance(self.ratoms, np.ndarray):
            self.ratoms = np.array(self.ratoms)

    @classmethod
    def from_CIF(cls, file):
        #pylint: disable-msg=unnecessary-comprehension
        """ Construct object from CIF file  """
        cif_data = cif.read(file).sole_block()
        # Unit Cell Parameters
        box = np.array([
            float(cif_data.find_values('_cell_length_a')[0]),
            float(cif_data.find_values('_cell_length_b')[0]),
            float(cif_data.find_values('_cell_length_c')[0])
        ])
        # Atom Symbols [e.g., Carbon=C, Oxygen=O, etc.]
        atom_symbols = [
            x for x in cif_data.find_loop('_atom_site_type_symbol')
        ]
        # Atom Labels [i.e., the specific type of the atom]
        atom_labels = [x for x in cif_data.find_loop('_atom_site_label')]
        # Atom Positions
        x = np.array(cif_data.find_loop('_atom_site_fract_x'),
                     dtype=float) * box[0]
        y = np.array(cif_data.find_loop('_atom_site_fract_y'),
                     dtype=float) * box[1]
        z = np.array(cif_data.find_loop('_atom_site_fract_z'),
                     dtype=float) * box[2]
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
                   atom_symbols=atom_symbols,
                   atom_labels=atom_labels,
                   ratoms=ratoms,
                   charges=charges)

    @classmethod
    def from_XYZ(cls, file, box):
        """ Construct object from XYZ file plus (orthorhombic) unit cell parameters """
        with open(file, mode='r') as f:
            lines = f.read().splitlines()
        atom_symbols = []
        ratoms = []
        charges = []
        for line in lines[2:]:
            entries = line.split()
            atom_symbols.append(entries[0])
            ratoms.append([float(x) for x in entries[1:4]])
        ratoms = np.array(ratoms)
        box = np.array(box)
        # atom_labels????
        return cls(box=box,
                   atom_symbols=atom_symbols,
                   ratoms=ratoms,
                   charges=charges)

    def to_XYZ(self, outfile):
        """ Write the MOF as a cartesian XYZ file """
        with open(outfile, mode='w') as f:
            f.write(str(len(self.ratoms)) + '\n')
            f.write('\n')
            for atom, pos in zip(self.atom_symbols, self.ratoms):
                f.write(atom)
                for idim in range(3):
                    f.write(' ' + str(pos[idim]))
                f.write('\n')

    def to_CIF(self, outfile):
        """ Write the MOF as a simplified, minimal CIF file """
        d = cif.Document()
        d.add_new_block(outfile.replace('.cif', ''))
        block = d.sole_block()
        block.set_pair('_symmetry_space_group_name_H-M', "'P 1'")
        block.set_pair('_symmetry_Int_Tables_number', str(1))
        block.set_pair('_cell_length_a', str(self.box[0]))
        block.set_pair('_cell_length_b', str(self.box[1]))
        block.set_pair('_cell_length_c', str(self.box[2]))
        block.set_pair('_cell_angle_alpha', str(90.0))
        block.set_pair('_cell_angle_beta', str(90.0))
        block.set_pair('_cell_angle_gamma', str(90.0))
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
            list(
                np.array([x[0] / self.box[0]
                          for x in self.ratoms]).astype(str)),
            list(
                np.array([x[1] / self.box[1]
                          for x in self.ratoms]).astype(str)),
            list(
                np.array([x[2] / self.box[2]
                          for x in self.ratoms]).astype(str))
        ]
        # Charges?
        if len(self.charges) == Natoms:
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
                x[idim] += -rmin - self.box[idim] / 2.

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
