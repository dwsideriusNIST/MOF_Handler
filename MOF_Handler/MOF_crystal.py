# -*- coding: utf-8 -*-
# pylint: disable-msg=invalid-name   #because I use snake_case
"""
Provide a class for a P-1 crystalline material with associated methods:
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
"""

import numpy as np
#from gemmi import cif


class MOF_crystal:
    """ Class for P-1 crystalline material """

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

        # Ensure that lists are nparrays
        if isinstance(self.box) != np.ndarray:
            self.box = np.array(self.box)
        if isinstance(self.ratoms) != np.ndarray:
            self.ratoms = np.array(self.ratoms)

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
