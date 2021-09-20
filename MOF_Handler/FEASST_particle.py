# -*- coding: utf-8 -*-
# pylint: disable-msg=invalid-name   #because I use snake_case
# pylint: disable-msg=too-many-locals
""" Function to generate a FEASST particle file """

import numpy as np


def FEASST_particle(mof, cutoff=10.0):
    """
       Generate a FEASST particle file (as a string)
       from an input mof_crystal object
    """
    # Generate the list of site types
    site_types = []
    atom_site_id = []

    # atom_symbols vs atom_labels
    for atom, LJ_params, charge in zip(mof.atom_labels, mof.LJ_params,
                                       mof.charges):
        #print(atom, LJ_params, charge)
        site_params = {
            'atom_label': atom,
            'sigma': LJ_params['sigma'],
            'epsilon': LJ_params['epsilon'],
            'charge': charge
        }
        if site_params not in site_types:
            #print(site_params)
            site_types.append(site_params)
    print(len(site_types), 'site types')
    # Number the site types
    counter = 0
    for site in site_types:
        site['id'] = counter
        counter += 1
    # Assign site ids to each atom
    counter = 0
    for atom, LJ_params, charge in zip(mof.atom_labels, mof.LJ_params,
                                       mof.charges):
        site_params = {
            'atom_label': atom,
            'sigma': LJ_params['sigma'],
            'epsilon': LJ_params['epsilon'],
            'charge': charge
        }
        #print('atom', site_params)
        for site in site_types:
            site_stub = {
                'atom_label': site['atom_label'],
                'sigma': site['sigma'],
                'epsilon': site['epsilon'],
                'charge': site['charge']
            }
            #print('match', site_stub)
            if site_stub == site_params:
                counter += 1
                atom_site_id.append(site['id'])
                break
        #break
    print(len(atom_site_id), 'assigned site ids')

    particle = '# LAMMPS-inspired data file\n\n'
    particle += str(len(mof.ratoms)) + ' sites\n'
    particle += str(len(site_types)) + ' site types\n\n'
    particle += 'Units\n\n'
    particle += 'length Angstrom\nenergy kJ/mol\ncharge elementary\n\n'
    particle += 'Site Labels\n\n'
    for index, site in enumerate(site_types):
        particle += str(index) + ' ' + site['atom_label'] + '\n'
    particle += '\nSite Properties\n\n'
    for index, site in enumerate(site_types):
        particle += str(index) + ' sigma ' + str(site['sigma'])
        particle += ' epsilon ' + str(site['epsilon'])
        particle += ' cutoff ' + str(cutoff)
        particle += ' charge ' + str(site['charge'])
        particle += '\n'
    particle += '\nSites\n\n'
    for index, frac_pos in enumerate(mof.ratoms):
        pos = np.dot(mof.H, frac_pos)
        site_type = atom_site_id[index]
        particle += str(index) + ' '
        particle += str(site_type) + ' '
        for idim in range(3):
            particle += '{:20.12f}'.format(pos[idim]) + ' '
        particle += '{:20.12f}'.format(mof.charges[index])
        particle += '\n'

    return particle
