# -*- coding: utf-8 -*-
"""Container for functions related to frac <-> real conversions"""
import numpy as np


def lattice_matrix(box, angles):
    """
      Construct the triclinic lattice matrix
    """
    #NOTE: this assumes that the angles vector is already in radians
    h_matrix = np.array(
        [[box[0], box[1] * np.cos(angles[2]), box[2] * np.cos(angles[1])],
         [
             0.e0, box[1] * np.sin(angles[2]), box[2] *
             (np.cos(angles[0]) - np.cos(angles[1]) * np.cos(angles[2])) /
             np.sin(angles[2])
         ],
         [
             0.e0, 0.e0,
             box[2] * np.sqrt((np.sin(angles[1]))**2 -
                              ((np.cos(angles[0]) -
                                np.cos(angles[1]) * np.cos(angles[2]))**2) /
                              (np.sin(angles[2]))**2)
         ]])
    return h_matrix
