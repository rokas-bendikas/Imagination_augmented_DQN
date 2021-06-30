#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 11:38:46 2021

@author: rokas
"""

from utils.utils import copy_weights


def optimise_model(shared_model, local_model, loss, lock):

    # Delete the gradients
    [o.zero_grad() for o in local_model.optimisers.values()]

    # Compute gradients
    [l.backward() for l in loss.values()]
    
    # Step in the model
    [o.step() for o in local_model.optimisers.values()]

    # The critical section begins
    lock.acquire()
    shared_model._copy_from_model(local_model)
    lock.release()
    # The critical section ends
