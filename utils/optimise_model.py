#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 11:38:46 2021

@author: rokas
"""

from utils.utils import copy_weights


def optimise_model(shared_model, local_model, loss, lock):
    
    # Delete the gradients
    [o.zero_grad() for o in local_model.optimisers]
    
    # Compute gradients
    [l.backward() for l in loss]
    
    # Step in the model
    [o.step() for o in local_model.optimisers]

    # The critical section begins
    lock.acquire()
    [copy_weights(s_model, l_model,True) for s_model,l_model in zip(shared_model.models.values(),local_model.models.values())]
    lock.release()
    # The critical section ends




