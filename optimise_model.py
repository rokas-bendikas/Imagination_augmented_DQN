#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 11:38:46 2021

@author: rokas
"""

from utils import copy_gradients


def optimise_model(shared_model, local_model, loss, optimiser, lock):
    # Compute gradients
    loss.backward()

    # The critical section begins
    lock.acquire()
    copy_gradients(shared_model, local_model)
    optimiser.step()
    lock.release()
    # The critical section ends

    local_model.zero_grad()

    return loss.item()


