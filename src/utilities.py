#!/usr/bin/python
# -*- coding: utf-8 -*-
#**************************************************************************
#
# $Id: utilities.py $
# $Revision: v01 $
# $Author: epichler $
# $Date: 2014-03-07 $
# Type: Python program.
#
# Comments: Module defining generic utilities functions.
#
# Usage:
# Import this file from other Python files.
#
# To do:
#   - 
#   - 
#
# Copyright © 2013-2014 Elgar E. Pichler & Avi M. Shapiro. All rights
# reserved.
#
#**************************************************************************


#**************************************************************************
#
# Modifications:
# 2014/03/07    E. Pichler      v01
#   - initial version
#
#**************************************************************************


#--------------------------------------------------------------------------

# ***** preliminaries *****

# --- import modules and packages ---
from __future__ import (
     division,
     print_function,
     )
import math
import numpy as np

# --- defaults ---


# ***** function definitions *****

# extended range function
def erange(start, stop=None, step=1, endpoint=False, ndigit=6, dtype=None):
    """
    Return a list of evenly spaced values, separated by step, within a
    given interval.
    This function defaults to the standard library range() function, but
    will also work with float arguments for start, stop, and step.
    Values are generated within the interval [start, stop) if endpoint
    is False, and within the interval [start, stop] if endpoint is True.
    Float values are compared to stop with precision ndigit.
    Values can forcibly be type-cast with dtype.
    """
    if stop is None:
        stop = start
        start = 0
    #print(start, stop, step, endpoint, ndigit, dtype)
    if dtype is None and type(start) is int and type(stop) is int and type(step) is int:
        dtype = int
    r = list(np.arange(start, stop + step / 2, step, dtype=dtype))
    stop_check = round(abs(stop), ndigit)
    if r and round(abs(r[-1]), ndigit) > stop_check:
        r.pop()
    if r and endpoint is False and round(abs(r[-1]), ndigit) == stop_check:
        r.pop()
    return r

#--------------------------------------------------------------------------





