#!/usr/bin/python
# -*- coding: utf-8 -*-
#**************************************************************************
#
# $Id: PGG_Simulator.py $
# $Revision: v18 $
# $Author: epichler $
# $Date: 2014-11-07 $
# Type: Python program.
#
# Comments: Runs a Public Goods Game simulator.
#
# Usage:
# Use the `-h' flag for further instructions on how to use this script.
#
# To do:
#   - if simulation is a continuation of a previous run all previous output
#     files have to be parsed and rewritten (the last record of the _ts file
#     has to be recomputed and all of the _pss file has to be recomputed)
#   - possibly introduce error checking for command line arguments, e.g.:
#     - all max values >= respective min values
#     - t_window >= (t_max - t_min)
#
# Copyright © 2012-2014 Elgar E. Pichler & Avi M. Shapiro.
# All rights reserved.
#
#**************************************************************************


#**************************************************************************
#
# Modifications:
# 2014/11/07    E. Pichler      v18
#   - merged again with PGG_Simulator_v02.py and added functionality from
#     that program variant
#   - introduced command line argument program_variant to select program
#     variant
#   - renamed utility container variable program to program_variant
# 2014/10/16    E. Pichler      v17
#   - eliminated computation and recording of powerlaw fit parameters
# 2014/08/07    E. Pichler      v16
#   - added computation and output of number of edges
# 2014/06/23    E. Pichler      v15
#   - added new command line argument t_graph, which is a list containing
#     timesteps at which the PGG graph should be output to a file
#   - moved nx.write_gml() from main program to simulate_pgg() in pgg.py
# 2014/03/07    E. Pichler      v14
#   - in main program changed looping over synergy and greediness so that
#     utilities.erange() is used, thereby hopefully avoiding unexpected
#     behavior because of float representation precision
# 2013/12/19    E. Pichler      v13
#   - set _alpha_delta_default to 0.05
# 2013/10/08    E. Pichler      v12
#   - corrected command line argument parsing
#   - minor cosmetic formatting changes
# 2013/09/16    E. Pichler      v12
#   - introduced or changed definition and usage of n_changers,
#     n_movers, n_unstable_nodes and made them arrays
# 2013/06/27    E. Pichler      v11
#   - changed type of n_cooperators container variable to list to enable
#     computation of steady state value for this variable
# 2013/06/03    E. Pichler      v10
#   - replaced computation of weighted average of a PGG graph's components'
#     average shortest path lengths with computation of a PGG graph's
#     largest component's shortest path length
#   - added corresponding container variables for recording of largest
#     component's shortest path length
# 2013/05/16    E. Pichler      v09
#   - added container variables for recording of
#       - number of PGG graph components
#       - weighted average of a PGG graph's components' average shortest
#         path lengths
#       - a PGG graph's average clustering component
# 2013/04/25    E. Pichler      v08
#   - adapted to move of pgg_v01.py back to pgg.py
# 2013/04/25    E. Pichler      v07
#   - added handling of power law parameters in utility container
# 2013/04/04    E. Pichler      v06
#   - changed (corrected) synergy min, max, delta to reasonable values
# 2013/03/11    E. Pichler      v05
#   - added command line argument debug flag
#   - continued adaptation of code to use of utility container
#   - merged with Avi Shapiro's changes from 20130211
#   - changed to conditional debug output
#   - added t_window command line argument and default value
#   - added initialization to steady state computation variables in container
#   - changed k default to 8 (as in Roca)
# 2013/01/26    E. Pichler      v04
#   - corrected header comments
#   - added command line argument for degree normalization factor
#   - redirected pgg.record_run_summary output to output_file_pss
#   - adapted code to availability of new utility container object
# 2012/12/07    E. Pichler      v03
#   - made use of pgg.py functions explicit
#   - added additional program options
#   - merged with Avi Shapiro's changes
# 2012/11/30	E. Pichler      v02
#   - as an intermediate solution branched Avi's pgg.py to pgg_v01.py and
#     utilized the branched version in this file
#   - added additional program options
# 2012/11/16	E. Pichler
#   - added first interface with Avi's code from
#       PublicGoods.py
#       animatePGG.py
#       pgg.py
#       playPGG.py
# 2012/11/09	E. Pichler
#   - defined additional min/max/delta for parameters
#   - added additional program options
#   - standardized parameter naming conventions
#   - added main loops
#   - added initial API definitions
# 2012/05/25	E. Pichler
#   - added additional program options
#   - introduced standardized output for input parameters and time series
#     recording
# 2012/04/18	E. Pichler      v01
#   - added to do list
# 2012/04/08	E. Pichler
#   - initial version
#
#**************************************************************************

#--------------------------------------------------------------------------

# ***** preliminaries *****

# --- import modules and packages ---
# - system packages
from __future__ import (
     division,
     print_function,
     )
import argparse
#import math
#import networkx as nx
import os
import random as rd
import time
# - project packages
import pgg
import utilities

# --- program specific defaults ---
#  - program name
program_name = os.path.basename(__file__)
#  - debug flag
_debug = False
#  - greediness
_alpha_delta_default = 0.05
_alpha_max_default = 1.0
_alpha_min_default = 0.
#  - graph type
#      choices: 1 ... Erdos-Renyi
#               2 ... Watts-Strogatz
#               3 ... Barabasi-Albert [default]
#               4 ... graph without edges
#               5 ... fully connected graph
_graph_type_default = 3
#  - graph input file
_input_name_g_default = ""
#  - "lattice" degree / degree scaling factor (k=8 in Roca)
_k_default = 8
#  - mu
_mu_default = 0.01
#  - players
_n_cooperators_init_default = 0
_n_players_default = 5000
#  - number of simulations per parameter set
_n_simulations_default = 5
#  - standard deviation of Gaussian noise
_noise_sigma_default = .1
#  - directory for output files
_output_dir = "../data/"
#  - program variant
#      choices: 1 ... partial tanh change probability function and Gaussian
#                     noise term in satisfaction function (Roca_2011-like)
#                     [default] 
#               2 ... smooth tanh change probability function and no noise
#                     term in satisfaction function
_program_variant_default = 1 
#  - stem of output script file
_script_name_stem = ""
#  - synergy
_synergy_delta_default = 0.5
_synergy_max_default = 10.       # should be k+1
_synergy_min_default = 1.
#  - time
_t_delta_default = 1
_t_max_default = 10000
_t_min_default = 0
_t_graph_default = [_t_max_default]
#  - steady state averaging time window
_t_window_default = 100

# --- function definitions ---
# PGG functions should be defined in pgg.py
# return a list of integers mapped from a string
def intlist(s):
    try:
        l = map(int, s.split(','))
        return l
    except:
        raise argparse.ArgumentTypeError("argument must be of type list of integers: i1,i2,i3,...")


# ***** main program *****

# define command line arguments and parser
parser = argparse.ArgumentParser(description='Run a Public Goods Game simulator.')
parser.add_argument('-D', dest='debug',
                    action="store_true", default=_debug,
                    help='debug flag [default: %(default)s]')
parser.add_argument('--alpha_delta', metavar='alpha_delta', dest='alpha_delta',
                    type=float, nargs='?', default=_alpha_delta_default,
                    help='greediness minimum [default: %(default)s]')
parser.add_argument('--alpha_max', metavar='alpha_max', dest='alpha_max',
                    type=float, nargs='?', default=_alpha_max_default,
                    help='greediness maximum [default: %(default)s]')
parser.add_argument('--alpha_min', metavar='alpha_min', dest='alpha_min',
                    type=float, nargs='?', default=_alpha_min_default,
                    help='greediness minimum [default: %(default)s]')
parser.add_argument('--graph_type', metavar='initial graph_type', dest='graph_type',
                    type=int, nargs='?', default=_graph_type_default,
                    help='graph types: 1 ... Erdos-Renyi, 2 ... Watts-Strogatz, 3 ... Barabasi-Albert, 4 ... graph without edges, 5 ... fully connected graph [default: %(default)s]')
parser.add_argument('--input_name_g', metavar='input_name_g', dest='input_name_g',
                    type=str, nargs='?', default=_input_name_g_default,
                    help='graph input file [default: %(default)s]')
parser.add_argument('--k', metavar='k', dest='k',
                    type=float, nargs='?', default=_k_default,
                    help='degree (scaling factor) / average degree [default: %(default)s]')
parser.add_argument('--mu', metavar='mu', dest='mu',
                    type=float, nargs='?', default=_mu_default,
                    help='memory loss parameter [default: %(default)s]')
parser.add_argument('--n_cooperators_init', metavar='n_cooperators_init', dest='n_cooperators_init',
                    type=int, nargs='?', default=_n_cooperators_init_default,
                    help='total number of initial cooperators [default: %(default)s]')
parser.add_argument('--n_players', metavar='n_players', dest='n_players',
                    type=int, nargs='?', default=_n_players_default,
                    help='total number of players [default: %(default)s]')
parser.add_argument('--n_simulations', metavar='n_simulations', dest='n_simulations',
                    type=int, nargs='?', default=_n_simulations_default,
                    help='number of simulations for given parameter set [default: %(default)s]')
parser.add_argument('--noise_sigma', metavar='noise_sigma', dest='noise_sigma',
                    type=float, nargs='?', default=_noise_sigma_default,
                    help='standard deviation of Gaussian noise [default: %(default)s]')
parser.add_argument('--output_dir', metavar='output_dir', dest='output_dir',
                    type=str, nargs='?', default=_output_dir,
                    help='directory for output files [default: %(default)s]')
parser.add_argument('--program_variant', metavar='program_variant', dest='program_variant',
                    type=int, nargs='?', default=_program_variant_default,
                    help='program variant: 1 ... partial tanh change probability function and Gaussian noise term in satisfaction function, 2 ... smooth tanh change probability function and no noise term in satisfaction function [default: %(default)s]')
parser.add_argument('--synergy_delta', metavar='synergy_delta', dest='synergy_delta',
                    type=float, nargs='?', default=_synergy_delta_default,
                    help='synergy factor delta [default: %(default)s]')
parser.add_argument('--synergy_max', metavar='synergy_max', dest='synergy_max',
                    type=float, nargs='?', default=_synergy_max_default,
                    help='synergy factor maximum [default: %(default)s]')
parser.add_argument('--synergy_min', metavar='synergy_min', dest='synergy_min',
                    type=float, nargs='?', default=_synergy_min_default,
                    help='synergy factor minimum [default: %(default)s]')
parser.add_argument('--t_graph', metavar='t_graph', dest='t_graph',
                    type=intlist, nargs='?', default=_t_graph_default,
                    help='list of times at which PGG graph is output [default: %(default)s]')
parser.add_argument('--t_max', metavar='t_max', dest='t_max',
                    type=int, nargs='?', default=_t_max_default,
                    help='number of generations to simulate game [default: %(default)s]')
parser.add_argument('--t_min', metavar='t_min', dest='t_min',
                    type=int, nargs='?', default=_t_min_default,
                    help='initial generation [default: %(default)s]')
parser.add_argument('--t_window', metavar='t_window', dest='t_window',
                    type=int, nargs='?', default=_t_window_default,
                    help='steady state averaging time window [default: %(default)s]')

# process command line arguments
args = parser.parse_args()
C = pgg.utility_container()
C.alpha_delta = args.alpha_delta
C.alpha_max = args.alpha_max
C.alpha_min = args.alpha_min
C.debug = args.debug
C.graph_type = args.graph_type
C.input_name_g = args.input_name_g
C.k = args.k
C.mu = args.mu
C.n_cooperators_init = args.n_cooperators_init
C.n_players = args.n_players
C.n_simulations = args.n_simulations
C.noise_sigma = args.noise_sigma
C.output_dir = args.output_dir
C.program_variant = args.program_variant
C.synergy_delta = args.synergy_delta
C.synergy_max = args.synergy_max
C.synergy_min = args.synergy_min
C.t_graph = args.t_graph
C.t_max = args.t_max
C.t_min = args.t_min
C.t_window = args.t_window
# initialize the other utility container variables
C.aspiration_average = [0]*C.t_window
C.aspiration_max = [0]*C.t_window
C.aspiration_min = [0]*C.t_window
C.clustering_coefficient_average = [0]*C.t_window
C.component_path_length_average = [0]*C.t_window
C.degree_average = [0]*C.t_window
C.degree_max = [0]*C.t_window
C.degree_min = [0]*C.t_window
C.largest_component_path_length = [0]*C.t_window
C.n_changers = [0]*C.t_window
C.n_components = [0]*C.t_window
C.n_cooperators = [0]*C.t_window
C.n_edges = [0]*C.t_window
C.n_edges_CC = [0]*C.t_window
C.n_edges_CD = [0]*C.t_window
C.n_edges_DD = [0]*C.t_window
C.n_largest_component = [0]*C.t_window
C.n_movers = [0]*C.t_window
C.n_unstable_nodes = [0]*C.t_window
C.payoff_average = [0]*C.t_window
C.payoff_max = [0]*C.t_window
C.payoff_min = [0]*C.t_window
#C.powerlaw_C = [0]*C.t_window
#C.powerlaw_gamma = [0]*C.t_window
C.satisfaction_average = [0]*C.t_window
C.satisfaction_max = [0]*C.t_window
C.satisfaction_min = [0]*C.t_window
C.t_delta = _t_delta_default

# debug mode check
if C.debug:
    print("alpha_delta =", C.alpha_delta)
    print("alpha_max =", C.alpha_max)
    print("alpha_min =", C.alpha_min)
    print("debug =", C.debug)
    print("graph_type =", C.graph_type)
    print("input_name_g =", C.input_name_g)
    print("k =", C.k)
    print("mu =", C.mu)
    print("n_cooperators_init =", C.n_cooperators_init)
    print("n_players =", C.n_players)
    print("n_simulations =", C.n_simulations)
    print("noise_sigma =", C.noise_sigma)
    print("output_dir =", C.output_dir)
    print("program_variant =", C.program_variant)
    print("synergy_delta =", C.synergy_delta)
    print("synergy_max =", C.synergy_max)
    print("synergy_min =", C.synergy_min)
    print("t_delta =", C.t_delta)
    print("t_graph =", C.t_graph)
    print("t_max =", C.t_max)
    print("t_min =", C.t_min)
    print("t_window =", C.t_window)

# execute the main program
if C.debug:
    print("----- ", program_name, ": start ... -----", sep="")
    #exit()

# initialize random number generator
rd.seed()

# begin of loop over synergy values
for C.synergy in utilities.erange(C.synergy_min, C.synergy_max, C.synergy_delta, True):
    # begin of loop over greediness values
    for C.alpha in utilities.erange(C.alpha_min, C.alpha_max, C.alpha_delta, True):
        # begin of loop over number of simulations
        for i in range(C.n_simulations):
            # get execution time stamp and output file names
            C.time_stamp = time.strftime("%Y%m%d%H%M%S", time.gmtime())
            C.index = C.time_stamp + "_%06d" % rd.randint(0, 999999)
            C.stem = C.output_dir + C.index
            C.output_name_ts = C.stem + "_ts.csv"     # output for time series
            C.output_name_pss = C.stem + "_pss.csv"   # output for parameters and steady state values
            # generate graph
            G = pgg.create_pgg_graph(C)
            # simulate PGG; loop over time
            pgg.simulate_pgg(G, C)
            # record run summary
            pgg.record_run_summary(G, C)
        # end of loop over number of simulations
    # end of loop over greediness values
# end of loop over synergy values

if C.debug:
    print("----- ", program_name, ": ... end -----", sep="")

#--------------------------------------------------------------------------
