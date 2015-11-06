#!/usr/bin/python
# -*- coding: utf-8 -*-
#**************************************************************************
#
# $Id: pgg.py $
# $Revision: v28 $
# $Author: epichler $
# $Date: 2014-11-07 $
# Type: Python module.
#
# Comments: Module defining Public Goods Game functions.
#
# Usage:
# Import this file from other Python files.
#
# To do:
#   - clarify issues on Ubuntu/Eclipse with
#       - import scipy as sp
#         from scipy import optimize
#         sp.optimize.leastsq()
#   - integrate use of Clausen_2009b algorithm for power law check and parameter fit
#   - implement steady state check in is_steady_state
#   - define read_pgg_graph
#     note that read_pgg_graph currently reads the PGG graph from a file but
#     initializes subsequently, clearly not what we want;
#     to simplify things, it might make sense to abandon the idea behind
#     read_pgg_graph; in the current form it is not functional;
#     note also that the previous and the new tss files would overlap on
#     the last and first record respectively
#   - why k = 4 and p = 0.4 for watts-strogatz
#   - why p = 0.4 for erdos-renyi
#   - make sure that nx.read_gml and nx.write_gml handle arbitrary graph
#     attributes
#
# Copyright © 2012-2014 Elgar E. Pichler & Avi M. Shapiro.
# All rights reserved.
#
#**************************************************************************

#**************************************************************************
#
# Modifications:
# 2014/11/07    E. Pichler      v28
#   - merged again with pgg_v02.py and added functionality from that
#     program variant
#   - renamed utility container variable program to program_variant
# 2014/10/16    E. Pichler      v27
#   - adapted changes from pgg_v02.py:
#     - added program to utility container
#     - added recording of C.k and C.program to record_run_summary()
#   - eliminated computation and recording of powerlaw fit parameters in
#       degree_pdf_fit()
#       record_graph_state()
#       record_run_summary()
# 2014/10/02    E. Pichler      v26
#   - added comments for various graph types in create_pgg_graph()
# 2014/09/14    A. Shapiro      v25
#   - added C.graph_type == 4, C.graph_type == 5 options in
#     create_pgg_graph() for empty and complete graphs, respectively
# 2014/09/01    E. Pichler      v24
#   - added explicit C.graph_type == 3 option in create_pgg_graph()
# 2014/08/07    E. Pichler      v23
#   - corrected computation of potential_neighbors in
#     accumulate_edge_changes() so that potential_neighbors does not
#     include self
#   - added computation and output of number of edges
#   - simplified argument list for record_graph_state()
# 2014/06/23    E. Pichler      v22
#   - moved conditional output of PGG graph via nx.write_gml() from main
#     program in PGG_Simulator to simulate_pgg()
# 2014/03/19    E. Pichler      v21
#   - to avoid output of irrelevant precision digits and to enable
#     correct filtering and binning in subsequent data processing
#     formatting specifications in print() statements were added in
#       record_graph_state()
#       record_run_summary()
# 2013/12/18    E. Pichler      v20
#   - disabled computation of average shortest path length for largest
#     component (because this is by far the computationally most expensive
#     computation
# 2013/10/08    E. Pichler      v19
#   - deleted most commented-out old code
#   - changed grouping and ordering of output columns in _ts and _pss files
#     to reflect temporal order of evaluation of variables (first PGG
#     topology description, then PGG players description, and then
#     impending update description
#   - normalize_edge_changes now follows after the accumulation of all edge
#     changes in compute_state_variables
#   - code from make_edge_changes has been folded into
#     update_player_connectivity; make_edge_changes has been deleted
#   - code from update_playoff_min and update_payoff_max has been folded
#     into update_playoff_min_max; those two functions have been deleted
#   - the dummy function change_connectivity has been deleted
#   - code from change_strategy has been folded into
#     update_player_strategies; change_strategy has been deleted
# 2013/09/26    E. Pichler      v18
#   - changed recording of values (state variables at time t now include all
#     computed values for that t)
#   - the above change necessitates
#     - the creation of the new functions
#         accumulate_strategy_changes
#         compute_state_variables
#         update_payoff_min_max
#         update_player_connectivities
#         update_player_strategies
#     - the deletion of the functions
#         pre_update_reset
#     - changes to the functions
#         initialize_pgg_graph
#         normalize_edge_changes
#         record_graph_state
#         simulate_pgg
#         update_graph_state
#     - and a reordering of function calling sequences
# 2013/09/16    E. Pichler      v17
#   - introduced or changed definition and usage of n_changers,
#     n_movers, n_unstable_nodes and made them arrays
# 2013/08/12    E. Pichler      v16
#   - added output of hostname and noise_sigma in run summary file
# 2013/06/27    A. Shapiro, E. Pichler      v15
#   - added correct updating and recording of n_cooperators variable
#   - changed type of n_cooperators container variable to list to enable
#     computation of steady state value for this variable
# 2013/06/27    A. Shapiro      v14
#   - powerlaw module now used to compute degree distribution fit in degree_pdf_fit()
#   - no longer storing power law normalization constant, just exponent
# 2013/06/27    E. Pichler      v13
#   - replaced len(G) computation of number of nodes in a graph with
#     corresponding and faster library call G.number_of_nodes()
#   - minimized largest_component_average_shortest_pathlength() and
#     average_shortest_component_pathlength() argument lists
# 2013/06/20    A. Shapiro      v12
#   - replaced average_shortest_component_pathlength even though we aren't
#     using it
# 2013/06/03    E. Pichler      v11
#   - replaced computation of weighted average of a PGG graph's components'
#     average shortest path lengths with computation of a PGG graph's
#     largest component's shortest path length
#   - added corresponding container variables for recording of largest
#     component's shortest path length
# 2013/05/16    E. Pichler      v10
#   - added code for
#       - average_shortest_component_pathlength for computation of a
#         weighted average of a PGG graph's components' average shortest
#         path lengths
#       - computation of average clustering coefficients of a PGG graph
# 2013/05/02    A. Shapiro, E. Pichler      v09
#   - minor corrections and typo fixes
# 2013/04/25    E. Pichler      v08
#   - moved pgg_v01.py back to pgg.py
#   - corrected initialization in initialize_pgg() so that exactly
#     n_cooperators are set to be cooperators
# 2013/04/25    E. Pichler      v07
#   - deleted obsolete code sections
#   - updated comments or delete obsolete comments
#   - replaced amp and exponent/index with a/C.powerlaw_C and
#     b/C.powerlaw_gamma, respectively
#   - cleaned up and aligned comments in cumulative_degree_fit() and
#     power_law_fit()
#   - eliminated unnecesary variables in record_graph_state()
#   - corrected header line output to _tss file in simulate_pgg()
#   - moved the following graph attributes into the utility container:
#       - G.graph['n_changers']       -> C.n_changers
#       - G.graph['n_movers']         -> C.n_movers
#       - G.graph['n_edge_additions'] -> C.n_edge_additions
#       - G.graph['n_edge_deletions'] -> C.n_edge_deletions
#   - simplified computation of n_movers and moved that computation from
#     accumulate_edge_changes() to normalize_edge_changes()
#   - deleted obsolete sections in to do list
# 2013/04/16    A. Shapiro      v06
#   - added functions
#       - cumulative_degree_fit
#       - power_law_fit
#   - modified data output code in
#       - record_graph_state
#       - record_run_summary
# 2013/04/10    E. Pichler      v05
#   - eliminated irrelevant strategy computation of payoff for player in payoff()
#   - corrected single cooperative player payoff computation
#   - completely rewrote edge deletion/addition handling
#   - changes to, or deletion (obsolete) or addition of the following functions:
#       - accumulate_edge_changes
#       - change connectivity
#       - compute_payoffs
#       - make_edge_changes
#       - normalize_edge_changes
#       - pre_update_reset
#       - reset_update_monitors
#       - update_connectivity
#       - update_graph_state
# 2013/03/11    E. Pichler      v04
#   - merged with Avi Shapiro's changes from 20130211
#   - added to utility container:
#       - index
#       - t_ss
#       - t_window
#       - degree_average
#       - payoff_average
#       - aspiration_average
#       - satisfaction_average
#   - in utility container deleted _tmp suffixed variables
#       - degree_tmp
#       - payoff_tmp
#       - aspiration_tmp
#       - satisfaction_tmp
#   - corrected record_run_summary output
#   - avoided variable and function name collisions in record_graph_state
#   - deleted references to and use of obsolete coopStat, defStat, and
#     degreeStat
#   - added output of header lines in output files
#   - is_steady_state now checks if enough time has passed for steady state
#     calculations
#   - cleaned up ToDo and comment sections; deleted old comments in code
# 2013/01/26    E. Pichler      v03
#   - corrected header comments
#   - renamed cooperator_count to count_cooperators
#   - renamed defector_count to count_defectors
#   - added utility_container class
#   - adapted code to availability of new utility container object
# 2012/12/07    E. Pichler      v02
#   - added docstrings to all functions
#   - moved changers and movers counters as attribute into graph class
#   - merged with Avi Shapiro's changes
# 2012/11/30
#   - 	E. Pichler      v01
#   - initial version (heavily based on and adapted from Avi Shapiro's
#     pgg.py, submitted on 2012/11/29)
#   - added many additional function for integration from PGG_Simulator.py
#   - adapted to PGG_Simulator.py naming conventions
#   - in update_position changed to first breaking links (since they are
#     responsible for the dissatisfaction at this time) and then creating
#     new links
#   - initial submit
#
#**************************************************************************

#--------------------------------------------------------------------------

# ***** preliminaries *****

# --- import modules and packages ---
from __future__ import (
     division,
     print_function,
     )
import random as rd
import math
import scipy as sp
#from scipy import optimize
import networkx as nx
from compiler.ast import flatten
import socket
import powerlaw
#import utilities

# --- defaults ---
# --- define utility container class ---
class utility_container:
    def __init__(self):
        self.alpha = 0
        self.alpha_delta = 0
        self.alpha_max = 0
        self.alpha_min = 0
        self.aspiration_average = []
        self.aspiration_max = []
        self.aspiration_min = []
        self.changers = set()
        self.clustering_coefficient_average = []
        #self.component_path_length_average = []
        self.debug = False
        self.degree_average = []
        self.degree_max = []
        self.degree_min = []
        self.edges_add = []
        self.edges_delete = []
        self.graph_type = 0
        self.index = ""
        self.input_name_g = ""
        self.k = 0
        #self.largest_component_path_length = []
        self.mu = 0
        self.movers = set()
        self.n_changers = []
        self.n_components = []
        self.n_cooperators = []
        self.n_cooperators_init = 0
        self.n_edge_additions = 0
        self.n_edge_deletions = 0
        self.n_edges = []
        self.n_edges_CC = []
        self.n_edges_CD = []
        self.n_edges_DD = []
        self.n_largest_component = []
        self.n_movers = []
        self.n_players = 0
        self.n_simulations = 0
        self.n_unstable_nodes = []
        self.noise_sigma = 0
        self.output_dir = ""
        self.output_file_g = None
        self.output_file_pss = None
        self.output_file_ts = None
        self.output_name_g = ""
        self.output_name_pss = ""
        self.output_name_ts = ""
        self.payoff_average = []
        self.payoff_max = []
        self.payoff_min = []
        #self.powerlaw_C = []
        #self.powerlaw_gamma = []
        self.program_variant = 0
        self.satisfaction_average = []
        self.satisfaction_max = []
        self.satisfaction_min = []
        self.stem = ""
        self.synergy = 0
        self.synergy_delta = 0
        self.synergy_max = 0
        self.synergy_min = 0
        self.t = 0
        self.t_delta = 0
        self.t_graph = []
        self.t_i = 0
        self.t_max = 0
        self.t_min = 0
        self.t_ss = 0
        self.t_window = 0
        self.time_stamp = ""


# ***** function definitions *****

# accumulate all connectivity changes
def accumulate_edge_changes(G, i, p, C):
    """
    Accumulate all connectivity changes.
    """
    # break a random link (only 1 link at this time)
    if rd.random() < p:
        neighbors = G.neighbors(i)
        if neighbors != []:
            j = rd.choice(neighbors)
            if j < i:
                C.edges_delete.append((j, i))
            else:
                C.edges_delete.append((i, j))
    # create a random link (only 1 link at this time)
    if rd.random() < p:
        potential_neighbors = list(nx.non_neighbors(G, i))
        #potential_neighbors = list(set(G.nodes()) - {i} - set(G.neighbors(i)))
        if potential_neighbors != []:
            j = rd.choice(potential_neighbors)
            if j < i:
                C.edges_add.append((j, i))
            else:
                C.edges_add.append((i, j))

# accumulate all strategy changes
def accumulate_strategy_changes(i, p, C):
    """
    Accumulate all strategy changes.
    """
    if rd.random() < p:
        C.changers.add(i)

# calculate a player's aspiration
def aspiration(player):
    """
    Calculate a player's aspiration.
    """
    return player['greediness'] * player['pi_max'] + \
           (1 - player['greediness']) * player['pi_min']

## compute the weighted average of averages of shortest component path lengths
#def average_shortest_component_pathlength(G, C):
#    """
#    Compute the weighted average of averages of shortest component path lengths.
#    """
#    n_components = nx.number_connected_components(G)
#    component_path_length_average = 0
#    for component in nx.connected_component_subgraphs(G):
#        component_path_length_average += nx.average_shortest_path_length(component) / component.number_of_nodes()
#    component_path_length_average *= C.n_players
#    return n_components, component_path_length_average

# compute the weighted average of averages of shortest component path lengths
def average_shortest_component_pathlength(G):
    """
    Compute the weighted average of averages of shortest component path lengths.
    """
    n_components = nx.number_connected_components(G)
    component_path_length_average = 0
    edgeless_nodes = 0  # nodes without edges do not contribute to path length average
    for component in nx.connected_component_subgraphs(G):
        if component.number_of_edges() == 0:
            edgeless_nodes += 1
        else:
            component_path_length_average += nx.average_shortest_path_length(component) * component.number_of_nodes()
    if G.number_of_nodes() == edgeless_nodes: # for the case G has 0 edges
        component_path_length_average = 0
    else:
        component_path_length_average /= (G.number_of_nodes() - edgeless_nodes)
    return n_components, component_path_length_average

# compute the cooperator and defector payoffs for a player's PGG
def compute_payoffs(G, player, synergy):
    """
    Compute the cooperator and defector payoffs for a player.
    """
    if len(G.neighbors(player)) == 0:
        # there are no payoffs if the player does not participate in any game
        return 0, 0
    else:
        n_cooperators = [G.node[i]['strategy'] for i in G.neighbors_iter(player)].count('C')
        payoff_D = synergy * n_cooperators / (1 + len(G.neighbors(player)))
        payoff_C = payoff_D - 1
        return payoff_C, payoff_D

# compute all the PGG graph state variables at time t
def compute_state_variables(G, C):
    """
    Compute all the PGG graph state variables at time t.
    """
    # get the current time index
    C.t_i = C.t % C.t_window

    # compute graph specific variables
    C.n_edges[C.t_i] = G.number_of_edges()
    
    # - count number of edge types: CC, CD (including DC), DD
    C.n_edges_CC[C.t_i] = 0
    C.n_edges_CD[C.t_i] = 0
    C.n_edges_DD[C.t_i] = 0
    for e in G.edges_iter():
        edge_type = G.node[e[0]]['strategy'] + G.node[e[1]]['strategy']
        if edge_type == 'CC':   
            C.n_edges_CC[C.t_i] += 1
        elif edge_type == 'CD' or edge_type == 'DC':
            C.n_edges_CD[C.t_i] += 1
        elif edge_type == 'DD':
            C.n_edges_DD[C.t_i] += 1
    
    C.n_components[C.t_i] = nx.number_connected_components(G)
    #C.n_largest_component[C.t_i], C.largest_component_path_length[C.t_i] = largest_component_average_shortest_pathlength(G)
    C.n_largest_component[C.t_i] = nx.connected_component_subgraphs(G)[0].number_of_nodes()
    C.clustering_coefficient_average[C.t_i] = nx.average_clustering(G)
    degrees = G.degree().values()
    C.degree_average[C.t_i] = sum(degrees) / C.n_players
    C.degree_min[C.t_i] = min(degrees)
    C.degree_max[C.t_i] = max(degrees)
    #C.powerlaw_C[C.t_i], C.powerlaw_gamma[C.t_i] = cumulative_degree_fit(G)
    #C.powerlaw_gamma[C.t_i] = degree_pdf_fit(G)

    # compute PGG specific variables
    # - reset payoffs for all players and connectivity and strategy change monitors
    reset_payoffs(G)
    reset_update_monitors(C)

    # - count the number of cooperators
    C.n_cooperators[C.t_i] = count_cooperators(G)

    # - compute payoffs for all players
    for i in G.nodes_iter():
        payoff_C, payoff_D = compute_payoffs(G, i, C.synergy)
        # first update a player's payoff with the payoff for the game for which he is
        # the game center ...
        payoff(G.node[i], payoff_C, payoff_D)
        # ... then pay off all the players participating in this game
        for j in G.neighbors_iter(i):
            payoff(G.node[j], payoff_C, payoff_D) # payoff neighbors
    payoffs = [G.node[i]['pi'] for i in G.nodes_iter()]
    C.payoff_average[C.t_i] = sum(payoffs) / C.n_players
    C.payoff_min[C.t_i] = min(payoffs)
    C.payoff_max[C.t_i] = max(payoffs)

    # - compute aspiration and satisfaction, and evaluate if strategy and edge
    #   changes will be necessary
    if C.program_variant == 2:
        for i in G.nodes_iter():
            # compute aspiration and satisfaction for the current generation
            G.node[i]['aspiration'] = aspiration(G.node[i])
            G.node[i]['satisfaction'] = satisfaction_2(G.node[i])
            # potentially change a player's strategy or participation in single games
            # depending on the player's satisfaction
            change_probability = .5 * (1 + math.tanh(-G.node[i]['satisfaction'] / C.k))
            accumulate_strategy_changes(i, change_probability, C)
            accumulate_edge_changes(G, i, change_probability, C)
    else: # presumably program_variant = 1
        for i in G.nodes_iter():
            # compute aspiration and satisfaction for the current generation
            G.node[i]['aspiration'] = aspiration(G.node[i])
            G.node[i]['satisfaction'] = satisfaction_1(G.node[i], C.noise_sigma)
            # if a player is unsatisfied, potentially change participation in single games
            # and change the players strategy
            if G.node[i]['satisfaction'] <= 0:
                change_probability = math.tanh(-G.node[i]['satisfaction'] / C.k)
                accumulate_strategy_changes(i, change_probability, C)
                accumulate_edge_changes(G, i, change_probability, C)
    normalize_edge_changes(C)
    aspirations = [G.node[i]['aspiration'] for i in G.nodes_iter()]
    C.aspiration_average[C.t_i] = sum(aspirations) / C.n_players
    C.aspiration_min[C.t_i] = min(aspirations)
    C.aspiration_max[C.t_i] = max(aspirations)
    satisfactions  = [G.node[i]['satisfaction'] for i in G.nodes_iter()]
    C.satisfaction_average[C.t_i] = sum(satisfactions) / C.n_players
    C.satisfaction_min[C.t_i] = min(satisfactions)
    C.satisfaction_max[C.t_i] = max(satisfactions)

    # - get the number of nodes that will change strategy, the number of nodes
    #   that will form or break connections to other nodes, and the total
    #   number of unstable nodes
    C.n_changers[C.t_i] = len(C.changers)
    C.n_movers[C.t_i] = len(C.movers)
    C.n_unstable_nodes[C.t_i] = len(C.changers | C.movers)

# count number of cooperators in a PGG
def count_cooperators(G):
    """
    Count the number of cooperators in a PGG graph.
    """
    return [G.node[i]['strategy'] for i in G.nodes_iter()].count('C')

# count number of defectors in a PGG
def count_defectors(G):
    """
    Count the number of defectors in a PGG graph.
    """
    return [G.node[i]['strategy'] for i in G.nodes_iter()].count('D')

# create PGG graph from scratch or read from a graph input file
def create_pgg_graph(C):
    """
    Create a PGG graph from scratch or read from a graph input file.
    """
    # first create graph ...
    # if graph_input is defined read the graph from the file
    if C.input_name_g != "":
        G = read_pgg_graph(C.input_name_g)
    # otherwise create a new graph
    else:
        # Erdos-Renyi graph with p=.05 for 2 nodes being connected
        if C.graph_type == 1:
            p = .05   # probability of 2 nodes being connected
            G = nx.erdos_renyi_graph(C.n_players, p)
        # Watts-Strogatz graph with k = 4 and p=.4
        elif C.graph_type == 2:
            k = 4
            p = 0.4
            G = nx.connected_watts_strogatz_graph(C.n_players, k, p)
        # Barabasi-Albert graph with average degree 2m
        elif C.graph_type == 3:
            m = 2   # average degree is 2*m
            G = nx.barabasi_albert_graph(C.n_players, m)
        # graph without edges
        elif C.graph_type == 4:
            G = nx.empty_graph(C.n_players, create_using=nx.Graph())
        # fully connected graph
        elif C.graph_type == 5:
            G = nx.complete_graph(C.n_players, create_using=nx.Graph())
        # Barabasi-Albert graph with average degree 2m
        else: # presumably graph_type = 3
            m = 2
            G = nx.barabasi_albert_graph(C.n_players, m)
    # ... then initialize the PGG graph and its attributes
    initialize_pgg_graph(G, C)
    return G

# # compute a graph's cumulative degree distribution power law fit parameters
# def cumulative_degree_fit(G):
#     """
#     Compute cumulative degree distribution power law fit parameters.
#     (To get a nice(r) power law fit this function employs 2 empirically determined
#     hacks:
#       - data for k=0 and k=1 are discarded
#       - the remaining upper 2/3 of the k range are discarded for the power law fit
#     To use a more objective power law fit function the algorithm described in
#     Clauset _2009b should be used.)
#     """
#     y = nx.degree_histogram(G)
#     # prepare data for fitting
#     y.reverse() # for inverse cumulative distribution
#     y = list(sp.cumsum(y, dtype=float) / G.number_of_nodes())
#     # trim data to facilitate power law using logs:
#     #   - discard data for k=0,1
#     y.pop()     # remove k=0 data
#     y.pop()     # remove k=1 data
#     y.reverse() # back to ordering for increasing degree
#     datalen = len(y)
#     x = [i+2 for i in range(datalen)]    # degrees [2, 3, 4, ...]
#     # fit to a * x^b
#     pinit = [1.0, -2.0] # initial parameter guess [a, b]
#     # trim data to eliminate tail effects:
#     #   - do not use data for upper 2/3 of k range in power law fit
#     a, b = power_law_fit(x[:datalen//3], y[:datalen//3], pinit)
#     return a, b

# compute a graph's degree distribution power law fit parameters
def degree_pdf_fit(G):
    """
    Compute power law probability distribution function fit parameters for
    node degrees.
    Uses powerlaw module described in Alstott_2013.pdf (http://arxiv.org/abs/1305.0215)
    based on the algorithm described in Clauset_2009b.pdf.
    Returns power law exponent gamma where p(x)~x^(-gamma)
    (Perhaps return xmin too, since then normalizing constant can be computed.)
    """
    fit = powerlaw.Fit(G.degree().values(), discrete=True)
    return fit.alpha

# a dummy function
def dummy_function():
    """
    A dummy function doing nothing and returning 0.
    """
    return 0

# initialize a PGG graph and its attributes
def initialize_pgg_graph(G, C):
    """
    Initialize a PGG graph and its attributes.
    """
    # initialize utility container attributes
    reset_update_monitors(C)

    # initialize PGG graph player attributes
    # first determine the cooperator/defector status for each node ...
    nodes = G.nodes()
    nodes_C = rd.sample(nodes, C.n_cooperators_init)
    nodes_D = list(set(nodes) - set(nodes_C))
    for i in nodes_C:
        G.node[i]['strategy'] = 'C'  # cooperators
    for i in nodes_D:
        G.node[i]['strategy'] = 'D'  # defectors
    # ... then determine the payoffs for each node ...
    reset_payoffs(G)
    for i in G.nodes_iter():
        payoff_C, payoff_D = compute_payoffs(G, i, C.synergy)
        # first update a player's payoff with the payoff for the game for which he is
        # the game center ...
        payoff(G.node[i], payoff_C, payoff_D)
        # ... then pay off all the players participating in this game
        for j in G.neighbors_iter(i):
            payoff(G.node[j], payoff_C, payoff_D)
    # ... and last set greediness, and min and max payoff for each node
    for i in G.nodes_iter():
        G.node[i]['greediness'] = C.alpha
        G.node[i]['pi_min'] = G.node[i]['pi']
        G.node[i]['pi_max'] = G.node[i]['pi']
        #G.node[i]['aspiration'] = aspiration(G.node[i])
        #G.node[i]['satisfaction'] = satisfaction(G.node[i], C.noise_sigma)

# test if steady state has been reached
def is_steady_state(C):
    """
    Test if steady state has been reached.
    (Currently always returns False.)
    """
    if C.t - C.t_min < C.t_window:
        return False
    else:
        # fill in real steady state test here ...
        # if steady state has been reached, set C.t_ss = C.t
        return False

# compute the average shortest component path length of the largest
# connected component of a graph
def largest_component_average_shortest_pathlength(G):
    """
    Compute the average shortest component path length of the largest
    connected component of a graph.
    """
    H = nx.connected_component_subgraphs(G)[0]
    return H.number_of_nodes(), nx.average_shortest_path_length(H)

# normalize edge change lists and count edge changes
def normalize_edge_changes(C):
    """
    Normalize the edge addition and deletion lists and count edge changes.
    """
    # create unique lists of edge changes
    #for i in C.edges_add:
    #    if i[1] < i[0]:
    #        i = (i[1], i[0])
    C.edges_add = sorted(set(C.edges_add))
    #for i in C.edges_delete:
    #    if i[1] < i[0]:
    #        i = (i[1], i[0])
    C.edges_delete = sorted(set(C.edges_delete))
    # count the edge changes
    C.n_edge_additions = len(C.edges_add)
    C.n_edge_deletions = len(C.edges_delete)

    # get the set of nodes that change their connectivity
    C.movers = set(flatten(C.edges_add + C.edges_delete))

# add a single game's payoff to a player's payoff
def payoff(player, payoff_C, payoff_D):
    """
    Add a single game's payoff to a single player's overall payoff.
    """
    if player['strategy'] == 'C':
        player['pi'] += payoff_C
    else:
        player['pi'] += payoff_D

# fit (x,y) data to a power law y = amp * (x**index)
def power_law_fit(x, y, pinit):
    """
    Fit (x, y) data to a power law, y = a * x^b.
    Returns a, b and associated fit error.
    (See also http://www.scipy.org/Cookbook/FittingData#head-5eba0779a34c07f5a596bbcf99dbc7886eac18e5)
    """
    # fit log of data to a line:
    #   y = a * x^b
    #   log(y) = log(a) + b * log(x)
    logx = sp.log10(x)
    logy = sp.log10(y)

    # define linear function to be fitted and error function
    fitfunc = lambda p, x: p[0] + p[1] * x
    errfunc = lambda p, x, y: (y - fitfunc(p, x))

    # find fit parameters; (alternatively try polyfit() function)
    out = sp.optimize.leastsq(errfunc, pinit, args=(logx, logy), full_output=1)
    pfinal = out[0]      # solved parameters p[0], p[1]
    #covar = out[1]       # some fit error info (?)
    a = 10.0**pfinal[0]  # power law constant
    b = -pfinal[1]       # power law exponent

    return a, b

# print the time series file header
def print_tsfile_header(f):
    """
    Print the headers row to a file.
    """
    # format for output data for time t:
    #   [time descriptor]
    #     t
    #   [graph descriptors]
    #     # of edges
    #     # of C-C edges
    #     # of C-D (and D-C) edges
    #     # of D-D edges
    #     # of components
    #     size of largest component
    #     largest component average shortest path length
    #     average clustering coefficient
    #     minimum degree
    #     maximum degree
    #     average degree
    ##    inverse cumulative degree distribution powerlaw constant
    ##    inverse cumulative degree distribution powerlaw exponent
    #   [PGG descriptors]
    #     # of cooperators
    #     minimum payoff
    #     maximum payoff
    #     average payoff
    #     minimum aspiration
    #     maximum aspiration
    #     average aspiration
    #     minimum satisfaction
    #     maximum satisfaction
    #     average satisfaction
    #   [update descriptors]
    #     # of unstable nodes
    #     # of strategy changers
    #     # of movers
    #     # of edge deletions
    #     # of edge additions
    print(
                                              #   [time descriptor]
          't',                                #     t
                                              #   [graph descriptors]
          'n_edges',                          #     # of edges
          'n_edges_CC',                       #     # of C-C edges
          'n_edges_CD',                       #     # of C-D (and D-C) edges
          'n_edges_DD',                       #     # of D-D edges
          'n_components',                     #     # of components
          'n_component_l',                    #     size of largest component
          #'component_l_path_length',         #     largest component average shortest path length
          'clustering_coefficient_average',   #     average clustering coefficient
          'degree_min',                       #     minimum degree
          'degree_max',                       #     maximum degree
          'degree_average',                   #     average degree
          #'powerlaw_C',                      #     inverse cumulative degree distribution powerlaw constant
          #'powerlaw_gamma',                  #     inverse cumulative degree distribution powerlaw exponent
                                              #   [PGG descriptors]
          'n_cooperators',                    #     # of cooperators
          'payoff_min',                       #     minimum payoff
          'payoff_max',                       #     maximum payoff
          'payoff_average',                   #     average payoff
          'aspiration_min',                   #     minimum aspiration
          'aspiration_max',                   #     maximum aspiration
          'aspiration_average',               #     average aspiration
          'satisfaction_min',                 #     minimum satisfaction
          'satisfaction_max',                 #     maximum satisfaction
          'satisfaction_average',             #     average satisfaction
                                              #   [update descriptors]
          'n_unstable_nodes',                 #     # of unstable nodes
          'n_changers',                       #     # of strategy changers
          'n_movers',                         #     # of movers
          'n_edge_deletions',                 #     # of edge deletions
          'n_edge_additions',                 #     # of edge additions
          file=f, sep='\t'
          )

# read graph from file
def read_pgg_graph(input_name_g):
    """
    Read a graph from a file with name input_name_g.
    """
    G = nx.read_gml(input_name_g)
    return G

# record graph state
def record_graph_state(C):
    """
    Record the state of a graph G at time t in file f.
    """
    # format for output data for time t:
    #   [time descriptor]
    #     t
    #   [graph descriptors]
    #     # of edges
    #     # of C-C edges
    #     # of C-D (and D-C) edges
    #     # of D-D edges
    #     # of components
    #     size of largest component
    #     largest component average shortest path length
    #     average clustering coefficient
    #     minimum degree
    #     maximum degree
    #     average degree
    ##    inverse cumulative degree distribution powerlaw constant
    ##    inverse cumulative degree distribution powerlaw exponent
    #   [PGG descriptors]
    #     # of cooperators
    #     minimum payoff
    #     maximum payoff
    #     average payoff
    #     minimum aspiration
    #     maximum aspiration
    #     average aspiration
    #     minimum satisfaction
    #     maximum satisfaction
    #     average satisfaction
    #   [update descriptors]
    #     # of unstable nodes
    #     # of strategy changers
    #     # of movers
    #     # of edge deletions
    #     # of edge additions

    print(
                                                              #   [time descriptor]
          '%d'   % C.t,                                       #     t
                                                              #   [graph descriptors]
          '%d'   % C.n_edges[C.t_i],                          #     # of edges
          '%d'   % C.n_edges_CC[C.t_i],                       #     # of C-C edges
          '%d'   % C.n_edges_CD[C.t_i],                       #     # of C-D (and D-C) edges
          '%d'   % C.n_edges_DD[C.t_i],                       #     # of D-D edges
          '%d'   % C.n_components[C.t_i],                     #     # of components
          '%d'   % C.n_largest_component[C.t_i],              #     size of largest component
          #'%.6f' % C.largest_component_path_length[C.t_i],   #     largest component average shortest path length
          '%.6f' % C.clustering_coefficient_average[C.t_i],   #     average clustering coefficient
          '%d'   % C.degree_min[C.t_i],                       #     minimum degree
          '%d'   % C.degree_max[C.t_i],                       #     maximum degree
          '%.6f' % C.degree_average[C.t_i],                   #     average degree
          #'%.6f' % C.powerlaw_C[C.t_i],                      #     inverse cumulative degree distribution powerlaw constant
          #'%.6f' % C.powerlaw_gamma[C.t_i],                  #     inverse cumulative degree distribution powerlaw exponent
                                                              #   [PGG descriptors]
          '%d'   % C.n_cooperators[C.t_i],                    #     # of cooperators
          '%.6f' % C.payoff_min[C.t_i],                       #     minimum payoff
          '%.6f' % C.payoff_max[C.t_i],                       #     maximum payoff
          '%.6f' % C.payoff_average[C.t_i],                   #     average payoff
          '%.6f' % C.aspiration_min[C.t_i],                   #     minimum aspiration
          '%.6f' % C.aspiration_max[C.t_i],                   #     maximum aspiration
          '%.6f' % C.aspiration_average[C.t_i],               #     average aspiration
          '%.6f' % C.satisfaction_min[C.t_i],                 #     minimum satisfaction
          '%.6f' % C.satisfaction_max[C.t_i],                 #     maximum satisfaction
          '%.6f' % C.satisfaction_average[C.t_i],             #     average satisfaction
                                                              #   [update descriptors]
          '%d'   % C.n_unstable_nodes[C.t_i],                 #     # of unstable nodes
          '%d'   % C.n_changers[C.t_i],                       #     # of strategy changers
          '%d'   % C.n_movers[C.t_i],                         #     # of movers
          '%d'   % C.n_edge_deletions,                        #     # of edge deletions
          '%d'   % C.n_edge_additions,                        #     # of edge additions
          file=C.output_file_ts, sep='\t'
          )

# write run parameters and steady state or final values to file
def record_run_summary(G, C):
    """
    Write run parameters and steady state or final values to file.
    """
    # format for output data for steady state:
    #   [simulation descriptors]
    #     index
    #     program variant
    #     hostname
    #   [PGG setup descriptors]
    #     # of players
    #     # of initial cooperators
    #     initial graph type
    #     synergy
    #     greediness
    #     normalization factor/temperature
    #     habituation factor
    #     noise sigma
    #   [time descriptors]
    #     t
    #     t of first ss
    #   [graph descriptors]
    #     ss # of edges
    #     ss # of C-C edges
    #     ss # of C-D (and D-C) edges
    #     ss # of D-D edges
    #     ss # of components
    #     ss size of largest component
    #     ss largest component average shortest path length
    #     ss average clustering coefficient
    #     ss minimum degree
    #     ss maximum degree
    #     ss average degree
    ##    ss inverse cumulative degree distribution powerlaw constant
    ##    ss inverse cumulative degree distribution powerlaw exponent
    #   [PGG descriptors]
    #     ss # of cooperators
    #     ss minimum payoff
    #     ss maximum payoff
    #     ss average payoff
    #     ss minimum aspiration
    #     ss maximum aspiration
    #     ss average aspiration
    #     ss minimum satisfaction
    #     ss maximum satisfaction
    #     ss average satisfaction
    #   [update descriptors]
    #     ss # of unstable nodes
    #     ss # of strategy changers
    #     ss # of movers

    f = open(C.output_name_pss, 'w')
    # record run parameters and steady state values
    print(
                                                 #   [simulation descriptors]
          'index',                               #     index
          'program_variant',                     #     program variant
          'hostname',                            #     hostname
                                                 #   [PGG setup descriptors]
          'n_players',                           #     # of players
          'n_cooperators_init',                  #     # of initial cooperators
          'graph_type',                          #     initial graph type
          'synergy',                             #     synergy
          'alpha',                               #     greediness
          'k',                                   #     normalization factor/temperature
          'mu',                                  #     habituation factor
          'noise_sigma',                         #     noise sigma
                                                 #   [time descriptors]
          't',                                   #     t
          't_ss',                                #     t of first ss
                                                 #   [graph descriptors]
          'n_edges_ss',                          #     ss # of edges
          'n_edges_CC_ss',                       #     # of C-C edges
          'n_edges_CD_ss',                       #     # of C-D (and D-C) edges
          'n_edges_DD_ss',                       #     # of D-D edges
          'n_components_ss',                     #     ss # of components
          'n_component_l_ss',                    #     ss size of largest component
          #'component_l_path_length_ss',          #     ss largest component average shortest path length
          'clustering_coefficient_average_ss',   #     ss average clustering coefficient
          'degree_min_ss',                       #     ss minimum degree
          'degree_max_ss',                       #     ss maximum degree
          'degree_average_ss',                   #     ss average degree
          #'powerlaw_C_ss',                      #     ss inverse cumulative degree distribution powerlaw constant
          #'powerlaw_gamma_ss',                  #     ss inverse cumulative degree distribution powerlaw exponent
                                                 #   [PGG descriptors]
          'n_cooperators_ss',                    #     ss # of cooperators
          'payoff_min_ss',                       #     ss minimum payoff
          'payoff_max_ss',                       #     ss maximum payoff
          'payoff_average_ss',                   #     ss average payoff
          'aspiration_min_ss',                   #     ss minimum aspiration
          'aspiration_max_ss',                   #     ss maximum aspiration
          'aspiration_average_ss',               #     ss average aspiration
          'satisfaction_min_ss',                 #     ss minimum satisfaction
          'satisfaction_max_ss',                 #     ss maximum satisfaction
          'satisfaction_average_ss',             #     ss average satisfaction
                                                 #   [update descriptors]
          'n_unstable_nodes_ss',                 #     ss # of unstable nodes
          'n_changers_ss',                       #     ss # of strategy changers
          'n_movers_ss',                         #     ss # of movers
          file=f, sep='\t'
          )
    print(
                                                                         #   [simulation descriptors]
                   C.index,                                              #     index
                   C.program_variant,                                    #     program variant
                   socket.gethostname(),                                 #     hostname
                                                                         #   [PGG setup descriptors]
          '%d'   % C.n_players,                                          #     # of players
          '%d'   % C.n_cooperators_init,                                 #     # of initial cooperators
          '%d'   % C.graph_type,                                         #     initial graph type
          '%.6f' % C.synergy,                                            #     synergy
          '%.6f' % C.alpha,                                              #     greediness
          '%.6f' % C.k,                                                  #     normalization factor/temperature
          '%.6f' % C.mu,                                                 #     habituation factor
          '%.6f' % C.noise_sigma,                                        #     noise sigma
                                                                         #   [time descriptors]
          '%d'   % C.t,                                                  #     t
          '%d'   % C.t_ss,                                               #     t of first ss
                                                                         #   [graph descriptors]
          '%.6f' % (sum(C.n_edges) / C.t_window),                        #     ss # of edges
          '%.6f' % (sum(C.n_edges_CC) / C.t_window),                     #     ss # of edges
          '%.6f' % (sum(C.n_edges_CD) / C.t_window),                     #     ss # of edges
          '%.6f' % (sum(C.n_edges_DD) / C.t_window),                     #     ss # of edges
          '%.6f' % (sum(C.n_components) / C.t_window),                   #     ss # of components
          '%.6f' % (sum(C.n_largest_component) / C.t_window),            #     ss size of largest component
          #'%.6f' % (sum(C.largest_component_path_length) / C.t_window), #     ss largest component average shortest path length
          '%.6f' % (sum(C.clustering_coefficient_average) / C.t_window), #     ss average clustering coefficient
          '%.6f' % (sum(C.degree_min) / C.t_window),                     #     ss minimum degree
          '%.6f' % (sum(C.degree_max) / C.t_window),                     #     ss maximum degree
          '%.6f' % (sum(C.degree_average) / C.t_window),                 #     ss average degree
          #'%.6f' % (sum(C.powerlaw_C) / C.t_window),                    #     ss inverse cumulative degree distribution powerlaw constant
          #'%.6f' % (sum(C.powerlaw_gamma) / C.t_window),                #     ss inverse cumulative degree distribution powerlaw exponent
                                                                         #   [PGG descriptors]
          '%.6f' % (sum(C.n_cooperators) / C.t_window),                  #     ss # of cooperators
          '%.6f' % (sum(C.payoff_min) / C.t_window),                     #     ss minimum payoff
          '%.6f' % (sum(C.payoff_max) / C.t_window),                     #     ss maximum payoff
          '%.6f' % (sum(C.payoff_average) / C.t_window),                 #     ss average payoff
          '%.6f' % (sum(C.aspiration_min) / C.t_window),                 #     ss minimum aspiration
          '%.6f' % (sum(C.aspiration_max) / C.t_window),                 #     ss maximum aspiration
          '%.6f' % (sum(C.aspiration_average) / C.t_window),             #     ss average aspiration
          '%.6f' % (sum(C.satisfaction_min) / C.t_window),               #     ss minimum satisfaction
          '%.6f' % (sum(C.satisfaction_max) / C.t_window),               #     ss maximum satisfaction
          '%.6f' % (sum(C.satisfaction_average) / C.t_window),           #     ss average satisfaction
                                                                         #   [update descriptors]
          '%.6f' % (sum(C.n_unstable_nodes) / C.t_window),               #     ss # of unstable nodes
          '%.6f' % (sum(C.n_changers) / C.t_window),                     #     ss # of strategy changers
          '%.6f' % (sum(C.n_movers) / C.t_window),                       #     ss # of movers
          file=f, sep='\t'
          )
    f.close()

# reset all the players' payoffs to 0
def reset_payoffs(G):
    """
    Reset all the players' payoffs to 0.
    """
    for i in G.nodes_iter():
        G.node[i]['pi'] = 0

# reset edge change variables
def reset_update_monitors(C):
    """
    Reset all edge change variables.
    """
    # reset strategy changers and movers sets
    C.changers.clear()
    C.movers.clear()
    # reset the number of edge additions and deletions
    C.n_edge_additions = 0
    C.n_edge_deletions = 0
    # reset lists for edge additions and deletions
    C.edges_add = []
    C.edges_delete = []

# calculate a player's satisfaction (program variant 1)
def satisfaction_1(player, noise_sigma):
    """
    Calculate a player's satisfaction (program variant 1).
    """
    return player['pi'] - player['aspiration'] + rd.gauss(0, noise_sigma)

# calculate a player's satisfaction (program variant 2)
def satisfaction_2(player):
    """
    Calculate a player's satisfaction (program variant 2).
    """
    return player['pi'] - player['aspiration']

# simulate a PGG
def simulate_pgg(G, C):
    """
    Simulate a PGG from t_min to t_max in t_delta time steps and record time series.
    """
    # open time series output file
    C.output_file_ts = open(C.output_name_ts, 'w')
    # write header line to time series file
    print_tsfile_header(C.output_file_ts)

    # initializations
    C.t = C.t_min
    # begin of loop over time
    while C.t <= C.t_max:
        # print current time step at 10 intervals if in debug mode
        if C.debug and C.t % int(C.t_max/10)==0:
            print("t=", C.t, sep="")
        # compute the graph state variables
        compute_state_variables(G, C)
        # record state of graph
        record_graph_state(C)
        # output PGG graph if necessary
        if C.t in C.t_graph:
            C.output_name_g = C.stem + "_g_" + str(C.t).zfill(6) + ".gml"
            nx.write_gml(G, C.output_name_g)
        # stop if steady state has been reached
        if is_steady_state(C):
            break
        # update state of graph
        C.t += C.t_delta
        if C.t > C.t_max:
            C.t -= C.t_delta
            break
        update_graph_state(G, C)
    # end of loop over time

    # close time series output file
    C.output_file_ts.close()

# update graph state
def update_graph_state(G, C):
    """
    Update the state of a PGG graph, i.e., play 1 round of the PGG to produce
    the t + 1 data.
    """
    # update minimum and maximum payoff values for all players
    update_payoff_min_max(G, C)
    # update player strategies
    update_player_strategies(G, C)
    # update player participation in single PGG games
    update_player_connectivities(G, C)

# update minimum and maximum payoff values for all players
def update_payoff_min_max(G, C):
    """
    Update minimum and maximum payoff values for all players.
    """
    for i in G.nodes_iter():
        # update a player's payoff_min.
        if G.node[i]['pi'] < G.node[i]['pi_min']:
            # decrease pi_min
            G.node[i]['pi_min'] = G.node[i]['pi']
        else:
            # increase pi_min
            G.node[i]['pi_min'] += C.mu * (G.node[i]['pi'] - G.node[i]['pi_min'])
        # update a player's payoff_max
        if G.node[i]['pi'] > G.node[i]['pi_max']:
            # increase pi_max
            G.node[i]['pi_max'] = G.node[i]['pi']
        else:
            # decrease pi_max
            G.node[i]['pi_max'] += C.mu * (G.node[i]['pi'] - G.node[i]['pi_max'])

# update player participation in single PGG games by adding and deleting edges
# in the PGG graph
def update_player_connectivities(G, C):
    """
    Update player participation in single PGG games by adding and deleting
    edges in the PGG graph.
    """
    # delete old edges
    for i in C.edges_delete:
        G.remove_edge(i[0], i[1])
    # add new edges
    for i in C.edges_add:
        G.add_edge(i[0], i[1])

# update player strategies
def update_player_strategies(G, C):
    """
    Update player strategies.
    """
    for i in C.changers:
        # for all changers switch strategy from C (collaborator) to
        # D (defector), or vice versa
        if G.node[i]['strategy'] == 'C':
            G.node[i]['strategy'] = 'D'
        else:
            G.node[i]['strategy'] = 'C'

#--------------------------------------------------------------------------
