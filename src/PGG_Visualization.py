#!/usr/bin/python
# -*- coding: utf-8 -*-
#**************************************************************************
#
# $Id: PGG_Visualization.py $
# $Revision: v2 $
# $Author: ashapiro $
# $Date: 2014-08-21 $
# Type: Python module.
#
# Comments: Module defining Public Goods Game functions.
#
# Usage:
# Import this file from other Python files.
#
# To do:
#   -
#
# Copyright © 2012-2014 Avi M. Shapiro & Elgar E. Pichler.
# All rights reserved.
#
#**************************************************************************

#**************************************************************************
#
# Modifications:
# 2014/08/21    A. Shapiro      v2
#   - changes fetch_ts to accept a file pattern of pss files rather
#       than a lookup table
#   - fetch_ts and thus plot_ts are now very slow
# 2014/08/14    A. Shapiro      v1
#   - initial version from IPython notebooks

#**************************************************************************

#--------------------------------------------------------------------------

# ***** preliminaries *****

# --- import modules and packages ---
from __future__ import (
     division,
     print_function,
     )
import glob
import os
import numpy as np
import pandas as pd
import pandasql as pds
import matplotlib as mpl
import matplotlib.pyplot as plt
import prettyplotlib as ppl
import networkx as nx
#import powerlaw
import csvparse
#import utilities

# ***** function definitions *****


# Create and maintain pss lookup table
def make_lookup(filedir, ofile):
    """
    Concatenates all *_pss.csv files in filedir into file ofile.
    
    Returns the concatenated data as panda dataframe.
    
    Usage: 
    newdf = make_lookup('./data_20140824_Shapiro/n100_t20000/', 'outfile.csv')
    """
    
    filelist = glob.glob(filedir + '*_pss.csv')
    
    lookup_df = pd.DataFrame()
    
    for filename in filelist:
        try:
            newrow = pd.read_csv(filename, sep='\t')
        except pd.parser.CParserError:
            print("File", filename, " is empty.")
        except:
            print("Error with", filename)
        else:
            lookup_df = lookup_df.append(newrow, 
                                         ignore_index=True)
    if not os.path.isfile(filedir + ofile):
        print("Creating new lookup table ", filedir + ofile)

    lookup_df.to_csv(filedir + ofile, sep='\t', index=False, float_format='%.6g')    
    return lookup_df


# --- Heat map functions ---
# Should clean up and write doc strings
def df2ndarray(df, field):
    """
    Converts pandas dataframe holding field data as a function of greediness and synergy 
    in an evenly spaced grid to a 2D numpy array.

    Returns a numpy array.

    Parameters:

        df : pandas dataframe with columns "synergy", "greediness", and field

        field : string naming the third column in df
    """

    if field not in list(df.columns):
        print('Wrong data column name')
        return 1
    
    # set up heatmap array
    index_df = pds.sqldf("""SELECT greediness FROM df GROUP BY greediness;""", locals())
    index = list(index_df['greediness'])
    column_df = pds.sqldf("""SELECT synergy FROM df GROUP BY synergy;""", locals())
    column = list(column_df['synergy'])
    # create the data frame that will hold the heatmap data
    d2 = np.empty([len(index), len(column)])
    d2.fill(np.NaN)
    d3 = pd.DataFrame(d2, index=index, columns=column,dtype=float)
    
    # populate data frame
    i = 'greediness'
    c = 'synergy'
    v = field

    for n, row in df.iterrows():
        d3.loc[row[i], row[c]] = row[v]
    
    # transform the pandas data frame into a numpy array for plotting with prettyplotlib
    d4 = d3.as_matrix()
    return d4, i, c, v, index, column
  
def plot_heatmap(fig, ax, df, field, *args, **kwargs):
    """
    Plots a heatmap of field from dataframe df as a function of parameters
    greediness and synergy.

    Returns a handle to the plot and colorbar axes.

    Parameters:

        fig: figure label

        ax: axis label

        df: pandas DataFrame

        field: string name of DataFrame column to plot

    Optional arguments:
    
        args: additional unnamed parameters passed to pcolormesh
        
    Optional keyword arguments:
    
        kwargs: dictionary of named arguments passed to pcolormesh
    """

    # convert dataframe to 2D array for plotting
    darray, i, c, v, indices, columns = df2ndarray(df, field)
    ax.set_xlabel(c) 
    ax.set_ylabel(i)
    ax.set_title(v)
    
    # if n_players_query == 1000:
    #     Z = darray[:-1,:-10] # without Pichler data use Z = darray[:-1,:-2]
    # else:
    #     Z = darray[:-1,:]
    Z = darray[:,:]
    
    # adjust colorbar scale for positive data
    kwargs.setdefault('vmax', max(1,Z.max()))
    kwargs.setdefault('vmin', min(0,Z.min()))
    # print(kwargs)
    
    # LaTeX stylized axes labels
    for j in xrange(len(indices)):
        if j % 4 != 0:
            indices[j] = ''
        else:
            indices[j] = '$'+str(indices[j])+'$'
            
    for j in xrange(len(columns)):
        # Label multiples of 5
        if j % 5 != 4:
            columns[j] = ''
        else:
            columns[j] = '$'+str(int(columns[j]))+'$'
            
        
    # returning a tuple requires prettyplotlib from github which is numbered 0.1.5
    # but is more recent than version 0.1.7 on PyPi
    p, cbar = ppl.pcolormesh(fig, ax, Z,
               xticklabels=columns, #xticklabels_rotation='vertical', 
               yticklabels=indices, 
               # vmin=min(0,v_min), vmax=max(v_max,1), 
               edgecolors='face',  # this seems to corrupt corner of box
               *args, **kwargs
               )

    return p, cbar
    
    
# --- Time series functions ---

# This version is not backwards-compatible plotting code used for poster
# due to additional arguments n_players_query, t_query
def fetch_ts(r_query, alpha_query, field_query, n_players_query, t_query,
             graph_query, n_cooperators_init_query, pssfile, *args, **kwargs):
    """
    Get time series matching query as rows in a Panda DataFrame
    
    Returns a pandas DataFrame with each time series in rows. 
    
    Parameters:
    
        r_query : int
          Synergy value 
          
        alpha_query: float
          Greediness value
          
        field_query: string
          Data header name of desired time series. String must match a column header in PSS file.
        
        n_players_query: float
          Number of players
          
        t_query: float
          Duration of simulation
        
        graph_query: int
          Type of initial graph
        
        n_cooperators_init_query: int 
          Initial number of cooperators
        
        
    Optional arguments:
    
        args: additional unnamed parameters
          Not currently implemented
        
    Optional keyword arguments:
    
        kwargs: dictionary of named arguments
          simnum: Index or a tuple of indices of time series to return if not returning all.
          scale: Scaling factor for field_query
    """
    
    # fetch matching TS files from PSS lookup table
    q = """
    SELECT [index] AS filehead
    FROM df
    WHERE synergy = %f
    AND alpha = %f
    AND n_players = %s 
    AND t = %s
    AND graph_type = %s
    AND n_cooperators_init = %s;
    """ % (r_query, alpha_query, n_players_query, 
           t_query, graph_query, n_cooperators_init_query)

    # get relative path filenames of ts files to search
    fnames = []
    
    for f in glob.glob(pssfile):
        d_match = csvparse.extract_data_sql(f, dtype={'index':str}, query=q)
        # compile data from multiple simulations
        temp = list(d_match.filehead)
        temp = [os.path.dirname(f)+'/'+str(temp[i])+'_ts.csv' for i in xrange(len(temp))]
        fnames += temp
        
    # Variables to return
    files_used = []
    ts_field_data = pd.DataFrame({})

    kwargs.setdefault('scale', 1)
    
    # extract TS data from files with correct filehead
    q = """
    SELECT %s / %s
    FROM df
    """ % (field_query, kwargs.get("scale"))
    
    # return specific time series simultations
    if kwargs.get("simnum") != None:
        # The logic here is to handle cases where "simnum" is an integer and an iterable
        try:
            iterator = iter(kwargs.get("simnum"))
        except TypeError:
            # fetch only one simulation
            files_used = fnames[kwargs.get("simnum")]
            ts_field_data = pd.DataFrame(csvparse.extract_data_sql(files_used, query=q)).T
        else:
            # fetch multiple simulations
            for ind in kwargs.get("simnum"):
                files_used = files_used + [fnames[ind]]
                ts_field_temp = pd.DataFrame(csvparse.extract_data_sql(fnames[ind], query=q)).T
                ts_field_data = pd.concat([ts_field_data, ts_field_temp])
       
    # return all simulations
    else:
        for fn in fnames:
            files_used = files_used + [fn]
            ts_field_temp = pd.DataFrame(csvparse.extract_data_sql(fn, query=q)).T
            ts_field_data = pd.concat([ts_field_data, ts_field_temp])
    
    return ts_field_data, files_used
    
def plot_ts(fig, ax, r_query, alpha_query, field_query, n_players_query, t_query, 
            graph_query, n_cooperators_init_query, pssfile, *args, **kwargs):
    """
    Plot a time series matching the query.
       
    Parameters:
    
        fig: Matplotlib Figure
          Figure for the plot
        
        ax: Matplotlib Axis
          Axis for the plot
    
        r_query : int
          Synergy value 
          
        alpha_query: float
          Greediness value
          
        field_query: string
          Data header name of desired time series. String must match a column header in PSS file.
    
        n_players_query: float
          Number of players
          
        t_query: float
          Duration of simulation
        
        graph_query: int
          Type of initial graph

        n_cooperators_init: int 
          Initial number of cooperators

    Optional arguments:
    
        args: additional unnamed parameters
          Not currently implemented
        
    Optional keyword arguments:
    
        kwargs: dictionary of named arguments
          simnum: Index or a tuple of indices of time series to return if not returning all.
          window: Integer number of time steps for moving average. Defaults to 100.
          scale: Scaling factor for field_query
    """
    
    ts, files = fetch_ts(r_query, alpha_query, field_query, n_players_query, 
                         t_query, graph_query, n_cooperators_init_query, 
                         pssfile, *args, **kwargs)

    try:
        kwargs.pop("simnum") # discard argument before plotting
    except KeyError:
        pass

    try: 
        kwargs.pop("scale")  # discard argument before plotting
    except KeyError:
        pass
    
    try:
        window = kwargs.pop("window")
    except KeyError:
        window = 100
         
    # Now kwargs can be sent to plot()
    
    kwargs.setdefault('label', field_query.replace("_", "\_"))    
    
    for sim in xrange(ts.shape[0]):
        p = ppl.plot(ax, pd.rolling_mean(ts.iloc[sim],window), *args, alpha=1, **kwargs) # brighten colors
        ax.set_title('synergy = '+str(float(r_query)) + ',  greediness = ' + str(float(alpha_query)))
        ax.set_xlabel('$t$')
        ax.set_ylabel('moving average of \n'+field_query.replace("_", "\_"))
        # ax.set_xlim(0,t_query)
    
    return p, files
    
    
# --- Degree distributions ---
    
def plot_degdist(fig, ax, fname, *args, **kwargs):
    """
    Plot a degree distribution for a graph.
       
    Parameters:
    
        fig: Matplotlib Figure
          Figure for the plot
        
        ax: Matplotlib Axis
          Axis for the plot
          
        fname: String
          Name of gml file
    
    Optional arguments:
    
        args: additional unnamed parameters
          Standard plot arguments
        
    Optional keyword arguments:
    
        kwargs: dictionary of named arguments
          Standard plot keyword arguments
  
    """
    
    G = nx.read_gml(fname)
    y, x = np.histogram(nx.degree(G).values(), 
                        bins=range(0, G.number_of_nodes()), 
                        density=True)
    ppl.plot(ax, x[:-1], y, *args, alpha=1, **kwargs)
    plt.xscale('log')
    plt.yscale('log')
    ax.set_xlabel('degree')
    ax.set_ylabel('degree\ndistribution')
    plt.tight_layout()