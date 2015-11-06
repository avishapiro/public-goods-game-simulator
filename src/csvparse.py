#!/usr/bin/python
# -*- coding: utf-8 -*-
#**************************************************************************
#
# $Id: csvparse.py $
# $Revision: v05 $
# $Author: epichler $
# $Date: 2014-04-17 $
# Type: Python program.
#
# Comments: Module defining CSV file parsing functions.
#
# Usage:
# Import this file from other Python files.
#
# To do:
#   - add a match_files_sql() equivalent for match_files(), that uses
#     pandas and pandasql capabilities as in extract_data_sql()
#   - check out SQLAlchemy (http://www.sqlalchemy.org/)
#     and if applicable consider rewrite using that module's fuctionality
#
# Copyright © 2013-2014 Elgar E. Pichler & Avi M. Shapiro. All rights
# reserved.
#
#**************************************************************************


#**************************************************************************
#
# Modifications:
# 2014/04/04    E. Pichler      v05
#   - added clarifying comments for extract_data_sql()
#   - cosmetic changes
# 2014/02/19    E. Pichler      v04
#   - added extract_data_sql(), which is a non-backward compatible
#     variant of extract_data()
#   - extract_data_sql()
#     - uses pandas and pandasql
#     - allows for more flexible query criteria
#     - allows for input data type casting
#     - allows for float output data format specification
#     - tackles the float precision issue in extract_data() via the float
#       output data format specification
# 2014/02/07    E. Pichler      v03
#   - added additional optional argument to extract_data() which allows
#     column data type specification for extracted data
#   - the current extract_data() version still processes and outputs floats
#     within machine precision error for floats!
# 2013/09/04    E. Pichler      v02
#   - renamed matching_file_index argument from reserved dir to directory
# 2013/08/12	E. Pichler      v01
#   - initial version
#
#**************************************************************************


#--------------------------------------------------------------------------

from __future__ import (
     division,
     print_function,
     )
#import sys
#import csv
import glob
#import inspect
import numpy as np
import matplotlib.mlab as mlab
#import twisted
import pkg_resources
pkg_resources.require("pandas >= 0.13.1")
pkg_resources.require("pandasql >= 0.3.1")
import pandas as pd
import pandasql as pds


# extract data from CSV files
# [This implementation of extract_data() contains somewhat of a kludge:
# data extraction from files will occur with desired data type casting if such
# data types are supplied, but the matched records are then semi-intelligently
# recast. At this moment the second type casting cannot be forced by the user.
# Also, the output for floats might still be messed up within machine precision!
# In the future this routine should probably be rewritten using pandas module
# calls.]
# [This function is now superceded by the function extract_data_sql().
# Note that extract_data_sql() is not backward compatible with extract_data()!]
def extract_data(pattern, query, headers, ctypes=None, fname=''):
    """
    Extract data from CSV files whose name matches pattern.
    Every record in a given file is checked if it satisfies the query
    condition(s). If the query condition(s) are satisfied, data from the
    columns specified by headers are extracted from that record. Collected
    records are returned in a numpy record array and, if a filename fname is
    specified, they are also written to fn in tab-separated CSV format.
    If no matching records are found an empty record array of type bool is
    returned.
    argument:       comment:
    pattern         a file name pattern for files from which records are to be
                    extracted
    query           conditions in the form of a dictionary (list of key-value
                    pairs) that need to be fulfilled for a record to be
                    extracted;
                    the query dictionary is specified in one of these forms:
                      dict(k1=v1, k2=v2, k3=v3, ...)
                      dict({"k1":v1, "k2":v2, "k3":v3, ...})
                      {"k1":v1, "k2":v2, "k3":v3, ...}
    headers         a list of strings specifying the column headers for the
                    columns which are to be extracted
    ctypes          if not None, is a dictionary mapping column number or
                    munged column name to a converter function;
                    the column type converter dictionary can be specified as:
                      {"k1":t1, "k2":t2, "k3":t3, ...}
                    where the t can be, e.g., str, int, float, bool.
    fname           if defined, the name of the CSV file (tab-separated) to
                    which extracted records are written
    """

    # get all file names that match pattern
    infiles = glob.glob(pattern)
    infiles.sort()

    # determine the query and header keys (in lowercase because csv2rec
    # lowercases headers), and query values
    qkeys = query.keys()
    qlckeys = [x.lower() for x in qkeys]
    qvalues = query.values()
    hlckeys = [x.lower() for x in headers]
    if ctypes:
        ctypes_lc = dict((key.lower(), value) for (key, value) in ctypes.items())
    else:
        ctypes_lc = None
    mkeys = set(qlckeys)
    mkeys = mkeys.union(hlckeys)
    mrows = []

    # check files for query patterns
    for f in infiles:
        d = mlab.csv2rec(f, delimiter='\t', converterd=ctypes_lc)
        # check if the data contain the necessary columns
        if mkeys <= set(d.dtype.names):
            # find the records that match the query
            darray = mlab.rec_keep_fields(d, qlckeys)
            imatch = np.array([False]*darray.size)
            for i in range(darray.size):
                if list(darray[i]) == qvalues:
                    imatch[i] = True
            # get data from records that matched the query
            if any(imatch):
                marray = mlab.rec_keep_fields(d, hlckeys)[imatch]
                for row in marray:
                    mrows.append(row.tolist())

    # write data from matching records to file if requested and return results
    if mrows:
        # The following does not work because the mlab.csv2rec() converterd
        # data type specifications are different from the
        # np.core.records.fromrecords() dtype data type specifications ...
        #results = np.core.records.fromrecords(mrows, dtype=ctypes_lc)
        # ... so, for now we cross our fingers and hope that
        # np.core.records.fromrecords() intuits the data types correctly, which
        # it seems to do (most of the time)
        results = np.core.records.fromrecords(mrows, names=headers)
    else:
        dt = [(h, bool) for h in headers]
        results = np.recarray(0, dtype=dt)
    if fname != '':
        mlab.rec2csv(results, fname, delimiter='\t')
    return results


# extract data from CSV files
# [This function supercedes the now obsolete function extract_data().
# Note that extract_data_sql() is not backward compatible with extract_data()!]
def extract_data_sql(ifile, dialect=None, dtype=None, query=None,
                        ofile=None, float_format=None):
    """
    Extract and accumulate data from CSV files based on an SQL query.
    Files which names match ifile pattern are parsed assuming column data types
    as specified in dtype.
    If a column data type is unspecified the parser semi-intelligently
    determines that column's data type.
    The read-in data are concatenated and stored in a pandas data frame called
    df. This data frame is queried and data are extracted/generated as specified
    by the SQL query.
    Collected/generated records are returned in a pandas data frame and if a
    filename ofile is specified, they are also written to ofile in tab-separated
    CSV format. Data types of result columns are determined by dtype, or are
    automatically determined semi-intelligently if not specified. Real numbers
    are written to ofile in a format specified by float_format.
    If no matching records are found an empty data frame is returned.
    argument:       comment:
    ifile           a file name pattern for input files from which records are
                    to be extracted
    dialect         a string or csv.Dialect instance to expose more ways to
                    specify the input file format
                    [default: dialect=None]
    dtype           a data type name or a dict of column name to data type; if
                    not specified, data types will be inferred;
                    the column type converter dictionary can be specified as:
                      {"k1":t1, "k2":t2, "k3":t3, ...}
                    where the t can be, e.g., bool, float, int, object, str.
                    [default: dtype=None]
    query           a string that contains SQL query that is used for data
                    extraction, filtering, and creation;
                    please note that the SQL query can only be applied to every
                    input file separately!;
                    an example is:
                    query = \"\"\"
                      SELECT C, A, AVG(B) "B_mean", [INDEX] "I"
                      FROM df
                      GROUP BY C, A
                    \"\"\"
                    it is assumed that data from all the input files will be
                    sequentially read into a data frame called df;
                    in this example the computed avg(B) will be written to the
                    third column with header B_mean, and the fourth column,
                    specified within brackets because INDEX is a reserved SQL
                    keyword, will be written to a column with header I;
                    also, note that AVG and GROUP BY are only applied on a per
                    input file basis;
                    [default: query=None]
    ofile           if defined, the name of the CSV file (tab-separated) to
                    which extracted records are written
                    [default: ofile=None]
    float_format    a format which takes a single float argument and returns
                    a formatted string; to be applied to floats in the results
                    data frame when writing to the output file (e.g.,
                    float_format='%.6g'
                    [default: float_format=None]
    """

    # initialize the results data frame
    results = pd.DataFrame()
    # check files for patterns
    if query is not None:
        infiles = glob.glob(ifile)
        infiles.sort()
        for f in infiles:
            # get data frame from CSV file
            df = pd.read_csv(f, dialect=dialect, sep='\t', dtype=dtype)
            # get results from data frame
            r = pds.sqldf(query, locals())
            # append single results to total results data frame
            if r is not None:
                if results.empty:
                    results = r
                else:
                    results = results.append(r, ignore_index=True)
    # write data from matching records to file if requested and return results
    if ofile:
        results.to_csv(ofile, index=False, sep='\t', float_format=float_format)
    return results


# return names of files that satisfy query conditions
def match_files(pattern, query):
    """
    Return list of file names of CSV files that satisfy query conditions, where
      pattern ... a file name pattern for files that are to be tested
      query ... a list of key=value pairs
    A file is a match if any row contains entries for all query conditions where
    each entry in the column labeled with key is identical to the corresponding
    value.
    Specify query as a dictionary in one of these forms:
      dict(k1=v1, k2=v2, k3=v3, ...)
      dict({"k1":v1, "k2":v2, "k3":v3, ...})
      {"k1":v1, "k2":v2, "k3":v3, ...}
    """
    # get all file names that match pattern
    infiles = glob.glob(pattern)
    infiles.sort()
    mlist = []
    # determine the query keys (in lowercase because csv2rec lowercases headers)
    # and query values
    qkeys = query.keys()
    qlckeys = [x.lower() for x in qkeys]
    qvalues = query.values()
    # check files for patterns
    for f in infiles:
        try:
            d = mlab.csv2rec(f, delimiter='\t')                
        except ValueError:
            print(str(f)+" cannot be read by csv2rec")
        else:
            # check if the data contain the necessary columns
            if set(qlckeys) <= set(d.dtype.names):
                darray = mlab.rec_keep_fields(d, qlckeys)
                for row in darray:
                    if list(row) == qvalues:
                        # a match has been found
                        mlist.append(f)
                        break
    # return a list of file names for files with matches
    return mlist


# # return names of files that satisfy query conditions
# def match_files_v01(pattern, query):
#     """
#     Return list of file names of CSV files that satisfy query conditions, where
#       pattern ... a file name pattern for files that are to be tested
#       query ... a list of key=value pairs
#     A file is a match if any row contains entries for all query conditions where
#     each entry in the column labeled with key is identical to the corresponding
#     value.
#     Specify query as a dictionary in one of these forms:
#       dict(k1=v1, k2=v2, k3=v3, ...)
#       dict({"k1":v1, "k2":v2, "k3":v3, ...})
#       {"k1":v1, "k2":v2, "k3":v3, ...})
#     """
#     infiles = glob.glob(pattern)
#     mlist = []
#     print('# total files: ' + str(len(infiles)))
#     print('infiles: ', infiles)
#
#     qkeys = query.keys()
#     qvalues = query.values()
#     print(qkeys, qvalues)
#
#     qlckeys = [x.lower() for x in qkeys]
#
#     print qkeys
#     print qlckeys
#     print qvalues
#
#     for f in infiles:
#         d = mlab.csv2rec(f, delimiter='\t')
#         dcols = [d[k] for k in qlckeys]
#         for r in drows:
#             if r.tolist() == qvalues:
#                 print "true:", r.tolist(), " = ", qvalues
#                 break
#             else:
#                 print "false:", r.tolist(), " != ", qvalues
#
#     print('# matching files: ' + str(len(mlist)))
#     print('mlist: ', mlist)






def matching_file_index(directory,**keywords):
    """
    Return list of filename stems of files holding data for parameters given
    by the keywords and values given by the argument.
    """
    # files to search for desired parameters
    flist = glob.glob(directory+'*pss.csv')
    # file list accumulator
    matching_files = [];
    print('# total files: ' + str(len(flist)))

    keys = keywords.keys()
    print(keys, [keywords[key] for key in keys])

    # check for valid keywords
    valid_keys = set(['index', 'graph_type', 'n_players', 'n_cooperators_init',
                      'synergy', 'alpha', 'mu', 't', 't_ss',  'n_cooperators_ss',
                      'n_components_ss', 'n_component_l_ss',
                      'component_l_path_length_ss', 'clustering_coefficient_average_ss',
                      'powerlaw_gamma_ss', 'degree_average_ss', 'degree_min_ss',
                      'degree_max_ss', 'payoff_average_ss', 'payoff_min_ss',
                      'payoff_max_ss', 'aspiration_average_ss', 'aspiration_min_ss',
                      'aspiration_max_ss', 'satisfaction_average_ss',
                      'satisfaction_min_ss', 'satisfaction_max_ss'])
    if set(keys) <= valid_keys:
        pass
    else:
        print("Invalid search key.")
        return
    ####

    keys.sort()

    for f in flist:
        fdata = mlab.csv2rec(f, delimiter='\t')
        if [fdata[kw] for kw in keys] == [keywords[kw] for kw in keys]:
            matching_files.append(f)


    print('# files: ' + str(len(matching_files)))
    # remove file extensions
    return [matching_files[i][0:-8] for i in range(len(matching_files))]

#--------------------------------------------------------------------------
