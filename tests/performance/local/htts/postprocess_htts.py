#! /usr/bin/env python
# -*- coding: utf-8 -*-

from math import sqrt
from sys import exit

from optparse import OptionParser

from numpy import std, mean

# 0: Delay [micro-seconds] - Independent Variable
# 1: Tasks - Independent Variable
# 2: OS-threads - Independent Variable
# 3: Total Walltime [seconds]
# 4+: Counters

DELAY = 0
TASKS = 1
STASKS = 2
OS_THREADS = 3
SAMPLES = 4 # generated by this script

LAST_IVAR = SAMPLES
GENERATED_IVARS = 1

# Returns the index of the independent variable that we use to differentiate
# datasets (dataset == line on the graph).
def dataset_key(row):
    return (int(row[DELAY]), int(row[TASKS]), int(row[STASKS]))

# Returns a list of all the independent variables. 
def ivars(row):
    return tuple(int(row[x]) for x in range(0, LAST_IVAR + 1 - GENERATED_IVARS))

# Returns a list of all the dependent variables. 
def dvars(row):
    return row[(LAST_IVAR + 1):]

op = OptionParser(usage="%prog [prefix]")
args = op.parse_args()[1]

if len(args) != 1:
    op.print_help()
    exit(1)

prefix          = args[0]
input_filename  = prefix + '.dat'
output_filename = 'post_' + prefix + '.dat' 
header_filename = 'post_' + prefix + '.gpi'

input_file  = open(input_filename, 'r')
output_file = open(output_filename, 'w')
header_file = open(header_filename, 'w')

print 'Prefix:      ', prefix
print 'Input File:  ', input_filename
print 'Output File: ', output_filename
print 'Header File: ', header_filename

master = {}
legend = []

try:
    while True:
        line = input_file.next()

        # Look for the legend 
        if line[0] == '#':
            if line[1] == '#' and line.find(':') != -1:
                row = line.split(':')
                legend.append([row[1].strip(), row[2].strip()])
            else:
                print >> output_file, line, 
            continue   

        # Look for blank lines
        if line == "\n":
            continue

        row = line.split()

        if not dataset_key(row) in master:
            master[dataset_key(row)] = {}

        if not ivars(row) in master[dataset_key(row)]:
            master[dataset_key(row)][ivars(row)] = []

        master[dataset_key(row)][ivars(row)].append(dvars(row))  

except StopIteration:
    pass

number_of_dvars = None

legend.insert(SAMPLES, ["SAMPLES", "Number of Samples - Independent Variable"])

for (key, dataset) in sorted(master.iteritems()):
    for (ivs, dvs) in sorted(dataset.iteritems()):
#        if sample_size is None:
#            sample_size = len(dvs)
#        else:
#            if sample_size > len(dvs):
#                missing = sample_size - len(dvs)
#                print "WARNING: Missing "+str(missing)+" sample(s) for "+\
#                      "("+", ".join(str(x) for x in ivs)+")"
#
        for dv in dvs:
            if number_of_dvars is None:
                number_of_dvars = len(dv)
            else:
                assert number_of_dvars is len(dv)

for i in range(0, LAST_IVAR + 1):
    print >> output_file, '## %i:%s:%s' % (i, legend[i][0], legend[i][1])

    print >> header_file, '%s="%i"' % (legend[i][0], i + 1) 

# + 1 is for the "samples" variable that we insert
for i in range(0, (len(legend) - (LAST_IVAR + 1)) * 3, 3):
    i0 = (LAST_IVAR + 1) + i
    i1 = (LAST_IVAR + 1) + (i / 3)

    print >> output_file, '## %i:%s_AVG:%s - Average'\
        % (i0, legend[i1][0], legend[i1][1]) 
    print >> output_file, '## %i:%s_STD:%s - Standard Deviation'\
        % (i0 + 1, legend[i1][0], legend[i1][1])
    print >> output_file, '## %i:%s_CI:%s - 95%% Confidence Interval'\
        % (i0 + 2, legend[i1][0], legend[i1][1])

    print >> header_file, '%s_AVG="%i"' % (legend[i1][0], i0 + 1) 
    print >> header_file, '%s_STD="%i"' % (legend[i1][0], i0 + 2)
    print >> header_file, '%s_CI="%i"' % (legend[i1][0], i0 + 3)

is_first = True

for (key, dataset) in sorted(master.iteritems()):
    if not is_first: 
        print >> output_file
        print >> output_file
    else:
        is_first = False

    print >> output_file, "\"%i μs, %i tasks\"" % (key[DELAY], key[TASKS])

    # iv is a list, dvs is a list of lists.
    for (iv, dvs) in sorted(dataset.iteritems()):
        for e in iv:
            print >> output_file, e,

        # Samples
        print >> output_file, len(dvs),

        for i in range(0, number_of_dvars):
            values = []
            for j in range(0, len(dvs)): 
                values.append(float(dvs[j][i]))

            stdev = std(values)
            ci = (1.96*stdev)/sqrt(len(dvs))
            print >> output_file, mean(values), stdev, ci,

        print >> output_file

