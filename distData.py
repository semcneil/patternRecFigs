#
#
# distData.py
#
# Plots distance data from the CEC 326 board using an LED on/off and CdS 
# light sensitive resistor to estimate distance.
#
# Seth McNeill
# 2022 November 04

import argparse  # for parsing input arguments
import datetime  # for time related functions
import numpy as np  # for datastructures
import matplotlib.pyplot as plt  # for actual plotting
import random  # for random sampling
import pdb  # for debugging
import sys  # to quit early (exit)
import pandas as pd  # for importing csv files
import glob  # for file seraching
from scipy.stats import norm


# https://stackoverflow.com/a/22579904
def normalIntersection(m1,m2,std1,std2):
  """Finds the intersection point(s) between normal distributions"""
  a_coeff = 1/(2*std1**2) - 1/(2*std2**2)
  b_coeff = m2/(std2**2) - m1/(std1**2)
  c_coeff = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1)
  return np.roots([a_coeff,b_coeff,c_coeff])

def plotClasses(fileGlobStr):
    """Plots files that match fileGlobStr"""
    files = glob.glob(fileGlobStr)
    frames = []
    for file in files:
        frames.append(pd.read_csv(file))
    
    x_axis = np.arange(-100,10000, 1)

    fig, ax = plt.subplots(figsize=(11,8.5))
    plt.rcParams['font.size'] = '18'
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(18)
    #plt.hist(frames, bins=counts[1], label=['Salmon','Seabass'], color=['b','r'])
    means = []
    stds = []
    for frame in frames:
        means.append(frame.mean())
        stds.append(frame.std())
        plt.hist(frame)
        plt.plot(x_axis,4000*norm.pdf(x_axis, means[-1], stds[-1]))
    plt.grid()
    plt.xlabel('LED On/Off Difference', fontsize=20)
    plt.ylabel('Count', fontsize=20)
    plt.title('Histogram of Light Intensity Differences')
    #plt.legend()
    plt.savefig('distHist.pdf')
    plt.savefig('distHist.png')
    plt.show()
 
    fig, ax = plt.subplots(figsize=(11,8.5))
    plt.rcParams['font.size'] = '18'
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(18)
#    for m,s in zip(means,stds):
    for ii in range(len(means)):
        plt.plot(x_axis,norm.pdf(x_axis, means[ii], stds[ii]))
        if ii > 0:
            crossings = normalIntersection(means[ii][0],means[ii-1][0],stds[ii][0],stds[ii-1][0])
            print(f'Optimal thresholds between {ii-1} and {ii} is at {crossings}')
            plt.plot(crossings, norm.pdf(crossings, means[ii],stds[ii]), 'o')
    plt.grid()
    plt.xlabel('LED On/Off Difference', fontsize=20)
    plt.ylabel('Likelihood', fontsize=20)
    plt.title('Normal Distributions of Light Intensity Differences')
    #plt.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', help='Double quoted glob string for files of interest')
    args = parser.parse_args()
    start_time = datetime.datetime.now()

    plotClasses(args.files)

    end_time = datetime.datetime.now()
    print(f"{__file__} took {end_time - start_time} s")