
# patternRecFigs.py
#
# The goal of this file is to show some pattern recognition ideas
#
# Seth McNeill
# 2022 March 29

import argparse  # for parsing input arguments
import datetime  # for time related functions
import numpy as np  # for datastructures
import matplotlib.pyplot as plt  # for actual plotting
import random  # for random sampling
import pdb  # for debugging
import sys  # to quit early (exit)
import pandas as pd  # for importing csv files


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', help='Name of file containing audio data as one number per line at 20kHz sample rate')
#    parser.add_argument('--env', help='Name of file containing environmental data')
    args = parser.parse_args()
    start_time = datetime.datetime.now()

    data = pd.read_csv(args.raw)
    sampleRate = 20000  #Hz

    #t = range(0,len(data)) #*1/sampleRate
    t = np.arange(0,len(data)*1/sampleRate,1/sampleRate)
    fig, ax = plt.subplots(figsize=(11,8.5))
    plt.rcParams['font.size'] = '18'
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(18) 
    plt.plot(data['time'], data['data'], label='Audio', color='b')
    plt.grid()
    plt.xlabel('Time (s)', fontsize=20)
    plt.ylabel('Audio Amplitude', fontsize=20)
    plt.title('One Second Recording')
#    plt.legend()
#    plt.savefig('posterior.pdf')
#    plt.savefig('posterior.png')
    plt.show()

    end_time = datetime.datetime.now()
    print(f"{__file__} took {end_time - start_time} s")