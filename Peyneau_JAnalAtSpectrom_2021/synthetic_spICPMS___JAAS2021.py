# -*- coding: utf-8 -*-
"""
Written by Pierre-Emmanuel Peyneau
This version: 2021-08-26
License: CC-BY 4.0 <URL: http://creativecommons.org/licenses/by/4.0/ >
"""

"""
Program generating synthetic sp-ICP-MS time scans

Unit of time = dwell time
"""

"""
Function <synthetic_exp>
Produces a single synthetic time scan
    Inputs:
        <ratep>: rate of the Poisson process (expressed in dwell time units) related to the starts of the particle events.
        <Nreadings>: number of readings of the synthetic time scan.
        <Delta_distribution>: list defining the probability distribution for the duration of particle events and related parameters.
        <eventdetection>: no (eventdetection = 0), yes (otherwise).
        <seedsim>: seed of the random generator.
    Outputs:
        With event detection:
            <Nspikes>: number of spikes.
            <Nspikes_theory_discr>: average number of spikes according to the theory.
            <Nspikes_theory_cont>: average number of spikes according to the theory for a zero dwell time.
            <Nparticles>: number of nanoparticles.
            <eventduration_mean>: average duration (in dwell time units) of a spike
        Without event detection:
            <Nspikes>: number of spikes.
            <Nspikes_theory_discr>: average number of spikes according to the theory.
            <Nspikes_theory_cont>: average number of spikes according to the theory for a zero dwell time.
            <Nparticles>: number of nanoparticles.

Function <synthetic_exp_MC>
Produces multiple synthetic time scans and computes related statistics (Monte Carlo simulations)
    Inputs:
        <ratep>: rate of the Poisson process (expressed in dwell time units) related to the starts of the particle events, for every time scan.
        <Nreadings>: number of readings of each time scan.
        <Delta_distribution>: list defining the probability distribution for the duration of particle events and related parameters.
        <eventdetection>: no (eventdetection = 0), yes (otherwise).
        <seedinit>: initial seed of the random generator.
        <Ntrials>: number of synthetic time scans.
    Outputs:
        With event detection:
            <ratep>: input parameter <ratep>.
            <Delta_ave>: average of particle event duration (input parameter contained in <Delta_distribution>).
            <Nspikes_mean>: average of the number of spikes.
            <Nspikes_std>: standard deviation of the number of spikes.
            <eventdur_mean>: average of the spike duration (in dwell time units).
            <eventdur_std>: standard deviation of the spike duration (in dwell time units).
        Without event detection:
            <ratep>: input parameter <ratep>.
            <Delta_ave>: average of particle event duration (input parameter contained in <Delta_distribution>).
            <Nspikes_mean>: average of the number of spikes.
            <Nspikes_std>: standard deviation of the number of spikes.

Examples:
    Simulation of a single 60 s time scan stemming from an average of 1000 nanoparticles, with a dwell time equal to 0.1 ms, a constant particle event duration of 0.9 ms and with particle detection:
        synthetic_exp(0.00167, 600000, ["dirac", 9], 1, 1)

    Simulation of a single 60 s time scan stemming from an average of 1000 nanoparticles, with a dwell time equal to 0.1 ms, a Gamma-distributed particle event duration of average 0.9 ms and standard deviation 0.3 ms, and without particle detection:
        synthetic_exp(0.00167, 600000, ["gamma", 9, 3], 0, 1)

    Monte Carlo simulation of 1,000 60 s time scans stemming from an average of 6,000 nanoparticles, with a dwell time equal to 0.1 ms, a constant particle event duration of 0.3 ms and with particle detection:
        synthetic_exp_MC(0.01, 600000, ["dirac", 3], 1, 1, 1000)
"""

import math
import random
import numpy as np



class Event:
    def __init__(self, t1, t2, countcum):
        self.tstart = t1
        self.tend = t2
        self.countcum = countcum
        self.above_zero = (self.countcum >= 0.)
    def __str__(self):
        return (f"Starts at: {self.tstart}\nEnds at: {self.tend}\n"
               +f"Cumulated counts: {self.countcum}\n"
               +f"Is above zero? {str(self.above_zero)}\n\n")
    def tave(self):
        return 0.5*(self.tstart + self.tend)
    def twidth(self):
        return self.tend - self.tstart
    def n_dwelltime(self):
        return self.tend - self.tstart + 1
    
    
    
def poisson_process(rate, maxtime):
    pptimes = []
    tmax = 0
    while(tmax <= maxtime):
        u = random.uniform(0, 1)
        dt = -(1./rate) * math.log(1-u) # simulation of an exponential distribution with parameter <rate>
        tmax += dt
        pptimes.append(tmax)
    pptimes.pop()   # the last element of this list is necessarily > maxtime ; it is thus erased
    return pptimes


def param(Delta_ave, Delta_std, Delta_distribution):
    if(Delta_distribution == "gamma"):
        k = (Delta_ave / Delta_std)**2
        theta = (Delta_std**2) / Delta_ave
        return k, theta
    if(Delta_distribution == "inversegaussian"):
        mu = Delta_ave
        lmbd = Delta_ave**3 / Delta_std**2
        return mu, lmbd
    

def peak_detection(count, background):
    start, end = [], []
    # count_extended is used to detect spikes at the very beginning or at the very end of a time scan
    count_extended = count[:]
    count_extended[0:0] = [-1]
    count_extended.append(-1)
    for i in range(1, len(count)+1):
        if (count_extended[i] > background and
            count_extended[i-1] <= background):
            start.append(i-1)
        if (count_extended[i] > background and
            count_extended[i+1] <= background):
            end.append(i-1)
    return start, end


def peak_cumulated(count, start, end):
    countcum = []
    for i in range(len(start)):
        cum = 0.
        for j in range(start[i], end[i]+1):
            cum += count[j]
        countcum.append(cum)
    return countcum



def synthetic_exp(ratep, Nreadings, Delta_distribution, eventdetection, seedsim):
    
    if(seedsim != -1):
        random.seed(seedsim)
    
    ppjumps = poisson_process(ratep, Nreadings)
    
    # Number of nanoparticles
    Nparticles = len(ppjumps)
    
    
    if(Delta_distribution[0] == "dirac"):
        Delta_ave = Delta_distribution[1]
    elif(Delta_distribution[0] == "gamma" or Delta_distribution[0] == "inversegaussian"):
        Delta_ave = Delta_distribution[1]
        Delta_std = Delta_distribution[2]
    elif(Delta_distribution[0] == "observed"):
        population = Delta_distribution[1]
        weights = Delta_distribution[2]
        Delta_ave = 0
        for i in range(len(Delta_distribution[1])):
            Delta_ave += Delta_distribution[1][i] * Delta_distribution[2][i]
        
    
    # Possible probability distributions for the duration of particle events: Dirac | gamma | inverse Gaussian | empiric (aka 'observed')
    if(Delta_distribution[0] == "dirac"):
        timepeak = [Delta_ave] * Nparticles
    if(Delta_distribution[0] == "gamma"):
        k, theta = param(Delta_ave, Delta_std, Delta_distribution[0])
        timepeak = np.random.gamma(k, theta, Nparticles)
    if(Delta_distribution[0] == "inversegaussian"):
        mu, lmbd = param(Delta_ave, Delta_std, Delta_distribution[0])
        timepeak = np.random.wald(mu, lmbd, Nparticles)
    if(Delta_distribution[0] == "observed"):
        timepeak = random.choices(population, weights, k=Nparticles)

    
    Delta_max = int(max(timepeak)) + 1

    cstval = 1. # hard-coded: can be modified at will
    
    if(eventdetection):
        # Synthetic time scan initialization
        # Only the first Nreadings first readings belong to the 'true' time scan
        synthetic_tra = [0] * (Nreadings + Delta_max);
    
        for j1 in range(len(ppjumps)):
            jumptime = ppjumps[j1]
            peaktime = timepeak[j1]
            # WARNING: timepeak can be 0 if the distribution of time peaks is very skewed
            if(timepeak[j1] == 0):
                raise Exception("Problem: timepeak = 0.")
            f = cstval / peaktime
            I1 = int(jumptime)
            I2 = int(jumptime + peaktime)
            if(I2 > I1 + 1):
                synthetic_tra[I1] += f * (I1 + 1 - jumptime)
                for j2 in range(I1+1, I2):
                    synthetic_tra[j2] += f
                synthetic_tra[I2] += f * (jumptime + peaktime - I2)
            elif(I2 == I1 + 1):
                synthetic_tra[I1] += f * (I1 + 1 - jumptime)
                synthetic_tra[I2] += f * (jumptime + peaktime - I2)
            else:   # (when I2 == I1)
                synthetic_tra[I1] += f*peaktime
        
        # We keep only the readings belonging to the 'true' time scan
        synthetic_tra = synthetic_tra[0:Nreadings]
    
        # Spike detection
        start_list, end_list = [], []
        start_list, end_list = peak_detection(synthetic_tra, 0)
    
        # Cumulated count in each spike
        countcum_list = []
        countcum_list = peak_cumulated(synthetic_tra, start_list, end_list)
    
        # Events
        event_list = []
        time_list = list(range(Nreadings))
        for i in range(len(start_list)):
            e = Event(time_list[start_list[i]], time_list[end_list[i]], countcum_list[i])
            event_list.append(e)
            
        # Number of detected spikes
        Nspikes = len(event_list)
        
        # Duration of each spike
        eventduration_list = []
        for i in range(len(event_list)):
            eventduration_list.append(event_list[i].n_dwelltime())
        
        # Average of the spike duration
        eventduration_mean = sum(eventduration_list)/len(eventduration_list)
    
    else:
        synthetic_bool = [False] * (max(Nreadings + Delta_max, Nreadings + 2));
        
        for j1 in range(len(ppjumps)):
            jumptime = ppjumps[j1]
            peaktime = timepeak[j1]
            I1 = int(jumptime)
            I2 = int(jumptime + peaktime)
            if(I2 > I1 + 1):
                synthetic_bool[I1] |= True 
                for j2 in range(I1+1, I2):
                    synthetic_bool[j2] |= True
                synthetic_bool[I2] |= True
            elif(I2 == I1 + 1):
                synthetic_bool[I1] |= True
                synthetic_bool[I2] |= True
            else:   # (when I2 == I1)
                synthetic_bool[I1] |= True
        
        synthetic_bool = synthetic_bool[0:Nreadings]
        synthetic_bool[-1] = False
        synthetic_bool.append(False)
        
        Nchanges = 0
        for i in range(Nreadings):
            if(synthetic_bool[i+1] ^ synthetic_bool[i]):
                Nchanges += 1
        Nspikes = (Nchanges+1)//2   # +1 to count the last event when it spans up to the very end of the time scan
    
    
    # Average number of spikes if the dwell time was equal to zero (Eq. 3 of the article <Number of spikes in single particle ICP-MS time scans:from the very dilute to the highly concentrated range>)
    Nspikes_theory_cont = ratep*Nreadings*math.exp(-Delta_ave*ratep)
 
    # Average number of spikes taking into account the nonzero dwell time (Eq. 4 of the article <Number of spikes in single particle ICP-MS time scans:from the very dilute to the highly concentrated range>)
    Nspikes_theory_discr = (math.exp(-2*ratep) * (math.exp(ratep)-1)/ratep) * ratep*Nreadings*math.exp(-Delta_ave*ratep)
    
    if(eventdetection):
        return Nspikes, Nspikes_theory_discr, Nspikes_theory_cont, Nparticles, eventduration_mean
    else:
        return Nspikes, Nspikes_theory_discr, Nspikes_theory_cont, Nparticles



def synthetic_exp_MC(ratep, Nreadings, Delta_distribution, eventdetection, seedinit, Ntrials):
    Nspikes = []
    eventdur = []
    Nparticles = []
    
    if(Delta_distribution[0] == "dirac" or Delta_distribution[0] == "gamma" 
       or Delta_distribution[0] == "inversegaussian"):
        Delta_ave = Delta_distribution[1]
    elif(Delta_distribution[0] == "observed"):
        Delta_ave = 0
        for i in range(len(Delta_distribution[1])):
            Delta_ave += Delta_distribution[1][i] * Delta_distribution[2][i]
    
    for itrial in range(Ntrials):
        result_trial = synthetic_exp(ratep, Nreadings, Delta_distribution, 
                                     eventdetection, seedinit + itrial)
        Nspikes.append(result_trial[0])
        Nparticles.append(result_trial[3])
        if(eventdetection):
            eventdur.append(result_trial[4])
        
    Nspikes_mean = sum(Nspikes)/Ntrials
    Nspikes_std = np.std(Nspikes)
    
    if(eventdetection):
        eventdur_mean = sum(eventdur)/Ntrials
        eventdur_std = np.std(eventdur)
    
    if(eventdetection):
        print(f"{ratep} ; {Delta_ave} ; {Nspikes_mean} ; {Nspikes_std} ; {eventdur_mean} ; {eventdur_std}")
    else:
        print(f"{ratep} ; {Delta_ave} ; {Nspikes_mean} ; {Nspikes_std}")    