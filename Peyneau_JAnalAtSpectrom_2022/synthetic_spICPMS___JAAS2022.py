# -*- coding: utf-8 -*-
"""
Written by Pierre-Emmanuel Peyneau
This version: 2022-04-07
Licence: CC-BY 4.0
"""

"""
Program generating synthetic sp-ICP-MS time scan

Unit of time = dwell time
"""

"""
Function <synthetic_exp>
Produces a single synthetic time scan
    Inputs:
        <ratep>: rate of the Poisson process (expressed in dwell time units) related to the starts of the particle events.
        <Nreadings>: number of readings of the synthetic time scan.
        <Delta_distribution>: list defining the probability distribution for the duration of particle events and related parameters.
        <size_distribution>: list defining the probability distribution for the diameter of particles and related parameters.
        <seedsim>: seed of the random generator.
    Outputs:
        <Nspikes>: number of spikes.
        <Nspikes_theory_discr>: average number of spikes according to the theory.
        <Nspikes_theory_cont>: average number of spikes according to the theory for a zero dwell time.
        <Nparticles>: number of nanoparticles.
        <eventduration_mean>: average duration (in dwell time units) of a spike.
        <Np_spike_mean>: average number of particles per spike.
        <Np_spike>: list gathering the number of particle events in each spike.
        <particle_countcum_list>: list gathering the number of counts associated with each particle.
        <countcum_list>: list gathering the number of counts associated with each spike.
        
Function <synthetic_exp_MC>
Produces multiple synthetic time scans and computes related statistics (Monte Carlo simulations)
    Inputs:
        <ratep>: rate of the Poisson process (expressed in dwell time units) related to the starts of the particle events, for every time scan.
        <Nreadings>: number of readings of each time scan.
        <Delta_distribution>: list defining the probability distribution for the duration of particle events and related parameters.
        <size_distribution>: list defining the probability distribution for the diameter of particles and related parameters.
        <seedinit>: initial seed of the random generator.
        <Ntrials>: number of synthetic time scans.
        <folder_name>: name of the folder used to store some output files.
    Outputs:
        <ratep>: input parameter <ratep>.
        <Delta_ave>: average of particle event duration (input parameter contained in <Delta_distribution>).
        <Nspikes_mean>: average of the number of spikes.
        <Nspikes_std>: standard deviation of the number of spikes.
        <eventdur_mean>: average of the spike duration (in dwell time units).
        <eventdur_std>: standard deviation of the spike duration (in dwell time units).
        <Np_spike_mean_mean>: mean number of particles per spike, calculated over all Monte Carlo trials.
        <Np_spike_mean_std> standard deviation of the number of particles per spike, calculated over all Monte Carlo trials.
        
        On screen :
            Empirical distribution of the number of particles per spike and comparison with the corresponding approximate geometric distribution. 
            
        Data files:
            size-particle.dat: histogram of the real particle sizes
            size-spikes.dat: histogram of the particle sizes stemming from the raw particle size distribution that can be inferred from the sp-ICP-MS signal.
            
Example:
    Monte Carlo simulation of one thousand 60 s time scans stemming from an average of 6,000 nanoparticles, with a dwell time equal to 0.1 ms, a constant particle event duration of 0.3 ms and a Gaussian particle size distribution with mean = 1 and std = 0.1:
        synthetic_exp_MC(0.01, 600000, ["dirac", 3], ["gaussian", 1, 0.1], 1, 1000, "examplefolder")
"""

import math
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import sys



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
    pptimes.pop()   # the last element of this list is necessarily > maxtime; it is thus erased
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



def synthetic_exp(ratep, Nreadings, Delta_distribution, size_distribution, seedsim):
    
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
        
    
    # Possible probability distributions for the duration of particle events:
    # Dirac | gamma | inverse Gaussian | empiric (aka 'observed')
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
    
    if(size_distribution[0] == 'dirac'):
        size_ave = size_distribution[1]
    elif(size_distribution[0] == 'gaussian' or size_distribution[0] == 'gamma'):
        size_ave = size_distribution[1]
        size_std = size_distribution[2]
    
    # Number of counts per particle event
    # Possible probability distributions: Dirac | gaussian | gamma
    if(size_distribution[0] == 'dirac'):
        particle_countcum_list = np.array([size_ave] * Nparticles)**3
    if(size_distribution[0] == 'gaussian'):
        # To get rid of negative values
        particle_countcum_list = np.maximum(np.random.normal(size_ave, size_std, Nparticles)**3, 0)
    if(size_distribution[0] == 'gamma'):
        k, theta = param(size_ave, size_std, 'gamma')
        particle_countcum_list = np.random.gamma(k, theta, Nparticles)**3
        
    particle_countcum_list = particle_countcum_list.tolist()
    
    # Synthetic time scan initialization
    # Only the first Nreadings first readings belong to the 'true' time sca
    synthetic_tra = [0] * (Nreadings + Delta_max);
    
    for j1 in range(len(ppjumps)):
        jumptime = ppjumps[j1]
        peaktime = timepeak[j1]
        # WARNING: timepeak can be 0 if the distribution of time peaks is very skewed
        if(timepeak[j1] == 0):
            raise Exception("Probleme : timepeak = 0.")
        f = particle_countcum_list[j1] / peaktime
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
    
    # Cumulated number of counts in each spike
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
        
    # Number of particles in each spike
    Np_spike = []
    ip = 0
    for ispike in range(Nspikes):
        Np = 0
        while ((ip < Nparticles) and (ppjumps[ip] <= event_list[ispike].tend + 1)):
            Np += 1
            ip += 1
        Np_spike.append(Np)
        
    # Frequency distribution of the number of particles in each spike
    Np_spike_mean = sum(Np_spike) / Nspikes
          
    # Duration of each spike
    eventduration_list = []
    for i in range(len(event_list)):
        eventduration_list.append(event_list[i].n_dwelltime())
        
    # Numerical and theoretical average spike duration
    eventduration_mean = sum(eventduration_list)/len(eventduration_list)
    esp_cond_duration = (1/ratep)/(1- math.exp(-ratep*Delta_ave)) * (1 - (1+ratep*Delta_ave)*math.exp(-ratep*Delta_ave))
    esp_cond_duration += math.exp(-ratep*(Delta_ave+1))/(ratep**2) * (math.exp(ratep)*(ratep*Delta_ave+2) - ratep*(Delta_ave+1) - 2) - math.exp(-ratep*(Delta_ave+2))/ratep * (ratep*(Delta_ave+2)+1)
        
    # Average number of spikes if the dwell time was equal to zero (Eq. 3 of the article <Number of spikes in single particle ICP-MS time scans:from the very dilute to the highly concentrated range>)
    Nspikes_theory_cont = ratep*Nreadings*math.exp(-Delta_ave*ratep)
 
    # Average number of spikes taking into account the nonzero dwell time (Eq. 4 of the article <Number of spikes in single particle ICP-MS time scans:from the very dilute to the highly concentrated range>)
    Nspikes_theory_discr = (math.exp(-2*ratep) * (math.exp(ratep)-1)/ratep) * ratep*Nreadings*math.exp(-Delta_ave*ratep)
    
    return Nspikes, Nspikes_theory_discr, Nspikes_theory_cont, Nparticles, eventduration_mean, Np_spike_mean, Np_spike, particle_countcum_list, countcum_list



def synthetic_exp_MC(ratep, Nreadings, Delta_distribution, size_distribution, seedinit, Ntrials, folder_name):
    Nspikes = []
    eventdur = []
    Nparticles = []
    Np_spike_mean = []
    Np_spike_cum = []
    
    MARGINCOEFF = 3
    NBINS = 1000 * MARGINCOEFF
    MINVALpc = MINVALc = 0.
    particle_size_binned = [0]*NBINS
    spike_size_binned = [0]*NBINS
    
    eachtrial = 0
    
    try:
        os.mkdir(folder_name)
    except:
        message = ' Folder '+folder_name+' already exists. '
        print('///'+' '*len(message)+'\\\\\\')
        print('|||'+message+'|||')
        print('\\\\\\'+' '*len(message)+'///'+'\n')
    
    if(Delta_distribution[0] == "dirac" or Delta_distribution[0] == "gamma" 
       or Delta_distribution[0] == "inversegaussian"):
        Delta_ave = Delta_distribution[1]
    elif(Delta_distribution[0] == "observed"):
        Delta_ave = 0
        for i in range(len(Delta_distribution[1])):
            Delta_ave += Delta_distribution[1][i] * Delta_distribution[2][i]
    
    for itrial in range(Ntrials):
        np.random.seed(seedinit + itrial)
        result_trial = synthetic_exp(ratep, Nreadings, Delta_distribution, 
                                     size_distribution, seedinit + itrial)
        Nspikes.append(result_trial[0])
        Nparticles.append(result_trial[3])
        eventdur.append(result_trial[4])
        Np_spike_mean.append(result_trial[5])
        if (eachtrial):
            print(result_trial[0])
        
        Np_spike = result_trial[6]
        particle_countcum = result_trial[7]
        countcum = result_trial[8]
        # Initialization
        if(itrial == 0):
            Np_spike_max = MARGINCOEFF*max(Np_spike)
            Np_spike_cum = [0]*(Np_spike_max + 1)
            MAXVALpc = MARGINCOEFF*max([x**(1./3) for x in particle_countcum])
            MAXVALc = MARGINCOEFF*max([x**(1./3) for x in countcum])
            BINWIDTHpc = (MAXVALpc - MINVALpc) / NBINS
            BINWIDTHc = (MAXVALc - MINVALc) / NBINS
        for i in range(1, Np_spike_max + 1):
            Np_spike_cum[i] += Np_spike.count(i)
        # We assume that the particles are spherical, hence the power one-third
        for elem in particle_countcum:
            binnum = int((elem**(1./3) - MINVALpc) // BINWIDTHpc)
            binindex = min(NBINS-1, binnum)
            particle_size_binned[binindex] += 1
        for elem in countcum:
            binnum = int((elem**(1./3) - MINVALc) // BINWIDTHc)
            binindex = min(NBINS-1, binnum)
            spike_size_binned[binindex] += 1        
        
    Nspikes_mean = sum(Nspikes)/Ntrials
    Nspikes_std = np.std(Nspikes)
    eventdur_mean = sum(eventdur)/Ntrials
    eventdur_std = np.std(eventdur)
    Np_spike_mean_mean = sum(Np_spike_mean)/Ntrials
    Np_spike_mean_std = np.std(Np_spike_mean)
    
    print(f"{ratep} ; {Delta_ave} ; {Nspikes_mean} ; {Nspikes_std} ; {eventdur_mean} ; {eventdur_std} ; {Np_spike_mean_mean} ; {Np_spike_mean_std}")
    
    geomparam = (math.exp(-2*ratep) * (math.exp(ratep)-1)/ratep) * math.exp(-Delta_ave*ratep)
    # Get rid of all the zeros at the end of the lists <Np_spike_cum>, <particle_countcum_binned> and <countcum_binned>
    for value in Np_spike_cum[::-1]:
        if(value == 0):
            Np_spike_cum.pop()
        else:
            break
    for value in particle_size_binned[::-1]:
        if(value == 0):
            particle_size_binned.pop()
        else:
            break
    for value in spike_size_binned[::-1]:
        if(value == 0):
            spike_size_binned.pop()
        else:
            break
        
    print("\n=== FREQUENCY DISTRIBUTION OF #PARTICLES/SPIKE: EMPIRICAL VS. GEOMETRIC DISTRIBUTION ===")
    for i in range(1, len(Np_spike_cum)):
        print(i, "   ", Np_spike_cum[i]/sum(Nspikes), "   ", (1 - geomparam)**(i-1) * geomparam)
    print('\n\n')
        
    fsizep = open(os.path.join(folder_name, 'size-particles.dat'), 'w')
    for i in range(len(particle_size_binned)):
        fsizep.write(str(i*BINWIDTHpc)+'   '+str(particle_size_binned[i]))
        fsizep.write('\n')
    fsizep.close()
    fsizes = open(os.path.join(folder_name, 'size-spikes.dat'), 'w')
    for i in range(len(spike_size_binned)):
        fsizes.write(str(i*BINWIDTHc)+'   '+str(spike_size_binned[i]))
        fsizes.write('\n')
    fsizes.close()
    