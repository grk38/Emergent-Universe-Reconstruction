import config

import numpy as np
from sympy.parsing.mathematica import parse_mathematica
from sympy import I, N
from sympy import Function
from sympy import symbols
from scipy.optimize import show_options
import sympy as sym
import scipy as sp
import math
import random
import time
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Process
from multiprocessing import Pool

global python_expr_r, python_expr_nS, function_symbols
python_expr_r = parse_mathematica(config.str_r)
python_expr_nS = parse_mathematica(config.str_nS)
function_symbols = config.input_symbols
#We turn ghe expression from mathematica code into python form

#The Planck 2018 constaints
rlim = 0.064
nSmin = 0.962514 - 0.00406408 #Modify or update them as needed 
nSmax = 0.962514 + 0.00406408


def nS(*args): # !!! The first input must always be the number of efolds. Eitherwise you would need to modify the score function
    global python_expr_nS, function_symbols
    expr_subs= python_expr_nS.subs(list(zip(function_symbols, args)))
    if expr_subs.is_real:
        return expr_subs.evalf() 
    else: return float('inf') #This is done to ensure , that imaginary results are quickly ruled out

    
def r(*args): 
    global python_expr_r, function_symbols
    expr_subs= python_expr_r.subs(list(zip(function_symbols, args)))
    if expr_subs.is_real:
        return expr_subs.evalf() 
    else: return float('inf')


Nmax, Nmin = 60, 50
class Solution:
    def __init__(self, genome):
        self.genome = genome
    
    def score(self):
        nS_value = nS(*self.genome)
        r_value = r(*self.genome)
        efolds = self.genome[0]
        dev_nS = abs(nS_value-nSmin)+abs(nS_value-nSmax)
        dev_r = abs(r_value-rlim)
        dev_N = abs(efolds-Nmin)+abs(efolds-Nmax)
        scoring = dev_r*dev_r + dev_nS*100 + dev_N*0.1 #Another possible substitution is dev_r*c, with the constant c being appropriate for your own specific problem.This method is computationally faster
        return scoring                                 #,but keep in mind that the more weight is put in dev_r the more likely the algorithm will try to minimize it, whilst ignoring the validity
                                                       # of the other constraints , turning into a "greedy" optimum instead of a local or global one
    #Thats the way i defined the fitness function . You could modify it as you like depending on your own problem


def generator(b):
    n = len(b)
    output = np.zeros(n)
    for index in range(n):
        lower_bound = b[index][0]
        upper_bound = b[index][1]
        l_b = math.log10(abs(lower_bound))
        u_b = math.log10(abs(upper_bound))
        cons_sign = -1 if lower_bound < 0 or upper_bound < 0 else 1
        output[index] = (10**(random.uniform(l_b,u_b)))*cons_sign
    return output
#The generator function is large-number-biased for inputs that differ more that 3 orders of magnitude. This snippet of code quarantes that there is at least one point between two concecutive orders
#of magnitude


def mutate(gene): # The mutating function
    starting_value = gene
    s = 1 #Choose this as you please. An interesting idea would be for the random variable to belong in a standard
#distribution with s defining the std. I tried it but the convergence rate was worse, along with being more computationally
#expensive
    random_value = np.random.uniform(-starting_value/s,starting_value/s)
    new_gene = starting_value + random_value
    return new_gene


#The code below is for multiprocessing
def calculate_score(obj):
    return obj.score()

def parallel(var_population):
    num_processes = multiprocessing.cpu_count()
    scores = np.zeros(len(var_population))
    objects = var_population.copy()
    num_objects = objects.shape[0]

    pool = multiprocessing.Pool(processes=num_processes) #This line will make sure python uses 100% of your CPU. If you dont want that you can reduce the number of processes
    results = pool.map(calculate_score, objects) 
    pool.close()
    pool.join()

    scores = np.array(results)

    return scores



def genetic_algo_start(bounds, size_population, fit_lim, iterations, loaded_population=None):
    #Making the first population manually
    global pops
    global best_pops
    global best_scores
    fit_lim = fit_lim+1
    best_scores=np.zeros((iterations,fit_lim+1))
    pops = np.zeros(iterations+1, dtype='object')
    best_pops = np.zeros(iterations+1, dtype='object')
    global population
    population = np.empty(size_population, dtype='object')  #The last parameter is necessary as numpy gets cranky
    #if we dont specify we are dealing with objects
    if loaded_population is None:
        for i in range(size_population):
            var_genome_0 = generator(bounds)
            sol = Solution(var_genome_0)
            population[i] = sol  
    elif isinstance(loaded_population, np.ndarray) and loaded_population.dtype == object:
        population = loaded_population[-1]
    
    pops[0]=population

    #Start loop
    
    for generation in range(iterations):
        #Initializing
        t1 = time.time()
        scoreboard = np.zeros(size_population)
        population = pops[generation]
        
        #Scoring
        scoreboard = parallel(population)
            
        best_candidates_pos = np.argpartition(scoreboard, fit_lim+1)[:fit_lim+1]
        #Fitness
        best_candidates = np.zeros(fit_lim+1,dtype='object')
        #Keeping the best players
        for index_position, position in enumerate(best_candidates_pos):
            var = population[position]
            best_candidates[index_position] = var
        #Keeping the score of the best players, so that we dont calculate it all over again.
        for index, pos in enumerate(best_candidates_pos):
            best_scores[generation][index] = scoreboard[pos]


        best_pops[generation]=best_candidates.copy()
        new_population = np.zeros(size_population, dtype='object')
        genetic_tree = np.zeros((size_population,len(bounds)))
        ranges = np.linspace(0,size_population, fit_lim+2) 
        #Crossover
        for indicator in range(len(ranges)-1): #The players are equally distributed in the population
            lower = math.floor(ranges[indicator])
            upper = math.floor(ranges[indicator+1] )
            var_genome = best_candidates[indicator].genome
            for index in range(int(lower),int(upper),1):
                genetic_tree[index]=var_genome

        #Mutate       
        new_population = np.array([Solution(np.array([mutate(gene) for gene in tree])) for tree in genetic_tree])
        #Keeping the best players of the previous generation
        for i in range(len(best_candidates)):
            new_population[i]=best_candidates[i]
        
        pops[generation+1]= new_population
        t2=time.time()
        elapsed_time =t2-t1
        print("Generation : "+str(generation+1)+" took {:.3f} seconds to finish".format(elapsed_time))
        print("***Time remaining: {:.3f} minutes***".format((iterations-generation-1)*elapsed_time/(60))) 
    #Finish for loop
    best_pops = best_pops[:-1] # The last element is zero , so we need to get rid of it
    print("Code finished compiling")
    return pops, best_pops, best_scores #It returns the populations of all the players, the populations of all the best players along with their scores


def clear_points(arr):
    values = np.empty( len(arr)*len(arr[0]), dtype=object)
    nS_arr = np.zeros(len(values))
    r_arr = np.zeros(len(values))
    index = 0
    for row in arr:
        for obj in row :
            test_nS = nS(*obj.genome)
            test_r = r(*obj.genome)
            if (test_nS>nSmin and test_nS<nSmax and test_r<rlim):
                values[index] = obj.genome
                nS_arr[index] = test_nS
                r_arr[index] = test_r
                index +=1
    values = values[0:index:1]
    nS_arr = nS_arr[0:index:1]
    r_arr = r_arr[0:index:1]
    return values, nS_arr, r_arr

def style(ax):
    plt.style.use('bmh')
    ax_names = ['top','right','bottom','left']
    for name in ax_names:
        ax.spines[name].set_linewidth(1.5)
        ax.spines[name].set_color('#333333')
    pass