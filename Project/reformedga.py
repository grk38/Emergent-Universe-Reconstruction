import numpy as np
from sympy.parsing.mathematica import parse_mathematica
from sympy import I, N
from sympy import Function
from sympy import symbols
from scipy.optimize import show_options, curve_fit
import sympy as sym
import math
import random
import time
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Pool
from sympy import sympify
from tqdm import tqdm
import warnings

class Solution:
    def __init__(self, genome,fitness_expression):
        self.genome = genome
        self.fitness_expression = fitness_expression
       
    def score(self,other):
        nS_value = other.nS(self.genome)
        r_value = other.r(self.genome)
        efolds = self.genome[0]
        dev_nS = abs(nS_value-other.nSmin)+abs(nS_value-other.nSmax)
        dev_r = abs(r_value-other.rlim)
        dev_N = abs(efolds-other.Nmin)+abs(efolds-other.Nmax)
        scoring = eval(self.fitness_expression, {
            'dev_nS': dev_nS,
            'dev_r': dev_r,
            'dev_N': dev_N
        }) #Another possible substitution is dev_r*c, with the constant c being appropriate for your own specific problem.This method is computationally faster
        return scoring                                 #,but keep in mind that the more weight is put in dev_r the more likely the algorithm will try to minimize it, whilst ignoring the validity
                                                       # of the other constraints , turning into a "greedy" optimum instead of a local or global one
    #Thats the way i defined the fitness function . You could modify it as you like depending on your own problem


class GenAlgo:
    def __init__(self,str_nS,str_r,input_symbols,mathematica_form=True,fitness_expression='dev_r * dev_r + dev_nS * 100 + dev_N * 0.1'):
        if mathematica_form:
            self.python_expr_r = parse_mathematica(str_r)
            self.python_expr_nS = parse_mathematica(str_nS)
        else:
            self.python_expr_r = str_r
            self.python_expr_nS = str_nS

        self.function_symbols = input_symbols
        self.fitness_expression = fitness_expression
        #We turn ghe expression from mathematica code into python form

        #The Planck 2018 constaints
        self.rlim = 0.064
        self.nSmin = 0.962514 - 0.00406408 #Modify or update them as needed 
        self.nSmax = 0.962514 + 0.00406408

        #The range of the e-folds
        self.Nmax = 60
        self.Nmin = 50

        #Now to make the actual functions
        self.nSsympyFunction = sym.lambdify(self.function_symbols,self.python_expr_nS,'numpy')
        self.rsympyFunction = sym.lambdify(self.function_symbols,self.python_expr_r,'numpy')
        pass


    def nS(self,args): # !!! The first input must always be the number of efolds. Eitherwise you would need to modify the score function
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                var = self.nSsympyFunction(*args)
                return var
        except Exception as e:
            return float('inf')

   
    def r(self,args): 
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                var = self.rsympyFunction(*args)
                return var
        except Exception as e:
            return float('inf')

    #    
    def generator(self,b):
        n = len(b)
        output = np.zeros(n)
        for index, value in enumerate(b):
            lower_bound,upper_bound = value
            l_b, u_b = (math.log10(abs(lower_bound)) , math.log10(abs(upper_bound)) )
            cons_sign = -1 if lower_bound < 0 or upper_bound < 0 else 1
            output[index] = (10**(random.uniform(l_b,u_b)))*cons_sign
        return output
#The generator function is large-number-biased for inputs that differ more that 3 orders of magnitude. This snippet of code quarantes that there is at least one point between two concecutive orders
#of magnitude

       
    def mutate(self,gene): # The mutating function
        self.scale = 1 #Choose this as you please. An interesting idea would be for the random variable to belong in a standard
    #distribution with s defining the std. I tried it but the convergence rate was worse, along with being more computationally
    #expensive
        return gene + np.random.uniform(-gene/self.scale,gene/self.scale)

    
    def start(self,bounds, size_population, fit_lim, iterations, loaded_population=None):
        #Making the first population manually
        global pops
        global best_pops
        global best_scores
        fit_lim = fit_lim+1
        best_scores=np.zeros((iterations,fit_lim+1),dtype=float)
        pops = np.zeros(iterations+1, dtype='object')
        best_pops = np.zeros(iterations+1, dtype='object')
        global population
        population = np.empty(size_population, dtype='object')  #The last parameter is necessary as numpy gets cranky
        #if we dont specify we are dealing with objects
        if loaded_population is None:
            for i in range(size_population):
                var_genome_0 = self.generator(bounds)
                sol = Solution(genome=var_genome_0,fitness_expression=self.fitness_expression)
                population[i] = sol  
        elif isinstance(loaded_population, np.ndarray) and loaded_population.dtype == object:
            population = loaded_population[-1]
        if len(bounds) != len(self.function_symbols):
            raise ValueError(f"Different dimensions. Bounds dimensions : {len(bounds)} and total symbols : {self.function_symbols}")
        
        pops[0]=population
        #Start loop
        for generation in tqdm(range(iterations),dynamic_ncols=True):
            #Initializing
            # t1 = time.perf_counter () 
            scoreboard = np.zeros(size_population)
            population = pops[generation]
            #Scoring

            scoreboard = np.array([obj.score(self) for obj in population])

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
                if index == 0 and scoreboard[pos] == float('inf'):
                    raise ValueError("The best player has a score of infinity. Check the bounds or the symbols used")
            
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
            new_population = np.array([Solution(genome=np.array([self.mutate(gene) for gene in tree]),fitness_expression=self.fitness_expression) for tree in genetic_tree])
            #Keeping the best players of the previous generation
            for i in range(len(best_candidates)):
                new_population[i]=best_candidates[i]
            
            pops[generation+1]= new_population
        best_pops = best_pops[:-1] # The last element is zero , so we need to get rid of it
        print("Code finished compiling")
        return pops, best_pops, best_scores #It returns the populations of all the players, the populations of all the best players along with their scores
       
    def process_point(self,obj):
        genome = obj.genome
        test_nS = self.nS(genome)
        test_r = self.r(genome)
        if self.nSmin < test_nS < self.nSmax and test_r < self.rlim:
            return [test_nS, test_r]
        else:
            return [None,None]
       
    def clear_points(self,arr):
        flattened = np.concatenate(arr).ravel()
        values = []
        nS_arr = []
        r_arr = []
        for obj in flattened:
            returnValue = self.process_point(obj)
            if returnValue[0] is not None:
                values.append(obj.genome)
                nS_arr.append(returnValue[0])
                r_arr.append(returnValue[1])
        return values, nS_arr, r_arr


def style(ax):
    plt.style.use('bmh')
    ax_names = ['top','right','bottom','left']
    for name in ax_names:
        ax.spines[name].set_linewidth(2.5)
        ax.spines[name].set_color('#333333')
    pass

def write_txt(name, data):
    f = open(name, "w")
    for point in data : 
        f.write(str(point))
        f.write('\n')   
    f.close()
    
    pass

def read_txt(name):
    with open(name, "r") as f :
        lines = f.readlines()
    
    data = [float(line) for line in lines]
    
    return data