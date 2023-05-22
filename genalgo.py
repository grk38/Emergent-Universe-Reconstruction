
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

global python_expr_r, python_expr_nS
python_expr_r = None
python_expr_nS = None

def func_initialize(a, b):
    global python_expr_nS, python_expr_r
    r_expr = parse_mathematica(a)
    nS_expr = parse_mathematica(b)
    python_expr_r = r_expr
    python_expr_nS = nS_expr
    # print("The r function in the language of Python is\n" + str(r_expr))
    # print("--------------------------------------------------------------------------------------------------------------------")
    # print("The nS function in the language of Python is\n" + str(nS_expr))
    pass



#Converting from Mathematica to Python
#The Planck 2018 constaints
rlim = 0.064
nSmin = 0.962514 - 0.00406408 #Modify them as needed
nSmax = 0.962514 + 0.00406408
#Converting the mathematica expressions to python. Just paste them below. !!!Copy them as plain text!!!

#print("--------------------------------------------------------------------------------------------------------------------")
#print("The nS function in the language of Python is\n"+str(python_expr_nsfinal))
# r_string = "Ne1"
# nS_string = "Ne1"
# update()

def nS(Ne1_val,n_val,b_val,c1_val,c2_val):
    global python_expr_nS
    Ne1, n, b, c1, c2 = sym.symbols('Ne1 n b c1 c2')
    expr = python_expr_nS
    expr_subs = expr.subs([(Ne1, Ne1_val), (n, n_val), (b, b_val), (c1,c1_val ), (c2, c2_val)])
    if expr_subs.is_real:
        return expr_subs.evalf() 
    else: return float('inf') #This is done to ensure , that imaginary results are quickly ruled out

    
def r(Ne1_val,n_val,b_val,c1_val,c2_val):
    global python_expr_r
    Ne1, n, b, c1, c2 = sym.symbols('Ne1 n b c1 c2')
    expr = python_expr_r
    expr_subs = expr.subs([(Ne1, Ne1_val), (n, n_val), (b, b_val), (c1,c1_val ), (c2, c2_val)])
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
        #print(nS_value,r_value)
        dev_nS = abs(nS_value-nSmin)+abs(nS_value-nSmax)
        dev_r = abs(r_value-rlim)
        dev_N = abs(efolds-Nmin)+abs(efolds-Nmax)
        scoring = dev_r*dev_r+dev_nS*100 +dev_N*0.1
        return scoring
    #Thats the way i defined the fitness function . You could modify it as you like depending on the order of your parameters

#bounds = np.array([(50,60),(-100,-1),(1,10**5),(1,10),(1,10)]) #The bounds for each variable. !!!They must exist!!! Usually for N we want 50 - 60
#Put them in ascending order meaning for N you must input (50,60).Try not to input zero. I dont think it will break the code
#it can handle infinities , but just to be safe from any overflowing issues
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

def param_initialize(a,b=1000,c=4,d=5):
    global bounds
    bounds = a.copy()
    global size_population
    global fit_lim
    global iterations
    size_population = b
    fit_lim = c
    iterations = d 
    pass



#Making our population
#size_population = 1000 #The complexity is O(size_population * iterations).Half the population , half the time
#fit_lim =4 # How many of the top condidates we are going to keep-1.This happened because i was not paying attention.
#therefore from now on, it's a "feature"
#!!!Its important to note that a smaller fit_lim produces a larger computational time. It may be a second or dozens of them
#depending on a lot of factors. Its so bizarre.
#iterations = 10 # How many iterations you want the code to run.Practically at 2-5 iterations the algorithm has convergeδ

#The most important aspects would be the bounds


def mutate(gene): # The mutating function
    starting_value = gene
    s = 1 #Choose this as you please. An interesting idea would be for the random variable to belong in a standard
#distribution with s defining the std. I tried it but the convergence rate was worse, along with being more computationally
#expensive
    random_value = np.random.uniform(-starting_value/s,starting_value/s)
    new_gene = starting_value + random_value
    return new_gene




def calculate_score(obj):
    return obj.score()

def parallel(var_population):
    num_processes = multiprocessing.cpu_count()
    scores = np.zeros(len(var_population))
    objects = var_population.copy()
    num_objects = objects.shape[0]

    pool = multiprocessing.Pool(processes=num_processes)
    results = pool.map(calculate_score, objects)
    pool.close()
    pool.join()

    scores = np.array(results)

    return scores



def genetic_algo_start():
    global bounds
    global size_population
    global fit_lim
    global iterations
    global python_expr_nS
    global python_expr_r
    #Making the first population manually
    global pops
    global best_pops
    global best_scores
    best_scores=np.zeros((iterations,fit_lim+1))
    pops = np.zeros(iterations+1, dtype='object')
    best_pops = np.zeros(iterations+1, dtype='object')
    global population
    population = np.empty(size_population, dtype='object')  #The last parameter is necessary as numpy gets cranky
    #if we dont specify we are dealing with objects
    for i in range(size_population):
        var_genome_0 = generator(bounds)
        sol = Solution(var_genome_0)
        population[i] = sol  
    pops[0]=population

    #Now i shall make the code iterative so that we can run it for numerous generations
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
        for indicator in range(len(ranges)-1):
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
    return pops, best_pops, best_scores

    
# def print_red(text):
#     print('\033[1;91m' + text + '\033[0m')

# print_red("Successfully launched the Python script")
# print("Initialize the functions and parameters of the algorithm. For help, call the function help()")

def help():
    while True:
        varinput = int(math.floor(float(input("Enter 0 for the nS and r initialization, 1 for the other parameters and bounds and 2 on how to start the algorithm:"))))
        print("")
        if varinput == 0:
            print("The function func_initialize takes two string arguments, the r and ns functions in that order.\nThe arguments must be strings that contain code in Mathematica language")
            print("In the case you have other symbols in your Mathematica code , you must open the script in text.editor\n and edit them yourself.Although tedious it is the only way to make the script\n not present issues.")
            break
        elif varinput == 1:
            print("The param_initialize function takes four arguments.The first one is the bounds of our algorithm in the form np.array([(),(),()]).\n The tuples must be in ascending order and preferably not zero")
            print("The other arguments are the size of the population , how many of the best players we keep in each\ngeneration-1 and lastly the number of generations we want the code to run")
            print("Their default values are 1000, 4 (meaning 5 best players in each generation) and 5")
            break
        elif varinput == 2:
            print("To start the algorithm you need to call the genetic_algo_start() function. It returns a tuple of all the populations and the best players in each generation\nIt is really important that you include if __name__ == '__main__': on top of it, else multiprocessing and jupyter are not compatible")
            break
        else:
            print("Not an option.Try again")



def clear(arr):
    values = []
    var_i = 0
    for row in arr:
        for obj in row:
            test_arr = obj.genome
            varns = nS(*test_arr)
            varr = r(*test_arr)

            if varr < rlim and varns > nSmin and varns < nSmax:
                values.append(test_arr)
                # print(var_i)
                var_i = var_i +1

    return np.array(values)

# strr = "(12 b^4 E^(-((4 Ne1)/n)) (-1+n)^4 ((2 E^(Ne1/n) (-1+(E^(-(Ne1/n)) (-1+n))/n) n^2)/(b^2 (-1+n))-(2 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^3)/(b^2 (-1+n)^2))^2)/((-1+(E^(-(Ne1/n)) (-1+n))/n)^4 n^8 (-1-(1/(2 (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^4))b^2 E^(-((2 Ne1)/n)) (-1+n)^2 ((2 E^(Ne1/n) (-1+(E^(-(Ne1/n)) (-1+n))/n) n^2)/(b^2 (-1+n))-(2 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^3)/(b^2 (-1+n)^2)) (-1-(24 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^4 (1/4 c1 (3-n-Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]) ((24 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^4)/(b^2 (-1+n)^2)+6 ((2 E^(Ne1/n) (-1+(E^(-(Ne1/n)) (-1+n))/n) n^2)/(b^2 (-1+n))-(2 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^3)/(b^2 (-1+n)^2)))^(-3+1/4 (3-n-Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)])) (-2+1/4 (3-n-Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)])) (-1+1/4 (3-n-Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]))+1/4 c2 (3-n+Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]) ((24 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^4)/(b^2 (-1+n)^2)+6 ((2 E^(Ne1/n) (-1+(E^(-(Ne1/n)) (-1+n))/n) n^2)/(b^2 (-1+n))-(2 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^3)/(b^2 (-1+n)^2)))^(-3+1/4 (3-n+Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)])) (-2+1/4 (3-n+Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)])) (-1+1/4 (3-n+Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]))))/(b^2 (-1+n)^2 (1/4 c1 (3-n-Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]) ((24 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^4)/(b^2 (-1+n)^2)+6 ((2 E^(Ne1/n) (-1+(E^(-(Ne1/n)) (-1+n))/n) n^2)/(b^2 (-1+n))-(2 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^3)/(b^2 (-1+n)^2)))^(-2+1/4 (3-n-Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)])) (-1+1/4 (3-n-Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]))+1/4 c2 (3-n+Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]) ((24 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^4)/(b^2 (-1+n)^2)+6 ((2 E^(Ne1/n) (-1+(E^(-(Ne1/n)) (-1+n))/n) n^2)/(b^2 (-1+n))-(2 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^3)/(b^2 (-1+n)^2)))^(-2+1/4 (3-n+Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)])) (-1+1/4 (3-n+Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]))))))^2 (1-(b^2 E^(-((2 Ne1)/n)) (-1+n)^2 ((2 E^(Ne1/n) (-1+(E^(-(Ne1/n)) (-1+n))/n) n^2)/(b^2 (-1+n))-(2 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^3)/(b^2 (-1+n)^2)))/(2 (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^4 (-1-(1/(2 (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^4))b^2 E^(-((2 Ne1)/n)) (-1+n)^2 ((2 E^(Ne1/n) (-1+(E^(-(Ne1/n)) (-1+n))/n) n^2)/(b^2 (-1+n))-(2 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^3)/(b^2 (-1+n)^2)) (-1-(24 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^4 (1/4 c1 (3-n-Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]) ((24 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^4)/(b^2 (-1+n)^2)+6 ((2 E^(Ne1/n) (-1+(E^(-(Ne1/n)) (-1+n))/n) n^2)/(b^2 (-1+n))-(2 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^3)/(b^2 (-1+n)^2)))^(-3+1/4 (3-n-Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)])) (-2+1/4 (3-n-Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)])) (-1+1/4 (3-n-Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]))+1/4 c2 (3-n+Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]) ((24 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^4)/(b^2 (-1+n)^2)+6 ((2 E^(Ne1/n) (-1+(E^(-(Ne1/n)) (-1+n))/n) n^2)/(b^2 (-1+n))-(2 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^3)/(b^2 (-1+n)^2)))^(-3+1/4 (3-n+Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)])) (-2+1/4 (3-n+Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)])) (-1+1/4 (3-n+Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]))))/(b^2 (-1+n)^2 (1/4 c1 (3-n-Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]) ((24 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^4)/(b^2 (-1+n)^2)+6 ((2 E^(Ne1/n) (-1+(E^(-(Ne1/n)) (-1+n))/n) n^2)/(b^2 (-1+n))-(2 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^3)/(b^2 (-1+n)^2)))^(-2+1/4 (3-n-Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)])) (-1+1/4 (3-n-Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]))+1/4 c2 (3-n+Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]) ((24 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^4)/(b^2 (-1+n)^2)+6 ((2 E^(Ne1/n) (-1+(E^(-(Ne1/n)) (-1+n))/n) n^2)/(b^2 (-1+n))-(2 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^3)/(b^2 (-1+n)^2)))^(-2+1/4 (3-n+Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)])) (-1+1/4 (3-n+Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]))))))))^2)"
# strns = "1-(-((2 b^2 E^(-((2 Ne1)/n)) (-1+n)^2 ((2 E^(Ne1/n) (-1+(E^(-(Ne1/n)) (-1+n))/n) n^2)/(b^2 (-1+n))-(2 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^3)/(b^2 (-1+n)^2)))/((-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^4))-(1/((-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^4))b^2 E^(-((2 Ne1)/n)) (-1+n)^2 ((2 E^(Ne1/n) (-1+(E^(-(Ne1/n)) (-1+n))/n) n^2)/(b^2 (-1+n))-(2 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^3)/(b^2 (-1+n)^2)) (-1-(24 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^4 (1/4 c1 (3-n-Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]) ((24 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^4)/(b^2 (-1+n)^2)+6 ((2 E^(Ne1/n) (-1+(E^(-(Ne1/n)) (-1+n))/n) n^2)/(b^2 (-1+n))-(2 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^3)/(b^2 (-1+n)^2)))^(-3+1/4 (3-n-Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)])) (-2+1/4 (3-n-Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)])) (-1+1/4 (3-n-Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]))+1/4 c2 (3-n+Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]) ((24 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^4)/(b^2 (-1+n)^2)+6 ((2 E^(Ne1/n) (-1+(E^(-(Ne1/n)) (-1+n))/n) n^2)/(b^2 (-1+n))-(2 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^3)/(b^2 (-1+n)^2)))^(-3+1/4 (3-n+Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)])) (-2+1/4 (3-n+Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)])) (-1+1/4 (3-n+Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]))))/(b^2 (-1+n)^2 (1/4 c1 (3-n-Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]) ((24 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^4)/(b^2 (-1+n)^2)+6 ((2 E^(Ne1/n) (-1+(E^(-(Ne1/n)) (-1+n))/n) n^2)/(b^2 (-1+n))-(2 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^3)/(b^2 (-1+n)^2)))^(-2+1/4 (3-n-Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)])) (-1+1/4 (3-n-Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]))+1/4 c2 (3-n+Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]) ((24 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^4)/(b^2 (-1+n)^2)+6 ((2 E^(Ne1/n) (-1+(E^(-(Ne1/n)) (-1+n))/n) n^2)/(b^2 (-1+n))-(2 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^3)/(b^2 (-1+n)^2)))^(-2+1/4 (3-n+Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)])) (-1+1/4 (3-n+Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)])))))+(b^2 E^(-((2 Ne1)/n)) (-1+n)^2 ((2 E^(Ne1/n) (-1+(E^(-(Ne1/n)) (-1+n))/n) n^2)/(b^2 (-1+n))-(2 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^3)/(b^2 (-1+n)^2)))/((-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^4 (-1-(1/(2 (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^4))b^2 E^(-((2 Ne1)/n)) (-1+n)^2 ((2 E^(Ne1/n) (-1+(E^(-(Ne1/n)) (-1+n))/n) n^2)/(b^2 (-1+n))-(2 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^3)/(b^2 (-1+n)^2)) (-1-(24 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^4 (1/4 c1 (3-n-Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]) ((24 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^4)/(b^2 (-1+n)^2)+6 ((2 E^(Ne1/n) (-1+(E^(-(Ne1/n)) (-1+n))/n) n^2)/(b^2 (-1+n))-(2 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^3)/(b^2 (-1+n)^2)))^(-3+1/4 (3-n-Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)])) (-2+1/4 (3-n-Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)])) (-1+1/4 (3-n-Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]))+1/4 c2 (3-n+Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]) ((24 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^4)/(b^2 (-1+n)^2)+6 ((2 E^(Ne1/n) (-1+(E^(-(Ne1/n)) (-1+n))/n) n^2)/(b^2 (-1+n))-(2 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^3)/(b^2 (-1+n)^2)))^(-3+1/4 (3-n+Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)])) (-2+1/4 (3-n+Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)])) (-1+1/4 (3-n+Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]))))/(b^2 (-1+n)^2 (1/4 c1 (3-n-Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]) ((24 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^4)/(b^2 (-1+n)^2)+6 ((2 E^(Ne1/n) (-1+(E^(-(Ne1/n)) (-1+n))/n) n^2)/(b^2 (-1+n))-(2 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^3)/(b^2 (-1+n)^2)))^(-2+1/4 (3-n-Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)])) (-1+1/4 (3-n-Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]))+1/4 c2 (3-n+Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]) ((24 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^4)/(b^2 (-1+n)^2)+6 ((2 E^(Ne1/n) (-1+(E^(-(Ne1/n)) (-1+n))/n) n^2)/(b^2 (-1+n))-(2 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^3)/(b^2 (-1+n)^2)))^(-2+1/4 (3-n+Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)])) (-1+1/4 (3-n+Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]))))))))/(1+(b^2 E^(-((2 Ne1)/n)) (-1+n)^2 ((2 E^(Ne1/n) (-1+(E^(-(Ne1/n)) (-1+n))/n) n^2)/(b^2 (-1+n))-(2 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^3)/(b^2 (-1+n)^2)))/(2 (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^4))"

# func_initialize(strr, strns)
# bounds = np.array([(50,60),(-100,-1),(1,10**7),(1,10),(1,10)])
# param_initialize(bounds,500,4,5)


if __name__ == '__main__':
    
    b=genetic_algo_start()[1]
    viable_arr = clear(b)
    #Save the arrays
    np.savetxt('test1.txt', viable_arr, fmt='%d')


# nSval_arr = np.apply_along_axis(nS, axis=1, arr=viable_arr)
# rval_arr = np.apply_along_axis(r, axis=1, arr=viable_arr)
strr = "(12 b^4 E^(-((4 Ne1)/n)) (-1+n)^4 ((2 E^(Ne1/n) (-1+(E^(-(Ne1/n)) (-1+n))/n) n^2)/(b^2 (-1+n))-(2 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^3)/(b^2 (-1+n)^2))^2)/((-1+(E^(-(Ne1/n)) (-1+n))/n)^4 n^8 (-1-(1/(2 (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^4))b^2 E^(-((2 Ne1)/n)) (-1+n)^2 ((2 E^(Ne1/n) (-1+(E^(-(Ne1/n)) (-1+n))/n) n^2)/(b^2 (-1+n))-(2 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^3)/(b^2 (-1+n)^2)) (-1-(24 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^4 (1/4 c1 (3-n-Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]) ((24 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^4)/(b^2 (-1+n)^2)+6 ((2 E^(Ne1/n) (-1+(E^(-(Ne1/n)) (-1+n))/n) n^2)/(b^2 (-1+n))-(2 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^3)/(b^2 (-1+n)^2)))^(-3+1/4 (3-n-Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)])) (-2+1/4 (3-n-Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)])) (-1+1/4 (3-n-Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]))+1/4 c2 (3-n+Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]) ((24 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^4)/(b^2 (-1+n)^2)+6 ((2 E^(Ne1/n) (-1+(E^(-(Ne1/n)) (-1+n))/n) n^2)/(b^2 (-1+n))-(2 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^3)/(b^2 (-1+n)^2)))^(-3+1/4 (3-n+Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)])) (-2+1/4 (3-n+Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)])) (-1+1/4 (3-n+Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]))))/(b^2 (-1+n)^2 (1/4 c1 (3-n-Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]) ((24 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^4)/(b^2 (-1+n)^2)+6 ((2 E^(Ne1/n) (-1+(E^(-(Ne1/n)) (-1+n))/n) n^2)/(b^2 (-1+n))-(2 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^3)/(b^2 (-1+n)^2)))^(-2+1/4 (3-n-Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)])) (-1+1/4 (3-n-Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]))+1/4 c2 (3-n+Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]) ((24 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^4)/(b^2 (-1+n)^2)+6 ((2 E^(Ne1/n) (-1+(E^(-(Ne1/n)) (-1+n))/n) n^2)/(b^2 (-1+n))-(2 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^3)/(b^2 (-1+n)^2)))^(-2+1/4 (3-n+Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)])) (-1+1/4 (3-n+Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]))))))^2 (1-(b^2 E^(-((2 Ne1)/n)) (-1+n)^2 ((2 E^(Ne1/n) (-1+(E^(-(Ne1/n)) (-1+n))/n) n^2)/(b^2 (-1+n))-(2 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^3)/(b^2 (-1+n)^2)))/(2 (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^4 (-1-(1/(2 (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^4))b^2 E^(-((2 Ne1)/n)) (-1+n)^2 ((2 E^(Ne1/n) (-1+(E^(-(Ne1/n)) (-1+n))/n) n^2)/(b^2 (-1+n))-(2 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^3)/(b^2 (-1+n)^2)) (-1-(24 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^4 (1/4 c1 (3-n-Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]) ((24 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^4)/(b^2 (-1+n)^2)+6 ((2 E^(Ne1/n) (-1+(E^(-(Ne1/n)) (-1+n))/n) n^2)/(b^2 (-1+n))-(2 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^3)/(b^2 (-1+n)^2)))^(-3+1/4 (3-n-Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)])) (-2+1/4 (3-n-Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)])) (-1+1/4 (3-n-Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]))+1/4 c2 (3-n+Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]) ((24 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^4)/(b^2 (-1+n)^2)+6 ((2 E^(Ne1/n) (-1+(E^(-(Ne1/n)) (-1+n))/n) n^2)/(b^2 (-1+n))-(2 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^3)/(b^2 (-1+n)^2)))^(-3+1/4 (3-n+Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)])) (-2+1/4 (3-n+Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)])) (-1+1/4 (3-n+Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]))))/(b^2 (-1+n)^2 (1/4 c1 (3-n-Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]) ((24 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^4)/(b^2 (-1+n)^2)+6 ((2 E^(Ne1/n) (-1+(E^(-(Ne1/n)) (-1+n))/n) n^2)/(b^2 (-1+n))-(2 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^3)/(b^2 (-1+n)^2)))^(-2+1/4 (3-n-Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)])) (-1+1/4 (3-n-Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]))+1/4 c2 (3-n+Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]) ((24 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^4)/(b^2 (-1+n)^2)+6 ((2 E^(Ne1/n) (-1+(E^(-(Ne1/n)) (-1+n))/n) n^2)/(b^2 (-1+n))-(2 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^3)/(b^2 (-1+n)^2)))^(-2+1/4 (3-n+Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)])) (-1+1/4 (3-n+Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]))))))))^2)"
strns = "1-(-((2 b^2 E^(-((2 Ne1)/n)) (-1+n)^2 ((2 E^(Ne1/n) (-1+(E^(-(Ne1/n)) (-1+n))/n) n^2)/(b^2 (-1+n))-(2 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^3)/(b^2 (-1+n)^2)))/((-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^4))-(1/((-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^4))b^2 E^(-((2 Ne1)/n)) (-1+n)^2 ((2 E^(Ne1/n) (-1+(E^(-(Ne1/n)) (-1+n))/n) n^2)/(b^2 (-1+n))-(2 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^3)/(b^2 (-1+n)^2)) (-1-(24 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^4 (1/4 c1 (3-n-Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]) ((24 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^4)/(b^2 (-1+n)^2)+6 ((2 E^(Ne1/n) (-1+(E^(-(Ne1/n)) (-1+n))/n) n^2)/(b^2 (-1+n))-(2 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^3)/(b^2 (-1+n)^2)))^(-3+1/4 (3-n-Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)])) (-2+1/4 (3-n-Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)])) (-1+1/4 (3-n-Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]))+1/4 c2 (3-n+Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]) ((24 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^4)/(b^2 (-1+n)^2)+6 ((2 E^(Ne1/n) (-1+(E^(-(Ne1/n)) (-1+n))/n) n^2)/(b^2 (-1+n))-(2 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^3)/(b^2 (-1+n)^2)))^(-3+1/4 (3-n+Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)])) (-2+1/4 (3-n+Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)])) (-1+1/4 (3-n+Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]))))/(b^2 (-1+n)^2 (1/4 c1 (3-n-Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]) ((24 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^4)/(b^2 (-1+n)^2)+6 ((2 E^(Ne1/n) (-1+(E^(-(Ne1/n)) (-1+n))/n) n^2)/(b^2 (-1+n))-(2 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^3)/(b^2 (-1+n)^2)))^(-2+1/4 (3-n-Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)])) (-1+1/4 (3-n-Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]))+1/4 c2 (3-n+Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]) ((24 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^4)/(b^2 (-1+n)^2)+6 ((2 E^(Ne1/n) (-1+(E^(-(Ne1/n)) (-1+n))/n) n^2)/(b^2 (-1+n))-(2 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^3)/(b^2 (-1+n)^2)))^(-2+1/4 (3-n+Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)])) (-1+1/4 (3-n+Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)])))))+(b^2 E^(-((2 Ne1)/n)) (-1+n)^2 ((2 E^(Ne1/n) (-1+(E^(-(Ne1/n)) (-1+n))/n) n^2)/(b^2 (-1+n))-(2 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^3)/(b^2 (-1+n)^2)))/((-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^4 (-1-(1/(2 (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^4))b^2 E^(-((2 Ne1)/n)) (-1+n)^2 ((2 E^(Ne1/n) (-1+(E^(-(Ne1/n)) (-1+n))/n) n^2)/(b^2 (-1+n))-(2 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^3)/(b^2 (-1+n)^2)) (-1-(24 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^4 (1/4 c1 (3-n-Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]) ((24 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^4)/(b^2 (-1+n)^2)+6 ((2 E^(Ne1/n) (-1+(E^(-(Ne1/n)) (-1+n))/n) n^2)/(b^2 (-1+n))-(2 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^3)/(b^2 (-1+n)^2)))^(-3+1/4 (3-n-Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)])) (-2+1/4 (3-n-Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)])) (-1+1/4 (3-n-Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]))+1/4 c2 (3-n+Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]) ((24 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^4)/(b^2 (-1+n)^2)+6 ((2 E^(Ne1/n) (-1+(E^(-(Ne1/n)) (-1+n))/n) n^2)/(b^2 (-1+n))-(2 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^3)/(b^2 (-1+n)^2)))^(-3+1/4 (3-n+Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)])) (-2+1/4 (3-n+Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)])) (-1+1/4 (3-n+Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]))))/(b^2 (-1+n)^2 (1/4 c1 (3-n-Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]) ((24 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^4)/(b^2 (-1+n)^2)+6 ((2 E^(Ne1/n) (-1+(E^(-(Ne1/n)) (-1+n))/n) n^2)/(b^2 (-1+n))-(2 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^3)/(b^2 (-1+n)^2)))^(-2+1/4 (3-n-Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)])) (-1+1/4 (3-n-Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]))+1/4 c2 (3-n+Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]) ((24 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^4)/(b^2 (-1+n)^2)+6 ((2 E^(Ne1/n) (-1+(E^(-(Ne1/n)) (-1+n))/n) n^2)/(b^2 (-1+n))-(2 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^3)/(b^2 (-1+n)^2)))^(-2+1/4 (3-n+Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)])) (-1+1/4 (3-n+Sqrt[1-2 n] Sqrt[(1+10 n+n^2)/(1-2 n)]))))))))/(1+(b^2 E^(-((2 Ne1)/n)) (-1+n)^2 ((2 E^(Ne1/n) (-1+(E^(-(Ne1/n)) (-1+n))/n) n^2)/(b^2 (-1+n))-(2 E^((2 Ne1)/n) (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^3)/(b^2 (-1+n)^2)))/(2 (-1+(E^(-(Ne1/n)) (-1+n))/n)^2 n^4))"
bounds = np.array([(50,60),(-10**6,-1),(1,10**7),(1,10),(1,10)])
param_initialize(bounds,1500,10,7)
func_initialize(strr, strns)
# fig, ax = plt.subplots()
# ax.scatter(nSval_arr, rval_arr)

def get_values(arr):
    nS_arr = np.zeros(len(arr))
    r_arr = np.zeros(len(arr))
    for index, obj in enumerate(arr):
        nS_arr[index] = nS(*obj)
        r_arr[index] = r(*obj)
    return nS_arr, r_arr




# ax.set_title('nSvs r')
# ax.set_xlabel('nS')
# ax.set_ylabel('r')
# plt.show()

# np.savetxt('viable.txt', viable_arr, fmt='%d')
# np.savetxt('nS.txt', nSval_arr, fmt='%d')
# np.savetxt('r.txt', rval_arr, fmt='%d')