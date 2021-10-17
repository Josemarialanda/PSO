import numpy as np
from PSO import PSO

# intertia weight (w)
w = np.random.uniform(low=0.1,high=0.5)
# cognitive parameter (c1)
c1 = np.random.uniform(low=1.2,high=2.0)
# social parameter    (c2)
c2 = np.random.uniform(low=1.2,high=2.0)

# test problem 1: the best known solution is f* = 1.3934651
test1    = lambda x1, x2 : (x1-2)**2+(x2-1)**2
test1_r1 = lambda x1, x2 : x1-2*x2+1
test1_r2 = lambda x1, x2 : (((x1**2)/4)+x2**2-1)
test1_r  = [test1_r1, test1_r2]
test1_bounds = [(-5,5),(-5,5)]

# pso = PSO(test1, restrictions = test1_r, pop_size=100, max_iter=200, bounds=test1_bounds, w=w, c1=c1, c2=c2, verbose=True)
# pso.run()
# print("\nSolution: ", pso.population.best_particle)

# test problem 2: the best known solution is f* = -6961.81381
test2    = lambda x1, x2 : (x1-10)**3+(x2-20)**3
test2_r1 = lambda x1, x2 : 100-(x1-5)**2-(x2-5)**2
test2_r2 = lambda x1, x2 : (x1-6)**2+(x2-5)**2-82.81
test2_r  = [test2_r1, test2_r2]
test2_bounds = [(13,100),(0,100)]

# pso = PSO(test2, restrictions = test2_r, pop_size=500, max_iter=200, bounds=test2_bounds, a=1, w=w, c1=c1, c2=c2, verbose=True)
# pso.run()
# print("\nSolution: ", pso.population.best_particle)

# test problem 3: the best known solution is f* = 680.630057
test3    = lambda x1, x2, x3, x4, x5, x6, x7 : (x1-10)**2+5*(x2-12)**2+x3**4+3*(x4-11)**2+10*x5**6+7*x6**2+x7**4-4*x6*x7-10*x6-8*x7
test3_r1 = lambda x1, x2, x3, x4, x5, x6, x7 : -127+2*x1**2+3*x2**4+x3+4*x4**2+5*x5
test3_r2 = lambda x1, x2, x3, x4, x5, x6, x7 : -282+7*x1+3*x2+10*x3**2+x4-x5
test3_r3 = lambda x1, x2, x3, x4, x5, x6, x7 : -196+23*x1+x2**2+6*x6**2-8*x7
test3_r4 = lambda x1, x2, x3, x4, x5, x6, x7 : 4*x1**2+x2**2-3*x1*x2+2*x3**2-5*x6-11*x7
test3_r  = [test3_r1, test3_r2, test3_r3, test3_r4]
test3_bounds = [(-10,10),(-10,10),(-10,10),(-10,10),(-10,10),(-10,10),(-10,10)]

# pso = PSO(test3, restrictions = test3_r, pop_size=100, max_iter=200, bounds=test3_bounds, w=w, c1=c1, c2=c2, verbose=True)
# pso.run()
# print("\nSolution: ", pso.population.best_particle)

# De Jong's function
# De Jong's function is also known as sphere model. It is continuos, convex and unimodal.
# the best known solution is f* = 0
deJong_5 = lambda x1, x2, x3, x4, x5: sum([xi**2 for xi in [x1, x2, x3, x4, x5]])
deJong_5_bounds  = [(-5.12,5.12),(-5.12,5.12),(-5.12,5.12),(-5.12,5.12),(-5.12,5.12)]

# pso = PSO(deJong_5, restrictions = [], pop_size=100, max_iter=200, bounds=deJong_5_bounds, s=0.1, w=w, c1=c1, c2=c2, verbose=True)
# pso.run()
# print("\nSolution: ", pso.population.best_particle)

# Axis parallel hyper-ellipsoid function
'''
The axis parallel hyper-ellipsoid is similar to De Jong's function 1. It is also known as the weighted sphere
model. Again, it is continuos, convex and unimodal.
'''
# the best known solution is f* = 0
def aphe_5(x1,x2,x3,x4,x5):
    x = [x1,x2,x3,x4,x5]
    i = [1,2,3,4,5]
    res = 0
    for j in range(5):
        xi = x[j]
        ii = i[j] 
        res = res + ii*xi**2
    return res
aphe_5_bounds  = [(-5.12,5.12),(-5.12,5.12),(-5.12,5.12),(-5.12,5.12),(-5.12,5.12)]

# pso = PSO(aphe_5, restrictions = [], pop_size=100, max_iter=200, bounds=aphe_5_bounds, s=0.1, w=w, c1=c1, c2=c2, verbose=True)
# pso.run()
# print("\nSolution: ", pso.population.best_particle)

# Rastrigin's function 6
'''
Rastrigin's function is based on De Jong's function with the addition of cosine
modulation to produce many local minima. Thus, the test function is highly multimodal.
However, the location of the minima are regularly distributed.
'''
# the best known solution is f* = 0
rastrigin_5 = lambda x1, x2, x3, x4, x5: 10*5+sum([xi**2-10*np.cos(2*np.pi*xi) for xi in [x1, x2, x3, x4, x5]])
rastrigin_5_bounds  = [(-5.12,5.12),(-5.12,5.12),(-5.12,5.12),(-5.12,5.12),(-5.12,5.12)]

# pso = PSO(rastrigin_5, restrictions = [], pop_size=500, max_iter=200, bounds=rastrigin_5_bounds, w=w, c1=c1, c2=c2, s=0.1, verbose=True)
# pso.run()
# print("\nSolution: ", pso.population.best_particle)

# Schwefel's function
'''
Schwefel's function [Sch81] is deceptive in that the global minimum is geometrically distant, over the parameter
space, from the next best local minima. Therefore, the search algorithms are potentially prone to convergence in
the wrong direction.
'''
# the best known solution is f* = -n*418.9829 (in this case n=5, so  f* = −2094.9145)
schwefel_5 = lambda x1, x2, x3, x4, x5: sum([-xi*np.sin(np.sqrt(np.abs(xi))) for xi in [x1, x2, x3, x4, x5]])
schwefel_5_bounds  = [(-500,500),(-500,500),(-500,500),(-500,500),(-500,500)]

# pso = PSO(schwefel_5, restrictions = [], pop_size=100, max_iter=200, bounds=schwefel_5_bounds, w=w, c1=c1, c2=c2, s=0.1, verbose=True)
# pso.run()
# print("\nSolution: ", schwefel_5(*pso.population.best_particle.position))

# Ackley's Path function
# the best known solution is f* = 0
def ackley_5(x1,x2,x3,x4,x5):
    a = 20
    b = 0.2
    c = 2*np.pi
    x = [x1,x2,x3,x4,x5]
    xi_2_sum   = sum([xi**2 for xi in x])
    xi_cos_sum = sum([np.cos(c*xi) for xi in x])
    n = len(x)
    return -a*np.exp(-b*np.sqrt(1/n*xi_2_sum))-np.exp(1/n*xi_cos_sum)+a+np.exp(1)
ackley_5_bounds  = [(-32.768,32.768),(-32.768,32.768),(-32.768,32.768),(-32.768,32.768),(-32.768,32.768)]

# pso = PSO(ackley_5, restrictions = [], pop_size=100, max_iter=200, bounds=ackley_5_bounds, w=w, c1=c1, c2=c2, s=0.1, verbose=True)
# pso.run()
# print("\nSolution: ", schwefel_5(*pso.population.best_particle.position))

# more examples at: http://www.geatbx.com/download/GEATbx_ObjFunExpl_v38.pdf


# g5 defined in [Michalewicz1996]: the best known solution is f* = 5126.4981
g5        = lambda x1, x2, x3, x4 : 3*x1+0.000001*x1**3+2*x2+0.000002/(3*x2**3)
g5_r1     = lambda x1, x2, x3, x4 : (-1)*(x4-x3+0.55)
g5_r2     = lambda x1, x2, x3, x4 :  (-1)*(x3-x4+0.55)

g5_r3     = lambda x1, x2, x3, x4 : np.abs(1000*np.sin(-x3-0.25)+1000*np.sin(-x4-0.25)+894.8-x1) - 0.005
g5_r4     = lambda x1, x2, x3, x4 : np.abs(1000*np.sin(x3-0.25)+1000*np.sin(x3-x4-0.25)+894.8-x2) - 0.005
g5_r5     = lambda x1, x2, x3, x4 : np.abs(1000*np.sin(x4-0.25)+1000*np.sin(x4-x3-0.25)+1294.8) - 0.005

g5_r      = [g5_r1, g5_r2, g5_r3, g5_r4, g5_r5]
g5_bounds = [(0,1200),(0,1200),(-0.55,0.55),(-0.55,0.55)]

# we can get a lower cvd if we use slightly larger values of s
# pso = PSO(g5, restrictions = g5_r, pop_size=100, max_iter=200, bounds = g5_bounds, s=0.1, w=w, c1=c1, c2=c2, verbose=True)
# pso.run()
# print(g5(*pso.population.best_particle.position))

# experiments (run with s=0.1, s=10 and try pop_size = 100, 500, 1000 and finally try num_experiments = 100 and 1000 with pop_size=100 and s=0.1)
# i.e try:
#  1) s=0.1, pop_size=100
#  2) s=0.1, pop_size=500
#  3) s=0.1, pop_size=1000
#  4) s=10,  pop_size=100
#  5) s=10,  pop_size=500
#  6) s=10,  pop_size=1000

def run_g5_experiments(num_experiments = 10, s=0.1, pop_size=100):
    X   = [0]*num_experiments
    CVD = [0]*num_experiments
    for i in range(num_experiments):
        w = np.random.uniform(low=0.1,high=0.5)
        c1 = np.random.uniform(low=1.2,high=2.0)
        c2 = np.random.uniform(low=1.2,high=2.0)
        pso = PSO(g5, restrictions = g5_r, pop_size=pop_size, max_iter=200, bounds = g5_bounds, s=s, w=w, c1=c1, c2=c2, verbose=True)
        pso.run()
        X[i]   = pso.X
        CVD[i] = pso.cvd
    return X, CVD

# experiment_1 = run_g5_experiments(s=0.1, pop_size=100)
# experiment_2 = run_g5_experiments(s=0.1, pop_size=500)
# experiment_3 = run_g5_experiments(s=0.1, pop_size=1000)
# experiment_4 = run_g5_experiments(s=10,  pop_size=100)
# experiment_5 = run_g5_experiments(s=10,  pop_size=500)
# experiment_6 = run_g5_experiments(s=10,  pop_size=1000)
# experiment_7 = run_g5_experiments(num_experiments=100, s=0.1,  pop_size=100)
# experiment_8 = run_g5_experiments(num_experiments=1000, s=0.1, pop_size=100)

# g7 defined in [Michalewicz1996]: the best known solution is f* = 24.3062091
g7 = lambda x1, x2, x3, x4, x5, x6, x7, x8, x9, x10: x1**2+x2**2+x1*x2-14*x1-16*x2+(x3-10)**2+4*(x4-5)**2+(x5-3)**2+2*(x6-1)**2+5*x7**2+7*(x8-11)**2+2*(x9-10)**2+(x10-7)**2+45
g7_r1     = lambda x1, x2, x3, x4, x5, x6, x7, x8, x9, x10: (-1)*(105-4*x1-5*x2+3*x1-9*x8)
g7_r2     = lambda x1, x2, x3, x4, x5, x6, x7, x8, x9, x10: (-1)*(-3*(x1-2)**2-4*(x2-3)**2-2*x3**2+7*x4+130)
g7_r3     = lambda x1, x2, x3, x4, x5, x6, x7, x8, x9, x10: (-1)*(-10*x1+8*x2+17*x7-2*x8)
g7_r4     = lambda x1, x2, x3, x4, x5, x6, x7, x8, x9, x10: (-1)*(-x1**2-2*(x2-2)**2+2*x1*x2-14*x5+6*x6)
g7_r5     = lambda x1, x2, x3, x4, x5, x6, x7, x8, x9, x10: (-1)*(8*x1-2*x2-5*x9+2*x10+12)
g7_r6     = lambda x1, x2, x3, x4, x5, x6, x7, x8, x9, x10: (-1)*(-5*x1**2-8*x2-(x3-6)**2+2*x4+40)
g7_r7     = lambda x1, x2, x3, x4, x5, x6, x7, x8, x9, x10: (-1)*(3*x1-6*x2-12*(x9-8)**2+7*x10)
g7_r8     = lambda x1, x2, x3, x4, x5, x6, x7, x8, x9, x10: (-1)*(-0.5*(x1-8)**2-2*(x2-4)**2-3*x5**2+x6+30)

g7_r      = [g7_r1, g7_r2, g7_r3, g7_r4, g7_r5, g7_r6, g7_r7, g7_r8]
g7_bounds = [(-10,10),(-10,10),(-10,10),(-10,10),(-10,10),(-10,10),(-10,10),(-10,10),(-10,10),(-10,10)]

# with a higher value of s, say s=10, we can better explore the solution space and obtain a solution with a lower CVD
# pso = PSO(g7, restrictions = g7_r, pop_size=500, max_iter=200, bounds = g7_bounds, s=0.1, w=w, c1=c1, c2=c2)
# pso.run()
# print("\nSolution: ", pso.population.best_particle)
# print(g7(*pso.population.best_particle.position))

# experiments (run with s=0.1, s=10 and try pop_size = 100, 500, 1000 and finally try num_experiments = 100 and 1000 with pop_size=100 and s=0.1)
# i.e try:
#  1) s=0.1, pop_size=100
#  2) s=0.1, pop_size=500
#  3) s=0.1, pop_size=1000
#  4) s=10,  pop_size=100
#  5) s=10,  pop_size=500
#  6) s=10,  pop_size=1000

def run_g7_experiments(num_experiments = 10, s=0.1, pop_size=100):
    X   = [0]*num_experiments
    CVD = [0]*num_experiments
    for i in range(num_experiments):
        w = np.random.uniform(low=0.1,high=0.5)
        c1 = np.random.uniform(low=1.2,high=2.0)
        c2 = np.random.uniform(low=1.2,high=2.0)
        pso = PSO(g7, restrictions = g7_r, pop_size=pop_size, max_iter=200, bounds = g7_bounds, s=s, w=w, c1=c1, c2=c2, verbose=True)
        pso.run()
        X[i]   = pso.X
        CVD[i] = pso.cvd
    return X, CVD

# experiment_1 = run_g7_experiments(s=0.1, pop_size=100)
# experiment_2 = run_g7_experiments(s=0.1, pop_size=500)
# experiment_3 = run_g7_experiments(s=0.1, pop_size=1000)
# experiment_4 = run_g7_experiments(s=10,  pop_size=100)
# experiment_5 = run_g7_experiments(s=10,  pop_size=500)
# experiment_6 = run_g7_experiments(s=10,  pop_size=1000)
# experiment_7 = run_g7_experiments(num_experiments=100, s=0.1, pop_size=100)
# experiment_8 = run_g7_experiments(num_experiments=1000, s=0.1, pop_size=100)

