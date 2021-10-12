import numpy as np
from inspect import signature
from copy import deepcopy

class Particle:
    
    # initialize nth dimensional particle
    def __init__(self, position, velocity) -> None:
        self.position  = np.array(position)
        self.velocity  = np.array(velocity)
        self.fitness   = 0
        
    def __repr__(self) -> str:
        msg = f'''
                POSITION : {self.position}
                VELOCITY : {self.velocity}
                FITNESS  : {self.fitness}
                '''
        return msg

class Population:
    
    # Initialize population
    def __init__(self, dim, pop_size, bounds) -> None:
        self.dim = dim
        self.pop_size = pop_size
        self.particles     = [0]*pop_size
        self.best_previous = [0]*pop_size
        for i in range(pop_size):
            if bounds == []: position = [np.random.uniform(-10000, 10000) for _ in range(dim)]
            else:
                position = [0]*dim
                for j in range(dim):
                    position[j] = np.random.uniform(bounds[j][0],bounds[j][1])
            velocity = [np.random.uniform(0, 100) for _ in range(dim)]
            self.particles[i] = Particle(position, velocity)
        # initially the best previous are simply the population particles
        self.best_previous = deepcopy(self.particles)
        self.best_particle = deepcopy(self.particles[0])
    
    def __repr__(self) -> str:
        msg = f" best particle: {self.best_particle}\n population:"
        for particle in self.particles:
            msg += repr(particle) + "\n"
        return msg
    
    def compare_generation(self):
        for i in range(self.pop_size):
            print(f'''
                    CURRENT POSITION       : {self.particles[i].position}
                    PREVIOUS BEST POSITION : {self.best_previous[i].position}
    particle [{i}]            
                    CURRENT FITNESS        : {self.particles[i].fitness}
                    PREVIOUS BEST FITNESS  : {self.best_previous[i].fitness}
                    
    ----------------------------------------------------------------------------------------
                    ''')

# solves single objective optimization problems
# with multiple restrictions of the form g(x) <= 0
class PSO:
    
    # initialize population and find best element
    def __init__(self, fitness_function, restrictions = [],
                       cf = 1,    # velocity constriction factor
                       w = 0.75,  # intertia weight
                       c1 = 1.75, # cognitive parameter
                       c2 = 1.75, # social parameter
                       pop_size = 100,
                       bounds = [], 
                       max_iter = 100) -> None:
        
        self.fitness_function = fitness_function
        self.dim = len(signature(fitness_function).parameters)
        self.restrictions = restrictions
        self.bound_restrictions = []*len(bounds)
        
        if bounds != []:
            for i in range(len(bounds)):
                lower = bounds[i][0]
                upper = bounds[i][1]
                g_lower = lambda *x : x[0]-upper
                g_upper = lambda *x : lower-x[1]
                self.bound_restrictions.append(g_lower)
                self.bound_restrictions.append(g_upper)
 
        if self.restrictions != [] or self.bound_restrictions != []:
            f = deepcopy(fitness_function)
            self.fitness_function = deepcopy(self.penalty(f))
        
        self.max_iter = max_iter; self.iter = 0              # iteration count
        self.cf = cf; self.w = w; self.c1 = c1; self.c2 = c2 # hyperparameters
        self.pop_size = pop_size
        self.population = Population(self.dim, pop_size, bounds)
        # compute fitness for each particle in the population
        fitness = [self.fitness_function(*particle.position) for particle in self.population.particles]
        for i in range(len(fitness)):
            self.population.particles[i].fitness = fitness[i]
        # find the best particles in population
        self.population.best_particle = deepcopy(self.sort_by_fitness()[0])
        
    def CVD(self, *x):
        a = 1 # a is a positive real number
        V = 0
        # restrictions
        for g in self.restrictions:
            V += np.max([0, g(*x)])
        # bound restrictions
        for i in range(len(self.bound_restrictions)):
           g = self.bound_restrictions[i]
           V += np.max([0, g(*x)])
        return V**a
    
    # this is the culprit
    def penalty(self, f):
        def gte(*x):
            z = f(*self.population.best_particle.position)
            return np.abs(f(*x)-z) #TODO <- something's wrong here =/ (I think the problem is here)
        
        # =(
        # return lambda *x : gte(*x) + 0.1*self.CVD(*x)**2
        # return lambda *x : f(*x) + 10000*self.CVD(*x)
        return lambda *x : f(*x) + 10**20*np.sum([np.abs(np.max([0,g(*x)])) for g in self.restrictions])
        # return f
        
    def print_population(self):
        print(self.population)
        
    def sort_by_fitness(self):
        key = lambda particle: self.fitness_function(*particle.position)
        return sorted(self.population.particles, key = key)
            
    def next(self):
        best_pop = self.population.best_particle                # best particle position in population
        for i in range(self.pop_size):
            xk      = self.population.particles[i].position     # current particle position
            vk      = self.population.particles[i].velocity     # current particle velocity
            xk_best = self.population.best_previous[i].position # previous best particle position
            r1, r2 = np.random.uniform(size = 2)
            # update velocity
            self.population.particles[i].velocity = self.cf*(self.w*vk+self.c1*r1*(xk_best-xk)+self.c2*r2*(best_pop.position-xk))
            # update position
            self.population.particles[i].position = xk + self.population.particles[i].velocity
            # update fitness
            self.population.particles[i].fitness  = self.fitness_function(*self.population.particles[i].position)

            # update previous best and population best
            f1 = self.population.particles[i].fitness
            f2 = self.population.best_previous[i].fitness
            f3 = best_pop.fitness
            # update previous
            if f1 < f2: self.population.best_previous[i] = deepcopy(self.population.particles[i])
            # update population best
            if f1 < f3: self.population.best_particle    = deepcopy(self.population.particles[i])

            
    def run(self):
        print(f"max iterations = {self.max_iter}")
        while self.iter < self.max_iter:
            self.next()
            fitness = self.population.best_particle.fitness
            cvd     = self.CVD(*self.population.best_particle.position)
            x       = self.population.best_particle.position
            print(f'''Ciclo {self.iter+1}. f: {fitness} CVD: {cvd} x: {x}''')
            self.iter = self.iter + 1
         


# test problem 1
test1    = lambda x1, x2 : (x1-2)**2+(x2-1)**2
test1_r1 = lambda x1, x2 : x1-2*x2+1
test1_r2 = lambda x1, x2 : (((x1**2)/4)+x2**2-1)
test1_r  = [test1_r1, test1_r2]
test1_bounds = [(-5,5),(-5,5)]

# pso = PSO(test1, restrictions = test1_r, pop_size=50, max_iter=10000, bounds = test1_bounds)
# pso.run()

# test problem 2
test2    = lambda x1, x2 : (x1-10)**3+(x2-20)**3
test2_r1 = lambda x1, x2 : 100-(x1-5)**2-(x2-5)**2
test2_r2 = lambda x1, x2 : (x1-6)**2+(x2-5)**2-82.81
test2_r  = [test2_r1, test2_r2]
test2_bounds = [(13,100),(0,100)]

# pso = PSO(test2, restrictions = test2_r, pop_size=50, max_iter=1000, bounds = test2_bounds)
# pso.run()

# g5 defined in [Michalewicz1996]
g5        = lambda x1, x2, x3, x4 : 3*x1+0.000001*x1**3+2*x2+0.000002/(3*x2**3)
g5_r1     = lambda x1, x2, x3, x4 : (-1)*(x4-x3+0.55)
g5_r2     = lambda x1, x2, x3, x4 :  (-1)*(x3-x4+0.55)
g5_r3     = lambda x1, x2, x3, x4 : 1000*np.sin(-x3-0.25)+1000*np.sin(-x4-0.25)+894.8-x1
g5_r4     = lambda x1, x2, x3, x4 : 1000*np.sin(x3-0.25)+1000*np.sin(x3-x4-0.25)+894.8-x2
g5_r5     = lambda x1, x2, x3, x4 : 1000*np.sin(x4-0.25)+1000*np.sin(x4-x3-0.25)+1294.8
g5_r      = [g5_r1, g5_r2, g5_r3, g5_r4, g5_r5]
g5_bounds = [(0,1200),(0,1200),(-0.55,0.55),(-0.55,0.55)]

# pso = PSO(g5, restrictions = g5_r, pop_size=50, max_iter=5000, bounds = g5_bounds)
# pso.run()

# g7 defined in [Michalewicz1996]
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

# pso = PSO(g7, restrictions = g7_r, pop_size=50, max_iter=20000, bounds = g7_bounds)
# pso.run()