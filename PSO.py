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
    def __init__(self, f, restrictions = [],
                       verbose = False,
                       a  = 2,    # CVD parameter
                       s  = 1500, # penalty parameter
                       cf = 1,    # velocity constriction factor
                       w  = 0.3,  # intertia weight
                       c1 = 1.75, # cognitive parameter
                       c2 = 1.75, # social parameter
                       pop_size = 100,
                       bounds = [], 
                       max_iter = 100) -> None:
        
        # objective function
        self.obj = deepcopy(f)
        # store best solutions
        self.X   = []
        # store solution CVD
        self.cvd = []
        
        self.verbose = verbose
        self.a = a
        self.s = s
        self.f = f
        self.dim = len(signature(f).parameters)
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
            self.f = deepcopy(self.penalty(deepcopy(f)))
        
        self.max_iter = max_iter; self.iter = 0              # iteration count
        self.cf = cf; self.w = w; self.c1 = c1; self.c2 = c2 # hyperparameters
        self.pop_size = pop_size
        self.population = Population(self.dim, pop_size, bounds)
        # compute fitness for each particle in the population
        fitness = [self.f(*particle.position) for particle in self.population.particles]
        for i in range(len(fitness)):
            self.population.particles[i].fitness = fitness[i]
        # find the best particles in population
        self.population.best_particle = deepcopy(self.sort_by_fitness()[0])
        
    def CVD(self, *x):
        V = 0
        # restrictions
        for g in self.restrictions:
            V += np.max([0, g(*x)])
        # bound restrictions
        for i in range(len(self.bound_restrictions)):
           g = self.bound_restrictions[i]
           V += np.max([0, g(*x)])
        return V**self.a
    
    def penalty(self, f):
        return lambda *x : f(*x) + self.s*self.CVD(*x)
        
    def print_population(self):
        print(self.population)
        
    def sort_by_fitness(self):
        key = lambda particle: self.f(*particle.position)
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
            self.population.particles[i].fitness  = self.f(*self.population.particles[i].position)

            # update previous best and population best
            f1 = self.population.particles[i].fitness
            f2 = self.population.best_previous[i].fitness
            f3 = best_pop.fitness
            # update previous
            if f1 < f2: self.population.best_previous[i] = deepcopy(self.population.particles[i])
            # update population best
            if f1 < f3: self.population.best_particle    = deepcopy(self.population.particles[i])
            
    def run(self):
        if self.verbose is True: print(f"max iterations = {self.max_iter}")
        while self.iter < self.max_iter:
            self.X.append(self.obj(*self.population.best_particle.position))
            self.cvd.append(self.CVD(*self.population.best_particle.position))
            fitness = self.population.best_particle.fitness
            cvd     = self.CVD(*self.population.best_particle.position)
            x       = self.population.best_particle.position
            if self.verbose is True: print(f'''Ciclo {self.iter+1}. f: {fitness} CVD: {cvd} x: {x}''')
            self.next()
            self.iter = self.iter + 1