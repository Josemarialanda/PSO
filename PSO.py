import numpy as np
from inspect import signature

from numpy.core.fromnumeric import size

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
    def __init__(self, dim, pop_size, soln_range) -> None:
        self.dim = dim
        self.pop_size = pop_size
        self.particles     = [0]*pop_size
        self.best_previous = [0]*pop_size
        for i in range(pop_size):
            position = [np.random.uniform(-soln_range, soln_range) for _ in range(dim)]
            velocity = [np.random.uniform(-100, 100) for _ in range(dim)]
            self.particles[i] = Particle(position, velocity)
        # initially the best previous are simply the population particles
        self.best_previous = self.particles
        self.best_particle = self.particles[0]
    
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

class PSO:
    
    # initialize population and find best element
    def __init__(self, fitness_function, obj = "min",
                       cf = 1,    # velocity constriction factor
                       w = 0.75,  # intertia weight
                       c1 = 1.75, # cognitive parameter
                       c2 = 1.75, # social parameter
                       pop_size = 100, soln_range = 1000, 
                       max_iter = 100) -> None:
        dim = len(signature(fitness_function).parameters)
        self.max_iter = max_iter
        self.iter = 0
        self.fitness_function = fitness_function
        self.obj = obj
        self.cf = cf
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.pop_size = pop_size
        self.population = Population(dim, pop_size, soln_range)
        # compute fitness for each particle in the population
        fitness = [self.fitness_function(*particle.position) for particle in self.population.particles]
        for i in range(len(fitness)):
            self.population.particles[i].fitness = fitness[i]
        # find the best particles in population
        self.population.best_particle = self.sort_by_fitness()[0]
        
    def print_population(self):
        print(self.population)
        
    def sort_by_fitness(self):
        reverse = False
        if self.obj == "max" : reverse = True
        key = lambda particle: self.fitness_function(*particle.position)
        return sorted(self.population.particles, reverse=reverse, key = key)
            
    def next(self):
        best_pop = self.population.best_particle                # best particle position in population
        for i in range(self.pop_size):
            xk      = self.population.particles[i].position     # current particle position
            xk_best = self.population.best_previous[i].position # previous best particle position
            vk      = self.population.particles[i].velocity     # current particle velocity
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
            if self.obj == "max":
                # update previous
                if f1 > f2: self.population.best_previous[i] = self.population.particles[i]
                # update population best
                if f1 > f3: self.population.best_particle    = self.population.particles[i]
            if self.obj == "min":
                # update previous
                if f1 < f2: self.population.best_previous[i] = self.population.particles[i]
                # update population best
                if f1 < f3: self.population.best_particle    = self.population.particles[i]
            
    def run(self):
        print(f"max iterations = {self.max_iter}")
        while self.iter < self.max_iter:
            self.next()
            print(self.population.best_particle)
            self.iter = self.iter + 1
         
  
g5 = lambda x1, x2 : 3*x1+0.000001*x1**3+2*x2+0.000002/(3*x2**3)
g7 = lambda x1, x2, x3, x4, x5, x6, x7, x8, x9, x10: x1**2+x2**2+x1*x2-14*x1-16*x2+(x3-10)**2+4*(x4-5)**2+(x5-3)**2+2*(x6-1)**2+5*x7**2+7*(x8-11)**2+2*(x9-10)**2+(x10-7)**2+45

pso = PSO(g7, pop_size=100, max_iter=100)
pso.run()