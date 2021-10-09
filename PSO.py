import numpy as np
from inspect import signature

class Particle:
    
    position : 'list[float]'
    velocity : 'list[float]'
    fitness  : float
    
    # initialize nth dimensional particle
    def __init__(self, position, velocity) -> None:
        self.position  = position
        self.velocity  = velocity
        self.fitness   = 0
        
    def __repr__(self) -> str:
        msg = f'''
                position = {self.position}
                velocity = {self.velocity}
                fitness  = {self.fitness}
                '''
        return msg

class Population:
    
    particles     : 'list[Particle]'
    best_particle : int # index of best particle
    
    # Initialize population
    def __init__(self, dim, pop_size, soln_range) -> None:
        self.particles = [0]*pop_size
        for i in range(pop_size):
            position = [np.random.uniform(-soln_range, soln_range) for _ in range(dim)]
            velocity = [np.random.uniform(-100, 100) for _ in range(dim)]
            self.particles[i] = Particle(position, velocity)
        self.best_particle = 0
    
    def __repr__(self) -> str:
        msg = f" best particle = {self.best_particle}\n"
        for particle in self.particles:
            msg += repr(particle) + "\n"
        return msg

class PSO:
    
    population : Population
    max_iter   : int
    # fitness_function : float, float, ... -> float
    
    # initialize population and compute fitness
    def __init__(self, fitness_function, pop_size = 100, soln_range = 1000, max_iter = 100) -> None:
        dim = len(signature(fitness_function).parameters)
        self.max_iter = max_iter
        self.iter = 0
        self.fitness_function = fitness_function
        self.population = Population(dim, pop_size, soln_range)
        # compute fitness for each particle in the population
        fitness = [self.fitness_function(*particle.position) for particle in self.population.particles]
        for i in range(len(fitness)):
            self.population.particles[i].fitness = fitness[i]
        # find the best particle in population
        # self.population.best_particle = self.sort_by_fitness()[0]
        
    def print_population(self):
        print(self.population)
        
    def sort_by_fitness(self):
        pass
            
    def update_particles(self):
        # update positions
        # update velocities
        pass        
    
    def update_bests(self):
        # update best particle
        pass
            
    def next(self):
        self.update_particles()
        self.update_bests()
            
    def run(self):
        if self.iter < self.max_iter:
            pass
         
  
g5 = lambda x1, x2 : 3*x1+0.000001*x1**3+2*x2+0.000002/(3*x2**3)
g7 = lambda x1, x2, x3, x4, x5, x6, x7, x8, x9, x10: x1**2+x2**2+x1*x2-14*x1-16*x2+(x3-10)**2+4*(x4-5)**2+(x5-3)**2+2*(x6-1)**2+5*x7**2+7*(x8-11)**2+2*(x9-10)**2+(x10-7)**2+45

pso = PSO(g5)
pso.print_population()