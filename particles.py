import numpy as np

class ParticleLike:
    def __init__(self):
        pass
    def update(self, attractor_pos):
        pass
    def known_best_position(self):
        pass
    def known_best_objective(self):
        pass
    def true_best_position(self):
        pass
    def true_best_objective(self):
        pass
    def all_best_positions(self):
        pass
    def nested_best_positions(self):
        pass
    def all_particles(self):
        pass

class Particle(ParticleLike):
    def __init__(self, init_pos, objective):
        self.objective = objective
        self.best_pos: np.ndarray = init_pos
        self.best_obj: float = self.objective(init_pos)
        self.n_dims: int = init_pos.shape[0]
    def update(self, attractor_pos):
        # calculate new position for particle
        mean = (self.best_pos + attractor_pos) / 2
        stdev = np.linalg.norm(self.best_pos - attractor_pos)
        current_pos = np.random.multivariate_normal(mean, np.identity(self.n_dims) * stdev).astype(self.best_pos.dtype)
        
        # check if new position is better than current best
        current_obj = self.objective(current_pos)
        if current_obj < self.best_obj:
            self.best_obj = current_obj
            self.best_pos = current_pos
    def known_best_position(self):
        return self.best_pos
    def known_best_objective(self):
        return self.best_obj
    def true_best_position(self):
        return self.best_pos
    def true_best_objective(self):
        return self.best_obj
    def all_best_positions(self):
        return [self.best_pos]
    def nested_best_positions(self):
        return self.best_pos
    def all_particles(self):
        return [self]

class SwarmParameters:
    def __init__(self,  
                 depth, 
                 inner_particles, 
                 init_lower_bounds, 
                 init_upper_bounds, 
                 division_factor, 
                 objective, 
                 attractor_type, 
                 multiplicity):
        # how many layers of particles exist below this one?
        self.depth: int = depth
        # how many actual particles are in this swarm?
        self.inner_particles: int = inner_particles
        # what are the lower and upper bounds of the search?
        self.init_lower_bounds: np.ndarray = init_lower_bounds
        self.init_upper_bounds: np.ndarray = init_upper_bounds
        # how will the search space be divided between child swarms?
        self.division_factor: int = division_factor
        # how are particle positions scored?
        self.objective = objective
        # how will the swarms interact with eachother in terms of sharing their best positions?
        # options:
        #  - separate: each swarm only knows about its own best position
        #  - child: each swarm knows about its best position and the best position of any child swarms
        #  - parent: each swarm knows about its best position, the best position of any children, and the best position of its parent (i.e. every swarm knows the global best)
        self.attractor_type: str = attractor_type
        # how many swarms will be searching the same space?
        self.multiplicity: int = multiplicity
    def copy(self):
        new_params = SwarmParameters(self.depth,
                                     self.inner_particles,
                                     self.init_lower_bounds,
                                     self.init_upper_bounds,
                                     self.division_factor,
                                     self.objective,
                                     self.attractor_type,
                                     self.multiplicity)
        return new_params

def generate_random_position(upper_bounds, lower_bounds):
    dims = np.empty_like(upper_bounds)
    for i in range(len(dims)):
        diff = upper_bounds[i] - lower_bounds[i]
        dims[i] = np.random.rand() * diff + lower_bounds[i]
    return dims

def divide_space(space_upper_bounds, space_lower_bounds, division_factor):
    if division_factor == 1:
        yield (space_upper_bounds, space_lower_bounds)
        return

    indices = np.zeros_like(space_upper_bounds)
    dimensions = indices.shape[0]
    break_out = False

    while not break_out:
        for d in range(dimensions):
            indices[d] += 1
            if indices[d] == division_factor and d == dimensions - 1:
                indices[d] = 0
                break_out = True
                break
            if indices[d] == division_factor:
                indices[d] = 0
            else:
                break
        divided_uppers = np.empty_like(space_upper_bounds)
        divided_lowers = np.empty_like(space_lower_bounds)
        for d in range(dimensions):
            diff = space_upper_bounds[d] - space_lower_bounds[d]
            divided_uppers[d] = (indices[d] + 1) * diff / division_factor + space_lower_bounds[d]
            divided_lowers[d] = (indices[d]) * diff / division_factor + space_lower_bounds[d]
        yield (divided_uppers, divided_lowers)

class Swarm(ParticleLike):
    def __init__(self, params: SwarmParameters):
        self.attractor_type: str = params.attractor_type
        self.particles: list[ParticleLike] = []
        for i in range(params.inner_particles):
            rand_pos = generate_random_position(params.init_upper_bounds, params.init_lower_bounds)
            self.particles.append(Particle(rand_pos, params.objective))
        if params.depth > 0:
            for space_upper, space_lower in divide_space(params.init_upper_bounds, params.init_lower_bounds, params.division_factor):
                for i in range(params.multiplicity):
                    new_params = params.copy()
                    new_params.depth -= 1
                    new_params.init_upper_bounds = space_upper
                    new_params.init_lower_bounds = space_lower
                    self.particles.append(Swarm(new_params))
        self.best_pos = self.particles[0].known_best_position()
        self.best_obj = self.particles[0].known_best_objective()
        # if the swarms are not separate, initialize the best objective to be the best among all child particles and swarms
        if self.attractor_type != 'separate':
            for particle in self.particles:
                obj = particle.known_best_objective()
                if obj < self.best_obj:
                    self.best_obj = obj
                    self.best_pos = particle.known_best_position()
        # otherwise, initialize the best objective to be the best among only the child particles
        else:
            for particle in self.particles:
                if isinstance(particle, Swarm):
                    continue
                obj = particle.known_best_objective()
                if obj < self.best_obj:
                    self.best_obj = obj
                    self.best_pos = particle.known_best_position()
    def update(self, attractor_pos):
        if self.attractor_type == 'separate':
            # ignore the parent attractor, update own attractor to align with best child particle
            for particle in self.particles:
                particle.update(self.best_pos)
            for particle in self.particles:
                if isinstance(particle, Swarm):
                    continue
                obj = particle.known_best_objective()
                if obj < self.best_obj:
                    self.best_obj = obj
                    self.best_pos = particle.known_best_position()
        elif self.attractor_type == 'child':
            # ignore the parent attractor, but update own attractor to align with best among all children
            for particle in self.particles:
                particle.update(self.best_pos)
            for particle in self.particles:
                obj = particle.known_best_objective()
                if obj < self.best_obj:
                    self.best_obj = obj
                    self.best_pos = particle.known_best_position()
        elif self.attractor_type == 'parent':
            # utilize the parent attractor, and update own attractor to align with best among all children
            for particle in self.particles:
                particle.update(attractor_pos)
            for particle in self.particles:
                obj = particle.known_best_objective()
                if obj < self.best_obj:
                    self.best_obj = obj
                    self.best_pos = particle.known_best_position()
            pass
        else:
            raise Exception()
    def known_best_position(self):
        return self.best_pos
    def known_best_objective(self):
        return self.best_obj
    def true_best_position(self):
        best_obj = self.best_obj
        best_pos = self.best_pos

        for particle in self.particles:
            obj = particle.true_best_objective()
            if obj < best_obj:
                best_obj = obj
                best_pos = particle.true_best_position()

        return best_pos
    def true_best_objective(self):
        best_obj = self.best_obj

        for particle in self.particles:
            obj = particle.true_best_objective()
            if obj < best_obj:
                best_obj = obj

        return best_obj
    def all_best_positions(self):
        positions = []
        for particle in self.particles:
            positions.extend(particle.all_best_positions())
        return positions
    def nested_best_positions(self):
        positions = []
        for particle in self.particles:
            positions.append(particle.nested_best_positions())
        return positions
    def all_particles(self):
        particles = []
        for particle in self.particles:
            particles.extend(particle.all_particles())
        return particles
