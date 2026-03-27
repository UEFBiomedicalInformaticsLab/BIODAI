from collections.abc import Iterable
from typing import Optional

from cross_validation.multi_objective.optimizer.pso.particle_mutation import ParticleMutation
from cross_validation.multi_objective.optimizer.pso.particle_updater import ParticleUpdater
from individual.fit_individual import FitIndividual
from individual.fit_individual_dense import FitIndividualDense
from individual.peculiar_individual_dense import PeculiarIndividualDense


class SwarmParticle(PeculiarIndividualDense):
    __velocity: list[float]
    __p_best: Optional[FitIndividual]
    __particle_updater: ParticleUpdater
    __particle_mutation: ParticleMutation

    def __init__(self, n_objectives: int,
                 particle_updater: ParticleUpdater, particle_mutation: ParticleMutation, seq=()):
        PeculiarIndividualDense.__init__(self, n_objectives=n_objectives, seq=seq)
        self.__p_best = None
        self.__particle_updater = particle_updater
        self.__particle_mutation = particle_mutation
        self.__velocity = [0]*len(seq)
        # Initialized at 0 as suggested in "Particle swarm optimization: Velocity initialization".

    def __current_as_best(self):
        self.__p_best = FitIndividualDense(fitness=self.get_test_fitness(), seq=self)

    def __update_p_best(self):
        if self.has_fitness():
            if self.__p_best is None or self.fitness.dominates(self.__p_best.fitness):
                self.__current_as_best()

    def p_best(self) -> FitIndividual:
        return self.__p_best

    def update_cinematic(self, g_best: Iterable[float], completion: float):
        """Updates p_best, applies flight equations and then mutation."""
        self.__update_p_best()
        vel = self.__velocity
        updated = self.__particle_updater.update_particle(
            x=self, v=vel, p_best=self.__p_best, g_best=g_best)
        for i, u in enumerate(updated):
            self[i]=u[0]
            vel[i]=u[1]
        self.__particle_mutation.mutate(particle=self, completion=completion)
        del self.fitness.values  # Invalidate fitness so it will be computed again.
