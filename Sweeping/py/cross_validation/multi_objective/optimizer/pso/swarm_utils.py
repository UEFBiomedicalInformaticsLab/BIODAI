from collections.abc import Sequence
from random import random, choice
from random import uniform

from cross_validation.multi_objective.optimizer.pso.particle_mutation import DummyParticleMutation, \
    ConstantParticleMutation, SlowingParticleMutation
from cross_validation.multi_objective.optimizer.pso.particle_updater import ParticleUpdater
from cross_validation.multi_objective.optimizer.pso.swarm_particle import SwarmParticle
from ga_strategy.ga_strategy_bitlist import create_individual_features
from individual.num_features import NumFeatures
from util.hyperbox.hyperbox import Interval


MUTATION_CHOICES = [
    DummyParticleMutation(), ConstantParticleMutation(std_dev=0.1), SlowingParticleMutation(std_dev=0.1)]


def init_particle(n_objectives: int, n_features: int, initial_features: NumFeatures, max_velocity: float,
                  w: Interval, c1: Interval, c2: Interval, theta: float) -> SwarmParticle:
    selected = create_individual_features(ind_size=n_features, num_features=initial_features)
    res_list = []
    for s in selected:
        if s:
            res_list.append(uniform(a=theta, b=1.0))
        else:
            res_list.append(uniform(a=0.0, b=theta))
    updater = ParticleUpdater(
        v_max=max_velocity,
        w=uniform(a=w.a(), b=w.b()),
        c1=uniform(a=c1.a(), b=c1.b()), c2=uniform(a=c2.a(), b=c2.b()))
    mutation = choice(MUTATION_CHOICES)
    return SwarmParticle(n_objectives=n_objectives, seq=res_list, particle_updater=updater, particle_mutation=mutation)


def init_swarm(pop_size: int, n_objectives: int, n_features: int, initial_features: NumFeatures, max_velocity: float,
               w: Interval, c1: Interval, c2: Interval, theta: float) -> Sequence[SwarmParticle]:
    return [init_particle(
        n_objectives=n_objectives, n_features=n_features, initial_features=initial_features,
        max_velocity=max_velocity, w=w, c1=c1, c2=c2, theta=theta)
        for _ in range(pop_size)]
