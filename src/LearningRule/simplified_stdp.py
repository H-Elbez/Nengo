"""Nengo implementations of Simplified STDP rules."""

import nengo
from nengo.builder import Builder
from nengo.builder.learning_rules import get_pre_ens, get_post_ens
from nengo.builder.operator import Operator
from nengo.builder.signal import Signal
from nengo.params import BoolParam, NumberParam, StringParam, Default
import numpy as np
import math 

class STDP(nengo.learning_rules.LearningRuleType):
    """Simplified Spike-timing dependent plasticity rule."""

    # Used by other Nengo objects
    modifies = 'weights'
    probeable = ('pre_trace', 'post_trace',"delta")

    # Parameters

    pre_tau = NumberParam('pre_tau', low=0, low_open=True)
    post_tau = NumberParam('post_tau', low=0, low_open=True)
    alf_p = NumberParam('alf_p', low=0, low_open=True)
    alf_n = NumberParam('alf_n', low=0, low_open=True)
    beta_p = NumberParam('beta_p', low=0, low_open=True)
    beta_n = NumberParam('beta_n', low=0, low_open=True)
    max_weight = NumberParam('max_weight')
    min_weight = NumberParam('min_weight')
    learning_rate = NumberParam("learning_rate", low=0, readonly=True, default=15e-3)
    def __init__(
            self,
            alf_p=0.05,
            alf_n=0.0001,
            beta_p=1.5,
            beta_n=0.5,
            max_weight=1.0,
            min_weight=0.0001,
            pre_tau=0.0168,
            post_tau=0.0337,
            learning_rate=Default,
    ):
        self.pre_tau = pre_tau
        self.post_tau = post_tau
        self.alf_p = alf_p
        self.alf_n = alf_n
        self.beta_p = beta_p
        self.beta_n = beta_n
        self.max_weight = max_weight
        self.min_weight = min_weight
        super().__init__(learning_rate)

@Builder.register(STDP)
def build_stdp(model, stdp, rule):
    conn = rule.connection
    pre_activities = model.sig[get_pre_ens(conn).neurons]['out']
    post_activities = model.sig[get_post_ens(conn).neurons]['out']
    pre_trace = Signal(np.zeros(pre_activities.size), name="pre_trace")
    post_trace = Signal(np.zeros(post_activities.size), name="post_trace")


    model.add_op(SimSTDP(
        pre_activities,
        post_activities,
        pre_trace,
        post_trace,
        model.sig[conn]['weights'],
        model.sig[rule]['delta'],
        pre_tau=stdp.pre_tau,
        post_tau=stdp.post_tau,
        alf_p=stdp.alf_p,
        alf_n=stdp.alf_n,
        beta_p=stdp.beta_p,
        beta_n=stdp.beta_n,
        max_weight=stdp.max_weight,
        min_weight=stdp.min_weight,
        learning_rate=stdp.learning_rate,
    ))

    # expose these for probes
    model.sig[rule]['pre_trace'] = pre_trace
    model.sig[rule]['post_trace'] = post_trace
    
    model.params[rule] = None  # no build-time info to return


class SimSTDP(Operator):
    def __init__(
            self,
            pre_activities,
            post_activities,
            pre_trace,
            post_trace,
            weights,
            delta,
            alf_p,
            alf_n,
            beta_p,
            beta_n,
            max_weight,
            min_weight,
            pre_tau,
            post_tau,
            learning_rate,
            tag=None
    ):
        super(SimSTDP,self).__init__(tag=tag)
        self.learning_rate = learning_rate
        self.alf_p = alf_p
        self.alf_n = alf_n
        self.beta_p = beta_p
        self.beta_n = beta_n
        self.pre_tau = pre_tau
        self.post_tau = post_tau
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.sets = []
        self.incs = []
        self.reads = [pre_activities, post_activities, weights]
        self.updates = [delta, pre_trace, post_trace]
        
    @property
    def delta(self):
        return self.updates[0]

    @property
    def post_activities(self):
        return self.reads[1]

    @property
    def post_trace(self):
        return self.updates[2]

    @property
    def pre_activities(self):
        return self.reads[0]

    @property
    def pre_trace(self):
        return self.updates[1]

    @property
    def weights(self):
        return self.reads[2]

    def make_step(self, signals, dt, rng):

        pre_activities = signals[self.pre_activities]
        post_activities = signals[self.post_activities]
        pre_trace = signals[self.pre_trace]
        post_trace = signals[self.post_trace]
        weights = signals[self.weights]
        delta = signals[self.delta]

        alphaP = self.learning_rate * (dt + self.alf_p) 
        alphaN = self.learning_rate * (dt + self.alf_n) 

        #alphaP = self.alf_p 
        #alphaN = self.alf_n
        def step_stdp():

            pre_trace[...] += ((dt / self.pre_tau) * (-pre_trace + pre_activities))

            post_trace[...] += ((dt / self.post_tau) * (-post_trace + post_activities))

            delta[...] = (( alphaP  *  np.exp( - self.beta_p * (( weights - self.min_weight )/( self.max_weight - self.min_weight )) )) * pre_trace[np.newaxis, :] - ( alphaN  * np.exp( - self.beta_n * (( self.max_weight - weights )/( self.max_weight - self.min_weight )) )) * post_trace[:, np.newaxis]) * post_activities[:, np.newaxis] * dt

            np.putmask(delta,((weights + delta) < self.min_weight),self.min_weight - weights)
            np.putmask(delta,((weights + delta) > self.max_weight),self.max_weight - weights)

        return step_stdp