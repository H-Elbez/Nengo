"""Nengo implementations of Simplified STDP rules."""

import nengo
from nengo.builder import Builder
from nengo.builder.learning_rules import get_pre_ens, get_post_ens
from nengo.builder.operator import Operator
from nengo.builder.signal import Signal
from nengo.params import BoolParam, NumberParam, StringParam
import numpy as np
import math 

# ================
# Frontend objects
# ================
#
# These objects are the ones that you include in your model description.
# They are applied to specific connections between groups of neurons.

class STDP(nengo.learning_rules.LearningRuleType):
    """Simplified Spike-timing dependent plasticity rule."""

    # Used by other Nengo objects
    modifies = 'weights'
    probeable = ('pre_trace', 'post_trace', 'pre_scale', 'post_scale')

    # Parameters
       # var max_weight = 1.0f
       # var min_weight = 0.0001f
       # var alf_p = 0.01f
       # var alf_n = 0.005f
       # var beta_p = 1.5f
       # var beta_n = 2.5f

    alf_p = NumberParam('alf_p', low=0, low_open=True)
    alf_n = NumberParam('alf_n', low=0, low_open=True)
    beta_p = NumberParam('beta_p', low=0, low_open=True)
    beta_n = NumberParam('beta_n', low=0, low_open=True)
    bounds = StringParam('bounds')
    max_weight = NumberParam('max_weight')
    min_weight = NumberParam('min_weight')

    def __init__(
            self,
            alf_p=0.01,
            alf_n=0.005,
            beta_p=1.5,
            beta_n=2.5,
            bounds='hard',
            max_weight=1.0,
            min_weight=0.0001
    ):
        self.alf_p = alf_p
        self.alf_n = alf_n
        self.beta_p = beta_p
        self.beta_n = beta_n
        self.bounds = bounds
        self.max_weight = max_weight
        self.min_weight = min_weight
        super(STDP, self).__init__()
# ===============
# Backend objects
# ===============
#
# These objects let the Nengo core backend know how to implement the rules
# defined in the model through frontend objects. They require some knowledge
# of the low-level details of how the Nengo core backends works, and will
# be different depending on the backend on which the learning rule is implemented.
# The general architecture of the Nengo core backend is described at
#   https://www.nengo.ai/nengo/backend_api.html
# but in the context of learning rules, each learning rule needs a build function
# that is associated with a frontend object (through the `Builder.register`
# function) that sets up the signals and operators that implement the rule.
# Nengo comes with many general purpose operators that could be combined
# to implement a learning rule, but in most cases it is easier to implement
# them using a custom operator that does the delta update equation.
# See, for example, `step_stdp` in the `SimSTDP` operator to see where the
# learning rule's equation is actually specified. The build function exists
# mainly to make sure to all of the signals used in the operator are the
# correct ones. This requires some knowledge of the Nengo core backend,
# but for learning rules they are all very similar, and this could be made
# more convenient through some new abstractions; see
#  https://github.com/nengo/nengo/pull/553
#  https://github.com/nengo/nengo/pull/1149
# for some initial attempts at making this more convenient.

@Builder.register(STDP)
def build_stdp(model, stdp, rule):
    conn = rule.connection
    pre_activities = model.sig[get_pre_ens(conn).neurons]['out']
    post_activities = model.sig[get_post_ens(conn).neurons]['out']
    pre_trace = Signal(np.zeros(pre_activities.size), name="pre_trace")
    post_trace = Signal(np.zeros(post_activities.size), name="post_trace")
    pre_scale = Signal(
        np.zeros(model.sig[conn]['weights'].shape), name="pre_scale")
    post_scale = Signal(
        np.zeros(model.sig[conn]['weights'].shape), name="post_scale")

    model.add_op(SimSTDP(
        pre_activities,
        post_activities,
        pre_trace,
        post_trace,
        pre_scale,
        post_scale,
        model.sig[conn]['weights'],
        model.sig[rule]['delta'],
        alf_p=stdp.alf_p,
        alf_n=stdp.alf_n,
        beta_p=stdp.beta_p,
        beta_n=stdp.beta_n,
        bounds=stdp.bounds,
        max_weight=stdp.max_weight,
        min_weight=stdp.min_weight,
    ))

    # expose these for probes
    model.sig[rule]['pre_trace'] = pre_trace
    model.sig[rule]['post_trace'] = post_trace
    model.sig[rule]['pre_scale'] = pre_scale
    model.sig[rule]['post_scale'] = post_scale

    model.params[rule] = None  # no build-time info to return


class SimSTDP(Operator):
    def __init__(
            self,
            pre_activities,
            post_activities,
            pre_trace,
            post_trace,
            pre_scale,
            post_scale,
            weights,
            delta,
            alf_p,
            alf_n,
            beta_p,
            beta_n,
            bounds,
            max_weight,
            min_weight,
    ):
        self.alf_p = alf_p
        self.alf_n = alf_n
        self.beta_p = beta_p
        self.beta_n = beta_n
        self.bounds = str(bounds).lower()
        self.max_weight = max_weight
        self.min_weight = min_weight

        self.sets = []
        self.incs = []
        self.i = 1
        self.reads = [pre_activities, post_activities, weights]
        self.updates = [delta, pre_trace, post_trace, pre_scale, post_scale]
        
    @property
    def delta(self):
        return self.updates[0]

    @property
    def post_activities(self):
        return self.reads[1]

    @property
    def post_scale(self):
        return self.updates[4]

    @property
    def post_trace(self):
        return self.updates[2]

    @property
    def pre_activities(self):
        return self.reads[0]

    @property
    def pre_scale(self):
        return self.updates[3]

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
        pre_scale = signals[self.pre_scale]
        post_scale = signals[self.post_scale]
        weights = signals[self.weights]
        delta = signals[self.delta]
        i = 0
        # Could be configurable
        beta_pscale = 1.
        beta_nscale = 1.

        if self.bounds == 'hard':

            def update_scales():
                pre_scale[...] = ((self.max_weight - weights) > 0. ).astype(np.float64) * beta_pscale
                post_scale[...] = -((self.min_weight + weights) < 0. ).astype(np.float64) * beta_nscale

        elif self.bounds == 'soft':

            def update_scales():
                pre_scale[...] = (self.max_weight - weights) * beta_pscale
                post_scale[...] = (self.min_weight + weights) * beta_nscale

        elif self.bounds == 'none':

            def update_scales():
                pre_scale[...] = beta_pscale
                post_scale[...] = -beta_nscale

        else:
            raise RuntimeError(
                "Unsupported bounds type. Only 'hard', 'soft' and 'none' are supported")

        def step_stdp():
            #update_scales()

            pre_trace[...]  += np.abs((dt * pre_activities) * ( self.i - pre_trace))
            post_trace[...] += np.abs((dt * post_activities) * ( self.i - post_trace))
            #pre_trace[...]  += (dt * pre_activities)
            #post_trace[...] += (dt * post_activities)

            # for two neurons
            # pre_activities ,pre_trace ,pre_scale 
            # post_activities ,post_trace ,post_scale
            # = (1,) (1,) (1, 1)

            # increase 
            # float(0.001 * math.exp(-1.5 * ((round(weight,6) - 0.0001)/(1.0 - 0.0001))))
            # decrease 
            # float(0.005 * math.exp(-2.5 * ((1.0 - round(weight,6))/(1.0 - 0.0001))))

            if(np.max(weights) > 1 or np.min(weights) < 0):
                weights[...] = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
             

            t = (post_trace[:,np.newaxis] - 50) - pre_trace
            t[t>0] = 0 # decrease
            t[t != 0] = 1 # increase

            delta[...] = ( self.alf_p  *  
            np.exp( -self.beta_p * (( weights - self.min_weight )/( self.max_weight - self.min_weight )) )
            ) * ( t - 0)
            - ( self.alf_n  * 
            np.exp( -self.beta_n * (( self.max_weight - weights )/( self.max_weight - self.min_weight )) )
            ) * ( 1 - t)

            self.i = self.i + 1

        return step_stdp