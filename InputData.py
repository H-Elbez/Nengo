from nengo.params import NdarrayParam ,NumberParam
from nengo.base import Process
from decimal import Decimal

class PresentInputWithPause(Process):
    """Present a series of inputs, each for the same fixed length of time.

    Parameters
    ----------
    inputs : array_like
        Inputs to present, where each row is an input. Rows will be flattened.
    presentation_time : float
        Show each input for this amount of time (in seconds).
    pause_time : float
        Pause time after each input (in seconds).
    """

    inputs = NdarrayParam("inputs", shape=("...",))
    presentation_time = NumberParam("presentation_time", low=0, low_open=True)
    pause_time = NumberParam("pause_time", low=0, low_open=True)

    def __init__(self, inputs, presentation_time,pause_time, **kwargs):
        self.inputs = inputs
        self.presentation_time = presentation_time
        self.pause_time = pause_time
        self.localT = 0
        self.index = 0
        super().__init__(
            default_size_in=0, default_size_out=self.inputs[0].size, **kwargs
        )

    def make_step(self, shape_in, shape_out, dt, rng, state):
        assert shape_in == (0,)
        assert shape_out == (self.inputs[0].size,)
        n = len(self.inputs)
        inputs = self.inputs.reshape(n, -1)
        presentation_time = float(self.presentation_time)
        pause_time = float(self.pause_time)
        self.localT = round((dt if self.localT == 0 else self.localT),2)

        def step_presentinput(t):
            #t = abs(t - pause_time)
            t = round(t,6)
            # Pause
            if t > ((presentation_time + pause_time ) * self.index + presentation_time) and t < round((presentation_time + pause_time) * (self.index + 1),6) :
                
                i = 0
                # return np.zeros((len(inputs)))
                return inputs[i % n]
            else:
            # Send input
                #if t >= (presentation_time + pause_time) * i:
                #    t = t - (pause_time * i)
                    
                i = int((self.localT - dt) / (presentation_time))
                self.localT += dt

                if t == round((presentation_time + pause_time) * (self.index + 1),6):
                    self.index +=1

                # i = 0
                return inputs[i % n]
       
        return step_presentinput