# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2022/4/10 18:46
# User      : Floyed
# Product   : PyCharm
# Project   : braincog
# File      : node.py


import abc
import math
from abc import ABC
import numpy as np
import random
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from einops import rearrange, repeat
from surrogate import *
from functools import reduce
import os

class BaseNode(nn.Module, abc.ABC):
    """
    Base class for neuron models

    :param threshold: The threshold that the neuron needs to reach to fire a pulse

    :param v_reset: Resting potential

    :param dt: Time step

    :param step: Simulation step

    :param requires_thres_grad: Whether to calculate the gradient with respect to the threshold, defaults to `False`

    :param sigmoid_thres: Whether to use sigmoid constraint to limit the threshold to [0, 1], defaults to `False`

    :param requires_fp: Whether to save the feature map during inference, requiring additional memory and time, defaults to `False`

    :param layer_by_layer: Whether to calculate the output of all steps at once. In the case of a large network model, this generally shortens the time of a single inference, defaults to `False`

    :param n_groups: Whether to use different weights at different time steps, defaults to `1`, i.e., no grouping
    :param mem_detach: Whether to truncate the membrane potential from the previous time step in the computational graph

    :param args: Other parameters

    :param kwargs: Other parameters
    """

    def __init__(self,
                 threshold=.5,
                 v_reset=0.,
                 dt=1.,
                 step=8,
                 requires_thres_grad=False,
                 sigmoid_thres=False,
                 requires_fp=False,
                 layer_by_layer=False,
                 n_groups=1,
                 *args,
                 **kwargs):

        super(BaseNode, self).__init__()
        self.threshold = Parameter(torch.tensor(threshold), requires_grad=requires_thres_grad)
        self.sigmoid_thres = sigmoid_thres
        self.mem = 0.
        self.spike = 0.
        self.dt = dt
        self.feature_map = []
        self.mem_collect = []
        self.requires_fp = requires_fp
        self.v_reset = v_reset
        self.step = step
        self.layer_by_layer = layer_by_layer
        self.groups = n_groups
        self.mem_detach = kwargs['mem_detach'] if 'mem_detach' in kwargs else False
        self.requires_mem = kwargs['requires_mem'] if 'requires_mem' in kwargs else False

    @abc.abstractmethod
    def calc_spike(self):
        """
        Calculate whether to issue a pulse based on the current memory, and then reset:

        :return: None
        """

        pass

    def integral(self, inputs):
        """
        Calculate the cumulative membrane potential based on the current inputs.

        :param inputs: Current synaptic input current

        :type inputs: torch.tensor

        :return: None
        """

        pass

    def get_thres(self):
        return self.threshold if not self.sigmoid_thres else self.threshold.sigmoid()

    def rearrange2node(self, inputs):
        if self.groups != 1:
            if len(inputs.shape) == 4:
                outputs = rearrange(inputs, 'b (c t) w h -> t b c w h', t=self.step)
            elif len(inputs.shape) == 2:
                outputs = rearrange(inputs, 'b (c t) -> t b c', t=self.step)
            else:
                raise NotImplementedError

        elif self.layer_by_layer:
            if len(inputs.shape) == 4:
                outputs = rearrange(inputs, '(t b) c w h -> t b c w h', t=self.step)
            elif len(inputs.shape) == 3:
                outputs = rearrange(inputs, '(t b) n c -> t b n c', t=self.step)
            elif len(inputs.shape) == 2:
                outputs = rearrange(inputs, '(t b) c -> t b c', t=self.step)
            else:
                raise NotImplementedError


        else:
            outputs = inputs

        return outputs

    def rearrange2op(self, inputs):
        if self.groups != 1:
            if len(inputs.shape) == 5:
                outputs = rearrange(inputs, 't b c w h -> b (c t) w h')
            elif len(inputs.shape) == 3:
                outputs = rearrange(inputs, ' t b c -> b (c t)')
            else:
                raise NotImplementedError
        elif self.layer_by_layer:
            if len(inputs.shape) == 5:
                outputs = rearrange(inputs, 't b c w h -> (t b) c w h')
            elif len(inputs.shape) == 4:
                outputs = rearrange(inputs, ' t b n c -> (t b) n c')
            elif len(inputs.shape) == 3:
                outputs = rearrange(inputs, ' t b c -> (t b) c')
            else:
                raise NotImplementedError

        else:
            outputs = inputs

        return outputs

    def forward(self, inputs):
        """
        The `torch.nn.Module` is the default function called to calculate the membrane potential input and the output pulse.

        When `self.requires_fp is True`, `self.feature_map` can be used to record the trace.

        `:param inputs: The current input membrane potential

        `:return: The output pulse`
        """

        if hasattr(self, 'parallel') and self.parallel is True:
            inputs = self.rearrange2node(inputs)
            if self.mem_detach and hasattr(self.mem, 'detach'):
                self.mem = self.mem.detach()
                self.spike = self.spike.detach()
            self.integral(inputs)

            self.calc_spike()

            if self.requires_fp is True:
                self.feature_map.append(self.spike)
            if self.requires_mem is True:
                self.mem_collect.append(self.mem)

            return self.rearrange2op(self.spike)

        elif self.layer_by_layer or self.groups != 1:
            inputs = self.rearrange2node(inputs)

            outputs = []
            for i in range(self.step):
                
                if self.mem_detach and hasattr(self.mem, 'detach'):
                    self.mem = self.mem.detach()
                    self.spike = self.spike.detach()
                self.integral(inputs[i])
                
                self.calc_spike()
                
                if self.requires_fp is True:
                    self.feature_map.append(self.spike)
                if self.requires_mem is True:
                    self.mem_collect.append(self.mem)
                outputs.append(self.spike)
            outputs = torch.stack(outputs)

            outputs = self.rearrange2op(outputs)
            return outputs
        else:
            if self.mem_detach and hasattr(self.mem, 'detach'):
                self.mem = self.mem.detach()
                self.spike = self.spike.detach()
            self.integral(inputs)
            self.calc_spike()
            if self.requires_fp is True:
                self.feature_map.append(self.spike)
            if self.requires_mem is True:
                self.mem_collect.append(self.mem)   
            return self.spike

    def n_reset(self):
        """
        Neuron reset, used when the model receives two unrelated inputs, to reset all states of a neuron.

        :return: None
        """
        self.mem = self.v_reset
        self.spike = 0.
        self.feature_map = []
        self.mem_collect = []
    def get_n_attr(self, attr):

        if hasattr(self, attr):
            return getattr(self, attr)
        else:
            return None

    def set_n_warm_up(self, flag):
        """
        Some training strategies treat neurons as activation functions of the ANN during the initial epochs. This setting determines whether to use this method:

        :param flag: True: Neurons become activation functions; False: No change

        :return: None
        """
        self.warm_up = flag

    def set_n_threshold(self, thresh):
        """
        Dynamically set the threshold for neurons

        :param thresh: Threshold

        :return:
        """
        self.threshold = Parameter(torch.tensor(thresh, dtype=torch.float), requires_grad=False)

    def set_n_tau(self, tau):
        """
        Dynamically set the decay coefficient of neurons for Leaky neurons

        :param tau: Decay coefficient

        :return:
        """
        if hasattr(self, 'tau'):
            self.tau = Parameter(torch.tensor(tau, dtype=torch.float), requires_grad=False)
        else:
            raise NotImplementedError

#============================================================================
# node的基类
class BaseMCNode(nn.Module, abc.ABC):
    """
    Base class for multi-compartment neuron models

    :param threshold: The threshold required for the neuron to fire a pulse

    :param v_reset: Resting potential

    :param comps: Different compartments of the neuron, e.g., ["apical", "basal", "soma"]
    """
    def __init__(self,
                 threshold=1.0,
                 v_reset=0.,
                 comps=[]):
        super().__init__()
        self.threshold = Parameter(torch.tensor(threshold), requires_grad=False)
        # self.decay = Parameter(torch.tensor(decay), requires_grad=False)
        self.v_reset = v_reset
        assert len(comps) != 0
        self.mems = dict()
        for c in comps:
            self.mems[c] = None 
        self.spike = None
        self.warm_up = False

    @abc.abstractmethod
    def calc_spike(self):
        pass
    @abc.abstractmethod
    def integral(self, inputs):
        pass        
    
    def forward(self, inputs: dict):
        '''
        Params:
            inputs dict: Inputs for every compartments of neuron 
        '''
        if self.warm_up:
            return inputs
        else:
            self.integral(**inputs)
            self.calc_spike()
            return self.spike

    def n_reset(self):
        for c in self.mems.keys():
            self.mems[c] = self.v_reset
        self.spike = 0.0

    def get_n_fire_rate(self):
        if self.spike is None:
            return 0.
        return float((self.spike.detach() >= self.threshold).sum()) / float(np.product(self.spike.shape))

    def set_n_warm_up(self, flag):
        self.warm_up = flag

    def set_n_threshold(self, thresh):
        self.threshold = Parameter(torch.tensor(thresh, dtype=torch.float), requires_grad=False)



class ThreeCompNode(BaseMCNode):
    """
    Three-compartment neuron model

    :param threshold: The threshold that the neuron needs to reach to fire a pulse

    :param v_reset: Resting potential

    :param tau: Cell body membrane potential time constant, used to control the decay of the cell body membrane potential

    :param tau_basal: Basal dendritic membrane potential time constant, used to control the decay of the basal dendritic cell body membrane potential

    :param tau_apical: Distal dendritic membrane potential time constant, used to control the decay of the distal dendritic cell body membrane potential

    :param comps: Different compartments of the neuron, e.g., ["apical", "basal", "soma"]

    :param act_fun: Pulse gradient surrogate function
    """
    def __init__(self,
                 threshold=1.0,
                 tau=2.0,
                 tau_basal=2.0,
                 tau_apical=2.0,
                 v_reset=0.0,
                 comps=['basal', 'apical', 'soma'],
                 dt=0.5,
                 act_fun=AtanGrad):
        g_B = 0.6
        g_L = 0.05
        super().__init__(threshold, v_reset, comps)
        self.tau = tau
        self.tau_basal = tau_basal
        self.tau_apical = tau_apical
        self.act_fun = act_fun(alpha=tau, requires_grad=False)
    
    def integral(self, basal_inputs, apical_inputs):
        '''
        Params:
            inputs torch.Tensor: Inputs for basal dendrite  
        '''

        self.mems['basal'] =  (self.mems['basal'] + basal_inputs) / self.tau_basal
        self.mems['apical'] =  (self.mems['apical'] + apical_inputs) / self.tau_apical

        self.mems['soma'] = self.mems['soma'] + (self.mems['apical'] + self.mems['basal'] - self.mems['soma']) / self.tau


    def calc_spike(self):
        self.spike = self.act_fun(self.mems['soma'] - self.threshold)
        self.mems['soma'] = self.mems['soma']  * (1. - self.spike.detach())
        self.mems['basal'] = self.mems['basal'] * (1. - self.spike.detach())
        self.mems['apical'] = self.mems['apical']  * (1. - self.spike.detach())

class ThreeCompNode_record(BaseMCNode):
    """
    Three-compartment neuron model

    :param threshold: The threshold that the neuron needs to reach to fire a pulse

    :param v_reset: Resting potential

    :param tau: Cell body membrane potential time constant, used to control the decay of the cell body membrane potential

    :param tau_basal: Basal dendritic membrane potential time constant, used to control the decay of the basal dendritic cell body membrane potential

    :param tau_apical: Distal dendritic membrane potential time constant, used to control the decay of the distal dendritic cell body membrane potential

    :param comps: Different compartments of the neuron, e.g., ["apical", "basal", "soma"]

    :param act_fun: Pulse gradient surrogate function
    """
    def __init__(self,
                 threshold=1.0,
                 tau=2.0,
                 tau_basal=2.0,
                 tau_apical=2.0,
                 v_reset=0.0,
                 comps=['basal', 'apical', 'soma'],
                 C_soma=0.0,
                 g_soma=0.0,
                 act_fun=AtanGrad):
        g_B = 0.6
        g_L = 0.05
        super().__init__(threshold, v_reset, comps)
        self.tau = tau
        self.tau_basal = tau_basal
        self.tau_apical = tau_apical
        self.act_fun = act_fun(alpha=tau, requires_grad=False)
        self.spikes_record = [] 
        #**Fixed conductance and capacitance: Random values ​​are generated only once.**
        self.C_soma = C_soma  # capacitance
        self.g_soma = g_soma  # electrical conductivity
    
    def integral(self, basal_inputs, apical_inputs):
        '''
        Params:
            inputs torch.Tensor: Inputs for basal dendrite  
        '''

        self.mems['basal'] =  (self.mems['basal'] + basal_inputs) / self.tau_basal
        self.mems['apical'] =  (self.mems['apical'] + apical_inputs) / self.tau_apical

        self.mems['soma'] = self.mems['soma'] +self.g_soma* (self.mems['apical'] + self.mems['basal'] - self.mems['soma']) / (self.tau*self.C_soma)


    def calc_spike(self):
        self.spike = self.act_fun(self.mems['soma'] - self.threshold)
        
        self.mems['soma'] = self.mems['soma']  * (1. - self.spike.detach())
        self.mems['basal'] = self.mems['basal'] * (1. - self.spike.detach())
        self.mems['apical'] = self.mems['apical']  * (1. - self.spike.detach())
    def save_spikes(self, step, save_path='./'):
        '''Segmented pulse data storage'''
        # Filenames can be generated based on the number of steps, ensuring that each saved file is not overwritten.
        self.spikes_record.append(self.spike.detach().cpu().numpy())
        file_name = f"spikes_record_step_{step}.pt"
    
        # Full path: Combines the filename and the save path
        full_path = os.path.join(save_path, file_name)
        
        # Save as .pt file
        torch.save(self.spikes_record, full_path)
        print(f"Spikes saved to {file_name}")
        # Clear the current pulse record list to prepare for saving the next segment.
        self.spikes_record = []

class ThreeCompNode_tclif(BaseMCNode):
    def __init__(self,
                 threshold=1.0,
                 tau=2.0,
                 tau_basal=2.0,
                 tau_apical=2.0,
                 v_reset=0.0,
                 comps=['basal', 'apical', 'soma'],
                 C_soma=0.0,
                 g_soma=0.0,
                 decay_factor: torch.Tensor = torch.full([1, 4], 0, dtype=torch.float),
                 act_fun=AtanGrad,
                 gamma: float = 0.5):
        g_B = 0.6
        g_L = 0.05
        super().__init__(threshold, v_reset, comps)
        self.act_fun = act_fun(alpha=tau, requires_grad=False)
        self.spikes_record = [] 

        self.C_soma = C_soma  
        self.g_soma = g_soma  
        self.decay = decay_factor
        self.decay_factor = torch.nn.Parameter(decay_factor)
        self.gamma = gamma
    
    def integral(self, basal_inputs, apical_inputs):
        '''
        Params:
            inputs torch.Tensor: Inputs for basal dendrite  
        '''
        self.mems['apical'] = self.mems['apical'] - torch.sigmoid(self.decay_factor[0][0]) * self.mems['soma'] + apical_inputs
        self.mems['basal'] = self.mems['basal'] - torch.sigmoid(self.decay_factor[0][3]) * self.mems['soma'] + basal_inputs
        self.mems['soma'] = self.mems['soma'] + torch.sigmoid(self.decay_factor[0][1]) * self.mems['apical']+ torch.sigmoid(self.decay_factor[0][2]) * self.mems['basal']

    def jit_soft_reset(self,v: torch.Tensor, spike: torch.Tensor, v_threshold: float):
        v = v - spike * v_threshold
        return v
    
    def calc_spike(self):
        self.spike = self.act_fun(self.mems['soma'] - self.threshold)
        # soft reset
        self.mems['apical'] = self.jit_soft_reset(self.mems['apical'], self.spike.detach(), self.gamma)
        self.mems['basal'] = self.jit_soft_reset(self.mems['basal'], self.spike.detach(), self.gamma)
        self.mems['soma'] = self.jit_soft_reset(self.mems['soma'], self.spike.detach(), self.threshold)

    def save_spikes(self, step, save_path='./'):
        self.spikes_record.append(self.spike.detach().cpu().numpy())
        file_name = f"spikes_record_step_{step}.pt"

        full_path = os.path.join(save_path, file_name)
        
        torch.save(self.spikes_record, full_path)
        print(f"Spikes saved to {file_name}")

        self.spikes_record = []


class ThreeCompNode_bilinear(BaseMCNode):
    def __init__(self,
                 threshold=1.0,
                 tau=2.0,
                 tau_basal=2.0,
                 tau_apical=2.0,
                 v_reset=0.0,
                 comps=['basal', 'apical', 'soma'],
                 dt=0.5,
                 act_fun=AtanGrad):
        g_B = 0.6
        g_L = 0.05
        super().__init__(threshold, v_reset, comps)
        self.tau = tau
        self.tau_basal = tau_basal
        self.tau_apical = tau_apical
        self.act_fun = act_fun(alpha=tau, requires_grad=False)
        init_w = -math.log(tau - 1.)
        self.alph_basal = nn.Parameter(torch.as_tensor(init_w))
        self.alph_apical = nn.Parameter(torch.as_tensor(init_w))
        self.alph_soma = nn.Parameter(torch.as_tensor(init_w))
        self.dt=dt
    
    def integral(self, basal_inputs, apical_inputs):
        '''
        Params:
            inputs torch.Tensor: Inputs for basal dendrite  
        '''

        self.mems['basal'] =   ((basal_inputs + self.mems['basal']) ) / self.tau_basal
        self.mems['apical'] =  ((apical_inputs + self.mems['apical'])) / self.tau_apical   
        self.mems['soma'] =self.mems['soma'] + ((self.mems['apical']*self.mems['basal'].expand(-1, self.mems['apical'].shape[1], -1))+ (self.mems['apical'] + self.mems['basal'].expand(-1, self.mems['apical'].shape[1], -1))- self.mems['soma'])/self.tau

        

    def calc_spike(self):
        self.spike = self.act_fun(self.mems['soma'] - self.threshold)
        self.mems['soma'] = self.mems['soma']  * (1. - self.spike.detach())
        self.mems['basal'] = self.mems['basal'] * (1. - self.spike.detach())
        self.mems['apical'] = self.mems['apical']  * (1. - self.spike.detach())
    
    def save_spikes(self, step, save_path='./'):
        self.spikes_record.append(self.spike.detach().cpu().numpy())
        file_name = f"spikes_record_step_{step}.pt"
    
        full_path = os.path.join(save_path, file_name)
  
        torch.save(self.spikes_record, full_path)
        print(f"Spikes saved to {file_name}")

        self.spikes_record = []






class ThreeCompNode_plif1(BaseMCNode):
    def __init__(self,
                 threshold=1.0,
                 tau=2.0,
                 tau_basal=2.0,
                 tau_apical=2.0,
                 v_reset=0.0,
                 comps=['basal', 'apical', 'soma'],
                 act_fun=AtanGrad):
        g_B = 0.6
        g_L = 0.05
        super().__init__(threshold, v_reset, comps)
        self.tau = tau
        self.tau_basal = tau_basal
        self.tau_apical = tau_apical
        self.act_fun = act_fun(alpha=tau, requires_grad=False)
        init_w = -math.log(tau - 1.)
        self.alph_basal = nn.Parameter(torch.as_tensor(init_w))
        self.alph_apical = nn.Parameter(torch.as_tensor(init_w))
        self.alph_soma = nn.Parameter(torch.as_tensor(init_w))
    
    def integral(self, basal_inputs, apical_inputs):
        '''
        Params:
            inputs torch.Tensor: Inputs for basal dendrite  
        '''

        self.mems['basal'] =  (self.mems['basal'] + basal_inputs) / self.tau_basal
        self.mems['apical'] =  (self.mems['apical'] + apical_inputs) / self.tau_apical

        self.mems['soma'] = self.mems['soma'] + (self.mems['apical']* self.alph_soma.sigmoid() + self.mems['basal']*(1- self.alph_soma.sigmoid()) - self.mems['soma'])/ self.tau


    def calc_spike(self):
        self.spike = self.act_fun(self.mems['soma'] - self.threshold)
        self.mems['soma'] = self.mems['soma']  * (1. - self.spike.detach())
        self.mems['basal'] = self.mems['basal'] * (1. - self.spike.detach())
        self.mems['apical'] = self.mems['apical']  * (1. - self.spike.detach())

class ThreeCompNode_plif3_human(BaseMCNode):
    def __init__(self,
                 threshold=1.0,
                 tau=2.0,
                 tau_basal=2.0,
                 tau_apical=2.0,
                 v_reset=0.0,
                 comps=['basal', 'apical', 'soma'],
                 dt=0.5,
                 act_fun=AtanGrad):
        g_B = 0.6
        g_L = 0.05
        super().__init__(threshold, v_reset, comps)
        self.tau = tau
        self.tau_basal = tau_basal
        self.tau_apical = tau_apical
        self.act_fun = act_fun(alpha=tau, requires_grad=False)
        init_w = -math.log(tau - 1.)
        self.alph_basal = nn.Parameter(torch.as_tensor(init_w))
        self.alph_apical = nn.Parameter(torch.as_tensor(init_w))
        self.alph_soma = nn.Parameter(torch.as_tensor(init_w))
        self.dt=dt
        self.spikes_record = [] 
    
    def integral(self, basal_inputs, apical_inputs):
        '''
        Params:
            inputs torch.Tensor: Inputs for basal dendrite  
        '''
        self.mems['basal'] =   ((basal_inputs + self.mems['basal']) ) / self.tau_basal
        self.mems['apical'] =  ((apical_inputs + self.mems['apical'])) / self.tau_apical
        self.mems['soma'] =self.mems['soma'] + ((self.mems['apical']*self.mems['basal'].expand(-1, self.mems['apical'].shape[1], -1))* 0.314+ (self.mems['apical'] + self.mems['basal'].expand(-1, self.mems['apical'].shape[1], -1))*(0.686)- self.mems['soma'])/self.tau
        
    def calc_spike(self):
        self.spike = self.act_fun(self.mems['soma'] - self.threshold)
        self.mems['soma'] = self.mems['soma']  * (1. - self.spike.detach())
        self.mems['basal'] = self.mems['basal'] * (1. - self.spike.detach())
        self.mems['apical'] = self.mems['apical']  * (1. - self.spike.detach())
    
    def save_spikes(self, step,epoch, save_path='./'):
        self.spikes_record.append(self.spike.detach().cpu().numpy())
        file_name = f"spikes_record_step_{step}_episode{epoch}.pt"

        full_path = os.path.join(save_path, file_name)

        torch.save(self.spikes_record, full_path)
        print(f"Spikes saved to {file_name}")

        self.spikes_record = []

class ThreeCompNode_plif3_macaque(BaseMCNode):
    def __init__(self,
                 threshold=1.0,
                 tau=2.0,
                 tau_basal=2.0,
                 tau_apical=2.0,
                 v_reset=0.0,
                 comps=['basal', 'apical', 'soma'],
                 dt=0.5,
                 act_fun=AtanGrad):
        g_B = 0.6
        g_L = 0.05
        super().__init__(threshold, v_reset, comps)
        self.tau = tau
        self.tau_basal = tau_basal
        self.tau_apical = tau_apical
        self.act_fun = act_fun(alpha=tau, requires_grad=False)
        init_w = -math.log(tau - 1.)
        self.alph_basal = nn.Parameter(torch.as_tensor(init_w))
        self.alph_apical = nn.Parameter(torch.as_tensor(init_w))
        self.alph_soma = nn.Parameter(torch.as_tensor(init_w))
        self.dt=dt
        self.spikes_record = [] 
    
    def integral(self, basal_inputs, apical_inputs):
        '''
        Params:
            inputs torch.Tensor: Inputs for basal dendrite  
        '''
        self.mems['basal'] =   ((basal_inputs + self.mems['basal']) ) / self.tau_basal
        self.mems['apical'] =  ((apical_inputs + self.mems['apical'])) / self.tau_apical
        self.mems['soma'] =self.mems['soma'] + ((self.mems['apical']*self.mems['basal'].expand(-1, self.mems['apical'].shape[1], -1))* 0.104+ (self.mems['apical'] + self.mems['basal'].expand(-1, self.mems['apical'].shape[1], -1))*(0.896)- self.mems['soma'])/self.tau
        
    def calc_spike(self):
        self.spike = self.act_fun(self.mems['soma'] - self.threshold)
        self.mems['soma'] = self.mems['soma']  * (1. - self.spike.detach())
        self.mems['basal'] = self.mems['basal'] * (1. - self.spike.detach())
        self.mems['apical'] = self.mems['apical']  * (1. - self.spike.detach())
    
    def save_spikes(self, step,epoch, save_path='./'):
        self.spikes_record.append(self.spike.detach().cpu().numpy())
        file_name = f"spikes_record_step_{step}_episode{epoch}.pt"

        full_path = os.path.join(save_path, file_name)

        torch.save(self.spikes_record, full_path)
        print(f"Spikes saved to {file_name}")

        self.spikes_record = []



class ThreeCompNode_plif3(BaseMCNode):
    def __init__(self,
                 threshold=1.0,
                 tau=2.0,
                 tau_basal=2.0,
                 tau_apical=2.0,
                 v_reset=0.0,
                 comps=['basal', 'apical', 'soma'],
                 dt=0.5,
                 act_fun=AtanGrad):
        g_B = 0.6
        g_L = 0.05
        super().__init__(threshold, v_reset, comps)
        self.tau = tau
        self.tau_basal = tau_basal
        self.tau_apical = tau_apical
        self.act_fun = act_fun(alpha=tau, requires_grad=False)
        init_w = -math.log(tau - 1.)
        self.alph_basal = nn.Parameter(torch.as_tensor(init_w))
        self.alph_apical = nn.Parameter(torch.as_tensor(init_w))
        self.alph_soma = nn.Parameter(torch.as_tensor(init_w))
        self.dt=dt
        self.spikes_record = [] 
    
    def integral(self, basal_inputs, apical_inputs):
        '''
        Params:
            inputs torch.Tensor: Inputs for basal dendrite  
        '''
        self.mems['basal'] =   ((basal_inputs + self.mems['basal']) ) / self.tau_basal
        self.mems['apical'] =  ((apical_inputs + self.mems['apical'])) / self.tau_apical
        self.mems['soma'] =self.mems['soma'] + ((self.mems['apical']*self.mems['basal'].expand(-1, self.mems['apical'].shape[1], -1))* self.alph_soma.sigmoid()+ (self.mems['apical'] + self.mems['basal'].expand(-1, self.mems['apical'].shape[1], -1))*(1-self.alph_soma.sigmoid())- self.mems['soma'])/self.tau
        
    def calc_spike(self):
        self.spike = self.act_fun(self.mems['soma'] - self.threshold)
        self.mems['soma'] = self.mems['soma']  * (1. - self.spike.detach())
        self.mems['basal'] = self.mems['basal'] * (1. - self.spike.detach())
        self.mems['apical'] = self.mems['apical']  * (1. - self.spike.detach())
    
    def save_spikes(self, step,epoch, save_path='./'):
        self.spikes_record.append(self.spike.detach().cpu().numpy())
        file_name = f"spikes_record_step_{step}_episode{epoch}.pt"

        full_path = os.path.join(save_path, file_name)

        torch.save(self.spikes_record, full_path)
        print(f"Spikes saved to {file_name}")

        self.spikes_record = []

class ThreeCompNode_plif3_mouse(BaseMCNode):
    def __init__(self,
                 threshold=1.0,
                 tau=2.0,
                 tau_basal=2.0,
                 tau_apical=2.0,
                 v_reset=0.0,
                 comps=['basal', 'apical', 'soma'],
                 dt=0.5,
                 act_fun=AtanGrad):
        g_B = 0.6
        g_L = 0.05
        super().__init__(threshold, v_reset, comps)
        self.tau = tau
        self.tau_basal = tau_basal
        self.tau_apical = tau_apical
        self.act_fun = act_fun(alpha=tau, requires_grad=False)
        init_w = -math.log(tau - 1.)
        self.alph_basal = nn.Parameter(torch.as_tensor(init_w))
        self.alph_apical = nn.Parameter(torch.as_tensor(init_w))
        self.alph_soma = nn.Parameter(torch.as_tensor(init_w))
        self.dt=dt
        self.spikes_record = [] 
    
    def integral(self, basal_inputs, apical_inputs):
        '''
        Params:
            inputs torch.Tensor: Inputs for basal dendrite  
        '''
        self.mems['basal'] =   ((basal_inputs + self.mems['basal']) ) / self.tau_basal
        self.mems['apical'] =  ((apical_inputs + self.mems['apical'])) / self.tau_apical
        self.mems['soma'] =self.mems['soma'] + ((self.mems['apical']*self.mems['basal'].expand(-1, self.mems['apical'].shape[1], -1))* 0.729+ (self.mems['apical'] + self.mems['basal'].expand(-1, self.mems['apical'].shape[1], -1))*(0.271)- self.mems['soma'])/self.tau

    def calc_spike(self):
        self.spike = self.act_fun(self.mems['soma'] - self.threshold)
        self.mems['soma'] = self.mems['soma']  * (1. - self.spike.detach())
        self.mems['basal'] = self.mems['basal'] * (1. - self.spike.detach())
        self.mems['apical'] = self.mems['apical']  * (1. - self.spike.detach())
    
    def save_spikes(self, step,epoch, save_path='./'):
        self.spikes_record.append(self.spike.detach().cpu().numpy())
        file_name = f"spikes_record_step_{step}_episode{epoch}.pt"
        full_path = os.path.join(save_path, file_name)

        torch.save(self.spikes_record, full_path)
        print(f"Spikes saved to {file_name}")

        self.spikes_record = []








