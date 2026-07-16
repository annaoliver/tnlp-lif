#ee
#mouse cho normal


import gc
from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA

import re
plt.switch_backend('agg')

regx=re.compile(r'\d+.\d+')


def get_population(name,N,tau_rp,g_m,C_m,g_AMPA_ext,g_GABA,g_AMPA_rec,g_NMDA,I_background,lamda_NA,tau_GABA,ampa_scale,nmda_scale,mu_ampa,mu_nmda,mu_gaba,gaba,dend_inh):
    """Get population of neurons

       name -- name of population
       tau_rp -- refractory period
       g_m -- diandao of neurons
       C_m -- capacitance of neurons
       """
    V_thr = -50. * mV
    V_reset = -55. * mV
    V_L = -70. * mV
    V_I = -70. * mV
    
    Mg2 = 1.
    c1=20 * pA
    c2=13.624 *pA
    c3=7.0
    c4=0 *pA
    c5=9.64 *pA
    c6=20 *pA
    c7=1*pA
    neuron_spacing = 50*umetre
    width = N/4.0*neuron_spacing
    neurons = NeuronGroup(
        N,
        """
        g_m :siemens
        C_m :farad
        V_thr :volt
        V_reset :volt
        V_L :volt
        tau_GABA :second
        V_I :volt
        g_GABA :siemens
        s_NMDA_tot:1
        lamda_NA:1
        g_AMPA_rec:siemens
        g_NMDA:siemens
        Mg2 :1
        I_background:amp
        c1:amp
        c2:amp
        c3:1
        c4:amp
        c5:amp
        c6:amp
        c7:amp
        ampa_scale:1
        nmda_scale:1
        mu_ampa:1
        mu_nmda:1
        mu_gaba:1
        gaba :1
        dend_inh :1
        

        
        g_AMPA_ext: siemens
        dv / dt = (- g_m * (v - V_L) - I_syn) / C_m : volt (unless refractory)
        I_syn = I_AMPA_ext + I_AMPA_rec*ampa_scale + I_NMDA_rec*nmda_scale + gaba*I_GABA_rec+I_soma_dend +I_background: amp
        I_AMPA_ext = g_AMPA_ext * (v - 0. * mV) * s_AMPA_ext : amp
        ds_AMPA_ext / dt = - s_AMPA_ext / ( 2. * ms) : 1
        I_GABA_rec =mu_gaba*g_GABA * (v - V_I) * s_GABA : amp
        ds_GABA / dt = - s_GABA / tau_GABA : 1 
        I_AMPA_rec =mu_ampa*lamda_NA* g_AMPA_rec * (v - 0. * mV) *s_AMPA : amp
        ds_AMPA / dt = - s_AMPA / (2. * ms): 1 
        I_NMDA_rec = mu_nmda*lamda_NA*g_NMDA * (v - 0. * mV) / (1 + Mg2 * exp(-0.062 * v / mV) / 3.57) * s_NMDA_tot : amp
        
        I_dend_inh=I_GABA_rec:amp
        I_dend_exc=I_AMPA_ext + I_AMPA_rec + I_NMDA_rec +I_background :amp
        num=-I_dend_inh/c6*dend_inh :1
        mid=(I_dend_exc +c3*I_dend_inh + c4)/c5*exp(num):1
        I_soma_dend=c1*tanh(mid) + c2 :amp

        """,
        threshold="v>V_thr",
        reset="v=V_reset",
        refractory=tau_rp,
        method='euler',
        name=name,
    )
    neurons.V_thr=V_thr
    neurons.V_reset = V_reset
    neurons.V_L = V_L
    neurons.g_m = g_m
    neurons.C_m = C_m
    neurons.g_AMPA_ext = g_AMPA_ext
    neurons.V_I = V_I
    neurons.tau_GABA = tau_GABA
    neurons.g_GABA = g_GABA
    neurons.g_AMPA_rec=g_AMPA_rec
    neurons.g_NMDA=g_NMDA
    neurons.lamda_NA=lamda_NA
    neurons.Mg2 = Mg2
    neurons.c1=c1
    neurons.c2=c2
    neurons.c3=c3
    neurons.c4=c4
    neurons.c5=c5
    neurons.c6=c6
    neurons.c7 = c7
    neurons.I_background=I_background
    neurons.ampa_scale=ampa_scale
    neurons.nmda_scale=nmda_scale
    neurons.mu_ampa=mu_ampa
    neurons.mu_nmda=mu_nmda
    neurons.mu_gaba=mu_gaba
    neurons.gaba=gaba
    neurons.dend_inh=dend_inh




    return neurons




def get_synapses(name, source, target,weight, eqs,g_AMPA_rec,g_NMDA,alpha_ampa,alpha_nmda,tau_NMDA_decay,tau_facil=None):
    """Construct connections and retrieve synapses

    name -- name of synapses
    source -- source of connections
    target -- target of connections
    weight -- weight matrix

    """


    synapses_eqs = """
    tau_NMDA_decay :second
    U :1
    tau_u :second
    tau_D :second
    alpha_ampa:1
    alpha_nmda:1
    

    
    dx / dt = (1- x) / tau_D : 1 (clock-driven)
    du / dt=(U-u)/tau_u :1 (clock-driven)    
    
    ds_NMDA / dt = - s_NMDA / tau_NMDA_decay: 1 (clock-driven)
    
    
    
    
    """.format(
        name
    )

    synapses_eqs+=eqs

    if tau_facil:

        synapses_action = """
        x += -u*x
        u += U*(1-u) 
        s_AMPA += x*u*alpha_ampa
        s_NMDA+= x*u*(1- s_NMDA)*alpha_nmda
        """
    else:
        synapses_action = """
        s_GABA +=1
        """

    synapses = Synapses(
        source,
        target,
        model=synapses_eqs,
        on_pre=synapses_action,
        method='euler',
        name=name,
    )

    
    alpha = 0.5 / ms
    # alpha_ampa=2.5*5
    # alpha_nmda=2.5*5

    U = 0.15
    tau_u = 1500 * ms
    tau_D = 2 * ms

    synapses.connect()


    synapses.tau_NMDA_decay = tau_NMDA_decay


    synapses.U = U
    synapses.tau_u = tau_u
    synapses.tau_D = tau_D

    synapses.w=weight
    synapses.x=1
    synapses.u=0

    synapses.alpha_ampa=alpha_ampa
    synapses.alpha_nmda=alpha_nmda

    return synapses

def weight_balance(w):
    raw=w.shape[0]
    col=w.shape[1]

    for num in range(col):
        sum_excitation=np.sum(w[:,num])
        w[:, num]+=-sum_excitation/raw
    return w


def input(num,t,P_E,P_I,sub_E,sub_I):
    f = 0.1
    C_ext = 900
    C_selection = int(f * C_ext)
    rate_selection = 50 * Hz
    stimuli1 = TimedArray(np.r_[np.zeros(40), np.ones(20), np.zeros(400)], dt=25 * ms)
    input_e = PoissonInput(P_E[:sub_E*num], 's_AMPA_ext', C_selection, rate_selection, 'stimuli1(t)')
    input_i=PoissonInput(P_I[:sub_I*num], 's_AMPA_ext', C_selection, rate_selection, 'stimuli1(t)')

    stimuli_reset = TimedArray(np.r_[np.zeros(int(40*t)), np.ones(2), np.zeros(400)], dt=25 * ms)
    input_reset_I = PoissonInput(P_E[:sub_E*num], 's_AMPA_ext', C_selection, rate_selection, 'stimuli_reset(t)')
    input_reset_E = PoissonInput(P_I[:sub_I*num], 's_AMPA_ext', C_selection, rate_selection, 'stimuli_reset(t)')


    return input_e,input_i,input_reset_E,input_reset_I,stimuli1,stimuli_reset

def get_0_1_array(w,p):
    '''按照数组模板生成对应的 0-1 矩阵，默认rate=0.2'''
    raw=w.shape[0]
    col=w.shape[1]
    array=np.ones((raw,col))
    rate=1-p
    zeros_num = int(array.size * rate)#根据0的比率来得到 0的个数
    new_array = np.ones(array.size)#生成与原来模板相同的矩阵，全为1
    new_array[:zeros_num] = 0 #将一部分换为0
    np.random.shuffle(new_array)#将0和1的顺序打乱
    re_array = new_array.reshape(array.shape)#重新定义矩阵的维度，与模板相同
    out=w*re_array
    return out

#将neuron的输出方式转换为放电序列时间矩阵
def change_matrix(spike_train,bin,t_start,t_end,N):
    t_delay=t_end-t_start
    R=np.zeros((t_delay,N))
    print(spike_train.t)
    for k in range(spike_train.t.shape[0]):
        t_=int(spike_train.t[k]/ms)
        n_=spike_train.i[k]
        print(t_)
        R[t_][n_]=1
    P=[]
    for k in range(int(t_delay/bin)):
        R_=R[k*bin:(k+1)*bin]
        R_=R_.sum(0)
        P.append(R_)
    return P

#计算神经元的皮尔逊相关系数；计算拟合后各个神经元衰减曲线的时间尺度
def time_scale(P,bin,t_start,t_end,N):
    t_delay=t_end-t_start
    count=int(t_delay/bin)

    ac=[]
    t=[]
    s_=int(t_start/bin)
    e_=int(t_end/bin)
    P=P[s_:e_]
    count=0
    for k in P:
        ac_=pearsonr(P[0],k)
        ac.append(ac_[0]) #相关系数
        t.append(count) #间隔时间
        count=count+1
    def func(t, tau):
        return np.exp(-t / tau)
    ac=np.nan_to_num(ac)
    popt1, pcov1 = curve_fit(func, t, ac, maxfev=20000, method='dogbox')
    tau=popt1[0]
    return ac,t,tau


def stable_spatial(P):
    pca=PCA(n_components=2)
    pca.fit(P)
    P_pca=pca.transform(P)
    var=pca.explained_variance_ratio_
    return P_pca,var

def rand_seq(mean,std,n):
    x=[]
    for i in range(n):
        a=np.random.normal(loc=mean, scale=std, size=None)
        x.append(a)
    x=np.array(x)
    return x


def main_work():
    # populations
    N = 1000
    
    N_I =int(0.3371*N)  #human E/I
    N_E = N-N_I
    sub =1  # number of input pool
    sub_E = int(N_E / sub)  # neuron number in input pool
    sub_I=int(N_I / sub)

    # membrane capacitance

    # Rin_e1=45* 1e6
    # Rin_e2=87* 1e6
    # Rin_i1=87* 1e6
    # Rin_i2=111* 1e6
    # tau_me1=15/1000
    # tau_me2=19.4/1000
    # tau_mi1=5.7/1000
    # tau_mi2=7.1/1000
    # c_e1=tau_me1/Rin_e1* 1e12 
    # c_e2=tau_me2/Rin_e2* 1e12
    # spiny1=np.full(int(N_E/2), c_e1)
    # spiny2=np.full(int(N_E/2), c_e2)
    # c_e = np.concatenate([spiny1, spiny2])
    # C_m_E=c_e*pF
    

    # c_i1=tau_mi1/Rin_i1* 1e12 
    # c_i2=tau_mi2/Rin_i2* 1e12
    # aspiny1=np.full(int(N_I/2), c_i1)
    # aspiny2=np.full(int(N_I/2), c_i2)
    # c_i=np.concatenate([aspiny1,aspiny2])
    # C_m_I=c_i*pF
    # membrane capacitance
    # C_m_E =rand_seq(164.96,59.11,N_E) * nF
    # fs=rand_seq( 59.58,10.59,int(N_I/3))
    # bt=rand_seq(79.36,14.83,int(N_I/3))
    # mc=rand_seq(81.12,28.96,int(N_I/3))
    # c_i=fs+bt+mc
    # C_m_I = c_i * nF
    

    # membrane leak
    # g_e1=1/Rin_e1 *1e9
    # g_e2=1/Rin_e2 *1e9 
    # g_e=np.concatenate([np.full(int(N_E/2), g_e1),np.full(int(N_E/2), g_e2)])
    # g_m_E=g_e* nS
    
    # g_i1=1/Rin_i1 *1e9
    # g_i2=1/Rin_i2 *1e9 
    # g_i=np.concatenate([np.full(int(N_I/2), g_i1),np.full(int(N_I/2), g_i2)])
    # g_m_I = g_i * nS

    # membrane capacitance

    Rin_e1=45* 1e6
    Rin_e2=87* 1e6
    Rin_i1=87* 1e6
    Rin_i2=111* 1e6
    tau_me1=15/1000
    tau_me2=19.4/1000
    tau_mi1=5.7/1000
    tau_mi2=7.1/1000
    c_e1=tau_me1/Rin_e1* 1e12 
    c_e2=tau_me2/Rin_e2* 1e12
    spiny1=np.full(int(N_E/2), c_e1)
    spiny2=np.full((N_E-int(N_E/2)), c_e2)
    c_e = np.concatenate([spiny1, spiny2])
    C_m_E=c_e*pF
    

    c_i1=tau_mi1/Rin_i1* 1e12 
    c_i2=tau_mi2/Rin_i2* 1e12
    aspiny1=np.full(int(N_I/2), c_i1)
    aspiny2=np.full((N_I-int(N_I/2)), c_i2)
    c_i=np.concatenate([aspiny1,aspiny2])
    C_m_I=c_i*pF
    

    # membrane leak
    g_e1=1/Rin_e1 *1e9
    g_e2=1/Rin_e2 *1e9 
    g_e=np.concatenate([np.full(int(N_E/2), g_e1),np.full((N_E-int(N_E/2)), g_e2)])
    g_m_E=g_e* nS
    
    g_i1=1/Rin_i1 *1e9
    g_i2=1/Rin_i2 *1e9 
    g_i=np.concatenate([np.full(int(N_I/2), g_i1),np.full((N_I-int(N_I/2)), g_i2)])
    g_m_I = g_i * nS
    # refractory period
    tau_rp_E = 2. * ms
    tau_rp_I = 1. * ms

    # external stimuli
    rate = 3 * Hz
    C_ext = 800


    # AMPA (excitatory)
    g_AMPA_ext_E = 2.08 * nS
    g_AMPA_rec_E = 0.104 * nS 
    g_AMPA_ext_I = 1.62 * nS
    g_AMPA_rec_I = 0.081 * nS 

    # NMDA (excitatory)
    g_NMDA_E = 0.327 * nS 
    g_NMDA_I = 0.258 * nS 

    # GABAergic (inhibitory)
    g_GABA_E = 1.25 * nS 
    g_GABA_I = 0.973 * nS 

    # subpopulations
    f = 0.1
    
    

    #background current
    I_background_e=310 *pA
    I_background_i = 300 * pA

    
    tau_GABA = 10. * ms
    tau_NMDA_decay = 100. * ms
    alph_ampa=[]
    alph_nmda=[]

    nm_pre_swi=197
    nm_pos_swi=425
    alph_nmda.append(nm_pos_swi/nm_pre_swi)
    am_pre_swi=58.6
    am_pos_swi=98.8
    alph_ampa.append(am_pos_swi/am_pre_swi)

    nm_pre_yue=127
    nm_pos_yue=319
    alph_nmda.append(nm_pos_yue/nm_pre_yue)
    am_pre_yue=52.5
    am_pos_yue=115
    alph_ampa.append(am_pos_yue/am_pre_yue)

    nm_pre_ping=154.5
    nm_pos_ping=385.6
    alph_nmda.append(nm_pos_ping/nm_pre_ping)
    am_pre_ping=53.4
    am_pos_ping=99
    alph_ampa.append(am_pos_ping/am_pre_ping)

    nm_pre_piz=168
    nm_pos_piz=361
    alph_nmda.append(nm_pos_piz/nm_pre_piz)
    am_pre_piz=65
    am_pos_piz=141
    alph_ampa.append(am_pos_piz/am_pre_piz)




    # para=np.load('./ampa-nmda/ampa-nmda.npy')
    alpha_ampa_record=[]
    alpha_nmda_record=[]
    Wie_self_record=[]
    Wii_other_record=[]
    ampa_scale_record=[]
    nmda_scale_record=[]
    e_spike=[]
    t_e_spike=[]
    i_spike=[]
    t_i_spike=[]

    ac_e_pre_record=[]
    t_e_pre_record=[]
    tau_e_pre_record=[]
    ac_i_pre_record=[]
    t_i_pre_record=[]
    tau_i_pre_record=[]
    ac_e_post_record=[]
    t_e_post_recoed=[]
    tau_e_post_record=[]
    ac_i_post_record=[]
    t_i_post_recoed=[]
    tau_i_post_record=[]
    para_record=[]

    
    # minist=np.load('./binary.npz')
    # binary_sequence=minist['binary_sequence']

    ee_=[1,2,3,4,5,6,7,8,9,10]

    for echo in range(5):
        for para_ee in ee_:
            
            
            remeber_num=4
            interval=0
            W_positive=para_ee
            
            p_ii=0.3
            p_ie=0.8
            p_ee=0.3
            p_ei=0.8


            
            
            lamda_NA=1

            # ampa_scale=alph_ampa[two]
            # nmda_scale=alph_nmda[two]
            ampa_scale=1
            nmda_scale=1

            fac=0

            spine_dense=0.5279
            EXN_IN=1-0.5258

            mu_ampa_e=0.5279*(1-EXN_IN)*(1-fac)
            mu_nmda_e=0.5279*(1-EXN_IN)*(1-fac)
            mu_gaba_e=0.5279*EXN_IN

            mu_ampa_i=0
            mu_nmda_i=0
            mu_gaba_i=1

            alpha_ampa=10
            alpha_nmda=10


            # #I_gaba charactor
            # gaba_e=1
            # gaba_i=1
            # #I_dend_inh charactor
            # dend_inh_e=1
            # dend_inh_i=1

            
            gaba_e=0.6
            gaba_i=1
            #I_dend_inh charactor
            dend_inh_e=1
            dend_inh_i=1



            P_E = get_population("P_E", N_E, tau_rp_E, g_m_E, C_m_E,g_AMPA_ext_E,g_GABA_E,g_AMPA_rec_E,g_NMDA_E,I_background_e,lamda_NA,tau_GABA,ampa_scale,nmda_scale,mu_ampa_e,mu_nmda_e,mu_gaba_e,gaba_e,dend_inh_e)
            P_I = get_population("P_I", N_I, tau_rp_I, g_m_I, C_m_I,g_AMPA_ext_I,g_GABA_I,g_AMPA_rec_I,g_NMDA_I,I_background_i,lamda_NA,tau_GABA,ampa_scale,nmda_scale,mu_ampa_i,mu_nmda_i,mu_gaba_i,gaba_i,dend_inh_i)

            

            
            
            # average field weight

            Wei_self = 19.948639
            Wei_other = 17.144662
            Wie_self = 17
            Wie_other = 13.155368
            Wii_self = 18.437482
            Wii_other = 0




            # compute weight
            Matrix_weight_E_E = np.zeros((N_E, N_E))
            Matrix_weight_E_I = np.zeros((N_E, N_I))
            Matrix_weight_I_E = np.zeros((N_I, N_E))
            Matrix_weight_I_I = np.zeros((N_I, N_I))

            Matrix_weight_E_I[:] = Wei_other
            Matrix_weight_I_E[:] = Wie_other
            Matrix_weight_I_I[:] = Wii_other


            for num in range(sub):
                Matrix_weight_E_E[num * sub_E:(num + 1) * sub_E, num * sub_E:(num + 1) * sub_E] = W_positive
                Matrix_weight_E_I[num * sub_E:(num + 1) * sub_E,num * sub_I:(num + 1) * sub_I]=Wei_self
                Matrix_weight_I_E[num * sub_I:(num + 1) * sub_I,num * sub_E:(num + 1) * sub_E]=Wie_self
                Matrix_weight_I_I[num * sub_I:(num + 1) * sub_I,num * sub_I:(num + 1) * sub_I]=Wii_self

                
            

            

            Matrix_weight_E_E=weight_balance(Matrix_weight_E_E)
            Matrix_weight_E_I = weight_balance(Matrix_weight_E_I)
            Matrix_weight_I_E = weight_balance(Matrix_weight_I_E)
            Matrix_weight_I_I=weight_balance(Matrix_weight_I_I)

            Matrix_weight_E_E=get_0_1_array(Matrix_weight_E_E,p_ee)
            Matrix_weight_E_I=get_0_1_array(Matrix_weight_E_I,p_ei)
            Matrix_weight_I_E=get_0_1_array(Matrix_weight_I_E,p_ie)
            Matrix_weight_I_I=get_0_1_array(Matrix_weight_I_I,p_ii)



            Matrix_weight_E_E2 = Matrix_weight_E_E.flatten()
            Matrix_weight_E_I2 = Matrix_weight_E_I.flatten()
            Matrix_weight_I_E2 = Matrix_weight_I_E.flatten()
            Matrix_weight_I_I2 = Matrix_weight_I_I.flatten()

            tau_rec = 810 * ms
            # E to E
            eqs='''
            s_NMDA_tot_post=w*s_NMDA :1(summed)
            
            w : 1  
            '''
            C_E_E = get_synapses(
                "C_E_E",
                P_E,
                P_E,
                Matrix_weight_E_E2,
                eqs,
                g_AMPA_rec_E,
                g_NMDA_E,
                alpha_ampa,
                alpha_nmda,
                tau_NMDA_decay,
                tau_rec,
            )

            # E to I
            eqs = '''
                s_NMDA_tot_post=w*s_NMDA :1(summed)
                w : 1  
                '''
            C_E_I = get_synapses(
                "C_E_I",
                P_E,
                P_I,
                Matrix_weight_E_I2,
                eqs,
                g_AMPA_rec_E,
                g_NMDA_E,
                alpha_ampa,
                alpha_nmda,
                tau_NMDA_decay,
                tau_rec,
            )

            # I to I
            neuron_spacing = 50*umetre
            ii=2
            width = N/4.0*neuron_spacing
            eqs = '''
                
                w : 1  
                '''
            C_I_I =C_I_E = get_synapses(
                "C_I_I",
                P_I,
                P_I,
                Matrix_weight_I_I2,
                eqs,
                g_AMPA_rec_I,
                g_NMDA_I,
                alpha_ampa,
                alpha_nmda,
                tau_NMDA_decay,
            )
            # C_I_I.w = ii
            # I to E
            eqs = '''
                
                w : 1  
                '''
            C_I_E = get_synapses(
                "C_I_E",
                P_I,
                P_E,
                Matrix_weight_I_E2,
                eqs,
                g_AMPA_rec_I,
                g_NMDA_I,
                alpha_ampa,
                alpha_nmda,
                tau_NMDA_decay,
            )


            # external noise
            C_P_E = PoissonInput(P_E, 's_AMPA_ext', C_ext, rate, '1')
            C_P_I = PoissonInput(P_I, 's_AMPA_ext', C_ext, rate, '1')

            #每个记忆在不同的时间触发
            
            #每个记忆在不同的时间触发
            
            
            f = 0.1
            C_ext = 800
            C_selection = int(f * C_ext)
            rate_selection = 25 * Hz
            
            # binary_data =binary_sequence[img_]

            # # 获取不为0的值的索引
            # non_zero_indices = np.where(binary_data != 0)

            # # non_zero_indices是一个包含不为0值的索引的元组
            # # 如果需要将其转换为一维数组，可以使用ravel()函数
            # selected_indices = non_zero_indices[0]
            
            # # 创建 SpikeGeneratorGroup 用于单独的泊松输入
            
            # times=int(400/(1/rate_selection*1000))
            # inter=1/rate_selection
            # spike_times=[]
            # for k in range(times+1):
            #     spike_times.append(1* second+k*inter)
            

            # repeated_array = [x for x in selected_indices for _ in range(len(spike_times))]  # 每个元素重复100次
            # repeated_array_=spike_times*len(selected_indices)
            
            
            # input_rates = [rate_selection if i in selected_indices else 0 * Hz for i in range(N_E)]
            # input_neurons = SpikeGeneratorGroup(N_E, repeated_array,repeated_array_)

            # # 连接泊松输入到神经元群
            # synapses = Synapses(input_neurons, P_E, on_pre='s_AMPA_ext +=1')
            # synapses.connect(i=selected_indices, j=selected_indices)
            
            # P_E_selected = []
            # for i in range(N_E):
            #     if i in selected_indices:
            #         P_E_selected.append(P_E[i])
            
            # 为选定神经元创建一个子集
            
                        
            # stimuli = TimedArray(np.r_[np.zeros(int(40)), np.ones(32), np.zeros(400)], dt=25 * ms)
            # input_e = PoissonInput(P_E_selected, 's_AMPA_ext', C_selection, rate_selection, 'stimuli(t)')
            #input_i=PoissonInput(P_I, 's_AMPA_ext', C_selection, rate_selection, 'stimuli(t)')
                
            


            # 定义刺激信号序列
            stimulus = np.r_[ 
                np.zeros(40),        # 从0秒到1秒，没有刺激（0）
                np.ones(16),         # 从1秒开始，持续400毫秒刺激（1）
                np.zeros(40),        # 刺激后，1秒间隔没有刺激（0）
                np.ones(2),          # 1秒间隔后，5毫秒的刺激（1）
                np.zeros(100)       # 之后没有刺激，直到仿真结束（0），这里1000是一个示例长度，具体可以根据仿真时间调整
            ]

            stimuli_reset = TimedArray(stimulus, dt=25 * ms)
            input_reset_E = PoissonInput(P_E, 's_AMPA_ext', C_selection, rate_selection, 'stimuli_reset(t)')
            input_reset_I = PoissonInput(P_I, 's_AMPA_ext', C_selection, rate_selection, 'stimuli_reset(t)')
            #input_reset_E = PoissonInput(P_I[:N_I], 's_AMPA_ext', C_selection, rate_selection, 'stimuli_reset(t)')
            
            #input_e,input_i,input_reset_E,input_reset_I,stimuli1,stimuli_reset=input(remeber_num,cue,P_E,P_I,sub_E,sub_I)

            
            
            e_neuron_all=SpikeMonitor(P_E)
            i_neuron_all=SpikeMonitor(P_I)

            
            r_E = PopulationRateMonitor(P_E)
            r_I = PopulationRateMonitor(P_I)
            monitor = StateMonitor(P_E, ['I_soma_dend', 'I_GABA_rec', 'I_NMDA_rec', 'I_AMPA_rec'], record=True)
            net = Network(collect())
            net.add(monitor)
            net.active=True
            net.run(5* second, report='stdout')

            I_soma_dend_values = monitor.I_soma_dend  # shape: (neurons_num, timepoints)
            I_GABA_rec_values = monitor.I_GABA_rec
            I_NMDA_rec_values = monitor.I_NMDA_rec
            I_AMPA_rec_values = monitor.I_AMPA_rec

            e_neuron = np.array(e_neuron_all.i)
            i_neuron = np.array(i_neuron_all.i)
            t_e = np.array(e_neuron_all.t)
            t_i = np.array(i_neuron_all.t)

            e_ = 200
            i_ = 40
            i_index = []
            i_t_index = []
            e_index = []
            e_t_index = []

            for k in range(i_neuron.shape[0]):
                group = int(i_neuron[k] / sub_I)
                if (i_neuron[k] < i_ + group * sub_I) and (i_neuron[k] > group * sub_I):
                    ind = i_neuron[k] - sub_I * group
                    index_ = ind + group * (e_ + i_) + e_
                    i_index.append(index_)
                    i_t_index.append(t_i[k])

            for k in range(e_neuron.shape[0]):
                group = int(e_neuron[k] / sub_E)
                if (e_neuron[k] < e_ + group * sub_E) and (e_neuron[k] > group * sub_E):
                    ind = e_neuron[k] - sub_E * group
                    index_ = ind + group * (e_ + i_)
                    e_index.append(index_)
                    e_t_index.append(t_e[k])
            size=5
            
            
            plt.scatter(i_t_index,i_index,c="#676FA3",s=size)
            plt.scatter(e_t_index,e_index,c="#FF5959",s=size)
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.title("Scatter Plot of two different datasets")
            plt.savefig("./human/ee/uncho/%f-%d.jpg"%(para_ee,echo))
            plt.close()
            array_dict={'e_neuron':e_neuron,
                        'i_neuron':i_neuron,
                        't_e':t_e,
                        't_i':t_i,
                        'I_soma_dend':I_soma_dend_values,
                        'I_GABA_rec':I_GABA_rec_values,
                        'I_NMDA_rec':I_NMDA_rec_values,
                        'I_AMPA_rec':I_AMPA_rec_values}

            np.savez('./human/ee/uncho/%f-%d.npz'%(para_ee,echo),**array_dict)

                
            

            net.active=False
            net.stop()

            
            
            
            
                
            

            del net
            del P_E
            del P_I
            del input_reset_I
            
            gc.collect()
    

            




    
main_work()