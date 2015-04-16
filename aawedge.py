'''
===================
aawedge.py
===================

Functions to build and plot seismic wedges.

Created April 2015 by Alessandro Amato del Monte (alessandro.adm@gmail.com)

Heavily inspired by Matt Hall and Evan Bianco's blog posts and code:

http://nbviewer.ipython.org/github/agile-geoscience/notebooks/blob/master/To_make_a_wedge.ipynb
http://nbviewer.ipython.org/github/kwinkunks/notebooks/blob/master/Spectral_wedge.ipynb
http://nbviewer.ipython.org/github/kwinkunks/notebooks/blob/master/Faster_wedges.ipynb
http://nbviewer.ipython.org/github/kwinkunks/notebooks/blob/master/Variable_wedge.ipynb

Also see Wes Hamlyn's tutorial on Leading Edge "Thin Beds, tuning and AVO" (December 2014):

https://github.com/seg/tutorials/tree/master/1412_Tuning_and_AVO

HISTORY
2015-04-10 first public release.
'''

import numpy as np
import matplotlib.pyplot as plt
import agilegeo


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def make_wedge(n_traces,layer_1_thickness,min_thickness,max_thickness,zz=0.1):
    '''
    Creates wedge-shaped model made of 3 units with variable thickness.

    INPUT
    n_traces
    layer_1_thickness
    min_thickness
    max_thickness
    zz: vertical sample rate, by default 0.1 m

    OUTPUT
    wedge: 2D numpy array containing wedge-shaped model made of 3 units
    '''

    padding_h   = 10
    padding_v   = 20*(1./zz)
    layer_1_thickness *= (1./zz)
    min_thickness *= (1./zz)
    max_thickness *= (1./zz)
    left_wedge  = padding_h
    right_wedge = n_traces-padding_h
    dz=float(max_thickness-min_thickness)/float(right_wedge-left_wedge)
    n_samples=max_thickness+padding_v+layer_1_thickness
    top_wedge=layer_1_thickness
    wedge = np.zeros((n_samples, n_traces))
    wedge[0:top_wedge,:]=1
    wedge[top_wedge:,:]=3
    wedge[top_wedge:top_wedge+min_thickness,left_wedge:right_wedge]=2
    for i,gg in enumerate(np.arange(left_wedge,right_wedge)):
        wedge[top_wedge+min_thickness:top_wedge+min_thickness+int(round(dz*i)),gg]=2
    print "wedge length: %.2f m" % (right_wedge-left_wedge)
    print "wedge minimum thickness: %.2f m" % (min_thickness*zz)
    print "wedge maximum thickness: %.2f m" % (max_thickness*zz)
    print "wedge vertical sampling: %.2f m" % (zz)
    print "wedge samples, traces: %dx%d" % (n_samples, n_traces)
    return wedge

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def assign_ai(model, aiprop):
    '''
    Assigns acoustic impedance to a rock model created with make_wedge.

    INPUT
    model: 2D numpy array containing values from 1 to 3
    aiprop: np.array([[vp1,rho1],[vp2,rho2],[vp3,rho3]])

    OUTPUT
    model_ai: 2D numpy array containing acoustic impedances
    '''
    model_ai=np.zeros(model.shape)
    code = 1
    for x in aiprop:
        model_ai[model==code] = x[0]*x[1]
        code += 1
    return model_ai

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def assign_vel(model, aiprop):
    '''
    Assigns velocity to a rock model created with make_wedge,
    to be used for depth-time conversion.

    INPUT
    model: 2D numpy array containing values from 1 to 3
    aiprop: np.array([[vp1,rho1],[vp2,rho2],[vp3,rho3]])

    OUTPUT
    model_vel: 2D numpy array containing velocities
    '''
    model_vel=np.zeros(model.shape)
    code=1
    for x in aiprop:
        model_vel[model==code] = x[0]
        code += 1
    return model_vel

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def assign_el(model, elprop):
    '''
    Assigns elastic properties (Vp, Vs, rho) to a rock model created with make_wedge.

    INPUT
    model: 2D numpy array containing values from 1 to 3
    elprop: np.array([[vp1,rho1,vs1],[vp2,rho2,vs2],[vp3,rho3,vs3]])

    OUTPUT
    model_vp: 2D numpy array containing Vp
    model_vs: 2D numpy array containing Vs
    model_rho: 2D numpy array containing densities
    '''
    model_vp=np.zeros(model.shape)
    model_vs=np.zeros(model.shape)
    model_rho=np.zeros(model.shape)
    code = 1
    for i in elprop:
        model_vp[model==code]  = i[0]
        model_vs[model==code]  = i[2]
        model_rho[model==code] = i[1]
        code += 1
    return model_vp,model_vs,model_rho

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def make_rc(model_ai):
    '''
    Computes reflectivities of an acoustic model created with make_wedge + assign_ai.

    INPUT
    model: 2D numpy array containing acoustic impedances

    OUTPUT
    rc: 2D numpy array containing reflectivities
    '''
    upper = model_ai[:-1][:][:]
    lower = model_ai[1:][:][:]
    rc=(lower - upper) / (lower + upper)
    if model_ai.ndim==1:
        rc=np.concatenate((rc,[0]))
    else:
        n_traces=model_ai.shape[1]
        rc=np.concatenate((rc,np.zeros((1,n_traces))))  # add 1 row of zeros at the end
    return rc

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def make_rc_elastic(model_vp,model_vs,model_rho,ang):
    '''
    Computes angle-dependent reflectivities of an elastic model created with make_wedge + assign_el.
    Uses Aki-Richards approximation.

    INPUT
    model_vp: 2D numpy array containing Vp values
    model_vs: 2D numpy array containing Vs values
    model_rho: 2D numpy array containing density values
    ang: list with near, mid, far angle, e.g. ang=[5,20,40]

    OUTPUT
    rc_near: 2D numpy array containing near-stack reflectivities
    rc_mid: 2D numpy array containing mid-stack reflectivities
    rc_far: 2D numpy array containing far-stack reflectivities
    '''
    from agilegeo.avo import akirichards
    [n_samples, n_traces] = model_vp.shape
    rc_near=np.zeros((n_samples,n_traces))
    rc_mid=np.zeros((n_samples,n_traces))
    rc_far=np.zeros((n_samples,n_traces))
    uvp  = model_vp[:-1][:][:]
    lvp  = model_vp[1:][:][:]
    uvs  = model_vs[:-1][:][:]
    lvs  = model_vs[1:][:][:]
    urho = model_rho[:-1][:][:]
    lrho = model_rho[1:][:][:]
    rc_near=akirichards(uvp,uvs,urho,lvp,lvs,lrho,ang[0])
    rc_mid=akirichards(uvp,uvs,urho,lvp,lvs,lrho,ang[1])
    rc_far=akirichards(uvp,uvs,urho,lvp,lvs,lrho,ang[2])
    rc_near=np.concatenate((rc_near,np.zeros((1,n_traces))))  # add 1 row of zeros at the end
    rc_mid=np.concatenate((rc_mid,np.zeros((1,n_traces))))
    rc_far=np.concatenate((rc_far,np.zeros((1,n_traces))))
    return rc_near, rc_mid, rc_far

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def make_synth(rc,wavelet):
    '''
    Convolves reflectivities with wavelet.

    INPUT
    rc: 2D numpy array containing reflectivities
    wavelet

    OUTPUT
    synth: 2D numpy array containing seismic data
    '''
    nt=np.size(wavelet)
    [n_samples, n_traces] = rc.shape
    synth = np.zeros((n_samples+nt-1, n_traces))
    for i in range(n_traces):
        synth[:,i] = np.convolve(rc[:,i], wavelet)
    synth = synth[np.ceil(len(wavelet))/2:-np.ceil(len(wavelet))/2, :]
    synth=np.concatenate((synth,np.zeros((1,n_traces))))
    return synth

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def make_synth_v2(rc,wavelet):
    '''
    Convolves reflectivities with wavelet.
    Alternative version using numpy apply_along_axis,
    slower than np.convolve with for loop.

    INPUT
    rc: 2D numpy array containing reflectivities
    wavelet

    OUTPUT
    synth: 2D numpy array containing seismic data
    '''
    nt=np.size(wavelet)
    [n_samples, n_traces] = rc.shape
    synth=np.zeros((n_samples+nt-1, n_traces))
    synth=np.apply_along_axis(lambda m: np.convolve(m,wavelet),axis=0,arr=rc)
    return synth

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def make_synth_v3(rc,wavelet):
    '''
    Convolves reflectivities with wavelet.
    Alternative version using scipy.ndimage.filters.convolve1d,
    slower than np.convolve with for loop.

    INPUT
    rc: 2D numpy array containing reflectivities
    wavelet

    OUTPUT
    synth: 2D numpy array containing seismic data
    '''
    from scipy.ndimage.filters import convolve1d
    nt=np.size(wavelet)
    [n_samples, n_traces] = rc.shape
    synth=np.zeros((n_samples+nt-1, n_traces))
    synth=convolve1d(rc,wavelet,axis=0)
    return synth

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def forward_model(model,aiprop,wavelet,dz,dt):
    """
    Meta function to do everything from scratch (zero-offset model).
    """
    earth = assign_ai(model, aiprop)
    vels = assign_vel(model, aiprop)
    earth_time=agilegeo.avo.depth_to_time(earth,vels,dz,dt,twt=True)
    rc = make_rc(earth_time)
    return make_synth(rc,wavelet)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def forward_model_elastic(model,elprop,wavelet,ang,dz,dt):
    """
    Meta function to do everything from scratch (angle-dependent models).
    """
    model_vp,model_vs,model_rho = assign_el(model,elprop)
    model_vp_time=agilegeo.avo.depth_to_time(model_vp,model_vp,dz,dt,twt=True)
    model_vs_time=agilegeo.avo.depth_to_time(model_vs,model_vp,dz,dt,twt=True)
    model_rho_time=agilegeo.avo.depth_to_time(model_rho,model_vp,dz,dt,twt=True)

    rc_near, rc_mid, rc_far=make_rc_elastic(model_vp_time,model_vs_time,model_rho_time,ang)
    near = make_synth(rc_near,wavelet)
    mid = make_synth(rc_mid,wavelet)
    far = make_synth(rc_far,wavelet)
    return near,mid,far

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def forward_model_elastic_decay(model,elprop,wav_near,wav_mid,wav_far,dz,dt):
    """
    Meta function to do everything from scratch (angle-dependent models).
    Uses angle-dependent wavelet to simulate frequency decay with offset.
    """
    model_vp,model_vs,model_rho = assign_el(model,elprop)
    model_vp_time=agilegeo.avo.depth_to_time(model_vp,model_vp,dz,dt,twt=True)
    model_vs_time=agilegeo.avo.depth_to_time(model_vs,model_vp,dz,dt,twt=True)
    model_rho_time=agilegeo.avo.depth_to_time(model_rho,model_vp,dz,dt,twt=True)

    rc_near, rc_mid, rc_far=make_rc_elastic(model_vp_time,model_vs_time,model_rho_time,ang)
    near = make_synth(rc_near,wav_near)
    mid = make_synth(rc_mid,wav_mid)
    far = make_synth(rc_far,wav_far)
    return near,mid,far

#------------------------------------------------
# PLOTTING FUNCTIONS
#------------------------------------------------

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_wavelet(wavelet,time):
    '''
    Plots wavelet.
    Required timescale can be calculated with:

    time=np.arange(-duration/2, duration/2 , dt)

    where duration and dt (sample rate) are also given input to calculate wavelet.
    '''
    plt.figure(figsize=(8,5))
    plt.plot(time,wavelet,lw=2,color='black')
    plt.fill_between(time,wavelet,0,wavelet>0.0,interpolate=False,hold=True,color='blue', alpha = 0.5)
    plt.fill_between(time,wavelet,0,wavelet<0.0,interpolate=False,hold=True,color='red', alpha = 0.5)
    plt.grid()
    plt.xlim(-0.1,0.1)
    locs,labels = plt.xticks()
    plt.xticks(locs[:-1], map(lambda x: "%d" % x, locs[:-1]*1000))
    plt.xlabel( 'two-way time (ms)')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_rock_grid(data,zz=1):
    '''
    Plots rock model created with make_wedge.

    INPUT
    data: 2D numpy array containing values from 1 to 3
    zz: vertical sample rate in depth
    '''
    import matplotlib.cm as cm
    cc=cm.get_cmap('copper_r',3)
    plt.figure(figsize=(12,6))
    plt.imshow(data,extent=[0,data.shape[1],data.shape[0]*zz,0],cmap=cc,interpolation='none',aspect='auto')
    cbar=plt.colorbar()
    cbar.set_ticks(range(1,4)); cbar.set_ticklabels(range(1,4))
    plt.grid()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_density(data,zz=1,seismic=True):
    '''
    Density plot of generic 2D numpy array (seismic or any property e.g., velocity).

    INPUT
    data: 2D numpy array containing seismic or elastic property
    zz: vertical sample rate in depth or time
    seismic: True to use red-blue colorscale
    '''
    plt.figure(figsize=(12,6))
    if seismic==True:
        clip=np.amax(abs(data))
        plt.imshow(data,extent=[0,data.shape[1],data.shape[0]*zz,0],cmap='RdBu',vmax=clip,vmin=-clip,aspect='auto')
    else:
        plt.imshow(data,extent=[0,data.shape[1],data.shape[0]*zz,0],cmap='PiYG',aspect='auto')
    plt.colorbar(), plt.grid()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_wiggle(data,zz=1,skip=1,gain=1,alpha=0.5,black=False):
    '''
    Wiggle plot of generic 2D numpy array.

    INPUT
    data: 2D numpy array
    zz: vertical sample rate in depth or time
    skip: interval to choose traces to draw
    gain: multiplier applied to each trace
    '''
    [n_samples,n_traces]=data.shape
    t=range(n_samples)
    plt.figure(figsize=(9.6,6))
    for i in range(0, n_traces,skip):
        trace=gain*data[:,i] / np.max(np.abs(data))
        plt.plot(i+trace,t,color='k', linewidth=0.5)
        if black==False:
            plt.fill_betweenx(t,trace+i,i, where=trace+i>i, facecolor=[0.6,0.6,1.0], linewidth=0)
            plt.fill_betweenx(t,trace+i,i, where=trace+i<i, facecolor=[1.0,0.7,0.7], linewidth=0)
        else:
            plt.fill_betweenx(t,trace+i,i, where=trace+i>i, facecolor='black', linewidth=0, alpha=alpha)
    locs,labels=plt.yticks()
    plt.yticks(locs,[n*zz for n in locs.tolist()])
    plt.grid()
    plt.gca().invert_yaxis()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_partial_stacks(near,mid,far,zz=1,label=''):
    '''
    Density plot of near, mid, far stacks.

    INPUT
    near, mid, far: 2D numpy arrays containing seismic
    zz: vertical sample rate in twt
    label
    '''
    clip=np.amax([abs(near), abs(mid), abs(far)])
    f, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,5))
    im0=ax[0].imshow(near,extent=[0,near.shape[1],near.shape[0]*zz,0],cmap='RdBu',vmax=clip,vmin=-clip,aspect='auto')
    ax[0].set_title(label+' (NEAR)',fontsize='small')
    im1=ax[1].imshow(mid,extent=[0,near.shape[1],near.shape[0]*zz,0],cmap='RdBu',vmax=clip,vmin=-clip,aspect='auto')
    ax[1].set_title(label+' (MID)',fontsize='small')
    im2=ax[2].imshow(far,extent=[0,near.shape[1],near.shape[0]*zz,0],cmap='RdBu',vmax=clip,vmin=-clip,aspect='auto')
    ax[2].set_title(label+' (FAR)',fontsize='small')
    ax[0].set_ylabel('twt [s]')
    cax = f.add_axes([0.925, 0.25, 0.02, 0.5])
    cbar=f.colorbar(im0, cax=cax, orientation='vertical')
    for i in range(len(ax)):
        ax[i].grid()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def update_xlabels(min_thickness,max_thickness,n_traces):
    '''
    Updates x_labels with actual thickness of model (in meters).
    '''
    locs,labels=plt.xticks()
    incr=(max_thickness-min_thickness)/(float(n_traces)-20)
    newlabels=(locs[1:-1]-10)*incr+min_thickness
    plt.xticks(locs[1:-1],[str(round(x,1))+'m' for x in newlabels])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def update_ylabels(lag,thickness,vel):
    '''
    Updates y_labels to add lag in two-way-time,
    given velocity of top layer having certain thickness.
    '''
    locs,labels=plt.yticks()
    lagtop=thickness/vel*2
    plt.yticks(locs[:-1],[round(y+lag-lagtop,3) for y in locs])