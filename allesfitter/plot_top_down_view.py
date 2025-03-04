#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 16:06:09 2019

@author:
Maximilian N. Günther
MIT Kavli Institute for Astrophysics and Space Research, 
Massachusetts Institute of Technology,
77 Massachusetts Avenue,
Cambridge, MA 02109, 
USA
Email: maxgue@mit.edu
Web: www.mnguenther.com
"""

from __future__ import print_function, division, absolute_import

#::: plotting settings
import seaborn as sns
sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5, color_codes=True)
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
sns.set_context(rc={'lines.markeredgewidth': 1})

#::: modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Circle
import matplotlib.ticker as plticker
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import random
import warnings
try:
    import rebound
    from rebound.particle import Particle
except:
    warnings.warn('Module "rebound" could not be imported. Orbital plots are not available.')
from itertools import cycle

#::: allesfitter codes
from allesfitter import config
from allesfitter.exoworlds_rdx.lightcurves.index_transits import get_first_epoch
from allesfitter.exoworlds_rdx.lightcurves.lightcurve_tools import calc_phase




def OrbitPlot(sim, figsize=None, lim=None, limz=None, Narc=100, xlabel='x', ylabel='y', zlabel='z', color=False, periastron=False, trails=True, show_orbit=True, lw=1., glow=False, slices=False, plotparticles=[], primary=None, fancy=False, ax=None):
    """
    Convenience function for plotting instantaneous orbits.

    Parameters
    ----------
    slices          : bool, optional
        Plot all three slices if set to True. Default is False and plots orbits only in the xy plane.
    figsize         : tuple of float, optional
        Tuple defining the figure size (default: (5,5))
    lim             : float, optional           
        Limit for axes (default: None = automatically determined)
    limz            : float, optional           
        Limit for z axis, only used if slices=True (default: None = automatically determined)
    unitlabel       : str, optional          
        String describing the units, shown on axis labels (default: None)
    color           : bool, str or list, optional            
        By default plots in black. If set to True, plots using REBOUND color cycle. If a string or list of strings, e.g. ['red', 'cyan'], will cycle between passed colors.
    periastron  : bool, optional            
        Draw a marker at periastron (default: False)
    trails          : bool, optional            
        Draw trails instead of solid lines (default: False)
    show_orbit      : bool, optional
        Draw orbit trails/lines (default: True)
    lw              : float, optional           
        Linewidth (default: 1.)
    glow            : bool (default: False)
        Make lines glow
    fancy           : bool (default: False)
        Changes various settings to create a fancy looking plot
    plotparticles   : list, optional
        List of particles to plot. Can be a list of any valid keys for accessing sim.particles, i.e., integer indices or hashes (default: plot all particles)
    primary         : rebound.Particle, optional
        Pimrary to use for the osculating orbit (default: Jacobi center of mass)

    Returns
    -------
    fig
        A matplotlib figure

    Examples
    --------
    The following example illustrates a typical use case.

    >>> sim = rebound.Simulation()
    >>> sim.add(m=1)
    >>> sim.add(a=1)
    >>> fig = rebound.OrbitPlot(sim)
    >>> fig.savefig("image.png") # save figure to file
    >>> fig.show() # show figure on screen

    """
    if slices:
        if figsize is None:
            figsize = (8,8)
        if ax is None:
            fig, ax = plt.subplots(2, 2, figsize=figsize)
        gs = gridspec.GridSpec(2, 2, width_ratios=[3., 2.], height_ratios=[2.,3.],wspace=0., hspace=0.) 
        OrbitPlotOneSlice(sim, plt.subplot(gs[2]), lim=lim, Narc=Narc, color=color, periastron=periastron, trails=trails, show_orbit=show_orbit, lw=lw, axes="xy",fancy=fancy, plotparticles=plotparticles, primary=primary, glow=glow)
        OrbitPlotOneSlice(sim, plt.subplot(gs[3]), lim=lim, limz=limz, Narc=Narc, color=color, periastron=periastron, trails=trails, show_orbit=show_orbit, lw=lw,fancy=fancy, axes="zy", plotparticles=plotparticles, primary=primary, glow=glow)
        OrbitPlotOneSlice(sim, plt.subplot(gs[0]), lim=lim, limz=limz, Narc=Narc, color=color, periastron=periastron, trails=trails, show_orbit=show_orbit, lw=lw,fancy=fancy, axes="xz", plotparticles=plotparticles, primary=primary, glow=glow)
        plt.subplot(gs[2]).set_xlabel(xlabel)
        plt.subplot(gs[2]).set_ylabel(ylabel)
      
        plt.setp(plt.subplot(gs[0]).get_xticklabels(), visible=False)
        plt.subplot(gs[0]).set_ylabel(zlabel)
        
        plt.subplot(gs[3]).set_xlabel(zlabel)
        plt.setp(plt.subplot(gs[3]).get_yticklabels(), visible=False)
    else:
        if figsize is None:
            figsize = (5,5)
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        OrbitPlotOneSlice(sim, ax, lim=lim, Narc=Narc, color=color, periastron=periastron, trails=trails, show_orbit=show_orbit, lw=lw,fancy=fancy, plotparticles=plotparticles, primary=primary, glow=glow)
    return plt.gcf(), ax




def get_color(color):
    """
    Takes a string for a color name defined in matplotlib and returns of a 3-tuple of RGB values.
    Will simply return passed value if it's a tuple of length three.

    Parameters
    ----------
    color   : str
        Name of matplotlib color to calculate RGB values for.
    """

    if isinstance(color, tuple) and len(color) == 3: # already a tuple of RGB values
        return color

    hexcolor = sns.colors.xkcd_rgb[color]

    hexcolor = hexcolor.lstrip('#')
    lv = len(hexcolor)
    return tuple(int(hexcolor[i:i + lv // 3], 16)/255. for i in range(0, lv, lv // 3)) # tuple of rgb values




def fading_line(x, y, color='black', alpha_initial=1., alpha_final=0., glow=False, **kwargs):
    """
    Returns a matplotlib LineCollection connecting the points in the x and y lists, with a single color and alpha varying from alpha_initial to alpha_final along the line.
    Can pass any kwargs you can pass to LineCollection, like linewidgth.

    Parameters
    ----------
    x       : list or array of floats for the positions on the (plot's) x axis
    y       : list or array of floats for the positions on the (plot's) y axis
    color   : matplotlib color for the line. Can also pass a 3-tuple of RGB values (default: 'black')
    alpha_initial:  Limiting value of alpha to use at the beginning of the arrays.
    alpha_final:    Limiting value of alpha to use at the end of the arrays.
    """
    if glow:
        glow = False
        kwargs["lw"] = 1
        fl1 = fading_line(x, y, color, alpha_initial, alpha_final, glow=False, **kwargs)
        kwargs["lw"] = 2
        alpha_initial *= 0.5
        alpha_final *= 0.5
        fl2 = fading_line(x, y, color, alpha_initial, alpha_final, glow=False, **kwargs)
        kwargs["lw"] = 6
        alpha_initial *= 0.5
        alpha_final *= 0.5
        fl3 = fading_line(x, y, color, alpha_initial, alpha_final, glow=False, **kwargs)
        return [fl3,fl2,fl1]

    color = get_color(color)
    cdict = {'red': ((0.,color[0],color[0]),(1.,color[0],color[0])),
             'green': ((0.,color[1],color[1]),(1.,color[1],color[1])),
             'blue': ((0.,color[2],color[2]),(1.,color[2],color[2])),
             'alpha': ((0.,alpha_initial, alpha_initial), (1., alpha_final, alpha_final))}
    
    Npts = len(x)
    if len(y) != Npts:
        raise AttributeError("x and y must have same dimension.")
   
    segments = np.zeros((Npts-1,2,2))
    segments[0][0] = [x[0], y[0]]
    for i in range(1,Npts-1):
        pt = [x[i], y[i]]
        segments[i-1][1] = pt
        segments[i][0] = pt 
    segments[-1][1] = [x[-1], y[-1]]

    individual_cm = LinearSegmentedColormap('indv1', cdict)
    lc = LineCollection(segments, cmap=individual_cm, **kwargs)
    lc.set_array(np.linspace(0.,1.,len(segments)))
    return lc




def OrbitPlotOneSlice(sim, ax, lim=None, limz=None, Narc=100, color=False, periastron=False, trails=False, show_orbit=True, lw=1., axes="xy", plotparticles=[], primary=None, glow=False, fancy=False):
    p_orb_pairs = []
    if not plotparticles:
        plotparticles = range(1, sim.N_real)
    for i in plotparticles:
        p = sim.particles[i]
        p_orb_pairs.append((p, p.calculate_orbit(primary=primary)))

    if lim is None:
        lim = 0.
        for p, o in p_orb_pairs: 
            if o.a>0.:
                r = (1.+o.e)*o.a
            else:
                r = o.d
            if r>lim:
                lim = r
        lim *= 1.15
    if limz is None:
        z = [p.z for p,o in p_orb_pairs]
        limz = 2.0*max(z)
        if limz > lim:
            limz = lim
        if limz <= 0.:
            limz = lim

    if axes[0]=="z":
        ax.set_xlim([-limz,limz])
    else:
        ax.set_xlim([-lim,lim])
    if axes[1]=="z":
        ax.set_ylim([-limz,limz])
    else:
        ax.set_ylim([-lim,lim])
        
    if fancy:
        ax.set_facecolor((0.,0.,0.))
        for pos in ['top', 'bottom', 'right', 'left']:
            ax.spines[pos].set_edgecolor((0.3,0.3,0.3))

    if color is not False:
        if isinstance(color, list):
            colors = []
            for c in color:
                colors.append(get_color(c))
        elif isinstance(color, str):
            colors = [get_color(color)]
        elif color == True:
            colors = [(1.,0.,0.),(0.,0.75,0.75),(0.75,0.,0.75),(0.75, 0.75, 0,),(0., 0., 0.),(0., 0., 1.),(0., 0.5, 0.)]
    else:
        if fancy:
            colors = [(181./206.,66./206.,191./206.)]
            glow = True
        else:
            colors = ["black"]
    coloriterator = cycle(colors)

#    coords = {'x':0, 'y':1, 'z':2}
#    axis0 = coords[axes[0]]
#    axis1 = coords[axes[1]]
   
    prim = sim.particles[0] if primary is None else primary 
    if fancy:
        sun = (256./256.,256./256.,190./256.)
        opa = 0.020
        size = 6000.
        for i in range(256):
            ax.scatter(getattr(prim,axes[0]),getattr(prim,axes[1]), alpha=opa, s=size*lw, facecolor=sun, edgecolor=None, zorder=3)
            size *= 0.95
        
        starcolor = (1.,1.,1.)
        mi, ma = ax.get_xlim()
        prestate = random.getstate()
        random.seed(1) #always same stars
        x, y = [], []
        #small stars
        for i in range(64):
            x.append(random.uniform(mi,ma))
            y.append(random.uniform(mi,ma))
        ax.scatter(x,y, alpha=0.05, s=8*lw, facecolor=starcolor, edgecolor=None, zorder=3)
        ax.scatter(x,y, alpha=0.1, s=4*lw, facecolor=starcolor, edgecolor=None, zorder=3)
        ax.scatter(x,y, alpha=0.2, s=0.5*lw, facecolor=starcolor, edgecolor=None, zorder=3)
        #medium stars
        x, y = [], []
        for i in range(16):
            x.append(random.uniform(mi,ma))
            y.append(random.uniform(mi,ma))
        ax.scatter(x,y, alpha=0.1, s=15*lw, facecolor=starcolor, edgecolor=None, zorder=3)
        ax.scatter(x,y, alpha=0.1, s=5*lw, facecolor=starcolor, edgecolor=None, zorder=3)
        ax.scatter(x,y, alpha=0.5, s=2*lw, facecolor=starcolor, edgecolor=None, zorder=3)
        random.setstate(prestate)

    else:
        ax.scatter(getattr(prim,axes[0]),getattr(prim,axes[1]), marker="*", s=35*lw, facecolor="black", edgecolor=None, zorder=3)
    
    proj = {}
    for p, o in p_orb_pairs:
        colori = next(coloriterator)
        
        prim = p.jacobi_com if primary is None else primary 
        if fancy:
            ax.scatter(getattr(p,axes[0]), getattr(p,axes[1]), s=25*lw, facecolor=colors, edgecolor=None, zorder=3)
        else:
            pass
            #::: !!!
            #::: !!!
            #::: plot the planet symbols
            #::: !!!
            #::: !!!
#            ax.scatter(getattr(p,axes[0]), getattr(p,axes[1]), s=p.r*lw, facecolor=colori, edgecolor=None, zorder=3)

       
        if show_orbit is True:
            alpha_final = 0. if trails is True else 1. # fade to 0 with trails

            hyperbolic = o.a < 0. # Boolean for whether orbit is hyperbolic
            if hyperbolic is False:
                pts = np.array(p.sample_orbit(Npts=Narc+1, primary=prim))
                proj['x'],proj['y'],proj['z'] = [pts[:,i] for i in range(3)]
                lc = fading_line(proj[axes[0]], proj[axes[1]], colori, alpha_final=alpha_final, lw=lw, glow=glow)
                if type(lc) is list:
                    for l in lc:
                        ax.add_collection(l)
                else:
                    ax.add_collection(lc)

            else:
                pts = np.array(p.sample_orbit(Npts=Narc+1, primary=prim, useTrueAnomaly=False))
                # true anomaly stays close to limiting value and switches quickly at pericenter for hyperbolic orbit, so use mean anomaly
                proj['x'],proj['y'],proj['z'] = [pts[:,i] for i in range(3)]
                lc = fading_line(proj[axes[0]], proj[axes[1]], colori, alpha_final=alpha_final, lw=lw, glow=glow)
                if type(lc) is list:
                    for l in lc:
                        ax.add_collection(l)
                else:
                    ax.add_collection(lc)
          
                alpha = 0.2 if trails is True else 1.
                pts = np.array(p.sample_orbit(Npts=Narc+1, primary=prim, trailing=False, useTrueAnomaly=False))
                proj['x'],proj['y'],proj['z'] = [pts[:,i] for i in range(3)]
                lc = fading_line(proj[axes[0]], proj[axes[1]], colori, alpha_initial=alpha, alpha_final=alpha, lw=lw, glow=glow)
                if type(lc) is list:
                    for l in lc:
                        ax.add_collection(l)
                else:
                    ax.add_collection(lc)

        if periastron:
            newp = Particle(a=o.a, f=0., inc=o.inc, omega=o.omega, Omega=o.Omega, e=o.e, m=p.m, primary=prim, simulation=sim)
            ax.plot([getattr(prim,axes[0]), getattr(newp,axes[0])], [getattr(prim,axes[1]), getattr(newp,axes[1])], linestyle="dotted", c=colori, zorder=1, lw=lw)
            ax.scatter([getattr(newp,axes[0])],[getattr(newp,axes[1])], marker="o", s=5.*lw, facecolor="none", edgecolor=colori, zorder=1)




def plot_top_down_view(params_median, params_star, a=None, timestep=None, scaling=5., colors=sns.color_palette('deep'), linewidth=2, plot_arrow=False, ax=None):
    
    sim = rebound.Simulation()
    sim.add(m=1)
    
    for i, companion in enumerate(config.BASEMENT.settings['companions_all']):
        if (i==0) and (timestep is None): 
            timestep = params_median[companion+'_epoch'] #calculate it for the timestep where the first companion is in transit
        first_epoch = get_first_epoch(timestep, params_median[companion+'_epoch'], params_median[companion+'_period'])
        phase = calc_phase(timestep, params_median[companion+'_period'], first_epoch)
        ecc = params_median[companion+'_f_s']**2 + params_median[companion+'_f_c']**2
        w = np.arccos( params_median[companion+'_f_c'] / np.sqrt(ecc) ) #in rad
        inc = params_median[companion+'_incl']/180.*np.pi
        if a is None:
            a1 = params_star['R_star'] / params_median[companion+'_radius_1'] #in Rsun 
            a1 *= 0.004650467260962157 #in AU
        else:
            a1 = a[i]
#        print(a, inc, ecc, w, phase*2*np.pi)
        if ecc>0:
            sim.add(a=a1, inc=inc-np.pi/2., e=ecc, omega=w, f=phase*2*np.pi)
        else:
            sim.add(a=a1, inc=inc--np.pi/2., f=phase*2*np.pi)
#    print(len(sim.particles))
    
#    print('Epoch, Period and mean anomaly, b:', sim.particles[0].M )
#    print('Mean anomaly, c:', sim.particles[0].M )
#    print('Mean anomaly, c:', sim.particles[0].M )
#    err
    
    fig, ax = OrbitPlot(sim, xlabel='AU', ylabel='AU', color=colors, lw=linewidth, ax=ax) #color=[sns.color_palette('deep')[i] for i in [0,1,3]],
    
    for i, companion in enumerate(config.BASEMENT.settings['companions_all']):
        
        R_companion = params_star['R_star'] * params_median[companion+'_rr'] # in Rsun
        R_companion *= 0.004650467260962157 #in AU
        R_companion *= scaling
    
        
        x = sim.particles.get(i+1).x
        y = sim.particles.get(i+1).y
        p = Circle((x,y), R_companion, color=colors[i])
        ax.add_artist(p)
        
    if plot_arrow:
        x0, x1 = ax.get_xlim()
        plt.arrow( 0.1*x1, 0, 0.7*x1, 0, color='silver', zorder=1 ) 
              
    plt.axis('equal')
#    ax.set(xlim=[-0.12,0.12], ylim=[-0.12,0.12])
#    loc = plticker.MultipleLocator(base=0.05) # this locator puts ticks at regular intervals
#    ax.xaxis.set_major_locator(loc)
#    ax.yaxis.set_major_locator(loc)

    return fig, ax


