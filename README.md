## Allesfitters

Allesfitters is a modified version of Allesfitter \citep{allesfitter-paper, allesfitter-code} with many new features.

To achieve faster modeling, I replaced ellc \citep{ellc} with PyTransit \citep{Parviainen2015} and RadVel \citep{Fulton2018RadVel}, which significantly accelerates the modeling process. However, this also means that Allesfitters currently cannot fit flares, starspots, or binary systems.

The most notable feature of Allesfitters is its incorporation of one of the most realistic Rossiter-McLaughlin models, \cite{Hirano2011}, implemented in tracit \citep{Hjorth2021, KnudstrupAlbrecht2022}. **Note that Allesfitters currently works only on x86 architecture**.

Compared to the \cite{Ohta2005} model, the \cite{Hirano2011} model introduces more parameters, especially microturbulent velocities ($V_\xi$) and macroturbulent velocities ($V_\zeta$). Some empirical relations can be used to calculate these parameters, followed by applying Gaussian priors on them (typically 1 km/s). For $V_\zeta$ and $V_\xi$, the relations from \cite{Doyle2014} and \cite{Bruntt2010} are preferred, respectively. Both have valid ranges (see code below). Outside these ranges, the empirical relations calibrated by the Gaia-ESO Survey working groups are adopted \citep{Blanco2014HighResolutionSpectralLibrary, Blanco2014iSpec, Blanco2019iSpec}.

If you use this code, I would appreciate it if you cite \citep{allesfitter-paper, allesfitter-code, Wang2024}.
Enjoy!

Tutorial for Installation and Usage: [Colab Link](https://colab.research.google.com/drive/1djLmR8l9Ujg-Ll7OH0NoWbHNctCPz4_q?usp=sharing)

```Python
import numpy as np
def _estimate_vmac_doyle2014(teff, logg, feh):
    """
    Estimate Macroturbulence velocity (Vmac) by using an empirical relation
    considering the effective temperature, surface gravity and metallicity.

    The relation was constructed by Doyle et al. (2014), which is only valid
    for the Teff range 5200 to 6400 K, and the log g range 4.0 to 4.6 dex.
    """
    t0 = 5777
    g0 = 4.44


    if teff < 5200 or teff > 6400 or logg < 4.0 or logg > 4.6:
        return np.nan

    if logg >= 3.5:
        if teff >= 5000:
            # main sequence and subgiants (RGB)
            vmac = 3.21 + 2.33e-3*(teff-t0) + 2e-6*(teff-t0)**2 - 2*(logg-g0)
        else:
            # main sequence
            vmac = 3.21 + 2.33e-3*(teff-t0) + 2e-6*(teff-t0)**2 - 2*(logg-g0)
    else:
        # Out of the calibrated limits
        vmac = 0.

    return vmac

def _estimate_vmac_ges(teff, logg, feh):
    """
    Estimate Microturbulence velocity (Vmic) by using an empirical relation
    considering the effective temperature, surface gravity and metallicity.

    The relation was constructed by Maria Bergemann for the Gaia ESO Survey.
    """
    t0 = 5500
    g0 = 4.0

    if logg >= 3.5:
        if teff >= 5000:
            # main sequence and subgiants (RGB)
            vmac = 3*(1.15 + 7e-4*(teff-t0) + 1.2e-6*(teff-t0)**2 - 0.13*(logg-g0) + 0.13*(logg-g0)**2 - 0.37*feh - 0.07*feh**2)
        else:
            # main sequence
            vmac = 3*(1.15 + 2e-4*(teff-t0) + 3.95e-7*(teff-t0)**2 - 0.13*(logg-g0) + 0.13*(logg-g0)**2)
    else:
        # giants (RGB/AGB)
        vmac = 3*(1.15 + 2.2e-5*(teff-t0) - 0.5e-7*(teff-t0)**2 - 0.1*(logg-g0) + 0.04*(logg-g0)**2 - 0.37*feh - 0.07*feh**2)

    return vmac

def estimate_vmac(teff, logg, feh, relation='GES'):
    """
    Estimate Microturbulence velocity (Vmic) by using an empirical relation
    considering the effective temperature, surface gravity and metallicity.

    By default, the selected relation was constructed by Maria Bergemann
    for the Gaia ESO Survey. Alternatively, "relation='Doyle2014'" implements
    a relation for dwrafs (Doyle et al, 2014).
    """
    if relation == 'Doyle2014':
        vmac = _estimate_vmac_doyle2014(teff, logg, feh)
    else:
        vmac = _estimate_vmac_ges(teff, logg, feh)
    vmac = float("%.2f" % vmac)
    return vmac

### vmic
def _estimate_vmic_ges(teff, logg, feh):
    """
    Estimate Microturbulence velocity (Vmic) by using an empirical relation
    considering the effective temperature, surface gravity and metallicity.

    The relation was constructed based on the UVES Gaia ESO Survey iDR1 data,
    results for the benchmark stars (Jofre et al. 2013),
    and globular cluster data from external literature sources.

    Source: http://great.ast.cam.ac.uk/GESwiki/GesWg/GesWg11/Microturbulence
    """
    t0 = 5500
    g0 = 4.0

    if logg >= 3.5:
        if teff >= 5000:
            # main sequence and subgiants (RGB)
            vmic = 1.05 + 2.51e-4*(teff-t0) + 1.5e-7*(teff-t0)**2 - 0.14*(logg-g0) - 0.05e-1*(logg-g0)**2 + 0.05*feh + 0.01*feh**2
        else:
            # main sequence
            vmic = 1.05 + 2.51e-4*(5000-t0) + 1.5e-7*(5000-t0)**2 - 0.14*(logg-g0) - 0.05e-1*(logg-g0)**2 + 0.05*feh + 0.01*feh**2
    else:
        # giants (RGB/AGB)
        vmic = 1.25 + 4.01e-4*(teff-t0) + 3.1e-7*(teff-t0)**2 - 0.14*(logg-g0) - 0.05e-1*(logg-g0)**2 + 0.05*feh + 0.01*feh**2
    vmic = float("%.2f" % vmic)
    return vmic

def _estimate_vmic_Bruntt2010(teff, logg, feh):
    # valid range: logg > 4, 5000< teff < 6500 
    # https://ui.adsabs.harvard.edu/abs/2010MNRAS.405.1907B/abstract
    t0 = 5700
    g0 = 4.0

    if logg < 4 or teff < 5000 or teff > 6500:
        return np.nan

    vmic = 1.01 + 4.5610e-4*(teff-t0) + 2.75e-7*(teff-t0)**2
  
    vmic = float("%.2f" % vmic)
    return vmic

def estimate_vmic(teff, logg, feh, relation='GES'):
    """
    Estimate Microturbulence velocity (Vmic) by using an empirical relation
    considering the effective temperature, surface gravity and metallicity.

    By default, the selected relation was constructed by Maria Bergemann
    for the Gaia ESO Survey. Alternatively, "relation='Doyle2014'" implements
    a relation for dwrafs (Doyle et al, 2014).
    """
    if relation == 'Bruntt2010':
        vmac = _estimate_vmic_Bruntt2010(teff, logg, feh)
    else:
        vmac = _estimate_vmic_ges(teff, logg, feh)
    vmac = float("%.2f" % vmac)
    return vmac


teff = 3678
logg = 4.5
feh = 0.23

vmic = estimate_vmic(teff, logg, feh, relation='Bruntt2010')
if np.isnan(vmic):
    vmic = estimate_vmic(teff, logg, feh, relation='GES')

vmac = estimate_vmac(teff, logg, feh, relation='Doyle2014')
if np.isnan(vmac):
    vmac = estimate_vmac(teff, logg, feh, relation='GES')

print(vmac,vmic)
```

Reference:

```python
@ARTICLE{Doyle2014,
       author = {{Doyle}, Amanda P. and {Davies}, Guy R. and {Smalley}, Barry and {Chaplin}, William J. and {Elsworth}, Yvonne},
        title = "{Determining stellar macroturbulence using asteroseismic rotational velocities from Kepler}",
      journal = {\mnras},
     keywords = {asteroseismology, line: profiles, planets and satellites: fundamental parameters, stars: rotation, Astrophysics - Solar and Stellar Astrophysics},
         year = 2014,
        month = nov,
       volume = {444},
       number = {4},
        pages = {3592-3602},
          doi = {10.1093/mnras/stu1692},
archivePrefix = {arXiv},
       eprint = {1408.3988},
 primaryClass = {astro-ph.SR},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2014MNRAS.444.3592D},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
@ARTICLE{Bruntt2010,
       author = {{Bruntt}, H. and {Bedding}, T.~R. and {Quirion}, P. -O. and {Lo Curto}, G. and {Carrier}, F. and {Smalley}, B. and {Dall}, T.~H. and {Arentoft}, T. and {Bazot}, M. and {Butler}, R.~P.},
        title = "{Accurate fundamental parameters for 23 bright solar-type stars}",
      journal = {\mnras},
     keywords = {Astrophysics - Solar and Stellar Astrophysics},
         year = 2010,
        month = jul,
       volume = {405},
       number = {3},
        pages = {1907-1923},
          doi = {10.1111/j.1365-2966.2010.16575.x},
archivePrefix = {arXiv},
       eprint = {1002.4268},
 primaryClass = {astro-ph.SR},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2010MNRAS.405.1907B},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
@ARTICLE{Blanco2014iSpec,
       author = {{Blanco-Cuaresma}, S. and {Soubiran}, C. and {Heiter}, U. and {Jofr{\'e}}, P.},
        title = "{Determining stellar atmospheric parameters and chemical abundances of FGK stars with iSpec}",
      journal = {\aap},
     keywords = {stars: atmospheres, stars: abundances, methods: data analysis, Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Solar and Stellar Astrophysics},
         year = 2014,
        month = sep,
       volume = {569},
          eid = {A111},
        pages = {A111},
          doi = {10.1051/0004-6361/201423945},
archivePrefix = {arXiv},
       eprint = {1407.2608},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2014A&A...569A.111B},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}


@ARTICLE{Blanco2014HighResolutionSpectralLibrary,
       author = {{Blanco-Cuaresma}, S. and {Soubiran}, C. and {Jofr{\'e}}, P. and {Heiter}, U.},
        title = "{The Gaia FGK benchmark stars. High resolution spectral library}",
      journal = {\aap},
     keywords = {stars: atmospheres, stars: abundances, Galaxy: general, galaxies: stellar content, Astrophysics - Solar and Stellar Astrophysics, Astrophysics - Astrophysics of Galaxies, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = 2014,
        month = jun,
       volume = {566},
          eid = {A98},
        pages = {A98},
          doi = {10.1051/0004-6361/201323153},
archivePrefix = {arXiv},
       eprint = {1403.3090},
 primaryClass = {astro-ph.SR},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2014A&A...566A..98B},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
@ARTICLE{Blanco2019iSpec,
       author = {{Blanco-Cuaresma}, Sergi},
        title = "{Modern stellar spectroscopy caveats}",
      journal = {\mnras},
     keywords = {techniques: spectroscopic, stars: abundances, stars: atmospheres, stars: fundamental parameters, Astrophysics - Solar and Stellar Astrophysics, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = 2019,
        month = jun,
       volume = {486},
       number = {2},
        pages = {2075-2101},
          doi = {10.1093/mnras/stz549},
archivePrefix = {arXiv},
       eprint = {1902.09558},
 primaryClass = {astro-ph.SR},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2019MNRAS.486.2075B},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
@article{Parviainen2015,
  author = {Parviainen, Hannu},
  doi = {10.1093/mnras/stv894},
  journal = {MNRAS},
  number = {April},
  pages = {3233--3238},
  title = {{PYTRANSIT: fast and easy exoplanet transit modelling in PYTHON}},
  url = {http://mnras.oxfordjournals.org/cgi/doi/10.1093/mnras/stv894},
  volume = {450},
  year = {2015}
}
@ARTICLE{ellc,
       author = {{Maxted}, P.~F.~L.},
        title = "{ellc: A fast, flexible light curve model for detached eclipsing binary stars and transiting exoplanets}",
      journal = {\aap},
     keywords = {binaries: eclipsing, methods: data analysis, methods: numerical, Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Earth and Planetary Astrophysics, Astrophysics - Solar and Stellar Astrophysics},
         year = 2016,
        month = jun,
       volume = {591},
          eid = {A111},
        pages = {A111},
          doi = {10.1051/0004-6361/201628579},
archivePrefix = {arXiv},
       eprint = {1603.08484},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2016A&A...591A.111M},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@ARTICLE{Fulton2018RadVel,
       author = {{Fulton}, Benjamin J. and {Petigura}, Erik A. and {Blunt}, Sarah and {Sinukoff}, Evan},
        title = "{RadVel: The Radial Velocity Modeling Toolkit}",
      journal = {\pasp},
     keywords = {Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Earth and Planetary Astrophysics},
         year = 2018,
        month = apr,
       volume = {130},
       number = {986},
        pages = {044504},
          doi = {10.1088/1538-3873/aaaaa8},
       eprint = {1801.01947},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2018PASP..130d4504F},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@ARTICLE{allesfitter-paper,
 author = {{G{\"u}nther}, Maximilian N. and {Daylan}, Tansu},
 title = "{Allesfitter: Flexible Star and Exoplanet Inference from Photometry and Radial Velocity}",
 journal = {\apjs},
 keywords = {Exoplanets, Binary stars, Stellar flares, Bayesian statistics, Astronomy software, Starspots, Astronomy data modeling, 498, 154, 1603, 1900, 1855, 1572, 1859, Astrophysics - Earth and Planetary Astrophysics, Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Solar and Stellar Astrophysics},
 year = 2021,
 month = may,
 volume = {254},
 number = {1},
 eid = {13},
 pages = {13},
 doi = {10.3847/1538-4365/abe70e},
 archivePrefix = {arXiv},
 eprint = {2003.14371},
 primaryClass = {astro-ph.EP},
 adsurl = {https://ui.adsabs.harvard.edu/abs/2021ApJS..254...13G},
 adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@MISC{allesfitter-code,
 author = {{G{\"u}nther}, Maximilian~N. and {Daylan}, Tansu},
 title = "{Allesfitter: Flexible Star and Exoplanet Inference From Photometry and Radial Velocity}",
 keywords = {Software },
 howpublished = {Astrophysics Source Code Library},
 year = 2019,
 month = mar,
 archivePrefix = "ascl",
 eprint = {1903.003},
 adsurl = {http://adsabs.harvard.edu/abs/2019ascl.soft03003G},
 adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}


@ARTICLE{Hirano2011,
       author = {{Hirano}, Teruyuki and {Suto}, Yasushi and {Winn}, Joshua N. and {Taruya}, Atsushi and {Narita}, Norio and {Albrecht}, Simon and {Sato}, Bun'ei},
        title = "{Improved Modeling of the Rossiter-McLaughlin Effect for Transiting Exoplanets}",
      journal = {\apj},
     keywords = {planets and satellites: general, planets and satellites: formation, stars: rotation, techniques: radial velocities, techniques: spectroscopic, Astrophysics - Earth and Planetary Astrophysics, Astrophysics - Solar and Stellar Astrophysics},
         year = 2011,
        month = dec,
       volume = {742},
       number = {2},
          eid = {69},
        pages = {69},
          doi = {10.1088/0004-637X/742/2/69},
archivePrefix = {arXiv},
       eprint = {1108.4430},
 primaryClass = {astro-ph.EP},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2011ApJ...742...69H},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@ARTICLE{Hjorth2021,
       author = {{Hjorth}, Maria and {Albrecht}, Simon and {Hirano}, Teruyuki and {Winn}, Joshua N. and {Dawson}, Rebekah I. and {Zanazzi}, J.~J. and {Knudstrup}, Emil and {Sato}, Bun'ei},
        title = "{A backward-spinning star with two coplanar planets}",
      journal = {Proceedings of the National Academy of Science},
     keywords = {Astrophysics - Earth and Planetary Astrophysics, Astrophysics - Solar and Stellar Astrophysics},
         year = 2021,
        month = feb,
       volume = {118},
       number = {8},
          eid = {e2017418118},
        pages = {e2017418118},
          doi = {10.1073/pnas.2017418118},
archivePrefix = {arXiv},
       eprint = {2102.07677},
 primaryClass = {astro-ph.EP},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2021PNAS..11817418H},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@ARTICLE{KnudstrupAlbrecht2022,
       author = {{Knudstrup}, E. and {Albrecht}, S.~H.},
        title = "{Orbital alignment of HD 332231 b. The warm Saturn HD 332231 b/TOI-1456 b travels on a well-aligned, circular orbit around a bright F8 dwarf}",
      journal = {\aap},
     keywords = {methods: observational, techniques: spectroscopic, techniques: photometric, planet-star interactions, planets and satellites: dynamical evolution and stability, Astrophysics - Earth and Planetary Astrophysics, Astrophysics - Solar and Stellar Astrophysics},
         year = 2022,
        month = apr,
       volume = {660},
          eid = {A99},
        pages = {A99},
          doi = {10.1051/0004-6361/202142726},
archivePrefix = {arXiv},
       eprint = {2111.14968},
 primaryClass = {astro-ph.EP},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2022A&A...660A..99K},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@ARTICLE{Ohta2005,
       author = {{Ohta}, Yasuhiro and {Taruya}, Atsushi and {Suto}, Yasushi},
        title = "{The Rossiter-McLaughlin Effect and Analytic Radial Velocity Curves for Transiting Extrasolar Planetary Systems}",
      journal = {\apj},
     keywords = {Stars: Planetary Systems, Stars: Individual: Henry Draper Number: HD 209458, Techniques: Spectroscopic, Astrophysics},
         year = 2005,
        month = apr,
       volume = {622},
       number = {2},
        pages = {1118-1135},
          doi = {10.1086/428344},
archivePrefix = {arXiv},
       eprint = {astro-ph/0410499},
 primaryClass = {astro-ph},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2005ApJ...622.1118O},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@ARTICLE{Wang2024,
       author = {{Wang}, Xian-Yu and {Rice}, Malena and {Wang}, Songhu and {Kanodia}, Shubham and {Dai}, Fei and {Logsdon}, Sarah E. and {Schweiker}, Heidi and {Teske}, Johanna K. and {Butler}, R. Paul and {Crane}, Jeffrey D. and {Shectman}, Stephen and {Quinn}, Samuel N. and {Kostov}, Veselin and {Osborn}, Hugh P. and {Goeke}, Robert F. and {Eastman}, Jason D. and {Shporer}, Avi and {Rapetti}, David and {Collins}, Karen A. and {Watkins}, Cristilyn N. and {Relles}, Howard M. and {Ricker}, George R. and {Seager}, Sara and {Winn}, Joshua N. and {Jenkins}, Jon M.},
        title = "{Single-star Warm-Jupiter Systems Tend to Be Aligned, Even around Hot Stellar Hosts: No T $_{eff}${\textendash}{\ensuremath{\lambda}} Dependency}",
      journal = {\apjl},
     keywords = {Planetary alignment, Exoplanet dynamics, Exoplanet evolution, Star-planet interactions, Exoplanets, Planetary theory, Exoplanet systems, Exoplanet astronomy, Planetary dynamics, Hot Jupiters, 1243, 490, 491, 2177, 498, 1258, 484, 486, 2173, 753, Astrophysics - Earth and Planetary Astrophysics},
         year = 2024,
        month = sep,
       volume = {973},
       number = {1},
          eid = {L21},
        pages = {L21},
          doi = {10.3847/2041-8213/ad7469},
archivePrefix = {arXiv},
       eprint = {2408.10038},
 primaryClass = {astro-ph.EP},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024ApJ...973L..21W},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}



```
