> [!WARNING]
> This is currently work in progress. Everything and anything is subject to change, including licensing information.

<p align="center">
    <img alt="GitHub License" src="https://img.shields.io/github/license/kabeleh/iDM">
    <br>
    <img alt="GitHub language count" src="https://img.shields.io/github/languages/count/kabeleh/iDM">
    <img alt="GitHub top language" src="https://img.shields.io/github/languages/top/kabeleh/iDM">
    <img alt="Codacy grade" src="https://img.shields.io/codacy/grade/68ad366818ce43e28d91b709155dac0f">
    <a href="https://github.com/psf/black"><img alt="Python Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
    <br>
    <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/kabeleh/iDM">
    <img alt="GitHub commit activity" src="https://img.shields.io/github/commit-activity/w/kabeleh/iDM">
    <img alt="GitHub code size in bytes" src="https://img.shields.io/github/languages/code-size/kabeleh/iDM">
    <img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/kabeleh/iDM">
</p>

# Dark Energy–Dark Matter Interactions
- Author: Kay Lehnert
- Related Publications: 
    - [Hitchhiker's Guide to the Swampland: The Cosmologist's Handbook to the string-theoretical Swampland Programme](https://arxiv.org/abs/2509.02632)
    - [PhD Thesis](url.com)

This repository contains the [`CLASS`](https://github.com/lesgourg/class_public) source code used for the cosmological simulations, the [`Cobaya`](https://cobaya.readthedocs.io/en/latest/) configuration files for the Markov chain Monte Carlo runs, and the [Mathematica](https://www.wolfram.com/mathematica/) notebooks with the derivations of the relevant equations for a swampland-inspired model of dark energy in the form of quintessence as a scalar field that is interacting with dynamical dark matter. The motivation, theory behind, and mathematical details of the model are explained in my PhD thesis [^4]. For the technical implementation of the mathematical equations into `CLASS`, please refer to the extensive [GitHub Wiki](https://github.com/kabeleh/iDM/wiki).


[^4]: TBA

## CLASS
The source code in this repository is a fork of the Cosmic Linear Anisotropy Solving System ([CLASS](https://github.com/lesgourg/class_public)) v3.3.4 by Julien Lesgourgues, Thomas Tram, Nils Schoeneberg et al.

<details>
<summary>Technical implementation of the cosmological model</summary>
    
The source code itself is commented and changes by us are indicated by the initials `KBL` as a comment.
Please note the extensive [Wiki Page](https://github.com/kabeleh/iDM/wiki) that documents the notable code changes and explains the physical and mathematical considerations behind those changes. 
</details>

<details>
<summary>How to compile and run the code</summary>
    
1. Either download the archive of the source code or `git pull` this repository to make a local copy.
2. Navigate to the folder that contains the root of the CLASS source code (the one that contains the *.ini-files as well as the `Makefile`).
3. <details>
    <summary>If you want to profit from Profile-Guided Optimizations (PGO)</summary>
    1. Uncomment the PGO flags in the makefile (read our comment regarding PGO).
    2. Run typical workload, e.g. several *.ini files with different model settings.
    3. Comment the PGO-creation compile flags and uncomment the PGO-implementation flags.
    </details>
4. Run `make clean; make class -j` to compile the code.
5. Run the cosmological simulation with `./class iDM.ini`
</details>

If you want to use this code, please cite [CLASS II: Approximation schemes](http://arxiv.org/abs/1104.2933) as well as `my thesis`.[^4]

## Cobaya
The Markov chain Monte Carlo (MCMC) runs to find the best-fit parameters of the model and compare it to $`\Lambda`$CDM were performed using [`Cobaya`](https://cobaya.readthedocs.io/en/latest/) v3.6.1 by Jesus Torrado and Antony Lewis. To assess our work, the full pipeline is openly available, with the MCMC products stored on [`Zenodo`](TODO).

The [`Zenodo`](TODO) repository contains
- Cobaya configuration-files (YAML)
- best-fit values from `--minimize` runs
- covariance matrices
- the full chains

These files allow you to fully reproduce my findings:
- The configuration-files recreate the exact MCMC-pipeline, choosing the likelihoods, starting parameters/priors, and run configuration.
- The best-fit values can be used to accelerate the reproduction of my MCMC chains. Those can be used as a starting value, i.e. your MCMC will then start at the best-fit value right away.
- The covariance matrices can also be passed to `Cobaya`, as initial guess. This also accelerates reproduction of my results.

The following naming scheme is applied to these files: 
```
        Sampler                 _    Likelihoods                            _    Type of Dark Energy                _    Dark Energy Conditions                           .    file-type
cobaya  < mcmc | polychord >    _    < CMB | SPA | PP | S | DESI | DES >    _    < LCDM | hyperbolic | DoubleExp >  _    < InitCond | tracking | coupled | uncoupled >    .    < txt | bestfit | covmat | yml >
```
The [types of dark energy](https://github.com/kabeleh/iDM/wiki/New-User-Input-Parameters#scalar-field) are explained in the [Wiki](https://github.com/kabeleh/iDM/wiki), as well as the [coupling](https://github.com/kabeleh/iDM/wiki/Coupling) the [initial conditions](https://github.com/kabeleh/iDM/wiki/Attractor-Solutions).

The data sets are the following:
- CMB: Planck 2018 (low TT|EE, marginalised high TT|TE|EE, lensed)
- SPA: SPT-3G + ACT DR6 + Planck (low TT)
- PP: Pantheon+
- DESI: DESI DR2
- S: SH0ES
- DES: DESY1 (shear)

## Mathematica
The Mathematica notebooks contain the derivation of the governing equations, equations of motions, the potentials and their derivatives, as well as additional tests for swampland-compatibility.

- `xPert.nb` uses the [xAct](https://xact.es) package [xPert](https://xact.es/xPert/index.html) to derive metric-independent equations of motions from the Lagrangian and the stress–energy tensor. It includes first-order perturbations.
- `xPand.nb` takes those equations and expresses them with respect to a metric and simplifies the equations using the Newtonian gauge. This notebook uses the [xPand](https://www2.iap.fr/users/pitrou/xpand.htm) plugin.
- `rho_cdm.nb` computes the derivatives of the interacting dark matter mass $m\left(\phi\right)=1-\tanh\left(c\phi\right)$ and expresses them in trigonometric functions.

## Python Helper
I used various little helper scripts to reduce human error, following the credo 'automate everything':
- `PostProcessing\EnergyConsumption.py`: Computations use energy. Each of my SLURM scripts ends with fetching the used energy of the run in Joule. This script collects all *.out files, looks for the table at the end of the ouput, and sums the energy usage. Luckily, the computations were hydro-powered and therefore carbon-neutral.
- `PostProcessing\data_postprocessing_getDist.py`: This file is mostly a [`GetDist`](https://getdist.readthedocs.io/en/latest/) wrapper: it loads the chains and best-fit data and creates the plots and LaTeX tables for the chosen parameters.
- `PostProcessing\getNumberOfDataPoints.py`: To compute the Bayesian Information Criterion (BIC) the number of data points per likelihood is required. To get the actual number used at runtime, this script can be interrupted in debugger mode. Different likelihoods store the data points in different ways, which made it necessary to retrieve this number by hand.
- `Cobaya\cobaya_progress.py`: Reads the *.progress file and creates a little plot with the number of accepted points and the convergence between the different MCMC chains.
- `Cobaya\create_cobaya_yaml.py`: Creates the [`Cobaya`](https://cobaya.readthedocs.io/en/latest/) configuration files, as well as the SLURM scripts to run them.

## Thesis
My PhD thesis can be found on [arXiv](TODO), [MURAL](TODO), as well as in the folder `Thesis`. This folder contains, besides the PDF, the `LaTeX` code. The code acts as a template for similar documents and makes the equations available for typesetting.

The thesis template is an adaptation of [TeXtured](https://github.com/jdujava/TeXtured) with inspiration taken from [Tony Zorman](https://tony-zorman.com/posts/phd-typesetting.html). It is compliant with Maynooth's [Doctoral Thesis Layout Recommendations](https://www.maynoothuniversity.ie/exams/postgraduate-information).


# FAIR Data Statement
Our research complies with FAIR data principles:
- **F**indable: All data is explained and made available on this `GitHub` repository or [`Zenodo`](TODO). The naming schemes are explained in this `README`. Keywords are used for Search Engine Optimisation and clear reference.
- **A**ccessible:
  - All data is available on this GitHub repository, except for the Markov chains, which are available on [Zenodo](TODO). Furthermore, the thesis is available on [arXiv](TODO) and on the Maynooth University Research Archive Library ([MURAL](TODO))
  - Plots use [scientific colourmaps](https://www.fabiocrameri.ch/colourmaps/) to increase readability for colour-vision deficient and colour-blind people, as well as on black-and-white printouts.
- **I**nteroperable:
  - The thesis' `LaTeX` code can be used with any `LaTeX` engine. If none is available, a PDF output is provided on [arXiv](TODO), [MURAL](TODO), as well as here on `GitHub`.
  - The `Mathematica` notebooks are provided here. These are not per se 'interoperable', since the closed source software `Mathematica` must be used to read them. To mitigate this shortcoming, PDF prints of the evaluated notebooks are provided here.
  - The `CLASS` source code can be compiled with any `C` compiler.
  - The `Cobaya` related setup files can be used with the open access tool [`Cobaya`](https://cobaya.readthedocs.io/en/latest/).
  - The `Cobaya` chains can be analyzed with various tools. We used [`GetDist`](https://getdist.readthedocs.io/en/latest/) through `Python`.
- **R**eusable:
  - The `LaTeX` code can act as a template for other theses. The equations can be used for other papers.
  - The `Mathematica` notebooks can be used to derive the perturbed equations of motions for other systems with only slight modifications, in particular setting the corresponding Lagrangian respectively action terms.
  - The `CLASS` code can easily be extended by implementing additional types of dark matter or dark energy potentials. Also, except for dark energy related changes, no breaking changes were implemented compared to the official `CLASS` code. Therefore, our code is perfectly compatible and suitable for extended evaluations of other cosmological models that are already implemented in the official `CLASS`.
  - The MCMC chains can be analysed with other tools to do additional statistics on them.
 
# AI Statement
No LLM has been used to write my thesis, to design my research, or to derive the equations. The `GitHub Copilot Chat` has been used in `VS Code` to write the `Python` helper scripts and `git` commit messages. 
 
# Carbon Footprint
Most computations have been performed on a hydro-powered supercomputer. The energy consumptions for all other involved devices is offset at [climeworks](https://climeworks.com/checkout/referral/AqO8j10d). Therefore, I consider my PhD as _probably_ carbon-neutral.

# Collaboration
Please reach out to me if you would like to collaborate on a similar project, or if you find bugs in my code!
