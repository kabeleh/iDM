# Dark Energy–Dark Matter Interactions
- Author: Kay Lehnert
- Publications: 
    - [Hitchhiker's Guide to the Swampland: The Cosmologist's Handbook to the string-theoretical Swampland Programme](https://arxiv.org/abs/2509.02632)
    - [PhD Thesis](url.com)

This repository contains the CLASS[^1], MontePython[^2], and Mathematica [^3] files for a swampland-inspired model of dark energy in the form of quintessence as a scalar field that is interacting with dynamical dark matter. The interactions and mathematical details of the model are all explained in my PhD thesis [^4].

[^1]: https://github.com/lesgourg/class_public
[^2]: https://github.com/brinckmann/montepython_public
[^3]: https://www.wolfram.com/mathematica/
[^4]: TBA

## CLASS
The source code in this repository is a fork of the Cosmic Linear Anisotropy Solving System ([CLASS](https://github.com/lesgourg/class_public)) v3.3.3 by Julien Lesgourgues, Thomas Tram, Nils Schoeneberg et al.

<details>
<summary>Technical implementation of the cosmological model</summary>
    
The source code itself is commented and changes by us are indicated by the initials `KBL` as a comment.
Please note the extensive [Wiki Page](https://github.com/kabeleh/iDM/wiki) that documents the notable code changes and explains the physical and mathematical considerations behind those changes. 
</details>

<details>
<summary>How to compile and run the code</summary>
    
1. Either download the archive of the source code or `git pull` this repository to make a local copy.
2. Navigate to the folder that contains the root of the CLASS source code (the one that contains the *.ini-files as well as the `Makefile`).
3. `make clean; make class -j` to compile the code.
4. Run the cosmological simulation with `./class iDM.ini`
</details>

If you want to use this code, please cite [CLASS II: Approximation schemes](http://arxiv.org/abs/1104.2933) as well as `my thesis`.

## MontePython
The Markov chain Monte Carlo (MCMC) runs to find the best-fit parameters of the model and compare it to $`\Lambda`$CDM were performed using [MontePython](https://github.com/brinckmann/montepython_public) v3.6.1 by Thejs Brinckmann, Benjamin Audren, et al.

This repository contains
- parameter-files
- best-fit values 
- covariance matrices

These files allow you to fully reproduce my findings:
- The parameter-files recreate the exact MCMC-pipeline, choosing the likelihoods and starting parameters.
- The best-fit values can be used to accelerate the reproduction of my MCMC chains. Those can be passed on to `MontePython` as a starting value, i.e. your MCMC will then start at the best-fit value right away.
- The covariance matrices can also be passed to `MontePython`, as initial guess. This also accelerates reproduction of my results.

The Markov chains themselves are too big to be stored on GitHub. You can find them at [Zenodo](TODO). This allows you to directly analyse the chains yourself.

The following naming scheme is applied to these files: 
```
Type of Dark Matter    -    Type of Dark Energy                                         -    Data Set                                           .    file-type
< CDM | H | I >        -    < pL | c | h | pNG | iPL | exp | SqE | Bean | DoublExp >    -    < plik | PPDESI | PlikPPDESI | CMB-SPA | Full >    .    < param | bestfit | covmat >
```
For example, the parameter file to test standard CDM with quintessence in the form of a simple exponential scalar field tested against Plank 2018 data is `CDM-exp-plik.param`.

The types of dark matter are explained in the [Wiki](https://github.com/kabeleh/iDM/wiki),
and so are the [types of dark energy](https://github.com/kabeleh/iDM/wiki/New-User-Input-Parameters#scalar-field).
The data sets are the following:
- plik: Planck 2018 (Plik),
- PPDESI: PantheonPlus + DESI DR2
- PlikPPDESI: Planck 2018 (Plik) + PantheonPlus + DESI DR2
- CMB-SPA: SPT-3G D1 + ACT DR6 + Planck (Plik)
- Full: CMB-SPA + PantheonPlus + DESI DR2

## Mathematica
The Mathematica notebooks contain the derivation of the governing equations, equations of motions, the potentials and their derivatives, as well as additional tests for swampland-compatibility.

- `xPert.nb` uses the [xAct](https://xact.es) package [xPert](https://xact.es/xPert/index.html) to derive metric-independent equations of motions from the Lagrangian and the stress–energy tensor. It includes first-order perturbations.
- `xPand.nb` takes those equations and expresses them with respect to a metric and simplifies the equations using the Newtonian gauge. This notebook uses the [xPand](https://www2.iap.fr/users/pitrou/xpand.htm) plugin.
- `rho_cdm.nb` computes the derivatives of the interacting dark matter mass $m\left(\phi\right)=1-\tanh\left(c\phi\right)$ and expresses them in trigonometric functions.

## Thesis
My PhD thesis can be found on [arXiv](TODO), [MURAL](TODO), as well as in the folder `Thesis`. This folder contains, besides the PDF, the `LaTeX` code. The code acts as a template for similar documents and makes the equations available for typesetting.

The thesis template is an adaptation of [TeXtured](https://github.com/jdujava/TeXtured) with inspiration taken from [Tony Zorman](https://tony-zorman.com/posts/phd-typesetting.html). It is compliant with Maynooth's [Doctoral Thesis Layout Recommendations](https://www.maynoothuniversity.ie/exams/postgraduate-information).


# FAIR Data Statement
Our research complies with FAIR data principles:
- **F**indable: All data is explained and made available on this `GitHub` repository. The naming schemes are explained in this `README`.
- **A**ccessible: All data is available on this GitHub repository, except for the Markov chains, which are available on [Zenodo](TODO). Furthermore, the thesis is available on [arXiv](TODO) and on the Maynooth University Research Archive Library ([MURAL](TODO))
- **I**nteroperable:
  - The thesis' `LaTeX` code can be used with any `LaTeX` engine. If none is available, a PDF output is provided on [arXiv](TODO), [MURAL](TODO), as well as here on `GitHub`.
  - The `Mathematica` notebooks are provided here. These are not per se 'interoperable', since the closed source software `Mathematica` must be used to read them. To mitigate this shortcoming, PDF prints of the evaluated notebooks are provided here.
  - The `CLASS` source code can be compiled with any `C` compiler.
  - The `MontePython` related setup files can be used with the open access tool [`MontePython`](https://github.com/brinckmann/montepython_public).
  - The `MontePython` chains can be analyzed with various tools. Besides open source alternatives, there is `Python` as well as `MATLAB` code for analysing the chains in the `CLASS` folder.
- **R**eusable:
  - The `LaTeX` code can act as a template for other theses. The equations can be used for other papers.
  - The `Mathematica` notebooks can be used to derive the perturbed equations of motions for other systems with only slight modifications, in particular setting the corresponding Lagrangian respectively action terms.
  - The `CLASS` code can easily be extended by implementing additional types of dark matter or dark energy potentials. Also, except for dark energy related changes, no breaking changes were implemented compared to the official `CLASS` code. Therefore, our code is perfectly compatible and suitable for extended evaluations of other cosmological models that are already implemented in the official `CLASS`.
  - The MCMC chains can be analysed with other tools to do additional statistics on them.
 
# Carbon Footprint
The energy consumptions for all involved devices is offset at [climeworks](https://climeworks.com/checkout/referral/AqO8j10d). Therefore, I consider my PhD as _probably_ carbon-neutral.

# Collaboration
Please reach out to me if you would like to collaborate on a similar project, or if you find bugs in my code!
