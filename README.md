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
The source code in this repository is a fork of the Cosmic Linear Anisotropy Solving System (CLASS) v3.3.3 by Julien Lesgourgues, Thomas Tram, Nils Schoeneberg et al., which can be pulled from https://github.com/lesgourg/class_public.

<details>
<summary>Technical implementation of the cosmological model</summary>

### Notable changes
To implement 

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

## Mathematica
The Mathematica notebooks contain the derivation of the governing equations, equations of motions, the potentials and their derivatives, as well as additional tests for swampland-compatibility.

Please reach out to me if you would like to collaborate on a similar project, or if you find bugs in my code!
