# brief report on March Meeting 2019

*from Mar 18, 2019, prepared for group meeting*

## R25.00014

Hilbert space properties of the many-body localization problem: from full ergodicity to multifractality. arxiv: 1812.10283, **Nicolas Macé**, Fabien Alet, Nicolas Laflorencie.

The participation entropy is defined as 

$$
S_q=\frac{1}{1-q}\ln \left(\sum_{\alpha=1}^N \vert \psi_\alpha\vert^{2q}\right)
$$

q to 1 limit (Shannon limit) gives $$S_1=-\sum_{\alpha}\vert \psi_\alpha\vert\ln \vert\psi_\alpha\vert^2$$. q=2 gives IPR, $$S_2=-\ln IPR$$.

The scaling law is $$S_q=D_q\ln N$$, D=1 for delocalized system while D=0 for localized system. The D value is jump drop down on ETH-MBL transitions by finite size fit from ED results.

Recast the many body version XXZ model to Fock space Anderson model.

Spin basis: sz configuration in zero sector. (without xy coupling) Fock basis: basis diagonalize free fermion part (without zz coupling). ED calculation. S1/ln N is crossing near MBL criticality. 

Employed so called shift invert ED, size up to 24! See [here](https://scipost.org/SciPostPhys.5.5.045/pdf) for details of the numerical methods.

Conclusion: $D_{q,MBL}\propto 1/h$, multifractal behavior across the phase instead of in the phase boundary.

Take home message: how about define localization length in Fock lattice, it may suffer from finite size effect less. How about construt graph with Anderson localization directly representing behavior of MBL?

## R25.00013

Stability of quasiperiodic chains to quantum avalanches, in preparation, **Anushya Chandran** and Philip Crowley. 

Highly similar with our ongoing work. As quoted in the abstract: 

> Our work identifies the first qualitative difference between random and quasi-periodic localization and suggests new experimental tests of avalanche instabilities.


Quasiperiodic potentials have no rare region may be more stable against thermal coupling. (*which I dont think is a reasonable argument*)

Setup: thermal bath (GOE random matrix) (6 sites fixed)+ 1D AA model (no interaction within AA chain), observable: half-chain EE. Tool: ED up to size 12.

No shift of critical point for short ranged coupling for case where the thermal bath only coupled to the outmost 2 sites in AA chain. Shift of critical point if thermal bath is coupled to the AA chain with exponential decay coupling. Bulk coupling is also considered (thermal bath within the chain), also critical crossing shift (avalanche).

## R25.00005

Many Body Localization Transition in Systems with Correlated Disorder, in preparation, **Rajdeep Sensarma**, Abhisek Samanta, Ahana Chakraborty.

Only take home message: one can setup the model with potential on long ranged correlation (say multivariate Guassian distribution controlled by several parameters) while still random instead of quasiperiodic. So it is a very important model setup to see the difference root between random and QP models.

