# More on MBL by QP potential

*from Nov 2018*

## TODO

- [x] check the relationships the decay length with varying wavevectors (not so clear now, seems not so relevant)
- [x] see the distribution of mix type potential
- [x] simulate the subband presudo random model to depict the root of the different universality classes
- [ ] what about all kinds of random hopping model or qp hopping model
- [ ] behavior in deep thermal and deep MBL phase: possible exploration on the MBL phase boundary
- [x] smooth bin for the distribution counting
- [ ] possible experiments relevance

## Experiments

* Very convincing fit for power law and exponential for two types of MBL phases.
* Thermal peak decays when one approaching deeply into the MBL phases.
* The estimated tri-critical point lying between W_qp = 2.2 to 2.3, the localization length is utilized in a parallel form.
* For W_qp=2.2, we see power law distribution of thermal block which is consitent with the weak MBL in the pure random limit.
* Test two band pseudo random model, very amazing crossing quality, and $\nu\approx 2.5$ near to the qp criticality. Meanwhile, for deep in the MBL phase of such model, exponential decay of max thermal length is observed.
* Test two band model with 0.5 transfer probability. It seems that we can indeed approximate critical exponent in this case as the same as random case. But the distribution is more smooth though take a similar behavior as small transfer probability case.
* For wave vector $\alpha=\sqrt{2}/2$, the critical value seems to be the same, inconsistent with the work by ED 1901.06971.
* For edge thermal bath inclusion, only max thermal block exhibit expected behavior while EE always show phase transition. Only linear scaling bath shows no phase transition. For all other cases, the critical value seems similar or even smaller.

## Pictures proposed

**Strong MBL vs. Weak MBL**: The different universality classes is the result of different MBL phases. MBL phases are in different types partially characterized by different scaling behavior of Griffith region. The distribution of max thermal block is power law in random-MBL while it is of form as exponetial decay in qp-MBL. So the qp-MBL is strong MBL which is more stable in some sense. 

The power is aroud 2 near criticality of random-MBL which meets the previous work. The reason of the different scaling law might be rooted in the so called **NN probability gap** in qp system. In such system, the energy differences between nearest localized single particle wavefunctions show a gap (though in the probability sense). Besides, we precisely locate the tricritical point between the two MBL phase and thermal phase, which is a strong support of our previous claim that qp-MBL criticality is stable under small randomness. This stableness can be justified by the non-vanishing of probability NN gap when small randomness is included. 

The weak MBL may be the one refered as MBL star.

Experiment relevance: small nu for faster decay, consitent with the strong MBL picture

Possible arguments for stongness: robust against thermal bath inclusion as the thermal process is totally independent reflected by the exponential probability

*Tittle:* Strong and weak many-body localization 

## Issue

* missing count for size 600: py script modification

## Further experiments on more general settings

* Results on different wavevectors

  Ref: arXiv:1901.06971, typical wavevector in their paper: $$\sqrt{2}/2,\sqrt{2}$$. ($$\sqrt{2}/2$$ is the most different one)

* Results on different interaction V

* Random hopping model (no well defined localization length?)

* QP hopping model

* Sum of different cosine potential

* Exactly solvable potential for mobility edge

  Ref: PRL 114, 146601 (2015)

  $$V_n=2\lambda \frac{\cos(2\pi nb+\phi)}{1-\alpha \cos(2\pi nb+\phi)}$$, $$\alpha\in (-1,1)$$, $$\alpha=0$$ is AA model limit. 

### New results

* Weirdly, wavevector smaller than 1 gives a shift of critical value towards larger value, and there is a three stage curve. But it is hard to differetiate them in chemical potential distribution. (spectrum after finite hopping term?)
* The universal property of double qp wave case seems to be close to random criticality. This may be attribute to that there is no nned in the two way case and actually the system nearly has no gap when multiple frequency wave is enabled.

## Background

* Time evolution of EE for a prepared initial state: thermal phase: power law increase to saturate; r-MBL: (PRL 109, 017202 (2012)) log increase in slow time (rapid increase in short time), with saturation of smaller value in finite system though predicted infinite increase for infinite system (the saturation seems irelevant of how ddep in MBL); Anderson localization, saturate in slow time. qp-MBL: (PRB 96, 075146 (2017)), log increase in slow time, too. As a comparison, it is reported (PRB 85, 094417(2012)) ee increase as loglogt at the generic critical point of random field trasverse ising chain.
* Many-body mobility edge: a large scale of ED: Phys. Rev. B **91**, 081103(R), also KL divergence and Hilbert space localization are studied an indicators of MBL.
* The slow dynamics of 1D qp experiments, is reported as a power law decay with time (PRL 119, 260401 (2017)). Actually the results is not so determinant, and there leaves possibility of exponential decay for slow time dynamics, in the supplemental material of this prl, there is also discussions on exponential decay fit. Non interacting version shows slower decay when approaching localized phase but with strong oscillation.  There may be an offset $\alpha_0$ in experiments for the decay power. Time evolution tools: "using a parallel solver as implemented in the SLEPc library".
* science.aaa7432, supp: formula for non-interacting imbalance evolving. stationary imbalance is expected as $I(\infty)\propto (w-w_c)^\eta$.
* time evolving of r-MBL,  Griffith region in thermal side, $S\propto t^{1/z}$, $I\propto t^{-\chi}$. Krylov space time evolution.
* [1812.10283](https://arxiv.org/pdf/1812.10283.pdf): formula on fractal dimensions
* Interesting quatities: typical DOS $$\rho(E)=\exp(\frac{1}{L}\sum_{i=1}^L\log \rho_i(E))$$. This order parameter is finite in the delocalized phase, zero in the localized phase, and goes to zero at the transition. This value can be calculated by kernel polynomial method efficiently. For details on this ,see PRL 114, 146601 (2015). This quantity is claimed to be easily generalize to many-body version.

## More MBL paper reading

### 1906.10701

Compute correlator in random Heisenberg model, including SzSz and S+S-. They found that the distribution of such correlator is deviating from Gaussian in thermal phase by charaterizing KL divergence (interesting to further understand why there is a deviation peak in thermal phase instead of MBL phase), which may be the reason for rare thermal region and Griffith effect. Beside, they apply pertubation theory to argue some features in their results distribution. It may also be interesting to note that there are two drop near center peak for spin flip correlator in strong MBL phase.

## Future work

- [x] couple to the thermal bath in RSRG method(fixed size vs. scaling size)
- [ ] critical behavior of random and quasiperiodic hopping model
- [x] try more on different wavevectors according to the paper: arxiv: 1901.06971: no qulitative change
- [ ] MBL phase boundary scaling form near AL critical point
- [ ] SSH model with quasiperiodic potential (topology enhanced localization or qp induced quantum scar?)
- [ ] Model with two different QP potentials
- [ ] Asymmetry of universality on MBL criticality two sides
- [ ] EE spectrum distribution, power law vs. expoenential decay?