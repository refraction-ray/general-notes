*Start writing from Mar 15, 2018*
*Only highlight points instead of the full picture*

* toc
{:toc}
# Canonical Quantization

## Free field

* Lorentz invariant of the delta function

  We choose the convention where for boson field $$[a(\vec{k_1}),a^\dagger (\vec{k_2})] = 2k^0 (2\pi)^3\delta^3(\vec{k_1}-\vec{k_2})$$ and the convention for normalization as $$\langle \vec{k}'\vert \vec{k}\rangle = 2k^0 (2\pi)^3 \delta^3(\vec{k}-\vec{k}')$$. The reason is that we can show $$k^0\delta^3$$ as a whole is Lorentz invariant.
  $$
  \begin{aligned}&  k_0\delta^3(\vec{k}) = k_0\delta^3(\vec{k'})\frac{d\vec{k'}}{d\vec{k}}=k_0\delta^3(\vec{k'})\frac{dk'_3}{dk_3}\\=&\delta^3(\vec{k'})\gamma(k_0+\beta k_0\frac{d \sqrt{\vec{k}^2+m^2}}{d k_3})= \delta^3(\vec{k'})k'\end{aligned}
  $$

* The frequently used relation in spectrum of propagators (proof using contour integral)
  $$
  \theta(\lambda)=\frac{i}{2\pi}\int_{-\infty}^{+\infty}d \omega \frac{e^{-i \omega\lambda}}{\omega +\epsilon}~,~~\epsilon>0.
  $$
  The introduction of $$\epsilon$$ is of physical meaning: equivalent to replace the mass as $$m^2\rightarrow m^2-i \epsilon$$ in $$L$$. In interacting case, $$\epsilon$$ is propotional to the width of decay.

* The propagator definition of fermion 
  $$
  \langle 0\vert T(\psi_\alpha(x)\bar{\psi}_\beta(y))\vert 0\rangle=\theta (x^0-y^0)\alpha\bar{\beta}-\theta(y^0-x^0)\bar{\beta}\alpha.
  $$
  Note the minus sign.

* Quantization of mass vector field (spin -1)

  * Requirement to reduce rudundancy freedom: $$\pi^0=\frac{\partial L}{\partial \dot{A^0}}=0$$.

  * The momentum $$\pi^i=E^i=\partial^iA^0-\partial^0 A^i$$.

  * The relation for polorization
    $$
    \sum_{\lambda=-1,0,1}\epsilon^\mu(k,\lambda)\epsilon^\nu(k,\lambda)=-g^{\mu\nu}+\frac{k^\mu k^\nu}{m^2}.
    $$

  * Express $$A^0$$ as $$A^0=\frac{1}{m^2}(J^0-\nabla \mathbf{E})$$.

  * EOM: $$\partial_\nu F^{\nu\mu}+m^2 A^\mu =J^\mu$$. 

## Interacting field and Feynman rules

* Logic flow of interacting field theory derivation

```mermaid
graph TD
A(Lagrangian) -->|impose commutation relation|B(canonical quantization)
    B --> |plane wave expansion|C(correlation of free fields)
    C --> |interaction picture of time evolution: Peskin 4.31 /connected diagrams/|D(correlation of interacting fields)
    D --> |LSZ reduction formula  /fully connected, amputated diagrams/| E(S matrix)
    E --> |scattering theory: Peskin 4.79 4.86| F(Decay rates or cross section)
    
```
* $$S=1+i T = 1+(2\pi)^4\delta^4(\Delta \vec{k})i M$$,  the common procedure for Feynman diagrams are designed for the calculation of amplitude $$i M$$, something between S matrix and multi-field correlation functions of interaction field theory.

* Optical theorem: due to the unitary (conservation of probability flow) of the scattering matrix S, we have

  $$-i (T-T^\dagger)=2Im\; T=T^\dagger T.$$

* Field in interaction picture is free field in Heisenberg picture in some sense (you can understand all fields in this framework as Heisenberg field).

# Problems

* Lecture Notes A31: where does the non-covariant term come from in the two point correlation function of massive vector fields? (seems there is no such term if plane wave expansion is directly applied?)

# Reference

* A solution to Peskin's QFT by Z-Z Xianyu: [pdf](https://zzxianyu.files.wordpress.com/2017/01/peskin_problems.pdf)