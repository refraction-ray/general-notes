# Autoencoder in phase transition
*Start writing from 28 Feb, 2018*

*Possible project tracking*

* toc
{:toc}

## Relevant models of many-body physics
### Heisenberg model for spin one

#### Todo list

- [ ] The critical value of $$S_z$$ term
- [ ] How to calculate the topo number from the finite size wave function
- [ ] Or are there any other way to determine the critical point utilizing the wave function and spectrum 

#### Hamiltonian terms

* biquadratic term: $$(\vec{S_i}\cdot\vec{S_j})^2$$ is not necessary to realize Haldane phase (SPT). But this term with coefficient $$1/3$$, make it a solvable model as AKLT limit. (We always treat the prefactor of Hersenberg terms as unity) For different coefficient of the biqudratic terms, see phase diagram here [so](https://physics.stackexchange.com/questions/215299/what-is-the-importance-of-the-biquadratic-interaction-in-the-aklt-model).
* if biquadratic term is zero, the critical value of $$S_z^2$$ term against Heisenberg term is around one for topological phase transition. see [ref](https://arxiv.org/pdf/1710.03105.pdf).

### Kitaev Chain Model 

#### Todo list

- [ ] Find the suitable Hamiltonian in 1D with Z topo classification and easy to carry out ED

## AE on the wavefunctions
### Todo list
- [ ] Try all kinds of models with phase transition, check the difference behaviors in the latent variable. For example, model with Z/Z_n topo transition,  with conventional second order phase transition, first order phase transition, X-Y phase transition, thermal-MBL transition,  FIQCP, etc.
- [ ] how to make the NN more robust, namely, it can determine topo (non)trivial states which is not the eigen states of training Hamiltonian
- [ ] finite size scaling on relevant model by using the latent variable, to locate the phase transition point
- [ ] … variant length AE? combine AE and RNN?
### Model Results
#### $Z_2$ order

Extended ALKT model (tuning $$j$$):
$$H=\sum_{ij}\vec{S_i}\cdot\vec{S_j}+u (\vec{S_i}\cdot\vec{S_j})^2+\sum_i j (S_{zi})^2$$

* $$u=\frac{1}{3}$$. size 6. 1d latent space: give two seperate constant value, with a clear jump, critical value around 0.6
* $$u=0$$. size 6. 1d latent space: also give two seperate constants, critical value around 0.435. different cell numbers in the hidden layer have no effect on the position of critical value
* $$u=0$$. size 6. 2d latent space: one similar to the above and the other one is constant all the time (redundancy). The critical value is the same $$j_c=0.435$$. *But this value is away from the numerical value around 1*
* $$u=0$$. size 4 for constant variable (local minimum actually); size 5 no good results. The only way to see other size is to increase the size rather than decreasing size.
* $$u=0$$. size 8. may lead to **continuos** change for latent variable (for dataset range from j=0 to j =3)

#### Conventional second order phase transition

Tranverse Ising model in 1D (tuning $$j$$): 

$$H = \sum_{ij}S_{zi}S_{zj}+j \sum_i S_{xi}$$

Special note: remeber to add a small symmtry breaking term $$\Delta S_{zi}$$ to make the ground state unique.

* 1d latent space: Continuos variant to indicate the order paramter (maybe a finite size scaling can be carried out)

## Other ML approaches and comparisons
### Todo list​

- [ ] PCA analysis and comparison with AE 