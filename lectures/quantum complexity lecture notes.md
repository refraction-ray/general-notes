# T. Cubit: Quantum Complexity Theory

[TOC]

*A quick note on lecture given by T. Cubit's summer school lecture on topics of computation theory in physics, see ref [1]*

## Complexity and computability

### Classical part

*most omitted in classical computations*

Rice Thm: Any nontrivial property of partial functions (languages) is undecidable. (Judge whether a TM is in some set of TM)

P/poly: defined via circuit, halting problem in in P/poly, no information on how to construct the circuit

Uhalt is in nonuniform P, though undecidable, uhalt is halting problem with unary encode, somewhat weird complexity class

? one can prove not every combinatoric problem can turn into a decision problem, but in practice you can.

uniform circuit family: there is TM, can design and construct circuits C_n in polytime.

### Quantum counterpart

BQP: there exit uniform polytime quantum circuits such that the output is 1 is larger than 2/3 for yes.

QMA (quantum Merlin-Arthur): there exist polytime uniform quantum verifier circiuts U, such that answer (classical) input x: YES,  $Pr(U \vert x\rangle \vert w\rangle ~output ~1)\geq 2/3$. The answer $\omega$ is poly(x) qubits.

$P\subseteq BQP$, $NP\subseteq QMA$

P vs. NP, P(BPP) vs. BQP (the very foundation of the usefulness of quantum computation)

BQP vs. QMA, NP vs. QMA

## Complexity in Physics

### Local Hamiltonian

$H=(C^d)^{\otimes n}$, local interaction, acts nontrivially only on k qubits, we call it k-local Hamiltonian if all terms including no more than k qubits.

Local Hamiltonian problem: k-local H on n qubits with m terms as input, output yes if the lowest eigenvalue is lower than $\alpha$ or no if it is higher than $\beta$.

Promise: $\beta-\alpha =\Omega(1/poly(n))$ (avoid real number issuses)

The input length $C_n^k d^{2k}$, the problem size is poly(n).

[Kitaev] local hamiltonian problem is QMA-hard(complete).

### Proof

Idea: encode the evolution of u in superposition in ground state (due to Feynman).

$\vert\Psi\rangle = \sum_{t=0}^T \vert \Psi_t\rangle\vert t\rangle$. computational register and clock memory, history state (*somewhat like state ensemble correspondence?*).

$H=I-\vert \Psi\rangle\langle \Psi\vert$. trivial one but not local.

$H=H_{in}+H_{prop}+H_{out}$ , Hilbert space $\mathcal{H}=C^2\otimes (C^2)^{\otimes \omega}\otimes (C^2)^{\otimes A}\otimes C^{T+1}$. output, witness, ancillas

the ground state of the constructed Hamiltonian lokks like a computation history, and related eigenvalue to the original problem.

To prove the YES instance can be reduced to k-local, we use $\langle\psi\vert H\vert\psi\rangle$, where $\psi$ is the state encoding computation history. We show that if $Pr(U\psi~ output ~no)\leq \epsilon$ then such energy is smaller than epsilon. For the opposite case, use geometric lemma to restrict the lower bound of the matrix sum, which show the reducibility of NO instance.

Contradiction with experiment: how about we construct such Hamiltonian in experiments and cool it down to measure the groud state energy. That would mean we can solve QMA hard problem quickly. The only way to get out of this is to assume we actually spend exponential time to cool down the system.

## Computability in physics

Undecidability means infinity hiding somewhere. 

Example 1 (Undecidability of Halting)

Example 2 Particle dynamics in 3D [Chris Moore 1990, 1991]

Smale horse shoe map. Baker's map. 

More general map on bi-infinite binary strings.

Lemma: any generalized shfit map is equivalent to a piecewise linear map in unit square.

Lemma: generalized shift map is Turing complete. So there is universal generalized shift map.

halting problem is reachability in generalized shift map.

3D box contain finite number of planar or parabolic surface. Shoot the ball from one hole of the box, does the ball get out from another hole? Undecidable

undecidability in 3D smooth potential of a classical single particle.

The example is related with butterfly effect, the hallmark of chaotic system. Therefore, we cannot predict the system with any precision. The undecidability is restricted by the finite precision of the input hole. 

Example 3 spectrum gap of infinite size 2D quantum system. [Nature 2015]

## Reference

1. http://www.dr-qubit.org/Boulder.html