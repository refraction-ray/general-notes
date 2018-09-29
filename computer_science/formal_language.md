# FORMAL LANG AND AUTOMATON

[TOC]

## Grammar

### definition

G = (V,T,P,S), V variable, T terminal, P production, S start symbol in V.

derivation vs. reduction

sentence: $T^{*}$, sentential form: $(T\cup V)^{*}$

all sentence is a language L(G)

### classification

1) type 0 grammar (all G) 

2) context sensitive grammar (CSG), type I, where right is no shorter than left in all production rules  

3) context free grammar (CFG), type II, CSG and all left part pf production is in V

4) regular grammar (RG), type III, CFG where all production rules are like A: wB or A: w, w is in $T^+$.

Thm: L is RL, iff G production rules in the pattern A: a or A: aB, where a is in T.

Linear grammar: A: w; A: wBx

right linear grammar: A: w; A: wB, the similar definition for left linear language and grammar.

Thm: left linear is equavilent to right linear grammar.

Thm: every grammar has equivalent grammar, where start S only in the left part of production rules.

All type can take empty sentence $\varepsilon$, with no difference in classfication.

## Finite automaton

### definition

$M = (Q, \Sigma, \delta, q_0,F)$. Q finite state set, q in Q is a state of M. Sigma input alphabet. delta: transition function, Q,Sigma: Q. q0 initial state, F final state set.

One can generalize the definition to $\delta'$ which is defined on Q times $\Sigma^{*}$.

comment: somehow like Markov type things, with no memory.

Language recognized by M, final state is in final state.

transition diagram.

instantaneous description (ID): insert current state into the string where pointer is located.

set(q_i): all elements in $\Sigma^*$ whose end state is q_i. this is a equivalent class partition.

{0^n1^n} is not a language can be accepted by FA, since the state is infinite.

### NFA

non-determinstic finite automaton:

$\delta$ is now a function from $Q\times \Sigma$ to $2^Q$ instead of Q, namely, for every step, it may has many state transfer. And as long as the final state has crossover with final state, it is said to be accepted. Somewhat behave as quantum stuff since the state at any moment is distributed in the whole state set.

Thm: NFA is equivalent with DFA.  Proof: let $Q'=2^Q$, obvious. (of course there may be inaccessible state in Q')

### epsilon-NFA

non-determinstic finite automaton with $\varepsilon$ - moves. $\delta$ is from $Q\times (\Sigma\cup\{\varepsilon\})$ to $2^Q$. Namely one can do a move without read next chracter.

Thm: e-NFA is equivalent with NFA.

Thm: FA is equivalent with RL.

## Regular expression

Regular expression can be defined recursively. Basic case $\empty, \{\varepsilon\}, a\in \Sigma$; induction case: $+$ the union set, $\times$ concatenation, $*$ closure.

Thm: regular expression is equivalent with FA.

## Properties of RL

### Pumping lemma

let L be a RL, there is a constant N of L, for any string z in L, if |z|>N, there must be u,v,w substring satisfying

1. z=uvw, 2. |uv|<N, 3. |v| non empty, 4. for any non integer i, uv^iw also in L.

### Closure property of RL

union, complementary, closure, product, intersection is closed on RL. 

### Myhill-Nerode Thm

equivalent relation $R_M$: defined based on specified automaton, strings reach the same state in M is the same class. $R_L$: defined based on language, any same string attached to the two both in or out of L. $R_M$ is a finer classification on $R_L$, and it actually corresponds minimal states of automaton required by the language L. In other words, if for some lang L, the no of equivalent classes is infinite, then L cannot be RL.

## CFL

### derivation tree

the tree with all leaves as terminal symbol while all non-leaf nodes non-terminal symbol. the yield is the leaves from left to right. similar definition can be applied to derivation subtree.

leftmost vs rightmost derivation

ambiguity for grammar (two different derivation tree for at least on yield), and inherent ambiguity for language. (non ambiguity language can also have ambiguity grammar.)

### simplification

* useless symbol (terminal or non-terminal symbols): two parts: someone can derive it (traceback from S), some sentence can be derived from it (traceback from terminal rules)
* epsilon production: $A\rightarrow X_1…X_m$ in P, if $X_i \in U$ where U is the reduced-to-null set, then add the rule in new P, but both with X in U and X as null.
* unit production or chain production: $A\rightarrow B$. 

### Chomsky normal form

$A\rightarrow T\vert VV$. Any CFG can transform as CNF

### Greibach normal form

$A\rightarrow TV^*$. Any CFG can transform as GNF. (left recursive transform technic)

## Pushdown automaton

PDA is $M=(Q,\Sigma,\Gamma,\delta,q_0,Z_0,F)$, Gamma is stack alphabet, Z0 is the only symbol in stack when start from state q0. this time the delta function pop the top symbol on the stack every time and push a series of symbols into the stack. The PDA can also define in the non derteminstic fashion and null movement. 

There are two way to define lang accepted by M, usual way like FDA, defined as L(M) which is strings drive M to final state F. And N(M) which is languages make the stack empty finally (if empty in the middle, the at least this banch goes to error state in the sense of NPDA). It can be shown, for any M you can find another M' that N(M')=L(M) and vice versa.

Thm: CFL is equivalent with PDA. (the pda just mimic the left most derivation of CFG) 

## Properties of CFL

### pumping lemma

CFL pumping lemma: there is N only dependent on the CFL L, for arbitrary z in L, if |z|>N, there is z=uvwxy, such that:

1. |vwx|<=N 2. |vx|>=1 3. uv^iwx^iy is still in L for non negative i

This lemma can be shown by careful considering of the derivation tree of CFG with two nodes labeled by the same non terminal close to the leaves.

The strong version: Ogden lemma: N refer for some certain terminal symbols. (make some proof easy.)

### Closure

CFL is closed under union, product, and closure computation.

However CFL is not closed under intersection and complementary.

Algorithm: dynamic programming on sentence judge in CFL: construct $V_{i,k}$ which is all non-terminals which can derive the sub string of sentence from the i pos and k length.

## Turing Machine

### Definition

Turing machine: $M=(Q,\Sigma,\Gamma,\delta,q_0,B,F)$.

the alphabet is a subset of tape symbol $\Gamma$, B is the blank symbol in Gamma. delta now is $Q\times\Gamma\rightarrow Q\times \Gamma\times \{R,L\}$. One can define the non-determinstic version of Turing machine, too.

The difference between TM and FDA is that, once the state reach to F, the machine halt and give the result instead of further checking like FDA and the possibility of leaving final state later. For non-deter version of any machine, it behaves like branches and error brach dont lead to error, but success branch lead to success. It is like a OR attempt.

All reasonable extension of turing machine you can imagine is equivalent to basic Turing machine in the sense of computability (multiple reading , multiple dimensions, multilple tapes, etc).  But they can be different in the sense of complexity. Say, nonderterminstic Turing machine is the same as basic Turing machine in terms of what can be computed. But they are different in the complexity hierarchy, unless P=NP.

### Computability

Church-Turing thesis: any problem that can be solved by 'effective algorithm' can be solved by one TM.

Not proved, just a belief, as 'algorithm' is not something well defined.

If encode everything into 0 and 1. Then a turing machine (program) can be encoded as an integer, the input is also an interger (data). Treat data and program equal footing as 0,1, which is just what is happening in modern computers. Every TM and every input data can be indexed by integer now. Diagonal arguments come to play a role. Considering L={w| w is the j-th data and the j-th TM dont accept it}. Such a lang cannot be accept by any TM. Moreover, L={Mw| M accept w} is called universal lang, which is r.e thought not r. (namely, all elements in L can be accept by a TM N in finite time, but non-lang part may trigger infinite loop)

### Lang equivalent

Turing machine is equivalent with type 0 lang: there is lang beyongd type 0!
LBA(linear bounded automaton) is equivalent with type 1 lang. LBA is just a Turing machine with a restriction on the tape (with two boundaries).