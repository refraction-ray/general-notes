# THEORY OF COUMPUTATION IN A NUTSHELL

*This is just an outline of basic concepts in computation theory, it is designed as a quick intro at group meeting. It is not self-contained, and you still need to refer to some standard textbooks on this field.*

[TOC]

## Computability theory

*The emphasis part may be omitted, and we don't cover theories of formal language here at all*

**The bold part is the highlight and focus at group meeting**

* Definition of Turing Machine

* *Understanding Non-deterministic Turing Machine*

* Decision problems and accepted language

* Decidable vs. recognizable

* *More variants on Turing Machine and their equivalence (in terms of computability)*

* **Church - Turing Thesis**

* Universal Turing Machine

* A_TM = {\<M,w\>|M is a TM and M accepts w} is undecidable though recognizable, i.e. **halting problem**

  Proof: reverse the universal TM on \<M,\<M\>\> and input itself. i.e. the old wisdom: this line is wrong.

* The diagonalization method (with argument on real number is more than integer)

* **Algorithms as integer while problems as real**

* *Unrecognizable language (undecidable language or its complement)*

* *decidable iff recognizable and co-recognizable*

* **the concept of problem reduction, mapping reducibility**

* examples on reduction: 1. true halting problem, 2. equivalence between TM (from the zero acceptance of TM)

* *reduction from computation history*

* *undecidable 'real problems': 1. integer root of polynomial, 2. Post Correspondence Problem*

## Complexity theory

* Big O notation
* worst case vs. average case analysis
* *hierarchy: computability (computable vs. uncomputable), complexity (tractable vs. intractable), algorithm theory (n^3 vs. n^2.87)*
* class P
* *all deterministic computation model are polynomially equivalent*
* *space complexity*
* P vs. Pseudo P algorithm (input length instead of value)
* two definitions of NP
* P vs. NP
* **NP hard and NP completeness**
* **polynomial time mapping reducible**
* **Cook-Levin Thm**
* 3SAT is NPC
* **proof procedure to show NPC**
* *cryptography related: 1. no safety proof and P=NP, 2. worst case hard vs. average case hard*

## Guidelines for further reading

After learning some basic knowledge on the notation and language of computation theory, I recommend you a tour to the interplay between statistical physics and computation theories. There are a series of blogs on this topic below. I have finished three sections covering this topic right now.

1. 2D spin glass is in P: [无向图最大权重匹配算法](https://re-ra.xyz/%E6%97%A0%E5%90%91%E5%9B%BE%E7%9A%84%E6%9C%80%E5%A4%A7%E6%9D%83%E9%87%8D%E5%8C%B9%E9%85%8D%E7%AE%97%E6%B3%95/), [Pfaffian Graph vs Perfect Matching](https://re-ra.xyz/Pfaffian-Graph-vs-Perfect-Matching/), [二维自旋系统作为 P 问题](https://re-ra.xyz/%E4%BA%8C%E7%BB%B4%E8%87%AA%E6%97%8B%E7%B3%BB%E7%BB%9F%E4%BD%9C%E4%B8%BA-P-%E9%97%AE%E9%A2%98/).
2. 2D spin glass with magnetic field and 3D spin glass is NP-complete: [cubic graph 上的 NP 完全问题](https://re-ra.xyz/cubic-graph-%E4%B8%8A%E7%9A%84-NP-%E5%AE%8C%E5%85%A8%E9%97%AE%E9%A2%98/), [经典自旋系统的 NP 完全](https://re-ra.xyz/%E7%BB%8F%E5%85%B8%E8%87%AA%E6%97%8B%E7%B3%BB%E7%BB%9F%E7%9A%84-NP-%E5%AE%8C%E5%85%A8/).
3. More elaboration on the meaning of NP hard in physics: [ 计算复杂度理论概念备忘](https://re-ra.xyz/%E8%AE%A1%E7%AE%97%E5%A4%8D%E6%9D%82%E5%BA%A6%E7%90%86%E8%AE%BA%E6%A6%82%E5%BF%B5%E5%A4%87%E5%BF%98/), [NP 困难物理系统的内涵与外延](https://re-ra.xyz/NP-%E5%9B%B0%E9%9A%BE%E7%89%A9%E7%90%86%E7%B3%BB%E7%BB%9F%E7%9A%84%E5%86%85%E6%B6%B5%E4%B8%8E%E5%A4%96%E5%BB%B6/).

There might be more series on this fascinating interplay in the future. The topics might include undecidable problems in physics, more on quantum complexity. Stay tuned.


## Main Reference

* M. Sipser, Introduction to the theory of computation.