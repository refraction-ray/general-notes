# Primary Number Theory

*This note is a brief summary of important concepts in primary number theory which might be useful in computer science and quantum computation. Start writing from Nov 6 2018.*

[TOC]

## Fundamentals

### Divide

**Definition:** d divides n, written as $d\vert n$, if there exits an integer k, s.t. $n=dk$.

**Properties** of divide:

1. (Transitivity) If $a\vert b$, $b\vert c$, then $a\vert c$.
2. If $d\vert a$, $d\vert b$, then $d\vert xa+yb$, where x and y are intergers.

### Fundamental Theorem of Arithmetic

**Theorem 1:** Any integer a greater than 1 has a prime factorization of the form $a=\Pi_i p_i^{a_i}$, where $p_i$ are distinct prime numbers. And such prime factorization is unique.

**Proof:** 1. It is easy to show the existence of such factorization by induction and the definition of composite number. 2. The uniqueness of such factorization can be easily shown by Euclid's lemma. And there is also direct argument by contradiction, see [here](https://en.wikipedia.org/wiki/Fundamental_theorem_of_arithmetic#Elementary_proof_of_uniqueness).

**Comment:** (Complexity class of prime factorization problem) The prime factorization problem is neither known in P nor in NPC. It is conjectured this problem might be in NPI. There is an efficient algorithm for this problem in term of quantum computer, which is well-known Shor algorithm and hence the problem is in BQP.

**Euclid's Lemma:** If prime p, $p\vert ab$, then $p\vert a$ or $p\vert b$ at least one of them is true.

**Proof**: We prove this lemma without Bezout's identity. Say $p\nmid a$, since $p\vert ab$, there is integer m s.t. $mp=ab$, namely $p:a=b:m$. Since $p,a$ is coprime, then $\frac{p}{a}$ is the simplest fraction. And all other fractions of the same value take the form $$\frac{np}{na}$$. That is to say, $b=np$, namely $p\vert b$. QED.

## Modular Arithmetic

### Basic

The arithmetic of remainders, add, minus and times are easily defined. Such formulas are written with $\mod n$ in the end. The inverse operation of multiplication: an effective division can sometimes defined, which we will elaborate in detail below.

**properties of modular arithmetic:**

Suppose $a_i=b_i\mod n$, we have

* $a+k=b+k\mod n$
* $ka=kb\mod n$
* $a_1\pm a_2=b_1\pm b_2\mod n$
* $a_1a_2=b_1b_2\mod n$
* $a^k=b^k\mod n$

### Greatest Common Divisor

**Definition:** The greates common divisor of a and b, written as $\gcd (a,b)$, is the greatest common elements from the divisor lists of a and b.

**Comments:** (Naive complexity) Based on the definition of gcd, the problem of finding gcd of two numbers is not easier than prime factorization. We will show clever algorithms finding gcd below which is in P.

**Definition** of coprime: a,b are said to be coprime, if $\gcd (a,b)=1$.

### Representation Theorem for the gcd

**Theorem 2:** $\gcd (a,b)$ is the least positive integer that can be written as $ax+by$, where x and y are integers (not necessarily positive ones).

**Proof:** Let $s=ax+by$ be the least positive number. Since $\gcd (a,b)\vert a,b$, then $\gcd(a,b)\vert s$. Namely, $\gcd (a,b)<s$. On the other hand, we can show $s\vert a,b$, but among such numbers, gcd is the largest one, therefore, $s<\gcd (a,b)$. So $s=gcd(a,b)$.

To show $s\vert a$ by contradiction, suppose $s\nmid a$, then $a=ks+r$, recall the expression of s, we have $0<r=a(1-kx)+b(-ky)<s$, which is contradict with the fact than s is the least positive number of such form. The same argument applies to $s\vert b$. QED.

**Corollary 2.1:** If $c\vert a,b$, then $c\vert \gcd (a,b)$.

**Definition** of multiplicative inverse: An integer b is said to be the integer a's multiplicative inverse modulo n if $ab=1\mod n$.

**Corollary 2.2:** If there are integers x and y, s.t. $ax+by=1$, then $\gcd (a,b)=1$, namely, a is coprime to b.

**Proof:** Obvious from the representation theorem.

**Corollary 2.3:** Let n be integer greater than 1. An integer a has a multiplicative inverse modulo n iff $\gcd (a,n)=1$. 

**Proof:** Suppose $aa^{-1}=1\mod n$, then $aa^{-1}+n(-k)=1$. And hence $\gcd (a,n)=1$.

**Comment:** (the effective division) We define $a^{-1}$ as the number satisfying $aa^{-1}=1\mod n$, and such an operation plays the role of division in modular arithmetic. Note $a$ must be coprime to n s.t. the inverse is well defined, then the elements of arithetic are subset of $Z_n$. The inverse in unique in terms of modular arithmetic, suppose $ba=1\mod n , ca=1\mod n$, then $b=c \mod n$. Since $\gcd (x,n)=1$,  $n\vert x(b−c)$ leads to $n\vert (b−c)$.

## Euclid's algorithm

In this section, we will focus on the euclid's algorithm for finding gcd, including its correctness and complexity analysis. In practical, there can be faster implementation than Euclid's method which heavily make use of the shift operations, see [here](http://blog.jobbole.com/106315/) for the optimal algorithm in implementation.

### Theorem for the equivalence of gcd

**Theorem 3:** Let r be the remainder when a is divided by b (a>b), then $\gcd (a,b)=\gcd (b,r)$.

**Proof:** Since $r=a-kb$ and $\gcd (a,b)\vert a,b$, we have $\gcd (a,b)\vert b,r$. By corollary 2.1, we have $lhs\vert rhs$.

On the other hand, $a=kb+r$, we have $\gcd(b,r)\vert b,a$, then $rhs\vert lhs$. QED.

### The algorithm finding gcd

**Euclid's Algorithm:** Iteratively replace the larger number with the remainder, until the remainder is zero. Then the divider (smaller number) in the last round is the gcd. The correctness of the algorithm is ensured by theorem 3.

**Comment:** (Algorithm finding the representation of gcd) As a byproduct, this algorithm can also be used to decompose gcd into standard representation as in theorem 2. Namely, find integer x,y, s.t. $ax+by=\gcd (a,b)$. This is done be go into the reverse direction on Euclid's algorithm, which take the larger and larger numbers into the expression, until we are back with the two numbers a and b, and the prefactor of them are x and y.

### Complexity analysis

Suppose a, b can be represented by L bit strings. Namely $a,b<2^L$. The key observation is that during the algorithm $r_{i+2}\leq r_{i}/2$, where $r_i$ is the remainder of i round. (Since $r_i=r_{i+1}+r_{i+2}$ if $r_{i+1}>r_i/2$). Therefore, we need only $2\log \min(a,b)=O(L)$ times of iterations. If we further consider the complexity of arithmetic operations in each iteration, the overall complexity is $O(L^3)$. And the same complexity applies to the problem of finding x and y of the representation for gcd.

### Finding multiplicative inverse

Euclid's Algorithm can also be used to find the modular inverse. This idea is directly from corollary 2.3. Since $aa^{-1}+n(-k)=1$, we only need to run Euclid's algorithm to find the representation of $\gcd (a,n)=1$. The corresponding coefficient is $a^{-1}$.

Now we are equiped with algorithms to solve linear equation in modular arithmetic as $ax+b=c\mod n$, where a and n are coprime. The solution is $x=a^{-1}(c-b)\mod n$, and can be solved efficiently.

## From Chinese remainder theorem to Fermat's little theorem

### Chinese remainder theorem

**Theorem 4.1:** Suppose $m_1…m_n$ are positive integers s.t. any pair of them are coprime. Then the system of equations 
$$
\begin{align}
x=&a_1\mod m_1\\
x=&a_2\mod m_2\\
...\\
x=&a_n \mod m_n
\end{align}
$$
has a solution. Moreover, any two solutions are equal modulo $M\equiv m_1m_2…m_n$.

**Proof:** Construct the solution explicitly. Define $M_i\equiv M/m_i$. Note $\gcd(m_i,M_i)=1$, we have $M_iN_i=1\mod m_i$. Then the solution is  $x = \sum_i a_i M_iN_i$. This can be verified noting the fact $M_iN_i=\delta_{ij}\mod m_j$.

Suppose x and x' are two different solutions to the system of equations. Then $x-x'=0\mod m_i$ and thus $x-x'=0\mod M$. QED.

**Comment:** (existence of a small solution) Based on the proof, there must be a solution $x<M$. If the explicitly constructed solution $x'>M$, we use the solution $x=x'\mod M$.

**Comment:** (deep structure of this thm) In fact, Chinese remainder theorem is deeply correlated with the rings decomposition and hence play a central role in the proof of primary number theory. That is to say, if we take a high level view from advanced algebra, most of the conclusions and progress from primary number theory are correlated with the decomposition of $Z/nZ$ rings.

### Fermat's little theorem

**Lemma 4.1:** Suppose p is prime and $1\leq k\leq p-1$, then $p\vert C_p^k$.

**Proof:** Consider $p(p-1)…(p-k+1)=C_p^k k(k-1)…1$. Since $lhs\vert p$, then $rhs\vert p$. However $p>k$, so $p\vert C_p^k$.

**Theorem 4.2:** (Fermat's little theorem) Suppose p is prime, then for any integer a: $a^p=a\mod p$. Moreover, if $p\nmid a$, then $a^{p-1}\mod p$.

**Proof**: The second half is easy. Since $p\nmid a$ means $\gcd(p,a)=1$. Then $a^{-1}\mod p$ is well defined. We have $a^{p-1}=a^{-1}a^p=a^{-1}a=1\mod p$.

WLOG, we prove the first half with positive a by induction on a. The case a=1 works for the theorem. Suppose the thm holds true for a, namely $a^p=a\mod p$. For $a+1$, we have $(a+1)^p=\sum_{k=0}^pC_p^k a^k$. Due to lemma 4.1, we have $(a+1)^p=a^p+1=a+1\mod p$. QED.

### Euler totient function

**Definition:** $\varphi(n)$ is defined as the number of positive integers that are coprime to n and less than n.

For prime p, it is obvious $\varphi(p)=p-1$. And for $p^\alpha$, the only non coprime number is $p,2p,3p…(p^{\alpha-1}-1)p$. Therefore, $\varphi(p^\alpha)=p^{\alpha-1}(p-1)$ which is the following lemma.

**Lemma 4.2:** For prime p, $\varphi(p^\alpha)=p^{\alpha-1}(p-1)$.

**Lemma 4.3:** If a and b are coprime, then $\varphi(ab)=\varphi(a)\varphi(b)$.

**Proof:** Cosider the system of equations, $x=x_a\mod a, x=x_b\mod b$. From theorem 4.1, there is a one to one correspondence between the solution $x<ab$ and the input $x_a<a,x_b<b$. Furthermore, if we add constrait $\gcd (x_a,a)=1,\gcd (x_b,b)=1$, this is equivalent to $\gcd(x,ab)=1$ (as x has no common nontrivial divisor with a or b). Therefore, there is a bijector between the two side $x$ vs $x_a,x_b$. There are $\varphi(a)\varphi(b)$ such pairs of $(x_a,x_b)$ and $\varphi(ab)$ pairs of $x$.

**Theorem 4.3:** (Expression of Euler function) For integer n with prime factorization $n=\prod_{j=1}^kp_j^{\alpha_j}$, we have $\varphi(n)=\prod_{j=1}^kp_j^{\alpha_j-1}(p_j-1)=n\prod_{j=1}^k(1-\frac{1}{p_j})$.

**Proof:** Combine lemma 4.2 and lemma 4.3, obvious.

### Euler's Theorem

**Theorem 4.4:** Suppose $\alpha$ is coprime to n, we have $a^{\varphi(n)}=1\mod n$.

**Proof:** We first show $a^{\varphi(p^\alpha)}=1\mod p^\alpha$, namely the case $n=p^\alpha$ where p is prime. When $\alpha =1$, we are back to Fermat's little theorem. By induction, assume this is true for $\alpha$, namely $a^{\varphi(p^\alpha)}=1+kp^\alpha$, then by lemma 4.2, we have $a^{\varphi(p^{\alpha+1})}=a^{p^\alpha(p-1)}=a^{p\phi(p^\alpha)}=(1+kp^\alpha)^p=1+\sum_{j} Cp^{j\alpha}=1\mod p^{\alpha+1}$.

For the more general case, note every integer has prime factorization. And these $p_i^{\alpha_i}$ terms are coprime with each other. Then $a^{\varphi(n)}=1\mod p_j^{\alpha_j}$ since $\varphi(n)=k\varphi(p_j^{\alpha_j})$. Since $p_j^{\alpha_j}$ are coprime to each other, we have the conclusion $a^{\varphi(n)}=1\mod n$.

**Comment:** (The algebra structure of modular arithmetic) Define all the elements from $Z_n$ which is coprime to n as $Z/nZ$. Such a structure is actually a ring with two operations (add and times in the modular sense). And $\vert Z/nZ\vert =\varphi(n)$. If we only focus on multiplication operation, this is a group. And we have the following theorem describing the structure of such groups.

**Theorem 4.5: ** $Z/nZ$ is cyclic iff $n=1,2,4,p^k,2p^k$,  where p is an odd prime and k>0.

**Proof:** It is straightforward to show the results if we are equiped knowledge with abstract algebra, see [this note](https://pi.math.cornell.edu/~mathclub/Media/mult-grp-cyclic-az.pdf) for the proof.

## References

* Michael A. Nielsen and Isaac L. Chuang, *Quantum Computation and Quantum Information*, mainly Appendix 4.

## TODO

- [ ] Bezout's identity