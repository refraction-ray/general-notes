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

**Comment:** (deep structure of this thm) In fact, Chinese remainder theorem is deeply correlated with the rings decomposition and hence play a central role in the proof of primary number theory. That is to say, if we take a high level view from advanced algebra, most of the conclusions and progress from primary number theory are correlated with the decomposition of $Z/nZ$ rings. In fact, thm 4.1 gives the duality between $Z_n$ and $\otimes_j Z_{p_j^{\alpha_j}}$ which is trivial. Besides thm 4.1 also gives the duality between $Z/nZ$ and $\otimes_j Z/(p_j^{\alpha_j}Z)$ which is nontrivial and actually the proof of Lemma 4.3 below.

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

## Reduction of factoring to order-finding

### Order

**Definition:** Suppose N is a positive integer and x is coprime to N, the order of x modulo N is defined to be the least positive integer r such that $x^r=1\mod N$.

**Comment:** Note the periodic behavior of $f(y)=x^y\mod N$ and $r\vert \varphi(N)$.

### The  First step

**Theorem 5.1:** Suppose N is a composite number and a is coprime to N. Suppose r is the order of a modulo N. If r is even and $a^{r/2}\neq -1\mod N$, then N has a nontrivial factor as $s=\gcd (a^{r/2}-1,N)$.

**Proof:** Since $N\vert (a^{r/2}-1)(a^{r/2}+1)$, and thus N must have a common factor with at least one of them. But $a^{r/2}\neq 1\mod N$ since r is order of a, namely $N\nmid a^{r/2}-1$. Besides, we have the condition $N\nmid a^{r/2}+1$, therefore both term contains nontrivial factor of N. Specifically, $s>1$ is a nontrivial factor of N.

**Comment:** nontrivial solution means that $x\neq \pm 1\mod N$.

### The Second Step

**Lemma 5.1:** Let p be an odd prime and  $2^d$ be the largest power of 2 that $2^d\vert \varphi(p^\alpha)$. Then with probability $1/2$, $2^d$ divides the order modulo $p^\alpha$ of a randomly chosen element from $Z/p^\alpha Z$.

**Proof:** Note that $\varphi(p^\alpha)=p^{\alpha-1}(p-1)$ is even so $d\geq 1$. By theorem 4.5, there exist a generator g, s.t. every element in $Z/p^\alpha Z$ can be written as $g^k\mod p^\alpha$ for k range from 1 to $\varphi(p^\alpha)$. Let r be the order of $g^k$ modulo $p^\alpha$, consider two cases. 1) When k is odd, from $g^{kr}=1\mod p^\alpha$, we have $\varphi(p^\alpha)\vert kr$. And thus $2^d\vert r$. 2) When k is even, $g^{k\varphi(p^\alpha)/2}=1^{k/2}\mod p^\alpha$. Thus $r\vert \varphi(p^\alpha)/2$. But since $2^{(d-1)}$ is the largest power of 2 which can divide $\varphi(p^\alpha)/2$, we conclude that $2^d\nmid \varphi(p^\alpha)$.

**Comment:** (structure of $Z/p^\alpha Z$) There are two classes of elements $g^k$. For even k, $2^d\vert r$ where r is the order of $g^k$. For odd k, $2^d\nmid r$.

**Theorem 5.2:** Suppose $N=\prod_{j=1}^m p_j^{\alpha_j}$ is the prime factorization of an odd composite positive integer. Let x be chosen randomly from $Z/NZ$, and let r be the order of x modular N. Then 
$$
Pr((r\vert 2) \and (x^{r/2}\neq-1\mod N))\geq 1-\frac{1}{2^{(m-1)}}
$$
**Proof:** We show that $Pr((r\nmid 2) \or (x^{r/2}=-1\mod N))\leq \frac{1}{2^{(m-1)}}$. And label them as condition 1 and condition 2.

By theorem 4.1, choosing x from $Z/nZ$ is equivalent to choosing $x_j$ independetly from $Z/p_j^{\alpha_j}Z$ and requiring the system of equations $x=x_j\mod p_j^{\alpha_j}$ for each j. Let $r_j$ be the order of $x_j$ modulo $p_j^{\alpha_j}$ and $2^{d_j}$ be the largest power of 2 that divides $r_j$. We will show that to satisfy condition 1 or 2, all $d_j$ have to take the same value as $d$. 

Note $r_j\vert r$ , since $n\vert x^r-1$ leads to $p_j^{\alpha_j}\vert x^r-1$, namely $x^r=1\mod p_j^{\alpha_j}$. 1) If r is odd, all $r_j$ is odd, then $d=d_j=0$. 2) If $x^{r/2}=-1\mod N$ , then $x^{r/2}=-1\mod p_j^{\alpha_j}$. Therefore, $r_j\nmid r/2$. Together with the fact $r_j\vert r$,  we have $r=s r_j=st 2^{d_j}$, where s and t are both odd integers. Thus we have $d_j=d$.

Let's now figure out the probability of all $d_j$ are equal. Say we randomly pick $x_1$ and determine $d_1$, then all $m-1$ terms must be the same. Suppose $2^{d_1}\vert \varphi(p_2^{\alpha_2})$, which is the largest power of 2 that divides phi. The probability of such event is no more than 1. Then based on the lemma 5.1, $P(2^{d_1}\vert r_2)\leq 1\times1/2=1/2$. In total, we show the probability is no more than $\frac{1}{2^{m-1}}$.

**Comments:** In other words, let N be odd and not a power of a prime (N at least be factorized into two different primes). $a$ is picked randomly from $Z/nZ$ (pick $1<a<N$ which is coprime to N),  say the order of a modulo N is r, then the probability that r is even and $a^{r/2}\neq \pm 1\mod N$ is no less than one half. This fact is directly from thm 5.2. $a^{r/2}\neq 1\mod N$ is due to the fact that r is the order of a (smallest integer make $a^r=1\mod N$).

### The algorithm for factoring

**Algorithm:** (factoring with subroutine of order finding)

1. Make sure the input N is not even (else return 2 as a factor) and not power of some prime (else return such prime as factor).
2. Randomly choose $1<x<N$. Compute $\gcd (x,N)$, if $\gcd(x,N)>1$, return it as a factor, else continue.
3. Determine the order r of x modulo N.
4. If r is even and $x^{r/2}\neq -1\mod N$, continue, else the algorithm fails, try another x to repeat the algorithm.
5. Compute $\gcd (x^{r/2}-1,N)$, and test to see which is a non-trivial factor of N. Return such factor.

**Correctness**: The correctness of the above algorithm is obvious following theorem following thm 5.1 and 5.2. In the first two steps, we are creating the condition of thm 5.2. The step 4 is successful with probability more than one half. So the randomized algorithm can achieve exponential success probability with several more trials. Step 5 is following thm 5.1.

**Complexity analysis**: It is obvious all steps can be done in polynomial time except step 3. The factor finding subroutine has no known classical algorithm with P complexity (though no theoretical lower bound of complexity, either). However, step 3 can be achieved by quantum algorithm named after Shor in polynomial times.

The complexity of step 1: check whether N is power of some prime. The approach is as following. We calculate $N^{1/k}$ for $k=2,…\log_2 N=L$. If any of these is an integer, we have found a factor. Otherwise, such N is not a power of an integer. So the complexity is still in polynomial time.

Finally the input of the above algorithm is promised to be a composite number. What if the input might be prime itself. Firstly, we already have determinstic primality test known as AKS primality test which is in P. And we can use such a subroutine before step 1 to make sure N is not a prime. Secondly, the algorithm itself is a good randomized primality test procedure. If the algorithm fails sufficient rounds, we are confident that the input N is a prime with high probability.

## References

* Michael A. Nielsen and Isaac L. Chuang, *Quantum Computation and Quantum Information*, mainly Appendix 4.

## TODO

- [ ] Bezout's identity