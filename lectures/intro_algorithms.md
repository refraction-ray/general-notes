#  Introduction to Algorithms

*Notes on MIT lectures, start writing from Aug 21, 2018*

* toc
{:toc}

## 1 Intro

### insertion sort

> iteration: sort first j of the array, then sort j+1 into it

$\Theta(n^2)$

### merge sort

> recursive: sort $A[1]$ to $A[ceil[n/2]]$and $A[ceil[n/2]+1]$ to $A[n]$
> until one element remain
> then merge the two groups (see below)

key subroutine: merge

> pick the smallest, then go on, every time you have two candiates from two subgroups to determine

for the complexity analysis, one again follow the recursive logic to get the recursive formula: $T(n)=2T(n/2)+\Theta(n)$, the first term on the right is for recursive and the second term is for the merge routine. You need solve a recurrence.

Recursion tree: make $\Theta(n)=cn$, following a brute-force expansion as a tree structure on paper. Every level of the tree count a $\Theta(n)$ and the depth of the tree is $\ln n$.

$\Theta(n\ln n)$

### complexity

worst time, asymptotic estimation

best time estimate make no sense as it can cheat

$\Theta$ notation for asymptotic analysis

## 2 Math basics

### asymptotic notation

* Big O notation ($\leq$)

$f(n)=O(g(n))$ means there are consts $C,n_0$, such that $0\leq f(n)\leq cg(n)$ for all $n>n_0$.

$O(g(n))$ can also be viewed as function sets with all possible $f(n)$.

macro convention: A set in a formula represents an anonymous function in that set. eg. $f=n^3+O(n^2)$.

large O in left hand, asymmetric understanding of the 'equal': is or belongs to. 

eg. $n^2+O(n)=O(n^2)$: for any $h$ in $O(n)$ there is a $f$ in $O(n^2)$, such that $n^2+h=f$.

* Big Omega notation ($\geq$)

$\Omega(g(n))$ is a set of functions which there exist const $c,n_0$, such that $0\leq cg(n)\leq f(n)$ for all $n\geq n_0$.

eg. $\sqrt{n}=\Omega(\ln n)$.

* Big Theta notation ($=$)

$\Theta(g(n))=O(g(n))\cap \Omega(g(n))$.

* little o($<$) and omega($>$) notation:

hold for all constant $c$ instead just for one, $n_0$ can dependent on $c$. c plays role of epsilon here.
eg. $2n^2=o(n^3)$, $n^2=\Theta(n^2)\neq o(n^2)$.

### recurrence solution

No general approach, of course.

* Substitution method

guess the answer form and match the coefficient.

loop hole: induction proof on $n=O(1)$. Since $1=O(1)$, by induction, assume $n-1=O(1)$, it is easy to conclude that $n=O(1)$ which is obviously not true. 
hint: never use O notation in induction, when n is expected to be infinite in some sense, the constant c in the O notation are actually increase each time which might end up infinite when n is infinite. Therefore when one use induction expand large O notation explicitly with constants.

in the subsitution method, one may want propose a O notation of the solution instead of explicit function form, and use induction to prove the O assumption. Such a bound guess is enough for complexity estimation, one dont need to exactly solve the recurrence, let alone there may be some O notation in the recurence at beginning. 

Sometimes, one need tighten the bound of O notation a little by add lower order expansions to prove the exact bound. eg $O(n^2)\sim c_1n^2-c_2n$ may be necessary for the proof. 

So basically the question, given $T(n)=4T(n/2)+n$, show that $T(n)<= c n^2$ for some constant n using induction, a cool problem because it is simpler to prove a tighter conclusion than the original one, which is hard to come up.

* Recursion Tree Method

not rigourous, just illustration.

eg. $T(n)=T(n/4)+T(n/2)+n^2$, expand in tree form with dot dot dot and add all thights up. Every step in expansion, writing constant above and $T$ below for another expansion. Finally estimate the leaf numbers and depth, then sum level by level. For the example above, it is a geometric series for level sum.

* Master method

Only applies to particular famlily of recurrence. $T(n)=aT(n/b)+f(n)$. $a\geq 1, b>1$, $f(n)$ should be asymptotically positive (for large enough n, f should be positive).

Thm: three cases below

Case 1) $f(n)=O(n^{\log_b a-\epsilon})$ for some $\epsilon>0$, then $T(n)=\Theta(n^{\log_b a})$.

Case 2) $f(n)=\Theta(n^{\log_b a}\log^k_2 n)$ for some $k\geq 0$, then $T(n)=\Theta(n^{\log_b a}\log_2^{k+1}n)$.

Case 3) $f(n)=\Omega(n^{\log_b a+\epsilon})$ for some $\epsilon>0$ and $af(n/b)\leq (1-\epsilon')f(n)$ for some $\epsilon'>0$, then $T(n)=\Theta(f(n))$.

How to apply: 1) compute $n^{log_b a}$, see f (compare f with) and jump to the specific case, note $k=0$ if not occur in f and $k<0$ case cannot be derived by master method.

This thm can be shown by recursive tree very easily.

## 3 Divide and Conquer

devide et impera

1. divide 2. conquer 3.combine

* binary search

find x in a already sorted array: 1 compare x with middle elements 2 reduce to find x in one sub array 3 combine: nothing. $T(n)=T(n/2)+\Theta(1)$, ie $T(n)=\log n$.

* powering a number

given number x and power interger $n\geq 0$, return $x^n$.

Naive alg: $\Theta(n)$. 

D&C: $f(x,n)=f(x,n/2)*f(x,n/2)$ , for odd n, just time one x more. Complexity: $\Theta(\log n)$

* Fibonacci number

Naive recursive: $\Omega(\phi^n)$ where $\phi=\frac{1+\sqrt 5}{2}$.

Bottom-up: compute in order, save them in cache: $\Theta(n)$

Naive recursive squaring: $F_n = round(\frac{\phi^n}{\sqrt 5})$, $\Theta(\log n)$, floating number destroy the approach

The correct approach: Thm: $\begin{pmatrix}F_{n+1}& F_n\\F_n& F_{n-1}\end{pmatrix}=\begin{pmatrix}1&1\\1&0\end{pmatrix}^n$. complexity $\Theta(\log n)$

* Matrix multiplication

$A=[a_{ij}], B=[b_{ij}]$ two matrix with $n*n$, return $C=[c_{ij}]$.

Naive: $\Theta(n^3)$

First B&C: divide matrix in four block for each one ($n/2\times n/2$) . $T(n)=8T(n/2)+\Theta(n^2)$, namely the complexity is still $\Theta(n^3)$.

Strassen's alg: somehow reduce number of multiplications from 8 to 7! Recall the division $A=[a,b,c,d]$ and $B=[e,f,g,h]$, we now compute seven $P_i$ each involving one matrix multiplication. And reudce the seven P to the four result blocks with only plus. Following these formula one can easily show the correctness of the alg. We now have subcubic alg for matrix multiplication: $T(n)=7T(n/2)+\Theta(n^2)$.

$\Theta(n^{\log 7})$.

*How to apply to non $2^n$ matrix. First, one can zero padding. And the method can similarly applies to non-square matrix too.*

Best so far alg is $\Theta(n^{2.376})$.

* VLSI (very large scale integration) layout

Problem: complete binary tree with n leaves, embed into agrid with minimum area. (embed meaning: nodes on vortex and link on edge of the square grid without crossing)

naive embedding: just drawing as the original structure of the tree on the grid. The height $H(n)=H(n/2)+\Theta(1)$, the weights $W(n)=2W(n/2)+O(1)$. The area is $\Theta(n\log n)$. Not a minimum configuration.

the H layout: every level lools like H letter where root node in the middle. Now the recurrence is $H(n)=2H(n/4)+\Theta(1)$, namely the $H(n)=\Theta(\sqrt n)$ and namely the area is of order n.

## 4 Quicksort

### quicksort

sorts in place (in terms of space); D&C; practical (with tuning)

> partition the input array into two subarray around pivot  x such that elements in the lower subarray $\leq x \leq$ elements in upper subarray
>
> recursively sort the two subarrays
>
> combine: trivial

subroutine partition:

> input A[p..q], pick pivot as A[p], compare all elements to the pivot by loop, if the elements is small than the pivot, exchange it with the one one the boundary, which is an integer index tracked by the loop at the same time

subroutine complexity: $\Theta(n)$. 

 tail recursive optimization

Analysis: assuming all elements are distinct. 

* Worst time analysis: Again $T(n)$ as the worst-case time. Worst case is that the input is sorted or reverse-sorted in which one of the partition has no elements. ie $T(n)=T(0)+T(n-1)+\Theta(n)$,  $T(n)=\Theta(n^2)$. 
* Best time analysis: partition split in the middle every time. $T(n)=2T(n/2)+\Theta(n)$, $T(n)=n\log n$.
* Suppose the split is alway 0.1:0.9. $T(n)=T(n/10)+T(9n/10)+\Theta(n)​$, do a recursion tree, and it gives every level no larger than $cn​$ with height between $\log_{10/9}n​$ and $\log_{10} n​$, and this gives $T(n)=\Theta(n\log n)​$, as lucky as the half split.
* Suppose we alternate lucky and unlucky case in the partition process. $L(n)=2U(n/2)+\Theta(n)$, and $U(n)=L(n-1)+\Theta(n)$. L and U for luck and unluck time. By combine the two, we have $L(n)=\Theta (n\log n)$.

### randomized quicksort

By picking random pivot every time. running time is independet on input array. no assumption on input distribution. no specific input can elicit the worst case behavior.

let $T(n)$ as the random variable. Define random variable $X_k=\delta(k=i)$, where i is the pivot out of n elements. $T(n)=\sum_k(T(k)+T(n-k)+\Theta(n))X_k$ with the probability of $1/n$.

Then $E[T(n)]=\sum E(X_k)*E(T+T-\Theta)$, due to the independence. $E[T(n)]=2/n(\sum_{k=2}^{n-1} E(T(k)))+\Theta(n)$. 

And prove $E=\Theta(n\log n)$ by substitution method of induction. One need the fact $\sum_{k=2}^{n-1}k\log k\leq 1/2 n^2\log n-1/8 n^2$, which is very easy to show by integral.

### *heapsort*

make use of complete binary tree (not necessary full binary tree) which can transform as a list with fixed relation to find the parent and child node indexes. if within such a tree, every node value are not less than their children, we call it max heap. 

two steps of heapsort: 1) make maxheap from array 2) pickup the max item(the first one) then maxheapify the new array

complexity: $\Theta(n \log n)$ due to $\Theta(\log n)$ time for heapify a node (insert node at the last position and make it move up to appropriate positions), in step 1, one need heapify around n nodes and in step 2, one heapify only root node each time sort one element. Note to build the heapify, it can only cost $O(n)$ time, there is a huge difference between using siftup or siftdown! 

See [this](https://stackoverflow.com/questions/9755721/how-can-building-a-heap-be-on-time-complexity) and [this](http://128kj.iteye.com/blog/1728555) for algorithm  on heapsort and see [wiki](https://en.wikipedia.org/wiki/Sorting_algorithm) on more aspects of sorting algorithms.

## 5 sort in linear time

How fast can we sort: it depends on models of what you can do with the elements.

all the above are comparison sorting models, you can only allowed to do compare pairs.

### map to the decision trees

Thm: no comparison sorting algorithm can run faster than $\Theta(n\log n)$.

Decision-tree model:

Give all the decision comparison branches with leaves as the sorting results.

Map decision tree to comparison sorts: one tree for each n.

Height of the tree is the worst time of sort alg.

Lower bound on decision tree sorting: the height is at least $n\log n$. Since no. of leaves is $n!$ , a tree with height h has at most $2^h$ leaves. Therefore $h_{min}=\log_2 n!\sim \log n^n$, DONE.

### sort beyond comparison model

* counting sort

each element of n-array input is an integer between 1 to k. 

require a length k asstitance array to keep the count of each category. $\Theta (k+n)$. 

* stable sort

so called stability preserve the order of equal elements, this property plays role when the elements are tuple or some thing with more info.

* radix sort

sort say integers with many digitals from the last (namely less importat at first) digital (using stable sort).

correctness proof (induction): induct on digit position that already sorting t, by assuming t-1 digits are already sorted.

comlexity analysis: using counting sort for each digit (one or several a time). assume each binary as b bits long, split it into single bit a time may not be optimal, so we set the split as b/r. digits are each r bits long. $T=b/r\Theta(2^{r}+n)$. Now one only need to minimize T in terms of r (optimal when $r\sim \log n$). Finally we have $T=O(\frac{b n}{\log n})\sim O(nd)$ where the number is in raner $n^d$.

* *bucket sort*

cousin of radix sort, first divide elements in some preset bins, and sort elements in each bin individually, finally link them together. if each bucket one element, the model reduce to counting sort. Say all elements are uniformly distributed between [0,1). Say we prepare n buckets for the sort and use insertion sort for each bin (have no idea why since insertion sort is slow). so $T=O(n)+\sum n_i^2$ where $\sum n_i^2=\sum\sum I_{ij}\sim n$, $I_{ij}$ denote the indicator random variable where i and j are in the same bucket.

## 6 Order statistics

given unsorted array, return the k-th smallest elements (elements of rank k).

Naive: sort first and return the k-th element. $\Theta(n\log n)$.

If $k=1$ or $k=n$, you can keep the minimum while scan the array. $\Theta(n)$

Medium case: a bit trickier

### randomized D&C (random select)

> recursively use the random partition from quicksort, which returns the final position index of pivot element (randomly chosen) k. assuming the rank we want is i, if i=k, done; if $i<k$ or $i>k$ , recusively use the partition subroutine for left of right subarray (remember the rank i we are looking for may get offsets if we now moves to the right subarray!).

Analysis: first assuming elements all distinct.

Best case: constant split every time, $T(n)=T(9/10n)+\Theta(n)$, namely $T(n)=\Theta(n)$.

Worst case: pivot always on the edge, $T(n)=T(n-1)+\Theta(n)$, namely $T(n)=n^2$.

Average case:  $ET(n)\leq\sum_k E[(T(\max(n-k-1,k))+\Theta(n))X_k]=1/n\sum_k E[T(\max(n-k-1,k))]+\Theta(n)$.

$ET(n)\leq 2/n\sum_{k=[n/2]}ET(k)$. Then use substitution induction method to show $ET=\Theta(n)$.

### worst-case linear-time order statistics

generate good pivot recursively.

> Divide the n elements array into 5 elements subarrays ([n/5]), omit the residue part.
>
> Recursively find median in each 5-elements group. $\Theta(n)$
>
> Recursively compute the median x of these medians. T(n/5)
>
> Use x as the pivot and do the partition $\Theta(n)$ and recursive call as above alg. T(3/4 n)

By the median choice, we have 3*[[n/5]/2] elements less and equal to x(the same for larger and equal). If n is large enough, this value is larger than $n/4$. 

The complexity recurrence $T(n)\leq T(n/5)+T(3/4n)+\Theta(n)$. Substitution induction to prove $T(n)=\Theta(n)$. If one want to get a linear upper bound, the intuition is make all the recurrence work unit together smaller than 1. Practically, this algorithm have large prefactor constants so maybe not so fast.

Why group of 5? Number larger than 5 also works. You need require $1/k+(k/2+1)/2k<1$, where I omit some [] function.

## 7 hashing

symbol table problem: table S holding n records, key-data pair

Operations: Insert (by pair), Delete (by key), Search (by key)

Direct access table: works when keys are distinct and are drawn a set U of m elements, namely two list one for keys and one for values. All operations take $\Theta(1)$ time in the worst case. Very impractical.

hash function H to map keys randomly into slot(index) table T. 

collision: resolve collisions by **chaining**. link records in the same slot into a list.

worst case analysis: all keys are hased to the same slot. (reduce to linked list). access takes $\Theta(n)$ time.

average case: simple uniform hashing. one hashtable, n keys and m slots. The load factor is $\alpha=n/m$, which is average no. of keys per slot.

Expected unsuccessful search time (return None): $\Theta(1+\alpha)$, first hash the value for the key and search the linked list.

### choose hash functions

regularity of key distributions should not affect unifomity of hash.

* Division method

$h(k)=k \mod m$. m with small factor is bad. Even mod even are always even, so m with factor 2, will map all even to even instead of uniformity. Pick m as a prime (not too close to power of 2 or 10) is good practice.

* Multiplication method

Assume the slot $m=2^r$, w-bit words, $h(k)=(Ak \mod 2^w).rightshift(w-r)$. A is an odd integer with bits the same as w. The formula is in the language of binary. A do not: not pick near power of 2.

### resolve collisions by open addressing

if collision use different hash function, probe table systematic till empty slots is found. $h(keys,probnum)\rightarrow slots$. deletion is difficult in this scheme, one can add a state to label the delete status instead of really delete items. 

* linear probing

$h(k,i)=(h'(k)+i) \mod m$. if filled, just scan down the nex slot. disadvantage: primary clustering (long runs of filled slots)

* double hashing

good scheme. $h(k,i)=(h_1(k)+i h_2(k))\mod m$. usually m is taken as power of 2 and force $h_2$ to be odd.

Analysis of open addressing:

assumption of uniform of hashing: each key is equally likely to have any of the $m$ permutations as its probe sequence independent of other keys.

Thm: E[#probes_of_fail]$\leq 1/(1-\alpha)$, of course $\alpha<1$ is required in open addressing scheme

Proof: for unsuccessful search, first hit and with $n/m$ probability to get collision, and then a second probe, again collision probability $(n-1)/(m-1)$ and so on. Note $(n-i)/(m-i)<\alpha$, sum them together, we have the thm results. The same result applies to successful search case. So make th hash table sparse to keep it fast.

##  8 hashing II

### universal hashing

idea: choose a hash function at random to avoid delibrately collision. *Within one hashtable, we dont randomly pick hash functions everytime, instead it is only selected when initilizing the hash table once. See [this post](https://stackoverflow.com/questions/10416404/finding-items-in-an-universal-hash-table) in stackoverflow*

U be a universe of keys, and H be a finite collection of hash functions: maping U to to sorts 1 to m.

Definition: We say H is universal if for all paris of distinct keys,  the number of h in H, which map the two keys into the same slot, is $|H|/m$. i.e. if h is chosen randomly from H, the probability of collision between two keys is $1/m$.

Thm: choose h randomly from H, suppose we hash n keys in T into m slots, for a given key x, the expecation number of collision with x is less than $\alpha=n/m$.

Proof: $C_x$ be the random variables denoting the total number of collsions of keys in T with x. $c_{xy}$ is nonzero only when keys x and y collision. namely $E(c_{xy})=1/m$. As $C_x=\sum_{y\in T-\{x\}}c_{xy}$, $E(C_x)=(n-1)/m$. 

### construction of universal hash functions

m is prime, decomose key k into r+1 digits based m (take mod of m step by step to get the r+1 digits decomp). pick $a$ at random, which is also based-m number, each a for a hash function. Define hash $h_a(k)=\sum a_ik_i\mod m$.

How big is the set of hash function $|H|$? Answer: $m^{r+1}$. 

Proof on H is universal: pick two distinct keys x, y with their decomposition representation in based-m. Say they are different in the zero digit. Count the number of $h_a$ which collide x,y, ie $h_a(x)=h_a(y)$. Namely $a_ix_i \equiv a_iy_i\mod m$, where summation convention is assumed,

$a_0(x_0-y_0)\equiv  \sum_{i=1}^r -a_i(x_i-y_i)  (\mod m)$ 

* Number theory fact detour

(from theory of finite fields) let m be **prime**, for any $z\in Z_m$ such that $z\not\equiv 0$, there exists a unique $z^{-1}\in Z_m$ such that $zz^{-1}\equiv 1 \mod m$.

So we now definitely have well defined $(x_0-y_0)^{-1}$ , therefore

$a_0 \equiv -\sum a_i(x_i-y_i) (x_0-y_0)^{-1}$, so freedom minus 1, and the number of possible $a$ to collide is $|H|/m$.

### perfect hashing

keys are all give at the beginning , determine the best hash function for constructing the static hash table. with space $m=O(n)$, make search $O(1)$ in time in the worst case (actually only two times of hashes).

idea: two-level scheme with universal hashing at two levels. If collide at level 1 slot, then the slot only save lists of random label $a$ for the second hash. no collision in level 2. if n collision in level 1, use $n_i^2$ slots for level 2 hash i. 

Thm: hash n keys to $n^2$ slots using h from universal set H, the expected number of collisions is less than $1/2$. The probability to collide is $1/n^2$, and no. of pairs key is $C_n^2$, hence the expected no. of collisions are products of the two factors, which is $1/2-1/(2n)$. (similar analysis as birthday paradox)
No collision at all probability is at least 1/2. 

Since the probability is very large, just try some random hash function, and one should hopefully work as $h_i$ from collision slots in level 1 to level 2.


* Markov inequality

For random variable $X\geq 0$ , $Pr[x\geq t]\leq \frac{E[x]}{t}$. (*too trivial to be a statement...*)

Space analysis: level 1, choose number of slots m to be equal to the number of keys n. let $n_i$ be random variable living in i slot in level 1. In level 2: the total space is $E=n+\sum_{i=1}^{n-1}E(n_i^2)=\Theta(n)$. Noting the counting of collsion is $\sum E(n_i^2)=\sum\sum E(I_{ij})$ , where I is the indicator variable give 1 when $h(i)=h(j)$. 

There are some middleware steps. First, if $\sum n_i^2> cn$ for some constant c, redo step 1, namely pick a new hash function h to rehash all things. The most nontrivial part of perfect hashing idea is actually those square level 2 slots are of the same order of n in total.

To summarize, pefect hashing is the type of an algorithm of designing hash function of fixed $n$ given keys which requires $O(1)$ times for accessing data by keys, and with space memory (slots in total) at most of order of keys $O(n)$. Furthermore, to find such function it should take no more than polyminal time with high probability (or in expectation context). (Actually we cannot assert this is always the case even for worst case, and this is a general property of algorithms with randomness. Say one flip the coins, he may not get heads for 1000 times of trials in theory, so we can only assert in expectation language or in high probability language). 

To recap the solution, we design a two level universal hashing. For colliding keys in the first round we find a second round hashing for each slot in first round and separate these keys in the second round. This is guaranteed by the square of the size of conflict keys in the first round in each slots. And similar to the birthday paradox, it is easy to show than the probability of avoding collision in the second round is lager than 0.5. In other words, it is fast to find effective hash function from the universal set for every second-round hash functions. The non-trivial part of that is in fact such a large space of slots in two levels are also of order $n$. In other words,  we can tune $c$ to make the probability of picking first round hashing probility also lager than 0.5. Therefore we can conclude that such a scheme can be designed in polyminal times, which finish the proof.

## 9 binary search trees (BST)

Some word on BST: insert - compare again and again, insert as a leaf. delete - find the exchange partner leaf, exchange and done.

balanced vs unbalanced: $\Theta(\log n)$ vs $\Theta(n)$

BST sort: 1) insert each item of array into BST, 2) do in-order tree-walk (recursively print left child then right, which lead to in-order traverse, i.e. left middle right sequence).

complexity: 1) $\Omega(n\log n)$ and $O(n^2)$ 2) $O(n)$ 

for 1) worst case: already sorted, best case: balanced tree in every step

Note the similarity between BST sort and quicksort, they make the same comparisons in different order.

Following the pivots of quicksort in each step (the first elements in remaining arrays), we have the exactly same tree as BST of the array.

### randomized BST sort

> randomly permute the input array and then call the usual BST sort.

Such an algorithm is equivalent to randomized version of quick sort. So the time complexity is the same.

$\Theta(n\log n)=E(\sum_{x}depth(x))$. Therefore, the average depth of elements in the tree is $\Theta(\log n)$. (not the height of the tree: which is the max depth of node).

Thm: E(height of rand BST) is $O(\log n)$. (the depth expecation is trivial but this thm is nontrivial)

### the proof 

Jensen's ineuqailty: convex function f and random variable x: $f(E[x])\leq E(f(x))$.

Basic idea, let $Y_n = 2^{X_n}$, where X is the height random variable, and we prove $E(Y_n)=O(n^3)$ first. 

If the root has rank k, then we have $X_n=1+\max(X_{k-1},X_{n-k})$. 

$Y_n=2\max(Y_{k-1},Y_{n-k})$. 

def indicator random variables: $Z_{nk}$ is 1 when the root has rank k and zero otherwise. $E(Z_{nk}=1/n)$.

$Y_n=\sum_{k=1}^n Z_{nk}2\max(Y_{k-1},Y_{n-k})$,

for expecation $E[Y_n]=2/n(\sum E[\max(Y_{k-1},Y_{n-k})])\leq 4/n \sum E(Y_k)$ ,  now we have the recurrence. We can use substitution method to prove this by induction $E[Y_n]\leq c n^3$. Done. (you have to use integral of cubic sum in the induction proof).

Precisely $E\approx 2.9882 \log n$.

## 10 balanced search trees

a tree guaranteed to be $\log n$ in height

including: AVL trees; 2-3 trees; 2-3-4 trees; B-trees; red-black trees; skip lists; treaps

### Red-black trees

binary search trees with extra color filed in each node.2

red-black property: 

1. Every node is either red or black

2. The root and leaves (imaginary null pointers, not real leaf nodes as usual) are all black

3. The parents of every red node is black

4. All simple path from node X to a descendant leaf of x: have same number of black nodes on the path. (black-height of x: exclude x itself) In other words: every node has consistent black-height definitions.

Thm: The height of red-black trees of n keys at most $2\log(n+1)$.

Proof sketch: Merge each red node to its parent black node (not a binary tree anymore, but 2-3-4 tree, all the leaves now has the same height which is the black-height in the original tree).

Now we have:

1. every internal nodes has 2-4 children
2. every leaves have the same depth, the height of this tree is denoted as $h'$

note there are $n+1$ leaves for n node trees. also in 2-3-4 tree, the number of leaves has to be beteen $2^{h'}$ to $4^{h'}$. Therefore $h'\leq \log(n+1)$. Furthermore $h\leq 2h'$ due to property 3 of red-black tree. Done the proof.

How to do relevant operations (insert or delete) to keep the tree of red-black trees.

* insert

> do BST insert, and pick red for it (property 3 is violated)
>
> color changes: move violation up by recoloring until we can fix it by rotation and recoloring. specifically, the subroutine is when the input node x is red and x.p is also red, we need to resolve the violation, do the loop: suppose x.p == x.p.p.l (x.p == x.p.p.r is similar), set y=x.p.p.r (uncle of x), if y is red, do case 1. elif  x == p.r, do case 2. if x == p.l do case3. 
>
> finally fix root as black
>
> case 1) x.p and x.uncle change to black, x.p.p change to red, recusively input x.p.p into the subroutine 
>
> case 2) left rotation on x.p, then x==x.p.l just go to case 3)
>
> case 3)  recolor x.p and x.p.p and right rotate of x.p.p done.

right-rotate of some node (left child turn to parent and the right child of original left child are not the left child of the original root). preserve binary search property. the reveser operation is left-rotate. rotation takes constant time.

It is nice the rotation takes $O(1)$ and less than recoloring, because recoloring doesn't affect the queries.

* *deletion*

first of all, if the node to be deleted has two non-leaves children, then use the similar delete method for binary tree(namely find the node with max value in left descendants), exchange them and delete that one which is at most with one non leave child. And if the node to be deleted has only leaves children, already done.

so we only care about cases where node with one non-leaves node child. again case by case, delete red, with black child instead of it done! delete black with red child, move up the child can recolor it to black, done! 

The only subtle case is black node with one black child. For detailed process, see [wiki](https://en.wikipedia.org/wiki/Red%E2%80%93black_tree#Removal).

*see visualization of rb trees in this [post](http://saturnman.blog.163.com/blog/static/5576112010969420383/).*

### *other types of balanced trees*

* treap

treap = tree (with value)+ heap (with priority), priority is random given at the beginnig of insertion.

For insert operation, first insert the node as binary tree, and use siftup in heapify to move it up. Since all operations in siftup are rotation which preserve the binary tree requirements, we can finnaly have a treap. Note this takes $O(\log n) $ rotations compared to RB tree, where only O(1) rotations.

In a word, treap is the dynamic version of randomized binary trees. We introduce priority set of values to realize the permutation of input array dynamically. So it can not gurantee the average cost when worst case happens.

* AVL trees

from any node, the difference height between left and right subtree are not greater than 1. single and double rotations are needed for insertion and deletion.

* B trees

2-3 or 2-3-4 trees are all special cases of general B trees. See detail about B-trees (including its variants) [here](https://blog.csdn.net/v_july_v/article/details/6530142).

database prefer trees than hashtables: see [this](https://stackoverflow.com/questions/7306316/b-tree-vs-hash-table)

* skiplist

see lecture later, no more notes here for skiplist.

* splay tree

idea: no strict restriction on whether the tree is balanced, but everytime a node is searched, the node would move up to the root via series of well-defined rotations which all preserve the probability of binary trees. See visualization [here](http://btechsmartclass.com/DS/U5_T5.html) for these so-called zig zag something rotations.

## 11 augmenting data structure

### order statistics trees (OS tree)

dynamic order statistics, tasks including select i-th smallest number or give the sorted rank of given elements

idea: keep the size of subtrees in each nodes of red-black trees (1 for lowest node, not null leaves).

size[x] = size[left[x]]+size[right[x]]+1

> input: root node x and the i-th 
>
> k = size of left[x] +1
>
> return x if i==k
>
> else return recusively the function with new input

$O(\log n)$ 

the size field is easy to maintain compared to directly rank info on nodes when dynamics is considered.

update subtree size filed when inserting or deleteing, and deal with rebalancing, when doing rotations, loos children and fix up.still $O(\log n)$. 

### interval trees

maintain of time interverals.

qurey: find interval with overlaps of given interval

underlying data structure: red black trees keyed on low endpoint

additional stored m value which is the largest high endpoint value in the subtree, namely $m(x)=\max\{ h(x), m(x.l), m(x.r)\}$ . 

for insert modifying: 1) tree insert, change node to m is m is the max when the ned node sink down to the tree. 2) rotations, new parents arrange m value based on their children to fix up.

interval-search (list one overlap), input interval is i

> set x to root, if i is overlap with x interval, return x
>
> else if x.l != None and low[i]<x.l.m recusively set x to x.l, else set x to x.r
>

 list all overlap intervals: find one and delete it $O(k\log n)$ output sensitive. Best algorithm now can do it in $O(k+\log n)$.

Thm: L is intevals sets in left subtree and the same to R. 1) if search goes right, then nothing overlap i in L. 2) If search goes left and nothing overlap in L, there is nothing overlap in right. (guarantee the algorithm correctness)

Proof: 1) search goes right, if L is empty, done. Since low[i] > x.l.m, the lhs is some high-end point in left subtree, so there is no possibility to be overlaped with i in left.

2) low[i]<x.l.m, and nothing overlap in left, for j in L, high[i]<low[j] (the other direction is restricted by condition). As the right half is further larger, no way to find overlap, done.

## 12 Skiplist

idea of first step: two sorted double linked list with cross pointer linking the same value, one list has less members.

search in such 2 linked list, walk on top list $L_1$ until go too far (transit to the lower list $L_0$ with all members then). 

complexity - search cost: $|L_1|+|L_0|/|L_1|$

to minimize it, we choose $|L_1|=\sqrt{L_0}$, the search cost is $O(\sqrt{n})$ compared to linked list $O(n)$.

now we add more sparse sort linked list on top of that, for k sort list, the complexity is $kn^{1/k}$.

if we have $k=\lg n$ sorted list, the complexity is $\lg n n^{1/\lg n}=2\lg n$.

now we make this idea of static set dynamic (insert & delete).

To insert x, first search where x fits in $L_0$, now you may want promote this node randomly. Half-Half: promote this node and repeat until the other half. If one always get heads, this promotion continues beyond the height of skiplist, you need to creat new level of linked list for more heads of coins. randomization make the worst case $O(n)$.

Delete is easy, just delete at all levels, done.

Lemma: number of levels in skip list is $O(\lg n)$ **with high probability** ($1-1/n^\alpha$). ($\alpha$ dependent on c in big O notation).

Proof of the lemma: (focus on failure probability) P(>$c\lg n$ levels)=P(some elements get promote more than $ c\lg n$ times) $\leq$ (union bound used here) n$\times$P(an element get promote more than such times)=$\frac{n}{2^{c\lg n}}=\frac{1}{n^{c-1}}$. Namely $\alpha = c-1$.

Thm: Any search in n-element skiplist, cost $O(\lg n)$ w.h.p. 

Proof: Trace the search path backwards. We pop up if the node is promoted. Such a path with up and left move with probability 0.5. Up move number is less than the height which is $O(\lg n)$ w.h.p. as the lemma.  The total number of moves = the number of moves till you get $c\lg n$ up moves w.h.p. Reduce the problem to a series of coin flip, which end when $c\lg n$ heads occur, the total number of such coin flips is $O(\lg n)$ w.h.p. Note there are two events. A for height of skiplist while B for the total number of coin flips under the condition A happens. You need to show the w.h.p can apply to A and B to conclude the proof.

Chernoff bound: let Y be a random variable, representing the total number of tails in a series of m independent coin flips, where each flip has a probability $p$ of coming up heads, for all $r>0$, we have:

$P[Y\geq (1+\delta)\mu]\leq \exp^{-\delta^2\mu/(2+\delta)}$. (see proof on this bound [here](https://www.cs.cmu.edu/afs/cs/academic/class/15859-f04/www/scribes/lec9.pdf))

Lemma: for any c, there is a const $d$, such that w.h.p the number of heads in flipping $d\lg n$ is at least $c\lg n$. ($d=8c$). 

*space complexsity expecation* 2n.

## 13 amortized analysis

### simple example on dynamic table

question: how large should a hash table be? as large as possible to make searching fast. space vs. time. the sweet point is $\Theta(n)$. What if we don't know n at beginning?

solution: dynamic tables. whenever the table is overflows we grow (double the size) it (malloc). create a larger table and move items form the old one to the new and free the old one.

analysis of the above dynamic solution: worst case of 1 insert $\Theta(n)$ therefore worst case if n inserts is $\Theta(n^2)$. (wrong analysis) if fact n inserts still take $\Theta(n)$ time in the worst case.

correct analysis: let $c_i$ be the cost of i-th insert which is i if i is the power of two or otherwise 1. 

$\sum_{i=1}^n c_i=n+\sum_{j=0}^{[\lg (n-1)]}2^j\leq 3n=\Theta(n)$.

Thus the average cost per insert is still $\Theta(1)$.

Though talking about average, there is no probability, it is about average performance in worst case.

### types of amortized analysis

* aggregate analysis

as above.

* accounting argument

charge ith operation a fictitious amortized cost $\hat{c}_i$,  fee is cosumed to perform the operation and the unused part is stored in the bank for later use. Bank balance must not go negative. Therefore $\sum c_i\leq \sum \hat{c}_i$.

case study: dynamic table. change 3 for each insert, one is going to pay for immediate insert and 2 is stored for doubling the table. when the table doubles, we use the stored money to move items. Thus the cost is bound by 3n.

* potential argument

framework of the potential method: start with some data structure $D_0$. operation make $D_{i-1}$ to $D_{i}$, the cost of operation is $c_i$. Define the potential $\Phi$ map $D$ to real value $R$, such that $\Phi(D_0)=0$ and $\Phi(D_i)\geq 0$. 

Define the amortized cost $\hat{c}_i=c_i+\Phi(D_i)-\Phi(D_{i-1})$. somehow the same thing of accounting method.

If $\Delta \Phi>0$, then $\hat{c}_i>c_i$ and vice versa. The total amortized cost is $\sum\hat{c}_i=\sum c_i +\Phi(D_n)\geq \sum c_i$.

case study: $\Phi(D_i)=2i-2^{[\lg i]}$. One can find the amortized cost by definition.

Different potential functions many yield different bounds.

## 14 Competive analysis

### self-organizing list

List L of n elements, operation: access of item cost is rank(x) which is the distance of x from the head of the list. L can be reordered by transposing neighbor elements $O(1)$. 

**Online**: a sequence S of operations is provided one at a time, for each operation, an online algorithm must execute the operation immediately. On the contrary, offline algorithm can see all of the S in advance.

The goal is to minimize the total cost of series of queries. $C_A(S)$

* Worst case analysis: adversary always accesses tail element, $C_A(S)=\Omega(|S|n)$.
* average case analysis: element x is accessed with probability $P(x)$, the expected cost on a sequence is $E[C_A(S)]=\sum p(x) rank(x)$, such value is minimized when L is sorted in decreasing order with respect to probability.

Heuristic: keep count of number of times an element is accessed and maintain the list in the order of this count.

In practice: move to the front  (MTF) of the previous accessed element, the cost is $2 rank(x)$.

### competive analysis

definition: An online algorithm A is $\alpha$-competive, if there is a constant k such that for and seq S of operations, the cost $C_A(S)\leq \alpha C_{OPT}(S)+k$. OPT is for the optimal **offline** algorithm. [God's algorithm]

Thm: MTF is 4-competive for self-organizing lists. 

Proof: $L_i$ is MTF list while $L_i^*$ is OPT's list. $c_i$ be MTF cost for ith query $2 rank_{L_{i-1}}(x)$. $c_i^*$ be OPT cost $rank_{L^*_{i-1}}(x)+t_i$ if OPT perform $t_i$ transposes. 

Define the potential function $\Phi(L_i)=2|\{x,y\}|$ for $x<y$ in $L_i$ and $y<x$ in $L_i^*$. Namely twice the number of inversions. Note $\Phi$ is always non-negative and $\Phi(L_0)=0$ if MTF and OPT have the same starting list.

How much does potential change from a transpose. A transpose create or destropy one inversion and $\Delta \Phi=\pm 2$. 

See what happen in the i-th access x. define $A,B,C,D={y\in L_{i-1}}$ for certain y. A is y before x in MTF while y after x in OPT, B is y before x and y before x, C is y after x and x before y, D is y after x and y after x. All the first is in terms of MTF list $L_{i-1}$ while the statement later is for OPT case. Now rank $r=|A|+|B|+1$ while $r^*=|A|+|C|+1$. 

When x is move to the front, it creates $|A|-|B|$ inversions. In OPT case, each transpose by OPT create no more than 1 inversions per transpose. $\Delta\Phi_i \leq 2(|A|-|B|+t_i)$.

Amortized cost $\hat{c}_i = c_i+\Delta \Phi_i\leq 2r+2(|A|-|B|+t_i)=4|A|+2t_i+2\leq 4(r^*+t_i)=c_i^*$.

$C_{MTF}(S)=\sum c_i=\sum \hat{c}_i -\Delta\Phi\leq 4C_{OPT}(S)-\Phi(L_{n})$. Done.

If we count transposes that move x toward the front of L as 'free' (O(1)), then MTF is 2-competitive. *(set the potential function as one times the inversions)*

What if they dont start with the same list. Then the potential function at the beginning might be as big as $2C_{n-1}^2=O(n^2)$. In this case $C_{MTF}\leq 4 C_{OPT}-\Theta(n^2)$, which is still 4 compatitive since $n^2$ is a constant as $|S|\rightarrow \infty$.

*The most hard part is how to find a successful potential function form, see more info about online algorithm and competitive analysis on this [document](http://www14.in.tum.de/personen/albers/papers/brics.pdf), specifically paging problem*

## 15 Dynamic programming

design techniques

example problem: longest common subsequence (LCS). given two sequence x and y, find **a** longest seq common to both.(maybe not unique). possible application: diff tool, dna seq comparison, etc.

LCS(x,y): the seq doesn't necessarily continous in x or y.

Naive solution:  check every subseq of x to see if it is also a subseq of y

Analysis: check on y $O(n)$, number of subsequence in x $O(2^m)$, worst case complexity $O(n 2^m)$.

* simplification stage

1) look at the length of LCS(x,y)  2) entend the alg to find LCS itself

strategy: consider prefixes of x and y. Define $c[i,j]=|LCS(x[1:i],y[1:j])|$.

Thm: c[i,j] is c[i-1,j-1]+1 if x[i]=y[j] and $\max\{c[i,j-1],c[i-1,j]\}$ otherwise. (trivial) 

Dynamic programming hallmark #1: **optimal substructure**: an optimal solution to a problem instance contains optimal solutions to sub problems.

Namely if z=LCS(x,y), then any prefix of z is LCS of some prefix of x and y.

Recursive alg for LCS 

> just recursively find c[i,j] based on the thm recurrence

worst case (anytime x[i] is not equal to y[j]): recursive tree, height m+n, the work is still exponential.

Dynamic programming hallmark #2: **overlaping subproblems**: a recursive solution contains small number of distinct subproblems repeated many times.

The subproblem space of LCS contains mn distinct subproblems.

memoization algorithm:

> keeps a table of c[i,j]
>
> if c[i,j] is none, compute it recursively and assige the value to c[i,j]
>
> otherwise return it

analysis: $\Theta(mn)$. amortized constant work for each entry. space complexity: $\Theta(mn)$.

dynamic programming (bottom-up)

fill the table row by row based on th recurrence.

analysis: time complexity $\Theta(mn)$. Reconstruct LCS by tracing backwards. (when meet diagonal crossong in cij table, record the elements correspondingly). space complexity: $\Theta(mn)$. can do $\Theta(\min\{m,n\})$. Everytime only keep one row for the bottom up build (or column by column).

* Hirschberg's alg

approach to keep reconstruction of LCS in $\Theta(\min\{m,n\})$ space. see [here](https://imada.sdu.dk/~rolf/Edu/DM823/E16/Hirschberg.pdf), the time complexity constant is much larger. It is a trick that save memory space at the cost of more time. Basic idea, recusively find the position the optimal path cross in middle column of the area. The position can be find by max the sum of elements from LCS and LCS from end table.

## 15 Greedy Algorithm

Minimum Spanning Tree (MST)

greedy alg: makes locally best choice, ignoring effect on future

Tree: connected acylic graph
spanning tree of a graph G: subsets of edges in G which form a tree and hit all vertices of G.

MST problem: given graph G=(V,E) and edge weight function w: E $\rightarrow$ R, find a spanning tree T of minimum weight $\sum_{e\in T}w(e)$. (may not be unique)

greedy properties:
1. optimal substructure
2. greedy choice property: locally optimal choices lead to globally optimal solution.

Optimal substructure for MST:
if edge e={u,v} is an edge of some MST, contract the edge (merge the two end points u and v), if there are other edges merging at the same time, take the minimum weights between them. Such a graph is G/e.

Thm: if T' is MST of G/e, than $T'\cup \{e\}$ is MST of G.

Proof: look at how a tree be contracted e. assume MST $e\in T^*$ of G.
T*-e is spanning tree of G/e. We know the weight of T' is no larger than  T*-e.
$w(T'\cup \{e\})=w(T')+W(e)\leq W(T^*-e)+w(e)=w(T^*)$.
Therefore $T'\cup\{e\}$ is a MST.

Dynamic porgram: ?
> guess edge e in MST
> contract the edge 
> recurse to find the MST
> decontract the edge
> add e to the MST

exponential time

Greedy-choice property for MST:
consider any cut (S,V-S) and crossing edges over this cut, suppose e is a least-weighted edge crossing the cut. then e is in some MST. 

Proof (cut and paste method):
let T* be a MST of G, if e is not in T*,  there must be an edge e' in T* that cross the cut. Remove e' and use e instead, T*-e'+e. Such new structure is a spanning tree and indeed MST. Done the proof by only modified edges cross the cut.

### Prim's algorithm
starting from a single node S and grow the tree one node by one node where minimum edge is pick across the new cut.
> maintain **priority queue** (may implemented by heap) Q on V-S, the key value of v is the $\min\{w(u,v)|u\in S\}$.
> init Q stores V, set the key of a node s to be zero, and other v set to infinity 
> do loop until Q is empty :
> extract max priority item u from the queue, update the key value of all items in the queue Q (partial update is enough, compare the original key with the distance of u), update the nodes' parent attr, which is return at the end as a MST structure.

correctness proof: $T_s$ is the subset of some MST of G in each iterative steps. (proof by induction making use of cut and paste).

complexity: same as Dijkstra, if use Fibonacci heap as priority queue, we have $O(V\lg V+E)$, the first term is extract-min of heap and the second term is decrease key of heap.

### Kruskal's algorithm

maintain connected components in MST-so-far T in a **union-find structure**.

> T = empty
> for v in V, make-set(v)
> sort E by weight
> for e={u,v} in E, if find-set(u) != find-set(v)
> T = T+e; union(u,v)

complexity: O(sort[E]+E\alpha(V)+V), the sort time is dominate. if weights is integers, it is linear time cost (radix sort).

correctness: tree T so far is some MST, induction proof

### *union-find*

set of groups of data, find given by data return group keys and there is union set op to merge groups. The average time cost per op is only $O(\alpha(n))$ where alpha is the inverse function of Ackermann function $A(n,n)$ which grows suprisingly fast. Therefore alpha grows very slow and hardly over 5 for practical data.

the structure is a forest, where all the roots are representative elements in each set. for union operation, attach one root to another (short to high to keep balance). For find, we use accelerate methods including path compression (attach the query to root directly everytime it is found), path splitting (every node on the path to their grandparent) or path halving.

### *all types of fancy heaps*

heaps are backbone of the priority queue. however, different types of heaps give very different time cost on operations, see [wiki](https://en.wikipedia.org/wiki/Priority_queue#Summary_of_running_times) for the complexity comparison table.

* Fibonacci heap

a fast implementation of priority queue. amortized complexity is constant for insert and findmin!! still $O(\lg n)$ for delete and extract-min. For union of two heaps, fibonacci heap can also achieve $O(1)$ compared to $O(n)$ of ordinary heap!

## 16 shortest path I

shortest path in graph $\delta(u,v)$: weighted graphs G=(V,E), real value for edge weights w(e). 

Note: negative edge weights may lead to non-existing of shortest path (just go forth and back forever). and disconnecting leads to infinity weighted shortest path.

checklist on optimal substructure: a subpath of shortest path is a shortest path. Proof (by cut and paste): trivial.

Triangle inequality: shortest path of u to v is no greater than the sum of shortest path of u to x and x to v. (trivial)

* single-source shortest paths problem

give a source vertex, how to get everywhere else with shortest weights

assume all weights is nonnegative in this section.

greedy 1) maintain set S of vertices whose shortest path weight from source s is known.  (initially pnly s is in S)  2) S grows one vertex per step, we add one of V-S whose estimated distance is minimum 3) update distance estimates: ones adjacent to v which is the newly coming to S

* Dijkstra's algorithm

> initial the distance array d[x] to infinity except the source s as x[s]=0, Q=V (Q is min priority queue, x  keyed on d[x])
> while Q !=0, u = extract-min(Q),  add u to set S, 
> for each v in Adj(u), if d[v]>d[u]+w(u,v), (relaxation step ) then lhs=rhs (the sametime a decrese key op in Q)

(how to get the actual path? shortest-path-tree) MST vs SPT: see [here](https://www.me.utexas.edu/~jensen/exercises/mst_spt/mst_spt.html).

* correctness proof:

Lemma1: invariant $d[v]\geq \delta(s,v)$ for all v anytime in the program. (trivial)

Proof (by induction and contradiction): consider the first violation, $d[v]<\delta(s,v)$, that is $d[u]+w(u,v)< \delta(s,v)$, impossible.

Lemma2: suppose we know shortest path s...u.v, and d[u]= $\delta(s,u)$, and suppose we relax (u,v) edge, then $d[v]=\delta(s,v)$. (trivial)

d[v] doesn't change after v is add to S, just prove d[v] is correct when v is adding to S, it cannot be larger afterwards due to relaxation step. And it is also lower bounded by Lemma1.

Proof by contradiction: if d[u] is the first incorrect term when adding u, then $d[u]>\delta(s,u)$. Let p be the shortest path from s to u. consider the first edge (x,y) where p exist S, bu induction d[x]=$\delta(s,x)$. by lemma2 (???) $d[y]\leq \delta(s,u)$ but $d[y]\geq d[u]$ due to the queue extract u instead of y.

* *another proof*

The above version is weird from lecture (confuse such an easy problem with lots of unnecesary stuff and so-called lemma..), just see [this one](https://web.engr.oregonstate.edu/~glencora/wiki/uploads/dijkstra-proof.pdf). (Note the rhs of the third formula is $l(Q)$ instead of $l(Q_x)$)

* analysis

O(V)+O(V extract-min)+O(E)+O(E decrease-keys) depends on the realization of the queue

Array $O(V^2)$, binary heap $O((V+E)\lg V)$, fibonacci heap $O(E+V\lg V)$.

* unweighted graphs

w=1 for all edges. BFS- breadth first search. BFS is Dijkstra alg, use FIFO queue instead of priority queue, and the relaxation step, if d[v] is infinite, then d[v]=d[u]+1 and add v to the end of the queue.

The complexity now is $O(V+E)$. Finding the shortest path weights is the natural byproduct of BFS.

## 17 shortest path II

negative weights in this section

### Bellman-Ford algorithm

it computes shortest path weights $\delta(s,v)$ from source to all vertices allowing negative weights and reports negative weight cycle.

> intialize d[v]=infinity d[s]=0
> for 1 to |V|-1: for each edge (u,v), d[v]=d[u]+w(u,v) if rhs is less
> for each e in E: if d[v]>d[u]+w(u,v), report negative loop error

complexity: $O(VE)$, slower than Dij alg.

correctness:
assuming no negative loop, only need to show at some point d[v] is right as $\delta(s,v)$.
let p=v_0-v_1...-v_k=v as a shortest path with the fewest possible edges. (p is a simple path no zero weight cycle)

induction: assume by induction that $d[v_i]=\delta(s,v_i)$ after i round.  During the next round, we relax $(v_{i-1},v_i)$,   then $d[v_i]$ is correct.

*another messy proof...*, just use this Lemma: after k loops, for any node u, d[u] is the smallest length of path from S to u contains at most k edges. (by induction)

corollary: if BF failes to converge after V-1 rounds, then there must be a negative cycle.

To find the cycle, just go backward form the first violation vertex when checking, until a vertex is met twice, there must be a negative cycle. And one can add assitant vertex connect to everyone else avoiding missing negative cycles when it is disconnected from s. For the two alg finding negative cycles, see [here](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.86.1981&rep=rep1&type=pdf)

New goal: find all pairs of shortest-path

### Linear programming

LP Problem: given m times n matrix A, m-vector b, n-vector c, find n-vector x that maximize $c^T x$, subject to $Ax\leq b$ (space region), or determine there is no such x.

solution list: simplex (exponential in the worst case but practical), ellipsoid (in polynomial time but not practical), interior point (polynominal time and practical), random sampling

Linear feasible problem, whether there is x exist obeying the condition. (no easier than LP)

difference constraints: linear feasibility problem, where each row of the matrix A has one 1 and one -1, and everything else in the row is 0.

constrain graph: convert simple constrains $x_i-x_j<w_{ij}$ into node x,y and edge w. Hence |V|=n |E|=m.

Thm: if  constraint graph has negative cycle, then the LP problem is infeasible.

Proof: consider a negative cycle, add all constraint we get $0\leq negative$.

Thm: no negative loop in G means the constrains is feasible.

Proof: add a vertex S with 0 edge to all vertex. it has shortest path from S. run source shortest path algorithm, to get d[v], such a d[v]=$\delta(s,v)$ is just a solution for the constraints.

Actually we can label d[v] for all vertex as zero at the beginning and avoid using the imaginary v0 node.

### VLSI layout

problems: arrange chips of some shape, and gurantee the minimum gap d (focus on 1D).

Bellman-Ford solution also maximizes x1+...xn subject to x<=0 and different constraints, also minimize max-min of x , the latter gives the solution of VLSI problem here.

1) show negative intitial in $d[s,u]$ cannot do better than all 0(easy), and then show positive initial in d[s] also cannot do better (the positive path never get passed otherwise the first segment belongs to delta due to optimal structure of shortest path which is positive violating constraints, so it do no help to increase delta) 2) (proof on the minimization of data span) suppose the minimum of constrains graph is x_i, and consider the path from v0 to vi, where x1 is the maximum. assume y1-yi, prove it is larger than xi, done.

## 18 shortest path III

Recall single source shortest path problem:

1. unweighted: BFS $O(V+E)$
2. positive weighted : Dijkstra's algorithm $O(E+V\lg V)$
3. general weights: Bellman-Ford $O(VE)$
4. DAG: topological sort $O(V+E)$

All pairs of shortest pathes:

1. unweighted graph: |V| times BFS $O(VE)$
2. positive weighted: run Dijkstra |V| times $O(VE+V^2\lg V)$

for general case, V times of Bellman-Ford is not fast enough. if the digram is dense, the $O(V^4)$. 3 algorithms for this general case. 

target output: n by n matrix with element as shortest path weights

### dynamic programming

let A be weighted adjacency matrix with elements as weights. def $d_{ij}^m$ is the shortest path from i to j with no larger than m edges. i.e $d^{n-1}=\delta(i,j)$ if no negative cycles.

the recurrence $d_{ij}^m=\min_k\{ d_{ik}^{m-1}+w_{kj}\}$.  Note zero edge to oneself is definied to make the recurrence nifty.

> use dynamic programming to fill the matrix $d_{ij}^m$ n-1 times

$O(V^4)$ no better than Bellman

recurrence and matrix multiplication, replace + with min and replace times with +. Namely, we have a new system of algebra $D^{(m)}=D^{(m-1)}\otimes A=A^m$. $A^0=I$  with zero in diagonal and infinity anywhere else. The algebra structure here is semi ring.

D&C to calculate the power of matrix. $O(n^3\lg n)$. Also detect negative loop: diagonal has negative value.

* Floyd-Warshall algrithom

def $c_ij^{(k)}$ is the weight if shortest path from i to j with intermediate vertices in set {1,2...k}. so $\delta(i,j)=c_{ij}^{(n)}$, $c_{ij}^{(0)}=a_{ij}$ which is weight matrix. The recurrence is $c^{(k)}_{ij}=\min \{ c_{ij}^{(k-1)}, c_{ik}^{(k-1)}+c_{kj}^{(k-1)}\}$. 

Using dynamical programming to build $c^{(n)}_{ij}$, the complexity is $O(V^3)$.

* Transitive closure program

given a graph, comput $t_{ij}$ which is 1 is there is a path or 0 if disconnected. solution: V times BFS $O(VE)$

the complexity for transitive closure is the same as matrix multiply

* Johnson's algorithm

graph reweighting, give function h: V to R, reweight each edge (u,v) in E by $w_h(u,v)=w(u,v)+h(u)-h(v)$. then for any two vertex u and v, all path (not only shortest path) are reweight by the same amout. (trivial)

if we find a function h make all weight positive we can then run Dijkstra. To satisfy this condition, we need $h(v)-h(u)\leq w(u,v)$ , we are faced with difference constraints. so basically the function h is the shortest path from vertex to imaginary vertex v0. 

Complexity: $O(VE+V^2\lg V)$.

## TODO

- [x] binary trees basic operations implementation
- [x] dynamics of randomized binary search trees
- [x] all kinds of balanced trees implementation 
- [ ] induction proof of height of red-black tree
- [x] OS rank algorithm in 11
- [x] is delete of skiplist robust against continous deletion?
- [x] fast fouries transform
- [x] NP completeness
- [ ] Primality Testing
- [x] small space complexity to reconstruct LCS (D&C)
- [x] Fibonacci heap for priority queue
- [x] union-find structure
- [x] bucket sort
- [x] all kinds of heaps: binominal, fibonacci, pairing
- [ ] linear programming
- [ ] graph theory in general
- [x] breadth and depth first search in graph
- [x] shortest-path-tree in 16
- [x] max flow
- [x] compute delta in all case in 17(even with negative loop) (cut all downstreams for their distance are negative infinity)
- [x] DAG-topological sort