# Reading Notes on *Introduction to Algorithms*

*Reading notes on the book, serving a supplemental materials of the lecture notes, start writting from Aug 28, 2018*

*Note all the emphasize part in the lecture notes are all actually contents from other materials, and in this note I will also include external materials and topics*

* toc
{:toc}
## Greedy Algorithm

### Huffman codes

used for compress data. idea: use vary length code for different characters which can be charactrized as binary tree, where 0 for left and 1 for right. optimal code at least is **full** binary tree, where all non-leaf nodes have two children. Therefore a file with C chars has a tree with C leaves and C-1 internal nodes. (proof by induction) The quantity to measure the totbits is $B(T)=\sum_{c\in leaf} f(c)d_T(c)$.

Given array f[c], n is the number of distinct chars, maintain a priority queue Q keyed on f

> first create the n terms for c as leaves
>
> recursively merge the two lowest frequent obj, the new node's frequency is the sum of old pairs

* Complxity: $O(n\lg n)$ for binary heap as Q.
* Correctness

Lemma 1: say x y with the lowest freq, there must exist a code where x y only diff in last digit

Proof: move x,y to the lowest level of the tree, and prove the difference in B is negative.

Lemma 2: optimal tree for C-x-y+z T', when replcace z to x,y, it is the optimal tree for C.

Proof: first we have B(T)=B(T')+f(x)+f(y). Supose such T is not optimal for C, there exist B(T'')<B(T). Reduce x,y in this tree to T we get T''', thus B(T''')=B(T'')-f[x]-f[y]<B(T)-f[x]-f[y]=B[T'], and this contradicts that T' is optimal tree.

### Theoretical foundations

Definition on matroids:

an ordered pair M=(S,l): 1) S is a finite set 2) l is a nonempty family of subsets of s, called the independent subsets of S: if B in l and A is subset of B, then A must also in l. We call l is hereditary.  3) if A,B in l and |A|<|B|, then there is some element x in B-A such that, A union x is in l. 

eg. for a undirectd graph G=(V,E), S=E, and A = acylic subset of E (forest)

*TBD*

## Heaps

Note search is always slow in heaps, so if you want to invoke the decrease-key operation, better pass the pointer in

### binominal heaps

comparison between different heaps:

![](https://user-images.githubusercontent.com/35157286/44724660-f34e9600-ab05-11e8-8c42-867faa7155d1.png)

* binominal trees

$B_k$ defined recursively, with root and children $B_0,...B_{k-1}$.

Properties (for $B_k$, node as depth 0): 

1. there are $2^k$ nodes
2. height is k
3. $C_k^i$ nodes at depth i
4. the root has max degree k

All can be proved by induction

* binominal heaps

heap H is a set of binominal trees satisfying following properties below

1. each tree obeys min-heap property
2. at most one tree for each k ($\lg n+1$ in total trees)

every node maintain degree value, the roots are in a linked list (denoting as root list, degree of root increase order sorted). sibling[node]: the right node.

* operations

  * find minimum

  assuming no key as infinity (search on the root list)

  > y=None, x= head[H], min = infty
  >
  > while x != None:{ if key[x]<min, then min=key[x], y=x} x=sibling[x]
  >
  > return y

  $O(\lg n)$

  * merge heaps

  idea subroutine: merge two $B_{k}$ make a new $B_{k+1}$.

  for H1 and H2, resort the link list of root lists by degrees, merge heap with same root degree until all with diff degree, again $O(\lg n)$.

  * insert node

  create a $B_0$ and call merge check, $O(\lg n)$

  * extract min

  find in in rootlist, remove it, and merge the child and remaing heap, $O(\lg n)$

  * decrease key

  bubbling up the key as binary heap, take the time of height which is $O(\lg n)$.

  * delete key

  decrese to -infinity and then extract min. $O(\lg n)$

summary: no advantage over binary heap if you don't need merge heaps. distinguish left and right strictly

### Fibonacci heaps

large constant factors and hard to implement, theoretically very fast.

children of one node are double-linked circular list, root list is also circular double linked list. degree of node is maintained as well, another boolean mark is kept (default false), too.

The tree is accessed by min(H) pointer. the number of nodes is saved externally as n[H]. t for number of trees and m for number of marked (mark is True) nodes. The potential of the heap is $\Phi(H)=t(H)+2m(H)$. assume D(n) as the upper bound of max degrees of root node.

Fibonacci heap is a collection of unordered binomial trees.

* Insert node

merge x as single heap and the remaining heap in the root list, $O(1)$

* find min

return the pointer of the H, $O(1)$

* merge heaps

just merge the root list $O(1)$

* extract min

remove min node and move all its children root into root list, then we consolidating the root list until every root in root list has a distinct degree value.

amortized $O(D(n))$

* decrease key

if the new value of x is less than its parent, we cuts x and its subtree off and treat x as a new root. Mark field: x was a root and then x was linked to another node and then one child is removed by cuts, we label x as True now. As soon as another child is cut, we cut x make it a new root. cuts all marked nodes the way up and change them to unmarked.

$O(1)$

* delete node

decrease key to minus infinity and extract min.

The *upper bound* of $D(n)$:

$D(n)\leq [\log_\phi n]$ where $\phi$ is the golden ratio.

Lemma 1: for any node x where d[x]=k, let $y_1,...y_k$ be its children in the time order, then $d[y_1]\geq 0$ and $d[y_i]\geq i-2$. Consider the extract-min call, where tree of the same degree be merged and the decrese key call, where only one child lost is allowed (otherwise, such node should be cut to move to the root list) 

Lemma 2: $F_{k+2}\geq \phi^k$ and $F_{k+2}=1+\sum_{i=0}^k F_i$ (proved by induction) 

Lemma 3: for any node x, k=d[x], size[x]$\geq F_{k+2}\geq \phi^k$.

Proof: suppose $s_k$ is the minimum size for nodes with degree k, then we have

$s(x)\geq s_k\geq 1+1+\sum_{i=2}^k s_{i-2}.$  Then use induction to prove.

Therefore, the max degree of any node $D(n)$ , such node has size at least $n\geq s_{D(n)}\geq \phi^{D(n)}$ , hence $D(n)$ is boudned by $\lg n$.

Intuition understanding of Fibonacci heap: [so](https://stackoverflow.com/questions/19508526/what-is-the-intuition-behind-the-fibonacci-heap-data-structure)

## Matrix

* definition and properties (trivial to physicists....)
* solving systems of linear equations

focus on A nonsigular case.

LUP decomposition: find matrix L,U,P, st. PA=LU. where L is unit lower-triangular matrix, U is upper trangular and P is permutation matrix. We can have LUx=b for the equations. define y=Ux, first we solve Ly=b. Then Ux=y. The traigular matrix can be solved by forward and back substitution. $\Theta(n^2)$.

So the remaining problem is to attack LUP decomp. Gaussian elimination. implemented it recursively based on 28.22. $\Theta(n^3)$.

* inverse

view the inverse problem as n Ax=delta_i linear equation problem. $\Theta(n^3)$.

Thm 1: multiplication is no harder than inversion

Thm 2: inversion is no harder than multiplication

see proof in the book (bored to type these matrix.... basically you need some clever construction on 3n by 3n matrix, the proof of thm2 is a bit harder, first construct positive definite symmetric matrix and find the inversion can be got by recursion, and finally show non-positive-definite case can be reduced to the above case)

Schur complement $S=C-BA_k^{-1}B^T$

## Parallel

dynamic multithreading: shared memory. 

pseudocode keywords: spawn - subroutine conexecute at the same time as its parents. sync - wait until all children spawn are done.

* underlying DAG

parallel instruction stream: DAG

vertices are threads, spawn edge vs continuation egde vs return edge.

denote $T_p$ is the running time on P processors. $T_1$ is the work of serial time (no parallel in this basic case). $T_{\infty}$ is the critical path length. (the longest path in the DAG)

eg, Fib(4) (assuming every threads cost the same time unit) $T_1$=17 (work A: the first spawn, work B: the second spawn, work C: sync and sum). $T_\infty=8$. 

Lower bound on $T_p\geq \frac{T_1}{P}$ and $T_p\geq T_\infty$.

define $T_1/T_p$ as the speedup on p processors. if this value is order p i.e. $\Theta(p)$, we say it is linear speed up. (perfect linear speedup, if the constant is 1) one never get super-linear speedup where the constant is larger than 1 in this model. 

Max possible speedup is $T_1/T_\infty$. this value is defined as parallelism $\bar{P}$.

* scheduling

Map computation to p procs. Done by runtime system. Online schedulers are complex. This section will only focus on off-line scheduling.

Greedy scheduler: do as much as possible on every step. 1) complete step: over p threads ready to run. execute any p of them. 2) Incomplete step: less than p threads ready, execute all of them. 

Thm: a greedy scheduler executes any computation G with work $T_1,T_\infty$ in time $T_P\leq T_1/P+T_\infty$ with p procs. therefore the greedy schedule is 2-competitive!

Proof: number of complete steps: no larger than T1/p. in terms of incomplete step, let G' be subgraph that remains to be executed, the critical path length is reduced by 1 every incomplete step. this implys the number of incomplete steps as $T_\infty$.

One get linear speedup when number of procs $p=O(\bar{P})$.

* algorithms: matrix multiplication

Matrix multiplication (n by n): D&C

> partion A and B into 2 by 2 submatrix  # O(1)
> spawn * 8
> sync
> add (C, T, n): 
> 	spawn *4
> 	sync

Analysis: let $M_p(n)$ be p-proc time for mult and $A_p$ for add. 

Work: $A_1(n)=4A_1(n/2)+\Theta(1)$, $A_1(n)=\Theta(n^2)$. similarly $M_1=\Theta(n^3)$.

critical path length: $A_\infty = A_\infty(n/2)+\Theta(1)$, $A_\infty=\Theta(\lg n)$. similarly, $M_\infty(n)=M_\infty(n/2)+\Theta(\lg n)$, $M_\infty=\Theta(\lg^2 n)$.

parallelism: for multiplication $\bar{P}=\Theta(n^3/\lg^2 n)$. $10^7$ for 1000 by 1000 matrix.

get rid of the assistance matrix T and only use C with spawn muladd function, and sync when 4 of them are calculated, so the recurrence would be $M_\infty(n)=2M_\infty(n/2)+\Theta(1)$, and the parallelism is $\Theta(n^2)$, which is fast in practice because we use less space.

*TBD*: $\lg n$ critical path for matrix multiplication

* algorithms: sorting

merge sort

> spawn two subarrays as merge sort and sync, then merge after sync

Work: $T_1(n)=2T_1(n/2)+\Theta(n)$, $T_1=\Theta(n\lg n)$.

cirtical path: $T_\infty(n)=T_\infty(n/2)+\Theta(n)$, $T_\infty=\Theta(n)$.

parallelism: $\bar{P}=\Theta(\lg n)$. tiny parallelism

parallel merge (merge A and B into C, l>m)

> p-m(A[1,...l],B[1,..m], C[1,..n]):
> find j st. B[j]\leq A[l/2]\leq B[j+1] using binary search
> spawn p-m(A[1,...l/2],B[1,...j],C)
> spawn p-m (A[l/2+1,...l], B[j+1,...m],C)
> sync

$PM_\infty(n)\leq PM_\infty(3/4n)+\Theta(\lg n)$, $PM_\infty=\Theta(\lg^2 n)$. $PM_1(n)=PM_1(\alpha n)+PM_1((1-\alpha)n)+\Theta(\lg n)$. substitution to show it is $PM_1(n)=\Theta(k)$ (the assumptionis $ak-b\lg k$ for one side proof)

now merge sort with p-m: $T_\infty=\Theta(\lg^3 n)$. This gives $\bar{P}=\Theta(n/\lg^2 n)$. And the best to date one is $n/\lg n$

## flow networks
### general notations
directed graph, two distinguished vertices as source s and sink t. each edge (u,v) in E with non-negative capacity C(u,v)

condition: netflow f  less than capacity on each edge, and flow conservation for all nodes f(u, V)=0 and skew symmetry f(u,v)=-f(v,u)

assumptions: no self loop edge is allowed. if there is two node with direct cycle, we introduce a new node to break this cycle (psotive flow is the sam as net flow then). if there is no edge, f=0.

aim: maximize the flow $|f|=\sum_{v\in V} f(s,v)=f(s,V)$

the application of implicit summation: $f(X,Y)=-f(Y,X)$, $f(X\cup Y, Z)=f(X,Z)+f(Y,Z)$ if there is no intersection of X and Y.

Thm: |f|=f(V,t)

Proof: |f| = f(s,V) = f(V,V) - f(V-s,V) = f(V,V-s) = f(V,t)+f(V,V-s-t) = f(V,t)

a cut is a bi-partition of nodes in a graph, (S,T), s in S and t in T. if f is a flow on G, then the flow across the cut is f(S,T), such cut not necessarily a real cut line. and capacity C(S,T) defined only on s to t direction, without counting on the reverse direction edge

Thm: value of flow for any cut is no larger than the capacity of any cut (max flow, min cut)

Lemma: for any flow f and any cut (S,T), we have |f|=f(S,T)

Proof: f(S,T) = f(S,V)-f(S,S) = f(S,V) = f(s,V)+f(S-s,V) =(flow conservation) f(s,V) = |f|

Residual network $G_f(V,E_f)$: strictly positive residual capacity. $C_f=C-f (u,v)$, you need double the edge, since f(v,u) = -f(u,v). usage: see whether augmenting path in Gf exists (a path from s to t)

Residual capacity of an augmenting path is $C_f(P)$ which equals the minimum value of edge capacity on the path of Gf

### Algorithm

Fork Fullkorsion Algorithm: 
> set all f(u,v)=0
> while an agugmenting path exists
> do augment on f by Cf

Thm (max flow min cut): the following statements are equavilent
1. |f|=c(S,T) for some cut (S,T)
2. f is a maximum flow
3. f admits no augmenting paths
Proof: 1 imply 2, 2 impy 3, 3 imply 1.
1:2 |f| is no larger than C(S,T), |f| cannot be increased when some capacity is meet
2:3 if there were an augmenting path, the flow value could be increase
3:1 suppose f admits no augmenting path, define set S is the reachable points in Gf, T=V-S
we can show |f|=c(S,T)

complexity analysis: the scheme of picking up augmenting path. if the path is worse selected (maybe DFS), the worst case takes very long time.

Edmonds-Karp algorithm: BFS path is a shortest path in Gf from s to t where each edge is count 1, only O(VE) augementations in the worst case. overall $O(VE^2)$ 

analysis: 
Lemma: $\delta_f(v)$ is the length of shortest path from source s to vertice v, such value does not decrease.
Proof: Let v be the node with smallest $\delta_{f'}(v)$ such that, the lemma breaks, and let u be the predecessor of v in Gf', we have $\delta_{f'}(v) =\delta_{f'}(u)+1>\delta_f(u)+1$
if (u,v) in Gf, $ lhs> \delta_f(v)+1$; if (u,v) not in Gf, we are augementing s-v-u in this case, namely (v,u) in Gf, $lhs=\delta_f(v)$.

define C(p)=C(u,v), every time there must be a critical edge be augmented, and for every E, two times augmentation push u far away from s 2, therefore V*E times is limited.

Dinic's algorithm: $O(V^2E)$, augmented all path at the same time, O(V) augmented round

application:
* elimination from sports standing
* person-task assign: bipart graph matching: find subset of edges such that no person for two tasks and no task is assigned two people. s link one side and t link the other side with weight 1. if the max flow is K, that means the max matching is also K.
* cover: bipart graph, each edge is connected at least one dark nodes (nodes may be dark). problem: given some edges to be covered, how many nodes should be colored. min cover K is the same as max flow K for the same graph.

## FFT

polynomial $a_kx^k$. 

evaluation: naive quadratic time, of course easily to linear time

Horner's Rule $A(x)=a_0+x(a_1+x(a_2+...))$. 

addition: A(x)+B(x), easy linear time

multiplication: A(x)B(x), naive quadratic time, goal $O(n\lg n)$

multiplication is the same as convolution

representation of polynomial: coefficient vector (slow in multiplication), roots (impossible in addition), samples (n points) (slow in evaluation)

Vandermonde matrix V $V_{jk}=x_j^k$, coeff to sample: S=VA quadratic time, A is the coeff vector. sample to coeff A=$V^{-1}S$, quadratic thought inverse is cubic, as you only need once to get inverse

D&C idea

> divide even and odd coeff as Aeven(x) and Aodd(x)
>
> recursively comput Aeven(y) and Aodd(y) for y in x^2
>
> A=Aeven(x^2)+xAodd(x^2)

analysis T(n,|X|)=2T(n/2,|X|)+O(n+|X|), initially |X|=n,|X| never goes down

nth roots of unity, then every recursive reduce |X| by half, this is FFT

by FFT, we transform vector and sample rep. of polynomia via FFT $O(n\lg n)$, such that the evaluation, addition and multiplication all be done in linear time which the total cost is nlgn.

in other words, it is just convoluton on one side is a simple times in other side in terms of Fourier like trans. so the intro of all contents in this section is redundency for physicts...

## van emde doas trees

storing interges 1 to u in trees, ops: insert, delete, successor, try in $O(\lg\lg u)$ time.

application: network router.

$T(u)=T(\sqrt{u})+O(1)$.

naive: size u boolean array. const time for insert and delete, successor O(u) in the worst case.

split the universe into clusters with size square of u, and save clusters summary 1/0. (sth like the skiplist) by this scheme, insert is O(1), successor is square root u time.

decompose x as $x=i\sqrt{u}+j$. high(x)= floor(x/sqrt(u)), low(x) = x mod sqrt(u)

recurse datastructure V as small vEB and summary structure as vEB

> Insert(V, x):
> ​	 Insert(V.cluster[high(x)],low(x))
>	Insert(V.summary, high(x))
>	
>successor(V,x):
>	i = high(x)
>	j = successor(V.cluster[i], low(x))
>	if j is infty:
>		i = successor(V.summary, i)
>		j= successor(V.cluster[i], -1)
>	return i,j

for insert $T(u)=2T(\sqrt{u})+O(1)$, substitute u as lg u and use master method, still cost lg u.

further improvements, store minimum for each vEB, then successor reduce to two recurvise calls. Moreover, store the max. then successor reduce to only one recursive call.

space O(u) to O(n\lg\lg u), by hashtable for array in cluster. it can be further reduced to O(n) by cutoff the recursive structure early. space recursive in the original case $S(M)=O(\sqrt{M})+(\sqrt{M}+1)S(\sqrt{M})$.

![](https://i.stack.imgur.com/qF5k5.png)

Let elaborate all things by graph, see this combined with algorithms.Trick, every time there are two recursive calls, one of them must be in const time.

## complexity theory

P: solvable in polynomial time; NP: decision problems solvable in nondeterministic polynomial time. eg 3SAT

NP completeness: NP & NP hard. NP-hard: X is NP-hard if every problem y in NP reduces to X.

Reudction from problem A to B: there is polytime algorithm converting A inputs to equivalent B inputs. (same yes and no answer)

Namely NP completeness is the critical point between NP and NP hard.

Just show 3SAT can reduce to the problem for the proof of NP hard. (Cook-Levin Thm states 3SAT is NP complete) (logic gate anyway in computation)

EXP: problem solvable in at most exponential time $2^{n^c}$.

R: problems solvable in finite time.

Almost every problem is not in R. (Most decision problems are not in R.)

Proof: program is just some binary string or just an integer. decision problem is a function map input (a number: data transform to binary) to 1,0 (equivalent to real number).  

### Approximation Algorithms

size n problem has approximation ratio $\rho(n)$, for any input, algorithm produces a solution with cost C such that max(C/Copt,Copt/C) is less than rho(n).

approximation scheme: takes input $\epsilon>0$, and for any fixed $\epsilon$, the scheme is a (1+\epsilon) approximation algorithm. 

Polynomial time approximation sheme(PTAS): Poly in n but not necessarily in epsilon. Fully PTAS is $O(n/\epsilon^2)$.

* Vertex Cover

Problem: Undirected graph G, find minimum set of vectices cover all edges. 

Max degree heuristic. The covered while unchoosed nodes loose degree by 1 each time. 
Worst case of this: optimal  n=k!,  our solution can be k!(1/k+1/(k-1)+...1) which is approxmated as k!logk. i.e. $\rho(n)=\Omega(\lg\lg n)$.

Approx Vertex Cover: 
> set C as empty set
> loop: Pick edge (u,v) arbitrarily, add u,v to C, and delete all edges from u or v
> return C

Proof: such an algorithm is a 2-approxiamtion algorithm. Let A be the set of picked edges. i.e. we have 2|A| vertex. We only need to show Copt is at least |A|. Since we need to cover every edge including all edges in A. One have to pick one vertex for each disjoint edge in A. Done.

* set cover

Problem: Given a set X and a family (possibly overlapping) subset Sm of X, pick some Si to make union of them as X, minimize the the number of Si.

Approximation Set Cover: $\ln n+1$ approximation algorithm. Pick the largest subset and remove elements in it and repeat until all elements all deleted.

Proof: Assume there is a Copt, with susets |Copt|=t. Let Xk be the set of elements in the iteration k. Xk of course can be covered by T. one of them covers at least |Xk|/t elements.
So the alg is gonna pick a set with more elements than |Xk|/t. So $|X_{k+1}|\leq (1-1/t)|X_k|$. SO $|X_k|\leq e^{-k/t}n$.

* partition problem and approximation scheme

Problem: Set S of n items with weights s1,s2...sn,(descending order) partition S into A and B, to minimize max of weights of A and B. if we define the sum of S as 2L, then the obeject is just the sum weight of one set and substract L. worst case is 2-approximation.

Approximation partition: 
Define m = floor(episilon)-1
First phase: Find an optimal partition A' B' of s1 to sm (brute force search $2^m=2^{1/\epsilon})$. (PTAS instead of full PTAS)
Second phase: iteratively add elements to the small weights set A or B due to the sorting order

Proof: assume w(A) is the larger on at the end, the ration is w(A)/L. suppose Sk is the last element added to A. This may be added in the 1st or 2nd phase. 1) if k is added to A in the first phase, namely all remaining elements are added to B, it is simply an optimal solution. 2) if k is added in the second phase, we know w(A)-Sk is less than w(B) at that time, i.e.$w(A)-S_k\leq 2L-w(A)$, $2L\geq (m+1)S_k$, so $w(A)/L\leq (L+S_{k}/2)/L\leq 1+\frac{S_k}{(m+1) S_k}=1+\epsilon$.

### fixed-parameter algorithms

idea: want exact results, but confine exponenttial dependence to a parameter. parameter K(x) is non negative interge where x is some input. we want polynomial in problem size but only exponential in parameter K. 

Parameterized problem

* vertex cover

k-vertex cover: given graph G=(V,E), integer K, problem: is there a vertex cover S in V whose size is no larger than K? Natural parameter: K.

Brute force: try all $C_V^k$ subsets of k verties. $O(EV^k)$. exponents of size depneds on k, bad case.

Parameterized problem is Fixed-Parameter Tractable(FPT) if it can be solved in f(k)n^O(1), where the exponent of the size n doesn't depnedent on k.

Bounded search tree algorithm: 

>  consider any edge (u,v)
>
> guess u in S or v in S
>
> delete u and incident edges, K--, recurse on G'K'
>
> do the same thing for v
>
> return OR of the two branch for yes or no answer
>
> base case: k=0, return whether |E|=0

Reveal complexity by recursive trees. $V2^k$, done.

Thm: you can solve problem in $f(k)n^c$ if and only if $f'(k)+n^{c'}$ if and only if it has a kernelization.

Proof: if n<=f(k): f(k)n^c<=f(k)^c+1; if n>f(k): f(k)n^c<=n^{c+1}, in both case f(k)n^c<=f(k)^{c+1}+n^{c+1}.

if f(k)>=n: already kernelized case, if n>f(k): FPT alg runs in )(n^{c+1}) time, output a canonical yes or no input.

* Kernelization

Polytime alg. converting input (n,k) into equivalent small input (n',k'). (small means n'<f(k) )

back to k-vertex cover problem. 1) if there is a self loop, delete the node and incident edges, decrease k and delete the node and incident edges of the node. 2) if there are many edges connecting two nodes, delete n-1 of such edges. 3) if vertex of degree larger than k, one must add it in the cover set. We now have a bounded graph. Now |E| should be less than k^2 to say yes. delete degree zero verteices. |V|<2k^2 to be judged. The remaining size is of order k^2, we are now having polynomial kernel!

The total complexity:  O(VE) (actually O(V)) for kernelization, bounded search tree O(k^22^k). Best to now: O(kV+1.274^k)

*  connection to approximation

take optimization problem with integer aim. consider decision problem: OPT<=k, parameterize by k.

Thm: optimization problem has EPTAS, then decision problem is FPT.

EPTAS: $\rho=f(1/\epsilon)n^{O(1)}$.  usual way to use: no FPT proof lead to no EPTAS shceme.

Proof: say the optimal problem is maximazation. run EPTAS with $\epsilon=1/(2k)$. it takes f(2k)n^O(1) time. This is just the correct answer, as the error bar is small than unity. (somewhat dejavu.)

* Metric TSP

Metric: with triangular inequality. Traveling Sales Man. It has 3/2-approximation alg.

2-approx alg: find the minimum spanning tree first, and do a DFS traversal, remov duplicates by just skip them. So it is small than two times weight of MST. And the cost of MST is bounded by the OPT by considering that one edge removing from OPT is a tree cost more than MST.

Lemma: cost of OPT in subset is no more than cost of OPT in the parent set in metric TSP.

perfect matchings of complete graph can be found in polymial time.

3/2-approx alg: Pick all odd degree nodes of MST(even number of nodes), find minimum weight perfect matching of them add these edges back to the MST. Then one can draw Euler circuit. Note the cost of matching is less than half of the cost of OPT in odd degree graph, we are now having 3/2-approxiamation.

## Distributed system

Algorithms run on multiple processors. 

Distributed Networks: $\Gamma(u)$ set of nerighbors of u, communication over edges.

### Synchronous distributed algorithms

[Slides](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-046j-design-and-analysis-of-algorithms-spring-2015/lecture-notes/MIT6_046JS15_lec19.pdf). For each node, only know local names by degree, have no idea on the global graph structure. Complexity focus on the number of round and communication cost.

* Leader Election

Simple case: all vertices are directly connected to all others. Thm: there is no determinstic alg guranteed to elect a leader in such case. So symmetry breaking is necessary.

UID: unique number assigned for each node, totally-ordered set.  for leader election, take 1 round and n^2 bit message.

Randomness: just auto generate UID randomly. 
Lemma: uid pool size is $r=n^2/(2\epsilon)$, then probability that all pick different uid is $1-\epsilon$. Such alg take expected time is $1/(1-\epsilon)$. 

Leader election on the rings, $O(\lg n)$ round.

* Maximal Independent Set

Choose a subset of nodes to from MIS. No two neighbors are both in the set, and cannot add any more nodes without violating independence. MIS(not unique, even not with the same number of nodes for diff answer, locally MIS actually).

Each pocess output in or out to indicate whether itself is in MIS. unsolvable for some graph with determinstic alg. Application: main route of communication.

Luby's MIS Algorithm:
In, out, active(initially), inactive reduce graph each phase. 
Round 1: pick a random value r every phase within n^5 and send it to all active neighbors. receive values from all neighbors, if r is max, output in. Announce join mess to all neighbors, any body receive output out. In and out both trun to inactive.

Correctness: obvious for independence and max. 
Complexity: with probability 1-1/n, all nodes decide within $4\lg n$ phases.
Proof: With probability 1-1/n^2, in each phase, all nodes choose different random value. For each phase, the active edges expectation reduce to half. Suppose in each phase w picked largest value of neighbor of w and neighbor of w's neighbor u. The probability is 1/(deg(u)+deg(w)). Probability u is killed is at least $\sum_{w\in \Gamma(u)}\frac{1}{deg(u)+deg(w)}$. The expect number of edge die is larger than $1/2\sum_u\sum_{w\in \gamma(u)}\frac{deg(u)}{deg(u)+deg(w)}$ (note two can be both killed in the same round). rewriten the sum over undirected edges, we can now prove the lemma edges decrease by half every phase. Now you can argue in expectation sense or w.h.p sense.

* Breadth-First Spanning Trees

root of the tree is predefined UID is assigned to every node. Distance equal depth in the tree of all vertex. output of every node is the parent in the tree. Naive: just send mess to its neighbors every round. The receiving should pick one of the sender as its parent.

such alg is nondeterminism but non-randomness. (always choose min UID to make it determinstic). complexity: number of round: max distance from the root. message complexity: O(|E|). some augments: child pointer, distance value.

How to end?  termination algorithm: send by no parent mess, then one can know itself is a leaf. The leaves sending done mess to its parents. And everyone do this if its subtree is done. finally the root know the construction is done.

Appliacations: fast boradcast from the root. or data collection from childrens by convergecasting.

* shortest path trees

now edges are with non-unity weights other settings are the same as above. every node know the weights of incident edges. every node output weighted distance and parents.

Bellman-Ford shortest paths algorithm:
state variables: dist. initially infinity, representing the known shortest path from the root. parent initially undifined, and uid.
> Each round: send dist to all neighbors, receive all dist and do relax, if dist decrease, then set parent as the receiver producing the new dist

Time complexity: n-1 rounds, message complexity: O(n|E|)

Invariant: r round past, every node got their shortest path for no more than r nodes.

Termination: send done when all children done, but all things are changing , so such process may invovled many times. (*complexity and correctness proof?*)

### Asynchronous distributed algorithms

[Slides](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-046j-design-and-analysis-of-algorithms-spring-2015/lecture-notes/MIT6_046JS15_lec20.pdf). No rounds now, processes get out of sync. Communication channel (I/O automaton), a FIFO queue for mess receive and send.

Max value: O(n|E|). real-time upper bounds. d for a channel to deliver the next mess and l for a process to perform its next step. O(diam n d). *Terminate?*

Async version of BFS trees. There is anomaly and not working in naive version. still O(diam d)! others only be faster. 
Improvement: use Bellman-Ford, bookmarking the hopping distance. O(diam n d)

Shortest spanning tree: same alg but exponential complexity!

## Cryptography

### Hash functions
collision resitance. random, determinstic, public. $h: \{0,1\}^*\rightarrow \{0,1\}^d$. poly time for hashing.

Random oracle. (not achievable in practice)
Oracle: on input x of string, if x is not in book, flip coin d (number of bits) times to determine h(x).

Desirable Properties:
One-Way (pre-image resistance): Infeasible to find any x such that h(x)=y
Collision Resistance: infeasible to find x and x', such that their image are the same
Target collision resistance: infeasible given x, to find x' such that their image are the same.
Pseudo randomeness: behavior indistighuishable from random.
Non malleability: infeasible given h(x) to produce h(x') where x and x' are related in some fashion.

CR imply TCR (TCR is weaker).
Suppose h is OW and TCR. change input x'0+x'1 instead of x0, now it is still one way but not TCR. (OW doesn't imply TCR)
construct h' x if |x| is less than n, otherwise normal h(x). (TCR doesn't imply OW)

Application: password save in database, file integrity, abstract of message to be signed, commitment (CR, OW and NM is needed for commitment).

### Encryption

Symmetric key encryption. $c=e_k(m)$, c:cipher text, k: key, m: message. and decryption function $m=d_k(c)$. reversible operation is needed (say permutation or exclusive OR). eg. AES (Advanced Encryption Standard). [Implementation of AES](https://blog.csdn.net/lrwwll/article/details/78069013).

Diffie Hellman Key exchange: $G=F_p^*$ finite field (mod prime p, inversible elements only).
Alice slecet random a. Public g and p. Compute g^a. Bob select b and g^b. g^b^a mod p is the same as g^a^b mod p. (color mixing analogy) Discrete log problem is considered 'hard'.

Diffie Hellman, give g^a, g^b, shoudn't work out g^ab. MIMA: certificate system.

RSA: Alice pick two large secret p and q. N=pq. choose and expoenent e which satisfies gcd(e, (p-1)(q-1)) = 1. (RSA pick 3, 17,65537). Alice public key (N,e). Decryption exponent obtained  using extended Euclidean Alg by Alice, ed = 1 mod (p-1)(q-1). Alice's private key (d,p,q).
Encryption: c = m^e mod N,  decryption: m= c^d mod N.

Thm: (Fermat little Thm) m^{p-1} = 1 (mod p). when p is prime.
Functionality: phi = (p-1)(q-1), since ed =1 mod phi , then ed = 1+ k phi. Two cases: 1) gcd(m, p) =1.$ (m^{p-1})^{kq-k} m = m (\mod p) $, ie m^{ed} = m (mod p)   2) gcd(m,p)=p, m mod p =0. m^{ed} = m

The same argument for q. Together make mod N work.

NP completer & crypto: Is N composite? This is in NP, but unkown is NPC. Is a graph 3-colorable?  This is NPC. Building crpto system based on 3-colorable graph.

Knapsack: S = b_iw_i, bi is 0 or 1, find assignment of bi to make ths euqal. NPC. Crypo based on this fails. (broken quickly). Worst case is not the average case. But factoring is always that hard on the average.

Super-increasing knapsack problem is easy (linear time). $w_i\geq \sum_{i=1}^{j-1}w_j$.

The original broken idea: private key is a super increasing knapsack. via private transformation (wi' = wi*2*N mod M) into a hard general knapsack problem which is the publickey. See [wiki](https://en.wikipedia.org/wiki/Merkle%E2%80%93Hellman_knapsack_cryptosystem) for details on this system and [instance](http://www.math.stonybrook.edu/~scott/blair/Breaking_Knapsack_Cryptosys.html) where it is broken.

* digital sign

correctness, unforgeability

the original attempt (broken): just inverse the encryption process. mutiplicative homo (just times the old tex t whose sign is the product of old sign)! or simply choose sign first and compute the message. 

improvements: use hash of message to sign, fix the two attacks before. (ad hoc security: no broken method no proof on security)

* Message authentification code (MAC)

only one with the same key to check and mac it. just a hash with some salts. Only some case works subtlely. (*special hash scheme with special contance scheme?*)

* Merkel tree

Problem: make sure the integrity of files on the cloud. files hashes as leaves. hashes them binarily. only store the root hash in local storage O(1). check time O(lg n).

## Linear programming

general settings: Some linear ineuqality and an optimal linear expression. (n variables and m constraints and one object) real version (if variable is restricted to integer value, the problem is NP complete)

Standard form: maximize  $\vec{c}\cdot\vec{x}$, subject to $Ax\leq b$ and all x is non-negative.

Certificate of optimality, LP duality. Thm: for stand form LP, there is a dual LP as min $\vec{b}\cdot\vec{y}$ subject to the constraints $A^Ty\geq c$ with y all nonnegative.

Convertion of LP into standard form. Switch sign for most case. Case: x does not follow non-negative property. Solution: replace it as x'-x'' and require them to be non negative. Case: equality constraint. Solution: <= and -<=.

* Maxflow

Constrains: skew symmetry, conservation, capacity. Multiple commodity max flow.

* shortest path

single source path d[v]. triangle inequality. $d[v]-d[u]<w(u,v)$ for all (u,v) in E. the objective max \sum d[v], it is max!!!

### simplex algorithm

Exponential cost. Represent LP in slack form. The orginal variables are call non-basic variable. Make inequalities as basic varibles, where zero is the bound. set all non-basic variables as zero. compute values of basic variables. Select a nonbasic varible whose coeeficient in obejctive is positive, increase the value of it as much as possible without violating any of the constraints. This variable becomes basic and some other variable become non-basic. (objective may change) Hopefulling, all the steps can converge finally.

## Cache

Memory hierachy: L1 to L4, RAM to disk. bandwidth and latency. latency becomes high, but bandwidth can match for diff levels (RAID for disk). blocking the infomation to mitigate latency. 
amortized cost over the block:  latency/block size+1/banwidth. spatial and temporal locality algorithms are need.

External memory model: two level-memory model. M/B blocks in cashe, with size B. Disk is infinite in size and also divided in block size B. Count the number of blocks IO (memory transfers).

Cache-oblivious model: algorithm doesn't know cache parameters B and M. Kick out policy: LRU or FIFO, n - competive.  optimal for all levels at the same time.

* Scanning
sum of an array. alg: just sum. cost: ceil[N/B]+1. (unkown boundary)

reverse the array
> for i in range(N/2): swap A[i] A[N-i-1]

cost O(N/B)

* Divide and Conquer

divide the problem into the problem size fits in O(1) blocks or fits in memory size M

* Median finding

5 column scheme with worst case O(N). remember store media array contiguously.
sort each 5 elements: cost O(N/B+1), recusively MT(N/5), partition O(N/B+1), recusively MT(7/10N). The base case should be used: MT(O(B))=O(1). We can also show MT(N)=O(N/B+1). (root cost determines the total cost)

* matrix multiplication

if two matrix is stored row by row and column by cloumn correspondingly, the total cost for naive dot product is O(N^3/B+N^2).

D&C solution: MT(N)=8MT(N/2)+O(N^2/B+1), though layout of output matrix is somewhat weird. The base case should be used: $MT(\sqrt{(M/3)})=O(M/B)$. finally $MT=N^3/\sqrt{M}B$.

* LRU

$LRU_M\leq 2 OPT_{M/2}$ . 

Proof: divide timeline into max phases of M/B distinct block access. LRU_M(phase)<=M/B

OPT_M/2(phase)>=1/2 M/B

* search

binary search : lg N/B

B- trees: $\Theta(\log_B N)$, optimal for comparison model, but alg have to know the cache parameter, not a cache oblivious.

van Emde Boas layouts: in what order should we store balance binary trees? save tree half, half. upper half with square root of N nodes, and lower half is a bunch of sub trees of square root of N nodes. and recursively follow the layout to store the tree.

cost 2\lg N/ 0.5\lg B = 4\log_B N, fourth of the optimal of B trees. in general cash oblivious weak than cash aware, just like online lose the offline. Actually such structure can be dynamic with memory transfer such order.

* sorting

N inserts into B trees: $\Theta(N\log_BN)$.

Mergesort: MT(N)=2MT(N/2)+O(N/B); base case MT(M)=O(M/B). MT=N/B lg (N/M).

M/B(-1) way of merge sort: MT(N)=M/B MT(N/(M/B))+O(N/B), MT=$N/B\log _{M/B}N/B$ , optimal. cache aware sorting alg.

cache oblivious sorting: can achieve the same bound. require tall-cache assumption. ($M\geq B^2$). $N^\epsilon$ way mergesort (funnel sort).

## TODO

- [ ] 16.4,16.5
- [x] cache related
- [x] distributed related
- [x] matrix stuff
- [x] Van Emde Boas tree
- [ ] 21
- [ ] 22.5
- [ ] 26.3-26.5
- [ ] 27
- [ ] 29
- [x] 30
- [ ] 31-33

## REFERENCE

* THE CLRS BOOK
* a solution manual: [link](http://sites.math.rutgers.edu/~ajl213/CLRS/CLRS.html)