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
Proof: Let v be the node with smallest $\delta_{f'}(v)$ such that, the lemma breaks, and let u be the predecessor of v in Gf', we have $\delta_{f'}(v)\=\delta_{f'}(u)+1>\delta_f(u)+1$
if (u,v) in Gf, $ lhs> \delta_f(v)+1$; if (u,v) not in Gf, we are augementing s-v-u in this case, namely (v,u) in Gf, $lhs=\delta_f(v)$.

define C(p)=C(u,v), every time there must be a critical edge be augmented, and for every E, two times augmentation push u far away from s 2, therefore V*E times is limited.

Dinic's algorithm: $O(V^2E)$, augmented all path at the same time, O(V) augmented round

application:
* elimination from sports standing
* person-task assign: bipart graph matching: find subset of edges such that no person for two tasks and no task is assigned two people. s link one side and t link the other side with weight 1. if the max flow is K, that means the max matching is also K.
* cover: bipart graph, each edge is connected at least one dark nodes (nodes may be dark). problem: given some edges to be covered, how many nodes should be colored. min cover K is the same as max flow K for the same graph.

## TODO

- [ ] 16.4,16.5
- [x] matrix stuff
- [ ] Van Emde Boas tree
- [ ] 21
- [ ] 22.5
- [ ] 26.3-26.5
- [ ] 27
- [ ] 29
- [ ] 30-35

## REFERENCE

* THE BOOK
* a solution manual: [link](http://sites.math.rutgers.edu/~ajl213/CLRS/CLRS.html)