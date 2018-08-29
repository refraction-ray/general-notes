# Reading Notes on *Introduction to Algorithms*

*Reading notes on the book, serving a supplemental materials of the lecture notes, start writting from Aug 28, 2018*

*Note all the emphasize part in the lecture notes are all actually contents from other materials*

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

## TODO

- [ ] matroid stuff
- [ ] matrix stuff