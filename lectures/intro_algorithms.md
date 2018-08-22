#  Introduction to Algorithm

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

complexity: $\Theta(n \log n)$ due to $\Theta(\log n)$ time for heapify a node, in step 1, one need heapify around n/2 nodes and in step 2, one heapify only root node each time sort one element.

See [this](http://javabypatel.blogspot.com/2017/05/analysis-of-heap-sort-time-complexity.html) for algorithm  on heapsort and see [wiki](https://en.wikipedia.org/wiki/Sorting_algorithm) on more aspects of sorting algorithms.

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

comlexity analysis: using counting sort for each digit (one or several a time). assume each binary as b bits long, split it into single bit a time may not be optimal, so we set the split as b/r. digits are each r bits long. $T=b/r\Theta(2^{r}+n)$, where n is 2 this case a fix number for counting system. Now one only need to minimize T in terms of r (optimal when $r\sim \log n$). Finally we have $T=O(\frac{b n}{\log n})\sim O(\log k)$ where k is the range of the number.