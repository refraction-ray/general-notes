# Compiler

*from Oct 13, 2018, everything about the compiler* 

[TOC]

## Syntax

### Backus Normal Form

V ::= expression (ts and vs), teminals are recoginzed by the zero occurence in left side, [] zero or one times, {} zero or many times in exteded BNF.

### preparation set

Null set of non-termianls: non-terminals which can generate empty in several steps

Follow(A): A is a nonterminal, the set of terminals that may follow A, may include lamba. i.e. all terminals that can occur on the right side of A in any sentence from the language.

First(V): the set of terminals can be as a beginner from something derived from V.

Follow and First function can defined on arbitrary long sequence of terminals and nonterminals. But these things are too many and should be computed on need instead of save as dicts.

The iterative logic to do the tasks above: coumpte_first() on arbitrary sequence, using compute_first to fill_first() on single nonterminals to get the array, fill_follow() which utilized compute_first()

Extension first_k(V), the k beginning sequence of V derivation

### top down parser

leftmost derivation, LL

LL(1), one look ahead, for A: X_1…X_m, if lookahead is in First(X_1…X_m) or if lambda in First(X_1…X_m) then the look ahead may be in Follow(A) too. Everytime such a lookahed is matched, the prediction is made. If no conflict for such a scheme, the grammar is said to be LL(1). 

`T[A][t]` A is a nonterminal to be matched and t is the lookahead token, Error also be a matrix element, the element is one of the production rule.

LL Driver pseudocode: 

```
stack.push(s)
a = first token
while !stack.empty()
	x = stack.pop()	
	if x is nonterminal and T[X][a] = X:Y_1...Y_m:
		stack.push(Y_m...Y_1)
	else if x is terminal and x=a:
		a = next token
	else:
		raise SytaxError
		
```

LL(1) table T and a driver can form an LL(1) parser. The driver is simply implemented by a stack storing the unmatched nonterminals. Such an implementation can simply include the semantic action also sharing the same symbol stack.

LL(1) conflicts: common prefixes and left recursion. 1) Production rules with same lhs share the same prefix of rhs (can be resolved by introduce a new nonterminal for the prefix). 2) the first symbol of rhs is the same as lhs (can also be resolved by introducing new nonterminals.). 

$\{[^i]^j\vert i\geq j\geq 0\}$ is not a lang in LL(k) for any k.

### bottom up parser

rightmost derivation, LR

There are two tables in shift-reduce parser. 1) Action table, (terminal and nonterminal symbols, state(int) ) = (S, A, R_n, Error) , A is accept while S is shift. 2) Goto table, (terminal and nonterminal symbols, state(int)) = state number or error

LR driver pseudocode

```
# S is the start state (int)
stack.push(S)
t = first token
while true:
	s = stack.pop()
	switch(action[s][t])
		case A:
			finish successfully
		case S:
			stack.push(Goto[s][t])
			t = next token
		case R_i:
			# Ri: X=Y_1...Y_m
			stack.pop(m states for rhs)
			s' = stack.top()
			stack.push(Goto[s'][X])	
```

#### LR(0)

configuration: dot in somewhere in the rhs of production rules, construct closure for configuration set: substitute the leftmost nonterminal right to the dot with its own rules with dot in the leftmost of rhs.

Sarting with the augmenting rule as configuration set $closure(\{S=\cdot \alpha \$\})$. The configuration dot auto advance lambda.

goto function given an old configuration set and a look ahead symbol, find the successor configuration set. (just advance the dot and recaculate the closure). If return is empty set, it implies syntax error. 

CFSM: characteristic finite state machine. build by start form start configuration set and try different tokens.

Some notations: 

* simple or prime phrase: phrase containing no smaller phrase, which are directly derived from a nonterminal.
* sentential form: something derived from start may including both terminal and nonterminal
* handle: the leftmost simple phrase for a sentential form.
* right sentential form: sentential from derived by rightmost derivation.
* viable prefix: (of a rightmost sentential form) the prefix doesn't extend beyond handle.

P: $S\rightarrow 2^Q$, S is a set of configurations from CFSM states, Q is the set of A or R_i or S. P(s) is defined as

$P(s)=\{R_i\vert B\rightarrow \rho\cdot \in s \& P_i:   B\rightarrow \rho\}\cup \{S\vert A\rightarrow \alpha\cdot a\beta \in s for a \in V_t\}$.

G is LR(0) iff for each s $\vert P(s)\vert=1$. More than 1 give two types of conflict: shift-reduce and reduce-reduce.

LR(0) grammar exists for any lang expressed in LR(k) grammar! (Note the difference between the power to parse any lang and any grammar)

#### LR(1)

The configuration is with one more lookahead l, which is the first element after the corresponding rules. Not the one directly following the dot!! This time the start configuration is with the token lambda. The closure function this time is defined as below.

```
closure(s):
	s' = s
	while(more configuration has beed added)
		if (B=d\dot Ar, l in s')
			Add all A=\dot g, u where u \in First(rl)
	return s'
```

similarly, LR(1) machine namely the finite state automaton can be constructed. The lookahead in shift phase, serving as the element behind the dot. All terminals are considered is such case due to the closure(). And the lookahead in reduce case, where there are more than 1 state with dot rightmost, to resolve the reduce reduce conflict. Note to differentiate the lookhead in configuration (which is the possible terminal after the full rule) and the lookhead in practical when parsing, which is the next token. They are totally different things.

#### SLR(1)

The FSM is the same as LR(0) case, no lookhead is included in the configuration sets. The lookhead is only used when there is some conflicts. The function P is defined as 

$P(s,a)=\{R_i\vert B\rightarrow \rho\cdot\in s, a\in Follow(B), P_i=B\rightarrow\rho\}\cup\{ S\vert A\rightarrow \alpha\cdot a\beta\in s, a \in V_t\}$.

G is SLR(1) iff for all state s and lookahead a, the set P is no larger than 1.

#### LALR(1)

LALR(1) first construct the finite state machine as LR(1) and then merge all states with the same configuration set omitting lookahead in the configurations.  We call the merged state the *core* of the corresponding states in LALR(1). The merged states also include the lookahead info, but with the union set of the original ones in LR(1) machine.

In terms of practical construction, the first LR(1) machine then merge policy is too resource consuming to use. The alternative approach is to build LR(0) CFSM first and propagate the lookaheads. There are two types of propagate links: one with normal dot advance and the other with closure expansion.

To sum up, for LR(0) and LR(1), you both need token input. In LR(0) case, such input is only used in goto table, which is states cross vocabs. The action table is only determined by stack top states instead of input token. The parser is general irrespective with LR(0) or LR(1) scheme as long as the action and goto table is well defined. The runtime parser case nothing about how these tables are generated.  The input token is only consumed in shift case. In reduce case, such token plays role in resolving conflicts in LR(1) case, and plays no role in LR(0). Or in other words, the action table for LR(0) is the same for each input token. The lookahead in LR(1) is the inner state to diff degenrated state in LR(0) case so that reduce-reduce conflicts may be resolved. As for the implementation of all above, i.e yacc, the only hard part is an efficient implementation of LALR(1) lookahead sets. How to correctly (not NOLALR) and efficiently get the lookheads for configurations in each states? 


### Earley's algorithm

general approach to CFG parsing. time and space consuming (beyond linear).

## Semantic processing

### General

* Action controlled vs Parser controlled semantic stacks
* Intermediate reps (IR): tuples or trees

### Symbol Table

* Implementation: Hash table vs binary tree
* Store names as a whole char string to save space
* Symbol table for nested scopes: Individual table for each scope: tables in a stack (scope stack) vs. a single symbol table, name is associated with a scope number, while the hash is still applied to names, the result is a hash table with linked list of terms with same name but different scope number. Multipass compiler in general use symbol table for each scope.
* Issue: import and export mechanism to overwrite the scope, oveload of names, forward reference, implicit decalration

### Storage Organization

* Static Allocation: static and extern in C

* Stack Allocation: activation record(AR), frame in python (control infor and local variables.)? Managed by a run-time stack. Within AR, each variable is label as the address offset determined at compile time. Better to allocate literals in functions in a static area called literal pool. For dynamic array, only a dope vector including the size and bounds of the array is put into AR. The array itself can allocated in the run-time stack atop of current AR.

  Static in C: lifetime across calls. Allocated as the literals outside AR. Coroutine like functions interface can be implemented by more than 1 stacks in heap allocation. Variables sharing by several local funcs: cactus stacks.

  Display register: address of ARs. Restore the display after each call.

  Procedure level AR vs. Block level AR

* Heap Allocation: new in C++, deallocation strategy is complicated. No dellocation policy. Explicit deallocation. Implicit deallocation (garbage collection): ref counts, mark and sweep garbage collection, work only be done when the heap space is full. Heap obj may have a refback pointer. Compaction may also performed together, to make the still-in-used heap obj together.

Program memory layout (bottom to top, not heap is developed from the top): Reserved Locations, Program Code, Literal Pool, Static Data, Similar things for libraries, Stack space,       Heap Space.

Formal Procedures: func as parameters. Pointers in AR, static link (for static structure, in the src sense, to restore the env context) and dynamic link (always point the previous one in the runtime stack). 

### Processing Declarations



## Misc

*a temporary memo here*

* type compatibility and scoping rules are context-sensitive
* ambiguity of CFG is undecidable, (by reducing this to PCP, see [this post](https://cstheory.stackexchange.com/questions/4352/how-is-proving-a-context-free-language-to-be-ambiguous-undecidable))
* python is not CFG, the lexer part is stronger than a simple reg matcher
* html is not CFG
* uncomputable: whether a given CFG is ambiguous, whether two grammar are the same

## Reference

1. C. N. Fischer, R. J. LeBlanc, Crafting a Compiler with C