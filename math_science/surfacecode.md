# A brief introduction to surface codes

[TOC]

## Overview

There are various quantum error correction schemes in literature such as Steane code or Shor code. These schemes often require several physical qubits to make a logical qubit. The tradeoff is that the threshold error rate for individual physical qubit should be very low (in the order of $10^{-5}$). Instead, surface code, emerged from the idea of toric code model by Kitaev[^Kitaev1997], might require thousands of physical qubits to compose one logical qubits. Theoretically speaking,  the minimum number of physical qubits involved in making one logical qubits is 13, however such an encoding is very fragile to errors. The advantage of a large number qubits encoding is the high tolerance on single physical qubit errors, the acceptable operation error can be as high as $1\%$. We will give a brief introduction on the surface code scheme in this note. The readers should have basic knowledge on quantum computation. The main reference of this note is [^Fowler2012].

## How to protect a quantum state

### Stabilizer form

In general, we have two ways to uniquely determine a quantum state. The first approach is to specify all amplitudes on a set of complete basis, i.e. wavefunctions. There is another approach to describe quantum states which is more suitable for error correction scenario. Instead giving all amplitudes for each basis, we can describe the given state as eigenstates of a set of compatible Hermitian operators (compatibility indicates these operators are commuting with each other). And these operators are called *stabilizer* to the quantum state. Namely, a quantum state can be determined by the eigenvalues ($\pm 1$) of stablizers. Not all states can be described efficiently in this fashion, but as we can see in the following sections, to implement surface code, it is enough for us to protect relevant states we are interested.

### Toric code stabilizer

We are now giving the description on the set of compatible operators to protect the surface code, which we denote as toric code stabilizer. The physical qubits array is shown as below. The white circles are data qubits while the black circles represent measure qubits.

<img width="60%"  src="https://user-images.githubusercontent.com/35157286/58756300-8f3f7f00-8528-11e9-9daf-0d6fd7af628f.png">

The measurement qubits are used to stabilize and manipulate the quantum state of the data qubits, and they are ingredients of toric code stabilizers. There are two types of measurement qubits, “measure-Z” qubits, colored green and “measure-X” qubits, colored yellow. These measurements are Z syndrome and X syndrome of errors, which give results of $\Pi_i Z_i$ and $\Pi_i X_i$, respectively. These product operators are so called toric code stabilizer. It is easy to show such operators commute with each other, say $Z_1Z_2Z_3Z_4$ and $X_1X_4X_5X_6$, this compatibility comes from two facts: local operators from different qubits commute, each Z product and X product operators must share even data qubits. 

Now, we can measure all these stabilizers again and again, with measurement period less than the coherence time of physical data qubits. By measurement, the quantum state must fall into the eigenstates of all these stabilizers, which we call the state *quiescent state*. If any errors happen on data qubit, these measurement output must be different somewhere. Say if one data bit got a Z flip, then the two X measurement outputs are changing sign while the two Z measurement related to such data qubit stay the same. This can be easily shown by $X_1X_2X_3X_4(Z_1\psi)=-Z_1(X_{1234}\psi)=-X_{1234}(Z_1\psi)$. Therefore, by the position and type of sign change from the measurement, we can infer the what and where the error is, and thus further correct it. As long as the error is sparse in the qubits array, we can accurately locate all errors and thus protect the quantum state as the same as the state in the beginning. Note such a protection is onlydesigned for further quantum computation set ups, and is not a universal scheme to protect any quantum states. Actually you cannot protect any quantum state by this setup since not all states have efficient and compact stabilizer formalism.

## How to make a logical qubit

It is not enough if we only have a protected quantum states, we need more freedom to do something interesting, more Hilbert space and yeah, qubits actually. 

<img width="60%" src="https://user-images.githubusercontent.com/35157286/58756297-7c2caf00-8528-11e9-9b3c-71cc2bcb1b2b.png">

### Array form

Let's directly see how can we treat the above 41 data qubits together with 40 measurement qubits as a logical bits. Wait a moment, 41 data qubits could have $2^{41}$ orthogonal quantum states, but 40 measurements of $\pm 1$ only give $2^{40}$ of them. Yes, such a setup has extra freedoms related to fixed measument outputs, and this is the beginning to construct the logical qubits. Maybe we are not protecting states by stabilizers, instead we are protecting qubits!

The operator $X_L=X_1X_2X_3X_4X_5$ is interesting since the measurement outcome after applying $X_L$ is the same as original quiescent state. Recall that a X flip would lead two neighbor Z measurements changing sign. After applying $X_1$, the measure qubit between $X_1,X_2$ gives opposite output. However, after applying $X_2$ in the following, the measurement bit flips again. By this logic, an operator string across the plaque will change nothing in terms of stabilizer measurement. In the same time, $X_L\psi$ and $\psi$ couldn't be the same state since five underlying data qubits have been flipped. Similarly, we can construct string operator $Z_L$ as shown in the figure which also keep all measurement outputs the same as before.

Maybe you are wondering whether we can construct more string operators that keep the measurements the same, yes, you can. But they are all linearly dependent on the above two which can be decomposed as one string operator in the above paragraph and the product of several stabilizers, which trivially give $\pm 1$.

Now, you can show the algebra between the two string operators $X_LZ_L=-Z_LX_L$. It is trivially true, since the only common part is $X_3, Z_3$ which gives a negative sign when interchanging with each other. Together with the other trivial fact that $X_L^2=Z_L^2=1$, we conclude that the two string operators here are actually Pauli algebra, the algebra of qubits. The 2 dimension Hilbert space for the logical qubit is also consistent with our initial judge, the extra freedom $2^{41}/2^{40}=2$. Therefore, by applying stabilizer measurements again and again, we are protecting more than quantum states. Instead, we are protecting a qubit, a so called logical qubit that is robust against nosie and error.

### One hole form

The above approach to construct logical qubit is straightforwad. But here we introduce another arrangement of the qubits array for logical qubits with holes. The approach is more suitable for further quantum gates construction on logical qubits, and better for scaling.

The basic arrangement of physical qubits is shown as below. The main idea of creating logical qubits in this approach is to create holes of measurement qubits, which can leave extra freedom. Note there is a hole of Z measure bit. The upper solid line represents the Z-cut edge. There is no requirement on other three directions of the array.

<img width="40%" src="https://user-images.githubusercontent.com/35157286/58757292-8c4e8980-853c-11e9-930d-f3d52b09e5e5.png">

Again, since one stabilizer measurement has been turned off, there is extra freedom which can finally be identified as logical qubit. In the similar fashion as the above section, we can identify $X_L=X_1X_2X_3$ and $Z_L=Z_3Z_4Z_5Z_6$ as shown in the figure as Pauli algebra which is independent of other stabilizers and keep the outputs of stabilizers the same while changing quantum state. We can actually create larger defects by turning off more measure qubits to make the code much robust against error, but we won't cover such configurations here. And in this note, we only focus on the minimal defects set up for error correction ($d=4$).

### Two holes form

We can generalize the above idea to two defects configuration, which could independent of edge cuts and more flexible. The basic set up is shown as below, now we don't need any boundaries of the array.

<img width="40%" src="https://user-images.githubusercontent.com/35157286/58757388-45619380-853e-11e9-8620-f44513242aed.png">

The basic idea is similar. As we turned off two Z measure bits, there are actually four extra freedoms with two sets of Pauli algebras as shown in the one hole form. But we only operate them in a correlated fashion, that we only care a subspace of them which is just one logical qubits. The corresponding $X_L$ and $Z_L$ string operators for the two holes form logical qubits is shown in the above figure. Note $X_L$ is equivalent to the product $X_{L1}X_{L2}$ up to a sign. Namely we actually pick $X_{L1}X_{L2}$ and $Z_{L2}$ as Pauli algebra for our new logical qubit from the four operator set $X_{L1}, X_{L2}, Z_{L1}, Z_{L2}$.

For all one hole form and two holes form qubits, we can construct another type with X measure defect similarly. It finally turns out to be important to have two different types of logical qubits for CNOT braiding on the logical qubits.

### Error correction scaling

After having the knowledge on constructing logical qubits by surface code, we give a brief review on the error correction efficiency and succeed probability. The figure below is the numerical simulation of surface code error corrections. The x axis is p, namely the typical probability of single physical qubit errors in each step. The y axis is $P_L$, i.e. the final error rate of the logical qubit after error correction. The threshold probability is $p_{th}\approx 0.57\%$ and d is the diameter(number of operators in the product of non-trivial string operator, width or height of the array for each logical qubit, or the perimeter of the hole for two hole form of encoding).

<img width="70%"  src="https://user-images.githubusercontent.com/35157286/58757021-80f85f80-8536-11e9-94eb-8441bf028592.png">

As from the figure, for physical qubits whose error rate $p<p_{th}$, the larger the array is, the smaller the error rate for the logical qubit. Actually the scaling rule is 
$$
P_L\propto (\frac{p}{p_{th}})^{d/2}.
$$
Instead, if the error rate for each single physical qubit is larger than the threshold, then surface code make no sense, since the more physical qubits involved, the larger probability error for logical qubits, the error rate is even larger than the error rate for single physical qubit.

Therefore, to utilize the power of surface code for quantum computation and error correction, the first priority is to lower the error rate of single physical qubit. Even if the error rate is smaller than the threshold, it is always a better idea to further reduce the single qubit error rate instead of blindly using more physical qubits as one logical qubit, which may take tens of thousands of them if the physical qubit error is close to the threshold. The estimated number of physical qubits for one logical qubit strongly dependents on the single bit error rate, which is shown schematically below. As we can see, for physical qubit error rate as small as $5*10^{-4}$, we may still need thousands of physical qubits for one qubits for a reliable universal computation system. Considering the number and quality of the state-of-art superconducting qubits we are contructing nowadays. Basically we have made progress on less than $1\%$ qubit for last thirty years.

<img width="70%" src="https://user-images.githubusercontent.com/35157286/58757107-79d25100-8538-11e9-8125-0bbfce0eb260.png">

Finally we briefly discuss the error correction method after the error detection. In surface code system, we don't actually apply X or Z quantum gates on error data qubits to correct it and recover the state we want. Instead we record all errors and their locations by classical softwares where we put "virtual" error correction single qubit gates in software level only. And these gates can be delayed evaluated and only applied at measurement time. Every further gate on this error qubits should firstly commute with the virtual error correction gate such that the error correction gate can always move back instead of real evaluation on the physical system. Such software assisted error correction scheme is more robust against quantum noise, otherwise, the error correction gates themselves can make new errors. 

## How to operate the logical qubit

It is not enough for us to have a logical qubit robust to noise if we want to build a quantum computer. For quantum computation, we need to operate the logical qubits such as initialize it, measure it and apply quantum gates on it. We now discuss how to implement these operations on logical qubit level. Of course, we assume that we can easily carry out all these operations on physical qubits. Otherwise, we have no way to control the logical qubits indirectly.

The obvious operations are X and Z gates on single qubits, which is just given by the string operators defining logical qubits. For other operations, it would be a little complicated.

### Initialization and measurement

For qubit state initialization, we only discuss initialization to ground state of $Z_L$ for Z measurement defect two hole qubits. The initialization is easy, apply measurement on all stabilizers for a full qubits array and then just quickly turn off two Z measurement qubits, since one of the Z loop is the Z operator for final logical qubits, we know its eigenvalue rightly before creating the defect. If $Z_L$ gives 1 instead of $-1$, we can apply the string operator connecting two holes $X_L$ on the system, and flip the logical qubit to the groundstate of $Z_L$.

For measurement on logical qubits, the idea is somewhat similar. If we want to measure $Z_L$ value of Z defect type logical qubits, then we just turn on the Z measurement bit in the hole, and we are done with the output. However, if one wants to measure in X direction of Z defect qubits, it is a bit involved. We should first turn off all Z measurement qubits in the $X_L$ string connecting two holes which seperate all data qubits in the string from any measurement qubits. Now we can measure all these unstablized separate data bits along x direction, the product gives the measure output of $X_L$. It is worth noting, after the measurement, the logical qubits is destroyed and now we can fill up the qubits array again and are ready for new qubits initializations.

Since there are operations for turning off some stabilizers, it seems to be error prone at first. However, after careful analysis on possible errors and how can we trace them in the process with several stabilizers off, we find that it still holds the full power of quantum error correction. Namely, maintaining the surface code error detection can be done by careful combinations of measurements before and after a stabilizer manipulation with the same fault tolerance as in the rest of the surface code array. We won't cover the details of error detection in such process in this note.	

### Moving qubits

Since many quantum gate operations require moving holes for the qubits, we will first summarize on how to move qubits in the 2D qubits array. The moving process is sketched as the figure below. Again, we only focus on error free version of qubits moving. The error happens in the moving process could in principle be well handled by careful design and analysis.

<img width="70%" src="https://user-images.githubusercontent.com/35157286/58757887-71811280-8546-11e9-989f-6cec7bfce12f.png">

Basically, we turn off one Z measurement qubit and measure the separate data bit in x direction $X_6$. Finally we turn on the stabilizer in the original position of the hole, and the hole is effectively going down by one step. Accordingly, both logical operators $X_L, Z_L$ have been changed. 

Note there may be a sign change for the definition of logical X and Z operators which can be determined by $X_6$ and the product of two Z stabilizers respectively. The sign change stuff is similar to Heisenberg picture. In the moving process, say, the $X_L$ string operator is redefined by multiplying $\pm 1$, for example $X'_{L}=X_{L}X_6$. Suppose a qubit, and you change the definition of operator $X$ to $-X$, what is the most easy way to keep the system giving consistent results as nothing changed? Then answer is to apply $Z$ operator to the underlying quantum states. So we may need to apply $X'_L$ or $Z'_L$ to the logical qubits such that the qubits moving process is consistent. In a word, a qubit moving is effectively equivalent to apply X or/and Z operators on the logical qubits. Moving the hole in some way in physical qubits is just applying some gates on the logical qubits. This observation is the basis why braiding could play the role of CNOT gate for logical qubits. These extra operators to make everything consistent are called as *byproduct operators*.

### Braiding

So called braiding is a special type of qubits move, where a defect move in a cycle in the space and come back to the original position, within the moving trajectory, there could also be other defects coming from other logical qubits. It is this process that creates entanglement between different logical qubits and finally realize the CNOT gate on the logical qubits.

We will look at a z defect qubit move around one hole from x defect qubit. It turns out the z defect qubit is the control bit while the x defect qubit is the target of CNOT. It is worth noting that, on the other hand, when x defect qubit go around one z qubit, the x defect qubit is still the target line. Since go around is relative and only types of qubit are important to determine which qubit is the target.

Here we only see one example of operation change which make $X_1I_2$ to $X_1X_2$ (see Fig below), by collecting enough transformations of such Pauli matrix product, we can finally conclude that the braiding with two qubits are just a CNOT on logical qubits (possibly with some extra Z or X gates on single qubit, which is the byproduct operators as we mentioned in qubit moving section). 

<img width="100%" src="https://user-images.githubusercontent.com/35157286/58762350-e9b8f980-8581-11e9-82ea-62d695b384ec.png">

As we can see from the above figure, this time $X'''_1$ operator is not the same as original $X_1$ operator up to $\pm1$. Instead, there is $X_2$ remaining after contracting the X loop with stabilizer squares. Therefore, when commuting $X_1I_2$ and the "braiding operation", we have "braiding operation" and $X_1X_2$ (omiting some extra single qubit byproduct operator). Namely, the braiding operation $C$ follows $C^{\dagger}X_1I_2C=X_1X_2$. Similarly one can derive the transformation on other four by four matrix. By these transformation relations, $C$ could be uniquely determined as $C=CNOT$.

Furthermore, if we want to implement CNOT gates on two Z or two X defect qubits, then we need assistance ancilla qubits in different type. In other words, CNOT gate can only directly implement by two different types of qubits.

### Hadamard and S, T gates

We won't cover these implementations in this note, interested readers could find the implementation in [^Fowler2012]. The implementation of S and T gates also requires the ancilla state injection, which is implemented by somewhat $d=1$ qubits and more error prone. So we have to introduce distillation circuit to "purify" these ancilla state.

The most important thing to know is that we can indeed implement all universal quantum gates in principle by surface code scheme on logical qubits, which renders surface code one of the most promising route towards universal quantum computers in the future.

## Outlook

### Physical implementation

The motivation for developing the surface code is, of course, to find a realistic and practical physical implementation for a quantum computer. Necessary conditions for surface code physical platform include:

1. It must meet the requirements for single-qubit and two-qubit gate and measurement fidelities, which should be lower than threshold probability.

2. It must have the ability to assemble and integrate a large number of nominally identical qubits.

3. Physical qubit coherence time should be larger than $1\mu s$.

The first two conditions are easy to understand. We will explain the third point a bit. Since the error correction scheme of surface code needs classical hardware to track and merge the error instead of doing error correction directly on the quantum system, then the processing time of classical calculation should match the measurement cycle time of surface code. Otherwise, one cannot locate the errors timely. But modern classical CPUs have relatively fixed frequency of $GHz$ order, so $200ns$ is a reasonable cycle time for surface code, which requires a longer lifetime of physical qubit of $1\mu s$ order.

These considerations appear to point to superconducting circuits as one of the best candidates for implementing the surface code: All of the physical and operating parameters for superconducting circuits fall into the ranges discussed here. There are clearly significant challenges in achieving sufficient gate and measurement fidelities though. 

### Estimation for Shor algorithm

If we could really build a quantum computer by surface code scheme, how many qubits and how long does it take for us to decompose 2000bit integer, which is a clear signal of quantum supremacy.   I only list the data here instead of a detailed analysis. Of course, such data is highly dependent on the error rate of single qubits. But based on several reasonable assumptions, the result is:

>  For Shor algorithm on 2000 bits integer,  the full quantum computer by surface code needs about $220 × 10^6$ physical qubits, operating for about 1 day.




## References

[^Fowler2012]: Surface codes: Toward practical large-scale quantum computation, Phys. Rev. A **86**, 032324 (2012).
[^Kitaev1997]: Fault-tolerant quantum computation by anyons, Annals Phys. **303** 2-30 (2003).