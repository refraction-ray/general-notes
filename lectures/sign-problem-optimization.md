# VMC Renaissance

**From differentiable programming to easing sign problem**

*2019 Aug*

*Quick recap and preparation for group meeting*

*not an organized or self-contained note, just some recaps and short comments with several links*

## Differentiable programming

Neural network is not about network, it is all about optimization on a scalar! (sounds super simple? The history of science and engineer is also all about optimization on some scalars.)

[What is differentiable programming](https://www.quora.com/What-is-Differentiable-Programming)

Auto differentiation (AD): actually has very long history, but it was overlooked for a long time. See [this post](https://justindomke.wordpress.com/2009/02/17/automatic-differentiation-the-most-criminally-underused-tool-in-the-potential-machine-learning-toolbox/) for its amazing prediction and the "dark age" of AD ten years ago.


Between symbolic and numerical differentiation, see [this post](https://loopvoid.github.io/2018/10/15/%E8%87%AA%E5%8A%A8%E5%BE%AE%E5%88%86%E6%B3%95/) for an intro. (The example shown in this post is not so good since it cannot show the difference on how the different directions require different set of forward values.)

For forward AD, one only need to keep the value of last layer, but if the input parameters are huge, you need to keep track each one for the AD process. For backward AD, one need to keep all forward values on nodes.

Applied to QR SVD and eigen decom of real matrix with well defined expression.

If the object is real, than complex decomp AD is also possible. [1907.13422](https://arxiv.org/pdf/1907.13422.pdf)

Technically, a ML library contains two essential parts, general acceleration(GPU accelerated linear algebra and vectorization to make linear algebra as compact and as often as possible), and auto differentiation.

Now there are codebase for source to source AD (meta programming), like [tangent](https://github.com/google/tangent). In some sense, the source to source version of AD is more like symbolic differentiation in a readable way.

The implementation of such AD infrastructure includes two aspects: abstraction on construction and compuatation on the graph and implementation of derivatives on all basic elemental operations.

AD is actually very interesting and illuminating in various fields. It provides an all-in-one optimization method and it is nearly a free lunch with the utilization of modern ML framework. It is universal and fast, meanwhile, it is so easy to implement. And, guess what, optimzation is all science and engineer about. This is why ML is so powerful, and most of the times, the power is puerly from AD infrastructure (derivative lists for primitive operations, graph flow, linear algebra acceleration and optimizers like adam) for scientific work beyond ML community.

AD has also came to condensed matter physics field, as shown by arXiv:1903.09650. I will update the paper list on AD in physics [here](https://github.com/quclub/Paper-reading/issues/38).

* Some more references

[An extended collection of matrix derivative results for forward and reverse mode algorithmic differentiation](https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf): dictionary on derivative forms of linear algerbra operations

[A Review of Automatic Differentiation and its Efficient Implementation](https://arxiv.org/pdf/1811.05031.pdf): including topics on checkpoints, higher order derivatives and so on

[autodiff.org](http://www.autodiff.org/)

## Wavefunction positivization

1906.04654

Marshall sign rules in Heisenberg model : [notes](https://www.escholar.manchester.ac.uk/api/datastream?publicationPid=uk-ac-man-scw:1a7072&datastreamId=POST-PEER-REVIEW-PUBLISHERS.PDF). Opposite sign of amplitude for spin configuration with different number parity of spin ups. Can be easily shown by directly considering $\lang \psi\vert H\vert\psi\rang$.

MPS input as wavefunction stick to local unitaries composed quantum circuit, the whole can be treated as the wavefunction after transformation

Case study model: 8 sites two-leg ladder with J1J2 Heisenberg model plus possible four spin exchange terms

Subtleties:

* Perfect smapling of MPS
* object function with sampling
* complex SVD gradient [tf issue 13641](https://github.com/tensorflow/tensorflow/issues/13641), extra gauge symmetry of matrix U and V, make backprop somewhat subtle: [so](https://math.stackexchange.com/questions/644327/how-unique-are-u-and-v-in-the-singular-value-decomposition). Note in 1903.09650, all involved tensors are literally real ones. This utility is the key of application of AD on general tensor networks!!

## Sign problem alleviation

A list of papers on allevating sign problem in Monte Carlo recently: [list](https://github.com/quclub/Paper-reading/issues/18)

* 1907.02076

To learn sketch on worldline QMC, see Troyer's paper on NPC of sign problem: [paper](https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.94.170201). For a more integrated introduction, please refer to Assaad's [paper](https://pawn.physik.uni-wuerzburg.de/~assaad/Reprints/assaad_evertz.pdf).

The single particle basis transformation doesn't have univeral features agnostic of system size.

Average sign determines the uncertainty (inverse) of the result and is given by $e^{-\Delta E}$.

Model: doped Fermion Hubbard model

Small size system: ED + [Nelder-Mead optimization](http://wuli.wiki/online/NelMea.html) (gradient free)

Large size system: The authors uses numerical differentiations in QMC simulation to locate the best single particle basis transformation (with random *not stochastic* gradient descent).

* 1906.02309

Object: quantity characterizing non-stoquasticality of Hamiltian matrix. (Not always correlated to average sign), norm of the non-stoquastic part of Hamiltonian, positive or zero for off diagonal terms.

Transform: on site (single qubit gate?) 

Sign Easing is NPC for 2-local Hamiltonian and simple on site Clifford operation.

Optimizer: conjugate gradient descent, Signal processing 89, 1704 (2009)

Parameter is sparse since the author only consider translation invariant case, indicating that all on site (two sites?) transformation are the same.

## Summary and comments

**VMC** may become the new paradigm everywhere in computational phsycis community in the short future. VMC is very old and used to find ground energy for complex systems. But in modern context, VMC can be viewed separately: **Variational**, which can be improved with all the modern optimization techniques especially AD infrastructures; **Monte Carlo**, which is reasonable samplings following relevant probability. MC is important to evaluate obeject function since the input may be too large to evaluate completely. It is not like batch update in ML, in which case the bulk of data is not so large and you can in princinple evaluate the overall loss function if you like (with terrible efficiency, though). However, for quantum data, say quantum wave functions, it is impossible to evaluate the whole thing for each amplitudes, this task is restricted by exponential large Hilbert space. So to evaluate some object function with such large input, we must utilize MC technique to sample and approximate the object function value. This is VMC Renaissance.

With this point of view, we can rethink the two cases study mentioned above:

1.  **V**: variational parameters: elements of quantum circuits unitaries, variational methods: AD, variational object: loss function characterzing sign structure of wave function.

   **MC**: perfect sampling from MPS, and related special treat on gradient of object function

2. **V**: variational parameters: elements of unitary for single particle basis transformaton, varitional methods: Nelder-Mead and numercial finite difference based "random" gradient descent, variational object: energy of corresponding hard core boson model characterizing how sever sign problem is in the fermion model.

   **MC**: some traditional QMC approach to evaluate the energy expectation for hard core boson model

There are also many other interesting works can be classified into VMC paradigm. The famous example including works on neural states starting from [this science paper](https://science.sciencemag.org/content/355/6325/602.abstract). Actually, the idea is super simple, the whole story is traditional VMC approach, with the only difference on wavefunction ansatz: RBM or neural networks here. 

Other examples are arXiv:1903.09650. It is about AD on tensornetworks, but if you look closely, it falls in VMC paradigm more or less. V here is AD together with quasi-Newton LBFGS algorithm for optimizer and tensor elements as input. The object is of course energy of matrix product wavefunction. But there is no explicit MC, since the energy operator can be easily traced in iPEPS without sampling to approximate.

Anyways, what I wanna convey here is combination of modern variational approaches with Monte Carlo sampling method is a very promising future direction to explore both in physics and ML fields.