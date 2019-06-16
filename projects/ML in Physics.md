# Review on Machine Learning in Physics

*Start writing from Jan 29, 2018*

*Related paper reading tracking, for more papers, you may refer to the ML in a Nutshell note*

* toc
{:toc}

## Case Study

### Nerural Network to Represent Ground State Wave Functions

**Reference**: Solving the Quantum Many-Body Problem with Artificial Neural Networks, Giuseppe Carleo and Matthias Troyer, Science **355** 6325 p602-606. (with insights from science).

**Keywords**: Restricted Boltzman Machine, Ground State Wavenfunction, Tensor Network, 1(2)-D Ising and Heisenberg Model, Monte Carlo Sampling

**Methods:**

* Construct RBM, and set up the output function (RBM variables(sum over latent variables) -> wavefunction amplitude) , the whole settings can represent wavefunctions in many-body system. And we use 1(2)-D spin system as examples.
* Symmetry considerations: construct NN as wavefunction under certain space group symmetries (here translation symmetry) to reduce the number of parameters. In the same spirit of shift-invariant RBM’s: *Sohn, K. & Lee, H. Learning Invariant Representations with Local Transformations. 1311–1318 (2012). Norouzi, M., Ranjbar, M. & Mori, G. Stacks of convolutional Restricted Boltzmann Machines for shift-invariant feature learning. In IEEE Conference on Computer Vision and Pattern Recognition, 2009. CVPR 2009, 2735–2742 (2009)*.
* Quantum quench in the above model in 1D and solve dynamics using NN. NN is trained at each time $$t$$, the network parameters cannot be reused. 

**Detail Issues**:

* How to make sure the amplitude of wavefunction is normalized? Maybe  a normalize layer is implicitly added in the last step. And I think make a NN description of the output layer and links may be better way to visualize the method.
* Equation E1: maybe reusing letters for different variable is not a good idea...

**Comments:**

* Not so machine learning style, the loss function and the traning process are rooted more in physics than ML. It is not a big suprise one can have convincing wave function from the RBM-NN, it behaves somewhat like how a tensor network mimic the wavefunctions.  But at least, potentially another way to construct wavefunction in exponetial large space by using polyminal large parameter space. As for time evolution part, as long as your aim is to minimize something like the difference with the target wavefunction, it is too hard to fail the task...
* And it is not a RBM, it is just a two layer NN, RBM has strict definition with the loss function as the maximum of probabilty of data set. You cannot call a two layer forwardfeed NN with sigmoid activation as a RBM, they are DIFFERENT strictly speaking. RBM is designed for data transformation and unsupervised training, which is obviously not the case in the paper.

### Quamtum Machine Learning Review

**Reference:** Quantum Machine Learning, *Nature* **549**, 195–202.

**Keywords:** Quantum Information, quantum computation

**Basics:** Four parts are included in this review. 

1. Machine Learning instruct the quantum physics: the design of Hamiltonian, the characterization of quantum states. In c) especially, they mention about how ML now apply to problems like phase transition, groud state in many-body physics.
2. Quantum physics help learning: d)-f) quantum experiments and algorithm as computation engine and g)-h) quantum mechanism help improving learning algorithm.  In g), a proposal on how quantum Gibbs sampling can replace traditional CD algorithm in training the RBM. And further proposal on so-called quantum boltzman machine. In g), they mention works on analogy between NN and tensor network or RG.
3. Relevant experiments in quantum physics to realize ML: photonics, NMR, chian of ions, and superconducting systems.
4. Frontier: complexity, possibility and quantum computer.


**Note:** The two versions of this paper in arxiv are …. just two totally different papers. no idea why. v1 is definitely better. It is trivial that nature or science published verison is always worse that the authors' original manuscript.

### Neural Network vs. RG

**Reference**: arxiv: 1802.02840

**Keywords**: NeuralRG, bijectors,MERA,MCMC

**Methods**:

* Construct the NN from statistical inference perspective, which is orthodox in modern machine learning interpretations. Also, link partition function in physics with the statistical inference is very typical ML based approach recently.  
* Test on MNIST dataset. Of course, everyone like MNIST. The features of the images are too obvious to make the dataset as a reasonable benchmark. 
* Treat the partition function in the statistics inference framework, which is promising. 

**Comments**: 

* This work just coincides my thoughts before, namely, view deep NN as RG flow with the depth of NN as flowing paramter as energy scale. Moreover, we might view the layer of NN as CFT while the bulk of NN as Ads.


### Phase Diagram Sampling by Active Learning

**Reference**: arXiv: 1803.03296

**Keywords**: active learning, Gaussian process

**Basics**:

* Recommendation system of active learning for trial state point outperforms the traditional grid search approach to get the phase diagram. 
* BTW, this paper has a fantastic explanation on E&E trade-off.
* The authors also propose the algorithm for batch recommendation. Recommend several points at one time. To do this, we need extra penalty term to avoid choosing two close points. 

**Comments**: 

* Very interesting idea in this work. There are two types of work on ML to solve physics problem. The first class: some fancy approach representing some function or something else to try to solve the physics problem by ML. In this class, actually we have no confidence on the correctness of the results from ML unless we have previous benchmark from theory or numerical study (which, in trun, weaken the meaning of ML approach in the same problem). Most of the study on ML in physics fall into this class. The second class: utilize ML approach to accelerate the algorithm in physics. This class of approaches have nothing to do with the correctness of final results. It is more or less a recommendation system. You can judge it as more or less effective but nothing to do with right or wrong. In this class, we can solve the physics problem faster without worrying the correctness or intrinsic meaning of the ML models behind. This work and series wirk on SLMC(self-learning Monte Carlo) fall into this class. And the idea on RBM wavefunction is also somewhat within this class.
* This work is the automatic realization of phase diagram calculation. In fact, we never use grid search to get the phase boundary if the calculation time is long. What we actually do is to see the existing points and find a most informative point manually to calculate the phase. Active learning is just doing the same thing but we don't need to do it by ourselves now.

## Misc in Reading and Thoughts

### Interesting or Involved Aspects

* The mathematical foundations for the ability of NN to describe anything are from established representability theorems, which guarantee the existence of network approximatesof high-dimensional functions, provided a sufficientlevel of smoothness and regularity is met in the function to be approximated. 

  **Reference**: 

  1. Kolmogorov, A. N. On the representation of continuous functions of several variables by superpositions of continuous functions of a smaller number of variables. Doklady Akademii Nauk SSSR 108, 179–182 (1961).
  2. Hornik, K. Approximation capabilities of multilayer feedforwardnetworks. Neural Networks 4, 251–257 (1991).
  3. Le Roux, N. & Bengio, Y. Representational Power of Restricted Boltzmann Machines and Deep Belief Networks. Neural Computation 20, 1631–1649 (2008).

### Possible Applications and My Ideas

* Gereneral route to the interplay between physics and ML
  1. Use ML methods to solve problems in physics
  2. Propose new ML approaches based on existed method in physics
  3. Try to understanding mechanism of ML by physics thoughts
  4. ? Quantum Machine Learning, see [case study](#quantum-machine-learning-review)

* Treat partition function $$Z$$ as a distribution and utilize variational inference tools to approximate it, which is an active field of ML. Some methods in ML might hopefully help:

  **Reference:**

  1. Automatic Differentiation Variational Inference [arixv](https://arxiv.org/pdf/1603.00788.pdf)
  2. Black Box Variational Inference [arxiv](https://arxiv.org/pdf/1401.0118.pdf)

* Note the 1 to 1 correspondence between quantum wave function and the NN. There might be something interesting when the quantum wave function plays the role as input data.

  1. Just use the NN ansatz of wavefunction as input, then the NN we train is **higher-order** NN.
  2. Instead of learning wave function amplitude directly, why not learn from the parameters of NN ansatz.
  3. Construct specific NN as novel states in quantum physics.
  4. Quantum wave function itself is a NN. How to combine it with the higher-order NN as some interactive scheme like parallel net or GAN?

* The method to solve the variant size problem in one network.

  1. Is RNN capable for quantum wavefunctions?
  2. What about some dimension reduction preprocessing?
  3. Use the first fixed number of eigenvalue from Hamiltonian or entanglment?
  4. What about the mask or padding mechanism of NN?
  5. What about other potential useful layers? like embedding layer in NLP.
  6. Or what about a brute force feature hashing? [ref](https://arxiv.org/pdf/0902.2206.pdf)

* Extensions of RBM.

  1. Quantum version of RBM especially. Classical RBM is classical Ising model what about transver Ising model as RBM?  See [this paper](https://arxiv.org/abs/1601.02036).
  2. What about spin-1 freedom on each node of RBM, does this contain more information than original binary RBM? (Use softmax or motinominal units, see 13.1 in [this guide](http://www.cs.toronto.edu/~hinton/absps/guideTR.pdf))
  3. What about the real valued RBM, say for XY model? See [this answer](https://www.quora.com/What-are-best-approaches-to-modeling-real-value-continuous-input-variables-in-RBM-pretraining) and reference therein.

* How does error bar (accuracy) of training scale with the quantity and quality of the data.

  1. Say the scaling dependence on the size and the number of images.

* RG vs. DNN

  1. RG as NN: there are already many proposals, like deep belief network, convolutional neural network or normalizing flows as real space RG process.
  2. RG of NN: a new direction, can we apply RG on the deep NN in the depth direction, so that we could effectively reduce the unnecessary layers of NN to keep the model minimal and avoid overfitting.

* Wavefunction or distribution variational ansatz

  1. RBM: there are various work on RBM as variation ansatz
  2. RNN: definitely RNN is a much interesting option, since it potentially can deal with different size wavefunction in the same model. Specifically, we can try utilize the many to one structure of LSTM for sequence regression: the input is basis (classical configuration) and the output is the amplitude (though may be unnormalized).

### Potentially Useful Resource

* General reference on thoughts and list of work on the bridge between ML and physics

  **Reference:**

  1. List of works and breif intro: [zhihu](https://zhuanlan.zhihu.com/p/25309182)
  2. List of papers: [github](https://physicsml.github.io/pages/papers.html)

* Papers Reading: To do list

  -[x] Why does deep and cheap learningg work so well? [arxiv](https://arxiv.org/pdf/1608.08225.pdf)
  -[ ] On the Equivalence of Restricted Boltzmann Machines and Tensor Network States [arxiv](https://arxiv.org/pdf/1701.04831)
  -[x] Quantum Machine Learning [arxiv](https://arxiv.org/pdf/1611.09347)
  -[x] Neural Network Renormalization Group [arxiv](https://arxiv.org/pdf/1802.02840.pdf)


