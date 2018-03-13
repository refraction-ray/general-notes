# Review on Machine Learning in Physics

*Start writing from Jan 29, 2018*

*Related paper reading tracking*

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


### Neural Network vs. RG

**Reference**: arxiv: 1802.02840

**Keywords**: NeuralRG, bijectors,MERA,MCMC

**Methods**:

* Construct the NN from statistical inference perspective, which is orthodox in modern machine learning interpretations. Also, link partition function in physics with the statistical inference is very typical ML based approach recently.  
* Test on MNIST dataset. Of course, everyone like MNIST, even a dog can come up with a new idea to do something called machine dog learning to classify and generate the digital images. The features of the images are too obvious to make the dataset as a reasonable benchmark. 

**Comments**: 

* This work just coincides my thoughts before, namely, view deep NN as RG flow with the depth of NN as flowing paramter as energy scale. Moreover, we might view the layer of NN as CFT while the bulk of NN as Ads.


## Misc in Reading and Thoughts

### Interesting or Involved Aspects

* The mathematical foundations for the ability of NN to describe anything are from established representability theorems, which guarantee the existence of network approximatesof high-dimensional functions, provided a sufficientlevel of smoothness and regularity is met in the function to be approximated. 

  **Reference**: 

  1. Kolmogorov, A. N. On the representation of continuous functions of several variables by superpositions of continuous functions of a smaller number of variables. Doklady Akademii Nauk SSSR 108, 179–182 (1961).
  2. Hornik, K. Approximation capabilities of multilayer feedforwardnetworks. Neural Networks 4, 251–257 (1991).
  3. Le Roux, N. & Bengio, Y. Representational Power of Restricted Boltzmann Machines and Deep Belief Networks. Neural Computation 20, 1631–1649 (2008).

### Possible Applications and Ideas

* Gereneral route to the interplay between physics and ML
  1. Use ML methods to solve problems in physics
  2. Propose new ML approaches based on existed method in physics
  3. Try to understanding mechanism of ML by physics thoughts
  4. ? Quantum Machine Learning, see [case study](#quantum-machine-learning-review)

* Treat partition function $$Z$$ as a distribution and utilize variational inference tools to approximate it, which is an active field of ML. Some methods in ML might hopefully help:

  **Reference:**

  1. Automatic Differentiation Variational Inference [arixv](https://arxiv.org/pdf/1603.00788.pdf)
  2. Black Box Variational Inference [arxiv](https://arxiv.org/pdf/1401.0118.pdf)

### Potentially Useful Resource

* General reference on thoughts and list of work on the bridge between ML and physics

  **Reference:**

  1. List of works and breif intro: [zhihu](https://zhuanlan.zhihu.com/p/25309182)
  2. List of papers: [github](https://physicsml.github.io/pages/papers.html)

* Papers Reading: To do list

  -[ ] Why does deep and cheap learningg work so well? [arxiv](https://arxiv.org/pdf/1608.08225.pdf)
  -[ ] On the Equivalence of Restricted Boltzmann Machines and Tensor Network States [arxiv](https://arxiv.org/pdf/1701.04831)
  -[x] Quantum Machine Learning [arxiv](https://arxiv.org/pdf/1611.09347)
  -[x] Neural Network Renormalization Group [arxiv](https://arxiv.org/pdf/1802.02840.pdf)


