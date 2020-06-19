## Differentiable NAS review

*2020 June*

[TOC]

NAS is an active field in AutoML (recent [review](https://arxiv.org/abs/2006.02903) and a nearly exhaustive [list](https://www.automl.org/automl/literature-on-neural-architecture-search/) of papers in this field), it aims at sota neural network architecture search which can hopefully be done in some automatic fashion. The original ideas are around [reinforce learning](https://arxiv.org/abs/1611.01578) where you can make action on the structure layerwise from a discrete set of options by some RNN controllers and [evolution algorithms](https://arxiv.org/pdf/1802.01548.pdf) where a population of network is kept, evaluated, mutated to find the best candidates. Such RL based or evolution algorithm based NAS are rather resource and time consuming since the intrinsic spirit is to search in a discrete domain with expoential large space of choices, other approaches based on [Bayesian optimization](https://arxiv.org/abs/1802.07191) faces the same challenge.

Instead, we focus on one type of NAS denoted as [DARTS](https://arxiv.org/pdf/1806.09055.pdf) (differentable architecture search) and its variants. In such setups, we try to relax the network space into continuous region where differtiation and gradient descent can be utilized, reducing training time from several (thousands of) GPU days to several GPU hours on CIFAR-10. 

### DARTS

In original [darts](https://arxiv.org/pdf/1806.09055.pdf), the search space is limited to micro structure within one cells. Two types of cell are assumed in the network: normal cell vs reduction cell. The NAS is done by determine the structure within the two types of cell, and the whole network can be completed by stacking two types of cells with any depth and input/output size requirements. Within each cell, two input (from the last 2 cells), four intermediate nodes and one output (concate of four intermediate nodes) are presented as DAG. For each edge, one need to search for the optimal connection layers, eg. conv with certain kernel size, or max/average pooling with give window size, zero/identity connections and so on. To make such search differentiable, one assume each edge is the weighted sum of different operations. The weight is the output of softmax of strcuture parameters in this edge. (The idea of supernetwork shares similarity with [one-shot architecture search](http://proceedings.mlr.press/v80/bender18a/bender18a.pdf))

In such setup, we have two sets of parameters: structure weights $$\alpha$$ which determines the network structure by choosing the largest one(or two) and dropping others, and conventional parameters in NN $$\omega$$. The two stage learning is that one firstly alternating (if they get optimized at the same time, the performance of final network structure is reported to be worse by the author) gradient descent for $$\alpha$$ and $$\omega$$ as a bi-optimization problem with validation data and training data. At the second stage, one prune all edges to keep the max weight connection types, and stack these cells as they want, then retraining $$\omega$$ from scratch as traditional NN training setups. As we will see below, the prune process between two stages is not so adibatic process and is the root of problems and drawbacks of darts design. Such NAS cell structure can be transfered to other data sets with possible layout changing in cell stack, and we believe the structure in all is near sota in these new data sets. 

It is worth noting, darts exploit second order Taylor expansion on the optimization of $$\alpha$$. The first order optimization is ok but with a little worse performance of the final network architecture. As we will see later, most of variants of DARTS directly utilize first order optimizations when they imporve the performance from other perspectives (second order optimization cost is larger than first order).

There are various following works focus on the drawback of DARTS. It turns out the original two-stage training procedure is problematic in some scenario and tend to find some worse network structures. The root may be in the large norm of Hessian of validation error in terms of structure parameters which indicates some non-flatten peak in energy landscape. Then after pruning procedure, the structure params move some distance, and the validation error increases drastically. Below we describe some approaches mitigating this together with some more sophisticated variants of darts.

### Some techniques in training

* early stopping in first stage trainning

early stopping when the Hessian of validation error goes up can effectively improve the performance of final structures.

**relevant work**:

[DARTS+](https://arxiv.org/pdf/1909.06035.pdf) (though the argument is not so relevant on why early stopping is good and why original darts may fail - bi optimization is not where the problem is, they notice original darts tend to find more skip connections and take this as early stopping criteria)

[UNDERSTANDING AND ROBUSTIFYING DIFFERENTIABLE ARCHITECTURE SEARCH](https://arxiv.org/pdf/1909.09656.pdf) (early stopping and L2 regularization are introduced, and the reason for the failure for original DARTS are correctely found as high curvature of validation error space together with the prune procedue in DARTS.)

* network parameters prethermalization

Before alternating updates structure and network parameters, one should first optimize the network parameter for tens of epochs as prethermalizations. Such methodology is adopted for almost all DARTS works.

* multiple parallel training procedures

One can train DARTS for different initialized parameters at the same time and found the cell structure that gives the best validation loss. If regularization is included, as suggested by 1909.09656 above, one can run multiple trainnings give different L2 regularization strength and pick the best model.

* training data spliting

Training data should be splited half half as training and validation datas used for updating structure and network parameters respectively. This is support as DARTS theoretical derivations as well as practical experiments for stable trainings.

* learning rates

Since DARTS is rather time consuming and less epoch training is important. So the learning rates should be aggressive enough to approach to optimal quickly. So annealing scheduled learning rates with large initial value or bold driver tricks or other tricks on learning rates adjustment should be tried and utilized, instead of plain Adam with very small initial learning rates (stable but too time consuming). 

**relevant work**

[sharpDARTS](https://arxiv.org/pdf/1903.09900.pdf): Cosine Power Annealing learning rate schedule and other better training/generalization tricks therein (such a max W weighting to suppress the max weighted operation for better minimum).

* regularization term on trainning loss term

L2 regularization is tested on 1909.09656, which gives performace improvements. Schduled Droppath is also considered in this work. 

One can also add L1 regularization terms, but such sparity restriction itself can play roles in pruning the mixed network, see YOSO work in next section for relevant ideas.

* smooth the validation error by near region average

**relevant work**:

[Smooth DARTS](https://arxiv.org/pdf/2002.05283.pdf) (the train loss is evaluated with different structure parameters in a small region, then the loss is averaged or worst case determined, other setups are the same as DARTS. In practice, in each run, we add some noise to structure parameters, one-line changer compared to irginal DARTS)

* operation level dropout and other explicit reset

Since DARTS prefer parameter free connections due to overfitting, one should design some mechanisms to suppress the weight of such operations such as skip connection.

**relevant work**:

[P-DARTS](https://arxiv.org/pdf/1904.12760.pdf): This work utilize dropout on skip connections with decaying rate (avoid fully kill skip connections, which is not good, either.) Also, this work forcely reset skip connection weights to zero for these skip connections beyond threhold of one cell. And then we can repeat this process: training, if number of dominate skip connections is larger than setting threhold, reset extra skip connections with zero weights and redo all this again and again until skip connection numbers is under control in final structure. As reported, such reseting policy must be combined with connection level dropout to give good results.

### Variants of DARTS

In this section, we explain some variants of DARTS with more redesign compared to the above work which only change some training tricks. These works are to solve other drawbacks of darts including: 1) still time consuming as the network with all operations mixed together is large and 2) the two stage setup is somehow artificial and one continuous process is expected.

* stochastic ensemble of networks

Imagine different network structrue has different weight (probability distributions), and we optimize the loss by training the parameterized probability distributions. In such a setup, we evaluate a network with one operations on the edge each time instead of multiple mix operations. Finally, we prune the network as before based on the probability distributions. To make such distribution sample AD aware, one can utilize score function or pathwise/reparameterization approach. The former is general but hampered by high variance. The latter is limited by has low variance. Since the distribution ansatz in DARTS case in category Bernoulli in each edge (factoriable by edge), such sample process has its own reparameterization trick: Gumbel trick with Gumbel distributions (anneling temperature on softmax is also utilized).

In such setups, we evaluate a separate neural network architecture on each run while still maintaining the differentiability of the whole flow.

*The stochastic methodolgy is what I considered after reading DARTS but before knowing all following works. Anyhow, there is always someone exploting the same idea before you*

**relevant work**

The following two works are nearly identical and the setup is presented above.

[SNAS](https://arxiv.org/abs/1812.09926): the focus in on the relation and link between this setup and other NAS methods, resouce contraint as one loss term is also considered

[GDAS](https://arxiv.org/pdf/1910.04465.pdf): note in this work, reduction cell seems to be fixed by hand 

[PARSEC](https://arxiv.org/pdf/1902.05116.pdf): this work utilize score function estimator of MC gradients instead of reparameterization by Gumbel trick in the above two works.

* annealing idea

**relevant work**

[ASAP](https://arxiv.org/pdf/1904.04123.pdf): merge two stage of darts into continuous one by incrementally prune ops. The softmax of weights is decorated with temperature T, by lowering T on the fly with an exponential decay, the prune is done one some weight go below the threhold.

* other continuos parameterization scheme for NAS

**relevant work**

[YOSO/DSO-NAS](https://arxiv.org/pdf/1811.01567.pdf): starting from a full connected DAG is possible, each output/input edge is rescaled by trainable parameters $$\lambda$$ determined by input layer, output layer and operation types of the two layers. (4-tuple). sparsity L1 regularization is added into loss (and thus optimization becomes challenging). APG-NAG optimization method is utilized due to L1 (see paper for details). 

Other memory or time cost are also considered as part of the loss.

* proximal algorithm idea

**relevant work**

[NASP](https://arxiv.org/pdf/1905.13577.pdf): keep structure parameters continuous but add new constainst that restrict them to one-hot. One can optimize such loss+constraint by [PA method](https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf). This approach seems to combine the benefits of annealing idea (one stage instead of two stage, no extra prune is needed) and stochastic settings (evaluate one network structure once and time effcient).

The optimization alg is show below. $$prox_C A$$ is taking the nearest point of A that in set C. C1 is one-hot restriction and C2 is box restriction.

<img width="653" alt="Screen Shot 2020-06-14 at 21 51 08" src="https://user-images.githubusercontent.com/35157286/84595253-594a1000-ae89-11ea-9bac-3b922caeefed.png">

* transfer capability/memory efficiency

In the 1st stage of DARTS, one learns the optimal network structure on given small datasets (usually CIFAR) and small number of cell stackings, while at the 2nd evaluation stage, one may need to evaluate a deeper stacked network with larger dataset (ImageNet for example), then the performance drop for such transfer is a concern. And this can also be viewed as the topic of memory effcient DARTS. Since as long as memory consumption is under control, transfer is not a big issue (one can always learn from the aim context instead of transfering from limited setups: small data set or shallow architecture).

**relevant work**

[P-DARTS](https://arxiv.org/pdf/1904.12760.pdf): this work aims at bridging the depth gap, i.e. 1st stage of DARTS training is carried out with small number of cells (8) and 2nd stage evaluation is carried out with large number of stacking cells (20). To avoid the performance drop with deeper structure, the authors proposed training DARTS in multiple stages. In each stage, the number of stacking cells increases, and to keep the computation effcient, the operations kept on each edge have to decrease respectively. Besides to overcome the biased preference towards param free connection operations in original DARTS, the authors introduced search space regularization, i.e. add dropout on skip connections with gradually decaying rate. Also this work utilize the reseting policy to keep skip-connections less. See training tricks parts for this.

[Proxyless DARTS](https://openreview.net/pdf?id=HylVB3AqYm): this work focus on directly training DARTS on ImageNet to bridge the transfer gap. To fit into GPU memory, the highlight of such work is the desgin of setups, in this persepctive, this work shares the philosophy of SNAS and NASP in the above. Similar to NASP, which introduce discrete restrictions to force the structure parameter to be one-hot, this work introduce binarized path to achieve this which is similar to category sample in SNAS. To make them BP aware, BinaryConnect approach is applied where $$\partial_p$$ is replace as $$\partial_g$$ where g is a one-hot sampled based on p (this also need evaluation of N pathes, so it is not the crux). The crux is, one always firstly choose two pathes in detach fashion (no BP here) and then do the probalistic choice only between the two operations and the training in each step only updates structure parameters for these two operations. *This method seems to be not as good as SNAS or NASP at least from theory perspective*. 

[PC-DARTS](https://arxiv.org/pdf/1907.05737.pdf): only partial of the channels are sent into mixed sum of operations while other channels are direct go through identity. This reduce memory consumptions in GPU to 1/K, when 1/K fraction of channels are chosen (channel mask is different for each output edge). Besides, edge normalization is also introduce for stability of trainings. Namely, extra sets of weights as structure parameters are introduced which are agnostic with operations but only related to edges. The final descision on operations and edges pruning are then determined by the product of two sets of structure weights. Note such edge normalization trick can be also used in other DARTS variants and not necessarily tie to partial channel ideas.