# Machine Learning in a Nutshell

*Start writting from Mar 4, 2018*

*A breif introduction to ML for physicists*

* toc
{:toc}
## Basic Concepts in Math and Machine Learning

### General Settings

Everytime we are faced with a dataset, how can we deal with it. So called data is just a group of number and we can arrange each data as a vector (for each component of the vector, we call it a feature), with or without labels. Labels in general is another vector (usually only one component) associate with data vectors. Now we have two different views on the whole dataset. We can  stack all of the data vectors in rows and make the whole dataset as a matrix. Or we can treat the data vector as a random variable whose distribution is specified by the whole dataset implicitly. Therefore, we can play with dataset in the context of **linear algebra** or **statistical inference**. And we can further gain more insights if we note the intrinsic connections between the two fields in math.

### Linear Algebra

* SVD

  For $$m\times n$$ matrix $$M$$ in field $$K$$, there exists a factorization, called a singular value decomposition of $$M$$, of the form
  $$
  \mathbf {M} =\mathbf {U} {\boldsymbol {\Sigma }}\mathbf {V} ^{\dagger},
  $$
  where $$U,V$$ is dimension $$m\times m, n\times n$$ unitary matrix in field $$K$$. The diagonal term of $$\Sigma$$ are single values. For singular vectors and construction of transformation matrix, see [here](https://en.wikipedia.org/wiki/Singular-value_decomposition#Singular_values,_singular_vectors,_and_their_relation_to_the_SVD).

  Truncated SVD, use only $$r\times r$$ matrix as $$\Sigma$$ to dramatically reduce the freedom representing the large matrix. For an application of truncated SVD in LSI (latent semantic indexing), see [this post](http://www.cnblogs.com/LeftNotEasy/archive/2011/01/19/svd-and-applications.html).



### Statistical Inference

* Covariance

  Covariance is defined for random variable

  $$\operatorname {cov} (X,Y)=\operatorname {E} {{\big [}(X-\operatorname {E} [X])(Y-\operatorname {E} [Y]){\big ]}}.$$

  For a random variable vector, its covariance matrix is defined as 

  $${\displaystyle \Sigma (\mathbf {X} )=\operatorname {cov} (\mathbf {X} ,\mathbf {X} ).}$$

  Suppose X is a random vector and A is a fixed transform vector, then we have:

  $${\displaystyle \Sigma (\mathbf {A} \mathbf {X} )=\operatorname {E} [\mathbf {A} \mathbf {X} \mathbf {X} ^{\mathrm {T} }\mathbf {A} ^{\mathrm {T} }]-\operatorname {E} [\mathbf {A} \mathbf {X} ]\operatorname {E} [\mathbf {X} ^{\mathrm {T} }\mathbf {A} ^{\mathrm {T} }]=\mathbf {A} \Sigma (\mathbf {X} )\mathbf {A} ^{\mathrm {T} }.}$$

* KL divergence (Kullback–Leibler divergence)

  Defined as $${\displaystyle D_{\mathrm {KL} }(P\|Q)=-\sum _{i}P(i)\,\log {\frac {Q(i)}{P(i)}}}$$, where $$P,Q$$ are two random variables and easily to be extended to the continuum version. Note this is not a symmtrical definition! This quantity chracterizes the 'difference' of the two distributions. It is positive definite and be zero iff $$P,Q$$ are the same distributions.

* Information entropy

  For discrete random variable $$X$$, the entropy $$H$$ is defined as $$H(X)=E(-\ln (P(X))=-\sum_{x_i}P(x_i)\ln P(x_i)$$. 

  We can also define conditional entropy as

  $${\displaystyle \mathrm {H} (X|Y)=E_X(H(Y\vert X=x))=\sum_x p(X=x)H(Y|X=x)=-\sum _{i,j}p(x_{i},y_{j})\log {\frac {p(x_{i},y_{j})}{p(y_{j})}}}.$$

  And the joint entropy:

  $$H(X,Y)=-\sum _{{x}}\sum _{{y}}P(x,y)\ln[P(x,y)].$$

  The joint entropy is greater than any individual entropy while no greater than the sum of them.

  Also there is concepts of information gain which measure the difference of information entropy $$IG(T,a)=H(T)-H(T\vert a)$$.

* Mutual information

  $${\displaystyle I(X;Y)=\sum _{y\in Y}\sum _{x\in X}p(x,y)\log {\left({\frac {p(x,y)}{p(x)\,p(y)}}\right)}},$$

  where $$X,Y$$ are two random variables. It is non-negative.

  Relations to conditional entropy:

  $${\displaystyle {\begin{aligned}I(X;Y)&{}\equiv \mathrm {H} (X)-\mathrm {H} (X|Y)\\&{}\equiv \mathrm {H} (Y)-\mathrm {H} (Y|X)\\&{}\equiv \mathrm {H} (X)+\mathrm {H} (Y)-\mathrm {H} (X,Y)\\&{}\equiv \mathrm {H} (X,Y)-\mathrm {H} (X|Y)-\mathrm {H} (Y|X)\end{aligned}}}$$

  Relation to KL divergence:

  $${\displaystyle \begin{aligned}I(X;Y)=&D(p(x,y)\vert\vert p(x)p(y))\\=& E_Y[D(p(x\vert y)\vert\vert p(x)))]\end{aligned}}$$

* Additive smoothing
  In statistics, additive smoothing, also called Laplace smoothing or Lidstone smoothing, is a technique used to smooth estimate probability of categorical data. Given an observation x = (x1, …, xd) N trials, a "smoothed" version of the data gives the probability estimator:
  $${\hat {\theta }}_{i}={\frac {x_{i}+\alpha }{N+\alpha d}}\qquad (i=1,\ldots ,d),$$

  where $$\alpha$$ is a small number called pseudocount. The original version of such formula comes from the [rule of succession](https://en.wikipedia.org/wiki/Rule_of_succession) ($$\alpha=1$$) which is designed to solve the [sunrise problem](https://en.wikipedia.org/wiki/Sunrise_problem). If you are confused with the prior ignorance and noninformative prior distributions, see [this doc](http://www.stats.org.uk/priors/noninformative/Smith.pdf).



###Jargon in ML

* Linear classfier

  Classfier is a blackbox to determine the label for the data.  The blackbox of linear classfier is the innerproduct between data vector and some weight vectors: $$f(\vec{\omega}\cdot \vec{x}+\vec{b})$$. And by adopt the kernel trick, all approaches for representing linear classfier can be converted into nonlinear algorithms (eg. PCA, SVM). It can be understood as the simple form of NN, too. Similarly, we can also define **quadratic classfier**.


* Outliner

  Anomaly. Data that fall far away its group or maybe wrong labeled.


* Batch size

  The number of data vectors for an update on parameters in NN. If batch size is one, we call it online learning. If batch size is more than 1, we call it batch learning.


* Perceptron

  It is for the "node" or "neuron" in NN. And sometimes, it specifically stands for single neuron NN with step function as activation functions which can be served as a binary classfier similar with logistic regression.


* Evaluation method

  * Holdout

    Divided the data into traning set and evaluation set from the beginning

  * Cross Validation

    Devide data into k group. Use k-1 group to fit the model and use the left one to evaluate. Repeat the process k times.

* Bias and variance

  Bias: predict value vs. real value (could be understood as the accuracy for training data).

  Variance: different prediction value in different realizations of model (in some sense the accuracy for validation data).

  The tradeoff: the more complicated the fitting model is, the smaller bias and the larger variance it becomes (over-fitting). See [this](http://scott.fortmann-roe.com/docs/BiasVariance.html) on the comparison and thoughts on these two terms. 

* Ensemble learning

  Definition: Train lots of similar model from sampling of the data, and let them vote for the right predictions. 

  For example, the so-called **bagging** method is to smaple the data with replacement for N times to make N fitting models for voting. Random forest is a typical application of bagging. 

  And if for the prediction, we don't use vote (the mode of predictions), instead train another model (usually logistic regression) with input as the output of the set of models and the output final prediction, such structure of ensemble learning is called **stacking**.

  And if we increase the weight of wrong-labeled data each time we train a model, all the weak model make the model set. This is **boosting**. There are various ways to reassign the weight of data, for example, [gradient boosting](https://en.wikipedia.org/wiki/Gradient_boosting).

* Generative vs. Discriminative model

  Naively, one can agure that generative model are designed for create task, like image generation; while discriminative model is designed for recognition task, like image recongnition. Such understanding is too simple to be correct.

  Suppose the data is $$x$$ and the label is $$y$$. Discriminative model learns $$P(y\vert x)$$ directly to make a predict on the label of newcoming data. Moreover, discriminative model can learn a hard boundary of categories rather than probability. Instead, generative model learns the join distribution $$P(x,y)$$ and use Bayes inference to make prediction (generative model can still work in classfication questions!). What is the advantage for learning the joint distribution? Instead of predicting $$P(y \vert x)$$, you can use it to generate data from $$P(x \vert y)$$. For more comparison on these two things, see [this question](https://stats.stackexchange.com/questions/12421/generative-vs-discriminative) and [this blog post](http://freemind.pluskid.org/machine-learning/discriminative-modeling-vs-generative-modeling/).

## NNN (Non-Neural-Network) approaches

### k-Means clustering

The aim is to partition N data vectors into k-groups. The aim function is the sum of intra group variance:
$$
{\displaystyle {\underset {\mathbf {S} }{\operatorname {arg\,min} }}\sum _{i=1}^{k}\sum _{\mathbf {x} \in S_{i}}\left\|\mathbf {x} -{\boldsymbol {\mu }}_{i}\right\|^{2}={\underset {\mathbf {S} }{\operatorname {arg\,min} }}\sum _{i=1}^{k}|S_{i}|\operatorname {Var} S_{i}}.
$$
This is an NP hard problem. There is an [algorithm](https://en.wikipedia.org/wiki/K-means_clustering#Algorithms) utilizing the iterative process to do the classfication, but no guarantee for optimal solution. The initial center of clusters can be chosed based on so called k-means++ algorithm.

### KNN (K-Nearest Neighbor)

KNN is supervised algorithm. The aim is to give the label of new input data based on original input data. The principle is to find the k neighbors of labeled data, and determine the label of the new one. For comparison between KNN and k-means, see [this post](http://blog.csdn.net/chlele0105/article/details/12997391).

### Mean Shift

Algorithm to locate the most dense part of dataset in the feature space. The basic idea is quite straightforward. Just find some point in feature space and then calcualte the weight center within some sphere centered by the start point. Then, iteratively move to the new center and repeat the calculation. Until we come to some fixed point and this is the most dense part in the space.

### Naive Bayes Classifier

Just a simple application of Bayes inference: $$P(A\vert B)=\frac{P(B\vert A)P(A)}{P(B)}$$. Naive means that each feature B is statistical independet on each other. And AODE classfier is Bayes classifier without the naive indepedent assumptions. The famous formular can also be summarized as

$$posterior =\frac{likelyhood\times prior}{marginal}.$$

In general, we use frequency of data to estimate the prior. And approximate the likelyhood in some funtion form before optimizing it.

### LDA (Linear Discriminant Analysis)

Alias: *Fisher's linear discriminant*. (Especially for two classes problem)

LDA is supervised learning whose aim is to find the optimal linear classifier for labeled high dimension data.  

Suppose we have set of data vector labeled in k classes. Mean value and covariance matrix is $$\mu_i$$ and $$\Sigma_i$$ respectively. Then the variance intra-class are $$\sum_{i=1}^k \vec{\omega}^T \Sigma_i \vec{\omega}$$. The variance inter-classes are $$\vec{\omega}^T \Sigma_b\vec{\omega}$$, where $$\Sigma _{b}={\frac {1}{C}}\sum _{i=1}^{C}(\mu _{i}-\mu )(\mu _{i}-\mu )^{T}$$. We use the ratio between variance inter and intra classes as the effectiveness of the classfication. To maximize this ratio, by utilizing the Larangian factor, we have the conclusion for LDA transform vector $$\omega$$ (the eigenvalue $$\lambda$$ correspoding to the ratio $$S=\frac{\sum_{i=1}^k \vec{\omega}^T \Sigma_i \vec{\omega}}{\vec{\omega}^T \Sigma_b\vec{\omega}}$$):
$$
\lambda (\sum_{i=1}^k\Sigma_i )\omega= \Sigma_b \omega .
$$

It is worth noting that LDA is actually a generatibe model instead of discirminative one. LDA assume the likelihood as Gaussian distribution with different mean vector but the same variance, and we can then max the posterior probability to get the coefficients. Such approach based on Bayes inference can be generalized as Gaussian discriminant analysis. See [this tutorial](https://people.eecs.berkeley.edu/~jrs/189/lec/07.pdf) for details on GDA and Q(uadratic)DA. And if we use GDA framework to calculate the posterior probability, we are about to get the logistic functions. 


### PCA (Principle Components Analysis)

PCA is similar with LDA, but data are without label. Therefore, PCA is unsupervised learning. At the begining, there is only one group of data. The aim of us is to use some project transform vector $$\omega$$ and make the variance largest after the projection. The calculation is similar with LDA case, and the final eigen solution is 
$$
\Sigma(X)\vec{\omega}=\lambda \vec{\omega}.
$$
Again, the larger the $$\lambda$$, the more effective of the dimension reduction axis.

Note that $$X^TX$$ itself can be recognised as proportional to the empirical sample covariance matrix of the dataset **X** (which is the dataset with zero empirical mean each column). The structure of data set matrix X: each of the rows represents a different repetition of the experiment, and each of the columns gives a particular kind of feature.

Or just get the principal values and vectors via SVD. $$\lambda$$ as eigenvalues for covariance matrix of $$X$$ is actually eqivalent to the squre of SVD value for $$X$$. In other words, SVD and PCA are the same thing (see the [explanation](https://math.stackexchange.com/questions/3869/what-is-the-intuitive-relationship-between-svd-and-pca/3871#3871) if you are confused with the equivalence).

### Kernel PCA

Map original data point $$x$$ to $$\phi(x)$$ in higher dimension, we only need to define the kernel function of inner product  $$\phi^T(x)\phi(y)=K(x,y)$$ for further calculation. See the [slides](http://www.cs.haifa.ac.il/~rita/uml_course/lectures/KPCA.pdf) or [blog](https://zhanxw.com/blog/2011/02/kernel-pca-%E5%8E%9F%E7%90%86%E5%92%8C%E6%BC%94%E7%A4%BA/) for details. In this way, by choosing appropriate kernel functions, PCA can handle non-linear knowledge in the data. 

### ICA (Independent Component Analysis)

For comparision between ICA and PCA, see [this answer](https://www.zhihu.com/question/28845451). The idea is somewhat similar to PCA. The difference is the critiria for basis choice: maximize the standard deviation or nonGaussianity.

### NMF (Non-negative Matrix Factorization)

It somewhat is similar to the ideas for truncated SVD but with only two matrix with non-negative elements. The factorization is not unique and there are many [iterative alogrithm](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization#Algorithms) to find such factor matrix.

### CCA (Canonical Correlation Analysis)

The way to find the two linear combination of two random variable vector respectively to maxmize the covariance of the two inner products (the most correlation direction). See [wiki](https://en.wikipedia.org/wiki/Canonical_correlation) for derivation details or [blog](http://www.cnblogs.com/jerrylead/archive/2011/06/20/2085491.html) for example applications and kernel extensions.

### SVM (Support Vector Machine)

Basically, it is to find an optimal hyperplane to seperate data, which is also one type of linear classfier with supervised learning. The aim of the separation is based on the evaluation of distance to the hyperplane from nearest data point, specifically we need to minimize $${\displaystyle \|{\vec {w}}\|} $$ subject to $${\displaystyle y_{i}({\vec {w}}\cdot {\vec {x}}_{i}-b)\geq 1,} $$ for $$ {\displaystyle i=1,\,\ldots ,\,n} $$, where $$y_i$$ is the label $$\pm1$$ of the data $$x_i$$.
The application of SVM require the data can be linearly separated. If not, either kernel trick or soft margin target can be applied.
See integrated [blog](http://blog.csdn.net/v_july_v/article/details/7624837) for more info on SVM (kernel trick and the formula derivation of optimal problem are included). For the difference between LDA and SVM, see [this question](https://stats.stackexchange.com/questions/243932/what-is-the-difference-between-svm-and-lda).

### Decision Trees

Decison trees is just a tree with nodes as the features condition while final leaves as the classfication. It is supervised.  Classification tree predicts discrete classes while regression tree predicts real numbers. CART is the short for classification and regression trees.

To generate such a tree with least nodes and smallest path, usually we generate nodes from top to bottom, and keep some index the extrem value through the constructing of the node. Namely, from the top, we pick the features of the node based on some value evaluation dependent on features. Such value evaluation includes information gain , gini coefficient gain and variance reduction in continuum case (ie. regression trees).  For information gain, I have give the formula before. For gini coefficient, the value is defined as $$gini(X)=1-\sum_{X=x_i}p(x_i)^2$$. For basic introduction on decision trees algorithm, see [here](https://www.ibm.com/developerworks/cn/analytics/library/ba-1507-decisiontree-algorithm/index.html).

To avoid overfitting which is common in decision tree generation, we need some algorithms on pre-pruning and post-pruning.  Besides, early stopping and return posterior probablity instead of categories might also be helful.

### Random Forest

Lots of trees make the forest. To generate each tree, we need data samples from all the data (dataset N and features M). We sample with replacement N times of data and with $$m<<M$$ features for each tree. No pruning process is needed for the forest. Use the mode of all the trees as the final prediction. Typically, the number of trees is of size 100 to 1000. One can use cross validation to find the optimal number of trees. It is interesting that instead of the mode, the variation of predictions amonst trees is also of meaning.

### Regression

* Regression analysis

  The conventional fit scheme for output as continuous variable, eg. linear regression. The loss function is the ordinary least square. 

  $$L_0=\sum_i(\hat{y_i}^2-y_i^2).$$

  To avoid overfitting and to get a more stable solution for fitting paramters $$\theta$$,  we introduce the below methods.

  * Ridge regression

    Add $$L_2$$ regularization term into the loss function. Only responsible for shrinking the value of fitting parameters. 

    $$L_2 = \sum_i \theta_i^2.$$

  * Lasso regression

    Add $$L_1$$ regularization term into the loss function.  Tend to reduce the number of freedom of parameters.

    $$L_1= \sum_i \vert \theta_i \vert.$$

  Of course, we have to tune hyperparamter $$\lambda$$ before regularization term to optimize the trade off between bias and variance.


* Logistic regression

  The fit scheme for output as two seperate values (0,1). So called logistic function: $$\sigma (t)={\frac {e^{t}}{e^{t}+1}}={\frac {1}{1+e^{-t}}}$$. Fit the data with $$\sigma(ax+b)$$. Unlike linear regression, there is no formular for $$a,b$$ dependent on the data provided. We should find the optimal paramters by iterations to optimal the aim which can be viewed as the error or the likelyhood from different aspects. And actually, such regression can be viewed as a two layer NN with sigmoid activation function and one output. 

  Why the function $$\sigma$$ is used to estimated the probability binary distribution? Try locate the boundary of the two classes via 

  $$\ln (\frac{P(true)}{P(false)})=\frac{\sigma(t)}{1-\sigma(t)}=ax+b=0.$$

  We are back to the linear class boundary and this is consitent with LDA derivation.


## NN (neural network) family

### Feed Forward Neural Network

Suppose we need to fit more complicated mapping with lots of parameters than cases in NNN approach, what is the efficient way to do this? This problem can be divided into two parts: structures of paramters to represent some map and efficient way to optimize these paramters. You may be able to name various approaches for how parameters can be organized to represent arbitrary mappings, however the second requirement restricts our imagination to very limit case. The one frequently used in CS field is so called neural networks. (Maybe tensor network in physics field is another possibility.)

### Convolutional Neural Network

### Recurrent Neural Network

### Autoencoder

### Boltzmann Machine

### Adversarial Networks



##Applications in Physics

*Mainly from talks of March Meeting 2018 in LA*

## Main Reference
### Series or books

* Some blog sites in Chinese: [blog](http://www.cnblogs.com/LeftNotEasy/), [blog](http://blog.csdn.net/v_july_v)
* Some blog sites in English: 

* Lei Wang's lecture notes: [link](http://wangleiphy.github.io/lectures/DL.pdf)
* Andrew Moore's slides: [link](https://www.autonlab.org/tutorials)
* Lectures of Universities: [Berkeley](https://people.eecs.berkeley.edu/~jrs/189/), [Stanford](http://cs229.stanford.edu/)
* Other reference series: [gitbook](https://wizardforcel.gitbooks.io/dm-algo-top10)

### Papers or blogs

See correponding references in the main text. 

