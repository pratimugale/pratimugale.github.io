---
layout: post
title:  "An Introduction to the Gradient-Free Classical Optimization Techniques"
date:   2025-01-04 01:55:49 -0500
categories: jekyll update
---

In this post, I will cover going through some optimization techniques, particularly ones used in Quantum Algorithms like the Quantum Approximate Optimization Algorithm. This post will cover the gradient free techniques like Constrained Optimization by Linear Approximation (COBYLA) and Nelder Mead. This post will aim to simplify the explanation by using 2D and 3D plots. When learning quantum optimization algorithms, these classical optimization techniques are often used to optimize the parameters in the parameterized quantum circuits. However, for me as a new learner, it has been challenging to readily find information about how these algorithms work, which this post aims to explain. The paper that has been referenced for majority of the blog is "A view of algorithms for optimization without derivatives" by Powell (2007) [[1](classical-optimizers#ref)], and the ideas and explanations belong to the author - this blogs simplifies, visually demonstrates and provides code for the techniques. 

## What problem are we aiming to solve?  
Given a function that takes "n" parameters, we want to find out the maxima or the minima of this function, and the value of the parameters for which this optima occurs - such a problem is an optimization problem. Such problems are important to solve in various problems, from Machine Learning to Quantum Algorithms. 

To find the optima of such functions, we need some kind of a method or an algorithm to systematically converge to a local or preferably global minima. Broadly, there are two kinds of methods - gradient based (in which the gradient or the derivative/slope needs to be calculated), and gradient-free methods. Gradient based methods such as the gradient descent algorithm is often used in Machine Learning to train a neural network so that the loss function is minimized. While this topic is not the subject of this blog's discussion, I would recommend checking out how gradient descent works - [this](https://uclaacm.github.io/gradient-descent-visualiser/) web page shows great animations of how the gradient descent algorithm works. In a Neural Network, the weights of the network are the parameters that need to be found such that the total error on all the inputs is minimized. In short, the gradient descent algorithm tries to find the direction of steepest descent and move in that direction. But how do we know the direction of steepest descent from a point? We need to find the gradient for each weight for this, which is not very elegant and cost effective. The Back-Propagation Algorithm is used to do this in an efficient manner. Refer to [these](https://www.cs.toronto.edu/~axgao/cs486686_f21/lecture_notes/Lecture_09_on_Neural_Networks_2.pdf) lecture notes by Alice Gao for an excellent explanation of how the gradient based Back-Propagation Algorithm works. 

<br>
<hr>
<h4><b>How do the gradient free algorithms work? </b></h4> 
<hr>

We now know that we are looking to find the minimum of a black box function by finding the optimum parameters. In the case of Quantum Algorithms like QAOA, we want to find 2p parameters $$\gamma$$ = [$$\gamma_1$$, ... $$\gamma_p$$] and $$\beta$$ = [$$\beta_1$$, ... $$\beta_p$$], where p is the number of parameters. 

From my understanding, gradient-based methods require making a lot of calls (executions) on Quantum Hardware [[3](classical-optimizers#ref)]. Frameworks like Qiskit and CUDA-Quantum provide other classical optimizers as well, like NelderMead and COBYLA; and from what I understand, these methods are preferred to minimize the amount of times that quantum circuit needs to be run.

The paper by Powell (2007) [[1](classical-optimizers#ref)] first explains a straightforward simplex method for unconstrained optimization, and then explains the Nelder Mead modification of the Simplex method and finally explains COBYLA. We will follow a similar structure but focus on the visual aspects of what is happening. 

<br>
<hr>
<h4><b>The Original Simplex Method (Spendley, Hext, Himsworth, 1962) [2]</b></h4> 
<hr>

Let us assume the following: 
1. F(x) is the function to be minimized (in this case we consider minimization)
2. $$x \in \mathbb{R}^n$$ i.e. x is a vector of length n, where each element in the vector is a real number. <br>
We can thus also write x = [$$x_1, ... x_n$$], where $$x_1, ... x_n \in \mathbb{R}$$
3. We thus have n variables (parameters). When n variables are present, this method requires taking n+1 points. We can take any n+1 points in the n-dimensional vector space. 
4. Evaluate the function on all n+1 points, and then arrange the values in ascending order. Let $$x_0$$, $$x_1$$, ... $$x_n$$ be the n+1 points now that they are sorted in ascending order of their function values. Thus, F($$x_0) \leq F(x_1) ... \leq F(x_n)$$). These points form a [simplex](https://en.wikipedia.org/wiki/Simplex) after connecting all the points with a line. We currently assume that the volume formed by this simplex is non-zero, meaning that the n vectors of the simplex i.e. $$\overrightarrow{(x_1 - x_0)}, ... \overrightarrow{(x_n - x_0)} $$ are linearly independent.
5. We are now ready to start an iteration of the original simplex method: 
-- Find a new point in the n-dimension space

<span style="margin-left: 60px;">$$
\begin{equation}
    \hat{x} = [\frac{2}{n} (\sum_{i=0}^{n-1} x_i)] - x_n            \label{eq:one}
\end{equation}
$$   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  (1) 
</span>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; What does this equation represent? It first calculates the centroid of the simplex formed by first n points (0 to n-1). Remember that a n-dimensional space can have up to k-simplices, where the k-th simplex has k+1 vertices. So, in the 3-D vector space, a 3-dimensional simplex or 3-simplex is a tetrahedron which has 4 vertices. Since we find the centroid of the first n points in an n-dimensional space, it forms an (n-1)-simplex. The centroid of this simplex is simply $$x_c = \frac{1}{n} (\sum_{i=0}^{n-1} x_i)$$. From (1), we see that $$\hat{x} = 2x_c - x_n$$. Where is this point located? Assume that there goes a line from $$x_n$$ through $$x_c$$ and extends beyond the centroid.
The point $$\hat{x}$$ is located on this line where the distance $$\hat{x} - x_c = x_c - x_n$$. We could say that the point $$\hat{x}$$ is at the opposite side of the centroid of simplex equidistant to $$x_n$$ such that all 3 are on the same line. This also implies that the volume of the n-simplex {$$x_0, x_1, ... x_{n-1}, x_n$$} is the same as the n-simplex {$$x_0, x_1, ... x_{n-1}, \hat{x}$$} 

-- Now that we know where $$\hat{x}$$ lies and what it denotes, the next task is to find the value of the function at this new point. Once found, compare F($$\hat{x}$$) with F($$x_{n-1}$$). 

---- If F($$\hat{x}$$) < F($$x_{n-1}$$), then replace $$x_n$$ with $$\hat{x}$$ as the (n+1)-th point of the original n-simplex. We already know If F($$\hat{x}$$), so sort the points according to the ascending order of their function values again F($$x_0) \leq F(x_1) ... \leq F(x_n)$$), and continue with another iteration, i.e. start with step 5 again. Since the volume of the original simplex did not change, we say that no contraction (of volume) occurred. This change makes sense because the value of F($$x_n$$) is larger than all other points. So if the other side of the simplex has a lower value, then we should move in that direction. But why do we test with the (n-1)-th point and not the n-th point? (F($$\hat{x}$$) < F($$x_{n-1}$$)). This is because of what would happen if $$F(x_{n-1} < F(\hat{x}) < F(x_n)$$. If this occurs, then replacing $$x_n$$ with $$\hat{x}$$ will make no difference in the ordering of the points. And if the order of the points don't change, then the centroid of the 1st n-1 points will also be the same. Thus, in a new iteration, $$\hat{x}$$ of the new iteration would be the $$x_n$$ of the previous iteration. The algorithm will keep on exchanging the last point from $$x_n$$ to $$\hat{x}$$ and will never converge. If we instead ensure that the new $$\hat{x}$$ is also lower than F($$x_{n-1}$$), then we know for sure that there is at least one point at which the function value is higher than $$\hat{x}$$ - making sure that the centroid of the first n-1 points would now be different. It ensures that we explore the space in different areas potentially in those ones that minimize the function.

---- else, i.e. if $$F(\hat{x}) \geq F(x_{n-1}$$), a contraction is performed, meaning a change is performed that reduces the volume of the simplex. It implies that the value of the function at this new point is at least more than the first n-1 points and potentially larger than the largest function value of any point in the simplex. Consider the (n-1)-simplex whose centroid was found, and the line that goes through the centroid to $$x_n$$ on one side and $$\hat{x}$$ on the other side. Thus the simplex is such that on both of its sides, there are points at which the value of the function is high. It might make sense that the lowest value of the function is somewhere closer to the first point of our simplex i.e. $$x_0$$ because this is the lowest value point that we currently know. Of course, it could happen that we get stuck in a local minima by contracting. Thus, to move closer to the lowest value, $$x_0$$ is kept as-is, and the rest of the points $$x_i$$ are replaced by $$\frac{1}{2}(x_0 + x_i) \forall x_i \in [1, 2, ... n]$$ We then calculate the values of the function n more times, arrange the new points again in ascending order and repeat from step 5.  

To visualize this algorithm, I implemented this original simplex method in Python [here](jekyll/update/2025/01/04/classical-optimizers.html)

<video width="540" height="360" controls>
  <source src="/assets/videos/NewScene.mp4" type="video/mp4">
</video>

Conclusion: 
From the above reasoning, what I understand is that the simplex method is a way of traversing n-dimensional space using an n-simplex such that we don't need gradients and we potentially move in good directions as seen above. 

The rest of the blog that explains where this algorithm doesn't work, the Nelder Mead modification of this original Simplex algorithm, and COBYLA is still under construction. I might add the rest of the content in another blog to not make this one too long. Please stay tuned!

<br>
<hr>
<h4 id="ref"><b>References </b></h4> 
<hr>
[1] M. J. D. Powell, “A View of Algorithms for Optimization without Derivatives,” Technical Report DAMTP2007/NA03, Department of Applied Mathematics and Theoretical Physics, University of Cambridge, Cambridge, 2007. 

[2] Spendley, W., Hext, G.R. and Himsworth, F.R. (1962) Sequential Application of Simplex Designs in Optimisation and Evolutionary Operation. Technometrics, 4, 441-461.
http://dx.doi.org/10.1080/00401706.1962.10490033

[3] Pellow-Jarman, A., McFarthing, S., Sinayskiy, I. et al. The effect of classical optimizers and Ansatz depth on QAOA performance in noisy devices. Sci Rep 14, 16011 (2024). https://doi.org/10.1038/s41598-024-66625-6