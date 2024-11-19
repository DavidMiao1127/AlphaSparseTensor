# AlphaSparseTensor: Discovering Faster Sparse Matrix Multiplication Algorithms on GPU for LLM inference

As Transformer models continue to grow in size and complexity, numerous high-fidelity pruning methods have been proposed to mitigate the increasing parameter count. However, transforming these theoretical computational savings into practical performance gains faces significant hurdles, including limited GPU support for unstructured sparsity, quality degradation from structured pruning, and mismatch between existing sparse kernels and Large Language Model (LLM) sparsity requirements. 
			
We introduce AlphaSparseTensor, a novel search methodology for efficient algorithms designed for arbitrary sparse matrix multiplication. Inspired by AlphaTensor's approach to matrix multiplication, AlphaSparseTensor reduces computational complexity by minimizing block multiplications and employs an automated workflow to map derived multiplication-addition orders to actual sparse matrix multiplication tasks, achieving performance advantages over existing software stacks on mainstream GPUs. We also introduce an efficient workflow that accommodates various sparsity and size loads, enabling the extraction of zero blocks and reducing block multiplications through dynamic planning. Furthermore, we optimize the GPU implementation of our matrix multiplication paradigm, enhancing performance through out-of-order execution and memory management.

This repository contains all the sparse matrix multiplication algorithms found by AlphaSparseTensor.



## Algorithms

The folder `algorithms` contains all the optimal sparse matrix multiplication strategies between $3\times 3\times 3$ and $5\times 5\times 5$ in both modular arithmetic ($\mathbb{Z}_2$) and standard arithmetic found by AlphaSparseTensor. More specifically, for matrix multiplication $\textbf{A}\textbf{B}=\textbf{C}$, in each matrix multiplication size, we sequentially introduced sparse elements at all positions of matrix A (i.e., setting them to 0, and they are not included in subsequent calculations), once at a time, and provided the corresponding multiplication strategies.

### File Directory

*	`moduler`：sparse matrix multiplication algorithms in moduler arithmetic ($\mathbb{Z}_2$) (i.e., each position in the matrix is either 0 or 1, except for the sparse positions that are determined to be 0.)
  *	`3-3-3`：algorithm for a $3\times 3$ matrix $\textbf{A}$ multiplied by another $3\times 3$ matrix $\textbf{B}$
    *	`333-a11.txt`：algorithm for the situation in which the (1,1) element of matrix $\textbf{A}$ is sure to be 0
    *	……
  *	……
*	`standard`：sparse matrix multiplication algorithms in moduler arithmetic
  *	……

### Interpretation of Algorithms

Take a  $2\times 2\times 2$ sparse matrix multiplication algorithm with the (1,1) element of matrix $\textbf{A}$ being 0 as an example. In our representation method corresponding to low-rank decomposition of three-dimensional spatial tensors, it would be written as:

```
(A22)*(B11+B22)*(C11+C22)
(A21+A22)*(B11)*(C21-C22)
(A22)*(B21-B11)*(C11+C21)
(A12)*(B22)*(-C11+C12)
(A21)*(B11+B12)*(C22)
(A12-A22)*(B21+B22)*(C11)
```

For the part contains A, each row corresponds to an intermediate variable $M_i$ calculated by adding or subtracting elements from A and B:

$$
\begin{array}{l}
M_1 = A_{22}(B_{11} + B_{22}) \\
M_2 = (A_{21} + A_{22})B_{11} \\
M_3 = A_{22}(B_{21} - B_{11}) \\
M_4 = A_{12}B_{22} \\
M_5 = A_{21}(B_{11} + B_{12}) \\
M_6 = (A_{12} - A_{22})(B_{21} + B_{22})
\end{array}
$$

For the part contains C, each row represents the role of the intermediate variable $M_i$ when calculating a specific element of the final matrix C. The coefficient in front of $C_{xy}$ represents the actual coefficient of $M_i$ when computing the element $(x,y)$ of matrix C:

$$
\begin{array}{l}
C_{11}=M_1+M_3-M_4+M_6\\
C_{12}=M_4\\
C_{21}=M_2+M_3\\
C_{22}=M_1-M_2+M_5
\end{array}
$$

The decomposition algorithm above is equivalent to the matrix multiplication calculation below:

$$
\left(
\begin{array}{cc}
 c _ { 11 } & c _ { 12 } \\
 c _ { 21 } & c _ { 22 }
\end{array}
\right) =
\left(
\begin{array}{cc}
0 & a _ { 12 } \\
 a _ { 21 } & a _ { 22 }
\end{array}
\right)
\cdot 
\left(
\begin{array}{cc}
 b _ { 11 } & b _ { 12 } \\
 b _ { 21 } & b _ { 22 }
\end{array}
\right)
$$

