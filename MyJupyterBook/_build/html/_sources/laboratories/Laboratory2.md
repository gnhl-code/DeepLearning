## Laboratory Task 2

Genheylou Felisilda - DS4A

---

Instruction: Perform a single forward pass and compute for the error.

$
x = \begin{bmatrix}
1 \\
0 \\
1 \\
\end{bmatrix}
$

$
y = \begin{bmatrix}
1 \\
\end{bmatrix}
$

$
f(z) = max(0, Z_n)
$

$
\text{hidden unit weights} =
\begin{bmatrix}
w_{11} & = & 0.2   & w_{12} & = & -0.3 \\
w_{13} & = & 0.4   & w_{14} & = & 0.1  \\
w_{15} & = & -0.5  & w_{16} & = & 0.2
\end{bmatrix}
$

$
\text{output unit weights} =
\begin{bmatrix}
w_{21} & = & -0.3 \\
w_{22} & = & -0.2
\end{bmatrix}
$

$
\theta =
\begin{bmatrix}
\theta_1 & = & -0.4 \\
\theta_2 & = & 0.2 \\
\theta_3 & = & 0.1
\end{bmatrix}
$


<div class="alert alert-block alert-success" style="font-family: Arial">
<p style="font-family:Times New Roman; text-align:justify; font-size:15px">

### **Solution**

**Hidden Layer**

Compute each hidden unit pre-activation:

$$
z_1 = (1)(0.2) + (0)(0.4) + (1)(-0.5) + (-0.4) = -0.7
$$

$$
z_2 = (1)(-0.3) + (0)(0.1) + (1)(0.2) + (0.2) = 0.1
$$

Apply activation $(a_i = f(z_i))$:

$$
a_1 = f(z_1) = f(-0.7) = 0, \quad
a_2 = f(z_2) = f(0.1) = 0.1
$$

So the hidden activations are:

$$
\mathbf{a} =
\begin{bmatrix}
0 \\
0.1
\end{bmatrix}
$$

---

**Output Layer**

Weights and bias:

$$
\mathbf{w}^{(o)} =
\begin{bmatrix}
-0.3 \\
-0.2
\end{bmatrix},
\quad
\theta^{(o)} = 0.1
$$

Pre-activation:

$$
z_o = (a_1)(-0.3) + (a_2)(-0.2) + \theta^{(o)}
$$

$$
z_o = (0)(-0.3) + (0.1)(-0.2) + 0.1 = 0.08
$$

Activation:

$$
\hat{y} = f(z_o) = f(0.08) = 0.08
$$

---

**Error Calculation**

Target:

$$
y = \begin{bmatrix} 1 \end{bmatrix}
$$

Error:

$$
E = \tfrac{1}{2}(y - \hat{y})^2
$$

$$
E = \tfrac{1}{2}(1 - 0.08)^2 = \tfrac{1}{2}(0.92^2) = 0.4232
$$

---

**Final Results:**

$$
\hat{y} = 0.08, \quad E = 0.4232
$$

</p>
</div>
