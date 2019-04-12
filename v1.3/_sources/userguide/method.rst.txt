Method of solution
==================

`eofs` uses a technique based on `singular value decomposition (SVD) <http://en.wikipedia.org/wiki/Singular_value_decomposition>`_ to compute the EOF solution [#]_. This avoids having to compute a potentially very large covariance matrix, making `eofs` usable for large data sets.

The input to an EOF analysis is a temporal-spatial field, represented in Python by an array or array-like structure of two or more dimensions. When an `eofs` solver class receives a field as input it is reshaped and stored internally as a two-dimensional array where time is the first dimension and all spatial dimensions are represented by the second dimension. It is a formal requirement of EOF analysis that this array have a time-mean of zero, therefore the `eofs` solver classes will by default subtract the mean along the first dimension.

Any missing values in the input array will be identified and removed [#]_, and the SVD of the array computed. The EOFs are the right singular vectors and the standardized PCs are the left singular vectors, while the singular values are proportional to the variances associated with each EOF mode. The SVD is computed in truncated form, where only singular vectors (EOFs/PCs) that correspond to non-zero singular values are returned. This is done to further reduce the computational cost of the analysis [#]_. Since a singular value is proportional to the variance explained by its associated EOF mode neglecting modes with a singular value of zero maintains a full solution.


Mathematical motivation for the SVD method
------------------------------------------

Consider a data set that consists of observations of a single geophysical variable at multiple positions in space :math:`x_1, x_2, \ldots, x_M` and at multiple times :math:`t_1, t_2, \ldots, t_N,`. These observations are arranged in a matrix :math:`\mathbf{F}` with dimension :math:`N \times M` such that each row of :math:`\mathbf{F}` is a map of observations at all points in space at a particular time, and each column is a time-series of observations at a particular point at all times.
The time-mean is then removed from of the :math:`M` time series to form the anomaly matrix :math:`\mathbf{A}` whose columns have zero-mean:

.. math::

   \mathbf{A} = \begin{pmatrix}
       a_{1,1} & a_{1,2} & \cdots & a_{1,M} \\
       a_{2,1} & a_{2,2} & \cdots & a_{2,M} \\
       \vdots  & \vdots  & \ddots & \vdots \\
       a_{N,1} & a_{N,2} & \cdots & a_{N,M}
   \end{pmatrix}.

Typically one would then compute the covariance matrix :math:`\mathbf{R} = \mathbf{A}^\intercal \mathbf{A}` and solve the eigenvalue problem:

.. math::
   :label: eig

   \mathbf{R C} = \mathbf{C} \Lambda,

where the columns of :math:`\mathbf{C}` are the eigenvectors (EOFs) and the eigenvalues (EOF variances) are on the leading diagonal of :math:`\Lambda`. The PCs :math:`\mathbf{P}` can then be computed from the projection of :math:`\mathbf{A}` onto the EOFs:

.. math::
   
   \mathbf{P} = \mathbf{A C}.

Since computing the covariance matrix can be an expensive operation, a different method is used in `eofs`. Instead of computing the covariance matrix, the SVD of :math:`\mathbf{A}` is computed:

.. math::

   \mathrm{SVD}\left(\mathbf{A}\right) = \mathbf{U} \Gamma \mathbf{V}^\intercal.

The columns of :math:`\mathbf{U}` and :math:`\mathbf{V}` are the singular vectors and the singular values are on the leading diagonal of :math:`\Gamma`. To demonstrate the equivalence of this method and the covariance matrix method we write the covariance matrix in two ways, first a rearranged form of :eq:`eig` [#]_:

.. math::
   :label: R1

   \mathbf{R} = \mathbf{C} \Lambda \mathbf{C}^\intercal,

and second the expression for the covariance matrix :math:`\mathbf{R}` after first taking the SVD of :math:`\mathbf{A}`:

.. math::
   :label: R2

   \mathbf{R} = \mathbf{A}^\intercal \mathbf{A} = \left( \mathbf{U} \Gamma \mathbf{V}^\intercal \right)^\intercal \left( \mathbf{U} \Gamma \mathbf{V}^\intercal \right) = \mathbf{V} \Gamma^\intercal \mathbf{U}^\intercal \mathbf{U} \Gamma \mathbf{V}^\intercal = \mathbf{V} \Gamma^\intercal \Gamma \mathbf{V}^\intercal.

Comparing :eq:`R1` and :eq:`R2` it is clear that :math:`\mathbf{C} = \mathbf{V}` and :math:`\Lambda = \Gamma^\intercal \Gamma`. A bonus of using the SVD method is that the singular vectors in :math:`\mathbf{U}` are the standardized PCs. This can be shown by first forming an expression for :math:`\mathbf{A}` in terms of the EOFs and the PCs:

.. math::

   \mathbf{A} = \mathbf{P} \mathbf{C}^\intercal,

which is the expression for reconstructing a field based on its EOFs and PCs. Defining a normalized PC :math:`\phi_j` as:

.. math::

   \phi_j = \dfrac{\mathbf{p}_j}{\sqrt{\lambda_j}},

where :math:`\mathbf{p}_j` is a column of :math:`\mathbf{P}`, and a diagonal matrix :math:`\mathbf{D}` with :math:`\sqrt{\lambda}_j` on the leading diagonal, and a matrix :math:`\Phi` of the ordered :math:`\phi_j` as columns, we can write a new expression for :math:`\mathbf{A}`:

.. math::

   \mathbf{A} = \Phi \mathbf{D} \mathbf{C}^\intercal.

This expression is exactly equivalent to the SVD of :math:`\mathbf{A}` and therefore the left singular vectors are the PCs scaled to unit variance.


.. rubric:: Footnotes

.. [#] Here SVD is referring to the linear algebra method of singular value decomposition. This is entirely different from the data analysis technique which analyses coupled covariance of two fields, also sometimes referred to as SVD because it makes use of the linear algebra techinque of the same name. The latter data analysis technique is often (and less ambiguously) referred to as maximum covariance analysis (MCA).

.. [#] An often used alternative and equally valid strategy is to set missing values to a constant value. This method yields the same solution since a constant time-series has zero-variance, but increases the computational cost of computing the EOFs as these values are still included in the SVD operation.

.. [#] For an :math:`N` by :math:`M` anomaly matrix the rank of the corresponding covariance matrix can be at most :math:`\mathrm{min} \left( m, n \right)`. The number of zero eigenvalues is therefore at least :math:`\left| m - n \right|`.

.. [#] This rearrangement is possible since the column eigenvectors in :math:`\mathbf{C}` are mutually orthogonal and hence :math:`\mathbf{C} \mathbf{C}^\intercal = \mathbf{I}` where :math:`\mathbf{I}` is the identity matrix.
