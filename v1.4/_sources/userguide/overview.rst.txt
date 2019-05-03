Overview
========

The technique of Empirical Orthogonal Function analysis, usually just referred to as EOF analysis, is very common in geophysical sciences.
The general aim of EOF analysis is to reduce the dimensionality of a spatial-temporal data set by transforming it to a new basis in terms of variance.
This transformation turns the input spatial-temporal data set into a set of maps representing patterns of variance, and a time series for each map that determines the contribution of that map to the original data set at each time.
The spatial patterns, which are by definition orthogonal to one-another, are referred to as the EOFs. The EOFs are basis functions in terms of variance. The time series of the EOFs are (in geophysics at least) referred to as the principal components, or PCs, and are the coefficients for each basis function at a given time.
This transformation is advantageous as often a large fraction of the total variance is contained within the first few spatial patterns, meaning the rest can be discarded whilst still maintaining the major variances present in the original data.

Since EOF analysis is really just expressing the original data set in terms of a new basis, the original data set can be completely reconstructed using the EOFs and PCs. However, in practice it is often only a subset of the EOFs that are of interest. Individual EOFs can sometimes have a physical interpretation assigned to them, or sometimes the EOFs and PCs are used to produce a data set that is truncated in terms of variance by reconstructing the original data set using only a limited number of EOFs.


EOF analysis with **eofs**
--------------------------

`eofs` uses solver classes to do EOF analysis. A solver class is a Python object that is initialized with a data set, that provides many methods for returning quantities of interest, such as the EOFs and PCs themselves. The typical usage pattern for `eofs` is:

1. Import the solver class for the type of data being worked with.

2. Create an instance of the solver class using the data to be analysed.
   
   * Other parameters such as weighting may also be specified as initialization time.

3. Call methods of the solver class to retrieve quantities of interest.


All solver possess the same methods, a summary of these methods and what they return is given in the following table:

.. tabularcolumns:: |L|L|

=========================  ====================================================
Method                     Description
=========================  ====================================================
 **pcs**                   Returns the (optionally scaled) principal component
                           time series (PCS).
 **eofs**                  Returns the (optionally scaled) empirical orthogonal
                           functions (EOFs).
 **eofsAsCorrelation**     Returns the EOFs expressed as the correlation
                           between each PCa nd the input data set at each point
                           in space
 **eofsAsCovariance**      Returns the EOFs expressed as the covariance between
                           each PC and the input data set at each point in
                           space
 **eigenvalues**           Returns the eigenvalues (decreasing variances)
                           associated with each EOF mode.
 **varianceFraction**      Returns the fractions of the total variance
                           accounted for by each EOF mode.
 **totalAnomalyVariance**  Returns the total variance in the input data set
                           (sum of the eigenvalues).
 **northTest**             Returns the typical errors associated with each
                           eigenvalue using *North's rule of thumb*.
 **reconstructedField**    Reconstructs the input data set using a given number
                           of EOFs.
 **projectField**          Projects a field onto the EOFs to produce a set of
                           pseudo-PCs.
 **getWeights**            Returns the spatial weights used for the analysis.
=========================  ====================================================

