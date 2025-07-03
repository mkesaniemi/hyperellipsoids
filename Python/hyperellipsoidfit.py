# Copyright (c) 2025 Martti Kesaniemi

import numpy as np
import warnings
from itertools import combinations
from scipy.linalg import eig

# struct for problem-specific but data-independent variables
from dataclasses import dataclass
@dataclass
class Persistent:
    nDim: np.ndarray 
    crossInds: np.ndarray 
    nCrossTerms: np.ndarray 
    nDSize: np.ndarray
    regMatrix: np.ndarray 
    epsilon: np.ndarray 
    crossTerms: np.ndarray 
    forceOrigin: np.ndarray 
    forceAxial: np.ndarray

########################################################
#HYPERELLIPSOIDFIT Fitting of N-dimensional ellipsoid
#   HYPERELLIPSOIDFIT(D) fits an N-dimensional 
#   hyperellipsoid to collection of N-dimensional data 
#   points given in variable D using a least-squares method 
#   described in 
#       Martti Kesäniemi and Kai Virtanen (2018),
#       "Direct Least Square Fitting of Hyperellipsoids,"
#       IEEE Transactions on Pattern Analysis and Machine Intelligence, 
#       40(1), 63-76. https://doi.org/10.1109/TPAMI.2017.2658574.
#   With input variable D, it is assumed that number of data points > 
#   number of dimensions, and the matrix is transposed accordingly.
#
#   HYPERELLIPSOIDFIT(D, R) uses a sphere-favoring regularization method to
#   increase the possibility to get a solution describing a hyperellipsoid,
#   and to obtain a solution for underdetermined problems.
#   R is the regularization parameter with default value 'eps'. 
#   Also negative values may be used to avoid spherical solutions.
#
#   HYPERELLIPSOIDFIT(D, R, M) uses the method chosen through
#   string M, where valid values for M are
#   'SOD': Default, Sum-Of-Discriminants. Described in 2D by
#       A. Fitzgibbon, M. Pilu, and R.B. Fisher, 
#       "Direct Least Square Fitting of Ellipses," 
#       IEEE Trans. Pattern Analysis and Machine Intelligence, 
#       vol. 21, no. 5, pp. 476-480, May 1999.
#   'HES': Ellipsoid-specific method. Described in 3D by
#       Q. Li and J.G. Griffiths, 
#       "Least Square Ellipsoid Spicific Fitting," 
#       Proc. IEEE Geometric Modeling and Processing, 
#       pp. 335-340, 2004.
#       Parameter eta can be used to loosen the ellipticity constraint.
#       With eta = inf, HES equals to SOD.
#
#   [Me, oe, success, A, regCoeff] = HYPERELLIPSOIDFIT returns the matrix 
#   Me and offset oe that maps points located on the surface of an unit 
#   hypersphere to the surface of the estimated ellipsoid, 
#   y = Me * x + oe;
#   If the solution doesn't decribe an ellipsoid, success is false,
#   otherwise success is true;
#   Parametric form of the quadric surface fitted to the data or 
#   normalized data is returned in vector A; 
#   Regularization parameter used is returned in regParam.
#
#   S = HYPERELLIPSOIDFIT returns all output parameters in struct S.
#
#   Other control parameters:
#   hyperellipsoidfit(D, M, R, ...
#       'eta', etaValue [1, inf], ...
#       'normalize', [true/false], ...
#       'forceOrigin', [true/false], ...
#       'forceAxial', [true/false])
#   eta: controls the constraint assuring hyperellipsoid specificity
#       with HES. eta = 1 assures the ellipsoid-specifity, and eta = inf 
#       equals to SOD. Valid values: eta >= 1. 
#       Default: eta = 1.
#   normalize: if false, input data is not normalized (centered and
#       scaled). 
#       Default: normalize = true.
#   forceAxial: if true, ellipsoid axis are fixed to coordinate axis.
#       Default: forceAxial = false.
#   forceOrigin: if true, ellipsoid center is fixed to origin.
#       Default: forceOrigin = false.
#
#   Copyright 2014-2025 by Martti Kesäniemi
#


########################################################
# Entry point for the ellipsoid fitting routine
def hyperellipsoidfit(data,
    regularization = np.finfo(float).eps, 
    method = "SOD", 
    forceOrigin = False,
    forceAxial = False,
    eta = 1,
    normalize = True):

    if eta < 1.0:
        raise Exception("Eta parameter value has to be >= 1.")
    
    if np.ndim(data) < 2:
        raise Exception('Two-dimensional data required');

    if any(x < 2 for x in np.shape(data)):
        raise Exception('All data dimensions have to be >= 2');
    
    if not hasattr(hyperellipsoidfit, "value"):
        hyperellipsoidfit.p = None  # initialized only once

    # Initialize output
    Me = np.nan; oe = np.nan; dist = np.inf;
    success = False;
    A = np.nan; regParam = regularization;

    # Transpose input matrix if seems appropriate
    dataSize = np.shape(data);
    if dataSize[1] > dataSize[0]:
        data = data.T;

    if normalize:
        if forceOrigin:
            [data, means, scales] = NormalizeData(data, False);
        else:
            [data, means, scales] = NormalizeData(data);
    else:
        means = np.zeros(1,data.shape[1])
        scales = np.ones(1,data.shape[1])

    if hyperellipsoidfit.p == None: 
        hyperellipsoidfit.p = Persistent(
            nDim = np.nan, 
            crossInds = np.nan, 
            nCrossTerms = np.nan, 
            nDSize = np.nan,
            regMatrix = np.nan, 
            epsilon = np.nan, 
            crossTerms = np.nan, 
            forceOrigin = np.nan, 
            forceAxial = np.nan);

    p = hyperellipsoidfit.p;

    if (p.nDim != data.shape[1]
        or p.forceOrigin != forceOrigin 
        or p.forceAxial != forceAxial):
        # Initialize dimension-related parameters
        p.nDim = data.shape[1]
        p.crossInds = np.array(list(combinations(np.arange(0, p.nDim), 2)))
        if forceAxial:
            p.nCrossTerms = 0
        else:
            p.nCrossTerms = p.crossInds.shape[0]    
        p.forceAxial = forceAxial

        if forceOrigin:
            p.nDSize = p.nDim + p.nCrossTerms + 1
        else:
            p.nDSize = p.nDim + p.nCrossTerms + p.nDim + 1    
        p.forceOrigin = forceOrigin
        
        # Create regularization matrix
        tmp = p.nDSize
        p.regMatrix = np.zeros((tmp,tmp))
        p.regMatrix[0:p.nDim, 0:p.nDim] = -2
        p.regMatrix.flat[np.arange(0, p.nDim) * (tmp+1)] = 2*(p.nDim-1)
        p.regMatrix.flat[(p.nDim + np.arange(0, p.nCrossTerms+1)) * (tmp+1)] = p.nDim
        p.epsilon = np.exp(np.log(np.finfo(float).eps)/2)

    # Compute second order cross terms
    p.crossTerms = np.zeros((data.shape[0], p.nCrossTerms));
    
    for ii in range(0, p.nCrossTerms):
        p.crossTerms[:,ii] = (
            data[:,p.crossInds[ii,0]] * data[:,p.crossInds[ii,1]])
    
    A, sucs = QuadraticConstraint(data, p, regParam, method, eta)

    if sucs:
        # Choose sign of solution vector
        if A[1] < 0:
            A = -A;
        if any(GetDiscriminants(A, p) < p.epsilon):
            sucs = false

    if sucs == False:
        print('Failed to find an ellipsoidal solution')
        return Me, oe, success, A, regParam, dist

    # Get distance
    if p.forceOrigin:
        D = np.hstack((data**2, p.crossTerms, -np.ones((data.shape[0],1))))
    else:
        D = np.hstack((data**2, p.crossTerms, data, -np.ones((data.shape[0],1))))
    dist = D @ A
    
    # Solve algebraic form
    Me, oe = GetMappingForm(A, p)
    Me = Me * np.mean(scales)
    oe = oe * np.mean(scales) + means.T
    success = True
    
    return Me, oe, success, A, regParam, dist

########################################################
# Data normalization to improve the numerical stability
def NormalizeData(data, bMeans = True, bScale = True):

    if bMeans:
        means  = data.mean(axis = 0);
        sdata  = data - means;
    else:
        means = np.zeros((1, data.shape[1]))
        sdata = data;
    
    if bScale:
        scale  = (sdata.max() - sdata.min())/2;
        sdata  = sdata / scale;
    else:
        scale = 1;
    
    scales = scale * np.ones((1, data.shape[1]))
    
    return sdata, means, scales

########################################################
# Solves the fitting problem using a quadratic constraint matrix
def QuadraticConstraint(
    ndata, p,
    regularization,
    method, eta):

    success = False
    
    A = np.zeros((p.nDim,1))
    if p.forceOrigin:
        D = np.hstack((ndata**2, p.crossTerms, -np.ones((ndata.shape[0],1))))
    else:
        D = np.hstack((ndata**2, p.crossTerms, ndata, -np.ones((ndata.shape[0],1))))
    
    # Populate constraint matrix according to the method
    constrMatrix = np.zeros((p.nDSize, p.nDSize));
    match method:
        case "HES" | "ES":
            # I: alpha = 2*(4-2*n) + 4*(n-1)*eta
            #          = 8 - 4*n + 4*n*eta - 4*eta
            #          = 4 * ((2-n) + (n-1)*eta)
            # J: beta  = 4-2*n
            # K: gamma = (1-n)*eta
            alpha =     4*((2 - p.nDim)/eta + (p.nDim-1))
            beta  =     (4 - 2*p.nDim) / eta
            gamma =     1 - p.nDim
        case "SOD":
            alpha =     4
            beta  =     0
            gamma =    -1
        case _:
            raise Exception('Unknown method');
    
    constrMatrix[0:p.nDim, 0:p.nDim] = alpha/2
    constrMatrix.flat[np.arange(0,p.nDim)*(p.nDSize+1)] = beta
    constrMatrix.flat[(((p.nDim+np.arange(0,p.nCrossTerms))) *
        (p.nDSize+1))] = gamma

    # Form scatter matrix
    S = D.T @ D
    
    # Solve eigensystem
    nQuadTerms = p.nDim + p.nCrossTerms

    # Break into blocks
    if p.forceOrigin:
        linTerms = 0
    else:
        linTerms = p.nDim

    C1 = constrMatrix[0:nQuadTerms, 0:nQuadTerms]
    # quadratic part of the constraint matrix
    S1 = S[0:nQuadTerms, 0:nQuadTerms]
    # quadratic part of the scatter matrix
    S2 = S[0:nQuadTerms, nQuadTerms+np.arange(0,linTerms+1)]           
    # combined part of the scatter matrix
    S3 = S[nQuadTerms:nQuadTerms+linTerms+1, nQuadTerms:nQuadTerms+linTerms+1]           
    # linear part of the scatter matrix    
    try:
        TS = np.linalg.solve(-S3, S2.T) # for getting a2 from a1
    except np.linalg.LinAlgError as e:
        if 'Singular matrix' in str(e):
            print('Ill-conditioned scatter matrix linear part!');
            return A, success
        else:
            raise
    
    if regularization != 0:
        TT = p.regMatrix[0:nQuadTerms, 0:nQuadTerms]
        M = S1 + S2 @ TS + regularization * TT # reduced scatter matrix
    else:
        M = S1 + S2 @ TS           # reduced scatter matrix
    
    e_val, e_vec = eig(M, C1)  # solve eigensystem
    for i in range(0,e_vec.shape[1]):
        e_vec[:,i] = e_vec[:,i]/np.max(abs(e_vec[:,i]))

    a1, success = ChoosePositiveEigenvalue(e_vec, e_val, p)
    
    if success==False:
        return A, success

    A = np.hstack((a1, TS @ a1))      # ellipse coefficients

    return A, success

########################################################
# Select the eigenvector providing a valid solution
def ChoosePositiveEigenvalue(e_vec, e_val, p):

    success = False 
    a = None;
    e_vec = np.real(e_vec);
    
    # Get index of the positive real finite eigenvalue
    # (which may be slightly negative in case of perfect fit!)
    I = np.where(
        np.isfinite(e_val) &
        (e_val > -p.epsilon) &
        (np.abs(np.imag(e_val)) < p.epsilon)
    )[0]
    if I.size == 0:
        return a, success
    
    # Drop non-ellipsoidal ones
    for ii in range(0,I.size):
        d = GetDiscriminants(e_vec[:,I[ii]], p);
        if any(x < 0 for x in d):
            I[ii] = -1

    I = I[I>=0]
    if I.size == 0:
        II = np.argmax(np.real(e_val))
        a = e_vec[:, II];  # eigenvector corresponding to largest eigenvalue
        return a, success
       
    # If still more than one, choose largest one
    II = np.argmax(np.real(e_val[I]))
    I = I[II]
    
    a = e_vec[:, I]  # eigenvector corresponding to chosen eigenvalue
    
    # Assure it creates an ellipsoid
    if any(GetDiscriminants(e_vec[:,I], p) < 0):
        return a, success
    
    success = True
    
    return a, success


########################################################
# Computed the discriminants corresponding to a solution vector
def GetDiscriminants( v, p ):

    if p.forceAxial: #All cross terms are zero, but discrs must still be > 0
        d = np.zeros((p.crossInds.shape[0], 1))
        for ii in range(0, p.crossInds.shape[0]):
            d[ii] = v[p.crossInds[ii,0]] * v[p.crossInds[ii,1]]
    else:
        d = np.zeros((p.nCrossTerms,1))
        for ii in range(0, p.nCrossTerms):
            d[ii] = (4 * v[p.crossInds[ii,0]] * v[p.crossInds[ii,1]] -
                v[p.nDim+ii]**2)
    
    return d


########################################################
# Transforms parametric representation of the ellipsoid to form
# M*v + o, where
# M is a matrix mapping a origo-centered unit sphere to the ellipsoid, and 
# o is the center point of the ellipsoid
def GetMappingForm( v, p ):

    # Scale v
    v = v/v[-1]
    v[p.nDim:-1] = v[p.nDim:-1]/2
    
    # Form matrix A in [x,1]'*A*[x,1] = 0
    A = np.diag(np.hstack((v[0:p.nDim], -1)))
    for ii in range(0,p.nCrossTerms):
        A[p.crossInds[ii,0], p.crossInds[ii,1]] = v[p.nDim+ii]
        A[p.crossInds[ii,1], p.crossInds[ii,0]] = v[p.nDim+ii]
    
    if p.forceOrigin:
        A[-1, 0:p.nDim] = 0
        A[0:p.nDim, -1] = 0
    else:
        A[-1, 0:p.nDim] = v[(p.nDim+p.nCrossTerms):-1]
        A[0:p.nDim, -1] = v[(p.nDim+p.nCrossTerms):-1]
    
    if p.forceOrigin==False:
        # get offset
        oe = np.linalg.solve(-A[0:p.nDim, 0:p.nDim], 
                             v[-p.nDim-1:-1].T)
        # Remove offset
        T = np.eye( p.nDim+1 )
        T[ p.nDim, 0:p.nDim ] = oe.T
        R = T @ A @ T.T
        R = R[0:p.nDim, 0:p.nDim] / -R[p.nDim, p.nDim]
    else:
        oe = np.zeros((p.nDim,1))
        R = A[0:p.nDim, 0:p.nDim]
    
    # solve Me from Me'*Me = R
    U, S, Vt = np.linalg.svd(R, full_matrices=True)
    V = Vt.T
    Me = np.real(V @ np.diag(1./np.sqrt(S)) @ V.T)
    
    return Me, oe    