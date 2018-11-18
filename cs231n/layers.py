from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    N, *D = x.shape
    x_  = np.reshape(x, (N, np.prod(D)))
    out = np.dot(x_, w) + b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    N, *D = x.shape
    dx = np.dot(dout, w.T).reshape((N, *D))
    dw = np.dot(x.reshape(N, -1).T, dout)
    db = np.dot(dout.T, np.ones((N,1)))[:,0]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = x * (x > 0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """

    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    dx = dout * (x >= 0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization arXiv:1502.03167v3  [cs.LG]  2 Mar 2015also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        # 
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        bmean = np.mean(x, axis=0)
        bvar = np.var(x, axis=0)
        x_ = (x - bmean)/np.sqrt(bvar + eps)
        out = gamma * x_ + beta

        running_mean = momentum * running_mean + (1 - momentum) * bmean
        running_var = momentum * running_var + (1 - momentum) * bvar

        cache = (gamma, bmean, bvar, x, x_, N, eps)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        out = (x - running_mean)/np.sqrt(running_var + eps)
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################

    gamma, bmean, bvar, x, x_, N, eps = cache

    dx_ = dout * gamma

    # bvareps = bvar + eps
    # const1 = -0.5 * (bvareps**(-1.5))
    # dbvar =  const1 * np.sum(dx_ * (x - bmean), axis=0)

    # const2 = 1 / np.sqrt(bvareps)
    # dx_const2 = -1 * dx_ * const2
    # dbvar2xbmeanm = (2/N) * dbvar * (x - bmean)
    # dbmean =  -1 * np.sum(dx_const2 + dbvar2xbmeanm, axis=0)
    # dx = dx_const2 + dbvar2xbmeanm + dbmean * (1/N)

    bvareps = bvar + eps
    xbmean = x - bmean
    dx_bvareps = dx_ * (1/np.sqrt(bvareps))

    const1 = 0.5 * (bvareps)**(-1.5) 
    dx_xbmean = dx_ * xbmean
    arg = dx_xbmean * (-1 * const1)
    dbvar = np.sum(arg, axis=0)

    arg = np.sum(-1 * dx_bvareps, axis=0)
    dbmean = arg + (dbvar * np.mean(-2 * (xbmean)))
    
    arg2 = dbvar * (2/N) * xbmean
    arg3 =  dbmean * (1/N)
    dx = dx_bvareps + arg2 + arg3 


    dgamma = np.sum(dout * x_, axis=0)
    dbeta = np.sum(dout, axis=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
     
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    gamma, bmean, bvar, x, x_, N, eps = cache

    dx_ = dout * gamma

    # bvareps = bvar + eps
    # const1 = -0.5 * (bvareps**(-1.5))
    # dbvar =  const1 * np.sum(dx_ * (x - bmean), axis=0)

    # const2 = 1 / np.sqrt(bvareps)
    # dx_const2 = -1 * dx_ * const2
    # dbvar2xbmeanm = (2/N) * dbvar * (x - bmean)
    # dbmean =  -1 * np.sum(dx_const2 + dbvar2xbmeanm, axis=0)
    # dx = dx_const2 + dbvar2xbmeanm + dbmean * (1/N)

    bvareps = bvar + eps
    xbmean = x - bmean
    dx_bvareps = dx_ * (1/np.sqrt(bvareps))

    const1 = 0.5 * (bvareps)**(-1.5) 
    dx_xbmean = dx_ * xbmean
    arg = dx_xbmean * (-1 * const1)
    dbvar = np.sum(arg, axis=0)

    arg = np.sum(-1 * dx_bvareps, axis=0)
    dbmean = arg + (dbvar * np.mean(-2 * (xbmean)))
    
    arg2 = dbvar * (2/N) * xbmean
    arg3 =  dbmean * (1/N)
    dx = dx_bvareps + arg2 + arg3 


    dgamma = np.sum(dout * x_, axis=0)
    dbeta = np.sum(dout, axis=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    
    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get('eps', 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    lmean = np.mean(x, axis=1, keepdims=True)
    lstd = np.sqrt(np.var(x, axis=1, keepdims=True) + eps)
    x_ = (x - lmean)/lstd
    out = gamma * x_ + beta

    cache = (gamma, beta, lmean, lstd, x, x_, eps)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    gamma, beta, lmean, lstd, x, x_, eps = cache
    N, D = dout.shape 

    dx_ = dout * gamma
    lvareps = lstd**2
    xlmean = x - lmean
    dx_lstd = dx_ * (1/lstd)

    const1 = 0.5 * (lvareps)**(-1.5) 
    dx_xlmean = dx_ * xlmean
    arg = dx_xlmean * (-1 * const1)
    dlvar = np.sum(arg, axis=0)

    arg = np.sum(-1 * dx_lstd, axis=0)
    dlmean = arg + (dlvar * np.mean(-2 * (xlmean)))
    
    arg2 = dlvar * (2/D) * xlmean
    arg3 =  dlmean * (1/D)
    dx = dx_lstd + arg2 + arg3

    dgamma = np.sum(dout * x_, axis=0)
    dbeta = np.sum(dout, axis=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        mask = np.random.rand(*x.shape) < p
        out = (x * mask) / p
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = (dout * mask) / dropout_param['p']
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # Get conv params for easier acces
    stride, pad = conv_param['stride'], conv_param['pad']
    # Get the shape of x
    N, C, H, W = x.shape
    # Get shape of w
    F, C, HH, WW = w.shape
    # Compute the shape of output volume
    Ho = 1 + (H + 2 * pad - HH) / stride
    Wo = 1 + (W + 2 * pad - WW) / stride
    try:
        assert (int(Ho) == Ho and int(Wo) == Wo)
    except AssertionError:
        raise ValueError("Invalid arguments for the convolution")
    else:
        Ho, Wo = int(Ho), int(Wo)
    
    # Zero pad
    x_ = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)
    # Convolve
    out = np.zeros((N, F, Ho, Wo))
    w_ = np.reshape(w, (F, -1)).T # This is useful for inner product
    #print(" RESHAPED FILTERS", w_.shape, w_)
    j = 0 # Initialize location in padded volume
    jo = 0 # Initialize location in output volume
    while j + WW <= W + 2 * pad:
        i = 0
        io = 0
        while i + HH <= H + 2 * pad:
            receptive_field = x_[..., i:i + HH, j:j + WW].reshape((N, -1))
            out_ = np.dot(receptive_field, w_) + b
            out[:, :, io, jo] = out_
            
            io += 1 
            i += stride
        jo += 1
        j += stride
     
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # Unpack the cache
    (x, w, b, conv_param) = cache
    # Get conv params for easier acces
    stride, pad = conv_param['stride'], conv_param['pad']
    # Get the shape of x (activations from previous layer)
    N, C, H, W = x.shape
    # Get shape of w (filters of current layer)
    F, C, HH, WW = w.shape
    # Get the shape of dout (output of the current conv at forward time)
    N, F, Ho, Wo = dout.shape
    # Initialize gradients with proper shape
    dx = np.zeros((N, C, H, W))
    dw = np.zeros((F, C, HH, WW))
    db = np.zeros((F, ))
    # Zero pad
    x_ = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)
    dx_ = np.pad(dx, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)
    for i in range(N): # loop over training examples
        
        # Select i training example
        x_i = x_[i]
        dx_i = dx_[i]

        for f in range(F): # loop over the channels of output volume
            for height in range(Ho): # loop over the height of output volulme
                for width in range(Wo): # loop over the width of output volume
                    
                    # Find the corners of the slice
                    vert_start = height * stride
                    vert_end = vert_start + HH
                    horiz_start = width * stride
                    horiz_end = horiz_start + WW

                    # Select the proper region in xi
                    x_i_slice = x_i[:, vert_start: vert_end, horiz_start: horiz_end]

                    # Update gradients for the window and the filter's parameters using the code formulas given above
                    vals = w[f,:,:,:] * dout[i, f, height, width]
                    dx_i[:, vert_start:vert_end, horiz_start:horiz_end] += vals

                    vals = x_i_slice * dout[i, f, height, width]
                    dw[f, :, :, :] += vals

                    db[f] += dout[i, f, height, width]
                    
        dx[i, :, :, :] = dx_i[:, pad:-pad, pad:-pad]
    
    #dw, _ = conv_forward_naive(np.swapaxes(x, 0, 1), np.swapaxes(dout, 0, 1), b, conv_param)
    #dw = np.swapaxes(dw, 0, 1)
    #db = np.sum(dout, axis=(0, 2, 3))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # Get conv params for easier acces
    stride = pool_param['stride']
    pool_height =  pool_param['pool_height']
    pool_width = pool_param['pool_width']
    # Get the shape of x
    N, C, H, W = x.shape
    # Compute the shape of output volume
    Ho = 1 + (H - pool_height) / stride
    Wo = 1 + (W - pool_width) / stride
    try:
        assert (int(Ho) == Ho and int(Wo) == Wo)
    except AssertionError:
        raise ValueError("Invalid arguments for the convolution")
    else:
        Ho, Wo = int(Ho), int(Wo)

    # Pool
    out = np.zeros((N, C, Ho, Wo))
    j = 0 # Initialize location in original volume
    jo = 0 # Initialize location in output volume
    while j + pool_width <= W:
        i = 0
        io = 0
        while i + pool_height <= H:
            receptive_field = x[..., i:i + pool_height, j:j + pool_width]
            out_ = np.max(receptive_field, axis=(-2, -1))
            out[:, :, io, jo] = out_
            
            io += 1 
            i += stride
        jo += 1
        j += stride
     
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # Unpack cache
    (x, pool_param) = cache
    # Get conv params for easier acces
    stride = pool_param['stride']
    pool_height =  pool_param['pool_height']
    pool_width = pool_param['pool_width']
    # Get dimentions of output layer
    N, F, Ho, Wo = dout.shape
    # Get dimentions of input layer
    N, F, H, W = x.shape
    # Initialize derivatives with zeros
    dx = np.zeros_like(x)
    
    for i in range(N): # loop over training examples
    
        # Select i training example
        xi = x[i]
    
        for f in range(F): # loop over the channels of output volume
    
            for height in range(Ho): # loop over the height of output volulme
                for width in range(Wo): # loop over the width of output volume
                    
                    # Find the corners of the slice
                    vert_start = height * stride
                    vert_end = vert_start + pool_height
                    horiz_start = width * stride
                    horiz_end = horiz_start + pool_width
    
                    # Select the proper region in xi
                    xi_slice = xi[f, vert_start: vert_end, horiz_start: horiz_end]
    
                    # Update gradients for the window using the indices of max values
                    mask = xi_slice == np.max(xi_slice)
                
                    vals = mask * dout[i, f, height, width]
                    dx[i, f, vert_start:vert_end, horiz_start:horiz_end] += vals
                    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    N, C, H, W = x.shape
    gamma  = np.repeat(gamma, H * W)
    beta = np.repeat(beta, H * W)
    x_ = np.reshape(x, (N, -1))
    out, cache = batchnorm_forward(x_, gamma, beta, bn_param)
    #gamma, bmean, bvar, x, x_, N, eps = cache

    out = np.reshape(out, (N, C, H, W))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    N, C, H, W = dout.shape
    dout = np.reshape(dout, (N, -1))
    dx, dgamma, dbeta = batchnorm_backward(dout, cache)
    dx = np.reshape(dx, (N, C, H, W))

    dgamma = np.reshape(dgamma, (C, H * W)).sum(axis=1)
    dbeta = np.reshape(dbeta, (C, H * W)).sum(axis=1)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get('eps',1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                # 
    ###########################################################################
    N, C, H, W = x.shape
    assert C % G == 0
    x = np.reshape(x, [N, G, C // G, H, W]) # now x has shape (N, G, C/G, H, W)
    mean = np.mean(x, axis=(2,3,4), keepdims=True)
    var = np.var(x, axis=(2,3,4), keepdims=True)
    out = (x - mean)/np.sqrt(var + eps)
    out = np.reshape(out, (N, C, H, W))
    out = out * gamma + beta
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
