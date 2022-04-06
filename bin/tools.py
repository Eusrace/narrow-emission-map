def idx_x2y(x,y):
    '''
    This function output where in x has same value of every y element(x,y can be different size)
    see https://stackoverflow.com/questions/8251541/numpy-for-every-element-in-one-array-find-the-index-in-another-array

    Parameters
    --------------------
    x: 1d np.array
        input array
    
    y: 1d np.array
        value for comparing with x array

    '''
    import numpy as np

    index = np.argsort(x)
    sorted_x = x[index]
    sorted_index = np.searchsorted(sorted_x, y)

    yindex = np.take(index, sorted_index, mode="clip")
    mask = x[yindex] != y

    result = np.ma.array(yindex, mask=mask)
    return result


