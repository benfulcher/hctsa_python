import numpy as np
from nitime import timeseries as ts
from functools import partial

# ------------------------------------------------------------------------------
def evaluateAllFunctions(y,functionDic):
    """
    Evalutes all functions specified in a list.
    """
    
    return {label: function(y) for label, function in functionDic.iteritems()}
    
    
    # dic = {}
    # for label in functionDic:
    #     # print label
    #     thisFunc = functionDic[label][0]
    #     dic[label] = thisFunc(*functionDic[label][1])
    #     # dic[label] = functionDic[label]
    #
    # return dic


def convertToFeatureVector(outputDic):
    """
    Converts an output dictionary to two lists: each feature has a name
    and a corresponding output.
    """
    
    nameList = []
    outputList = []
    
    for label in outputDic:
        if type(outputDic[label]) is dict:
            for subLabel in outputDic[label]:
                nameList.append(label+'_'+subLabel)
                outputList.append(outputDic[label][subLabel])
        else:
            nameList.append(label)
            outputList.append(outputDic[label])
            
    return nameList, outputList

def vectorize(time_series):
    
    """
    Converts an input time-series object to a numpy ndarray vector
    
    Arguments
    ---------
    
    time_series: some time-series format
    """
    
    if type(time_series) is ts.TimeSeries:
        return time_series.data
    elif type(time_series) is np.ndarray:
        return time_series
    else:
        raise TypeError('time series must be either an ndarray or nitime TimeSeries object')


def makeRowVector(time_series):
    """
    Ensures a vector is a row, as required by many numpy functions (like diff)
    
    Arguments
    ---------
    
    time_series: a np.ndarray (use vectorize)
    """
    
    if len(time_series.shape) == 1:
        # For some reason if you only pass a single argument it does this shit:
        time_series.shape = (1,len(time_series))
        return time_series
    elif time_series.shape[0] != 1:
        return np.transpose(time_series)
    else:
        return time_series

# ------------------------------------------------------------------------------
# Time-series statistics
# ------------------------------------------------------------------------------
def ST_Length(y):
    """
    Returns the time-series length
    """
    
    # Make the input a row vector of numbers:
    y = makeRowVector(vectorize(y))
    
    # How long is this row vector?:
    return y.shape[1]

def DN_Means(y,meanType='arithmetic'):
    
    # Make the input a row vector of numbers:
    y = makeRowVector(vectorize(y))
    
    # Return mean:
    if meanType in set(['arithmetic','norm']):
        return np.mean(y);

def DN_Spread(y,stdType='std'):
    
    # Make the input a row vector of numbers:
    y = makeRowVector(vectorize(y))
    
    # Return standard deviation of time-series values:
    if stdType == 'std':
        return np.std(y);

def EN_CID(y):
    """
    CID measure from Batista, G. E. A. P. A., Keogh, E. J., Tataw, O. M. & de
    Souza, V. M. A. CID: an efficient complexity-invariant distance for time
    series. Data Min Knowl. Disc. 28, 634-669 (2014).
    
    Arguments
    ---------

    y: a nitime time-series object, or numpy vector

    """

    # Make the input a row vector of numbers:
    y = makeRowVector(vectorize(y))

    # Prepare the output dictionary
    out = {}
    
     # Original definition (in Table 2 of paper cited above)
    out['CE1'] = np.sqrt(np.mean(np.power(np.diff(y),2))); # sum -> mean to deal with non-equal time-series lengths

    # Definition corresponding to the line segment example in Fig. 9 of the paper
    # cited above (using Pythagoras's theorum):
    out['CE2'] = np.mean(np.sqrt(1 + np.power(np.diff(y),2)));

    return out

def SB_MotifTwo(y,binarizeHow='diff'):
    """
    Looks at local motifs in a binary symbolization of the time series, which is performed by a
    given binarization method
    
    Arguments
    ---------

    y: a nitime time-series object, or numpy vector

    """
    
    # Make the input a row vector of numbers:
    y = makeRowVector(vectorize(y))

    # Make binarization on incremental differences:
    if binarizeHow == 'diff':
        yBin = ((np.sign(np.diff(y)))+1.)/2.
    else:
        raise ValueError(binarizeHow)
        
    # Initialize output dictionary
    out = {}
    
    # Where the difference is 0, 1
    r0 = yBin==0
    r1 = yBin==1
    

    out['u'] = np.mean(r1)
    out['d'] = np.mean(r0)
    out['h'] = -(out['u']*np.log2(out['u']) + out['d']*np.log2(out['d']))
    
    return out

def DN_Burstiness(y):
    """
    Returns the 'burstiness' statistic from:
    Goh and Barabasi, 'Burstiness and memory in complex systems' Europhys. Lett.
    % 81, 48002 (2008)
    """
    
    # Ensure the input is a row vector of numbers:
    y = makeRowVector(vectorize(y))
    
    # Compute the burstiness statistic, B:
    B = (np.std(y) - np.mean(y))/(np.std(y) + np.mean(y))
    return B

# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Main body of code
# ------------------------------------------------------------------------------

# Define the time series
timeSeriesData = np.random.randn(1,1000)

# Set functions and parameters:
functionsToCall = {'mean': partial(DN_Means, meanType='arithmetic'),
                    'std': partial(DN_Spread, stdType='std'),
                    'length': ST_Length,
                    'EN_CID': EN_CID,
                    'SB_MotifTwo_diff': partial(SB_MotifTwo,binarizeHow='diff'),
                    'burstiness': DN_Burstiness}

# Evaluate functions:
resultsDic = evaluateAllFunctions(timeSeriesData,functionsToCall)

# Turn this into a list of names and outputs:
functionNames, outputs = convertToFeatureVector(resultsDic)

for i in np.arange(len(functionNames)):
    print functionNames[i],' = ',outputs[i]

