from io import BufferedIOBase
import numpy as np
from pywt import wavedec, upcoef, Wavelet, dwt, waverec
import statistics

def getZeroCrossingRate(arr):
    my_array = np.array(arr)
    return ((my_array[:-1] * my_array[1:]) < 0).sum()
def wrcoef(data, coef_type='d', wname='db6', level=9):
    w = Wavelet(wname)
    a = data
    ca = []
    cd = []
    for i in range(level):
        (a, d) = dwt(a, w)
        ca.append(a)
        cd.append(d)
    rec_a = []
    rec_d = []
    for i, coeff in enumerate(ca):
        coeff_list = [coeff, None] + [None] * i
        rec_a.append(waverec(coeff_list, w))
    for i, coeff in enumerate(cd):
        coeff_list = [None, coeff] + [None] * i
        rec_d.append(waverec(coeff_list, w))
    if coef_type == 'd':
        return rec_d
    return rec_a


def waveletDecompExtract(besaOutput = None,waveletFunction = None): 
    #waveletDecompExtract Extraction of Wavelet Decomposition Features
#   Input:time series, besaOutput, and the wavelet function used for
#   wavelet decomposition, waveletFunction
#   Output: n x m matrix consisting of n features and m channels
    
    waveletDecompFeatures_full = []
    for i in range(27):

        # Wavelet Decomposition
        # Calculation The Coefficients Vectors
        cA6, cD6, cD5, cD4, cD3, cD2 , cD1= wavedec(besaOutput[:,i], wavelet = waveletFunction, level = 6)

        # Calculation The Coefficients Vectors
        D1 = wrcoef(besaOutput[:,i], 'd',waveletFunction,1)
        D2 = wrcoef(besaOutput[:,i], 'd',waveletFunction,2)[1]
        D3 = wrcoef(besaOutput[:,i], 'd',waveletFunction,3)[2]
        D4 = wrcoef(besaOutput[:,i], 'd',waveletFunction,4)[3]
        D5 = wrcoef(besaOutput[:,i], 'd',waveletFunction,5)[4]
        D6 = wrcoef(besaOutput[:,i], 'd',waveletFunction,6)[5]
        A6 = wrcoef(besaOutput[:,i], 'a',waveletFunction,6)[-1]
      
        waveletDecompFeatures = [statistics.mean(D3.tolist()), statistics.mean(D4.tolist()), 
                    statistics.mean(D5.tolist()), statistics.mean(D6.tolist()), 
                    statistics.mean(A6.tolist()), statistics.stdev(D3.tolist()),
                    statistics.stdev(D4.tolist()), statistics.stdev(D5.tolist()), 
                    statistics.stdev(D6.tolist()), statistics.stdev(A6.tolist())]

        # waveletDecompFeatures[i].append(statistics.mean(D3))
        # count = count + 1
        # waveletDecompFeatures[i].append(statistics.mean(D4))
        # count = count + 1
        # waveletDecompFeatures[i].append(statistics.mean(D5))
        # count = count + 1
        # waveletDecompFeatures[i].append(statistics.mean(D6))
        # count = count + 1
        # waveletDecompFeatures[i,count] = statistics.mean(A6)
        # count = count + 1
        # # STD
        # waveletDecompFeatures[i].append(statistics.stdev(D3))
        # count = count + 1
        # waveletDecompFeatures[i].append(statistics.stdev(D4))
        # count = count + 1
        # waveletDecompFeatures[i].append(statistics.stdev(D5))
        # count = count + 1
        # waveletDecompFeatures[i].append(statistics.stdev(D6))
        # count = count + 1
        # waveletDecompFeatures[i].append(statistics.stdev(A6))
        # count = count + 1
        # Energy/Power
#             waveletDecompFeatures(i,count)=sum(D3.^2);count=count+1;
#             waveletDecompFeatures(i,count)=sum(D4.^2);count=count+1;
#             waveletDecompFeatures(i,count)=sum(D5.^2);count=count+1;
#             waveletDecompFeatures(i,count)=sum(D6.^2);count=count+1;
#             waveletDecompFeatures(i,count)=sum(A6.^2);count=count+1;
# Normalized Energy/Power
        totalPower = sum(D3 ** 2) + sum(D4 ** 2) + sum(D5 ** 2) + sum(D6 ** 2) + sum(A6 ** 2)
        waveletDecompFeatures += [sum(D3 ** 2) / totalPower, sum(D4 ** 2) / totalPower,
                                sum(D5 ** 2) / totalPower, sum(D6 ** 2) / totalPower,
                                sum(A6 ** 2) / totalPower]
      
        # Normalized Number of zero crossings
        # hzcd = dsp.ZeroCrossingDetector
        waveletDecompFeatures += [getZeroCrossingRate(D3), 
                                    getZeroCrossingRate(D4),
                                    getZeroCrossingRate(D5),
                                    getZeroCrossingRate(D6),
                                    getZeroCrossingRate(A6)]
        waveletDecompFeatures_full.append(waveletDecompFeatures)
    
    waveletDecompFeatures_full = np.transpose(waveletDecompFeatures_full)
    return waveletDecompFeatures_full