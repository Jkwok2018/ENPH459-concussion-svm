from io import BufferedIOBase
import numpy as np
from pywt import wavedec, upcoef, Wavelet, dwt, waverec
import statistics

def getZeroCrossingRate(self,arr):
    my_array = np.array(arr)
    return float("{0:.2f}".format((((my_array[:-1] * my_array[1:]) < 0).sum())/len(arr)))
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
        D2 = wrcoef(besaOutput[:,i], 'd',waveletFunction,2)
        D3 = wrcoef(besaOutput[:,i], 'd',waveletFunction,3)
        D4 = wrcoef(besaOutput[:,i], 'd',waveletFunction,4)
        D5 = wrcoef(besaOutput[:,i], 'd',waveletFunction,5)
        D6 = wrcoef(besaOutput[:,i], 'd',waveletFunction,6)
        A6 = wrcoef(besaOutput[:,i], 'a',waveletFunction,6)
     
        waveletDecompFeatures = [statistics.mean(D3), statistics.mean(D4), 
                    statistics.mean(D5), statistics.mean(D6), 
                    statistics.mean(A6), statistics.stdev(D3),
                    statistics.stdev(D4), statistics.stdev(D5), 
                    statistics.stdev(D6), statistics.stdev(A6)]
        

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
        hzcd = dsp.ZeroCrossingDetector
        waveletDecompFeatures += [step(hzcd,D3), step(hzcd,D4), step(hzcd,D5), step(hzcd,D6), step(hzcd,A6)]
    
    waveletDecompFeatures = transpose(waveletDecompFeatures)
    
    
    return waveletDecompFeatures