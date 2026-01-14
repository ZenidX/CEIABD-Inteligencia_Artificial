import numpy as np
import scipy.io as sio
from numpy.fft import fft
import matplotlib.pyplot as plt

def cutvowel(file_address, start, end):
    Fs, audio = sio.wavfile.read(file_address)
    audiocut = audio[int(float(start)*Fs):int(float(end)*Fs)]
    audiocut = audiocut[:,0]
    return Fs, audiocut
    
    
def wav2vec(cut, Fs): #(file_address, start, end):
    # Fs, audio = sio.wavfile.read(file_address)
    # cut = audio[int(start*Fs):int(end*Fs)]
    # cut = cut[:,0]
    lpf = 15
    zerofilt = 15
    maxfilt = 25
    
    fourierofcut = fft(cut)
    Fsmall = fourierofcut[0:300]
    Fsmall = np.sqrt((np.real(Fsmall) ** 2) + np.imag(Fsmall) ** 2)
    Fsmall[0:30] = 0
    outoffilter = np.zeros(len(Fsmall) - lpf, dtype = np.float64)
    for i in range(len(Fsmall) - lpf):
        for j in range(lpf):
            outoffilter[i] += Fsmall[i+j]

    
    filter2 = maxfilt
    #calculate first max
    Ffon = np.zeros(3, dtype=np.int64)
    Ffon[0] = np.argmax(outoffilter)
    outoffilter[(Ffon[0] - filter2):(Ffon[0] + filter2)] = 0

    #calculate second max
    Ffon[1] = np.argmax(outoffilter)
    outoffilter[(Ffon[1] - filter2):(Ffon[1] + filter2)] = 0

    #calculate third max
    Ffon[2] = np.argmax(outoffilter)
    outoffilter[(Ffon[2] - filter2):(Ffon[2] + filter2)] = 0

    #from fft to Hz freq
    #Ffon.sort()
    Ffon = Ffon * (Fs / len(fourierofcut))

    return Ffon


def distancebv(vect1, vect2):
    return np.sqrt((vect1[0]-vect2[0]) ** 2 + (vect1[1]-vect2[1]) ** 2 + (vect1[2]-vect2[2]) ** 2)

def displaycase(testcase, dictmatrix, testmatrix):
    return 1
    