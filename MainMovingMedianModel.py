import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import random
from sklearn.linear_model import LinearRegression
import timeit

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
#import seaborn as sns; sns.set_theme(color_codes=True)
import math


# for each object
# Read an object's file
# Seperate Data Pts by Filter
# For each filter:
# init output array (filter_output_array_<color>) for this filter
# Sort ALL data pts by time (least to greatest)
# For each batch of n(19) points
# Pick first n) pts find Time Median.
# Sort n pts by mag of the specific filter
# Record Time and Mag Median into Tuple/Dictionary and append this to filter_output_array_<color>
# Here we will have filter_output_array_red, filter_output_array_infra ...
os.chdir("/Users/ArjunShrivastava/School/SIP/Astronomy Notebooks/Astronomy SIP Projects/AstronomyLab/Moving Median Model /full")
my_files = glob.glob('*[0-10].mjdmag')
#print(my_files[9])
ranger = 200
uOutput = [[] for i in range(ranger)]
gOutput = [[] for i in range(ranger)]
rOutput = [[] for i in range(ranger)]
iOutput = [[] for i in range(ranger)]
i2Output = [[] for i in range(ranger)]
zOutput = [[] for i in range(ranger)]


u_timeMedian = [[] for i in range(ranger)]
g_timeMedian = [[] for i in range(ranger)]
r_timeMedian =[[] for i in range(ranger)]
i_timeMedian = [[] for i in range(ranger)]
i2_timeMedian = [[] for i in range(ranger)]
z_timeMedian = [[] for i in range(ranger)]


timeAxisU = [[] for i in range(ranger)]
magAxisU = [[] for i in range(ranger)]

timeAxisG = [[] for i in range(ranger)]
magAxisG = [[] for i in range(ranger)]

timeAxisR = [[] for i in range(ranger)]
magAxisR = [[] for i in range(ranger)]

timeAxisI = [[] for i in range(ranger)]
magAxisI = [[] for i in range(ranger)]

timeAxisI2 = [[] for i in range(ranger)]
magAxisI2 = [[] for i in range(ranger)]

timeAxisZ = [[] for i in range(ranger)]
magAxisZ = [[] for i in range(ranger)]

U_RMS = []
G_RMS = []
R_RMS = []
I_RMS = []
I2_RMS = []
Z_RMS = []

U_M = []
G_M = []
R_M = []
I_M = []
I2_M = []
Z_M = []

#print(uOutput)
def is_space(e):
    # This function removes the spaces in each line in the objectFile array.
    return e != ''

count = 0
def read_one_object(filepath, index):
    # This function reads data file for a specific object returns the data for the specfied filters.

    #print(filepath)
    objectFile = np.loadtxt(filepath, skiprows=14, delimiter='\n', unpack=1, dtype=str)
    # Here we get all the data from a single file. The delimeter gives the criteria of how the data is seperated in the array.


    # Initialized lists where we put the MDJ and Magnitude.

    for line in objectFile:
        #         print("Line :", line)
        x = line.split(" ")
        #         print(x)
        x = list(filter(is_space, x))
        # print(x[0], x[1], x[8])
        # Using split and filter, we remove all the spaces in a single line so there are no indexs where there are spaces.

        data = (float(x[0]), float(x[1]))
        #print(data)

        if (x[8] == 'U'):
            uOutput[index].append(data)

        elif (x[8] == 'G'):
            gOutput[index].append(data)
        elif (x[8] == 'R'):
            rOutput[index].append(data)

        elif (x[8] == 'I'):
            iOutput[index].append(data)

        elif (x[8] == 'I2'):
            i2Output[index].append(data)
        elif (x[8] == 'Z'):
            zOutput[index].append(data)
            # This for loop appends the data from x (each line) and appends it to the specified filter array

    return uOutput[index], gOutput[index], rOutput[index], iOutput[index], i2Output[index], zOutput[index]
for i in range(ranger):
    r = False
    if(i == 0 and r == True):
        uOutput[i], gOutput[i], rOutput[i], iOutput[i], i2Output[i], zOutput[i] = read_one_object('CFHTLS-VAR-J141937.69+524110.8.mjdmag', i)
    else:
        ind = random.randrange(0, 1000, 1)
        uOutput[i], gOutput[i], rOutput[i], iOutput[i], i2Output[i], zOutput[i] = read_one_object(my_files[ind], i)
#print(uOutput, len(uOutput[0]) + len(gOutput[0]) + len(rOutput[0])+ len(iOutput[0]) + len(i2Output[0]) + len(zOutput[0]))


def key_func(t):
    return t[1]
t = [(0,4),(3,6), (1,3),(-1,4)]
u = sorted(t,key=key_func)


def time_func(filterType):
    return t[0]
#We specify to sort only the MJD's in the tuples.


# Array of time medians and mag medians.
filterID = ""


def sorter(filterType, filterID, index):
    # print("x", filterID)
    # if(filterID == uOutput):
    #  filterID == "uOutput"
    #     if(filterID == gOutput):
    #
    # filterID == "gOutput"
    #     if(filterID == rOutput):
    #          filterID == "rOutput"
    #     if(filterID == iOutput):
    #          filterID == "iOutput"
    #     if(filterID == i2Output):
    #          filterID == "i2Output"
    #     if(filterID == zOutput):
    #          filterID == "zOutput"

    filterSORTED = sorted(filterType, key=time_func)
    # print(filterSORTED)
    return MedMaker(filterSORTED, filterID, index)

# Call the sorting funtion to order the tuples in the list by time.
# print(uOutputSORTED)
# uOutputSORTED contains the list where the tuples are organized by time.
def mag_func(filterType):
    return t[1]


# Sorting function to sort the tuples by magnitude in the U list.


def MedMaker(filterSORTED, filterID, index):
   # print("x", filterID)

    start = 0
    end = 18
    med = 9
   # print(filterID)
    while (end < len(filterSORTED)):

        # Allows us to record the median values.
        # print(med)
        # print(uOutputSORTED[med][0], med)
        timeMed = filterSORTED[med][0]
        # We put the index of the median into the sorted time array to get the median value.

        tempArray = filterSORTED[start:end + 1]
        # Temporarily, we put the the current the mangitudes 19 points of time in this run of the loop into an array.
        # The start and end keep appending to make it a moving median.
        tempArraySORTED = sorted(tempArray, key=mag_func)

        if (filterID == "uOutput"):
            # print(True)
            u_timeMedian[index].append((timeMed, tempArraySORTED[9][1]))
            if (end + 1 == len(filterSORTED)):
                return u_timeMedian

        if (filterID == "gOutput"):
            g_timeMedian[index].append((timeMed, tempArraySORTED[9][1]))
            if (end + 1 == len(filterSORTED)):
                return g_timeMedian

        if (filterID == "rOutput"):
            r_timeMedian[index].append((timeMed, tempArraySORTED[9][1]))
            if (end + 1 == len(filterSORTED)):
                return r_timeMedian

        if (filterID == "iOutput"):
            i_timeMedian[index].append((timeMed, tempArraySORTED[9][1]))
            if (end + 1 == len(filterSORTED)):
                return i_timeMedian

        if (filterID == "i2Output"):
            i2_timeMedian[index].append((timeMed, tempArraySORTED[9][1]))
            if (end + 1 == len(filterSORTED)):
                return i2_timeMedian

        if (filterID == "zOutput"):
            z_timeMedian[index].append((timeMed, tempArraySORTED[9][1]))
            if (end + 1 == len(filterSORTED)):
                return z_timeMedian

        start += 1
        med += 1
        end += 1



def collectMedians():
    for i in range(ranger):
        sorter(uOutput[i], "uOutput", i)
        sorter(gOutput[i], "gOutput", i)
        sorter(rOutput[i], "rOutput", i)
        sorter(iOutput[i], "iOutput", i)
        sorter(i2Output[i], "i2Output", i)
        sorter(zOutput[i], "zOutput", i)

collectMedians()



def InitializePlot():
    plt.clf()
    f = plt.figure(figsize=(20, 20))


    plt.xlabel('Mean of Magnitudes')
    plt.ylabel('RMS-Median')


f = InitializePlot()
def coList(ind):
    for t in u_timeMedian[ind]:
        timeAxisU[ind].append(t[0])
        magAxisU[ind].append(t[1])
    for t in g_timeMedian[ind]:
        timeAxisG[ind].append(t[0])
        magAxisG[ind].append(t[1])

    for t in r_timeMedian[ind]:
        timeAxisR[ind].append(t[0])
        magAxisR[ind].append(t[1])
    for t in i_timeMedian[ind]:
        timeAxisI[ind].append(t[0])
        magAxisI[ind].append(t[1])
    for t in i2_timeMedian[ind]:
        timeAxisI2[ind].append(t[0])
        magAxisI2[ind].append(t[1])
    for t in z_timeMedian[ind]:
        timeAxisZ[ind].append(t[0])
        magAxisZ[ind].append(t[1])

for i in range(ranger):
    coList(i)



origUTime = [[] for i in range(ranger)]
origUMag = [[] for i in range(ranger)]

origGTime = [[] for i in range(ranger)]
origGMag = [[] for i in range(ranger)]

origRTime = [[] for i in range(ranger)]
origRMag = [[] for i in range(ranger)]

origITime = [[] for i in range(ranger)]
origIMag = [[] for i in range(ranger)]

origI2Time = [[] for i in range(ranger)]
origI2Mag= [[] for i in range(ranger)]

origZTime = [[] for i in range(ranger)]
origZMag = [[] for i in range(ranger)]
for tup in uOutput[0]:
    origUTime[0].append(tup[0])
    origUMag[1].append(tup[1])
for tup in gOutput[0]:
    origGTime[0].append(tup[0])
    origGMag[1].append(tup[1])
for tup in rOutput[0]:
    origRTime[0].append(tup[0])
    origRMag[1].append(tup[1])
for tup in iOutput[0]:
    origITime[0].append(tup[0])
    origIMag[1].append(tup[1])
for tup in i2Output[0]:
    origI2Time[0].append(tup[0])
    origI2Mag[1].append(tup[1])
for tup in zOutput[0]:
    origZTime[0].append(tup[0])
    origZMag[1].append(tup[1])


#fig, axis = plt.subplots(2, 2)




"""def RMSCalc(mags, filter):
    magArray = np.array(mags)
    mean_square_mag = np.mean(magArray*magArray)

    return np.sqrt(mean_square_mag)"""


def RMSArray(U, G, R, I, I2, Z):
    for i in U:
        #print(i, "llll")
        #if(len(i) < 1): print("found")
        #print(RMSCalc(i))
        if(i == []):
            #print(True)
            continue
        else:
            #print(i)
            U_M.append(np.median(i))
            #x = RMSCalc(i, "U")
            #U_RMS.append(abs(x**2 - np.mean(i)**2))

            U_RMS.append(np.std(i))
    for i in G:
        #print(i, "llll")
        #if(len(i) < 1): print("found")
        #print(RMSCalc(i))
        if(i == []):
            #print(True)
            continue
        else:
            #y = RMSCalc(i, "G")
            G_M.append(np.median(i))
            G_RMS.append(np.std(i))
    for i in R:
        #print(i, "llll")
        #if(len(i) < 1): print("found")
        #print(RMSCalc(i))
        if(i == []):
            continue
            #print(True)
        else:

            R_M.append(np.median(i))
            R_RMS.append(np.std(i))


RMSArray(magAxisU, magAxisG, magAxisR, magAxisI, magAxisI2, magAxisZ)


poly = PolynomialFeatures(degree=3, include_bias=False)
U_M2 = []

G_M2 = []

R_M2 = []

R_RMS2 = []

G_RMS2 = []

U_RMS2 = []

U_logRMS = np.log10(U_RMS)
R_logRMS = np.log10(R_RMS)
G_logRMS = np.log10(G_RMS)


U_M = np.array(U_M)
U_M2 = poly.fit_transform(U_M.reshape(-1, 1))

G_M = np.array(G_M)
G_M2 = poly.fit_transform(G_M.reshape(-1, 1))

R_M = np.array(R_M)
R_M2 = poly.fit_transform(R_M.reshape(-1, 1))

np.nan_to_num(G_M, neginf=0)
np.nan_to_num(G_RMS, neginf=0)

np.nan_to_num(R_M, neginf=0)
np.nan_to_num(R_RMS, neginf=0)




#print(U_logRMS)

#fig, axis = plt.subplots(3, 1)
"""DataFrameRed = []
for i in range """



#print(R_RMS)
#print(U_M)
plt.ylabel('log RMS')
plt.xlabel('Median RMag')
plt.subplot(3, 1, 1)
#plt.xlim(16, 36)
#plt.ylim(-5, 1)
plt.scatter(R_M, R_logRMS,c='red', edgecolors='none', s=2)

"""poly_reg_model = LinearRegression()
poly_reg_model.fit(R_M2, R_logRMS)
R_RMS2 = poly_reg_model.predict(R_M2)
plt.scatter(R_M, R_RMS2, s = 1)"""
#print(R_RMS2)

plt.subplot(3, 1, 3)
plt.scatter(U_M, U_logRMS,c='blue', edgecolors='none', s=2)
#plt.xlim(16, 36)
#plt.ylim(-5, 1)
plt.ylabel('log RMS')
plt.xlabel('Median UMag')




plt.subplot(3, 1, 2)
plt.scatter(G_M, G_logRMS,c='green', edgecolors='none', s=2)
#plt.xlim(16, 27)
#plt.ylim(-5, 1)
plt.ylabel('log RMS')
plt.xlabel('Median GMag')
"""poly_reg_model = LinearRegression()
poly_reg_model.fit(G_M2, G_logRMS)
G_RMS2 = poly_reg_model.predict(G_M2)
plt.scatter(G_M, G_RMS2, s = 1, c='green')"""


#plt.scatter(G_M, G_RMS, c='green', edgecolors='none', s=9)
#plt.scatter(R_M, R_RMS, c='magenta', edgecolors='none', s=9)
#plt.scatter(I_M, I_RMS, c='gold', edgecolors='none', s=2)
#plt.scatter(I2_M, I2_RMS, c='gold', edgecolors='none', s=2)
#plt.scatter(Z_M, Z_RMS, c='red', edgecolors='none', s=2)

plt.show()



start = timeit.default_timer()

#Your statements here

stop = timeit.default_timer()

print('Time: ', stop - start)

"""
plt.ylim(23, 18)
plt.scatter(origUTime[0], origUMag[1],  c='blue', edgecolors='none', s=2 )
plt.scatter(origGTime[0], origGMag[1],  c='green', edgecolors='none', s=2 )
plt.scatter(origRTime[0], origRMag[1],  c='purple', edgecolors='none', s=2 )
plt.scatter(origITime[0], origIMag[1],  c='gold', edgecolors='none', s=2 )
plt.scatter(origUTime[0], origUMag[1],  c='gold', edgecolors='none', s=2 )
plt.scatter(origZTime[0], origZMag[1],  c='red', edgecolors='none', s=2)
plt.plot(timeAxisU[0], magAxisU[0], c='blue')
plt.plot(timeAxisG[0], magAxisG[0], c='green')
plt.plot(timeAxisR[0], magAxisR[0], c='purple')
plt.plot(timeAxisI[0], magAxisI[0], c='gold')
plt.plot(timeAxisI2[0], magAxisI2[0], c='gold')
plt.plot(timeAxisZ[0], magAxisZ[0], c='red')

"""