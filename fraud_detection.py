# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 14:15:15 2020

@author: Menghe Dou
Net ID: md4492
"""

import csv
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import math

from collections import Counter

###
# Problem 1: Read and clean Iranian election data 
###

def extract_election_vote_counts(filename, column_name):
    
    result = []
    # read the csv file by using DictReader
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        # in this way, every row is a key and every is the value                    
        for row in reader:
            for column in column_name:
                # convert each string data to integer
                int_result = int(row[column].replace(',',''))
                result.append(int_result)
    
    return result

###
# Problem 2: Make a histogram 
###        
    
def ones_and_tens_digit_histogram(numbers):

    digit_count = []
    result = []

    for num in numbers:
        # fill the number with two digits if it has a single unit
        num = str(num).zfill(2)
        # save the number in ones digit to list
        digit_count.append(num[len(num)-1])
        # save the number in tens digit to list
        digit_count.append(num[len(num)-2])
        
    # count each number in ones and tens digits   
    frequency = Counter(digit_count)

    for i in range(10):
        # extract frequency values according to keys and 
        # calculate the probability of occurency of each number
        result.append(frequency[str(i)]/sum(frequency.values()))                     
       
    return result        
  
###
# Problem 3: Plot election data 
###

def plot_iranian_least_digits_histogram (histogram):
    
    histogram = extract_election_vote_counts("election-iran-2009.csv", 
                              ["Ahmadinejad", "Rezai", "Karrubi", "Mousavi"])
    # set x-axis values
    x_digit = range(0,10)
    # set y-axis values with the frequency of the ones and tens digit
    y_frequency = ones_and_tens_digit_histogram(histogram)
    
    # create an empty figure
    fig, ax = plt.subplots()
    
    # draw the ideal line - y=0.1 and set it blue
    ax.axhline(y=0.1, xmin=0.0, xmax=9.0, color='blue')

    # draw the Iran digit plot and set it green
    ax.plot(x_digit, y_frequency,'green')
    # add legend of two lines
    ax.legend(["Ideal", "Iran"], loc='upper right')
    
    # set the labels of x and y axis
    plt.xlabel('Digit')  
    plt.ylabel('Frequency')
    
    # set scale of x and y axis
    plt.axis([0, 9, 0.06, 0.16])
    # save the picture with name "iran-digits"
    plt.savefig('iran-digits.png') 
    # show the plot
    plt.show()

    return    
       
###
# Problem 4: Smaller samples have more variation 
###    

def plot_distribution_by_sample_size():

    # set x-axis values
    x_digit = range(0,10)
    
    # create 5 different-size collections of random numbers as y values
    y1 = ones_and_tens_digit_histogram(np.random.randint(0, 99, 10))
    y2 = ones_and_tens_digit_histogram(np.random.randint(0, 99, 50))
    y3 = ones_and_tens_digit_histogram(np.random.randint(0, 99, 100))
    y4 = ones_and_tens_digit_histogram(np.random.randint(0, 99, 1000))
    y5 = ones_and_tens_digit_histogram(np.random.randint(0, 99, 10000)) 

    # create a figure wuth given size
    fig, ax = plt.subplots(figsize=(15,6))
    
    # set scale of x axis
    plt.xlim(0, 9)
    
    # draw the ideal line - y=0.1 and set it blue
    ax.axhline(y=0.1, xmin=0.0, xmax=9.0, color='blue')

    # draw 5 different-size set of random numbers and 
    #compare with the ideal line
    ax.plot(x_digit, y1)
    ax.plot(x_digit, y2)
    ax.plot(x_digit, y3)
    ax.plot(x_digit, y4)
    ax.plot(x_digit, y5)
    
    # add legend of these lines and set location of legend
    ax.legend(["Ideal", "10 random numbers", "50 random numbers", "100 random numbers",
               "1000 random numbers", "10000 random numbers"], loc='upper right')
    
    # set labels of x and y axis
    plt.xlabel('Digit')  
    plt.ylabel('Frequency')
    
    # set title of the graph
    plt.title('Distribution of last two digits')
    
    # save the picture with name "random-digits"
    plt.savefig('random-digits.png') 
    # show the plot
    plt.show()

    return  

###    
# Problem 5: Comparing variation of samples
###

def mean_squared_error(numbers1, numbers2):
    
    # give MSE an initial value
    mse = 0
    # calculate mse baesd on the index of numbers list
    for i in range(len(numbers1)):
        # MSE equation
        mse += (numbers1[i]-numbers2[i])**2   
            
    return mse

###    
# Problem 6: Comparing variation of samples
###  

def calculate_mse_with_uniform(histogram):
    
    # write uniform distribution of y = 0.1
    # represented by ten replicatable numbers - 0.1
    uniform = [0.1] * 10
    
    # calculate the frequency of given dataset
    frequency = ones_and_tens_digit_histogram(histogram)
    
    # get the MSE between given dataset and uniform distribution
    result = mean_squared_error(frequency, uniform)
    
    return result


def compare_iranian_mse_to_samples(number):    
   
    random_group = []
    for i in range(10000):
        random_group.append(np.random.randint(0, 99, 120))
    
    small_list = []
    large_list = []     
    for group in random_group:
        random_mse = calculate_mse_with_uniform(group)

        if random_mse >= number:
            large_list.append(random_mse)
        if random_mse < number:
            small_list.append(random_mse)

    print("Quantity of MSEs larger than or equal to the 2009 Iranian election MSE:", 
          len(large_list))
    print("Quantity of MSEs smaller than the 2009 Iranian election MSE:", 
          len(small_list))
    print("2009 Iranian election null hypothesis rejection level p:", 
          len(small_list)/10000)
    
    return               
        
###
# Problem 8: Other datasets 
###
    
def extract_election_vote_us(filename, column_name):
    
    result = []
    # extract us election data and exclude the comma in it
    # since this dataset has a different compile style, it needs to add 
    # "encoding='latin-1'" to read
    df = pd.read_csv(filename, thousands=r',',usecols = column_name, encoding='latin-1')
    
    # get each data according to the location of row and column
    for row in range(len(df)):
        for column in column_name:
            result.append(df.at[row, column])
            
    # clean the data by removing "nan" value and transform the data to integer
    clean_result = [int(x) for x in result if str(x) != 'nan']  
    
    return clean_result
 

def compare_us_mse_to_samples():
    
    # get the us election data by using former function
    us_data = extract_election_vote_us("election-us-2008.csv", 
                                   ["Obama", "McCain", "Nader", "Barr", "Baldwin", "McKinney"])
    # calculate the mse between us data and uniform distribution
    us_mse = calculate_mse_with_uniform(us_data)
    
    random_group2 = []
    # create 10000 random datasets by using a for loop
    for i in range(10000):
        # create random data by using np.random.randint
        random_group2.append(np.random.randint(0, 99, len(us_data)))
        # with this package, we can get the given-size dataset in a given range
        # since we only need the last two digits in this question, the range 
        # can be set between 0-99
    
    small_list = []
    large_list = []     
    for group in random_group2:
        # in random_group2, each index represents a group of random number
        random_mse = calculate_mse_with_uniform(group)
        # then calculate the MSE of each random dataset
        
        # save the comparison result to their specific lists in order to get the
        # number of answers individually
        if random_mse >= us_mse:
            large_list.append(random_mse)
        if random_mse < us_mse:
            small_list.append(random_mse)
    
    # print the required results
    print("2008 US election MSE:", us_mse)
    print("Quantity of MSEs larger than or equal to the 2008 US election MSE:", 
          len(large_list))
    print("Quantity of MSEs smaller than the 2008 US election MSE:", 
          len(small_list))
    print("2008 United States election null hypothesis rejection level p:", 
          len(small_list)/10000)
    
    return


##################################################################
# Part 2
##################################################################

###
# Problem 9-11
###  

# create histogram of benford distribution
def benford_distribution():
    
    prob_d = []
    # calculate d by using given equation when d belongs [1, 10). 
    for d in range(1,10):
        # use math package to calculate log function
        prob_d.append(math.log10(1+1/d))

    return prob_d


# create sampling datapoints with e^r
def sample_fit_benford_distribution(number):
    
    sample_list = []
    # get a group of random numbers with given size
    for i in range(number):
        # select a random number between 0-30
        r = random.uniform(0,30)
        # calculate each number with equation of e^r
        datapoint = (math.e)**r
        # save the processed ranfom numbers 
        sample_list.append(datapoint)
    
    first_digit = [] 
    # extract the first digit of each number based on the index
    for num in sample_list:
        # save all first digits
        first_digit.append(str(num)[0])
    
    # get the frequency of each first digit in a dictionary
    frequency = Counter(first_digit)

    sample_his = []
    for i in range(1,10):
        # extract frequency values according to keys and 
        # calculate the probability of occurency of each number
        sample_his.append(frequency[str(i)]/sum(frequency.values())) 

    return sample_his


# create sampling datapoints with pi*(e^r) 
def sample_pi_fit_benford_distribution(number):
    
    # get the random numbers in the same way with above
    sample_pi_list = []
    for i in range(number):
        r = random.uniform(0,30)
        # calculate each number with equation of pi*e^r
        datapoint2 = math.pi*((math.e)**r)
        sample_pi_list.append(datapoint2)
    
    # extract the first digit of each number again
    first_digit = []    
    for num in sample_pi_list:
        first_digit.append(str(num)[0])
    
    frequency2 = Counter(first_digit)

    sample_pi_his = []
    for i in range(1,10):
        # extract frequency values according to keys and 
        # calculate the probability of occurency of each number
        sample_pi_his.append(frequency2[str(i)]/sum(frequency2.values())) 

    return sample_pi_his
     
    
# plot above three lines in one figure    
def plot_benford_distribution_law():
           
    # list the number show on the first digit    
    x_digit = range(1,10)
    
    # create an empty figure with given size
    fig, ax = plt.subplots(figsize=(12,8))
    
    # set scales of x and y axis
    plt.axis([1, 9, 0, 0.35])
    
    # set y values with benford distribution and two other random datasets
    y1 = benford_distribution()
    y2 = sample_fit_benford_distribution(1000)
    y3 = sample_pi_fit_benford_distribution(1000)
    
    # plot the sample datapoints compared with Benford distribution
    ax.plot(x_digit, y1, color= 'blue')
    ax.plot(x_digit, y2)
    ax.plot(x_digit, y3)
    
    # add legend and set its location 
    ax.legend(["Benford", "1000 samples", "1000 samples, scaled by $\pi$"], 
              loc='upper right')
    
    # set label of x and y axis
    plt.xlabel('First digit')  
    plt.ylabel('Frequency')
    
    # save the picture with name "scale-invariance"
    plt.savefig('scale-invariance.png') 
    # show the plot
    plt.show()

    return      
  
plot_benford_distribution_law()      


###
# Problem 12-13
###  

# This function is used to calculate US population histogram
def US_population_digit_histogram(filename, column_name):
    
    result = []
    # use the same way to read csv file as the problem 1
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for column in column_name:
                # exclude 'X' and 0 in column
                if row[column] != 'X' and row[column] != 0:
                    # transform string to integer
                    int_result = int(row[column])
                    # save the integer result
                    result.append(int_result)
    
    # extract each first digit of US population data              
    first_digit = []    
    for data in result:
        first_digit.append(str(data)[0])
    
    frequency = Counter(first_digit)

    US_population_his = []
    for i in range(1,10):
        # extract frequency values according to keys and 
        # calculate the probability of occurency of each number
        US_population_his.append(frequency[str(i)]/sum(frequency.values())) 
    
    return US_population_his
     
# This function is used to calculate literature population histogram
def literature_population_histogram(file):
    
    #  split the text data by using the specific delimiter '\t'
    text_column = [x.split('\t')[1] for x in open(file).readlines()]
    # read the second column with index 1 
    
    # clean the text data
    clean_data = []
    for i in text_column:
        # we can remove '\n' by using rstrip
        data = i.rstrip()
        # remove every comma of data
        clean_data.append(int(data.replace(',','')))
        
    # extract each first digit of literature population data    
    first_digit = []    
    for num in text_column:
        first_digit.append(str(num)[0])
    
    frequency = Counter(first_digit)

    text_his = []
    for i in range(1,10):
        # extract frequency values according to keys and 
        # calculate the probability of occurency of each number
        text_his.append(frequency[str(i)]/sum(frequency.values()))
    
    return text_his

# plot the US and literature population histogram compared with benford distribution
def plot_population_data():  
    
    # list the number can show as the first digit    
    x_digit = range(1,10)
    
    # create an empty figure with given size
    fig, ax = plt.subplots(figsize=(12,8))
    
    # set scales of x and y axis
    plt.axis([1, 9, 0, 0.35])
    
    # set y values with benford distribution and two other population datasets
    y1 = benford_distribution()
    y2 = US_population_digit_histogram("SUB-EST2009_ALL.csv", ["POPCENSUS_2000"])
    y3 = literature_population_histogram("literature-population.txt")
    
    # plot the two population datasets compared with Benford distribution
    ax.plot(x_digit, y1)
    ax.plot(x_digit, y2)
    ax.plot(x_digit, y3)
    
    # add legend and set its location 
    ax.legend(["Benford", "US(all)", "Literature Places"], loc='upper right')
    
    # set label of x and y axis
    plt.xlabel('First digit')  
    plt.ylabel('Frequency')
    
    # save the picture with name "population-data"
    plt.savefig('population-data.png') 
    # show the plot
    plt.show()
    
    return

###
# Problem 14: Smaller samples have more variation 
### 

def plot_benford_samples():  
    
    x_digit = range(1,10)
    
    # set y1 value with benford distribution
    y1 = benford_distribution()
    
    # sample with 10 randomly-selected values
    y2 = sample_fit_benford_distribution(10)
        
    # sample with 50 randomly-selected values
    y3 = sample_fit_benford_distribution(50)
        
    # sample with 100 randomly-selected values
    y4 = sample_fit_benford_distribution(100)     
    
    # sample with 1000 randomly-selected values
    y5 = sample_fit_benford_distribution(1000) 
        
    # sample with 10000 randomly-selected values
    y6 = sample_fit_benford_distribution(10000)         
    
    # create an empty figure with given size
    fig, ax = plt.subplots(figsize=(12,8))
    
    # set scales of x and y axis
    plt.axis([1, 9, 0, 0.35])
    
    # plot each sample datasets compared with Benford distribution
    # thicken the lines in order to observe the figure clearly
    ax.plot(x_digit, y1, linewidth=3)
    ax.plot(x_digit, y2, linewidth=3)
    ax.plot(x_digit, y3, linewidth=3)
    ax.plot(x_digit, y4, linewidth=3)
    ax.plot(x_digit, y5, linewidth=3)
    ax.plot(x_digit, y6, linewidth=3, color = 'yellow')    
    
    # add legends and set its location 
    ax.legend(["Benford", "10 samples", "50 samples", "100 samples", 
               "1000 samples", "10000 samples"], loc='upper right')
    
    # set labels of x and y axis
    plt.xlabel('First digit')  
    plt.ylabel('Frequency')
    
    # save the picture with name "benford-samples"
    plt.savefig('benford-samples.png') 
    # show the plot
    plt.show()
    
    return

###
# Problem 15: Comparing variation of samples 
### 

# write a function to extract first digit in convenience of the following question
def first_digit_histogram(numbers):
    
    # extract the first digit in the exactly same way with the former
    first_digit = []    
    for i in numbers:
        first_digit.append(str(i)[0])
            
    frequency = Counter(first_digit)

    random_his = []
    for j in range(1,10):
        random_his.append(frequency[str(j)]/sum(frequency.values()))

    return random_his

def comparing_variation_of_samples():
    
    # get the benford distribution histogram
    benford_his = benford_distribution()
    # get the literature population histogram
    literature_his = literature_population_histogram("literature-population.txt")
    
    # calculate mse between the two histogram
    literature_mse = mean_squared_error(benford_his,literature_his)
    
    # get the literature population data
    literature_data = []
    text_column = [x.split('\t')[1] for x in open("literature-population.txt").readlines()]
    # clean the literature population data
    for i in text_column:
        data = i.rstrip()
        literature_data.append(int(data.replace(',','')))
    
    # get the US Popcensus data
    popcensus_data = []
    with open("SUB-EST2009_ALL.csv") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for column in ["POPCENSUS_2000"]:
                # exclude 'X' and 0 in column
                if row[column] != 'X' and row[column] != 0:
                    # trsform string to integer
                    int_result = int(row[column])
                    # save the integer result
                    popcensus_data.append(int_result)
      
    # get random datasets by using popcensus data  
    # and each datasets size is the number of literature data
    random_group = []
    for i in range(10000):
        # by using random.sample we can get random numbers in given dataset and given size
        random_group.append(random.sample(popcensus_data, k=len(literature_data)))
    
    larger_equal = []
    smaller = [] 
    for group in random_group:
        # get first digit histogram of each random data sample
        us_his = first_digit_histogram(group)
        # calculate mes between each datasets and benford distribution
        us_mse = mean_squared_error(us_his, benford_his)
        if us_mse >= literature_mse:
            larger_equal.append(us_mse)
        if us_mse < literature_mse:
            smaller.append(us_mse)

    # print the required results
    print("Comparison of US MSEs to literature MSE:")
    print("larger/equal:", len(larger_equal))
    print("smaller:", len(smaller))
    
    return 
  
    
# use main function to run the specific functions    
def main(): 
    
    # get Iran election data by using former function
    Iran_data = extract_election_vote_counts("election-iran-2009.csv", 
                              ["Ahmadinejad", "Rezai", "Karrubi", "Mousavi"])
    # calculate Iran MSE
    Iran_mse = calculate_mse_with_uniform(Iran_data)
    # print Iran MSE in required form
    print("2009 Iranian election MSE:", Iran_mse)
    # run the other two function to print all calculation results
    compare_iranian_mse_to_samples(Iran_mse)
    compare_us_mse_to_samples()    

if __name__ == "__main__":     
    main() 

    
    
    
    
    
    

    