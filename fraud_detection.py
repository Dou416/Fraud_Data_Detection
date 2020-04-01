# -*- coding: utf-8 -*-

import csv
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import math

from collections import Counter

###
# Read and clean Iranian election data 
###

def extract_election_vote_counts(filename, column_name):
    
    result = []
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)                    
        for row in reader:
            for column in column_name:
                int_result = int(row[column].replace(',',''))
                result.append(int_result)
    
    return result

###
# Make a histogram 
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
          
    frequency = Counter(digit_count)

    for i in range(10):
        result.append(frequency[str(i)]/sum(frequency.values()))                     
       
    return result        
  
###
# Plot election data 
###

def plot_iranian_least_digits_histogram (histogram):
    
    histogram = extract_election_vote_counts("election-iran-2009.csv", 
                              ["Ahmadinejad", "Rezai", "Karrubi", "Mousavi"])
    x_digit = range(0,10)

    y_frequency = ones_and_tens_digit_histogram(histogram)
    
    fig, ax = plt.subplots()

    ax.axhline(y=0.1, xmin=0.0, xmax=9.0, color='blue')

    ax.plot(x_digit, y_frequency,'green')
    ax.legend(["Ideal", "Iran"], loc='upper right')
    
    plt.xlabel('Digit')  
    plt.ylabel('Frequency')

    plt.axis([0, 9, 0.06, 0.16])
    # save the picture 
    plt.savefig('iran-digits.png') 
    # show the plot
    plt.show()

    return    
       
###
# Smaller samples have more variation 
###    

def plot_distribution_by_sample_size():

    x_digit = range(0,10)
    
    y1 = ones_and_tens_digit_histogram(np.random.randint(0, 99, 10))
    y2 = ones_and_tens_digit_histogram(np.random.randint(0, 99, 50))
    y3 = ones_and_tens_digit_histogram(np.random.randint(0, 99, 100))
    y4 = ones_and_tens_digit_histogram(np.random.randint(0, 99, 1000))
    y5 = ones_and_tens_digit_histogram(np.random.randint(0, 99, 10000)) 

    fig, ax = plt.subplots(figsize=(15,6))
    
    plt.xlim(0, 9)

    ax.axhline(y=0.1, xmin=0.0, xmax=9.0, color='blue')

    ax.plot(x_digit, y1)
    ax.plot(x_digit, y2)
    ax.plot(x_digit, y3)
    ax.plot(x_digit, y4)
    ax.plot(x_digit, y5)

    ax.legend(["Ideal", "10 random numbers", "50 random numbers", "100 random numbers",
               "1000 random numbers", "10000 random numbers"], loc='upper right')
    
    plt.xlabel('Digit')  
    plt.ylabel('Frequency')

    plt.title('Distribution of last two digits')

    plt.savefig('random-digits.png') 

    plt.show()

    return  

###    
# Comparing variation of samples
###

def mean_squared_error(numbers1, numbers2):

    mse = 0
    for i in range(len(numbers1)):
        # MSE equation
        mse += (numbers1[i]-numbers2[i])**2   
            
    return mse

###    
# Comparing variation of samples
###  

def calculate_mse_with_uniform(histogram):

    uniform = [0.1] * 10

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
# Other datasets 
###
    
def extract_election_vote_us(filename, column_name):
    
    result = []
    df = pd.read_csv(filename, thousands=r',',usecols = column_name, encoding='latin-1')
    
    for row in range(len(df)):
        for column in column_name:
            result.append(df.at[row, column])

    clean_result = [int(x) for x in result if str(x) != 'nan']  
    
    return clean_result
 

def compare_us_mse_to_samples():

    us_data = extract_election_vote_us("election-us-2008.csv", 
                                   ["Obama", "McCain", "Nader", "Barr", "Baldwin", "McKinney"])
    us_mse = calculate_mse_with_uniform(us_data)
    
    random_group2 = []
    for i in range(10000):
        random_group2.append(np.random.randint(0, 99, len(us_data)))
    
    small_list = []
    large_list = []     
    for group in random_group2:
        random_mse = calculate_mse_with_uniform(group)

        if random_mse >= us_mse:
            large_list.append(random_mse)
        if random_mse < us_mse:
            small_list.append(random_mse)
    
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

# create histogram of benford distribution
def benford_distribution():
    
    prob_d = []
    for d in range(1,10):
        prob_d.append(math.log10(1+1/d))

    return prob_d


# create sampling datapoints with e^r
def sample_fit_benford_distribution(number):
    
    sample_list = []
    for i in range(number):
        r = random.uniform(0,30)
        datapoint = (math.e)**r
        sample_list.append(datapoint)
    
    first_digit = [] 
    # extract the first digit of each number based on the index
    for num in sample_list:
        # save all first digits
        first_digit.append(str(num)[0])
    
    frequency = Counter(first_digit)

    sample_his = []
    for i in range(1,10):
        # extract frequency values according to keys and 
        # calculate the probability of occurency of each number
        sample_his.append(frequency[str(i)]/sum(frequency.values())) 

    return sample_his


# create sampling datapoints with pi*(e^r) 
def sample_pi_fit_benford_distribution(number):
    
    sample_pi_list = []
    for i in range(number):
        r = random.uniform(0,30)
        datapoint2 = math.pi*((math.e)**r)
        sample_pi_list.append(datapoint2)
    
    # extract the first digit of each number again
    first_digit = []    
    for num in sample_pi_list:
        first_digit.append(str(num)[0])
    
    frequency2 = Counter(first_digit)

    sample_pi_his = []
    for i in range(1,10):
        sample_pi_his.append(frequency2[str(i)]/sum(frequency2.values())) 

    return sample_pi_his
     
    
# plot above three lines in one figure    
def plot_benford_distribution_law():
             
    x_digit = range(1,10)
    
    fig, ax = plt.subplots(figsize=(12,8))
    
    plt.axis([1, 9, 0, 0.35])
    
    y1 = benford_distribution()
    y2 = sample_fit_benford_distribution(1000)
    y3 = sample_pi_fit_benford_distribution(1000)

    ax.plot(x_digit, y1, color= 'blue')
    ax.plot(x_digit, y2)
    ax.plot(x_digit, y3)
    
    ax.legend(["Benford", "1000 samples", "1000 samples, scaled by $\pi$"], 
              loc='upper right')
    
    plt.xlabel('First digit')  
    plt.ylabel('Frequency')

    plt.savefig('scale-invariance.png') 
    plt.show()

    return      
  
plot_benford_distribution_law()      


# Calculate US population histogram
def US_population_digit_histogram(filename, column_name):
    
    result = []
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for column in column_name:
                # exclude 'X' and 0 in column
                if row[column] != 'X' and row[column] != 0:
                    int_result = int(row[column])
                    result.append(int_result)
               
    first_digit = []    
    for data in result:
        first_digit.append(str(data)[0])
    
    frequency = Counter(first_digit)

    US_population_his = []
    for i in range(1,10):
        US_population_his.append(frequency[str(i)]/sum(frequency.values())) 
    
    return US_population_his
     
# Calculate literature population histogram
def literature_population_histogram(file):
    
    #  split the text data by using the specific delimiter '\t'
    text_column = [x.split('\t')[1] for x in open(file).readlines()]
    
    # clean the text data
    clean_data = []
    for i in text_column:
        data = i.rstrip()
        # remove every comma of data
        clean_data.append(int(data.replace(',','')))
        
    first_digit = []    
    for num in text_column:
        first_digit.append(str(num)[0])
    
    frequency = Counter(first_digit)

    text_his = []
    for i in range(1,10):
        text_his.append(frequency[str(i)]/sum(frequency.values()))
    
    return text_his

# plot the US and literature population histogram compared with benford distribution
def plot_population_data():  
     
    x_digit = range(1,10)

    fig, ax = plt.subplots(figsize=(12,8))

    plt.axis([1, 9, 0, 0.35])

    y1 = benford_distribution()
    y2 = US_population_digit_histogram("SUB-EST2009_ALL.csv", ["POPCENSUS_2000"])
    y3 = literature_population_histogram("literature-population.txt")

    ax.plot(x_digit, y1)
    ax.plot(x_digit, y2)
    ax.plot(x_digit, y3)

    ax.legend(["Benford", "US(all)", "Literature Places"], loc='upper right')

    plt.xlabel('First digit')  
    plt.ylabel('Frequency')

    plt.savefig('population-data.png') 
    plt.show()
    
    return

###
# Smaller samples have more variation 
### 

def plot_benford_samples():  
    
    x_digit = range(1,10)
    
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

    fig, ax = plt.subplots(figsize=(12,8))

    plt.axis([1, 9, 0, 0.35])
    
    ax.plot(x_digit, y1, linewidth=3)
    ax.plot(x_digit, y2, linewidth=3)
    ax.plot(x_digit, y3, linewidth=3)
    ax.plot(x_digit, y4, linewidth=3)
    ax.plot(x_digit, y5, linewidth=3)
    ax.plot(x_digit, y6, linewidth=3, color = 'yellow')    

    ax.legend(["Benford", "10 samples", "50 samples", "100 samples", 
               "1000 samples", "10000 samples"], loc='upper right')

    plt.xlabel('First digit')  
    plt.ylabel('Frequency')
    plt.savefig('benford-samples.png') 
    plt.show()
    
    return

###
# Comparing variation of samples 
### 

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

    benford_his = benford_distribution()
    literature_his = literature_population_histogram("literature-population.txt")

    literature_mse = mean_squared_error(benford_his,literature_his)

    literature_data = []
    text_column = [x.split('\t')[1] for x in open("literature-population.txt").readlines()]

    for i in text_column:
        data = i.rstrip()
        literature_data.append(int(data.replace(',','')))
    
    # get the US Popcensus data
    popcensus_data = []
    with open("SUB-EST2009_ALL.csv") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for column in ["POPCENSUS_2000"]:
                if row[column] != 'X' and row[column] != 0:
                    int_result = int(row[column])
                    popcensus_data.append(int_result)
      
    random_group = []
    for i in range(10000):
        random_group.append(random.sample(popcensus_data, k=len(literature_data)))
    
    larger_equal = []
    smaller = [] 
    for group in random_group:
        us_his = first_digit_histogram(group)
        us_mse = mean_squared_error(us_his, benford_his)
        if us_mse >= literature_mse:
            larger_equal.append(us_mse)
        if us_mse < literature_mse:
            smaller.append(us_mse)

    print("Comparison of US MSEs to literature MSE:")
    print("larger/equal:", len(larger_equal))
    print("smaller:", len(smaller))
    
    return 
  
      
def main(): 
    
    Iran_data = extract_election_vote_counts("election-iran-2009.csv", 
                              ["Ahmadinejad", "Rezai", "Karrubi", "Mousavi"])
    # calculate Iran MSE
    Iran_mse = calculate_mse_with_uniform(Iran_data)
    print("2009 Iranian election MSE:", Iran_mse)

    compare_iranian_mse_to_samples(Iran_mse)
    compare_us_mse_to_samples()    

if __name__ == "__main__":     
    main() 

    
    
    
    
    
    

    
