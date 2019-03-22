# Author: Nimish Mishra

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures



def load_csv(FILEPATH):
    #
    # @param: path of the file to be loaded
    # @return: Pandas dataframe after reading the file
    #
    data = pd.read_csv(FILEPATH)
    return data

def peak(data):
    #
    # @param: pandas dataframe 
    #
    print(data.head(10)) # Displays the first 10 rows of @param

def plot_data(wind_velocity, power):
    #
    # @param: list of wind velocities; list of corresponsing power output
    #

    plt.figure() # Forces a new window for every new plot
    plt.scatter(wind_velocity, power)
    plt.xlabel('Wind velocity (m/s)')
    plt.ylabel('Total Power Output from model (W)')
    plt.title('Scatter plot of training data from simulations')
    plt.show(block = False) # Allows further code execution while the window is open

def prepare_data(data):
    #
    # @param: the pandas dataframe 
    # @return: the columns power, wind velocity after extraction from the dataframe; train data (after adding an intercept term to list of wind velocities)
    #
    power = data.Power
    wind_velocity = data['Wind velocity']
    plot_data(wind_velocity, power)
    train_data = []
    # Adding the intercept term
    for i in range(len(wind_velocity)):
        velocity = wind_velocity[i]
        train_data.append([1, velocity])
    return power, train_data, wind_velocity

def build_model():
    #
    #@return: A regressor model for further predictions
    #

    data = load_csv('/Users/nimishmishra/Desktop/HUAssignment/Updated Data - Sheet1.csv')
    peak(data)
    power, train_data, wind_velocity = prepare_data(data)
    power = power/(1e6) # Scaling down the output labels
    # The implementation of LinearRegression() from sklearn.linear_model works well for straight line graphs. To learn polynomial functions, 
    # higher order terms need to be added to the existing linear order terms. PolynomialFeatures(degree) serves to extend existing data into a polynomial of
    # required degree. Hence, complex functions can be learnt by the LinearRegression() model

    poly = PolynomialFeatures(degree = 2) 
    train_data = np.array(poly.fit_transform(train_data))
    regressor  = LinearRegression()
    print('Training...')
    regressor.fit(train_data, power)
    plt.figure()
    plt.scatter(wind_velocity, power, color = 'red')
    plt.plot(wind_velocity, regressor.predict(train_data), color = 'blue')
    plt.title('Line of best fit on the training data')
    plt.xlabel('Wind velocity (m / s)')
    plt.ylabel('Total Power output ( x 1e06 W)')
    plt.show(block = False)
    return regressor

def predict(velocity, model):
    to_predict = [[1, velocity]] # Adding the intercept term to the velocity for which power output needs to be predicted
    # Same logic of creation of higher order terms
    polynomial_creator = PolynomialFeatures(degree = 2) 
    poly_to_predict = np.array(polynomial_creator.fit_transform(to_predict))
    prediction = model.predict(poly_to_predict)
    print('Predicted power output for ' + str(velocity) + ' m/s is ' + str(prediction) + ' * 10e6 W')

def calculate_statistics():
    # Loading the result sheet
    final_data = load_csv('/Users/nimishmishra/Desktop/HUAssignment/Wind speed data - Sheet1.csv')
    peak(final_data)
    print('\nWind velocity statistics\n(wind velocities considered here are the mean wind gust\nvelocities in the most productive period of the year in these places)')
    wind_velocity = final_data['Mean Wind velocity (m/s)[29]']
    mean = np.mean(np.array(wind_velocity))
    standard_deviation = np.std(np.array(wind_velocity))
    variance = np.var(np.array(wind_velocity))
    print('\nMean : ' + str(mean) + "\nStandard Deviation: " + str(standard_deviation) + "\nVariance: " + str(variance))

    print('\nTotal Predicted Power output statistics\n')
    power_output = final_data['Total model power output']
    mean = np.mean(np.array(power_output))
    standard_deviation = np.std(np.array(power_output))
    variance = np.var(np.array(power_output))
    print('\nMean : ' + str(mean) + "\nStandard Deviation: " + str(standard_deviation) + "\nVariance: " + str(variance))

    improved = final_data[final_data['Improvement'] == 'y']['Improvement'] # Filtering rows that have performance improvement
    not_improved = final_data[final_data['Improvement'] == 'n']['Improvement'] # Rows that don't show performance improvement
    same_performance = final_data[final_data['Improvement'] == 's']['Improvement']
    improved = improved.values # Conversion to numpy array
    not_improved = not_improved.values
    same_performance = same_performance.values
    plt.figure()
    # Bar chart comparing number of sites showing performance improvement and no performance improvement
    plt.bar([1, 2, 3], [len(improved), len(not_improved), len(same_performance)], width = 0.4, tick_label = ['Improvement', 'No Improvement', 'Nearly same performance'])
    plt.title('Number of sites of improvements (out of ' + str(len(improved) + len(not_improved) + len(same_performance)) + ' sites)')
    plt.ylabel('Number of sites')
    plt.show(block = False)

    

velocity = 12 #The velocity for which power output has to be predicted
model = build_model()
predict(velocity, model)
calculate_statistics()
plt.show()
    