#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    cleaned_data = []
    for index in range(0,len(ages)):
        cleaned_data.append((ages[index],net_worths[index],predictions[index]-net_worths[index]))
        
    cleaned_data.sort(key= lambda cleaned_data: cleaned_data[2], reverse = True)
    del cleaned_data[0:10]
  
    return cleaned_data

