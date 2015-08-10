#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Alex Korbonits
@korbonits
"""

# kaggle-walmart-stormy-weather

# This python file is to be used for looking at the data and outputing a submission file for the above-referenced competition.

import graphlab as gl
import re

key = gl.SFrame.read_csv('key.csv')

weather = gl.SFrame.read_csv('weather.csv')

train = gl.SFrame.read_csv('train.csv')

test = gl.SFrame.read_csv('test.csv')

"""
If one joins train and key, one easily gets the station column. Then you can just join train and weather!

## Let's create to little hashes
store_station = dict([(key[i]['store_nbr'],key[i]['station_nbr']) for i in range(0,len(key))])
station_store = dict([(key[i]['station_nbr'],key[i]['store_nbr']) for i in range(0,len(key))])

## Let's add the weather station as a column in the training data
train['station_nbr'] = train['store_nbr'].apply(lambda x: store_station[x]) ## sweet
"""
## Let's clean up some column names:

def clean_column_names(dataframe):

	cols = sorted(dataframe.column_names())

	renamed = sorted([re.findall(r'"(\w+)"',col)[0] for col in cols])

	for col in cols:
		if re.findall(r'"(\w+)"',col)[0] in renamed:
			dataframe[re.findall(r'"(\w+)"',col)[0]] = dataframe[col]
			dataframe.remove_column(col)

	return dataframe

weather = clean_column_names(weather)

## Let's join train and key
train = train.join(key,on=['store_nbr'],how='inner')

## Let's join train and weather:
train = train.join(weather,on=['date','station_nbr'],how='inner') ## booyah

## Need to do some more data/feature engineering here before I can model.
train['date'] = train['date'].str_to_datetime()

train.split_datetime('date',limit=['year','month','day']) ## need numeric types for regression

## Features
features = ['date.year',
 'date.month',
 'date.day',
 'store_nbr',
 'item_nbr',
 # 'units',
 'station_nbr',
 'avgspeed',
 'codesum',
 'cool',
 'depart',
 'dewpoint',
 'heat',
 'preciptotal',
 'resultdir',
 'resultspeed',
 'sealevel',
 'snowfall',
 'stnpressure',
 'sunrise',
 'sunset',
 'tavg',
 'tmax',
 'tmin',
 'wetbulb']

m = gl.boosted_trees_regression.create(train, features=features, target='units') ## can this improve with numeric types for other columns?

prediction = m.predict(test)

def zeroed(x):

	if x >= 0.0:
		return x
	else:
		return 0

prediction = prediction.apply(lambda x: zeroed(x))

# Create log-units column
import math
train['log-units'] = train['units'].apply(lambda x: math.log(1 + x))

m1 = gl.boosted_trees_regression.create(train, features=features, target='log-units') ## can this improve with numeric types for other columns?

p1 = m1.predict(test).apply(lambda x: math.exp(x)-1).apply(lambda x: x if x > 0 else 0)

dates = gl.SFrame.read_csv('test.csv')['date']

id_col = test['store_nbr'].apply(lambda x: str(x) + '_') + test['item_nbr'].apply(lambda x: str(x) + '_') + dates.apply(lambda x: str(x))

def make_submission(prediction, filename='submission.txt'):
    with open(filename, 'w') as f:
        f.write('id,units\n')
        submission_strings = id_col + ',' + prediction.astype(str)
        for row in submission_strings:
            f.write(row + '\n')

# make_submission(p1, 'submission2.txt')

def parameter_search(training, validation, target):
    """
    Return the optimal parameters in the given search space.
    The parameter returned has the lowest validation rmse.
    """
    job = gl.toolkits.model_parameter_search(gl.boosted_trees_regression.create,
                                             training_set = training,
                                             validation_set = validation,
                                             features = features,
                                             target = target,
                                             max_depth = [10, 15, 20],
                                             min_child_weight = [5, 10, 20],
                                             step_size = [0.05],
                                             max_iterations = [500])


    # When the job is done, the result is a dictionary containing all the models
    # being generated, and a SFrame containing summary of the metrics, for each parameter set.
    result = job.get_results()
    
    models = result['models']
    summary = result['summary']
    
    sorted_summary = summary.sort('validation_rmse', ascending=True)
    print sorted_summary
       
    optimal_model_idx = sorted_summary[0]['model_id']

    # Return the parameters with the lowest validation error. 
    optimal_params = sorted_summary[['max_depth', 'min_child_weight']][0]
    optimal_rmse = sorted_summary[0]['validation_rmse']

    print 'Optimal parameters: %s' % str(optimal_params)
    print 'RMSE: %s' % str(optimal_rmse)
    return optimal_params

training, validation = train.random_split(0.8)

params_log_units = parameter_search(training, validation, target='log-units')

m_log_units = gl.boosted_trees_regression.create(train, target='log-units', features=features, **params_log_units)

final_prediction = m_log_units.predict(test).apply(lambda x: math.exp(x)-1).apply(lambda x: x if x > 0 else 0)
