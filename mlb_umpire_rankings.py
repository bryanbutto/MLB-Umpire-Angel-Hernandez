'''
MLB Umpire Rankings
The data was taken from a Kaggle dataset on MLB umpire performance scorecards from 2015-2022.

https://www.kaggle.com/datasets/mattop/mlb-baseball-umpire-scorecards-2015-2022/data
'''

# Importing Packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from scipy import stats
import statistics
from functools import reduce

# Importing Dataset

ump = pd.read_csv('mlb-umpire-scorecard.csv')

# Head

ump.head()

# Tail

ump.tail()

# Datatypes

ump.info()

# Removing 'ND' values

ump = ump[ump.pitches_called != 'ND']

# Datatypes

ump.count()

# Converting Datatypes

ump['date'] = pd.to_datetime(ump['date'])
ump['pitches_called'] = pd.to_numeric(ump.pitches_called, errors='coerce')
ump['incorrect_calls'] = pd.to_numeric(ump.incorrect_calls, errors='coerce')
ump['expected_incorrect_calls'] = pd.to_numeric(ump.expected_incorrect_calls, errors='coerce')
ump['correct_calls'] = pd.to_numeric(ump.correct_calls, errors='coerce')
ump['expected_correct_calls'] = pd.to_numeric(ump.expected_correct_calls, errors='coerce')
ump['correct_calls_above_expected'] = pd.to_numeric(ump.correct_calls_above_expected, errors='coerce')
ump['accuracy'] = pd.to_numeric(ump.accuracy, errors='coerce')
ump['expected_accuracy'] = pd.to_numeric(ump.expected_accuracy, errors='coerce')
ump['accuracy_above_expected'] = pd.to_numeric(ump.accuracy_above_expected, errors='coerce')
ump['consistency'] = pd.to_numeric(ump.consistency, errors='coerce')
ump['favor_home'] = pd.to_numeric(ump.favor_home, errors='coerce')
ump['total_run_impact'] = pd.to_numeric(ump.total_run_impact, errors='coerce')

# Show Datatypes again

ump.info()

# Descriptive Statistics

ump.describe()

'''
Looking at the descriptive statistics above, there were 18,093 games in the dataset. The average number of pitches called per game was 154.6 with an average of 11.7 incorrect calls and 142.9 correct calls. The average accuracy was 92.4% and the average consistency was 93.2%.
'''

# Boxplot of 'pitches_called'

plt.boxplot(ump['pitches_called'])
plt.show()

# Boxplot of 'incorrect_calls'

plt.boxplot(ump['incorrect_calls'])
plt.show()

# Boxplot of 'correct_calls'

plt.boxplot(ump['correct_calls'])
plt.show()

# Boxplot of 'accuracy'

plt.boxplot(ump['accuracy'])
plt.show()

# Boxplot of 'consistency'

plt.boxplot(ump['consistency'])
plt.show()

# Grouping by Umpire

ump_group = ump.groupby('umpire')

# Counting number of games for each Umpire

ump_games = ump_group['id'].count().sort_values(ascending=False)
ump_games = pd.DataFrame(ump_games).reset_index()
ump_games

# Average accuracy for each Umpire

ump_accuracy = ump_group['accuracy'].mean().sort_values(ascending=False)
ump_accuracy = pd.DataFrame(ump_accuracy).reset_index()
ump_accuracy

# Average consistency for each Umpire

ump_consistency = ump_group['consistency'].mean().sort_values(ascending=False)
ump_consistency = pd.DataFrame(ump_consistency).reset_index()
ump_consistency

# Total pitches called for each Umpire

ump_pitches_called = ump_group['pitches_called'].sum().sort_values(ascending=False)
ump_pitches_called = pd.DataFrame(ump_pitches_called).reset_index()
ump_pitches_called

# Total incorrect calls for each Umpire

ump_incorrect_calls = ump_group['incorrect_calls'].sum().sort_values(ascending=False)
ump_incorrect_calls = pd.DataFrame(ump_incorrect_calls).reset_index()
ump_incorrect_calls

# Total correct calls for each Umpire

ump_correct_calls = ump_group['correct_calls'].sum().sort_values(ascending=False)
ump_correct_calls = pd.DataFrame(ump_correct_calls).reset_index()
ump_correct_calls

# Merging all 6 dfs with ump data together

ump_data = [ump_games, ump_accuracy, ump_consistency, ump_pitches_called, ump_incorrect_calls, ump_correct_calls]
merged_ump_data = reduce(lambda  left,right: pd.merge(left,right,on=['umpire'],
                                            how='outer'), ump_data)
merged_ump_data

merged_ump_data = pd.DataFrame(merged_ump_data)
merged_ump_data.describe()

'''
Angel Hernandez
How does Angel Hernandez compare to other umpires in accuracy and consistency?
'''

# Filtering for Angel Hernadez

angel = ump.loc[ump['umpire'] == 'Angel Hernandez']
angel

# Descriptive Statistics for Angel Hernadez

angel.describe()

'''
Looking at the descriptive statistics above for Angel Hernandez, he called a total of 227 games behind the plate over the 8 year period from 2015-2022.
The average number of pitches he called per game was 155.8 compared to league average of 154.6.
He had an average of 12.7 incorrect calls compared to the league average of 11.7.
He had an average of 143.1 correct calls compared to the league average of 142.9.
He had an average accuracy of 91.9% compared to the league average of 92.4%.
He had an average consistency of 92.7% compared to the league average of 93.2%.
'''

# Histogram for Angel Hernadez for Incorrect Calls

plt.hist(angel['incorrect_calls'])
plt.show()

# Histogram for Angel Hernadez for Accuracy

plt.hist(angel['accuracy'])
plt.show()

# Histogram of 'accuracy' for All umps

plt.hist(ump['accuracy'], color='blue', range=(80,100))
plt.title('Accurcy for All Umpires')
plt.show()

# Histogram of 'accuracy' for Angel Hernadez

plt.hist(angel['accuracy'], color='red', range=(80,100))
plt.title('Accurcy for Angel Hernandez')
plt.show()

# Histogram of 'accuracy' for All umps and for Angel Hernadez

plt.hist(ump['accuracy'], color='blue')
plt.hist(angel['accuracy'], color='red')
plt.title('All Umps vs Angel Hernandez')
plt.legend(['All Umps', 'Angel Hernandez'])
plt.show()

# Average Accurarcy for all umpires and for Angel Hernandez

print("Average Umpire Accuracy:", ump['accuracy'].mean())
print("Angel Hernandez Accuracy:", angel['accuracy'].mean())

# Average Consistency for all umpires and for Angel Hernandez

print("Average Umpire Consistency:", ump['consistency'].mean())
print("Angel Hernandez Consistency:", angel['consistency'].mean())

'''
Where Does Angel Hernandez rank amount umpires in games, accuracy, consistency, pitches called, incorrect calls, and correct calls? There are 124 umpires in the dataset
'''

# Games

ump_games.loc[ump_games['umpire'] == 'Angel Hernandez']

'''
Angel Hernandez ranks 8th in total games called among the 124 umpires with 227 games called
'''

# Accuracy

ump_accuracy.loc[ump_accuracy['umpire'] == 'Angel Hernandez']

'''
Angel Hernandez ranks 85th in average accuracy among the 124 umpires with an average accuracy of 91.9%
'''

# Consistency

ump_consistency.loc[ump_consistency['umpire'] == 'Angel Hernandez']

'''
Angel Hernandez ranks 108th in average consistency among the 124 umpires with an average consitency of 92.7%
'''

# Pitches called

ump_pitches_called.loc[ump_pitches_called['umpire'] == 'Angel Hernandez']

'''
Angel Hernandez ranks 7th in total pitches called among the 124 umpires with a total of 35,363 pitches called
'''

# Incorrect Calls

ump_incorrect_calls.loc[ump_incorrect_calls['umpire'] == 'Angel Hernandez']

'''
Angel Hernandez ranks 4th in total incorrect calls among the 124 umpires with 2,877 incorrect calls
'''

# Correct calls

ump_correct_calls.loc[ump_correct_calls['umpire'] == 'Angel Hernandez']

'''
Angel Hernandez ranks 8th in total correct calls among the 124 umpires with 32,486 correct calls
'''

# ANOVA (Analysis of Variance)
# ANOVA to determine if there is a statistically significant difference in means between the average accuracy and consistency between all umpires and Angel Hernandez

# ANOVA for Accuracy

f_oneway(ump['accuracy'], angel['accuracy'])

# ANOVA for Consistency

f_oneway(ump['consistency'], angel['consistency'])

# T Test
# T Test to determine the statistical significance of the difference in means between the average accuracy and consistency between all umpires and Angel Hernandez

# T Test for Accuracy

stats.ttest_1samp(angel['accuracy'], ump['accuracy'].mean())

# T Test for Consistency

stats.ttest_1samp(angel['consistency'], ump['consistency'].mean())

'''
Given the low pvalues below 0.05 for both ANOVA and the T Test, we can reject the null hypothesis and conclude that Angel Hernandez was a statistically significantly below average umpire according the the metrics of accuracy and consistency.
'''
