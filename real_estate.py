# importing libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as sm

# loading the file & exploration
df=pd.read_csv('Real estate.csv')
df.head()
df.info()

# initial model 
model = sm.ols('house_price_of_unit_area~transaction_date+house_age+distance_to_the_nearest_MRT_station+number_of_convenience_stores+latitude+longitude',data=df).fit()
print('MODEL')
print(model.summary())

# outliers
# cook's D value
CooksD=model.get_influence().cooks_distance
# calculate the sample size
n = len(df)
# add variable to the data containing the Cooks D 
df['Outlier']=CooksD[0]>4/n
df.head()

# inspecting outliers
print(df[df.Outlier==True])
outlier = df[df.Outlier==True]
outlier.head()

# heatmap 
sns.heatmap(df.corr(), annot=True,cmap='Reds')


# non-linear relationships 
independent_variable1 = df['transaction_date']
independent_variable2 = df['house_age']
independent_variable3 = df['distance_to_the_nearest_MRT_station']
independent_variable4 = df['number_of_convenience_stores']
independent_variable5 = df['latitude']
independent_variable6 = df['longitude']
dependent_variable = df['house_price_of_unit_area']

# graph1 (same could be done with a pairplot, but for the sake of clarity has been divided)
plt.figure()
sns.regplot(x = independent_variable1,y = dependent_variable,
            data=df,line_kws={'color':'black'},lowess = True)
plt.title('RegPlot Transaction Date')
# graph2
plt.figure()
sns.regplot(x = independent_variable2,y = dependent_variable,
           data=df,line_kws={'color':'black'},lowess = True)
plt.title('RegPlot House Age')
# graph3
plt.figure()
sns.regplot(x = independent_variable3,y = dependent_variable,
           data=df,line_kws={'color':'black'},lowess = True)
plt.title('RegPlot Distance to MRT')
# graph4
plt.figure()
sns.regplot(x = independent_variable4,y = dependent_variable,
            data=df,line_kws={'color':'black'},lowess = True)
plt.title('RegPlot Convenience Stores')
# graph5
plt.figure()
sns.regplot(x = independent_variable5,y = dependent_variable,
            data=df,line_kws={'color':'black'},lowess = True)
plt.title('RegPlot Latitude')
# graph6
plt.figure()
sns.regplot(x = independent_variable6,y = dependent_variable,
            data=df,line_kws={'color':'black'},lowess = True)
plt.title('RegPlot Longitude')

# CREATION OF NEW VARIABLES - HOUSE_AGE
df['transformed_house_age'] = df['house_age'] ** -0.4
# check for and handle infinite or NaN values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(subset=['transformed_house_age'], inplace=True)
# testing of new variables 
model_house_age = sm.ols('house_price_of_unit_area~house_age', data=df).fit()
print('model_house_age')
print(model_house_age.summary())
model_transformed_house_age = sm.ols('house_price_of_unit_area~transformed_house_age', data=df).fit()
print('model_transformed_house_age')
print(model_transformed_house_age.summary())

# CREATION OF NEW VARIABLES - NEAREST_MRT
df['transformed_distance_to_the_nearest_MRT_station'] = df['distance_to_the_nearest_MRT_station'].apply(lambda x: np.log1p(x))
# testing of new variables 
model_MRT = sm.ols('house_price_of_unit_area~distance_to_the_nearest_MRT_station', data=df).fit()
print('model_MRT')
print(model_MRT.summary())
model_transformed_MRT = sm.ols('house_price_of_unit_area~transformed_distance_to_the_nearest_MRT_station', data=df).fit()
print('model_transformed_MRT')
print(model_transformed_MRT.summary())

# CREATION OF NEW VARIABLES - LATITUDE
best_r_squared_latitude = 0
best_exponent_latitude = 0
# specify a range of exponent values from -10 to 10 for latitude
exponent_range_latitude = np.linspace(-10, 10, num=100)
for exponent_latitude in exponent_range_latitude:
    df['transformed_latitude'] = df['latitude'] ** exponent_latitude
    model_latitude = sm.ols(f'house_price_of_unit_area~transformed_latitude', data=df).fit()
    r_squared_latitude = model_latitude.rsquared

    if r_squared_latitude > best_r_squared_latitude:
        best_r_squared_latitude = r_squared_latitude
        best_exponent_latitude = exponent_latitude
print(f'Best R-squared for Latitude: {best_r_squared_latitude} with Exponent: {best_exponent_latitude}')
# creation of new variables - transformed_house_age
df['transformed_latitude'] = df['latitude'] ** -8.98989898989899
# testing of new variables 
model_latitude = sm.ols('house_price_of_unit_area~latitude', data=df).fit()
print('model_latitude')
print(model_latitude.summary())
model_transformed_latitude = sm.ols('house_price_of_unit_area~transformed_latitude', data=df).fit()
print('model_transformed_latitude')
print(model_transformed_latitude.summary())

# CREATION OF NEW VARIABLES - LONGITUDE
best_r_squared_longitude = 0
best_exponent_longitude = 0
# specify a range of exponent values from -10 to 10 for longitude
exponent_range_longitude = np.linspace(-10, 10, num=100)
for exponent_longitude in exponent_range_longitude:
    df['transformed_longitude'] = df['longitude'] ** exponent_longitude
    model_longitude = sm.ols(f'house_price_of_unit_area~transformed_longitude', data=df).fit()
    r_squared_longitude = model_longitude.rsquared
    if r_squared_longitude > best_r_squared_longitude:
        best_r_squared_longitude = r_squared_longitude
        best_exponent_longitude = exponent_longitude
print(f'Best R-squared for Longitude: {best_r_squared_longitude} with Exponent: {best_exponent_longitude}')
# creation of new variables - transformed_house_age
df['transformed_longitude'] = df['longitude'] ** -5.555555555555555
# testing of new variables 
model_longitude = sm.ols('house_price_of_unit_area~longitude', data=df).fit()
print('model_longitude')
print(model_longitude.summary())
model_transformed_longitude = sm.ols('house_price_of_unit_area~transformed_longitude', data=df).fit()
print('model_transformed_longitude')
print(model_transformed_longitude.summary())


# final model 
model_final = sm.ols('house_price_of_unit_area~transaction_date+house_age+transformed_house_age+distance_to_the_nearest_MRT_station+transformed_distance_to_the_nearest_MRT_station+number_of_convenience_stores+latitude+transformed_latitude+longitude+transformed_longitude',data=df).fit()
print('model_final')
print(model_final.summary())