#!/usr/bin/env python
# coding: utf-8

# # Housing Sales Price Prediction

# In[1]:


#Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import math


# In[2]:


pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)


# ## Read data

# In[3]:


d_tr=pd.read_csv(r"C:\Users\prath\Desktop\dataset python\Property_Price_Train.csv")


# In[4]:


d_tr.head()


# In[5]:


d_tr.tail()


# In[6]:


d_tr.columns


# In[7]:


d_tr.describe()


# In[8]:


#d_tr=d_tr.drop


# In[9]:


d_te=pd.read_csv(r"C:\Users\prath\Desktop\dataset python\Property_Price_Test.csv")


# In[10]:


d_te.head()


# In[11]:


d_te.tail()


# In[12]:


d_te.describe()


# In[13]:


#Check shape of datasets


# In[14]:


d_tr.shape,d_te.shape


# In[15]:


#Check null values and drop columns [which is having more than 50% missing values]


# In[16]:


per=(100*d_tr.isnull().sum()/len(d_tr))
per


# In[17]:


d_tr=d_tr.drop(["Lane_Type","Pool_Quality","Fence_Quality","Miscellaneous_Feature"],axis=1)


# In[18]:


d_tr.isnull().sum()


# In[19]:


#find out each has value count


# In[20]:


d_tr.Lot_Extent.value_counts()
d_tr.Basement_Height.value_counts()
d_tr.Basement_Condition.value_counts()
d_tr.Exposure_Level.value_counts()
d_tr.BsmtFinType1.value_counts()
d_tr.BsmtFinType2.value_counts()
d_tr.Brick_Veneer_Type.value_counts()
d_tr.Brick_Veneer_Area.value_counts()
d_tr.Electrical_System.value_counts()
d_tr.Fireplace_Quality.value_counts()
d_tr.Garage.value_counts()
d_tr.Garage_Built_Year.value_counts()
d_tr.Garage_Finish_Year.value_counts()
d_tr.Garage_Quality.value_counts()
d_tr.Garage_Condition.value_counts()


# In[21]:


d_tr.Lot_Extent=d_tr.Lot_Extent.fillna(d_tr.Lot_Extent.mean())
d_tr.Basement_Height=d_tr.Basement_Height.fillna("Fa")
d_tr.Basement_Condition=d_tr.Basement_Condition.fillna("Po")
d_tr.Exposure_Level=d_tr.Exposure_Level.fillna("Mn")
d_tr.BsmtFinType1=d_tr.BsmtFinType1.fillna("LwQ")
d_tr.BsmtFinType2=d_tr.BsmtFinType2.fillna("GLQ")
d_tr.Brick_Veneer_Type=d_tr.Brick_Veneer_Type.fillna("BrkCmn")
d_tr.Brick_Veneer_Area=d_tr.Brick_Veneer_Area.fillna(d_tr.Brick_Veneer_Area.mean())
d_tr.Electrical_System=d_tr.Electrical_System.fillna("Mix")
d_tr.Fireplace_Quality=d_tr.Fireplace_Quality.fillna("Po")
d_tr.Garage=d_tr.Garage.fillna("2Types")
d_tr.Garage_Built_Year=d_tr.Garage_Built_Year.fillna(d_tr.Garage_Built_Year.mean())
d_tr.Garage_Finish_Year=d_tr.Garage_Finish_Year.fillna("Fin")
d_tr.Garage_Quality=d_tr.Garage_Quality.fillna("Po")
d_tr.Garage_Condition=d_tr.Garage_Condition.fillna("Ex")


# In[22]:


d_tr.isnull().sum()


# In[23]:


#Dtype conversion


# In[24]:


d_tr.dtypes


# In[25]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[26]:


d_tr.Zoning_Class=le.fit_transform(d_tr.Zoning_Class)
d_tr.Road_Type=le.fit_transform(d_tr.Road_Type)
d_tr.Property_Shape=le.fit_transform(d_tr.Property_Shape)
d_tr.Land_Outline=le.fit_transform(d_tr.Land_Outline)
d_tr.Utility_Type=le.fit_transform(d_tr.Utility_Type)
d_tr.Lot_Configuration=le.fit_transform(d_tr.Lot_Configuration)
d_tr.Property_Slope=le.fit_transform(d_tr.Property_Slope)
d_tr.Neighborhood=le.fit_transform(d_tr.Neighborhood)
d_tr.Condition1=le.fit_transform(d_tr.Condition1)
d_tr.Condition2=le.fit_transform(d_tr.Condition2)
d_tr.House_Type=le.fit_transform(d_tr.House_Type)
d_tr.House_Design=le.fit_transform(d_tr.House_Design)
d_tr.Roof_Design=le.fit_transform(d_tr.Roof_Design)
d_tr.Roof_Quality=le.fit_transform(d_tr.Roof_Quality)
d_tr.Exterior1st=le.fit_transform(d_tr.Exterior1st)
d_tr.Exterior2nd=le.fit_transform(d_tr.Exterior2nd)
d_tr.Brick_Veneer_Type=le.fit_transform(d_tr.Brick_Veneer_Type)
d_tr.Exterior_Material=le.fit_transform(d_tr.Exterior_Material)
d_tr.Exterior_Condition=le.fit_transform(d_tr.Exterior_Condition)
d_tr.Foundation_Type=le.fit_transform(d_tr.Foundation_Type)
d_tr.Basement_Height=le.fit_transform(d_tr.Basement_Height)
d_tr.Basement_Condition=le.fit_transform(d_tr.Basement_Condition)
d_tr.Exposure_Level=le.fit_transform(d_tr.Exposure_Level)
d_tr.BsmtFinType1=le.fit_transform(d_tr.BsmtFinType1)
d_tr.BsmtFinType2=le.fit_transform(d_tr.BsmtFinType2)
d_tr.Heating_Type=le.fit_transform(d_tr.Heating_Type)
d_tr.Heating_Quality=le.fit_transform(d_tr.Heating_Quality)
d_tr.Air_Conditioning=le.fit_transform(d_tr.Air_Conditioning)
d_tr.Electrical_System=le.fit_transform(d_tr.Electrical_System)
d_tr.Kitchen_Quality=le.fit_transform(d_tr.Kitchen_Quality)
d_tr.Functional_Rate=le.fit_transform(d_tr.Functional_Rate)
d_tr.Fireplace_Quality=le.fit_transform(d_tr.Fireplace_Quality)
d_tr.Garage=le.fit_transform(d_tr.Garage)
d_tr.Garage_Finish_Year=le.fit_transform(d_tr.Garage_Finish_Year)
d_tr.Garage_Quality=le.fit_transform(d_tr.Garage_Quality)
d_tr.Garage_Condition=le.fit_transform(d_tr.Garage_Condition)
d_tr.Pavedd_Drive=le.fit_transform(d_tr.Pavedd_Drive)
d_tr.Sale_Type=le.fit_transform(d_tr.Sale_Type)
d_tr.Sale_Condition=le.fit_transform(d_tr.Sale_Condition)


# In[27]:


d_tr.dtypes


# In[28]:


#Basemodel building


# In[29]:


data_x=d_tr.iloc[:,0:75]
data_x.head()


# In[30]:


data_y=d_tr.iloc[:,-1]
data_y.head()


# In[31]:


d_te=d_te.iloc[:,1:]
d_te.head()


# In[32]:


import sklearn
from sklearn.model_selection import train_test_split

data_xtrain,data_xtest,data_ytrain,data_ytest=train_test_split(data_x,data_y,test_size=0.2,random_state=101)
data_xtrain.shape,data_xtest.shape,data_ytrain.shape,data_ytest.shape


# In[33]:


from sklearn import linear_model
ln=linear_model.LinearRegression()

ln.fit(data_xtrain,data_ytrain)


# In[34]:


prediction=ln.predict(data_xtest)
#prediction


# In[35]:


#parameters of model B1,B2,...,B5
ln.coef_


# In[36]:


#intercept on y axis [B0]
ln.intercept_


# In[37]:


#to select best fit model


# In[38]:


rsquare=ln.score(data_xtrain,data_ytrain)
rsquare


# In[39]:


adjR2=1-(((1-rsquare)*(75-1))/(75-5-1))
adjR2


# In[40]:


from sklearn import metrics


# In[41]:


MAE=metrics.mean_absolute_error(data_ytest,prediction)
MAE


# In[42]:


MSE=metrics.mean_squared_error(data_ytest,prediction)
MSE


# In[43]:


#different types of error


# In[44]:


error=data_ytest-prediction
error


# In[45]:


abs_error=np.abs(error)
abs_error


# In[46]:


MAPE=np.mean(abs_error/data_ytest)*100
MAPE


# In[47]:


accuracy=100-MAPE
accuracy


# TEST data 

# In[48]:


100*d_te.isnull().sum()/len(d_te)


# In[49]:


d_te=d_te.drop(["Lane_Type","Fireplace_Quality","Pool_Quality","Fence_Quality","Miscellaneous_Feature"],axis=1)


# In[50]:


d_te.isnull().sum()


# In[51]:


d_te.Zoning_Class.value_counts()
d_te.Lot_Extent.value_counts()
d_te.Utility_Type.value_counts()
d_te.Exterior1st.value_counts()
d_te.Exterior2nd.value_counts()
d_te.Brick_Veneer_Type.value_counts()
d_te.Brick_Veneer_Area.value_counts()
d_te.Basement_Height.value_counts()
d_te.Basement_Condition.value_counts()
d_te.Exposure_Level.value_counts()
d_te.BsmtFinType1.value_counts()
d_te.BsmtFinSF1.value_counts()
d_te.BsmtFinType2.value_counts()
d_te.BsmtFinSF2.value_counts()
d_te.BsmtUnfSF.value_counts()
d_te.Total_Basement_Area.value_counts()
d_te.Underground_Full_Bathroom.value_counts()
d_te.Underground_Half_Bathroom.value_counts()
d_te.Kitchen_Quality.value_counts()
d_te.Functional_Rate.value_counts()
d_te.Garage.value_counts()
d_te.Garage_Built_Year.value_counts()
d_te.Garage_Finish_Year.value_counts()
d_te.Garage_Size.value_counts()
d_te.Garage_Area.value_counts()
d_te.Garage_Quality.value_counts()
d_te.Garage_Condition.value_counts()
d_te.Sale_Type.value_counts()


# In[52]:


d_te.Zoning_Class=d_te.Zoning_Class.fillna("RHD")
d_te.Lot_Extent=d_te.Lot_Extent.fillna(d_te.Lot_Extent.mean())
d_te.Utility_Type=d_te.Utility_Type.fillna("AllPub")
d_te.Exterior1st=d_te.Exterior1st.fillna("CB")
d_te.Exterior2nd=d_te.Exterior2nd.fillna("Stone")
d_te.Brick_Veneer_Type=d_te.Brick_Veneer_Type.fillna("BrkCmn")
d_te.Brick_Veneer_Area=d_te.Brick_Veneer_Area.fillna(d_te.Brick_Veneer_Area.mean())
d_te.Basement_Height=d_te.Basement_Height.fillna("Fa")
d_te.Basement_Condition=d_te.Basement_Condition.fillna("Po")
d_te.Exposure_Level=d_te.Exposure_Level.fillna("Mn")
d_te.BsmtFinType1=d_te.BsmtFinType1.fillna("LwQ")
d_te.BsmtFinSF1=d_te.BsmtFinSF1.fillna(d_te.BsmtFinSF1.mean())
d_te.BsmtFinType2=d_te.BsmtFinType2.fillna("GLQ")
d_te.BsmtFinSF2=d_te.BsmtFinSF2.fillna(d_te.BsmtFinSF2.mean())
d_te.BsmtUnfSF=d_te.BsmtUnfSF.fillna(d_te.BsmtUnfSF.mean())
d_te.Total_Basement_Area=d_te.Total_Basement_Area.fillna(d_te.Total_Basement_Area.mean())
d_te.Underground_Full_Bathroom=d_te.Underground_Full_Bathroom.fillna(3.0)
d_te.Underground_Half_Bathroom=d_te.Underground_Half_Bathroom.fillna(2.0)
d_te.Kitchen_Quality=d_te.Kitchen_Quality.fillna("Fa")
d_te.Functional_Rate=d_te.Functional_Rate.fillna("MS")
d_te.Garage=d_te.Garage.fillna("CarPort")
d_te.Garage_Built_Year=d_te.Garage_Built_Year.fillna(d_te.Garage_Built_Year.mean())
d_te.Garage_Finish_Year=d_te.Garage_Finish_Year.fillna("Fin")
d_te.Garage_Size=d_te.Garage_Size.fillna(5.0)
d_te.Garage_Area=d_te.Garage_Area.fillna(d_te.Garage_Area.mean())
d_te.Garage_Quality=d_te.Garage_Quality.fillna("Po")
d_te.Garage_Condition=d_te.Garage_Condition.fillna("Ex")
d_te.Sale_Type=d_te.Sale_Type.fillna("Con")


# In[53]:


d_te.isnull().sum()


# In[54]:


d_te.dtypes


# In[55]:


d_te.Zoning_Class=le.fit_transform(d_te.Zoning_Class)
d_te.Road_Type=le.fit_transform(d_te.Road_Type)
d_te.Property_Shape=le.fit_transform(d_te.Property_Shape)
d_te.Land_Outline=le.fit_transform(d_te.Land_Outline)
d_te.Utility_Type=le.fit_transform(d_te.Utility_Type)
d_te.Lot_Configuration=le.fit_transform(d_te.Lot_Configuration)
d_te.Property_Slope=le.fit_transform(d_te.Property_Slope)
d_te.Neighborhood=le.fit_transform(d_te.Neighborhood)
d_te.Condition1=le.fit_transform(d_te.Condition1)
d_te.Condition2=le.fit_transform(d_te.Condition2)
d_te.House_Type=le.fit_transform(d_te.House_Type)
d_te.House_Design=le.fit_transform(d_te.House_Design)
d_te.Roof_Design=le.fit_transform(d_te.Roof_Design)
d_te.Roof_Quality=le.fit_transform(d_te.Roof_Quality)
d_te.Exterior1st=le.fit_transform(d_te.Exterior1st)
d_te.Exterior2nd=le.fit_transform(d_te.Exterior2nd)
d_te.Brick_Veneer_Type=le.fit_transform(d_te.Brick_Veneer_Type)
d_te.Exterior_Material=le.fit_transform(d_te.Exterior_Material)
d_te.Exterior_Condition=le.fit_transform(d_te.Exterior_Condition)
d_te.Foundation_Type=le.fit_transform(d_te.Foundation_Type)
d_te.Basement_Height=le.fit_transform(d_te.Basement_Height)
d_te.Basement_Condition=le.fit_transform(d_te.Basement_Condition)
d_te.Exposure_Level=le.fit_transform(d_te.Exposure_Level)
d_te.BsmtFinType1=le.fit_transform(d_te.BsmtFinType1)
d_te.Exterior_Material=le.fit_transform(d_te.Exterior_Material)
d_te.Exterior_Condition=le.fit_transform(d_te.Exterior_Condition)
d_te.Foundation_Type=le.fit_transform(d_te.Foundation_Type)
d_te.Basement_Height=le.fit_transform(d_te.Basement_Height)
d_te.Basement_Condition=le.fit_transform(d_te.Basement_Condition)
d_te.Exposure_Level=le.fit_transform(d_te.Exposure_Level)
d_te.BsmtFinType2 =le.fit_transform(d_te.BsmtFinType2 )
d_te.Heating_Type=le.fit_transform(d_te.Heating_Type)
d_te.Heating_Quality=le.fit_transform(d_te.Heating_Quality)
d_te.Air_Conditioning=le.fit_transform(d_te.Air_Conditioning)
d_te.Electrical_System=le.fit_transform(d_te.Electrical_System)
d_te.Kitchen_Quality=le.fit_transform(d_te.Kitchen_Quality)
d_te.Functional_Rate=le.fit_transform(d_te.Functional_Rate)
d_te.Garage=le.fit_transform(d_te.Garage)
d_te.Garage_Finish_Year=le.fit_transform(d_te.Garage_Finish_Year)
d_te.Garage_Quality=le.fit_transform(d_te.Garage_Quality)
d_te.Garage_Condition=le.fit_transform(d_te.Garage_Condition)
d_te.Pavedd_Drive=le.fit_transform(d_te.Pavedd_Drive)
d_te.Sale_Type=le.fit_transform(d_te.Sale_Type)
d_te.Sale_Condition=le.fit_transform(d_te.Sale_Condition)


# In[56]:


d_te.isnull().sum()


# In[57]:


d_te.shape


# In[58]:


d_te.boxplot(figsize=(35,25))


# In[59]:


fig,ax=plt.subplots(19,4,figsize=(50,45))
sns.countplot("Building_Class",data=d_tr,ax=ax[0][0])
sns.countplot("Zoning_Class",data=d_tr,ax=ax[0][1])
sns.countplot("Lot_Extent",data=d_tr,ax=ax[0][2])
sns.countplot("Lot_Size",data=d_tr,ax=ax[0][3])
sns.countplot("Road_Type",data=d_tr,ax=ax[1][0])
sns.countplot("Property_Shape",data=d_tr,ax=ax[1][1])
sns.countplot("Land_Outline",data=d_tr,ax=ax[1][2])
sns.countplot("Utility_Type",data=d_tr,ax=ax[1][3])
sns.countplot("Lot_Configuration",data=d_tr,ax=ax[2][0])
sns.countplot("Property_Slope",data=d_tr,ax=ax[2][1])
sns.countplot("Neighborhood",data=d_tr,ax=ax[2][2])
sns.countplot("Condition1",data=d_tr,ax=ax[2][3])
sns.countplot("Condition2",data=d_tr,ax=ax[3][0])
sns.countplot("House_Type",data=d_tr,ax=ax[3][1])
sns.countplot("House_Design",data=d_tr,ax=ax[3][2])
sns.countplot("Overall_Material",data=d_tr,ax=ax[3][3])
sns.countplot("House_Condition",data=d_tr,ax=ax[4][0])
sns.countplot("Construction_Year",data=d_tr,ax=ax[4][1])
sns.countplot("Remodel_Year",data=d_tr,ax=ax[4][2])
sns.countplot("Roof_Design",data=d_tr,ax=ax[4][3])
sns.countplot("Roof_Quality",data=d_tr,ax=ax[5][0])
sns.countplot("Exterior1st",data=d_tr,ax=ax[5][1])
sns.countplot("Exterior2nd",data=d_tr,ax=ax[5][2])
sns.countplot("Brick_Veneer_Type",data=d_tr,ax=ax[5][3])
sns.countplot("Brick_Veneer_Area",data=d_tr,ax=ax[6][0])
sns.countplot("Exterior_Material",data=d_tr,ax=ax[6][1])
sns.countplot("Exterior_Condition",data=d_tr,ax=ax[6][2])
sns.countplot("Foundation_Type",data=d_tr,ax=ax[6][3])
sns.countplot("Basement_Height",data=d_tr,ax=ax[7][0])
sns.countplot("Basement_Condition",data=d_tr,ax=ax[7][1])
sns.countplot("Exposure_Level",data=d_tr,ax=ax[7][2])
sns.countplot("BsmtFinType1",data=d_tr,ax=ax[7][3])
sns.countplot("BsmtFinSF1",data=d_tr,ax=ax[8][0])
sns.countplot("BsmtFinType2",data=d_tr,ax=ax[8][1])
sns.countplot("BsmtFinSF2",data=d_tr,ax=ax[8][2])
sns.countplot("BsmtUnfSF",data=d_tr,ax=ax[8][3])
sns.countplot("Total_Basement_Area",data=d_tr,ax=ax[9][0])
sns.countplot("Heating_Type",data=d_tr,ax=ax[9][1])
sns.countplot("Heating_Quality",data=d_tr,ax=ax[9][2])
sns.countplot("Air_Conditioning",data=d_tr,ax=ax[9][3])
sns.countplot("Electrical_System",data=d_tr,ax=ax[10][0])
sns.countplot("First_Floor_Area",data=d_tr,ax=ax[10][1])
sns.countplot("Second_Floor_Area",data=d_tr,ax=ax[10][2])
sns.countplot("LowQualFinSF",data=d_tr,ax=ax[10][3])
sns.countplot("Grade_Living_Area",data=d_tr,ax=ax[11][0])
sns.countplot("Underground_Full_Bathroom",data=d_tr,ax=ax[11][1])
sns.countplot("Underground_Half_Bathroom",data=d_tr,ax=ax[11][2])
sns.countplot("Full_Bathroom_Above_Grade",data=d_tr,ax=ax[11][3])
sns.countplot("Half_Bathroom_Above_Grade",data=d_tr,ax=ax[12][0])
sns.countplot("Bedroom_Above_Grade",data=d_tr,ax=ax[12][1])
sns.countplot("Kitchen_Above_Grade",data=d_tr,ax=ax[12][2])
sns.countplot("Kitchen_Quality",data=d_tr,ax=ax[12][3])
sns.countplot("Rooms_Above_Grade",data=d_tr,ax=ax[13][0])
sns.countplot("Functional_Rate",data=d_tr,ax=ax[13][1])
sns.countplot("Fireplaces",data=d_tr,ax=ax[13][2])
sns.countplot("Fireplace_Quality",data=d_tr,ax=ax[13][3])
sns.countplot("Garage",data=d_tr,ax=ax[14][0])
sns.countplot("Garage_Built_Year",data=d_tr,ax=ax[14][1])
sns.countplot("Garage_Finish_Year",data=d_tr,ax=ax[14][2])
sns.countplot("Garage_Size",data=d_tr,ax=ax[14][3])
sns.countplot("Garage_Area",data=d_tr,ax=ax[15][0])
sns.countplot("Garage_Quality",data=d_tr,ax=ax[15][1])
sns.countplot("Garage_Condition",data=d_tr,ax=ax[15][2])
sns.countplot("Pavedd_Drive",data=d_tr,ax=ax[15][3])
sns.countplot("W_Deck_Area",data=d_tr,ax=ax[16][0])
sns.countplot("Open_Lobby_Area",data=d_tr,ax=ax[16][1])
sns.countplot("Enclosed_Lobby_Area",data=d_tr,ax=ax[16][2])
sns.countplot("Three_Season_Lobby_Area",data=d_tr,ax=ax[16][3])
sns.countplot("Screen_Lobby_Area",data=d_tr,ax=ax[17][0])
sns.countplot("Pool_Area",data=d_tr,ax=ax[17][1])
sns.countplot("Miscellaneous_Value",data=d_tr,ax=ax[17][2])
sns.countplot("Month_Sold",data=d_tr,ax=ax[17][3])
sns.countplot("Year_Sold",data=d_tr,ax=ax[18][0])
sns.countplot("Sale_Type",data=d_tr,ax=ax[18][1])
sns.countplot("Sale_Condition",data=d_tr,ax=ax[18][2])
sns.countplot("Sale_Price",data=d_tr,ax=ax[18][3])


# In[60]:


d_te.columns


# In[61]:


fig,ax=plt.subplots(20,4,figsize=(50,45))
sns.countplot("Building_Class",data=d_te,ax=ax[0][0])
sns.countplot("Zoning_Class",data=d_te,ax=ax[0][1])
sns.countplot("Lot_Extent",data=d_te,ax=ax[0][2])
sns.countplot("Lot_Size",data=d_te,ax=ax[0][3])
sns.countplot("Road_Type",data=d_te,ax=ax[1][0])
sns.countplot("Property_Shape",data=d_te,ax=ax[1][1])
sns.countplot("Land_Outline",data=d_te,ax=ax[1][2])
sns.countplot("Utility_Type",data=d_te,ax=ax[1][3])
sns.countplot("Lot_Configuration",data=d_te,ax=ax[2][0])
sns.countplot("Property_Slope",data=d_te,ax=ax[2][1])
sns.countplot("Neighborhood",data=d_te,ax=ax[2][2])
sns.countplot("Condition1",data=d_te,ax=ax[2][3])
sns.countplot("Condition2",data=d_te,ax=ax[3][0])
sns.countplot("House_Type",data=d_te,ax=ax[3][1])
sns.countplot("House_Design",data=d_te,ax=ax[3][2])
sns.countplot("Overall_Material",data=d_te,ax=ax[3][3])
sns.countplot("House_Condition",data=d_te,ax=ax[4][0])
sns.countplot("Construction_Year",data=d_te,ax=ax[4][1])
sns.countplot("Remodel_Year",data=d_te,ax=ax[4][2])
sns.countplot("Roof_Design",data=d_te,ax=ax[4][3])
sns.countplot("Roof_Quality",data=d_te,ax=ax[5][0])
sns.countplot("Exterior1st",data=d_te,ax=ax[5][1])
sns.countplot("Exterior2nd",data=d_te,ax=ax[5][2])
sns.countplot("Brick_Veneer_Type",data=d_te,ax=ax[5][3])
sns.countplot("Brick_Veneer_Area",data=d_te,ax=ax[6][0])
sns.countplot("Exterior_Material",data=d_te,ax=ax[6][1])
sns.countplot("Exterior_Condition",data=d_te,ax=ax[6][2])
sns.countplot("Foundation_Type",data=d_te,ax=ax[6][3])
sns.countplot("Basement_Height",data=d_te,ax=ax[7][0])
sns.countplot("Basement_Condition",data=d_te,ax=ax[7][1])
sns.countplot("Exposure_Level",data=d_te,ax=ax[7][2])
sns.countplot("BsmtFinType1",data=d_te,ax=ax[7][3])
sns.countplot("BsmtFinSF1",data=d_te,ax=ax[8][0])
sns.countplot("BsmtFinType2",data=d_te,ax=ax[8][1])
sns.countplot("BsmtFinSF2",data=d_te,ax=ax[8][2])
sns.countplot("BsmtUnfSF",data=d_te,ax=ax[8][3])
sns.countplot("Total_Basement_Area",data=d_te,ax=ax[9][0])
sns.countplot("Heating_Type",data=d_te,ax=ax[9][1])
sns.countplot("Heating_Quality",data=d_te,ax=ax[9][2])
sns.countplot("Air_Conditioning",data=d_te,ax=ax[9][3])
sns.countplot("Electrical_System",data=d_te,ax=ax[10][0])
sns.countplot("First_Floor_Area",data=d_te,ax=ax[10][1])
sns.countplot("Second_Floor_Area",data=d_te,ax=ax[10][2])
sns.countplot("LowQualFinSF",data=d_te,ax=ax[10][3])
sns.countplot("Grade_Living_Area",data=d_te,ax=ax[11][0])
sns.countplot("Underground_Full_Bathroom",data=d_te,ax=ax[11][1])
sns.countplot("Underground_Half_Bathroom",data=d_te,ax=ax[11][2])
sns.countplot("Full_Bathroom_Above_Grade",data=d_te,ax=ax[11][3])
sns.countplot("Half_Bathroom_Above_Grade",data=d_te,ax=ax[12][0])
sns.countplot("Bedroom_Above_Grade",data=d_te,ax=ax[12][1])
sns.countplot("Kitchen_Above_Grade",data=d_te,ax=ax[12][2])
sns.countplot("Kitchen_Quality",data=d_te,ax=ax[12][3])
sns.countplot("Rooms_Above_Grade",data=d_te,ax=ax[13][0])
sns.countplot("Functional_Rate",data=d_te,ax=ax[13][1])
sns.countplot("Fireplaces",data=d_te,ax=ax[13][2])
sns.countplot("Garage",data=d_te,ax=ax[13][3])
sns.countplot("Garage_Built_Year",data=d_te,ax=ax[14][0])
sns.countplot("Garage_Finish_Year",data=d_te,ax=ax[14][1])
sns.countplot("Garage_Size",data=d_te,ax=ax[14][2])
sns.countplot("Garage_Area",data=d_te,ax=ax[14][3])
sns.countplot("Garage_Quality",data=d_te,ax=ax[15][0])
sns.countplot("Garage_Condition",data=d_te,ax=ax[15][1])
sns.countplot("Pavedd_Drive",data=d_te,ax=ax[15][2])
sns.countplot("W_Deck_Area",data=d_te,ax=ax[15][3])
sns.countplot("Open_Lobby_Area",data=d_te,ax=ax[16][0])
sns.countplot("Enclosed_Lobby_Area",data=d_te,ax=ax[16][1])
sns.countplot("Three_Season_Lobby_Area",data=d_te,ax=ax[16][2])
sns.countplot("Screen_Lobby_Area",data=d_te,ax=ax[16][3])
sns.countplot("Pool_Area",data=d_te,ax=ax[17][0])
sns.countplot("Miscellaneous_Value",data=d_te,ax=ax[17][1])
sns.countplot("Month_Sold",data=d_te,ax=ax[17][2])
sns.countplot("Year_Sold",data=d_te,ax=ax[17][3])
sns.countplot("Sale_Type",data=d_te,ax=ax[18][0])
sns.countplot("Sale_Condition",data=d_te,ax=ax[18][1])


# In[62]:


d_tr=d_tr.drop_duplicates()
d_te=d_te.drop_duplicates()


# In[63]:


d_tr.shape,d_te.shape


# In[64]:


d_tr1=d_tr["Id"]


# In[65]:


d_tr=d_tr.drop(["Id"],axis=1)


# In[66]:


d_tr.shape


# In[67]:


# Check correlation


# In[68]:


d=d_tr.corr()


# In[143]:


plt.figure(figsize=(55,45))
heatmap=sns.heatmap(d,linewidth=1,annot=True,cmap=plt.cm.Blues)
plt.title("Heatmap using Seaborn Method")
plt.show()


# In[144]:


d_tr.columns


# In[145]:


c=pd.DataFrame(d_tr,columns={'Garage_Finish_Year', 'Garage_Size', 'Garage_Area', 'Garage_Quality',
       'Garage_Condition', 'Pavedd_Drive', 'W_Deck_Area', 'Open_Lobby_Area',
       'Enclosed_Lobby_Area', 'Three_Season_Lobby_Area', 'Screen_Lobby_Area',
       'Pool_Area', 'Miscellaneous_Value', 'Month_Sold', 'Year_Sold',
       'Sale_Type', 'Sale_Condition','Sale_Price'})


# In[146]:


c1=c.corr()


# In[147]:


plt.figure(figsize=(35,25))
heatmap=sns.heatmap(c1,linewidth=1,annot=True,cmap=plt.cm.Blues)
plt.title("Heatmap using Seaborn Method")
plt.show()


# In[148]:


d_tr.boxplot(figsize=(87,88))


# In[149]:


d_tr.Lot_Extent.value_counts()
d_tr.Lot_Size.value_counts()
d_tr.Brick_Veneer_Area.value_counts()
d_tr.BsmtFinSF1.value_counts()
d_tr.BsmtUnfSF.value_counts()
d_tr.Total_Basement_Area.value_counts()
d_tr.First_Floor_Area.value_counts()
d_tr.Second_Floor_Area.value_counts()
d_tr.Grade_Living_Area.value_counts()
d_tr.Garage_Area.value_counts()
d_tr.W_Deck_Area.value_counts()
d_tr.Open_Lobby_Area.value_counts()
d_tr.Enclosed_Lobby_Area.value_counts()
d_tr.Sale_Price.value_counts()


# In[150]:


#d_tr.boxplot("Lot_Extent")
#d_tr.boxplot("Lot_Size")
#d_tr.boxplot("Brick_Veneer_Area")
#d_tr.boxplot("BsmtFinSF1")
#d_tr.boxplot("BsmtUnfSF")
#d_tr.boxplot("Total_Basement_Area")
#d_tr.boxplot("BsmtUnfSF")
#d_tr.boxplot("Second_Floor_Area")
#d_tr.boxplot("Grade_Living_Area")
#d_tr.boxplot("Garage_Area")
#d_tr.boxplot("W_Deck_Area")
#d_tr.boxplot("Open_Lobby_Area")
#d_tr.boxplot("Enclosed_Lobby_Area")
#d_tr.boxplot("Sale_Price")


# In[151]:


d_tr.boxplot(["Lot_Extent","Lot_Size","Brick_Veneer_Area","BsmtFinSF1","BsmtUnfSF","Total_Basement_Area","BsmtUnfSF","Second_Floor_Area","Grade_Living_Area","Garage_Area","W_Deck_Area","Open_Lobby_Area","Enclosed_Lobby_Area","Sale_Price"],figsize=(45,35))


# In[152]:


#d_tr.Lot_Extent.hist()
#d_tr.Lot_Size.hist()
#d_tr.Brick_Veneer_Area.hist()
#d_tr.BsmtFinSF1.hist()
#d_tr.BsmtUnfSF.hist()
#d_tr.Total_Basement_Area.hist()
#d_tr.First_Floor_Area.hist()
#d_tr.Second_Floor_Area.hist()
#d_tr.Grade_Living_Area.hist()
#d_tr.Garage_Area.hist()
#d_tr.W_Deck_Area.hist()
#d_tr.Open_Lobby_Area.hist()
#d_tr.Enclosed_Lobby_Area.hist()
#d_tr.Sale_Price.hist()


# In[153]:


col=pd.DataFrame(d_tr,columns={"Lot_Extent","Lot_Size","Brick_Veneer_Area","BsmtFinSF1","BsmtUnfSF","Total_Basement_Area","BsmtUnfSF","Second_Floor_Area","Grade_Living_Area","Garage_Area","W_Deck_Area","Open_Lobby_Area","Enclosed_Lobby_Area","Sale_Price"})
col.shape


# In[154]:


fig_with_outlier_treat=col.hist(figsize=(25,30),bins=50,color="brown",edgecolor="black",xlabelsize=8,ylabelsize=8)


# In[ ]:





# In[155]:


iqr1=d_tr["Lot_Extent"].quantile(0.75)-d_tr["Lot_Extent"].quantile(0.25)
iqr1


# In[156]:


iqr2=d_tr["Lot_Size"].quantile(0.75)-d_tr["Lot_Size"].quantile(0.25)
iqr2


# In[157]:


iqr3=d_tr["Brick_Veneer_Area"].quantile(0.75)-d_tr["Brick_Veneer_Area"].quantile(0.25)
iqr3


# In[158]:


iqr4=d_tr["BsmtFinSF1"].quantile(0.75)-d_tr["BsmtFinSF1"].quantile(0.25)
iqr4


# In[159]:


iqr5=d_tr["BsmtUnfSF"].quantile(0.75)-d_tr["BsmtUnfSF"].quantile(0.25)
iqr5


# In[160]:


iqr6=d_tr["Total_Basement_Area"].quantile(0.75)-d_tr["Total_Basement_Area"].quantile(0.25)
iqr6


# In[161]:


iqr7=d_tr["First_Floor_Area"].quantile(0.75)-d_tr["First_Floor_Area"].quantile(0.25)
iqr7


# In[162]:


iqr8=d_tr["Second_Floor_Area"].quantile(0.75)-d_tr["Second_Floor_Area"].quantile(0.25)
iqr8


# In[163]:


iqr9=d_tr["Grade_Living_Area"].quantile(0.75)-d_tr["Grade_Living_Area"].quantile(0.25)
iqr9


# In[164]:


iqr14=d_tr["Sale_Price"].quantile(0.75)-d_tr["Sale_Price"].quantile(0.25)
iqr14


# In[165]:


up1=d_tr["Lot_Extent"].quantile(0.75)+(3*iqr1)
lw1=d_tr["Lot_Extent"].quantile(0.25)-(3*iqr1)
print(up1,lw1)


# In[166]:


up2=d_tr["Lot_Size"].quantile(0.75)+(3*iqr2)
lw2=d_tr["Lot_Size"].quantile(0.25)-(3*iqr2)
print(up2,lw2)


# In[167]:


up3=d_tr["Brick_Veneer_Area"].quantile(0.75)+(3*iqr3)
lw3=d_tr["Brick_Veneer_Area"].quantile(0.25)-(3*iqr3)
print(up3,lw3)


# In[168]:


up4=d_tr["BsmtFinSF1"].quantile(0.75)+(3*iqr4)
lw4=d_tr["BsmtFinSF1"].quantile(0.25)-(3*iqr4)
print(up4,lw4)


# In[169]:


up5=d_tr["BsmtUnfSF"].quantile(0.75)+(2.5*iqr5)
lw5=d_tr["BsmtUnfSF"].quantile(0.25)-(2.5*iqr5)
print(up5,lw5)


# In[170]:


up6=d_tr["First_Floor_Area"].quantile(0.75)+(3*iqr6)
lw6=d_tr["First_Floor_Area"].quantile(0.25)-(3*iqr6)
print(up6,lw6)


# In[171]:


up7=d_tr["First_Floor_Area"].quantile(0.75)+(3*iqr7)
lw7=d_tr["First_Floor_Area"].quantile(0.25)-(3*iqr7)
print(up7,lw7)


# In[172]:


up8=d_tr["Second_Floor_Area"].quantile(0.75)+(1.5*iqr8)
lw8=d_tr["Second_Floor_Area"].quantile(0.25)-(1.5*iqr8)
print(up8,lw8)


# In[173]:


up9=d_tr["Grade_Living_Area"].quantile(0.75)+(3*iqr9)
lw9=d_tr["Grade_Living_Area"].quantile(0.25)-(3*iqr9)
print(up9,lw9)


# In[174]:


up10=d_tr["Garage_Area"].mean()+3*d_tr["Garage_Area"].std()
lw10=d_tr["Garage_Area"].mean()-3*d_tr["Garage_Area"].std()
print(up10,lw10)


# In[175]:


up11=d_tr["W_Deck_Area"].mean()+3*d_tr["W_Deck_Area"].std()
lw11=d_tr["W_Deck_Area"].mean()-3*d_tr["W_Deck_Area"].std()
print(up11,lw11)


# In[176]:


up12=d_tr["Open_Lobby_Area"].mean()+3*d_tr["Open_Lobby_Area"].std()
lw12=d_tr["Open_Lobby_Area"].mean()-3*d_tr["Open_Lobby_Area"].std()
print(up12,lw12)


# In[177]:


up13=d_tr["Enclosed_Lobby_Area"].mean()+3*d_tr["Enclosed_Lobby_Area"].std()
lw13=d_tr["Enclosed_Lobby_Area"].mean()-3*d_tr["Enclosed_Lobby_Area"].std()
print(up13,lw13)


# In[178]:


up14=d_tr["Sale_Price"].quantile(0.75)+(3*iqr14)
lw14=d_tr["Sale_Price"].quantile(0.25)-(3*iqr14)
print(up14,lw14)


# In[179]:


d_tr.describe()


# In[180]:


#d_tr.Lot_Extent.describe()
#d_tr.Lot_Size.describe()
#d_tr.Brick_Veneer_Area.describe()
#d_tr.BsmtFinSF1.describe()
#d_tr.BsmtUnfSF.describe()
#d_tr.Total_Basement_Area.describe()
#d_tr.First_Floor_Area.describe()
#d_tr.Second_Floor_Area.describe()
#d_tr.Grade_Living_Area.describe()
#d_tr.Garage_Area.describe()
#d_tr.W_Deck_Area.describe()
#d_tr.Open_Lobby_Area.describe()
#d_tr.Enclosed_Lobby_Area.describe()
#d_tr.Sale_Price.describe()


# In[181]:


d_tr.loc[d_tr["Lot_Extent"]>136.0,"Lot_Extent"]=136.0
d_tr.loc[d_tr["Lot_Size"]>23765.0,"Lot_Size"]=23765.0
d_tr.loc[d_tr["Brick_Veneer_Area"]>658.0,"Brick_Veneer_Area"]=658.0
d_tr.loc[d_tr["BsmtFinSF1"]>2848.0,"BsmtFinSF1"]=2848.0
d_tr.loc[d_tr["BsmtUnfSF"]>2269.25,"BsmtUnfSF"]=2269.25
d_tr.loc[d_tr["Total_Basement_Area"]>2807.5,"Total_Basement_Area"]=2807.5
d_tr.loc[d_tr["First_Floor_Area"]>2920.0,"First_Floor_Area"]=2920.0
d_tr.loc[d_tr["Second_Floor_Area"]>1820.0,"Second_Floor_Area"]=1820.0
d_tr.loc[d_tr["Grade_Living_Area"]>3723.0,"Grade_Living_Area"]=3723.0
d_tr.loc[d_tr["Garage_Area"]>1102.9411493792454,"Garage_Area"]=1102.9411493792454
d_tr.loc[d_tr["W_Deck_Area"]>466.5623805627213,"W_Deck_Area"]=466.5623805627213
d_tr.loc[d_tr["Open_Lobby_Area"]>250.2146600613432,"Open_Lobby_Area"]=250.2146600613432
d_tr.loc[d_tr["Enclosed_Lobby_Area"]>208.64074693070367,"Enclosed_Lobby_Area"]=208.64074693070367
d_tr.loc[d_tr["Sale_Price"]>466150.0,"Sale_Price"]=466150.0


# In[182]:


d_tr.columns


# In[183]:


d_tr.describe()


# In[184]:


#d_tr.Lot_Extent.describe()
#d_tr.Lot_Size.describe()
#d_tr.Brick_Veneer_Area.describe()
#d_tr.BsmtFinSF1.describe()
#d_tr.BsmtUnfSF.describe()
#d_tr.Total_Basement_Area.describe()
#d_tr.First_Floor_Area.describe()
#d_tr.Second_Floor_Area.describe()
#d_tr.Grade_Living_Area.describe()
#d_tr.Garage_Area.describe()
#d_tr.W_Deck_Area.describe()
#d_tr.Open_Lobby_Area.describe()
#d_tr.Enclosed_Lobby_Area.describe()
#d_tr.Sale_Price.describe()


# In[185]:


col=pd.DataFrame(d_tr,columns={"Lot_Extent","Lot_Size","Brick_Veneer_Area","BsmtFinSF1","BsmtUnfSF","Total_Basement_Area","BsmtUnfSF","Second_Floor_Area","Grade_Living_Area","Garage_Area","W_Deck_Area","Open_Lobby_Area","Enclosed_Lobby_Area","Sale_Price"})
col.shape


# In[186]:


fig_without_outler=col.hist(figsize=(25,30),bins=50,color="brown",edgecolor="black",xlabelsize=8,ylabelsize=8)


# In[187]:


after=d_tr.boxplot(["Lot_Extent","Lot_Size","Brick_Veneer_Area","BsmtFinSF1","BsmtUnfSF","Total_Basement_Area","BsmtUnfSF","Second_Floor_Area","Grade_Living_Area","Garage_Area","W_Deck_Area","Open_Lobby_Area","Enclosed_Lobby_Area","Sale_Price"],figsize=(45,35))


# In[188]:


#d_tr.boxplot("Lot_Extent")
#d_tr.boxplot("Lot_Size")
#d_tr.boxplot("Brick_Veneer_Area")
#d_tr.boxplot("BsmtFinSF1")
#d_tr.boxplot("BsmtUnfSF")
#d_tr.boxplot("Total_Basement_Area")
#d_tr.boxplot("BsmtUnfSF")
#d_tr.boxplot("Second_Floor_Area")
#d_tr.boxplot("Grade_Living_Area")
#d_tr.boxplot("Garage_Area")
#d_tr.boxplot("W_Deck_Area")
#d_tr.boxplot("Open_Lobby_Area")
#d_tr.boxplot("Enclosed_Lobby_Area")
#d_tr.boxplot("Sale_Price")


# In[189]:


#Skewness


# In[190]:


from scipy.stats import skew


# In[191]:


for coloumn in d_tr:
    print(coloumn)
    print(skew(d_tr[coloumn]))
    
    plt.figure()
    sns.distplot(d_tr[coloumn])
    plt.show()


# In[192]:


d_tr["Brick_Veneer_Area"]=np.sqrt(d_tr["Brick_Veneer_Area"])
d_tr["Sale_Price"]=np.sqrt(d_tr["Sale_Price"])


# In[193]:


#Building_Class,Utility_Type,Property_Slope,Condition1,Condition2,House_Type,Roof_Design,Roof_Quality,Brick_Veneer_Area,BsmtFinSF2,Heating_Type,LowQualFinSF
#,Underground_Half_Bathroom,Kitchen_Above_Grade,Three_Season_Lobby_Area,Pool_Area,Miscellaneous_Value,Sale_Price


# In[194]:


#Building_Class,Utility_Type,Property_Slope,Condition1,Condition2,House_Type,Roof_Design,Roof_Quality,Brick_Veneer_Area,BsmtFinSF2,Heating_Type,LowQualFinSF
#,Underground_Half_Bathroom,Kitchen_Above_Grade,Three_Season_Lobby_Area,Pool_Area,Miscellaneous_Value


# In[195]:


d_tr["Building_Class"]=np.sqrt(d_tr.Building_Class)
d_tr["Utility_Type"]=np.sqrt(d_tr.Utility_Type)
d_tr["Property_Slope"]=np.sqrt(d_tr.Property_Slope)
d_tr["Condition1"]=np.sqrt(d_tr["Condition1"])
d_tr["Condition2"]=np.sqrt(d_tr["Condition2"])
d_tr["House_Type"]=np.sqrt(d_tr["House_Type"])
d_tr["Roof_Design"]=np.sqrt(d_tr["Roof_Design"])
d_tr["Roof_Quality"]=np.sqrt(d_tr["Roof_Quality"])
d_tr["Brick_Veneer_Area"]=np.sqrt(d_tr["Brick_Veneer_Area"])
d_tr["BsmtFinSF2"]=np.sqrt(d_tr["BsmtFinSF2"])
d_tr["Underground_Half_Bathroom"]=np.sqrt(d_tr["Underground_Half_Bathroom"])
d_tr["Kitchen_Above_Grade"]=np.sqrt(d_tr["Kitchen_Above_Grade"])
d_tr["Three_Season_Lobby_Area"]=np.sqrt(d_tr["Three_Season_Lobby_Area"])
d_tr["Pool_Area"]=np.sqrt(d_tr["Pool_Area"])
d_tr["Miscellaneous_Value"]=np.sqrt(d_tr["Miscellaneous_Value"])


# # Lasso

# In[196]:


from sklearn.linear_model import Lasso
l=Lasso()


# In[197]:


l.fit(data_xtrain,data_ytrain)


# In[198]:


pred=l.predict(data_xtest)
pred


# In[199]:


l.coef_


# In[213]:


#data_frame=pd.DataFrame({"Importance":list(l.coef_),"columns":list(data_x)})
#data_frame


# In[201]:


xnew=d_tr[["Building_Class","Zoning_Class","Lot_Extent","Lot_Size","Property_Shape","Lot_Configuration","Neighborhood","House_Type","Overall_Material","House_Condition","Construction_Year","Remodel_Year","Exterior1st","Brick_Veneer_Type","Brick_Veneer_Area","Exterior_Material","Basement_Height","Basement_Condition","Exposure_Level","BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","Total_Basement_Area","Heating_Quality","First_Floor_Area","Second_Floor_Area","LowQualFinSF","Bedroom_Above_Grade","Kitchen_Quality","Functional_Rate","Fireplaces","Fireplace_Quality","Garage_Built_Year","Garage_Size","Garage_Area","W_Deck_Area","Open_Lobby_Area","Enclosed_Lobby_Area","Three_Season_Lobby_Area","Screen_Lobby_Area","Pool_Area","Miscellaneous_Value","Month_Sold","Year_Sold","Year_Sold","Sale_Condition"]]
xnew.shape


# In[202]:


ynew=d_tr["Sale_Price"]
ynew


# In[203]:


df_xtrain,df_xtest,df_ytrain,df_ytest=train_test_split(xnew,ynew,test_size=0.2,random_state=101)
df_xtrain.shape,df_xtest.shape,df_ytrain.shape,df_ytest.shape


# In[204]:


from sklearn import linear_model
ln=linear_model.LinearRegression()


# In[205]:


ln.fit(df_xtrain,df_ytrain)


# In[206]:


pr=ln.predict(df_xtest)
pr


# In[207]:


ln.coef_


# In[208]:


ln.intercept_


# In[209]:


rsquare2=ln.score(df_xtrain,df_ytrain)
rsquare2


# In[210]:


adjR2=1-(((1-rsquare)*(1167-1))/(1167-46-1))
adjR2


# In[211]:


from sklearn import metrics

MSE=metrics.mean_squared_error(df_ytest,pr)
MSE


# In[212]:


from scipy import stats

slope, intercept, r, p, std_err=stats.linregress(df_ytest,pr)

def myfunc(df_ytest):
    return slope * df_ytest + intercept

mymodel = list(map(myfunc, df_ytest))

plt.scatter(df_ytest, pr)
plt.plot(df_ytest, mymodel)
plt.show()


# In[ ]:




