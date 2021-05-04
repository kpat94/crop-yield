#%%
import torch
print(torch.__version__)
import pandas as pd
print(pd.__version__)
from torch.utils.data import DataLoader,ConcatDataset
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

#%%
# Read fertilizer data in the form of nutrients
######### Not required
FertilizerNutrients = pd.read_csv("Inputs_FertilizersNutrient_E_All_Data_NOFLAG.csv", engine='python');
FertilizerNutrients.head

#%%
InputPesiticides = pd.read_csv("Inputs_Pesticides_Use_E_All_Data_NOFLAG.csv", engine='python')
InputPesiticides.head
# %%
# Read fertilizer data in the form of products
FertilizerProducts = pd.read_csv("Inputs_FertilizersProduct_E_All_Data_NOFLAG.csv", engine='python')
FertilizerProducts.head
# %%
# Read the crop yield data
ProductionYield = pd.read_csv("Production_Crops_E_All_Data_NOFLAG.csv",engine='python')
ProductionYield.head
# %%
print(FertilizerNutrients.columns);
#%%
print(FertilizerNutrients['Area'])
#%%
# Filter out the data for Australia
is_Country_Australia = FertilizerNutrients['Area']=="Australia"
Australia_Fertilizers_Nutrients = FertilizerNutrients[is_Country_Australia]
print(Australia_Fertilizers_Nutrients.head)

#%%
# Get the fertilizer usage in terms of agriculture
is_Element_Agricultural_Use = Australia_Fertilizers_Nutrients['Element']=="Agricultural Use"
Australia_Fertilizers_Nutrients_Agricultural = Australia_Fertilizers_Nutrients[is_Element_Agricultural_Use]
print(Australia_Fertilizers_Nutrients_Agricultural)
# %%
print("Fertilizer products",FertilizerProducts.columns)
print("\n")
print("Input pesticides", InputPesiticides.columns)
#%%
# Filter out the data for Australia
is_Country_Australia_Products = FertilizerProducts['Area']=="Australia"
Australia_Fertilizers_Products = FertilizerProducts[is_Country_Australia_Products]

is_Country_Australia_Pesticides = InputPesiticides['Area']=="Australia"
Australia_Pesticides = InputPesiticides[is_Country_Australia_Pesticides]

# Get the fertilizer usage for agricultural use
is_Agricultural = Australia_Fertilizers_Products['Element']=="Agricultural Use"
Australia_Fertilizers_Products_Agricultural = Australia_Fertilizers_Products[is_Agricultural]
print("Fertilizers \n")
print(Australia_Fertilizers_Products_Agricultural.columns)
print(Australia_Fertilizers_Products_Agricultural['Item'])
print("\n")
print("Pesticides \n")
print(Australia_Pesticides.columns)
print(Australia_Pesticides['Item'])

#%%
# 7 years for training - 2007 to 2013
# 3 years fro validation - 2014 to 2016
# 2017 for testing
X_seven_years_fertlizers = Australia_Fertilizers_Products_Agricultural.drop(['Area Code', 'Area', 'Item Code', 'Element Code','Element',
       'Unit','Y2002', 'Y2003', 'Y2004', 'Y2005', 'Y2006', 'Y2014', 'Y2015', 'Y2016', 'Y2017'], axis=1)
print(X_seven_years_fertlizers)
print(X_seven_years_fertlizers.columns)

X_seven_years_pesticides = Australia_Pesticides.drop(['Area Code', 'Area', 'Item Code', 'Element Code', 'Element',
       'Unit', 'Y1990', 'Y1991', 'Y1992', 'Y1993', 'Y1994', 'Y1995', 'Y1996', 'Y1997', 'Y1998', 'Y1999', 'Y2000', 'Y2001', 'Y2002', 'Y2003', 'Y2004', 'Y2005', 'Y2006', 'Y2014', 'Y2015', 'Y2016', 'Y2017'], axis=1)
print(X_seven_years_pesticides)
print(X_seven_years_pesticides.columns)

#%%
# Drop Ammonia anyhydrous and Ammonia nitrate from 2007 to 2013. It has not been used in these years.
X_seven_years_fertlizers = X_seven_years_fertlizers.drop([602, 607], axis=0)
print(X_seven_years_fertlizers)
print(X_seven_years_fertlizers.columns)

#%%
# 2009 to 2013
interpolate_2009_to_2013 = X_seven_years_fertlizers.interpolate()
interpolate_2009_to_2013

#%%
#  Interpolate missing fertlizers data
#  Replace na with mean values of 2004,2005,2009,2010
interpolate_2004_2005 = Australia_Fertilizers_Products_Agricultural[['Y2004','Y2005']].copy().interpolate()
index_names = interpolate_2004_2005.index
index_names
interpolate_2004_2005=interpolate_2004_2005.drop([602,607], axis=0)
interpolate_2004_2005['mean'] = interpolate_2004_2005.mean(axis=1)
print("Mean of 2004 and 2005 ==== \n")

# Use this for 2007
print(interpolate_2004_2005)

interpolate_2009_2010 = Australia_Fertilizers_Products_Agricultural[['Y2009','Y2010']].copy().interpolate()
index_names = interpolate_2009_2010.index
index_names

interpolate_2009_2010['mean'] = interpolate_2009_2010.mean(axis=1)
print("Mean of 2009 and 2010 ==== \n")
# Use this for 2008
interpolate_2009_2010=interpolate_2009_2010.drop([602,607], axis=0)
print(interpolate_2009_2010)

# 2007
i=0
for i in range(1,16):
       X_seven_years_fertlizers.iloc[0:i,1:2] = interpolate_2004_2005['mean'].iloc[0:i]

# 2008
j=0
for j in range(1,16):
       X_seven_years_fertlizers.iloc[0:j,2:3] = interpolate_2009_2010['mean'].iloc[0:j]

X_seven_years_fertlizers


# %%
print(ProductionYield.columns)
#%%
ProductionYield.head
#%%
Australia = ProductionYield['Area']=="Australia"
Production = ProductionYield['Element']=="Production"
Australia_yield = ProductionYield[Australia]
Australia_yield_production = Australia_yield[Production]
print(ProductionYield[Australia])
#%%
# Get the yield data for the year 2017 only
Australia_yield_2017 = Australia_yield_production.drop(['Area Code', 'Area', 'Item Code', 'Element Code','Element',
       'Unit','Y1961', 'Y1962', 'Y1963', 'Y1964', 'Y1965', 'Y1966', 'Y1967', 'Y1968', 'Y1969', 'Y1970', 'Y1971', 'Y1972', 'Y1973', 
       'Y1974', 'Y1975', 'Y1976', 'Y1977', 'Y1978', 'Y1979', 'Y1980', 'Y1981', 'Y1982', 'Y1983', 'Y1984', 'Y1985', 'Y1986', 'Y1987',
        'Y1988', 'Y1989', 'Y1990', 'Y1991', 'Y1992', 'Y1993', 'Y1994', 'Y1995', 'Y1996', 'Y1997', 'Y1998', 'Y1999', 'Y2000', 'Y2001', 
        'Y2002', 'Y2003', 'Y2004', 'Y2005', 'Y2006', 'Y2007', 'Y2008', 'Y2009', 'Y2010', 'Y2011', 'Y2012', 'Y2013', 'Y2014', 'Y2015', 
        'Y2016', 'Y2018'], axis=1)
print(Australia_yield_2017)
print(Australia_yield_2017.columns)
# %%
# Get the apple yield for 2017 Australia
Apples = Australia_yield_2017['Item']=="Apples"
Australia_yield_2017_apples=Australia_yield_2017[Apples]
print(Australia_yield_2017_apples)
# %%
# Use 2016 data for training and 2017 data for testing
X = Australia_Fertilizers_Products_Agricultural.drop(['Area Code', 'Area', 'Item Code', 'Item', 'Element Code', 'Element',
       'Unit','Y2002', 'Y2003', 'Y2004', 'Y2005', 'Y2006', 'Y2007', 'Y2008',
       'Y2009', 'Y2010', 'Y2011', 'Y2012', 'Y2013', 'Y2015','Y2014'],axis=1)
# 2017 for testing
X_test=X['Y2017']
print(X_test.to_frame().columns)
print(X_test)
#%%
# Remove 2017 data from the train set
X_train=X.drop(['Y2017'],axis=1)
print("Input training data \n",X_train)
print("\n")
print("Input testing data \n",X_test)
#%%

Y = Australia_yield_production.drop(['Area Code', 'Area', 'Item Code', 'Item', 'Element Code', 'Element',
       'Unit', 'Y1961', 'Y1962', 'Y1963', 'Y1964', 'Y1965', 'Y1966', 'Y1967',
       'Y1968', 'Y1969', 'Y1970', 'Y1971', 'Y1972', 'Y1973', 'Y1974', 'Y1975',
       'Y1976', 'Y1977', 'Y1978', 'Y1979', 'Y1980', 'Y1981', 'Y1982', 'Y1983',
       'Y1984', 'Y1985', 'Y1986', 'Y1987', 'Y1988', 'Y1989', 'Y1990', 'Y1991',
       'Y1992', 'Y1993', 'Y1994', 'Y1995', 'Y1996', 'Y1997', 'Y1998', 'Y1999',
       'Y2000', 'Y2001', 'Y2002', 'Y2003', 'Y2004', 'Y2005', 'Y2006', 'Y2007',
       'Y2008', 'Y2009', 'Y2010', 'Y2011', 'Y2012', 'Y2013', 'Y2014', 'Y2017', 'Y2018'], axis=1)
Y_train = Y[Apples]
# 2017 for testing
Y_test = Australia_yield_2017_apples['Y2017']
print("Output training \n",Y_train['Y2016'])
print("\n")
print("Output testing \n",Y_test)
# %%
X_train_dl = DataLoader(X_train[:1])
print(len(X_train_dl.dataset))
Y_train_dl = DataLoader(Y_train)
print(len(Y_train_dl.dataset))
# %%
# TODO: Refactor
x = torch.Tensor(X_train.dropna().to_numpy())
print("Apple yield for 2016  \n", Y_train['Y2016'])
print("\n")
y = torch.mean(torch.Tensor(Y_train['Y2016'].to_numpy()))
print("X --> \n",x)
print("\n")
print(x.shape)
print("Y --> \n",y)
print("\n")
print(y.shape)

# %%
class NeuralNet(nn.Module):
       def __init__(self, D_in, H1, H2, H3, D_out):
              super(NeuralNet, self).__init__()

              self.linear1 = nn.Linear(D_in, H1)
              self.linear2 = nn.Linear(H1, H2)
              self.linear3 = nn.Linear(H2, H3)
              self.linear4 = nn.Linear(H3, D_out)

       def forward(self, x):
              y_pred = self.linear1(x).clamp(min=0)
              y_pred = self.linear2(y_pred).clamp(min=0)
              y_pred = self.linear3(y_pred).clamp(min=0)
              y_pred = self.linear4(y_pred)
              return y_pred

H1, H2, H3 = 500, 1000, 200
D_in, D_out = 1, 1

#%%
# Normalization of inputs
x_data = (x - x.mean())/(x.max() - x.min())
print(x_data)
#%%
# Rescale the output to match the range of the input
y_data = (y/1000000)
print(y_data)
#%%

loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 0.00001

model1 = NeuralNet(D_in, H1, H2, H3, D_out)

optimizer = torch.optim.SGD(model1.parameters(), lr=learning_rate)

losses1 = []
#%%
for t in range(50):
       y_pred = model1(x)

       loss = loss_fn(y_pred, y)
       print(t, loss.item())
       losses1.append(loss.item())

       optimizer.zero_grad()
       loss.backward()
       optimizer.step()       
#%%
# BEST 
model2 = NeuralNet(D_in, H1, H2, H3, D_out)

loss_fn = torch.nn.MSELoss(reduction='sum')

optimizer = torch.optim.SGD(model2.parameters(), lr=1e-4)

losses2 = []

for t in range(500):
       y_pred=model2(x_data)
       print(torch.mean(y_pred))

       loss = loss_fn(y_pred, y_data)
       print(t, loss.item())
       losses2.append(loss.item())

       optimizer.zero_grad()
       loss.backward()
       optimizer.step()

print("Mean loss  --> \n", torch.mean(torch.Tensor(losses2)))


#%%
model3 = NeuralNet(D_in, H1, H2, H3, D_out)
optimizer = torch.optim.SGD(model3.parameters(), lr=1e-4*2)
losses3 = []

for t in range(500):
       y_pred = model3(x_data)

       loss=loss_fn(y_pred, y_data)
       print(t, loss.item())
       losses3.append(loss.item())

       optimizer.zero_grad()
       loss.backward()
       optimizer.step()

print("Mean loss  --> \n", torch.mean(torch.Tensor(losses3)))
#%%
model4 = NeuralNet(D_in, H1, H2, H3, D_out)
optimizer = torch.optim.Adam(model4.parameters(), lr=1e-4*2)
losses4 = []

for t in range(500):
       y_pred = model4(x_data)

       loss = loss_fn(y_pred, y_data)
       print(t, loss.item())
       losses4.append(loss.item())

       optimizer.zero_grad()
       loss.backward()
       optimizer.step()

print("Mean loss  --> \n", torch.mean(torch.Tensor(losses4)))

# %%
# %%
# Save the model that has the best accuracy
torch.save(model4.state_dict(), './fcn_trained.dat')
# Load the trained parameters
model2.load_state_dict(torch.load('./fcn_trained.dat'))
# Set the model to evaluation mode
model2.eval()
#%%
def predict(input_values):
       with torch.no_grad():
              output = model2(input_values)

       return output

# %%

x_eval = torch.Tensor(X_test.dropna().to_numpy())
# Apple yield of Australia for 2017
print("Apple yield for 2017 \n ", Australia_yield_2017_apples["Y2017"])
y_eval = torch.mean(torch.Tensor(Australia_yield_2017_apples["Y2017"].to_numpy()))
y_eval = y_eval/1000000
#%%
# Normalize
x_eval_data = (x_eval-x_eval.mean())/(x_eval.max()-x_eval.min())
x_eval_data = torch.reshape(x_eval_data,(12,1))
# %%
y_eval_pred = predict(x_eval_data)
# y_eval = torch.reshape(y_eval, (12,1))
print("ACTUAL    ",y_eval)
print("PREDICTED   ",torch.mean(y_eval_pred))
# %%
# RMSE
prediction_loss = np.sqrt(mean_squared_error(y_eval, torch.mean(y_eval_pred)))
print("Prediction loss --> ", prediction_loss)

# %%
