#!/usr/bin/env python
# coding: utf-8

# %

# In[1]:


import torch
print(torch.__version__)
import pandas as pd
print(pd.__version__)
from torch.utils.data import DataLoader,ConcatDataset
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


# %

# ## Read pesticides and fertilizer products data

# In[3]:

#%%
def load_data():
       pesticides_frame = pd.read_csv("Inputs_Pesticides_Use_E_All_Data_NOFLAG.csv", engine='python')
       pesticides_frame.head
  
       # Read fertilizer data in the form of products
       fertilizers_frame = pd.read_csv("Inputs_FertilizersProduct_E_All_Data_NOFLAG.csv", engine='python')
       fertilizers_frame.head
       
       # Read the crop yield data
       yield_frame = pd.read_csv("Production_Crops_E_All_Data_NOFLAG.csv",engine='python')
       yield_frame.head

       # %<br>
       # Get the fertilizer usage in terms of agriculture

       # In[4]:     
       print("Fertilizer products",fertilizers_frame.columns)
       print("\n")
       print("Input pesticides", pesticides_frame.columns)
       
       # Filter out the data for Australia
       is_Country_Australia_Products = fertilizers_frame['Area']=="Australia"
       Australia_Fertilizers_Products = fertilizers_frame[is_Country_Australia_Products]
       # In[5]:
       is_Country_Australia_Pesticides = pesticides_frame['Area']=="Australia"
       Australia_Pesticides = pesticides_frame[is_Country_Australia_Pesticides]

       # Get the fertilizer usage for agricultural use
       # In[6]:
       is_Agricultural = Australia_Fertilizers_Products['Element']=="Agricultural Use"
       Australia_Fertilizers_Products_Agricultural = Australia_Fertilizers_Products[is_Agricultural]
       print("Fertilizers \n")
       print(Australia_Fertilizers_Products_Agricultural.columns)
       print(Australia_Fertilizers_Products_Agricultural['Item'])
       print("\n")
       print("Pesticides \n")
       print(Australia_Pesticides.columns)
       print(Australia_Pesticides['Item'])
       
       # %<br>
       # 7 years for training - 2007 to 2013<br>
       # 3 years fro validation - 2014 to 2016<br>
       # 2017 for testing
       # In[7]:
       X_ten_years_fertlizers = Australia_Fertilizers_Products_Agricultural.drop(['Area Code', 'Area', 'Item Code', 'Element Code','Element',
              'Unit','Y2002', 'Y2003', 'Y2004', 'Y2005', 'Y2006', 'Y2017'], axis=1)
       print(X_ten_years_fertlizers)
       print(X_ten_years_fertlizers.columns)
       fertilizer_2017 = Australia_Fertilizers_Products_Agricultural['Y2017']
       print("2017 \n",fertilizer_2017)

       # In[8]:
       X_ten_years_pesticides = Australia_Pesticides.drop(['Area Code', 'Area', 'Item Code', 'Element Code', 'Element',
              'Unit', 'Y1990', 'Y1991', 'Y1992', 'Y1993', 'Y1994', 'Y1995', 'Y1996', 'Y1997', 'Y1998', 'Y1999', 'Y2000', 'Y2001', 'Y2002', 'Y2003', 'Y2004', 'Y2005', 'Y2006', 'Y2017'], axis=1)
       print(X_ten_years_pesticides)
       print(X_ten_years_pesticides.columns)
       pesticides_2017 = Australia_Pesticides['Y2017']
       print("2017 \n",pesticides_2017)

       # %<br>
       # Drop Ammonia anyhydrous and Ammonia nitrate from 2007 to 2013. It has not been used in these years.
       # In[9]:
       X_ten_years_fertlizers = X_ten_years_fertlizers.drop([602, 607], axis=0)
       print(X_ten_years_fertlizers)
       print(X_ten_years_fertlizers.columns)
       fertilizer_2017 = fertilizer_2017.drop([602, 607], axis=0)
       print(fertilizer_2017)
       
       # %<br>
       # 2009 to 2013
       # ## Handle missing values from 2009 to 2013
       # In[10]:
       interpolate_2009_to_2016 = X_ten_years_fertlizers.interpolate()
       interpolate_2009_to_2016
       interpolate_2017 = fertilizer_2017.interpolate()
       print("interpolated 2017 fertilizers \n",interpolate_2017)
       # TODO: Don't just interpolate. Use the means of the previous 2  years
       # interpolate_pesticides_2017 = pesticides_2017.interpolate()
       # print("interpolated 2017 pesticides \n",interpolate_pesticides_2017)

       # %<br>
       #  Interpolate missing fertlizers data<br>
       #  Replace na with mean values of 2004,2005,2009,2010
       # ##  Replace 2007 nan values with the mean of 2004 and 2005

       # In[11]:

       interpolate_2004_2005 = Australia_Fertilizers_Products_Agricultural[['Y2004','Y2005']].copy().interpolate()
       index_names = interpolate_2004_2005.index
       index_names
       interpolate_2004_2005=interpolate_2004_2005.drop([602,607], axis=0)
       interpolate_2004_2005['mean'] = interpolate_2004_2005.mean(axis=1)
       print("Mean of 2004 and 2005 ==== \n")
       print(interpolate_2004_2005)
       pesticides_2015_2016 = Australia_Pesticides[['Y2015','Y2016']]
       pesticides_2015_2016['mean'] = pesticides_2015_2016.mean(axis=1)
       print("Mean of 2015 and 2016 === \n")
       print(pesticides_2015_2016)

       # Use this for 2007

       # In[ ]:
       # ##  Replace 2008 nan values with the mean of 2009 and 2010

       # In[12]:
       interpolate_2009_2010 = Australia_Fertilizers_Products_Agricultural[['Y2009','Y2010']].copy().interpolate()
       index_names = interpolate_2009_2010.index
       index_names

       # In[13]:
       interpolate_2009_2010['mean'] = interpolate_2009_2010.mean(axis=1)
       print("Mean of 2009 and 2010 ==== \n")
       # Use this for 2008
       interpolate_2009_2010=interpolate_2009_2010.drop([602,607], axis=0)
       print(interpolate_2009_2010)


       # ## Populate 2007 nan values with computed mean
       # In[14]:
       # Fertilizers
       i=0
       for i in range(1,16):
              interpolate_2009_to_2016.iloc[0:i,1:2] = interpolate_2004_2005['mean'].iloc[0:i]

       #Pesticides
       j=0
       for j in range(1,7):
              pesticides_2017.iloc[0:j,] = pesticides_2015_2016['mean'].iloc[0:j]

       # ## Populate 2008 nan values with computed mean
       # In[15]:
       j=0
       for j in range(1,16):
              interpolate_2009_to_2016.iloc[0:j,2:3] = interpolate_2009_2010['mean'].iloc[0:j]
                
       interpolate_2009_to_2016.iloc[15, 1] = 1437658.50
       print(interpolate_2009_to_2016.iloc[15, 1])
       interpolate_2009_to_2016.iloc[15, 2] = 171351.00

       # In[16]:
       X_ten_years_fertilizers_interpolated = interpolate_2009_to_2016
       X_ten_years_fertilizers_interpolated
       fertilizer_2017_interpolated = interpolate_2017
       print(fertilizer_2017_interpolated)

       # #### TODO: Combine pesticdes and fertilizers data
       # In[17]:
       print(X_ten_years_pesticides)
       print(X_ten_years_fertilizers_interpolated)

       pesticides_without_Item = X_ten_years_pesticides.copy().drop(['Item'], axis=1)
       pesticides_2017_wo_item = pesticides_2017
       fertilizers_without_Item = X_ten_years_fertilizers_interpolated.copy().drop(['Item'], axis=1)
       fertilizer_2017_wo_item = fertilizer_2017_interpolated
       # torch.tensor(X_ten_years_pesticides.values.astype(np.float64))
       print(X_ten_years_pesticides.values)
       print(X_ten_years_fertilizers_interpolated.values)
       print(pesticides_2017_wo_item.values)
       print(fertilizer_2017_wo_item.values)

       print(pesticides_without_Item)
       pesticides_tensor = torch.from_numpy(pesticides_without_Item.values)
       print("\t Pesticides tensor", pesticides_tensor)
       pesticides_2017_tensor = torch.from_numpy(pesticides_2017_wo_item.values)

       print(fertilizers_without_Item)
       fertilizers_tensor = torch.from_numpy(fertilizers_without_Item.values)
       fertilizer_2017_tensor = torch.from_numpy(fertilizer_2017_wo_item.values)
       print("\t Fertilizers tensor", fertilizers_tensor)

       # In[18]:
       print("Train : \n",pesticides_tensor.shape)
       print("Train: \n",fertilizers_tensor.shape)
       print("Test: \n",pesticides_2017_tensor.shape)
       print("Test: \n",fertilizer_2017_tensor.shape)

       # TODO: Normalize inputs before concatenation
       pesticides_and_fertilizers = torch.cat((pesticides_tensor, fertilizers_tensor), dim=0)
       print(pesticides_and_fertilizers.shape)
       print("\n")
       print("Training X data for 10 years: ",pesticides_and_fertilizers)
       print("\n")
       pesticides_and_fertilizers_2017 = torch.cat((pesticides_2017_tensor, fertilizer_2017_tensor), dim=0)
       print(pesticides_and_fertilizers_2017.shape)
       print("\n")
       print("Testing X data for 1 year: ",pesticides_and_fertilizers_2017)
       print("\n")

       # Training + Validation data : Input
       X_train_ten = pesticides_and_fertilizers;
       X_test = pesticides_and_fertilizers_2017

       # #### TODO: Create output vector

       # In[19]:
       print(yield_frame.columns)
       #%%
       yield_frame.head
       #%%
       Australia = yield_frame['Area']=="Australia"
       Production = yield_frame['Element']=="Production"
       Australia_yield = yield_frame[Australia]
       Australia_yield_production = Australia_yield[Production]
       print(yield_frame[Australia])
       #%%
       # Get the yield data for the year 2017 only
       Australia_yield_2017 = Australia_yield_production.drop(['Area Code', 'Area', 'Item Code', 'Element Code','Element',
              'Unit','Y1961', 'Y1962', 'Y1963', 'Y1964', 'Y1965', 'Y1966', 'Y1967', 'Y1968', 'Y1969', 'Y1970', 'Y1971', 'Y1972', 'Y1973', 
              'Y1974', 'Y1975', 'Y1976', 'Y1977', 'Y1978', 'Y1979', 'Y1980', 'Y1981', 'Y1982', 'Y1983', 'Y1984', 'Y1985', 'Y1986', 'Y1987',
              'Y1988', 'Y1989', 'Y1990', 'Y1991', 'Y1992', 'Y1993', 'Y1994', 'Y1995', 'Y1996', 'Y1997', 'Y1998', 'Y1999', 'Y2000', 'Y2001', 
              'Y2002', 'Y2003', 'Y2004', 'Y2005', 'Y2006', 'Y2007', 'Y2008', 'Y2009', 'Y2010', 'Y2011', 'Y2012', 'Y2013', 'Y2014', 'Y2015', 
              'Y2016', 'Y2018'], axis=1)
       print("\n")
       print("Test Y data: ",Australia_yield_2017)
       print("\n")
       print(Australia_yield_2017.columns)
       print(Australia_yield_2017['Item'].unique())
       print("Number of unique crops in 2017: ",len(Australia_yield_2017['Item'].unique()))

       Australia_yield_2007_to_2016 = Australia_yield_production.drop(['Area Code', 'Area', 'Item Code', 'Element Code','Element',
              'Unit','Y1961', 'Y1962', 'Y1963', 'Y1964', 'Y1965', 'Y1966', 'Y1967', 'Y1968', 'Y1969', 'Y1970', 'Y1971', 'Y1972', 'Y1973', 
              'Y1974', 'Y1975', 'Y1976', 'Y1977', 'Y1978', 'Y1979', 'Y1980', 'Y1981', 'Y1982', 'Y1983', 'Y1984', 'Y1985', 'Y1986', 'Y1987',
              'Y1988', 'Y1989', 'Y1990', 'Y1991', 'Y1992', 'Y1993', 'Y1994', 'Y1995', 'Y1996', 'Y1997', 'Y1998', 'Y1999', 'Y2000', 'Y2001', 
              'Y2002', 'Y2003', 'Y2004', 'Y2005', 'Y2006', 'Y2017','Y2018'], axis=1)
       print(Australia_yield_2007_to_2016.columns)
       print(Australia_yield_2007_to_2016['Item'].unique())
       print("\n")
       print("Train Y data: ",Australia_yield_2007_to_2016)
       print("\n")
       print("Number of unique crops from 2007 to 2016",len(Australia_yield_2007_to_2016['Item'].unique()))

       # In[20]:
       yield_without_Item = Australia_yield_2007_to_2016.copy().drop(['Item'],axis=1)
       yield_without_Item_2017 = Australia_yield_2017.copy().drop(['Item'],axis=1)

       print(yield_without_Item)
       yield_tensor = torch.from_numpy(yield_without_Item.values)
       print("Shape: ", yield_tensor.shape)
       print("\t Yield tensor", yield_tensor)
       print(yield_without_Item_2017)
       yield_tensor_2017 = torch.from_numpy(yield_without_Item_2017.values)
       print("Shape: ", yield_tensor_2017.shape)
       print("\t Yield tensor", yield_tensor_2017)

       # Training + Validation: Output
       Y_train_ten = yield_tensor
       # Test Y
       Y_test = yield_tensor_2017

       print("Train X: \n",X_train_ten.shape)
       print("Train Y: \n",Y_train_ten.shape)
       print("Test X: \n",X_test.shape)
       print("Test Y: \n",Y_test.shape)
       
       return [X_train_ten, Y_train_ten, X_test, Y_test]
#%%

training_input = load_data()[0]
training_ouput = load_data()[1]
test_input = load_data()[2]
test_output = load_data()[3]

x = training_input
y = training_ouput
#%%

# Transforms

# Training data
x_tensor = torch.Tensor(x.float())
x_data = (x - x.mean())/(x.max() - x.min())
test = torch.Tensor([[0,4,5]])
x_tensor_normalized = x.normal_()
y_data = (y.normal_())

transformed_x = torch.reshape(x_tensor_normalized,(1,220))
transformed_y = y_data
y = torch.reshape(y, (1,1010))

#%%
# Test data
transformed_test_x = torch.Tensor(test_input.float())
transformed_test_x = transformed_test_x.normal_()
transformed_test_y = torch.Tensor(test_output.float())
transformed_test_y = transformed_test_y.normal_()
#%%
#%%
# Define model
class NeuralNet(nn.Module):
       def __init__(self, D_in, H1, H2, H3, H4, H5, H6, H7, H8, H9, D_out):
              super(NeuralNet, self).__init__()
              self.linear1 = nn.Linear(D_in, H1)
              self.linear2 = nn.Linear(H1, H2)
              self.linear3 = nn.Linear(H2, H3)
              self.linear4 = nn.Linear(H3, H4)
              self.linear5 = nn.Linear(H4, H5)
              self.linear6 = nn.Linear(H5, H6)
              self.linear7 = nn.Linear(H6, H7)
              self.linear8 = nn.Linear(H7, H8)
              self.linear9 = nn.Linear(H8, H9)
              self.linear10 = nn.Linear(H9, D_out)
              self.relu = nn.ReLU()
       def forward(self, x):
              y_pred = self.linear1(x)
              y_pred = torch.tanh(self.linear2(y_pred))
              y_pred = self.linear3(y_pred)
              y_pred = torch.tanh(self.linear4(y_pred))
              y_pred = self.linear5(y_pred)
              y_pred = torch.tanh(self.linear6(y_pred))
              y_pred = self.linear7(y_pred)
              y_pred = torch.sigmoid(self.linear8(y_pred))
              y_pred = self.linear9(y_pred)
              y_pred = torch.sigmoid(self.linear10(y_pred))
              return y_pred

# In[ ]:


from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from random import randint
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

# In[73]:


def get_accuracy(model, predicted, actual, threshold_percentage):
    num_items = len(actual)
    X = predicted.view(num_items)
    Y = actual.view(num_items)
    num_correct = torch.sum(torch.abs(X-Y)<torch.abs(threshold_percentage*Y))
    accuracy = (num_correct.item()*100.0/num_items)
    return (num_correct, accuracy)
    

def random_split_training(trainset_x, trainset_y):
       # TODO:
       # Select 2 random numbers between 1 and 10
       # Use them to index into x and y.
       # That will give us the training and validation set
       # Rest of the indices are thus our training set
       index1 = randint(0,9)
       index2 = randint(0,9)
       indices=[]
       i=0
       for i in range(0,10):
              if(i not in [index1,index2]):
                 indices.append(i)    
       
       indices = torch.tensor(indices)
       val_indices = torch.tensor([index1, index2])
       t_subset_x = torch.index_select(trainset_x, 1, indices)
       t_subset_y = torch.index_select(trainset_y, 1, indices)
       v_subset_x = torch.index_select(trainset_x, 1,val_indices)
       v_subset_y = torch.index_select(trainset_y, 1, val_indices)
       return (t_subset_x, t_subset_y, v_subset_x, v_subset_y)
# 

# In[76]:
#%%     
# Function to train and validate data. Save the best model based on validation loss
# 
# params:
# config: hyperparameter search space
# 
def train_crop_yield(config):
       net = NeuralNet(176, config['H1'], config['H2'], config['H3'], config["H4"], config["H5"], config["H6"], config["H7"],config["H8"],config["H9"],8)
       criterion = torch.nn.MSELoss()
       optimizer = torch.optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)
       # train_set = torch.cat((transformed_x, transformed_y), dim=0)
       # test_set = torch.cat((test_input, test_output), dim=0)
       
       x_by_years = transformed_x.reshape(22,10)
       y_by_years = transformed_y.reshape(101,10)
       
       train_subset_x, train_subset_y, val_subset_x, val_subset_y = random_split_training(x_by_years, y_by_years)
       
       for epoch in range(500):
              running_loss = 0.0
              epoch_steps = 0
              # Zero the accumulated gradients
              optimizer.zero_grad()
              # forward + backward + optimize
              output = net(train_subset_x.float().reshape(1,176))
              loss = criterion(output, train_subset_y.float())
              loss.backward()
              optimizer.step()
              # print statistics
              running_loss += loss.item()
              epoch_steps += 1
              if epoch%5==4:
                     print("[%d] loss: %.3f"%(epoch+1, running_loss/epoch_steps))
                     running_loss=0.0
       
       # Validation loss
       net = NeuralNet(44, config['H1'], config['H2'], config['H3'], config["H4"], config["H5"], config["H6"], config["H7"],config["H8"],config["H9"],2)
       val_loss = 0.0
       val_steps = 0
       total = 0
       correct = 0
       with torch.no_grad():
              val_output = net(val_subset_x.float().reshape(1,44))
              total = val_subset_y.size(0)
              correct = torch.sum(torch.abs(val_output-val_subset_y)<torch.abs(0.10*val_subset_y))
              loss = criterion(val_output, val_subset_y.float())
              val_loss += loss.cpu().numpy()
              torch.save((net.state_dict(), optimizer.state_dict()), './trained_net.dat')
              tune.report(loss=(val_loss), accuracy=(correct/total))
       print("\n Finished Training \n")
              
#%%
# Evaluation:
def test_accuracy(net, device='cpu'):
       net.eval()
       with torch.no_grad():
              predicted = net(transformed_test_x.float().reshape(1,22))
              return get_accuracy(net, predicted.reshape(101,1), transformed_test_y.reshape(101,1), 0.50)
       
# Define hyperparameters
config = {
       "H1":tune.sample_from(lambda _: 2**np.random.randint(7,9)),
       "H2":tune.sample_from(lambda _: 2**np.random.randint(7,9)),
       "H3":tune.sample_from(lambda _: 2**np.random.randint(7,12)),
       "H4":tune.sample_from(lambda _: 2**np.random.randint(7,12)),
       "H5":tune.sample_from(lambda _: 2**np.random.randint(7,12)),
       "H6":tune.sample_from(lambda _: 2**np.random.randint(8,10)),
       "H7":tune.sample_from(lambda _: 2**np.random.randint(8,10)),
       "H8":tune.sample_from(lambda _: 2**np.random.randint(7,9)),
       "H9":tune.sample_from(lambda _: 2**np.random.randint(7,9)),
       "lr":tune.loguniform(1e-2, 1e-1)
}

# Scheduler. Randomly try out combination of hyperparameters
scheduler = ASHAScheduler(
       metric="loss",
       mode="min",
       max_t=10,
       grace_period=1,
       reduction_factor=2
)

reporter=CLIReporter(metric_columns=["loss","accuracy", "training_iteration"])

result = tune.run(
       partial(train_crop_yield),
       config=config,
       num_samples=10,
       scheduler=scheduler
)

best_trial = result.get_best_trial("loss", "min", "last")

print("Best trial config: {}".format(best_trial.config))
print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
print("Best trial fnal validation accuracy: {}".format(best_trial.last_result["accuracy"]))

best_trained_model = NeuralNet(22, best_trial.config["H1"], best_trial.config["H2"], best_trial.config["H3"],best_trial.config["H4"],best_trial.config["H5"],best_trial.config["H6"],best_trial.config["H7"],best_trial.config["H8"],best_trial.config["H9"],101)

device="cpu"
best_trained_model.to(device)

# best_checkpoint_dir = best_trial.checkpoint.value
# print(" \n",type(best_trial.evaluated_params))
# model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))
# best_trained_model.load_state_dict(model_state)

num_correct, test_acc = test_accuracy(best_trained_model, device)
print("Best trial test set accuracy: {}".format(test_acc))

#%%
# %%

# %%

# %%

# %%
