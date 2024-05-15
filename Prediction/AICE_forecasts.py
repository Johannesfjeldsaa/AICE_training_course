#!/usr/bin/env python
# coding: utf-8

# In[10]:


import os
import time
import h5py
import datetime
import numpy as np
function_path = "/lustre/storeB/users/cyrilp/COSI/Scripts/Operational/Operational_chain/"
from Attention_Res_UNet_prod import *
#
tf.keras.mixed_precision.set_global_policy("mixed_float16")
#print("GPUs available: ", tf.config.list_physical_devices('GPU'))
#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#print(physical_devices)
#tf.config.experimental.set_memory_growth(physical_devices[0], True)


# In[11]:


#
model_name = "SIC_Attention_Res_UNet" 
#
paths = {}
paths["predictors"] = os.getcwd() + "/forecast_files/"
paths["static"] = "/lustre/storeB/project/fou/hi/oper/aice/static/"
paths["output"] = os.getcwd() + "/forecast_files/"
#
filename_standardization = paths["static"] + "Stats_standardization_20130103_20201231_weekly.h5"


# # Model parameters

# In[12]:


model_params = {"list_predictors": ["LSM", "ECMWF_T2M_cum", "ECMWF_wind_x_cum", "ECMWF_wind_y_cum", "SICobs_AMSR2_SIC"],
                "patch_dim": (480, 544),
                "batch_size": 1,
                "n_filters": [32, 64, 128, 256, 512, 1024],
                "activation": "relu",
                "kernel_initializer": "he_normal",
                "batch_norm": True,
                "pooling_type": "Average",
                "dropout": 0,
                }


# # Load predictors

# In[13]:


def load_predictors(paths):
    today_date = (datetime.datetime.now()).strftime("%Y%m%dT000000Z")
    filename = paths["predictors"] + "AICE_predictors_" + today_date + ".h5"
    with h5py.File(filename, "r") as file:
        Dataset = {}
        Dataset["x"] = file["x"][:]
        Dataset["y"] = file["y"][:]
        Dataset["lat"] = file["lat"][:,:]
        Dataset["lon"] = file["lon"][:,:]
        Dataset["LSM"] = file["LSM"][:,:]
        Dataset["SICobs_AMSR2"] = file["SICobs_AMSR2"][:,:]
        Dataset["ECMWF_T2M"] = file["ECMWF_T2M"][:,:,:]
        Dataset["ECMWF_wind_x"] = file["ECMWF_x_wind"][:,:,:]
        Dataset["ECMWF_wind_y"] = file["ECMWF_y_wind"][:,:,:]
    return(Dataset)


# # Standardization data

# In[14]:


def load_standardization_data(file_standardization, lead_time):
    standard = {}
    hf = h5py.File(file_standardization, "r")
    for var in hf:
        if "ECMWF" in var:
            standard[var] = np.array(hf[var])[lead_time]
        else:
            standard[var] = hf[var][()]
    hf.close()
    return(standard)


# # Data generator

# In[15]:


class Data_generator(tf.keras.utils.Sequence):
    def __init__(self, list_predictors, lead_time, standard, Dataset, dim):
        self.list_predictors = list_predictors
        self.lead_time = lead_time
        self.standard = standard
        self.Dataset = Dataset
        self.dim = dim
        self.n_predictors = len(list_predictors)
    #
    def normalize(self, var, var_data):
        norm_data = (var_data - self.standard[var + "_min"]) / (self.standard[var + "_max"] - self.standard[var + "_min"])
        return(norm_data)
    #
    def data_generation(self): # Generates data containing batch_size samples
        # Initialization
        X = np.full((1, *self.dim, self.n_predictors), np.nan)
        #
        # Generate data
        for v, var in enumerate(self.list_predictors):
            if var == "LSM":
                var_data = self.Dataset["LSM"]
            elif var == "SICobs_AMSR2_SIC":
                var_data = self.Dataset["SICobs_AMSR2"]
            elif "ECMWF" in var:
                var_data = self.Dataset[var.replace("_cum", "")][self.lead_time,:,:]
            #
            X[0,:,:,v] = self.normalize(var, var_data)
        #
        return(X)


# # Function make_predictions

# In[16]:


class make_predictions:
    def __init__(self, Dataset, model_params, filename_standardization, paths):
        self.Dataset = Dataset
        self.model_params = model_params
        self.filename_standardization = filename_standardization
        self.paths = paths
    #
    def SIC_from_normalized_SIC(self, variable_name, field, standard):
        Predicted_SIC = field * (standard[variable_name + "_max"] - standard[variable_name + "_min"]) + standard[variable_name + "_min"]
        Predicted_SIC[Predicted_SIC > 100] = 100
        Predicted_SIC[Predicted_SIC < 0] = 0
        return(Predicted_SIC)
    #
    def predict(self):
        lead_times = np.linspace(0, 9, 10, dtype = int)
        SIC_pred = np.full((len(lead_times), self.model_params["patch_dim"][0], self.model_params["patch_dim"][1]), np.nan)
        #
        for leadtime in lead_times:
            file_model_weights = self.paths["static"] + "UNet_leadtime_" + str(leadtime) + "_days.h5"
            standard = load_standardization_data(self.filename_standardization, leadtime)
            unet_model = Att_Res_UNet(**self.model_params).make_unet_model()
            unet_model.load_weights(file_model_weights)
            #
            params_test = {"list_predictors": self.model_params["list_predictors"],
                           "lead_time": leadtime,
                           "standard": standard,
                           "Dataset": self.Dataset,
                           "dim": self.model_params["patch_dim"],
                          }
            #
            pred_sample = Data_generator(self.model_params["list_predictors"], leadtime, standard, self.Dataset, self.model_params["patch_dim"]).data_generation()
            tp0 = time.time()
            predictions_SIC = np.squeeze(unet_model.predict(pred_sample))
            tp1 = time.time()
            print("Prediction for lead time: " + str(leadtime) + ", " + str(tp1 - tp0))
            predictions_SIC = self.SIC_from_normalized_SIC("TARGET_AMSR2_SIC", predictions_SIC, standard)
            predictions_SIC[:,:][Dataset["LSM"] == 0] = np.nan
            predictions_SIC[predictions_SIC < 3] = 0
            SIC_pred[leadtime,:,:] = np.copy(predictions_SIC)
            del unet_model
        return(SIC_pred)


# # Function write_hdf5_forecasts

# In[17]:


def write_hdf5_forecasts(Dataset):
    today_date = (datetime.datetime.now()).strftime("%Y%m%dT000000Z")
    timestamps = []
    #
    for lt in range(0, 10):
        timestamps.append((datetime.datetime.strptime(today_date, "%Y%m%dT000000Z") + datetime.timedelta(days = lt)).timestamp())
    #
    path_output = paths["output"] 
    if os.path.exists(path_output) == False:
        os.system("mkdir -p " + path_output)    
    output_filename = path_output + "AICE_forecasts_" + today_date + ".h5"
    if os.path.isfile(output_filename):
        os.system("rm " + output_filename)
    #
    hf = h5py.File(output_filename, 'w')
    hf.create_dataset("time", data = timestamps)
    hf.create_dataset("x", data = Dataset["x"])
    hf.create_dataset("y", data = Dataset["y"])
    hf.create_dataset("lat", data = Dataset["lat"])
    hf.create_dataset("lon", data = Dataset["lon"])
    hf.create_dataset("SIC", data = Dataset["SIC_pred"])
    hf.close()


# In[20]:


t0 = time.time()
#
print("............................................................................................")
print("AICE forecasts")
Dataset = load_predictors(paths)
t1 = time.time()
print("Load dataset", t1 - t0)
Dataset["SIC_pred"] = make_predictions(Dataset, model_params, filename_standardization, paths).predict()
t2 = time.time()
print("Make predictions", t2 - t1)
write_hdf5_forecasts(Dataset)
t3 = time.time()
print("write hf5 file", t3 -t2)
#
tf = time.time()
print("AICE forecasts computing time", tf - t0)
print("............................................................................................")

