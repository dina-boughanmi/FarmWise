from django.http import JsonResponse
from sklearn.preprocessing import StandardScaler , LabelEncoder
import numpy as np
import pandas as pd
import torch  # Needed for tensor operations
import os
from django.conf import settings
import tensorflow as tf
import joblib

# Define your column names
sequential_cols = ['Crop (t-2)', 'Crop (t-1)', 'Crop (t)']
static_cat_cols = ['Soil_type', 'soil salinity', 'area', 'Disease', 'Pest']
numerical_cols = [
    'Etm', 'Water Requirements', 'days_of_water_usage',
    'Fertilizer N', 'Fertilizer K', 'Fertilizer P',
    'Root Depth', 'Residue', 'plant spacing', 'Total_Fertilizer_Index'
]


def adding_static_data(df,old_crop_list,curre_crop):
  matches = df[df['Crop'] == curre_crop]
  if not matches.empty:
    corres_data = matches.sample(n=1).iloc[0]
  else:
    return print("No matching rows found.")

  # Set sequential crop values
  corres_data['Crop (t-2)'] = old_crop_list[0]
  corres_data['Crop (t-1)'] = old_crop_list[1]
  corres_data['Crop (t)'] =old_crop_list[2]

  # Drop unused column
  if 'Next Crop' in corres_data:
      corres_data = corres_data.drop('Next Crop')

  # Convert to numeric
  for col in ["Etm", "Water Requirements", "days of water usage", "Fertilizer N",
                "Fertilizer K", "Fertilizer P", "Root Depth", "Residue"]:
    corres_data[col] = pd.to_numeric(corres_data[col], errors='coerce')

  # Rename to match model input
  corres_data.rename({'days of water usage': 'days_of_water_usage'}, inplace=True)

  # Add engineered feature
  corres_data['Total_Fertilizer_Index'] = (
        corres_data['Fertilizer N'] +
        corres_data['Fertilizer P'] +
        corres_data['Fertilizer K']
    )
  return corres_data.to_dict() #return dictionary



def encode_static(feat_dict,scaler,df_static_cat):
    df_in = pd.DataFrame([feat_dict])
    nums = pd.DataFrame(scaler.transform(df_in[numerical_cols]), columns=numerical_cols)
    cats = pd.get_dummies(df_in[static_cat_cols]).reindex(columns=df_static_cat.columns, fill_value=0)
    return pd.concat([nums, cats], axis=1).astype(np.float32).values


def encode_sequential(crop_sequence, seq_encoders):
    crop3_code = seq_encoders['Crop (t-2)'].transform([crop_sequence[0]])[0]
    crop2_code = seq_encoders['Crop (t-1)'].transform([crop_sequence[1]])[0]
    crop1_code = seq_encoders['Crop (t)'].transform([crop_sequence[2]])[0]
    crop_sequence_encoded = [crop3_code, crop2_code, crop1_code]
    return crop_sequence_encoded


def encode_gcn(static_input_encoding,gcn_model):
    x_np=static_input_encoding
    x_t=torch.tensor(x_np,dtype=torch.float32)
    #edge_i=torch.tensor([[0],[0]],dtype=torch.long)
    edge_i = torch.zeros((2, 1), dtype=torch.long) 
    gcn_model.eval()
    with torch.no_grad(): out=gcn_model(x_t,edge_i)
    return out.numpy()

def predict_next(seq,gcn_input,target_le,model,top_k=3):
    s_arr = np.array([seq])
    g_arr = gcn_input.reshape(1, -1)
    probs=model.predict([s_arr,g_arr])[0]
    idx=np.argsort(probs)[::-1][:top_k]
    return [target_le.classes_[i] for i in idx]
