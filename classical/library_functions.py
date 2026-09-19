# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 12:10:42 2023

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 11:30:22 2023

@author: Admin
"""
import os
import csv
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.ticker import FormatStrFormatter
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D, Conv2DTranspose, MaxPooling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
import tensorflow as tf
from keras.utils import to_categorical
from keras.datasets import mnist
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import math
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
# demonstrate data standardization with sklearn

def Max(array):
  length=len(array)
  max_tam=-10000
  for i in range(length):
    if(max_tam < max(array[i])):
      max_tam = max(array[i])
  return max_tam

# define a function for peak signal-to-noise ratio (PSNR)
def psnr(target, ref):
         
    # assume RGB image
    target_data = target.astype(float)
    ref_data = ref.astype(float)

    diff = ref_data - target_data
    diff = diff.flatten('C')

    rmse = math.sqrt(np.mean(diff ** 2.))
    return 20 * math.log10(Max(target)/ rmse)

#This is the function which helps me split the string
def Split_String(name):
  return os.path.basename(name)

# =============================================================================
def List_Name_Image(path):
  list_name=[]
  files_path=[os.path.abspath(x) for x in os.listdir(path)]
  #print(files_path)

  for i in files_path:
      list_name.append(Split_String(i))
  return list_name
# =============================================================================

def Load_Data(Path,list_name):
  X=[] #Dùng để lưu dữ liệu load lên từ thư mục
  for i in tqdm(list_name, desc="📊 Loading Data", unit="file", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'):
    filename = Path +'/' + i
    #print(filename)
    with open(filename) as f:
      reader = csv.reader(f)
      tam=[]
      line = [row for row in reader]
      for k in range(4,len(line)):
        tam1=[]
        for l in range(len(line[k])):
          tam1.append(float(line[k][l]))
        tam.append(tam1)

      draft = np.array(tam)

      rotated_image = np.rot90(draft,2)
  
      rotated_image = np.diff(rotated_image)

      if((('Step_R' in i)==True) or (('Step_T' in i)==True)):
        Draft_1 = np.zeros((32,31))
        for i in range(32):
          Draft_1[i]=rotated_image[i][::-1]
        rotated_image = Draft_1

      # Copy cột cuối cùng (cột 30) sang cột 31
      last_column = rotated_image[:, -1:]
      rotated_image = np.column_stack([rotated_image, last_column])
      
      # ============ TÍNH ĐẠO HÀM GRADIENT (∇H) ============
      # Tính đạo hàm riêng theo 2 chiều: ∂H/∂y (dọc), ∂H/∂x (ngang)
      grad_y = np.gradient(rotated_image, axis=0)      # ∂H/∂y
      grad_x = np.gradient(rotated_image, axis=1)      # ∂H/∂x
      # Độ lớn gradient: ||∇H|| = sqrt((∂H/∂x)² + (∂H/∂y)²)
      gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
      # ================================================
      
      #print(rotated_image)

      # Lưu cả field value + gradient magnitude
      field_with_gradient = np.dstack([rotated_image, gradient_magnitude])
      X.append(field_with_gradient)
      tam = np.array(X)
      print(len(X))
      f.close
      
  return X 

def Load_Data_Test(Path,list_name):
  X=[] #Dùng để lưu dữ liệu load lên từ thư mục
  for i in tqdm(list_name, desc="📊 Loading Test Data", unit="file", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'):
    filename = Path +'/' + i
    with open(filename) as f:
      reader = csv.reader(f)
      tam=[]
      line = [row for row in reader]

      for k in range(len(line)):
        tam1=[]
        for l in range(len(line[k])):

          tam1.append(float(line[k][l]))
        tam.append(tam1)

      draft = np.array(tam)

      rotated_image = np.rot90(draft,1)

      draft_1 = np.zeros((32,31))
      
      for o in range(31):
        draft_1[o]=rotated_image[o]

      # Copy cột cuối cùng (cột 30) sang cột 31
      last_column = draft_1[:, -1:]
      rotated_image = np.column_stack([draft_1, last_column])

      X.append(rotated_image)

      f.close

  return X 

def Load_Data_Train_Test(path_file):
  path = path_file#"/content/drive/MyDrive/Dipole Model/Train_Test_Data/Trainning_Data/Trainning"

  List_Name_Image_Ellipse = List_Name_Image(path)
  
  list_tam=np.array(List_Name_Image_Ellipse)

  y_name =  list_tam

  #print(y_name)
  X_data=Load_Data(path,list_tam)
  
  X_data=np.array(X_data)

  return X_data, np.array(y_name)

def Train_Test_Split(standardized,name_data):

  x_data = np.array(standardized)
  # x_real_data_high = np.array(x_real_data_high)

  y_train, y_test, y_train_name, y_test_name = train_test_split( x_data, name_data, test_size=0.3, random_state=4 )

  x_train=[]
  x_test=[]
  # x_real_data_low=[]

  for i in range(len(y_train)):
    img = y_train[i]
    img = cv2.resize(img, (16, 16), interpolation = cv2.INTER_CUBIC)
    x_train.append(img)

  for i in range(len(y_test)):
    img = y_test[i]
    img = cv2.resize(img, (16, 16), interpolation = cv2.INTER_CUBIC)
    x_test.append(img)

  x_train=np.array(x_train)
  y_train=np.array(y_train)
  x_test=np.array(x_test)
  y_test=np.array(y_test)


  return x_train,y_train,x_test,y_test,y_train_name, y_test_name

def Train_Test_Split_Test(name_path):

  path = name_path #"/content/drive/MyDrive/Dipole Model/Train_Test_Data/Testing_Data/Experiment_1"

  List_Name_Image_Test = List_Name_Image(path)

  list_tam=np.array(List_Name_Image_Test)

  y_name_test =  list_tam

  X_data_test = Load_Data_Test(path,list_tam)
  X_data_test = np.array(X_data_test)

  return X_data_test,y_name_test

def Real_Data_Test_Split(x_real_data_high):

  x_real_data_low=[]

  for i in range(len(x_real_data_high)):
    img = x_real_data_high[i]
    img = cv2.resize(img, (16, 16), interpolation = cv2.INTER_CUBIC)
    x_real_data_low.append(img)

  x_real_data_low = np.array(x_real_data_low)

  return x_real_data_low, x_real_data_high

def Creating_Low_Image(list_data):

  list_data=np.array(list_data)

  x_real_data_low=[]

  for i in range(len(list_data)):
    img = list_data[i]
    img = cv2.resize(img, (16, 16), interpolation = cv2.INTER_CUBIC)
    x_real_data_low.append(img)

  x_real_data_low = np.array(x_real_data_low)

  return x_real_data_low

def Error_Image(real_image,predicted_image):
  list_error=[]
  for i in range(len(real_image)):
    S_real=np.sum(real_image[i])
    S_predicted=np.sum(predicted_image[i])
    error = np.float16(S_predicted/S_real)
    error=error**(1/2)
    list_error.append(error)
  return list_error

def del_row_col_Real_Dta(Matrix):
  tam=[]
  for i in range(len(Matrix)):
    m = np.delete(Matrix[i],[Matrix[i].shape[0]-1],0)
    m = np.delete(m,np.s_[m.shape[1]-1],1)
    tam.append(m)
  return np.array(tam)

def del_row_col_Dipole_Model(Matrix):
  tam=[]
  for i in range(len(Matrix)):
    m = np.delete(Matrix[i],np.s_[Matrix[i].shape[1]-1],1)
    tam.append(m)
  return np.array(tam)

def vmin_vmax(Matrix_A,Matrix_B):
  max_A = np.max(Matrix_A)
  min_A = np.min(Matrix_A)
  
  max_B = np.max(Matrix_B)
  min_B = np.min(Matrix_B)

  if(max_A>max_B):
    vmax=max_A
  else:
    vmax=max_B

  if(min_A<min_B):
    vmin=min_A
  else:
    vmin=min_B
  
  return vmax,vmin

def SSIM(list_img_original,list_img_predicted):
    
    
  result=[]
  for i in range(len(list_img_original)):
    result.append(ssim(list_img_original[i],list_img_predicted[i]))
  return result

def Plot_2D(list_data,file_name,V_min,V_max,wspace=0.05):
    list_data=np.array(list_data)
    num_samples = list_data.shape[0]
    
    # In từng sample riêng lẻ - computed color range
    for idx in range(num_samples):
        fig=plt.figure(figsize=(3,3))
        vmin_flag=V_min
        vmax_flag=V_max
        
        x_plot=list_data[idx]
        plt.imshow(x_plot,vmin=vmin_flag,vmax=vmax_flag,origin="lower",interpolation="bilinear",cmap=plt.cm.jet,aspect="equal")
        plt.axis('off')
        
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(file_name + f'_sample_{idx}' + '.png', format='png', bbox_inches='tight', pad_inches=0)
    
    # In từng sample riêng lẻ - fixed color range
    for idx in range(num_samples):
        fig=plt.figure(figsize=(3,3))
        vmin_flag=-0.05
        vmax_flag=0.05
        
        x_plot=list_data[idx]
        plt.imshow(x_plot,vmin=vmin_flag,vmax=vmax_flag,origin="lower",interpolation="bilinear",cmap=plt.cm.jet,aspect="equal")
        plt.axis('off')
        
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(file_name + f'_fixed_sample_{idx}' + '.png', format='png', bbox_inches='tight', pad_inches=0)

def Plot_2D_None(list_data,file_name,V_min,V_max,wspace=0.05):
    list_data=np.array(list_data)
    num_samples = list_data.shape[0]
    
    # In từng sample riêng lẻ
    for idx in range(num_samples):
        fig=plt.figure(figsize=(3,3))
        vmin_flag=V_min
        vmax_flag=V_max
        
        x_plot=list_data[idx]
        plt.imshow(x_plot,vmin=vmin_flag,vmax=vmax_flag,origin="lower",interpolation=None,cmap=plt.cm.jet,aspect="equal")
        plt.axis('off')
        
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(file_name + f'_sample_{idx}' + '.png', format='png', bbox_inches='tight', pad_inches=0)

def Plot_2D_Single(sample_2d, file_name, V_min=None, V_max=None, figsize=(4,4)):
  """Plot a single 2D sample with jet colormap and colorbar."""
  sample = np.array(sample_2d)
  if V_min is None:
    V_min = float(np.min(sample))
  if V_max is None:
    V_max = float(np.max(sample))

  fig, ax = plt.subplots(figsize=figsize)
  im = ax.imshow(sample, vmin=V_min, vmax=V_max, origin="lower", interpolation="bilinear", cmap=plt.cm.jet, aspect="equal")
  ax.axis('off')
  cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
  cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
  cbar.set_label('Value')
  plt.tight_layout()
  fig.savefig(file_name + '.png', format='png', dpi=200, bbox_inches='tight')
  plt.close(fig)

def Plot_2D_Grid(list_data, file_name, V_min, V_max, cols=5, figsize=(14, 6)):
  """Plot multiple samples in a grid with one shared colorbar."""
  list_data = np.array(list_data)
  num_samples = list_data.shape[0]
  try:
    cols = int(cols)
  except Exception:
    cols = 5
  if cols < 1:
    cols = 1
  rows = int(np.ceil(num_samples / cols))
  fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)

  norm = plt.Normalize(vmin=V_min, vmax=V_max)
  last_im = None

  for idx in range(rows * cols):
    ax = axes[idx // cols][idx % cols]
    if idx < num_samples:
      x_plot = list_data[idx]
      last_im = ax.imshow(x_plot, vmin=V_min, vmax=V_max, origin="lower", interpolation="bilinear", cmap=plt.cm.jet, aspect="equal")
      ax.axis('off')

      # Circle label like the example layout
      circle = Circle((0.14, 0.85), 0.1, transform=ax.transAxes, fill=False, color='black', linewidth=2)
      ax.add_patch(circle)
      ax.text(0.14, 0.85, f"{idx+1}", transform=ax.transAxes, ha='center', va='center', fontsize=12, fontweight='bold')
    else:
      ax.axis('off')

  if last_im is not None:
    cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.01)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    cbar.set_label('Value')

  plt.tight_layout()
  fig.savefig(file_name + '.png', format='png', dpi=200, bbox_inches='tight')
  plt.close(fig)

def Plot_Line(list_data,file_name):
  list_data=np.array(list_data)
  num_samples = list_data.shape[0]
  
  # First figure with up to 5 samples
  num_first = int(np.minimum(5, num_samples))
  fig, axes = plt.subplots(1, num_first, figsize=(17,4), sharey=True)
  
  if num_first == 1:
    axes = [axes]
  
  for i in range(num_first):
    axes[i].plot(list_data[i][16])
    axes[i].set_ylim(-0.3, 0.3)
  
  plt.tight_layout()
  plt.show()
  fig.savefig(file_name + '_1' + '.png', format='png')
  
  # Second figure with remaining samples (if any)
  if num_samples > 5:
    num_second = num_samples - 5
    fig, axes = plt.subplots(1, num_second, figsize=(17,4), sharey=True)
    
    if num_second == 1:
      axes = [axes]
    
    for i in range(num_second):
      axes[i].plot(list_data[5+i][16])
      axes[i].set_ylim(-0.3, 0.3)
    
    plt.tight_layout()
    plt.show()
    fig.savefig(file_name + '_2' + '.png', format='png')

def Plot_Line_Mix(list_data_real, list_data_predict, file_name):
  list_data_real = np.array(list_data_real)
  list_data_predict = np.array(list_data_predict)
  num_samples = list_data_real.shape[0]
  
  # First figure with up to 5 samples
  num_first = int(np.minimum(5, num_samples))
  fig, axes = plt.subplots(1, num_first, figsize=(17,4), sharey=True)
  
  if num_first == 1:
    axes = [axes]
  
  for i in range(num_first):
    axes[i].plot(list_data_predict[i][16])
    axes[i].plot(list_data_real[i][16])
    axes[i].set_ylim(-0.3, 0.3)
  
  plt.tight_layout()
  plt.show()
  fig.savefig(file_name + '_1' + '.png', format='png')
  
  # Second figure with remaining samples (if any)
  if num_samples > 5:
    num_second = num_samples - 5
    fig, axes = plt.subplots(1, num_second, figsize=(17,4), sharey=True)
    
    if num_second == 1:
      axes = [axes]
    
    for i in range(num_second):
      axes[i].plot(list_data_predict[5+i][16])
      axes[i].plot(list_data_real[5+i][16])
      axes[i].set_ylim(-0.3, 0.3)
    
    plt.tight_layout()
    plt.show()
    fig.savefig(file_name + '_2' + '.png', format='png')

def Plot_Bar(list_data, file_name, color='jet', vmin=None, vmax=None):
  """
  Vẽ bar chart từng sample riêng lẻ với gradient màu
  list_data: array các sample (3D array)
  file_name: đường dẫn file output
  color: colormap (mặc định: 'jet')
  vmin, vmax: min/max values cho colormap
  """
  list_data = np.array(list_data)
  num_samples = list_data.shape[0]
  
  # Tính vmin, vmax nếu không được cung cấp
  if vmin is None:
    vmin = np.min(list_data)
  if vmax is None:
    vmax = np.max(list_data)
  
  # In từng sample riêng lẻ
  for idx in range(num_samples):
    fig, ax = plt.subplots(figsize=(12, 3))
    
    x_data = list_data[idx].flatten()  # Flatten 2D array thành 1D
    x_axis = np.arange(len(x_data))
    
    # Normalize dữ liệu cho colormap
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.get_cmap(color)
    colors = cmap(norm(x_data))
    
    # Vẽ bar chart với gradient màu
    bars = ax.bar(x_axis, x_data, color=colors, width=0.8)
    ax.set_xlabel('Position')
    ax.set_ylabel('Value')
    ax.grid(axis='y', alpha=0.3)
    
    # Thêm colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    tick_values = np.linspace(vmin, vmax, 5)
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels([f"{t:.2f}" for t in tick_values])
    cbar.set_label('Value')
    
    plt.tight_layout()
    fig.savefig(file_name + f'_sample_{idx}' + '.png', format='png', bbox_inches='tight', pad_inches=0.1, dpi=100)
    plt.close(fig)

def Plot_Bar_Fixed(list_data, file_name, color='jet', vmin=-0.05, vmax=0.05):
  """Bar chart with fixed color scale to mirror fixed_sample heatmaps."""
  Plot_Bar(list_data, file_name, color=color, vmin=vmin, vmax=vmax)


# =============================================================================
# NEW FUNCTION: Load data from folder and align with labels CSV
# =============================================================================
def Load_Data_With_Labels(data_folder_path, labels_csv_path):
  """
  Load data from folder and align with labels CSV to ensure correct matching.
  
  Args:
    data_folder_path: Path to folder containing CSV data files
    labels_csv_path: Path to labels CSV file with columns: filename, width, length, depth
    
  Returns:
    X: numpy array of data (N, H, W)
    y: numpy array of labels (N, 3) - [width, length, depth]
    filenames: list of filenames that were successfully loaded
  """
  import pandas as pd
  
  # 1. Load labels from CSV
  labels_df = pd.read_csv(labels_csv_path)
  labels_dict = {}
  for idx, row in labels_df.iterrows():
    try:
      filename = str(row['filename']).strip()
      w = float(row['width'])
      l = float(row['length'])
      d = float(row['depth'])
      labels_dict[filename] = [w, l, d]
    except (ValueError, KeyError):
      continue
  
  # 2. Get all CSV files from data folder, sorted for consistency
  csv_files = sorted([f for f in os.listdir(data_folder_path) if f.endswith('.csv')])
  
  # 3. Load data and match with labels
  X = []
  y = []
  matched_filenames = []
  
  for csv_file in tqdm(csv_files, desc="📊 Loading Data with Labels", unit="file", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'):
    if csv_file not in labels_dict:
      print(f"  [WARNING] {csv_file} not found in labels CSV")
      continue
    
    # Load data from CSV
    try:
      filepath = os.path.join(data_folder_path, csv_file)
      with open(filepath) as f:
        reader = csv.reader(f)
        tam = []
        line = [row for row in reader]
        for k in range(4, len(line)):
          tam1 = []
          for l in range(len(line[k])):
            tam1.append(float(line[k][l]))
          tam.append(tam1)
        
        draft = np.array(tam)
        rotated_image = np.rot90(draft, 2)
        rotated_image = np.diff(rotated_image)
        
        if (('Step_R' in csv_file) == True) or (('Step_T' in csv_file) == True):
          Draft_1 = np.zeros((32, 31))
          for i in range(32):
            Draft_1[i] = rotated_image[i][::-1]
          rotated_image = Draft_1
        
        # Copy last column to fill 32x32
        last_column = rotated_image[:, -1:]
        rotated_image = np.column_stack([rotated_image, last_column])
        
        # ============ TÍNH ĐẠO HÀM GRADIENT (∇H) ============
        # Tính đạo hàm riêng theo 2 chiều: ∂H/∂y (dọc), ∂H/∂x (ngang)
        grad_y = np.gradient(rotated_image, axis=0)      # ∂H/∂y
        grad_x = np.gradient(rotated_image, axis=1)      # ∂H/∂x
        # Độ lớn gradient: ||∇H|| = sqrt((∂H/∂x)² + (∂H/∂y)²)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        # ================================================
        
        # Lưu cả field value + gradient magnitude
        field_with_gradient = np.dstack([rotated_image, gradient_magnitude])
        
        X.append(field_with_gradient)
        y.append(labels_dict[csv_file])
        matched_filenames.append(csv_file)
        
    except Exception as e:
      print(f"  ERROR loading {csv_file}: {str(e)}")
      continue
  
  X = np.array(X)
  y = np.array(y)
  
  # VERIFICATION: Check alignment between X, y, and filenames
  print(f"\n{'='*70}")
  print("DATA-LABEL ALIGNMENT VERIFICATION")
  print(f"{'='*70}")
  print(f"Total files loaded: {len(matched_filenames)}")
  print(f"X shape: {X.shape}")
  print(f"y shape: {y.shape}")
  
  if len(X) != len(y):
    print(f"  ERROR: Mismatch! X has {len(X)} samples but y has {len(y)} labels")
  elif len(X) != len(matched_filenames):
    print(f"  ERROR: Mismatch! X has {len(X)} samples but only {len(matched_filenames)} filenames")
  else:
    print(f"  OK: All {len(X)} samples, labels, and filenames are aligned")
  
  # Show first 3 samples for manual verification
  print(f"\nFirst 3 samples (for manual verification):")
  n_show = 3 if len(matched_filenames) >= 3 else len(matched_filenames)
  for i in range(n_show):
    print(f"  {i}: {matched_filenames[i]:<40} | W={y[i,0]:.4f}, L={y[i,1]:.4f}, D={y[i,2]:.4f} | Shape={X[i].shape}")
  print(f"{'='*70}\n")
  
  return X, y, matched_filenames

