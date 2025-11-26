import numpy as np
import sklearn 
import seaborn
import matplotlib.pyplot as plt
import pandas as pd

#################################################################################
# 1. Read the data and plot the data
#################################################################################

data = pd.read_csv("/home/asier/Escritorio/GitHub/MicroDatosIA/08-regression-classification/Data/student_exam_data.csv")
print('############ Read data ############\n')
print(data.head())

