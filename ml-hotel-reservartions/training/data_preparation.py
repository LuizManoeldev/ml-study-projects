import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Leitura do arquivo
base_reservations = pd.read_csv('../csv_files/reservations.csv')


# Removendo colunas
base_reservations = base_reservations.drop(['repeated_guest', 'no_of_previous_bookings_not_canceled'], axis = 1)

target_column = 'label_avg_price_per_room'


columns = [target_column] + [col for col in base_reservations.columns if col != target_column]
base_reservations = base_reservations[columns]

mapping = {1: 0, 2: 1, 3: 2}
base_reservations['label_avg_price_per_room'] = base_reservations['label_avg_price_per_room'].replace(mapping)

# Dividindo base de treinamento e teste 
train_size = int(0.7 * len(base_reservations))

base_treinamento = base_reservations.iloc[:train_size,:]
base_teste = base_reservations.iloc[train_size: ,:]


# Refatorando colunas classificatorias
base_treinamento = pd.get_dummies(base_treinamento, columns=['type_of_meal_plan', 'room_type_reserved', 'market_segment_type', 'booking_status'])
base_teste = pd.get_dummies(base_teste, columns=['type_of_meal_plan', 'room_type_reserved', 'market_segment_type', 'booking_status'])


# Convertendo para Float32
base_treinamento = base_treinamento.astype('float32')
base_teste = base_teste.astype('float32')


# Colunas de parametro e alvo para teste
X_teste = base_teste.iloc[:,1:25].values
y_teste = base_teste.iloc[:, 0].values

#Convertendo bases para CSV

#base_treinamento.to_csv('../csv_files/hotel_reservations_train_xgboost.csv', header = False, index = False)
#base_teste.to_csv('../csv_files/hotel_reservations_test_xgboost.csv', header = False, index = False)

