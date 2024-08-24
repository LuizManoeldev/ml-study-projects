from sagemaker import image_uris
import sagemaker
from sagemaker.tuner import HyperparameterTuner, IntegerParameter, ContinuousParameter
from sagemaker.serializers import CSVSerializer

from sagemaker_config import session, role, output_location, s3_train_data, s3_test_data
from data_preparation import X_teste, y_teste

from sklearn.metrics import mean_absolute_error, mean_squared_error

import numpy as np



container = image_uris.retrieve(framework = 'xgboost', region='us-east-1', version='1.7-1')

xgboost = sagemaker.estimator.Estimator(image_uri = container,
                                        role = role,
                                        instance_count = 1,
                                        instance_type = 'ml.m5.2xlarge',
                                        output_path = output_location,
                                        sagemaker_session = session,
                                        use_spot_instances= True,
                                        max_run=3600,
                                        max_wait=7200
                                        )


# Set initial hyperparameters
xgboost.set_hyperparameters(
    num_round = 200, 
    num_class = 3, 
    objective='multi:softmax',
    eval_metric='merror'
)



train_input = sagemaker.inputs.TrainingInput(s3_data = s3_train_data, content_type='csv', s3_data_type = 'S3Prefix')
validation_input = sagemaker.inputs.TrainingInput(s3_data = s3_test_data, content_type='csv', s3_data_type = 'S3Prefix')
data_channels = {'train': train_input, 'validation': validation_input}

job_name = 'xgboost-70-train-base-v1'

xgboost.fit(data_channels, job_name= job_name)


#Deploy Test

xgboost_regressor = xgboost.deploy(initial_instance_count = 1, instance_type = 'ml.m4.xlarge')

xgboost_regressor.serializer = CSVSerializer()  

# Fazendo a previsão
previsoes = xgboost_regressor.predict(X_teste)

# Decodificando os bytes para string
previsoes_str = previsoes.decode('utf-8')

# Dividindo a string em valores numéricos
previsoes_list = previsoes_str.split(',')

# Convertendo a lista de strings para um array NumPy de floats
previsoes = np.array(previsoes_list, dtype=np.float64)

# Arredondando os valores para o inteiro mais próximo e convertendo para inteiros
previsoes = np.round(previsoes).astype(np.int32)

#print(previsoes)

mapping1 = {0:1, 1: 2, 2:3}
mapping2 = {0.0:1, 1.0:2, 2.0:3}

def apply_mapping(arr, mapping):
    return np.vectorize(lambda x: mapping[x])(arr)

# Converter y_teste
y_teste = apply_mapping(y_teste, mapping1)

# Converter previsoes para inteiros e depois para a escala desejada
previsoes = previsoes.astype(int)  # Converter float para int
previsoes = apply_mapping(previsoes, mapping1)  # Aplicar o mapeamento para a escala de 1, 2, 3

mae = mean_absolute_error(y_teste, previsoes)
mse = mean_squared_error(y_teste, previsoes)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(y_teste, previsoes)

# Calculando acurácia
accuracy = 100 - mape

print('MAE:', mae)
print('MSE:', mse)
print('MAPE:', mape)
print('Accuracy:', accuracy)