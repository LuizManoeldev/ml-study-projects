import boto3
import xgboost as xgb
import tarfile
from data_preparation import X_teste, y_teste
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize the S3 client
s3 = boto3.client('s3', region_name='us-east-1')

# Specify your bucket name and model path in S3
bucket_name = 'sagemaker-curso-bucket'
#model_key = 'modelos/hotel-reservation/xgboost/output/xgboost-training-job-hotel-reservations-v3/output/model.tar.gz'  
#model_key = 'modelos/hotel-reservation/xgboost/output/xgboost-training-job-hotel-reservations-v5-pretunning/output/model.tar.gz'  
#model_key = 'modelos/hotel-reservation/xgboost/output/xgboost-training-job-hotel-reservations-v6/output/model.tar.gz'  

# model_key = 'modelos/hotel-reservation/xgboost/output/xgboost-70-train-base-v1/output/model.tar.gz'  


# # Download the model file to a local directory
local_model_path = 'model/model_ester.tar.gz'  
# s3.download_file(bucket_name, model_key, local_model_path)

# print(f"Downloaded model artifact from S3: {local_model_path}")

extracted_model_dir = 'model'

with tarfile.open(local_model_path, 'r:gz') as tar:
    tar.extractall(path=extracted_model_dir)

model_file = 'model/xgboost-model'

model = xgb.Booster()
model.load_model(model_file)    

dtest = xgb.DMatrix(X_teste)

predictions2 = model.predict(dtest)


print(classification_report(y_teste, predictions2))

print("Accuracy: ", accuracy_score(y_teste, predictions2))

conf_matrix = confusion_matrix(y_teste, predictions2)

plt.figure(figsize=(7, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[1, 2, 3], yticklabels=[1, 2, 3])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()