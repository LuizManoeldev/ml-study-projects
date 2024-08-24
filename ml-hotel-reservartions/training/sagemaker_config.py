import sagemaker
import boto3
from sagemaker import Session
import sagemaker.amazon.common as smac
import os, io

session = sagemaker.Session()

bucket_name = 'sagemaker-curso-bucket'

subpasta_modelo = 'modelos/hotel-reservation/xgboost'
subpasta_dataset = 'datasets/hotel-reservations'
key_train = 'hotel-train-data-xgboost'
key_test = 'hotel-test-data-xgboost'

role = "AmazonSageMaker-ExecutionRole-20240702T173175"

s3_train_data = 's3://{}/{}/train/{}'.format(bucket_name, subpasta_dataset, key_train)
s3_test_data = 's3://{}/{}/test/{}'.format(bucket_name, subpasta_dataset, key_test)
output_location = 's3://{}/{}/output'.format(bucket_name, subpasta_modelo)

# print(s3_train_data)
# print(s3_test_data)
# print(output_location)



with open('./csv_files/hotel_reservations_train_xgboost.csv', 'rb') as f:
    s3_train_path = os.path.join(subpasta_dataset, 'train', key_train).replace('\\', '/')
    boto3.Session().resource('s3').Bucket(bucket_name).Object(s3_train_path).upload_fileobj(f)
    
    
with open('./csv_files/hotel_reservations_test_xgboost.csv', 'rb') as f:
    s3_test_path = os.path.join(subpasta_dataset, 'test', key_test).replace('\\', '/')
    boto3.Session().resource('s3').Bucket(bucket_name).Object(s3_test_path).upload_fileobj(f)

