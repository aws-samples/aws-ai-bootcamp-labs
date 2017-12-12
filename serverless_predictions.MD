# Serverless Predictions at Scale

## Set up the AWS CodeStar project

1. Navigate to the AWS CodeStar service console at https://console.aws.amazon.com/codestar/home#/projects 
2. Click the **Create a new project** tile to launch the quickstart wizard. This takes you to the template selection screen in which you select to build a **Python**-based **Web Service** using **AWS Lambda**.
3. The next screen asks you to provide details for the project being created. In the "Project name" field enter ***mxnet-resnet***. For the purpose of this lab we will rely on AWS CodeCommit to store the project's source code. Note, that the project name used before is also used in the "Repository name" field. Click the **Next** button to continue with the setup.
4. Take time to review the project details and the main components of the AWS CodePipeline and resources AWS CodeStar provisions on your behalf. Make sure to provide AWS CodeStar with permissions to provision and administer the underlying AWS resources, i.e. keep the checkbox selected before your continue by clicking the **Create Project** button.
5. As AWS CodeStar provisions your resources in the background follow the instructions to set up your code editor and tools. Select **Command line tools** and click the **Next** button. Choose your operating system, connection method, set up Git, generate the credentials and eventually clone the repository to your development environment.

## Adjust CodeStarWorker-mxnet-resnet-CloudFormation IAM role to be able to put a bucket policy on the S3 model store

1. Navigate to the Identity and Access Management console at https://console.aws.amazon.com/iam/home#/roles
2. Search and filter for the role ***CodeStarWorker-mxnet-resnet-CloudFormation***
3. Click on the role name ***CodeStarWorker-mxnet-resnet-CloudFormation*** and review the inline policy attached to the role.
4. Over the course of the next couple of steps we are going to create an S3 bucket for storing model files. In order to govern these resources using CodeStar it is necessary to adjust the default policy. Click **Edit Policy** and add the following block to the “Statement” part inline policy. Make sure to replace the `${AWS::Region}` with the acronym of the region that you are using, e.g. eu-central-1, and the `${AWS::AccountId}` with your AWS Account ID, e.g. `123456789012`. Click **Save** to persist your changes to the inline policy. 

```
{
    "Action": [
        "s3:PutBucketPolicy"
    ],
    "Resource": [
        "arn:aws:s3:::mxnet-resnet-${AWS::Region}-${AWS::AccountId}-models"
    ],
    "Effect": "Allow"
}
```

## Add S3 bucket to host pre-trained models

1. Open the ***template.yml*** file and add a S3 bucket resource. This bucket will later be used to store the pre-trained models. Our Lambda function will later download the models from this defined location to the function execution environment. Commit and push your changes to the CodeCommit repository. After you have pushed your changes CodePipeline picks up your changes and will create an S3 bucket which can be used to host pre-trained models. 

```
AWSTemplateFormatVersion: 2010-09-09
Transform:
- AWS::Serverless-2016-10-31
- AWS::CodeStar

Parameters:
  ProjectId:
    Type: String
    Description: CodeStar projectId used to associate new resources to team members

Resources:
  HelloWorld:
    Type: AWS::Serverless::Function
    Properties:
      Handler: index.handler
      Runtime: python2.7
      Role:
        Fn::ImportValue:
          !Join ['-', [!Ref 'ProjectId', !Ref 'AWS::Region', 'LambdaTrustRole']]
      Events:
        GetEvent:
          Type: Api
          Properties:
            Path: /
            Method: get
        PostEvent:
          Type: Api
          Properties:
            Path: /
            Method: post

  ModelStoreBucket:
    Type: "AWS::S3::Bucket"
    Properties: 
      BucketName: 
        !Join ['-', [!Ref 'ProjectId', !Ref 'AWS::Region', !Ref 'AWS::AccountId', 'models']]
  ModelStoreBucketPolicy:
    Type: "AWS::S3::BucketPolicy"
    Properties: 
      Bucket: 
        !Ref ModelStoreBucket
      PolicyDocument: 
        Statement:
          - Effect: "Allow"
            Action: 
            - "s3:GetObject"
            Resource: 
              Fn::Join: ["", ["arn:aws:s3:::", !Ref ModelStoreBucket, "/*"]]
            Principal:
              AWS:
              - Fn::ImportValue: !Join ['-', [!Ref 'ProjectId', !Ref 'AWS::Region', 'LambdaTrustRole']]

Outputs:
  ModelStoreBucket:
    Description: A reference to S3 bucket to use as model store.
    Value: !Ref ModelStoreBucket
```

## Download the pre-trained models and copy them to the S3 model store bucket

1. Download the pre-trained model files from http://data.mxnet.io/models site. The ResNet-18 files can be found in the imagenet/resnet/18-layers subfolder. The resnet-18-symbol.json file contains the model definition of the pre-trained model. The resnet-18-0000.params file is a binary containing the parameters. Also, make sure to fetch the text file containing the labels from http://data.mxnet.io/models/imagenet/resnet/synset.txt.
2. Copy the files to the S3 bucket used as model store.

The following code outlines CLI commands that can be used to automate the above steps on macOS/Linux. **Note: Make sure to configure the AWS CLI before executing those commands, e.g. set the region to the region where you created the CodeStar project.**

```
# create temp. directory
mkdir -p /tmp/resnet18
# enter temp. directory
pushd $_
# download pre-trained model files
wget "http://data.mxnet.io/models/imagenet/resnet/synset.txt" -O resnet-18-synset.txt
wget "http://data.mxnet.io/models/imagenet/resnet/18-layers/resnet-18-0000.params" -O resnet-18-0000.params
wget "http://data.mxnet.io/models/imagenet/resnet/18-layers/resnet-18-symbol.json" -O resnet-18-symbol.json
# assemble name of S3 bucket to use as model store
region=$(aws configure get region)
account=$(aws sts get-caller-identity --query 'Account' --output text)
bucket="mxnet-resnet-${region}-${account}-models" 
# sync files to model store bucket
aws s3 sync . s3://${bucket}
# leave temp. directory
popd
```

## Add the Lambda function code to the repository

1. Create a file named ***lambda_function.py*** based on the source code given below. Add the file to the repository, push and commit the changes to the repository.

```
import os
import boto3
import json
import tempfile
import urllib2
import mxnet as mx
import numpy as np
import cv2

# get model store config from environment
f_bucket = os.environ['MODEL_STORE_BUCKET_NAME'] 
f_prefix = os.environ['MODEL_STORE_PREFIX'] 
f_params = os.environ['MODEL_PARAMS_FILE_NAME'] 
f_symbol = os.environ['MODEL_SYMBOL_FILE_NAME'] 
f_synset = os.environ['MODEL_SYNSET_FILE_NAME'] 

prediction_model_check_point = 0
prediction_model_prefix = '/tmp/resnet-18'

s3 = boto3.resource('s3')
s3_client = boto3.client('s3')

# load pre-trained model files to temporary files
s3_client.download_file(f_bucket, f_prefix + f_params, '/tmp/' + f_params)
s3_client.download_file(f_bucket, f_prefix + f_symbol, '/tmp/' + f_symbol)
s3_client.download_file(f_bucket, f_prefix + f_synset, '/tmp/' + f_synset)

with open('/tmp/' + f_synset, 'r') as f:
  synsets = [l.rstrip() for l in f]

def lambda_handler(event, context):
  print("Received event.")
  if event['httpMethod'] == 'GET':
    url = event['queryStringParameters']['url']
  else:
    return None

  print("Determined URL = " + url)
  prediction_sym, arg_params, aux_params = mx.model.load_checkpoint(prediction_model_prefix, prediction_model_check_point)
  print("Loaded model checkpoint from file.")
  prediction_model = mx.mod.Module(symbol=prediction_sym, label_names=['softmax_label'])
  print("Created model.")
  prediction_model.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 224))], label_shapes=[('softmax_label', (1,))])
  print("Bound the symbols to construct executors.")
  prediction_model.set_params(arg_params=arg_params, aux_params=aux_params, allow_missing=True)
  print("Assigned parameter and aux state values.")

  req = urllib2.urlopen(url)
  arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
  cv2_img = cv2.imdecode(arr, -1)
  img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
  if img is None:
    return None

  print("Loaded image.")
  img = cv2.resize(img, (224, 224))
  img = np.swapaxes(img, 0, 2)
  img = np.swapaxes(img, 1, 2) 
  img = img[np.newaxis, :] 
  print("Prepared image for model forward.")
  print(img.shape)

  pred_data_iter = mx.io.NDArrayIter(data={'data': img}, batch_size=1)
  predictions = prediction_model.predict(eval_data=pred_data_iter).asnumpy()

  # sort predictions based on their probability
  predictions = np.squeeze(predictions)
  predictions_idxs = np.argsort(predictions)
  top_predictions_idxs = predictions_idxs[::-1][0:5]
  print("Indices of top predictions %s" % top_predictions_idxs)
  
  out = '{ "predictions": [' 
  for i in top_predictions_idxs:
    out += '{ "probability" : "%f", "class" : "%s" },' % (predictions[i], synsets[i]) 
  out = out[:-1]
  out += "] }"
  return out

def lambda_proxy_handler(event, context):
  body = lambda_handler(event, context);
  if body is None:
    body = {}
  
  resp = {}
  resp['statusCode'] = 200
  resp['headers'] = {}
  resp['headers']['Content-Type'] = 'application/json'
  resp['body'] = body 
  return resp
```

## Adjust the build specification to package the function code

1. Navigate to the AWS CodeStar service console at https://console.aws.amazon.com/codestar/home#/projects 
2. Locate the ***mxnet-resnet** *project and open the dashboard
3. From the dashboard click the **Build** on the left navigator and you will be taken to the overview of the ***mxnet-resnet*** build project in the AWS CodeBuild console. Click the **Edit project** button to adjust the project build settings.
4. On the build settings page update the image to be used by clicking the **Update image** button. Select **Specify a Docker image**, choose **Linux** as the "Environment type", **Other** as the "Custom image type" and ***amazonlinux:latest*** as the "Custom image ID". Click the **Update** button to apply the changes.
5. Edit the ***buildspec.yml*** file to update the build specification. The specification installs dependencies to correctly package a ZIP containing the dependencies and execution code for our function code. Make sure to push and commit the changes to the repository.

```
version: 0.2

phases:
  install:
    commands:
      # install Python 2.7
      - yum install -y python27
      - yum install -y python27-virtualenv
      # install AWS CLI
      - yum install -y aws-cli
      # install zip utility
      - yum install -y zip
      # install libgomp
      - yum install -y libgomp
      # set up and configure virtual environment
      - mkdir -p $HOME/env/mxnet
      - cd $HOME/env/mxnet
      - virtualenv .
      - source bin/activate
      # upgrade pip
      - pip install --upgrade pip
      # install dependencies 
      - "pip install mxnet==0.11.0 --only-binary=:all:"
      - "pip install opencv-python --only-binary=:all:"
  build:
    commands:
      # create distribution directory for Lambda package
      - mkdir -p $VIRTUAL_ENV/dist
      # copy function code to distribution directory
      - cp -rf $CODEBUILD_SRC_DIR/*.py $VIRTUAL_ENV/dist
      # copy dependencies to distribution directory
      - cp -rf $VIRTUAL_ENV/lib/python2.7/site-packages/* $VIRTUAL_ENV/dist
      - cp -rf $VIRTUAL_ENV/lib64/python2.7/site-packages/* $VIRTUAL_ENV/dist
      - cp /usr/lib64/libgomp.so.1 $VIRTUAL_ENV/dist
      # minimize Lambda package size and zip
      - cd $VIRTUAL_ENV/dist
      - rm -rf pip*
      - rm -rf setuptools*
      - zip -r9 $CODEBUILD_SRC_DIR/mxnet-lambda.zip .
      - du -sh $CODEBUILD_SRC_DIR/mxnet-lambda.zip
  post_build:
    commands:
      # package local artifcats
      - cd $CODEBUILD_SRC_DIR
      - aws cloudformation package --template template.yml --s3-bucket $S3_BUCKET --output-template template-export.yml
artifacts:
  type: zip
  files:
    - template-export.yml
```



## Adjust CodeStarWorker-mxnet-resnet-Lambda IAM role to be able to download the pre-trained models from S3

1. Navigate to the Identity and Access Management console at https://console.aws.amazon.com/iam/home#/roles
2. Search and filter for the role ***CodeStarWorker-mxnet-resnet-Lambda***
3. Click on the role name  ***CodeStarWorker-mxnet-resnet-Lambda*** and review the attached policies
4. Click **Attach policy**, filter for *AmazonS3ReadOnlyAccess*, select the checkbox and click **Attach Policy** 

## Adjust SAM template to include API endpoints for prediction

1. Open the ***template.yml*** file and add a function definition to serve predictions based on the pre-trained models.

```
AWSTemplateFormatVersion: 2010-09-09
Transform:
- AWS::Serverless-2016-10-31
- AWS::CodeStar

Parameters:
  ProjectId:
    Type: String
    Description: CodeStar projectId used to associate new resources to team members

Resources:
  HelloWorld:
    Type: AWS::Serverless::Function
    Properties:
      Handler: index.handler
      Runtime: python2.7
      Role:
        Fn::ImportValue:
          !Join ['-', [!Ref 'ProjectId', !Ref 'AWS::Region', 'LambdaTrustRole']]
      Events:
        GetEvent:
          Type: Api
          Properties:
            Path: /
            Method: get
        PostEvent:
          Type: Api
          Properties:
            Path: /
            Method: post
  Predict:
    Type: AWS::Serverless::Function
    Properties:
      Handler: lambda_function.lambda_proxy_handler
      Runtime: python2.7
      CodeUri: ./mxnet-lambda.zip
      MemorySize: 1024
      Timeout: 120
      Role:
        Fn::ImportValue:
          !Join ['-', [!Ref 'ProjectId', !Ref 'AWS::Region', 'LambdaTrustRole']]
      Events:
        GetEvent:
          Type: Api
          Properties:
            Path: /predict
            Method: get
      Environment:
        Variables:
          MODEL_STORE_BUCKET_NAME: !Join ['-', [!Ref 'ProjectId', !Ref 'AWS::Region', !Ref "AWS::AccountId", 'models']]
          MODEL_STORE_PREFIX: ""
          MODEL_PARAMS_FILE_NAME: "resnet-18-0000.params"
          MODEL_SYMBOL_FILE_NAME: "resnet-18-symbol.json"
          MODEL_SYNSET_FILE_NAME: "resnet-18-synset.txt"
          
  ModelStoreBucket:
    Type: "AWS::S3::Bucket"
    Properties:
      BucketName:
        !Join ['-', [!Ref 'ProjectId', !Ref 'AWS::Region', !Ref "AWS::AccountId", 'models']]
  ModelStoreBucketPolicy:
    Type: "AWS::S3::BucketPolicy"
    Properties:
      Bucket:
        !Ref ModelStoreBucket
      PolicyDocument:
        Statement:
          - Effect: "Allow"
            Action:
            - "s3:GetObject"
            Resource:
              Fn::Join: ["", ["arn:aws:s3:::", !Ref ModelStoreBucket, "/*"]]
            Principal:
              AWS:
              - Fn::ImportValue: !Join ['-', [!Ref 'ProjectId', !Ref 'AWS::Region', 'LambdaTrustRole']]

Outputs:
  ModelStoreBucket:
    Description: A reference to the S3 bucket to use as model store.
    Value: !Ref ModelStoreBucket
```

## Call the endpoint

1. Navigate to the AWS CodeStar service console at https://console.aws.amazon.com/codestar/home#/projects 
2. Locate the ***mxnet-resnet*** project and open the dashboard
3. Identify the **Application endpoints** tile and click the link.
4. Manipulate the URL from /Prod to /Prod/predict?url=https://images-na.ssl-images-amazon.com/images/G/01/img15/pet-products/small-tiles/23695_pets_vertical_store_dogs_small_tile_8._CB312176604_.jpg 
5. Open the URL to the image which was used for the prediction, e.g. https://images-na.ssl-images-amazon.com/images/G/01/img15/pet-products/small-tiles/23695_pets_vertical_store_dogs_small_tile_8._CB312176604_.jpg
