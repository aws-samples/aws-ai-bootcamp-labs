# AWS-AI-Bootcamp-Labs
This library has a collection of Notebooks and code examples for AWS AI Bootcamps.

### Content

[MNIST MXNet Demo Notebook](Notebooks/FashionMNIST_MXNet_Demo.ipynb)

[Amazon Lex Demo Notebook](Notebooks/Lex_Demo.ipynb)

[Amazon Polly Demo Notebook](Notebooks/Polly_Demo.ipynb)

[Amazon Rekognition Demo Notebook](Notebooks/Rekognition_Demo.ipynb)

[Amazon Machine Learning Demo Notebook](Notebooks/AmazonML_Demo.ipynb)

### Launch EC2 instance using the deep learning AMI and open fashion MNIST MXNet demo

1. Create EC2 IAM role for the workshop as described [here](http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/iam-roles-for-amazon-ec2.html#create-iam-role). We will apply permission policies as documented in each notebook
2. Launch EC2 Instance using the Ubuntu deep learning AMI in eu-west-1, Ireland (p2.xlarge - $0.972/hour) http://amzn.to/2j3FdOZ
3. Connect via SSH and tunnel port 8888:
    * Linux, Mac:
        - `ssh -i user.pem -L 8888:localhost:8888 ubuntu@ec2-ip-ip-ip-ip.region.compute.amazonaws.com`
    * Windows: 
        - Follow the instructions [here](http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/putty.html) to download PuTTY and to convert your private key
        - Host Name: `ubuntu@ec2-ip-ip-ip-ip.region.compute.amazonaws.com`
        - Expand Connection and choose Auth, select your .ppk file
        - Expand Connection > SSH, choose Tunnels, specify Source Port: `8888`, Destination: `localhost:8888`
        - Choose Add and Open
4. Clone aws-ai-bootcamp-labs github repository `git clone https://github.com/awslabs/aws-ai-bootcamp-labs`
5. Start jupyter notebook: `nohup jupyter notebook &`
6. `tail nohup.out` to get the login token
    * look for `http://localhost:8888/?token=<your_login_token>`
7. Open demo notebook (FashionMNIST_MXNet_Demo.ipynb)
    * Select Kernel > Change kernel > Python 2
8. Follow steps in notebook
