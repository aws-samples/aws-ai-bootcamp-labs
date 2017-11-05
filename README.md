# AWS-AI-Bootcamp-Labs
This library has a collection of Notebooks and code examples for AWS AI Bootcamps.

### Content

[MNIST MXNet Demo Notebook](Notebooks/FashionMNIST_MXNet_Demo.ipynb)

[Amazon Lex Demo Notebook](Notebooks/Lex_Demo.ipynb)

[Amazon Polly Demo Notebook](Notebooks/Polly_Demo.ipynb)

[Amazon Rekognition Demo Notebook](Notebooks/Rekognition_Demo.ipynb)

[Amazon Machine Learning Demo Notebook](Notebooks/AmazonML_Demo.ipynb)

### Launch EC2 instance using the deep learning AMI and open fashion MNIST MXNet demo

1. Create EC2 IAM role for the workshop, either apply `AdministratorAccess` policy or use policies as described in each notebook
2. Launch EC2 Instance using the Ubuntu deep learning AMI in eu-west-1, Ireland (p2.xlarge - $0.972/hour) http://amzn.to/2j3FdOZ
3. Connect via SSH: `ssh -i user.pem ubuntu@ec2-ip-ip-ip-ip.region.compute.amazonaws.com`
4. Clone aws-ai-bootcamp-labs github repository `git clone https://github.com/awslabs/aws-ai-bootcamp-labs`
4. Start jupyter notebook: `nohup jupyter notebook &`
5. Create SSH tunnel: `ssh -f -i user.pem -N -L 8888:127.0.0.1:8888 ubuntu@ip-ip-ip-ip.region.compute.amazonaws.com'
6. Open notebook landing page and follow instructions provided `tail nohup.out` to get token
8. Open demo notebook (FashionMNIST_MXNet_Demo.ipynb)
9. Follow steps in notebook