# dev diary

## structure
config == 暂时没用
dataset == 数据集的初始文件（SMSSpamCollection）
           中间文件
            最终文件 vectorised_sms_data.csv
preprocess == 各种预处理脚本
            应用顺序=cleanup ->sbert
## work flow
* preprocess
* * get data (done)
* * clean (totally done)
* * analyse data (done)
* divide data
* select model
* train model

## get data
I downloaded data set form ucl machine learning repo.  
the data set is a text file,like:  
- ``ham	Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...``  
- ``ham	Ok lar... Joking wif u oni...``  
- ``spam	Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry 
 question(std txt rate)T&C's apply 08452810075over18's``
- ``ham	U dun say so early hor... U c already then say...``

## clean data
it's done in **cleanUp.py**.
several steps:
- lowercase
- remove punctuation(标点)
- replace numbers with "numbers" (a string, "1" -> "numbers")
- replace urls with "urls"
- remove extra whitespace
- remove stop words
- lemmatization(换回原型)

## analyse
### *不平衡的数据*
Number of all messages: 5574
Number of ham messages: 4827
Number of spam messages: 747

