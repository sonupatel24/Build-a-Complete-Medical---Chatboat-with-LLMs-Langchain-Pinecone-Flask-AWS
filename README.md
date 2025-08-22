
# Build-a-Complete-Medical---Chatboat-with-LLMs-Langchain-Pinecone-Flask-AWS


Clone the repo 

````bash
git clonehttps://github.com/sonupatel24/Build-a-Complete-Medical---Chatbot-with-LLMs-Langchain-Pinecone-Flask-AWS.git
 ````


 ### step 1 : create a conda envirement 

 ````bash
 conda create -n medibot python=3.10 -y
 ````


 ````bash
 conda activate medibot
 ````


 # step 2 install the requirements

````bash
pip install -r requirements.txt
```

#3. Create a .env file in the root directory and add your Pinecone & openai credentials as follows:
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
GOOGLE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# 4. run the following command to store embeddings to pinecone
python store_index.py

# Finally run the following command
python app.py


Now,
open up localhost:

#Techstack Used:
Python
LangChain
Flask
GEMINI
Pinecone





