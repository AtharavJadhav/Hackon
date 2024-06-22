# Amazon HackOn Team CodeBreachers (VNIT) Product 2:

## Brief


So this is our second product: Do It with Amazon. We decided to make this product so as to get as many people to open up Amazon as possible and increase the instinctive buying tendency of the customers. Here is a simulated application of Amazon mobile, after clicking the search with the image option. Here we added the option to Do It With Amazon, and our philosophy is that a person can find the solution to his/her daily problems by talking with Amazon, be it cooking a special dish, or fixing the plumbing issues. 


After clicking the button, we can see our chat interface, where the user can input an image of their problem, or something they want to do, and give a brief description of their problem. After clicking send the user will receive the response.


We have integrated the place where customers can buy the products listed in the response Direct from Amazon.


Along with that, we have added the regional language option too so that a large multitude of users can use our products.

## How to use

Step 1: Clone the repository: 
```
git clone github.com/AthravJadhav/Hackon.git

```

Step 2: Move to the "backend" directory:

```
cd backend

```

Step 3: Edit the paths in the hackon.py file

Step 4: Downlaod the 3 models along with all thier files and store it in the following tree structure:

```

Hackon
|
|--- Amazon_Dataset
|
|--- fronend
|
|--- backend
|
|--- models
        |
        |--- translation_model
                    |
                    |--- files for translation model
        |
        |--- chatbot_model
                    |
                    |--- files for chatbot model
        |
        |--- classification_model
                    |
                    |--- files for classification model

```
The links for the models are:

1) [translation model] (https://huggingface.co/Helsinki-NLP/opus-mt-en-hi)
2) [chatbot_model] (https://huggingface.co/anakin87/zephyr-7b-alpha-sharded)
3) [classification model] (https://huggingface.co/google/vit-base-patch16-224)

Adjust the paths wherever neccesaary in the hackon.py file.

Step 5: Install the libraries:

```
!pip install -q transformers peft accelerate bitsandbytes safetensors sentencepiece streamlit chromadb langchain sentence-transformers sacremoses pypdf langchain-community pillow torchvision
```

Step 6: Start the backend code:

```
python hackon.py
```

Step 7: Start the frontend & view the application on mobile or web

```
cd frontend
npx expo start
```

## ThankYou
