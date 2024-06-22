import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline, MarianMTModel, MarianTokenizer
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import PyPDFDirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from IPython.display import display
from sentence_transformers import SentenceTransformer, util

import os
from transformers import ViTForImageClassification, ViTFeatureExtractor
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer

from transformers import DefaultDataCollator
from torch.utils.data import Dataset
from PIL import Image

class ImageClassificationDataset(Dataset):
    def __init__(self, folder_path, transform):
        self.dataset = ImageFolder(root=folder_path, transform=transform)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return {"pixel_values": image, "labels": label}

# Define a custom data collator
class CustomDataCollator(DefaultDataCollator):
    def __call__(self, features):
        pixel_values = torch.stack([f["pixel_values"] for f in features])
        labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)
        return {"pixel_values": pixel_values, "labels": labels}

# Function to train the image classifier
def train_image_classifier(dataset_path):
    feature_extractor = ViTFeatureExtractor.from_pretrained("C:/Users/Atharav Jadhav/OneDrive/Desktop/Hackon/models/image_classification_model")
    model = ViTForImageClassification.from_pretrained("C:/Users/Atharav Jadhav/OneDrive/Desktop/Hackon/models/image_classification_model", num_labels=len(os.listdir(dataset_path)))

    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
    ])

    dataset = ImageClassificationDataset(folder_path=dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=8,
        num_train_epochs=3,
    )

    data_collator = CustomDataCollator()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    trainer.train()
    return model, feature_extractor

# Example usage:
dataset_path = "C:/Users/Atharav Jadhav/OneDrive/Desktop/Hackon/amazon_dataset/train_images"
image_classifier_model, feature_extractor = train_image_classifier(dataset_path)

# fixing unicode error in google colab
import locale
locale.getpreferredencoding = lambda: "UTF-8"

import torch
from PIL import Image

class Chatbot:
    def __init__(self, translation_model_name, source_language_code, target_language_code, model, tokenizer, retriever, llm, chat_history, image_classifier, feature_extractor):
        self.translation_model = MarianMTModel.from_pretrained(translation_model_name)
        self.translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
        self.source_language_code = source_language_code
        self.target_language_code = target_language_code

        self.model = model
        self.tokenizer = tokenizer
        self.retriever = retriever
        self.llm = llm
        self.chat_history = chat_history
        self.similarity_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self.image_classifier = image_classifier
        self.feature_extractor = feature_extractor

    def translate(self, text):
        inputs = self.translation_tokenizer(text, return_tensors="pt", truncation=True)
        translation = self.translation_model.generate(**inputs)
        translated_text = self.translation_tokenizer.batch_decode(translation, skip_special_tokens=True)[0]
        return translated_text

    def classify_image(self, image_path):
        image = Image.open(image_path)
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        outputs = self.image_classifier(**inputs)
        predicted_class = outputs.logits.argmax(-1).item()
        return self.image_classifier.config.id2label[predicted_class]

    def create_conversation(self, query, image_path=None):
        try:
            combined_query = query
            if image_path:
                image_label = self.classify_image(image_path)
                combined_query = f"{query} Here is an image related to '{image_label}'."

            memory = ConversationBufferMemory(
                memory_key='chat_history',
                return_messages=False
            )
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.retriever,
                memory=memory,
                get_chat_history=lambda h: h,
            )

            result = qa_chain({'question': combined_query, 'chat_history': self.chat_history})
            filtered_response = self.filter_response(combined_query, result['answer'])
            self.chat_history.append((combined_query, filtered_response))
            return filtered_response, self.chat_history

        except Exception as e:
            self.chat_history.append((query, str(e)))
            return str(e), self.chat_history

    def filter_response(self, query, response):
        relevant_sentences = self.semantic_similarity_scoring(query, response)
        return ' '.join(relevant_sentences)

    def semantic_similarity_scoring(self, query, response):
        query_embedding = self.similarity_model.encode(query, convert_to_tensor=True)
        response_sentences = response.split('. ')
        response_embeddings = self.similarity_model.encode(response_sentences, convert_to_tensor=True)

        similarities = util.pytorch_cos_sim(query_embedding, response_embeddings)[0]
        scored_sentences = [(similarity, sentence) for similarity, sentence in zip(similarities, response_sentences)]

        scored_sentences.sort(reverse=True, key=lambda x: x[0])
        combined_response = []
        combined_length = 0
        max_length = 500
        threshold = 0.5

        for score, sentence in scored_sentences:
            if score >= threshold:
                combined_response.append(sentence)
                combined_length += len(sentence.split())
                if combined_length > max_length:
                    break

        if not combined_response:
            combined_response = response_sentences[:3]

        return combined_response

    # def on_submit_button_click(self, b):
    #     query = self.query_input.value
    #     use_hindi = self.hindi_checkbox.value
    #     image_path = None

    #     if self.image_upload.value:
    #         image_bytes = self.image_upload.value[0]['content']
    #         with open("uploaded_image.jpg", "wb") as f:
    #             f.write(image_bytes)
    #         image_path = "uploaded_image.jpg"

    #     if use_hindi:
    #         translated_query = self.translate(query)
    #     else:
    #         translated_query = query

    #     response, self.chat_history = self.create_conversation(translated_query, image_path)

    #     if use_hindi:
    #         translated_response = self.translate(response)
    #     else:
    #         translated_response = response

    #     with self.response_output:
    #         print(translated_response)

    #     self.chat_history_output.clear_output(wait=True)
    #     with self.chat_history_output:
    #         for entry in self.chat_history:
    #             print(f"User: {entry[0]}")
    #             print(f"Bot: {entry[1]}")

class ChatbotTraining:
    def __init__(self, translation_model_name, source_language_code, target_language_code, model_name, folder_path, embedding_model_name):
        self.translation_model = MarianMTModel.from_pretrained(translation_model_name)
        self.translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
        self.source_language_code = source_language_code
        self.target_language_code = target_language_code

        self.model = self.load_quantized_model(model_name)
        self.tokenizer = self.initialize_tokenizer(model_name)


        self.folder_path = folder_path
        self.documents = self.load_documents()

        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        self.all_splits = self.text_splitter.split_documents(self.documents)

        self.embedding_model_name = embedding_model_name
        self.model_kwargs = {"device": "cuda"}
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name, model_kwargs=self.model_kwargs)

        self.vectordb = Chroma.from_documents(documents=self.all_splits, embedding=self.embeddings, persist_directory="chroma_db")
        self.retriever = self.vectordb.as_retriever()

        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            use_cache=True,
            device_map="auto",
            max_length=2048,
            do_sample=True,
            top_k=5,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        self.llm = HuggingFacePipeline(pipeline=self.pipeline)

    def load_quantized_model(self, model_name):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            #load_in_4bit=True,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config
        )
        return model

    def initialize_tokenizer(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.bos_token_id = 1
        return tokenizer

    def load_documents(self):
        pdf_loader = PyPDFDirectoryLoader(self.folder_path)
        #text_loader = TextLoader(self.folder_path)

        pdf_documents = pdf_loader.load()
        #text_documents = text_loader.load()

        documents = pdf_documents
        return documents

# Usage Example
translation_model_name = "C:/Users/Atharav Jadhav/OneDrive/Desktop/Hackon/models/translation_model"  #Helsinki-NLP/opus-mt-en-hi
source_language_code = "hi"
target_language_code = "en"
model_name = "C:/Users/Atharav Jadhav/OneDrive/Desktop/Hackon/models/chatbot_model" #anakin87/zephyr-7b-alpha-sharded
folder_path = 'C:/Users/Atharav Jadhav/OneDrive/Desktop/Hackon/amazon_dataset/train_pdfs'
embedding_model_name = "sentence-transformers/all-mpnet-base-v2"

chatbot_training = ChatbotTraining(translation_model_name, source_language_code, target_language_code, model_name, folder_path, embedding_model_name)

chatbot_1 = Chatbot(
    translation_model_name, source_language_code, target_language_code,
    chatbot_training.model, chatbot_training.tokenizer,
    chatbot_training.retriever, chatbot_training.llm, [],
    image_classifier_model, feature_extractor
)

from fastapi import FastAPI, Form, File
from pydantic import BaseModel
import requests
import base64
from fastapi.middleware.cors import CORSMiddleware
import spacy # type: ignore

app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as necessary
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Response(BaseModel):
    response: str
    objects: list

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

@app.post("/chat_with_image", response_model=Response)
async def chat_with_image(prompt: str = Form(...), image: File = Form(...), use_hindi: bool = Form(False)):
    if image.value:
        image_bytes = image.value[0]['content']
        with open("uploaded_image.jpg", "wb") as f:
            f.write(image_bytes)
        image_path = "uploaded_image.jpg"

    if use_hindi:
        translated_prompt = chatbot_1.translate(prompt)
    else:
        translated_prompt = prompt

    response, chat_history = chatbot_1.create_conversation(translated_prompt, image_path)

    if use_hindi:
            translated_response = chatbot_1.translate(response)
    else:
            translated_response = response

    # Extract objects from the response text
    objects = extract_objects(translated_response)

    return Response(response=translated_response, objects=objects)

def extract_objects(text: str) -> list:
    doc = nlp(text)
    objects = []
    for ent in doc.noun_chunks:
        objects.append(ent.text)
    return objects

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)