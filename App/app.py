import os
from flask import Flask, render_template, request, jsonify
from langchain_community.embeddings import HuggingFaceEmbeddings
from llama_index.core import VectorStoreIndex, ServiceContext, Document
from llama_index.llms.openai import OpenAI
from llama_index.core import SimpleDirectoryReader
from gtts import gTTS
import base64
import re
import speech_recognition as sr
from llama_index.core.prompts.base import ChatPromptTemplate
import openai
from flask import send_file

app = Flask(__name__)
openai.openai_key = os.getenv("OPENAI_API_KEY")
messages = []
history = []

@app.route("/")
def index():
    return render_template("index.html", messages=messages, history=history)

@app.route("/chat", methods=["POST"])
def chat():
    user_question = request.json["user_question"]
    response_text, additional_questions, audio_data, document_session = generate_response(user_question)
    appendMessage('user', user_question)
    appendMessage('assistant', response_text, type='response')

    # Append additional questions to the messages
    if additional_questions:
        for i, question in enumerate(additional_questions, start=1):
            appendMessage("user" , additional_questions)
            appendMessage('assistant', f"Additional Question {i}: {question}", type='additional_question')

    return jsonify({"response_text": response_text, "additional_questions": additional_questions, "audio_data": audio_data, "document_session": document_session})

def load_data():
    reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
    docs = reader.load_data()
    
    llm = OpenAI(model="gpt-3.5-turbo", temperature="0.1", systemprompt="""Use the books in data file is source for the answer.Generate a valid 
                 and relevant answer to a query related to 
                 construction problems, ensure the answer is based strictly on the content of 
                 the book and not influenced by other sources. Do not hallucinate. The answer should 
                 be informative and fact-based. """)
    service_content = ServiceContext.from_defaults(llm=llm)
    index = VectorStoreIndex.from_documents(docs, service_context=service_content)
    return index

def query_chatbot(query_engine, user_question):
    response = query_engine.query(user_question)
    return response.response if response else None
def initialize_chatbot(data_dir="./data", model="gpt-3.5-turbo", temperature=0.4):
    documents = SimpleDirectoryReader(data_dir).load_data()
    llm = OpenAI(model=model, temperature=temperature)

    additional_questions_prompt_str = (
        "Given the context below, generate only one additional question different from previous additional questions related to the user's query:\n"
        "Context:\n"
        "User Query: {query_str}\n"
        "Chatbot Response: \n"
    )

    new_context_prompt_str = (
        "We have the opportunity to only one generate additional question different from previous additional questions based on new context.\n"
        "New Context:\n"
        "User Query: {query_str}\n"
        "Chatbot Response: \n"
        "Given the new context, generate only one additional questions different at each time from previous additional questions related to the user's query."
        "If the context isn't useful, generate only one additional questions different at each from previous time from previous additional questions based on the original context.\n"
    )

    chat_text_qa_msgs = [
        (
            "system",
            """Generate only one  additional questions  that facilitate deeper exploration of the main topic 
            discussed in the user's query and the chatbot's response. The questions should be relevant and
              insightful, encouraging further discussion and exploration of the topic. Keep the questions concise 
              and focused on different aspects of the main topic to provide a comprehensive understanding.""",
        ),
        ("user", additional_questions_prompt_str),
    ]
    text_qa_template = ChatPromptTemplate.from_messages(chat_text_qa_msgs)

    
    chat_refine_msgs = [
        (
            "system",
            """Based on the user's question '{prompt}' and the chatbot's response '{response}', please 
            generate only one additional questions related to the main topic. The questions should be 
            insightful and encourage further exploration of the main topic, providing a more comprehensive 
            understanding of the subject matter.""",
        ),
        ("user", new_context_prompt_str),
    ]
    refine_template = ChatPromptTemplate.from_messages(chat_refine_msgs)
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(
        text_qa_template=text_qa_template,
        refine_template=refine_template,
        llm=llm,
    )

    return query_engine

def generate_response(user_question):
    index = load_data()
    chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

    response = chat_engine.chat(user_question)
    if response:
        response_text = response.response

        tts = gTTS(text=response_text, lang='en')
        tts.save('output.wav')

        # Read the audio file and convert it to base64 encoding
        with open('output.wav', 'rb') as audio_file:
            audio_data = base64.b64encode(audio_file.read()).decode('utf-8')

        # Generate additional questions
        additional_questions = generate_additional_questions(response_text)

        # Generate document session if available
        document_session = extract_document_section(response_text)

        return response_text, additional_questions,audio_data, document_session

    return None, None, None, None  # Return placeholders if no response


def generate_additional_questions( user_question):
    
    # Use the chat engine to generate additional questions
    additional_questions = []
    words = ["apple" , "mango" , "orange"]
    for word in words :
        question = query_chatbot(initialize_chatbot(), user_question)
        additional_questions.append(question if question else None)
        

    return additional_questions

def appendMessage(role, message, type='message'):
    messages.append({"role": role, "content": message, "type": type})
    history.append({"role": role, "content": message, "type": type})

@app.route("/history", methods=["GET"])
def get_history():
    return jsonify({"history": history})

@app.route("/voice_command", methods=["POST"])
def voice_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    
    try:
        user_question = recognizer.recognize_google(audio)
        print(f"User Command: {user_question}")
        response_text, additional_questions, audio_data = generate_response(user_question)
        appendMessage('user', user_question)
        appendMessage('assistant', response_text, type='response')

        # Append additional questions to the messages
        if additional_questions:
            for i, question in enumerate(additional_questions, start=1):
                appendMessage('assistant', f"Additional Question {i}: {question}", type='additional_question')

        return jsonify({"response_text": response_text, "additional_questions": additional_questions, "audio_data": audio_data})
    except sr.UnknownValueError:
        print("Could not understand audio")
        return jsonify({"error": "Could not understand audio"})
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return jsonify({"error": f"Could not request results from Google Speech Recognition service; {e}"})


def extract_document_section(response_text):
    pattern = r'\[SECTION_START\](.*?)\[SECTION_END\]'
    match = re.search(pattern, response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return "No session was found"

if __name__ == "__main__":
    app.run( debug= True)