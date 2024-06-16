from transformers import BertTokenizer, BertForQuestionAnswering
from flask import Flask, request, jsonify, render_template
import torch

# Initialize the Flask application
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

# Load the pre-trained BERT model and tokenizer for question answering
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

# Sample context for the chatbot to answer questions from
context = """
Lung cancer is a type of cancer that begins in the lungs, characterized by uncontrolled cell growth in the lung tissues. It is a significant health challenge in South Africa, with an estimated 8,000 new cases diagnosed annually. Lung cancer is the leading cause of cancer-related deaths among men and one of the top five cancers affecting women. The primary risk factor for lung cancer is smoking, with approximately 20% of the adult population identified as smokers, contributing to its high incidence. Environmental factors, such as exposure to asbestos and industrial pollutants, also play a role. Additionally, South Africa’s high HIV/AIDS prevalence, with about 13% of the adult population living with HIV, exacerbates the lung cancer burden, as immunocompromised individuals are at higher risk. Late-stage diagnosis is common due to limited access to healthcare services and inadequate screening programs, resulting in poorer outcomes. Treatment access is further hindered by the high costs associated with chemotherapy, radiation, and surgical interventions, which are often beyond the reach of many South Africans relying on the overburdened public healthcare system. Public health efforts, including anti-smoking campaigns and initiatives for early detection, are ongoing, but there is a pressing need for more comprehensive and accessible screening programs, along with enhanced support systems for patients and their families.
"""

@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Get the message from the POST request
        data = request.get_json()
        question = data['question']
        
        # Log the received question
        app.logger.info(f"Received question: {question}")
        
        # Tokenize the input message and context
        inputs = tokenizer.encode_plus(question, context, return_tensors='pt')

        # Get the model's output
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract the answer start and end logits
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

         # Get the most likely start and end token positions
        start_index = torch.argmax(start_logits, dim=-1).item()
        end_index = torch.argmax(end_logits, dim=-1).item()

        # Check if the indices are valid
        if start_index <= end_index and start_index < len(inputs['input_ids'][0]) and end_index < len(inputs['input_ids'][0]):
            # Convert token indices back to tokens
            input_ids = inputs['input_ids'].squeeze().tolist()
            answer_tokens = tokenizer.convert_ids_to_tokens(input_ids[start_index:end_index+1])

            # Clean the answer and convert it to lowercase
            answer = tokenizer.convert_tokens_to_string(answer_tokens).lower()
        else:
            answer = "I'm sorry, i don't have the information you are looking for."

        # Log the answer
        app.logger.info(f"Answer: {answer}")

        # Return the answer as a JSON response
        return jsonify({'answer': answer})
    except Exception as e:
        app.logger.error(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)