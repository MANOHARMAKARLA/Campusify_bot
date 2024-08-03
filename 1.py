from flask import Flask, request, jsonify, render_template
import os
from PyPDF2 import PdfReader
from transformers import pipeline
from spellchecker import SpellChecker
import torch

app = Flask(__name__)
UPLOAD_BASE_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_BASE_FOLDER

# Ensure base upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Check if GPU is available
device = 0 if torch.cuda.is_available() else -1

# Initialize summarization and QA models with GPU support if available
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)
qa_model = pipeline("question-answering", device=device)
spell = SpellChecker()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload_page():
    return render_template('upload.html')

@app.route('/chatbot')
def chatbot_page():
    return render_template('chatbot.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return {'error': 'No file part'}, 400
    file = request.files['file']
    if file.filename == '':
        return {'error': 'No selected file'}, 400
    
    year = request.form.get('year')
    semester = request.form.get('semester')
    
    if not year or not semester:
        return {'error': 'Year and semester are required'}, 400

    if file:
        filename = file.filename
        # Create year and semester folders if they don't exist
        year_folder = os.path.join(app.config['UPLOAD_FOLDER'], f'Year_{year}')
        semester_folder = os.path.join(year_folder, f'Semester_{semester}')
        os.makedirs(semester_folder, exist_ok=True)
        filepath = os.path.join(semester_folder, filename)
        file.save(filepath)
        
        return {'message': 'File uploaded successfully'}, 200

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    year = data.get('year')
    semester = data.get('semester')
    query = data.get('query')
    
    if not year or not semester or not query:
        return jsonify({'error': 'Year, semester, and query are required'}), 400
    
    # Correct spelling in the query
    corrected_query = " ".join([spell.correction(word) for word in query.split()])

    # Define folder paths
    year_folder = os.path.join(app.config['UPLOAD_FOLDER'], f'Year_{year}')
    semester_folder = os.path.join(year_folder, f'Semester_{semester}')
    
    all_results = []

    # Iterate over all PDF files in the directory
    for filename in os.listdir(semester_folder):
        if filename.endswith('.pdf'):
            filepath = os.path.join(semester_folder, filename)
            answer = search_pdf(filepath, corrected_query)
            if answer:
                all_results.append(f"Results from {filename}:\n{answer}")
    
    if all_results:
        return jsonify({'answer': "\n\n".join(all_results)}), 200
    else:
        return jsonify({'answer': 'No relevant information found.'}), 200

def search_pdf(filepath, query):
    try:
        with open(filepath, 'rb') as f:
            reader = PdfReader(f)
            if not reader.pages:
                return "No pages found in the PDF."
            
            text = ""
            for page in reader.pages:
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + " "
                except Exception as e:
                    return f"Error processing page: {str(e)}"
            
            # Correct spelling in the extracted text
            corrected_text = " ".join([spell.correction(word) for word in text.split()])

            # Use the QA model to find the answer
            qa_result = qa_model(question=query, context=corrected_text)
            answer = qa_result['answer']

            # Summarize the answer if it's too long
            if len(answer) > 200:
                summary = summarizer(answer, max_length=200, min_length=50, do_sample=False)
                return summary[0]['summary_text']
            else:
                return answer
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

@app.route('/files', methods=['POST'])
def list_files():
    data = request.json
    year = data.get('year')
    semester = data.get('semester')
    
    if not year or not semester:
        return jsonify({'error': 'Year and semester are required'}), 400
    
    year_folder = os.path.join(app.config['UPLOAD_FOLDER'], f'Year_{year}')
    semester_folder = os.path.join(year_folder, f'Semester_{semester}')
    
    if not os.path.exists(semester_folder):
        return jsonify({'error': 'Semester folder does not exist'}), 400

    pdf_files = [f for f in os.listdir(semester_folder) if f.endswith('.pdf')]
    return jsonify({'files': pdf_files}), 200

if __name__ == '__main__':
    app.run(debug=True)
