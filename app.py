from flask import Flask, request, jsonify, render_template
import os
from PyPDF2 import PdfReader
from transformers import pipeline

app = Flask(__name__)
UPLOAD_BASE_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_BASE_FOLDER

# Ensure base upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize summarization model with a specific model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

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
    
    # Define folder paths
    year_folder = os.path.join(app.config['UPLOAD_FOLDER'], f'Year_{year}')
    semester_folder = os.path.join(year_folder, f'Semester_{semester}')
    
    # Search for the PDFs
    all_results = []

    # Iterate over all PDF files in the directory
    for filename in os.listdir(semester_folder):
        if filename.endswith('.pdf'):
            filepath = os.path.join(semester_folder, filename)
            answer = search_pdf(filepath, query)
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
            snippet_length = 500  # Length of context around the query
            results = []
            fallback_texts = []

            if not reader.pages:
                return "No pages found in the PDF."

            for page_num, page in enumerate(reader.pages):
                try:
                    text = page.extract_text()
                    if text is None:
                        continue

                    index = text.lower().find(query.lower())
                    if index != -1:
                        start_index = max(index - snippet_length // 2, 0)
                        end_index = min(index + snippet_length // 2, len(text))
                        snippet = text[start_index:end_index]
                        snippet = snippet.replace('\n', ' ').strip()
                        if len(snippet) > snippet_length:
                            snippet = snippet[:snippet_length] + '...'
                        results.append(f"Page {page_num + 1}: {snippet}")
                    else:
                        if len(text) > snippet_length:
                            fallback_texts.append(text[:snippet_length])

                except Exception as e:
                    return f"Error processing page {page_num + 1}: {str(e)}"

            if results:
                combined_results = " ".join(results)
                summary = summarizer(combined_results, max_length=200, min_length=50, do_sample=False)
                return summary[0]['summary_text']
            elif fallback_texts:
                combined_fallbacks = " ".join(fallback_texts)
                summary = summarizer(combined_fallbacks, max_length=200, min_length=50, do_sample=False)
                return "I couldn't find an exact match, but here's some related information: " + summary[0]['summary_text']
            return None  # No relevant information found
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

@app.route('/files', methods=['POST'])
def list_files():
    data = request.json
    year = data.get('year')
    semester = data.get('semester')
    
    if not year or not semester:
        return jsonify({'error': 'Year and semester are required'}), 400
    
    # Define folder paths
    year_folder = os.path.join(app.config['UPLOAD_FOLDER'], f'Year_{year}')
    semester_folder = os.path.join(year_folder, f'Semester_{semester}')
    
    if not os.path.exists(semester_folder):
        return jsonify({'error': 'Semester folder does not exist'}), 400

    pdf_files = [f for f in os.listdir(semester_folder) if f.endswith('.pdf')]
    return jsonify({'files': pdf_files}), 200

if __name__ == '__main__':
    app.run(debug=True)
