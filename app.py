import pandas as pd
import numpy as np
import json
import joblib
import re
import os
import traceback
import ast
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

app = Flask(__name__)

# Configuration
DATA_FILE = "jee_mains_train2.csv"
MODEL_DIR = str(Path.home() / "quiz_models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATHS = {
    'vectorizer': os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"),
    'model': os.path.join(MODEL_DIR, "logreg_model.pkl"),
    'encoder': os.path.join(MODEL_DIR, "label_encoder.pkl"),
    'metadata': os.path.join(MODEL_DIR, "model_metadata.json")
}

# Text preprocessing
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Model training
def train_model():
    try:
        print("\n=== Training Question Classifier ===")
        
        df = pd.read_csv(DATA_FILE)
        df = df.dropna(subset=['question', 'subject', 'chapter'])
        df = df[df['subject'].str.strip() != '']
        
        # Create target variable
        df['target'] = df['subject'] + '|' + df['chapter'].astype(str)
        
        # Filter rare classes
        class_counts = df['target'].value_counts()
        valid_classes = class_counts[class_counts >= 5].index
        df = df[df['target'].isin(valid_classes)]
        
        # Preprocess questions
        df['clean_question'] = df['question'].apply(preprocess_text)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['clean_question'], df['target'], 
            test_size=0.2, random_state=42, stratify=df['target']
        )
        
        # Build pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', LogisticRegression(class_weight='balanced', max_iter=1000))
        ])
        
        # Hyperparameter tuning
        param_grid = {
            'tfidf__max_features': [5000, 10000],
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'clf__C': [0.1, 1, 10]
        }
        
        grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        
        # Save best model
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        joblib.dump(best_model.named_steps['tfidf'], MODEL_PATHS['vectorizer'])
        joblib.dump(best_model.named_steps['clf'], MODEL_PATHS['model'])
        
        # Save encoder and metadata
        label_encoder = LabelEncoder().fit(df['target'])
        joblib.dump(label_encoder, MODEL_PATHS['encoder'])
        
        metadata = {
            'accuracy': accuracy,
            'best_params': grid_search.best_params_,
            'classes': label_encoder.classes_.tolist(),
            'training_date': pd.Timestamp.now().isoformat()
        }
        with open(MODEL_PATHS['metadata'], 'w') as f:
            json.dump(metadata, f)
            
        print(f"\nTraining complete. Model Accuracy: {accuracy:.2%}\n")
        return True
    except Exception as e:
        print(f"\nTraining failed: {str(e)}\n")
        traceback.print_exc()
        return False

# Model loading
def load_models():
    try:
        vectorizer = joblib.load(MODEL_PATHS['vectorizer'])
        model = joblib.load(MODEL_PATHS['model'])
        encoder = joblib.load(MODEL_PATHS['encoder'])
        metadata = json.load(open(MODEL_PATHS['metadata']))
        
        print(f"\nModel loaded successfully. Accuracy: {metadata['accuracy']:.2%}\n")
        return vectorizer, model, encoder, metadata
    except Exception as e:
        print(f"\nModel loading failed: {str(e)}\n")
        traceback.print_exc()
        return None, None, None, None

def parse_options(options_str):
    """Parse options from string to list of dictionaries"""
    try:
        if isinstance(options_str, str):
            # Try to parse as JSON first
            try:
                options = json.loads(options_str)
                if isinstance(options, list):
                    return options
            except json.JSONDecodeError:
                # If JSON parsing fails, try ast.literal_eval
                try:
                    options = ast.literal_eval(options_str)
                    if isinstance(options, list):
                        return options
                except:
                    pass
        
        # If all parsing fails, return default options
        return [
            {"identifier": "A", "content": "Option A"},
            {"identifier": "B", "content": "Option B"},
            {"identifier": "C", "content": "Option C"},
            {"identifier": "D", "content": "Option D"}
        ]
    except Exception as e:
        print(f"Error parsing options: {str(e)}")
        return [
            {"identifier": "A", "content": "Option A"},
            {"identifier": "B", "content": "Option B"},
            {"identifier": "C", "content": "Option C"},
            {"identifier": "D", "content": "Option D"}
        ]

def parse_correct_option(correct_option):
    """Parse correct option from various formats"""
    try:
        if isinstance(correct_option, str):
            # Handle JSON array format
            if correct_option.startswith('[') and correct_option.endswith(']'):
                correct = json.loads(correct_option)
                if isinstance(correct, list) and len(correct) > 0:
                    return str(correct[0]).strip().upper()[0]
            
            # Handle string format
            return str(correct_option).strip().upper()[0]
        
        # Handle list format
        if isinstance(correct_option, list) and len(correct_option) > 0:
            return str(correct_option[0]).strip().upper()[0]
        
        # Default to A if parsing fails
        return 'A'
    except:
        return 'A'

def load_questions(file_path, subject=None, chapters=None):
    """Load questions with enhanced filtering and validation"""
    try:
        df = pd.read_csv(file_path)
        print(f"\nLoaded {len(df)} questions from {file_path}\n")
        
        # Basic data validation
        required_columns = ['question', 'options', 'correct_option', 'subject', 'chapter']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Filter by subject if specified
        if subject and subject.lower() not in ["all", "random"]:
            df = df[df['subject'].str.lower() == subject.lower()]
        
        # Filter by chapters if specified
        if chapters and "all" not in chapters and "random" not in chapters:
            df = df[df['chapter'].isin(chapters)]
        
        # Clean data
        df = df.dropna(subset=['question', 'options', 'correct_option'])
        
        # Sample questions (max 15)
        question_count = min(15, len(df))
        if question_count == 0:
            return []
            
        questions = df.sample(n=question_count).to_dict('records')
        
        # Process each question
        processed_questions = []
        for idx, q in enumerate(questions):
            try:
                # Parse options
                options = parse_options(q['options'])
                
                # Get correct option
                correct_option = parse_correct_option(q['correct_option'])
                
                processed_questions.append({
                    'id': idx,
                    'question': q['question'],
                    'options': options,
                    'correct_option': correct_option,
                    'explanation': q.get('explanation', 'No explanation available'),
                    'subject': q.get('subject', 'Unknown'),
                    'chapter': str(q.get('chapter', 'Unknown'))
                })
            except Exception as e:
                print(f"Error processing question {idx}: {str(e)}")
                continue
                
        return processed_questions
        
    except Exception as e:
        print(f"\nError loading questions: {str(e)}\n")
        traceback.print_exc()
        return []

def get_subjects():
    try:
        df = pd.read_csv(DATA_FILE)
        subjects = df['subject'].dropna().unique().tolist()
        return ['all'] + sorted(subjects)
    except Exception as e:
        print(f"\nError getting subjects: {str(e)}\n")
        return ['all', 'Physics', 'Chemistry', 'Mathematics']

def get_chapters(subject):
    try:
        df = pd.read_csv(DATA_FILE)
        if subject.lower() in ["all", "random"]:
            chapters = df['chapter'].dropna().unique().tolist()
        else:
            chapters = df[df['subject'].str.lower() == subject.lower()]['chapter'].dropna().unique().tolist()
        return ['all'] + sorted(chapters)
    except Exception as e:
        print(f"\nError getting chapters: {str(e)}\n")
        return ['all']

@app.route('/')
def index():
    try:
        model_status = "Model not loaded"
        accuracy = 0
        
        # Check if model exists
        model_exists = all(os.path.exists(path) for path in MODEL_PATHS.values())
        
        if not model_exists:
            print("\nInitiating model training...")
            if train_model():
                _, _, _, metadata = load_models()
                accuracy = metadata.get('accuracy', 0)
                model_status = f"New model trained (Accuracy: {accuracy:.2%})"
            else:
                model_status = "Model training failed"
        else:
            _, _, _, metadata = load_models()
            accuracy = metadata.get('accuracy', 0) if metadata else 0
            model_status = f"Pre-trained model loaded (Accuracy: {accuracy:.2%})"
        
        return render_template('index.html',
                            subjects=get_subjects(),
                            all_chapters=get_chapters("all"),
                            model_status=model_status)
    except Exception as e:
        print(f"\nIndex route error: {str(e)}\n")
        return render_template('index.html', 
                            error_message="System initialization failed",
                            subjects=['all'],
                            all_chapters=['all'])

@app.route('/get_chapters', methods=['POST'])
def get_chapters_route():
    try:
        data = request.get_json()
        subject = data.get('subject', 'all')
        return jsonify({"chapters": get_chapters(subject)})
    except Exception as e:
        print(f"\nChapter route error: {str(e)}\n")
        return jsonify({"error": str(e)}), 500

@app.route('/start_quiz', methods=['POST'])
def start_quiz():
    try:
        subject = request.form.get('subject', 'all')
        chapters = request.form.getlist('chapters')
        
        print(f"\nStarting quiz: Subject={subject}, Chapters={chapters}\n")
        
        # Load questions
        quiz_data = load_questions(DATA_FILE, subject, chapters)
        if not quiz_data:
            return render_template('index.html',
                                error_message="No questions found for selected criteria",
                                subjects=get_subjects(),
                                all_chapters=get_chapters("all"))
        
        return render_template('quiz.html',
                            questions=quiz_data,
                            questions_json=json.dumps(quiz_data))
        
    except Exception as e:
        print(f"\nQuiz start error: {str(e)}\n")
        return render_template('index.html',
                            error_message=f"Error generating quiz: {str(e)}",
                            subjects=get_subjects(),
                            all_chapters=get_chapters("all"))

@app.route('/evaluate_quiz', methods=['POST'])
def evaluate_quiz():
    try:
        # Get user answers
        user_answers = {}
        for key, value in request.form.items():
            if key.startswith('question_'):
                q_id = key.split('_')[1]
                user_answers[q_id] = value.upper()[0] if value else ''  # Take first character only
        
        # Get original questions
        questions_data = request.form.get('questions_data', '[]')
        questions = json.loads(questions_data)
        
        results = []
        score = 0
        correct_count = 0
        attempted = 0
        
        for q in questions:
            q_id = str(q['id'])
            user_answer = user_answers.get(q_id, '').strip()
            correct_answer = q['correct_option'].strip().upper()[0]  # Take first character
            
            # Calculate score
            is_correct = False
            if user_answer:
                attempted += 1
                if user_answer == correct_answer:
                    score += 4
                    correct_count += 1
                    is_correct = True
                else:
                    score -= 1
            
            # Prepare answer texts
            user_answer_text = "Not attempted"
            if user_answer:
                option = next((opt for opt in q['options'] if opt['identifier'] == user_answer), None)
                user_answer_text = f"{user_answer}: {option['content']}" if option else "Invalid option"
            
            correct_option = next((opt for opt in q['options'] if opt['identifier'] == correct_answer), None)
            correct_answer_text = f"{correct_answer}: {correct_option['content']}" if correct_option else "Answer not found"
            
            results.append({
                'question': q['question'],
                'options': q['options'],
                'user_answer': user_answer_text,
                'correct_answer': correct_answer_text,
                'is_correct': is_correct,
                'explanation': q['explanation'],
                'subject': q['subject'],
                'chapter': q['chapter']
            })
        
        total = len(questions)
        accuracy = (correct_count / total) * 100 if total > 0 else 0
        
        print(f"\nQuiz Evaluation Results:")
        print(f"Score: {score}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Correct: {correct_count}/{total}")
        print(f"Attempted: {attempted}/{total}\n")
        
        return render_template('results.html',
                            results=results,
                            score=score,
                            accuracy=round(accuracy, 2),
                            correct_responses=correct_count,
                            attempted=attempted,
                            total_questions=total)
        
    except Exception as e:
        print(f"\nEvaluation error: {str(e)}\n")
        return render_template('index.html',
                            error_message=f"Error evaluating quiz: {str(e)}",
                            subjects=get_subjects(),
                            all_chapters=get_chapters("all"))

if __name__ == "__main__":
    print("\n=== Starting JEE Quiz Application ===")
    app.run(host='0.0.0.0', port=5001, debug=True)