# 🧠 ML-Based MCQ Quiz Generator

This project is a Machine Learning-powered quiz generator designed to help students prepare for competitive exams like **JEE**. It uses **logistic regression** to analyze a dataset of previous years' questions and generate relevant, topic-specific multiple-choice questions.

Students can customize their quizzes by selecting the subject, chapter, or topic. The platform generates a 15-question quiz with a 30-minute timer and evaluates responses automatically.

---

## 🚀 Features

- ✅ Topic-based quiz generation using ML (Logistic Regression)
- ⏱️ 15 questions
- 📊 Automatic scoring: +4 for correct, -1 for wrong
- 📈 Post-quiz evaluation with score, accuracy, and time analytics
- 💬 HTML UI integrated with a Python backend for interactivity
- 🤖 AI-based chatbot (planned) for a smarter student experience

---

## 📁 Project Structure

- `index.html` – Home page to select quiz options  
- `quiz.html` – Displays quiz questions with timer  
- `results.html` – Auto evaluation and feedback  
- `app.py` – Python backend using Flask  
- `jee_mains_train2.csv` – Dataset of previous years' questions  
- `model.pkl` – Trained Logistic Regression model  

---

## 🧪 How It Works

1. User selects subject/chapter/topic
2. Model filters and selects 15 MCQs from dataset
3. Timer starts (30 minutes)
4. Upon submission, backend:
   - Calculates score
   - Computes accuracy
   - Displays detailed results and analysis

---

## 📦 Future Improvements

- Add support for other exams like **NEET**
- Build an intelligent, conversational **chatbot tutor**
- Enable student progress tracking across sessions
- Improve UI/UX for mobile responsiveness

---


## 🙋‍♀️ Author

**Aastha Baid and Sahithi G**  

---

## 🙏 Thank You!

Feel free to fork, contribute, and share this project to help more students prepare smarter. Good luck to all aspirants!
