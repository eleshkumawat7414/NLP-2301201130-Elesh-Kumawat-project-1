# NLP-2301201130-Elesh-Kumawat-project-1

Create a chatbot that answers student queries about university information such as admissions, fees, timetable, hostel, exams, etc.

Concepts Used

Text preprocessing (tokenization, stopword removal, lemmatization)

TF-IDF or Word Embeddings for query matching

Cosine similarity for retrieving the best answer

Tools

Python (NLTK, Scikit-learn, or spaCy)

Dataset: Self-created CSV with "Question" and "Answer"

Optional: Flask or Streamlit for web deployment

Example Dataset

Question

Answer

Admission fee is 5000.

How much is the

admission fee?

How can I apply for a hostel?

When will exams start?

Fill the hostel form online at hostel.university.edu.

Exams will begin in December as per the academic calendar.

Workflow

Preprocess FAQ dataset (tokenize, remove stopwords, lemmatize).

2. Preprocess user input query.

3. Compute similarity between user query and dataset questions.

4. Return the best matching answer.

Extensions

Use BERT embeddings for better semantic matching.

Add multi-language support for international students

Deploy on Telegram, WhatsApp, or a web portal
