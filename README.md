# 🧠 ResuMind AI

**AI-powered Resume Intelligence, Job Matching & Interview Assistant.**

ResuMind AI helps you analyze your resume, match it with job descriptions, identify skill gaps, and prepare for interviews using **Google Gemini AI** and **Hybrid RAG**.

## 🚀 Live Demo

👉 **[Try ResuMind AI](https://ai-resume-intelligence-ewmnhaehwreunf6wmgj55k.streamlit.app/)**

## ✨ Features

* 📄 **Resume Analysis** — strengths, skills, weaknesses & suggestions
* 🎯 **Job Match** — resume vs job description with match score
* 💬 **RAG Chat** — ask questions from your uploaded documents
* 🎤 **AI Interview** — generate questions & evaluate answers
* 🔎 **Hybrid Search** — FAISS + BM25 + Cross-Encoder
* 🤖 **Gemini AI** — intelligent, grounded responses

## 🛠️ Tech Stack

`Python` · `Streamlit` · `Google Gemini` · `FAISS` · `BM25` · `Sentence Transformers` · `PyPDF`

## ⚙️ Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
pip install -r requirements.txt
```

Create `.env`:

```env
GEMINI_API_KEY=your_api_key
```

Run:

```bash
streamlit run main.py
```

## ☁️ Deployment

Deploy on **Streamlit Community Cloud** and add `GEMINI_API_KEY` under **Secrets**.

⚠️ Never commit your API key or `.env` file.

## ⭐ Support

If you like **ResuMind AI**, give the repository a ⭐!

---

### 👨‍💻 Author

**Amit Kumar Singh**

> *Turn your resume into your career advantage.* 🚀
