# Sentiment Analysis Web Application

This project is a web-based sentiment analysis tool developed using a FastAPI backend and a React frontend. It leverages a Hugging Face Transformers model to classify user-entered text into positive or negative sentiment, along with a confidence score. The interface is simple and responsive, designed for real-time interaction. Users can input a sentence, trigger the analysis with a click, and receive the result instantly.

The backend API is built with FastAPI and loads a pre-trained DistilBERT sentiment model. Optionally, a fine-tuned version can be used if available in the local model directory. A CLI utility is also included to fine-tune the model using custom labeled data in JSONL format. The frontend is built with React and communicates with the backend using a POST request to the `/predict` endpoint.

The application supports both manual local setup and containerized deployment using Docker and Docker Compose. Dependencies are managed using `requirements.txt` for Python and `package.json` for Node.js. The `.gitignore` file ensures that unnecessary files like `node_modules`, model outputs, and virtual environments are excluded from version control.

### Features

- Real-time sentiment prediction (positive/negative)
- Confidence score display
- Clean and responsive user interface
- Optional fine-tuning with custom data
- REST API integration with React frontend
- Docker support for easy deployment

### Technologies Used

- FastAPI
- Hugging Face Transformers
- React
- Python 3.10
- Node.js
- Docker & Docker Compose

### Local Setup Instructions

1. **Backend**
   - Navigate to `backend/`
   - Install dependencies: `pip install -r requirements.txt`
   - Start server: `uvicorn app:app --host 0.0.0.0 --port 8000`

2. **Frontend**
   - Navigate to `frontend/`
   - Install dependencies: `npm install`
   - Start frontend: `npm start`
   - Visit: `http://localhost:3000`

### Docker Setup (Optional)

```bash
docker-compose up --build
