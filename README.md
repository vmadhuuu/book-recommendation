# AI-Powered Book Recommendation System

A deep learning-based book recommendation system deployed on AWS, using PyTorch, FastAPI, React and Dockerized microservices for scalable, real-time recommendations.  


## Project Overview
This project builds an AI-driven book recommendation system that processes 10K+ books and provides personalized recommendations based on user preferences. The system is deployed using AWS ECS, with an optimized backend powered by **FastAPI** and a frontend built with **React and Tailwind CSS**.  

### Key Features: 
- Deep Learning Model: Multi-layer **fully connected neural network** with **batch normalization, dropout, and LeakyReLU activation**, trained with **AdamW optimizer** and **learning rate scheduling**.
- Automated Data Processing: Feature engineering, normalization, and encoding of categorical data (authors, language, ratings, etc.).
- Handling Imbalanced Data: Weighted loss function and **class-aware sampling** to improve predictions for underrepresented books.
- Scalable Microservices Architecture: Backend API using **FastAPI**, frontend with **React**, and **Nginx as a reverse proxy**.
- CI/CD Pipeline: Deployed using **AWS CodeBuild, Docker, and ECS**, reducing deployment time by **80%**.
- Multi-Container Orchestration: Docker Compose for seamless **integration between backend, frontend, and Nginx**.
- Cloud-Native Deployment: AWS-based architecture with **model checkpointing for automated retraining**.  

---

## Tech Stack
| **Component**  | **Technology Used**  |
|---------------|----------------------|
| **Deep Learning**  | PyTorch, NumPy, Scikit-Learn  |
| **Web Framework**  | FastAPI  |
| **Frontend**  | React, Tailwind CSS  |
| **Reverse Proxy**  | Nginx  |
| **Deployment**  | AWS ECS, AWS CodeBuild, Docker, Docker Compose  |
| **CI/CD**  | GitHub Actions, AWS CodePipeline  |

---

## Configuration Files
- `buildspec.yml`: AWS CodeBuild configuration
- `docker-compose.yml`: Local container orchestration
- `requirements.txt`: Python dependencies
- `package.json`: Node.js dependencies

## **🚀 Installation & Setup**  

###  Prerequisites
Ensure you have the following installed:  
- Python 3.8+  
- Docker & Docker Compose  
- Node.js & npm/yarn  
- AWS CLI (if deploying to AWS)  

### **Clone the Repository**  
```sh
git clone https://github.com/yourusername/book-recommendation-system.git
cd book-recommendation-system
```

### **Backend Setup**  
```sh
cd backend
pip install -r requirements.txt
python book_recommender.py
```

### **Frontend Setup**  
```sh
cd frontend
npm install
npm start
```

### **Running with Docker**  
To run the full stack using Docker Compose:  
```sh
docker-compose up --build
```

---

## **API Endpoints**  

| **Endpoint**  | **Method**  | **Description**  |
|--------------|------------|------------------|
| `/recommend/{book_title}`  | GET  | Returns a list of recommended books based on the input title. |
| `/health`  | GET  | Health check for the API. |

---

## **Model Training & Evaluation**  
- The model is trained using **CrossEntropyLoss with label smoothing (0.1)**.  
- Learning rate is adjusted using **ReduceLROnPlateau scheduler**.  
- **Early stopping** is implemented to prevent overfitting.  
- Model is saved with **checkpointing for retraining**.  

---

## **Deployment on AWS ECS**  
To deploy the application on AWS:  
```sh
aws ecr create-repository --repository-name book-recommendation
docker build -t book-recommendation .
docker tag book-recommendation:latest <aws-account-id>.dkr.ecr.<region>.amazonaws.com/book-recommendation:latest
docker push <aws-account-id>.dkr.ecr.<region>.amazonaws.com/book-recommendation:latest
```
Use `awscli` and `CodeBuild` to automate deployments.

---

## **Future Enhancements**  
- **Improve recommendation accuracy with Transformer-based models** (e.g., BERT embeddings).
- **User-based collaborative filtering** for more personalized recommendations.
- **Deploy using Kubernetes (EKS) for enhanced scalability**.  




