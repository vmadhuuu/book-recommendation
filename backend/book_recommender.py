import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from fastapi import FastAPI
from difflib import get_close_matches
import uvicorn
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset

# Load and preprocess dataset
def prepare_data(file_path):
    df = pd.read_csv(file_path, on_bad_lines='skip', encoding='utf-8')
    
    # Normalize dataset titles
    df['title'] = df['title'].str.strip().str.lower()
    
    # Filter books with sufficient ratings
    df = df[df['ratings_count'] > 50]
    
    # Feature Selection
    df = df[['title', 'authors', 'average_rating', 'language_code', 'num_pages', 'ratings_count']]
    
    # Handle missing values
    df = df.dropna()
    
    # Encode categorical features
    encoder_title = LabelEncoder()
    encoder_authors = LabelEncoder()
    encoder_lang = LabelEncoder()
    
    df['title_encoded'] = encoder_title.fit_transform(df['title'])
    df['authors_encoded'] = encoder_authors.fit_transform(df['authors'])
    df['language_encoded'] = encoder_lang.fit_transform(df['language_code'])
    
    # Normalize numerical features
    scaler = MinMaxScaler()
    df[['average_rating', 'num_pages', 'ratings_count']] = scaler.fit_transform(
        df[['average_rating', 'num_pages', 'ratings_count']]
    )
    
    # Prepare feature matrix and labels
    X = df[['authors_encoded', 'average_rating', 'language_encoded', 
            'num_pages', 'ratings_count']].values
    y = df['title_encoded'].values
    
    return df, X, y, encoder_title

class BookRecommendationNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, num_books=10311):
        super(BookRecommendationNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, num_books)
        
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        x = self.dropout(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout(self.relu(self.bn2(self.fc2(x))))
        x = self.dropout(self.relu(self.bn3(self.fc3(x))))
        return self.fc4(x)

class BookRecommender:
    def __init__(self, model_path=None):
        self.df = None
        self.X = None
        self.y = None
        self.model = None
        self.encoder_title = None
        self.X_tensor = None
        self.y_tensor = None
        self.model_dir = os.getenv('MODEL_DIR', '/app/models')
        self.model_path = os.path.join(self.model_dir, 'best_model.pth')
        os.makedirs(self.model_dir, exist_ok=True)

    def save_model(self, model_state, is_best=False):
        """Save model state with proper path handling"""
        if is_best:
            save_path = self.model_path
        else:
            save_path = os.path.join(self.model_dir, f'checkpoint_model.pth')
        
        torch.save(model_state, save_path)
        print(f"Model saved to {save_path}")

    def load_model(self):
        """Load model with proper path handling"""
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))
            print(f"Loaded model from {self.model_path}")
            return True
        return False
        
    def prepare_data(self, file_path):
        self.df, self.X, self.y, self.encoder_title = prepare_data(file_path)
        self.X_tensor = torch.tensor(self.X, dtype=torch.float32)
        self.y_tensor = torch.tensor(self.y, dtype=torch.long)
        return self.df, self.X_tensor, self.y_tensor
    
    def create_model(self):
        input_dim = self.X.shape[1]
        num_books = len(self.df['title'].unique())
        self.model = BookRecommendationNN(input_dim, num_books=num_books)
        return self.model
    
    def train_model(self, epochs=100, batch_size=64):
        if self.model is None:
            self.create_model()
            
        # Calculate class weights
        class_counts = torch.bincount(self.y_tensor)
        weights = 1.0 / class_counts.float()
        weights = weights / weights.sum()
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
        
        # Optimizer setup
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=0.001,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Create data loader
        dataset = TensorDataset(self.X_tensor, self.y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        max_patience = 10
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            num_batches = 0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            scheduler.step(avg_loss)
            
            if epoch % 2 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
            
            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                # Save best model
                self.save_model(self.model.state_dict(), is_best=True)
            else:
                patience_counter += 1
                self.save_model(self.model.state_dict(), is_best=False)
                
            if patience_counter >= max_patience:
                print(f"Early stopping triggered at epoch {epoch}")
                # Load best model
                self.load_model()
                break
    
    def get_recommendations(self, book_title, n_recommendations=5):
        book_title = book_title.strip().lower()
        
        # Find closest matching book title
        available_titles = self.df['title'].str.lower().tolist()
        closest_match = get_close_matches(book_title, available_titles, n=1, cutoff=0.6)
        
        if not closest_match:
            return None
        
        # Get book details
        book_title = closest_match[0]
        book_index = self.df[self.df['title'].str.lower() == book_title].index[0]
        book_vector = torch.tensor([self.X[book_index]], dtype=torch.float32)
        
        self.model.eval()
        with torch.no_grad():
            similarities = torch.nn.functional.cosine_similarity(
                self.model(self.X_tensor),
                self.model(book_vector)
            )
        
        # Get top N recommendations
        top_indices = torch.argsort(similarities, descending=True)[1:n_recommendations+1].tolist()
        recommended_books = [self.df.iloc[int(i)]['title'] for i in top_indices]
        
        return {"book_title": book_title, "recommended_books": recommended_books}

# FastAPI app
app = FastAPI()
recommender = BookRecommender()

@app.on_event("startup")
async def startup_event():
    # Initialize the recommender system
    df, X_tensor, y_tensor = recommender.prepare_data('books.csv')
    recommender.create_model()
    # Load pre-trained model if exists
    try:
        recommender.model.load_state_dict(torch.load('best_model.pth'))
        print("Loaded pre-trained model")
    except:
        print("Training new model")
        recommender.train_model()

@app.get("/recommend/{book_title}")
def recommend_books(book_title: str, n_recommendations: int = 5):
    recommendations = recommender.get_recommendations(book_title, n_recommendations)
    if recommendations is None:
        return {"error": "Book not found in the dataset"}
    return recommendations

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)