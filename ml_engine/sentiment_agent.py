import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax

class SentimentAgent:
    def __init__(self):
        """
        Initializes the FinBERT model. 
        We use 'ProsusAI/finbert' which is fine-tuned for financial contexts.
        """
        self.model_name = "ProsusAI/finbert"
        print(f"⏳ Loading Sentiment Agent ({self.model_name})...")
        
        # Load pre-trained model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertForSequenceClassification.from_pretrained(self.model_name)
        
        # Move to GPU if available (optional, usually CPU is fast enough for inference)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval() # Set to evaluation mode
        
        print(f"✅ Sentiment Agent Ready on {self.device}")

    def get_sentiment(self, text):
        """
        Analyzes a single headline.
        Returns: 
          - label (str): 'positive', 'negative', 'neutral'
          - score (float): A continuous value from -1 (Very Neg) to +1 (Very Pos)
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = softmax(logits, dim=1).cpu().numpy()[0] # [Prob_Pos, Prob_Neg, Prob_Neut]
            
            # FinBERT labels are usually: 0: Positive, 1: Negative, 2: Neutral 
            # (Check specific model config if unsure, but ProsusAI standard is Pos/Neg/Neu or similar)
            # Actually ProsusAI/finbert config is: {0: 'positive', 1: 'negative', 2: 'neutral'}
            labels = self.model.config.id2label
            
            # Get the highest probability label
            predicted_id = np.argmax(probs)
            predicted_label = labels[predicted_id]
            
            # CALCULATE COMPOSITE SCORE (Innovation)
            # Instead of just "Positive", we want a number for the Math Model.
            # Score = Prob(Positive) - Prob(Negative)
            # Range: -1.0 to +1.0
            # Example: 90% Pos, 10% Neg -> 0.9 - 0.1 = +0.8
            
            # Map probabilities correctly based on model config
            # We need to find which index corresponds to 'positive' and 'negative'
            pos_idx = -1
            neg_idx = -1
            
            for idx, label in labels.items():
                if label == 'positive': pos_idx = idx
                elif label == 'negative': neg_idx = idx
            
            score = probs[pos_idx] - probs[neg_idx]

        return predicted_label, float(score), probs

    def analyze_daily_headlines(self, headlines_list):
        """
        Aggregates sentiment for a whole day.
        """
        if not headlines_list:
            return "neutral", 0.0
        
        total_score = 0
        
        print(f"\n🔍 Analyzing {len(headlines_list)} headlines...")
        
        for headline in headlines_list:
            label, score, _ = self.get_sentiment(headline)
            print(f"   📄 '{headline[:50]}...' -> {label.upper()} (Score: {score:.4f})")
            total_score += score
            
        # Average score for the day
        avg_score = total_score / len(headlines_list)
        
        # Determine final daily label
        if avg_score > 0.15: final_label = "bullish"
        elif avg_score < -0.15: final_label = "bearish"
        else: final_label = "neutral"
        
        return final_label, avg_score