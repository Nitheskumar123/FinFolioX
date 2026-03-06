import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
from ml_engine.mcp_server import MCPDataServer

class SentimentAgent:
    def __init__(self):
        """
        Initializes the FinBERT model and the MCP Data Router.
        """
        self.model_name = "ProsusAI/finbert"
        print(f"⏳ Loading Sentiment Agent ({self.model_name})...")

        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertForSequenceClassification.from_pretrained(self.model_name)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # Initialize the MCP Server connection
        self.mcp_server = MCPDataServer()

        print(f"✅ Sentiment Agent Ready on {self.device}")

    def get_sentiment(self, text):
        """Analyzes a single string of text."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = softmax(logits, dim=1).cpu().numpy()[0]

            labels = self.model.config.id2label

            predicted_id = int(np.argmax(probs))
            predicted_label = labels[predicted_id]

            # Defensive label mapping to prevent id2label crashes
            pos_idx, neg_idx = -1, -1
            for idx, label in labels.items():
                if label.lower() == "positive": pos_idx = idx
                elif label.lower() == "negative": neg_idx = idx
            
            # Safety fallback if model config unexpectedly changes
            if pos_idx == -1 or neg_idx == -1:
                return "neutral", 0.0, probs

            score = float(probs[pos_idx] - probs[neg_idx])

        return predicted_label, score, probs

    def analyze_with_mcp(self, ticker):
        """
        Phase 22 Pipeline:
        1. Requests contextual payload from MCP.
        2. Scores each item with FinBERT.
        3. Multiplies FinBERT score by the MCP Tier Weight (SEC vs Reddit).
        """
        mcp_payload = self.mcp_server.get_global_context_payload(ticker)
        
        weighted_scores = []
        total_weight_applied = 0.0

        print("\n🔍 [FinBERT] Processing MCP Intelligence Payload...")

        for item in mcp_payload:
            source = item['source']
            text = item['text']
            tier_weight = item['tier_weight']

            # Short text guard to prevent FinBERT garbage output
            if len(text.strip()) < 10:
                continue

            label, raw_score, probs = self.get_sentiment(text)
            confidence = float(np.max(probs))

            # Reduce weight if FinBERT is unsure
            confidence_multiplier = confidence if confidence >= 0.65 else 0.30
            final_item_weight = tier_weight * confidence_multiplier
            
            adjusted_score = raw_score * final_item_weight

            weighted_scores.append(adjusted_score)
            total_weight_applied += final_item_weight

            print(f"   🏛️ [{source}] (Tier Weight: {tier_weight:.1f})")
            print(f"      📄 '{text[:60]}...' -> {label.upper()} (Raw: {raw_score:.2f})")

        # Safe floating point zero-division check
        if total_weight_applied < 1e-6:
            final_score = 0.0
        else:
            final_score = sum(weighted_scores) / total_weight_applied

        # Hard clamp
        final_score = float(np.clip(final_score, -0.75, 0.75))

        if final_score > 0.15:
            final_label = "bullish"
        elif final_score < -0.15:
            final_label = "bearish"
        else:
            final_label = "neutral"

        return final_label, final_score