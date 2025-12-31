"""
BERT-based Sentiment Analysis Inference Script
Performs sentiment classification on SST-5 dataset using a fine-tuned BERT model.

Usage:
    python inference.py --mode interactive
    python inference.py --mode batch --input file.txt --output results.txt
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path


# Configuration
CONFIG = {
    'model_name': 'bert-base-uncased',
    'num_labels': 5,
    'max_seq_length': 64,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # Path to saved model
    'model_path': 'finetuned-bert/bert_focal/best_model.pt',  # Update this path if needed
    
    # Label mapping
    'label_list': {
        0: "Very Negative",
        1: "Negative", 
        2: "Neutral",
        3: "Positive",
        4: "Very Positive"
    }
}


class SentimentAnalyzer:
    """
    Sentiment analyzer using fine-tuned BERT model
    """
    
    def __init__(self, model_path: str, config: dict):
        """
        Initialize the sentiment analyzer
        
        Args:
            model_path: Path to the saved model weights
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device(config['device'])
        
        # Load tokenizer
        print(f"Loading tokenizer: {config['model_name']}")
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        
        # Load model
        print(f"Loading model from: {model_path}")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config['model_name'],
            num_labels=config['num_labels']
        )
        
        # Load saved weights
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully on device: {self.device}")
    
    def predict(self, text: str) -> dict:
        """
        Predict sentiment for a single text
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary with prediction results
        """
        # Tokenize input
        encoding = self.tokenizer(
            text,
            max_length=self.config['max_seq_length'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
        
        # Extract results
        pred_label = torch.argmax(logits, dim=1).item()
        confidence = probs[0, pred_label].item()
        probabilities = probs[0].cpu().numpy()
        
        # Create result dictionary
        result = {
            'text': text,
            'predicted_label': pred_label,
            'predicted_sentiment': self.config['label_list'][pred_label],
            'confidence': confidence,
            'probabilities': {
                self.config['label_list'][i]: float(probabilities[i])
                for i in range(self.config['num_labels'])
            }
        }
        
        return result
    
    def predict_batch(self, texts: list) -> list:
        """
        Predict sentiment for multiple texts
        
        Args:
            texts: List of input texts
            
        Returns:
            List of prediction results
        """
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results


def print_result(result: dict, verbose: bool = True):
    """
    Print prediction result in a formatted way
    
    Args:
        result: Prediction result dictionary
        verbose: Whether to show detailed probabilities
    """
    print("\n" + "="*80)
    print(f"Text: {result['text']}")
    print(f"Predicted Sentiment: {result['predicted_sentiment']}")
    print(f"Confidence: {result['confidence']:.4f}")
    
    if verbose:
        print("\nProbability Distribution:")
        for label, prob in sorted(
            result['probabilities'].items(),
            key=lambda x: -x[1]
        ):
            bar_length = int(prob * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            print(f"  {label:15s}: {bar} {prob:.4f}")
    print("="*80)


def interactive_mode(analyzer: SentimentAnalyzer):
    """
    Interactive mode for single predictions
    
    Args:
        analyzer: SentimentAnalyzer instance
    """
    print("\n" + "="*80)
    print("SENTIMENT ANALYSIS - INTERACTIVE MODE")
    print("="*80)
    print("Enter sentences to analyze (type 'exit' to quit)\n")
    
    while True:
        try:
            text = input("Enter text: ").strip()
            
            if text.lower() == 'exit':
                print("Exiting...")
                break
            
            if not text:
                print("Please enter a valid text.\n")
                continue
            
            result = analyzer.predict(text)
            print_result(result, verbose=True)
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}\n")


def batch_mode(analyzer: SentimentAnalyzer, input_file: str, output_file: str):
    """
    Batch mode for processing file input
    
    Args:
        analyzer: SentimentAnalyzer instance
        input_file: Path to input file with sentences
        output_file: Path to output file for results
    """
    print(f"\nProcessing batch from: {input_file}")
    
    # Read input file
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return
    
    with open(input_file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(texts)} sentences to process...")
    
    # Process predictions
    results = analyzer.predict_batch(texts)
    
    # Write results to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, result in enumerate(results, 1):
            f.write(f"{i}. {result['text']}\n")
            f.write(f"   Sentiment: {result['predicted_sentiment']}\n")
            f.write(f"   Confidence: {result['confidence']:.4f}\n")
            f.write(f"   Probabilities:\n")
            for label, prob in sorted(
                result['probabilities'].items(),
                key=lambda x: -x[1]
            ):
                f.write(f"     {label:15s}: {prob:.4f}\n")
            f.write("\n")
    
    print(f"Results saved to: {output_file}")
    print(f"\nSummary:")
    for result in results[:5]:  # Show first 5 results
        print(f"  '{result['text'][:50]}...' -> {result['predicted_sentiment']}")
    if len(results) > 5:
        print(f"  ... and {len(results) - 5} more sentences")


def demo_mode(analyzer: SentimentAnalyzer):
    """
    Demo mode with predefined test sentences
    
    Args:
        analyzer: SentimentAnalyzer instance
    """
    print("\n" + "="*80)
    print("SENTIMENT ANALYSIS - DEMO MODE")
    print("="*80)
    
    test_sentences = [
        "This movie is not good at all, I didn't like it.",
        "I love this terrible movie!",
        "This is amazing!",
        "This is awful and terrible.",
        "The movie was okay, nothing special.",
        "I absolutely hate this masterpiece!",
        "This is the worst best movie ever made.",
    ]
    
    print(f"\nRunning demo with {len(test_sentences)} test sentences...\n")
    
    for i, text in enumerate(test_sentences, 1):
        result = analyzer.predict(text)
        print(f"{i}. {text}")
        print(f"   → {result['predicted_sentiment']} (confidence: {result['confidence']:.4f})\n")


def main():
    """
    Main function with argument parsing
    """
    parser = argparse.ArgumentParser(
        description="Sentiment Analysis Inference using Fine-tuned BERT"
    )
    parser.add_argument(
        '--mode',
        choices=['interactive', 'batch', 'demo'],
        default='interactive',
        help='Mode of operation (default: interactive)'
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Input file path for batch mode (one sentence per line)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='inference_results.txt',
        help='Output file path for batch mode (default: inference_results.txt)'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=CONFIG['model_path'],
        help='Path to the saved model weights'
    )
    parser.add_argument(
        '--device',
        choices=['cuda', 'cpu'],
        default=CONFIG['device'],
        help='Device to use for inference'
    )
    
    args = parser.parse_args()
    
    # Update config with command-line arguments
    CONFIG['device'] = args.device
    CONFIG['model_path'] = args.model_path
    
    # Initialize analyzer
    try:
        analyzer = SentimentAnalyzer(args.model_path, CONFIG)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure the model path is correct.")
        print("Expected model path structure:")
        print("  finetuned-bert/bert_focal/best_model.pt")
        return
    
    # Run selected mode
    if args.mode == 'interactive':
        interactive_mode(analyzer)
    
    elif args.mode == 'batch':
        if not args.input:
            print("Error: --input is required for batch mode")
            return
        batch_mode(analyzer, args.input, args.output)
    
    elif args.mode == 'demo':
        demo_mode(analyzer)


if __name__ == '__main__':
    main()
