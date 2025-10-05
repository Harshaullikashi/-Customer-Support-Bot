"""
Customer Support Bot with Document Training
A comprehensive agentic workflow that trains on documents and provides intelligent responses
with feedback-based improvement capabilities.
"""

import logging
import random
import os
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import re

# Core ML libraries
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np

# Document processing
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SupportBotAgent:
    """
    Advanced Customer Support Bot with document training and feedback-based learning.
    
    Features:
    - Document processing (PDF and text files)
    - Semantic search using sentence transformers
    - Question-answering with transformer models
    - Feedback simulation and response adjustment
    - Comprehensive logging and decision tracking
    - Fallback handling for out-of-scope queries
    """
    
    def __init__(self, document_path: str, model_name: str = "distilbert-base-uncased-distilled-squad"):
        """
        Initialize the SupportBotAgent.
        
        Args:
            document_path: Path to the training document (PDF or TXT)
            model_name: Hugging Face model for question answering
        """
        self.document_path = document_path
        self.model_name = model_name
        self.document_text = ""
        self.sections = []
        self.section_embeddings = None
        self.qa_model = None
        self.embedder = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
        # Feedback tracking
        self.feedback_history = []
        self.response_history = []
        self.improvement_strategies = {
            "not helpful": "rephrase_with_context",
            "too vague": "add_details",
            "unclear": "simplify_explanation",
            "good": "maintain_response"
        }
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize models and process document
        self._initialize_models()
        self._load_and_process_document()
        
        logging.info(f"SupportBotAgent initialized successfully with document: {document_path}")
    
    def _setup_logging(self):
        """Set up comprehensive logging system."""
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # Configure logging
        log_filename = f"logs/support_bot_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()  # Also log to console
            ]
        )
        
        self.log_file = log_filename
        logging.info("Logging system initialized")
    
    def _initialize_models(self):
        """Initialize NLP models for question answering and embeddings."""
        try:
            logging.info("Initializing NLP models...")
            
            # Initialize question-answering model
            self.qa_model = pipeline(
                "question-answering",
                model=self.model_name,
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Initialize sentence transformer for embeddings
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize TF-IDF vectorizer for keyword matching
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            logging.info("NLP models initialized successfully")
            
        except Exception as e:
            logging.error(f"Error initializing models: {str(e)}")
            raise
    
    def _load_and_process_document(self):
        """Load and preprocess the training document."""
        try:
            logging.info(f"Loading document: {self.document_path}")
            
            # Determine file type and load accordingly
            if self.document_path.lower().endswith('.pdf'):
                self.document_text = self._load_pdf(self.document_path)
            else:
                self.document_text = self._load_text(self.document_path)
            
            # Clean and preprocess text
            self.document_text = self._clean_text(self.document_text)
            
            # Split into sections
            self.sections = self._split_into_sections(self.document_text)
            
            # Create embeddings
            self._create_embeddings()
            
            # Create TF-IDF matrix for keyword matching
            self._create_tfidf_matrix()
            
            logging.info(f"Document processed successfully. Found {len(self.sections)} sections")
            
        except Exception as e:
            logging.error(f"Error processing document: {str(e)}")
            raise
    
    def _load_pdf(self, path: str) -> str:
        """Load text from PDF file."""
        try:
            with open(path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logging.error(f"Error loading PDF: {str(e)}")
            raise
    
    def _load_text(self, path: str) -> str:
        """Load text from text file."""
        try:
            with open(path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logging.error(f"Error loading text file: {str(e)}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\:\;\-\(\)]', '', text)
        return text.strip()
    
    def _split_into_sections(self, text: str) -> List[str]:
        """Split document into meaningful sections."""
        # Split by double newlines first
        sections = [section.strip() for section in text.split('\n\n') if section.strip()]
        
        # Further split long sections by sentences
        final_sections = []
        for section in sections:
            if len(section) > 500:  # If section is too long, split by sentences
                sentences = re.split(r'[.!?]+', section)
                current_chunk = ""
                for sentence in sentences:
                    if len(current_chunk + sentence) > 300:
                        if current_chunk.strip():
                            final_sections.append(current_chunk.strip())
                        current_chunk = sentence
                    else:
                        current_chunk += sentence + ". "
                if current_chunk.strip():
                    final_sections.append(current_chunk.strip())
            else:
                final_sections.append(section)
        
        return [section for section in final_sections if len(section) > 50]
    
    def _create_embeddings(self):
        """Create embeddings for document sections."""
        try:
            logging.info("Creating document embeddings...")
            self.section_embeddings = self.embedder.encode(
                self.sections, 
                convert_to_tensor=True,
                show_progress_bar=True
            )
            logging.info("Embeddings created successfully")
        except Exception as e:
            logging.error(f"Error creating embeddings: {str(e)}")
            raise
    
    def _create_tfidf_matrix(self):
        """Create TF-IDF matrix for keyword matching."""
        try:
            logging.info("Creating TF-IDF matrix...")
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.sections)
            logging.info("TF-IDF matrix created successfully")
        except Exception as e:
            logging.error(f"Error creating TF-IDF matrix: {str(e)}")
            raise
    
    def find_relevant_section(self, query: str, method: str = "semantic") -> Tuple[str, float]:
        """
        Find the most relevant section for a query.
        
        Args:
            query: User query
            method: "semantic" for embeddings or "keyword" for TF-IDF
            
        Returns:
            Tuple of (section_text, confidence_score)
        """
        try:
            if method == "semantic" and self.section_embeddings is not None:
                # Use semantic similarity
                query_embedding = self.embedder.encode(query, convert_to_tensor=True)
                similarities = util.cos_sim(query_embedding, self.section_embeddings)[0]
                best_idx = similarities.argmax().item()
                confidence = similarities[best_idx].item()
                
                logging.info(f"Semantic search: Found section {best_idx} with confidence {confidence:.3f}")
                return self.sections[best_idx], confidence
                
            elif method == "keyword" and self.tfidf_matrix is not None:
                # Use keyword matching
                query_vector = self.tfidf_vectorizer.transform([query])
                similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
                best_idx = similarities.argmax()
                confidence = similarities[best_idx]
                
                logging.info(f"Keyword search: Found section {best_idx} with confidence {confidence:.3f}")
                return self.sections[best_idx], confidence
                
            else:
                # Fallback to simple keyword matching
                query_words = set(query.lower().split())
                best_section = ""
                best_score = 0
                
                for section in self.sections:
                    section_words = set(section.lower().split())
                    score = len(query_words.intersection(section_words)) / len(query_words)
                    if score > best_score:
                        best_score = score
                        best_section = section
                
                logging.info(f"Fallback search: Found section with score {best_score:.3f}")
                return best_section, best_score
                
        except Exception as e:
            logging.error(f"Error finding relevant section: {str(e)}")
            return "", 0.0
    
    def answer_query(self, query: str, context_method: str = "semantic") -> str:
        """
        Generate an answer for a user query.
        
        Args:
            query: User query
            context_method: Method to find relevant context
            
        Returns:
            Generated answer
        """
        try:
            logging.info(f"Processing query: {query}")
            
            # Find relevant context
            context, confidence = self.find_relevant_section(query, context_method)
            
            # Check if we have sufficient context
            if not context or confidence < 0.1:
                logging.warning(f"Low confidence ({confidence:.3f}) for query: {query}")
                return self._handle_out_of_scope_query(query)
            
            # Generate answer using QA model
            try:
                result = self.qa_model(question=query, context=context)
                answer = result["answer"]
                answer_confidence = result["score"]
                
                logging.info(f"Generated answer with confidence: {answer_confidence:.3f}")
                
                # If QA model confidence is too low, try alternative approach
                if answer_confidence < 0.3:
                    logging.warning("Low QA confidence, trying alternative approach")
                    return self._generate_alternative_answer(query, context)
                
                return answer
                
            except Exception as e:
                logging.error(f"Error in QA model: {str(e)}")
                return self._generate_alternative_answer(query, context)
                
        except Exception as e:
            logging.error(f"Error answering query: {str(e)}")
            return "I apologize, but I encountered an error processing your request. Please try rephrasing your question."
    
    def _handle_out_of_scope_query(self, query: str) -> str:
        """Handle queries that are not covered by the document."""
        logging.info(f"Handling out-of-scope query: {query}")
        
        # Check for common patterns and provide helpful responses
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["contact", "phone", "email", "support"]):
            return "I don't have specific contact information in my training data, but I recommend checking our website or reaching out to our general support team."
        
        elif any(word in query_lower for word in ["price", "cost", "fee", "subscription"]):
            return "I don't have current pricing information in my training data. Please visit our website or contact our sales team for the most up-to-date pricing details."
        
        elif any(word in query_lower for word in ["bug", "error", "problem", "issue"]):
            return "I don't have specific troubleshooting information for that issue in my training data. Please contact our technical support team with detailed error messages."
        
        else:
            return "I don't have enough information in my training data to answer that question accurately. Please contact our support team for assistance with this specific inquiry."
    
    def _generate_alternative_answer(self, query: str, context: str) -> str:
        """Generate an alternative answer when QA model fails."""
        # Extract relevant sentences from context
        sentences = re.split(r'[.!?]+', context)
        relevant_sentences = []
        
        query_words = set(query.lower().split())
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            if len(query_words.intersection(sentence_words)) > 0:
                relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            return ". ".join(relevant_sentences[:3]) + "."
        else:
            return context[:200] + "..." if len(context) > 200 else context
    
    def simulate_feedback(self, query: str, response: str) -> str:
        """
        Simulate user feedback for a response.
        
        Args:
            query: Original user query
            response: Generated response
            
        Returns:
            Simulated feedback
        """
        # Simple feedback simulation based on response characteristics
        feedback_options = ["good", "not helpful", "too vague", "unclear"]
        
        # Analyze response characteristics
        response_length = len(response.split())
        
        if response_length < 10:
            # Very short responses are likely "too vague"
            weights = [0.1, 0.2, 0.6, 0.1]
        elif response_length > 100:
            # Very long responses might be "unclear"
            weights = [0.3, 0.2, 0.1, 0.4]
        elif "don't have" in response.lower() or "not enough information" in response.lower():
            # Out-of-scope responses are likely "not helpful"
            weights = [0.1, 0.7, 0.1, 0.1]
        else:
            # Balanced feedback for normal responses
            weights = [0.4, 0.2, 0.2, 0.2]
        
        feedback = random.choices(feedback_options, weights=weights)[0]
        
        # Store feedback in history
        self.feedback_history.append({
            "query": query,
            "response": response,
            "feedback": feedback,
            "timestamp": datetime.now().isoformat()
        })
        
        logging.info(f"Simulated feedback: {feedback}")
        return feedback
    
    def adjust_response(self, query: str, response: str, feedback: str) -> str:
        """
        Adjust response based on feedback.
        
        Args:
            query: Original user query
            response: Current response
            feedback: User feedback
            
        Returns:
            Adjusted response
        """
        logging.info(f"Adjusting response based on feedback: {feedback}")
        
        strategy = self.improvement_strategies.get(feedback, "maintain_response")
        
        if strategy == "rephrase_with_context":
            # Rephrase with more context
            context, _ = self.find_relevant_section(query)
            if context:
                return f"Let me provide more detailed information: {context[:300]}..."
            else:
                return f"To better assist you: {response} Could you provide more specific details about what you're looking for?"
        
        elif strategy == "add_details":
            # Add more specific details
            context, _ = self.find_relevant_section(query)
            if context:
                return f"{response}\n\nAdditional details: {context[:200]}..."
            else:
                return f"{response}\n\nFor more specific information, please contact our support team."
        
        elif strategy == "simplify_explanation":
            # Simplify the explanation
            sentences = response.split('. ')
            simplified = sentences[0] if sentences else response
            return f"In simple terms: {simplified}"
        
        else:
            # Maintain current response
            return response
    
    def run_query_with_feedback(self, query: str, max_iterations: int = 2) -> Dict:
        """
        Process a query with feedback loop.
        
        Args:
            query: User query
            max_iterations: Maximum number of feedback iterations
            
        Returns:
            Dictionary with query, final response, and iteration history
        """
        logging.info(f"Starting query processing with feedback loop: {query}")
        
        iteration_history = []
        current_response = self.answer_query(query)
        
        iteration_history.append({
            "iteration": 0,
            "response": current_response,
            "feedback": None
        })
        
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print(f"{'='*60}")
        print(f"Initial Response: {current_response}")
        
        # Feedback loop
        for iteration in range(1, max_iterations + 1):
            feedback = self.simulate_feedback(query, current_response)
            
            iteration_history.append({
                "iteration": iteration,
                "response": current_response,
                "feedback": feedback
            })
            
            print(f"\nFeedback: {feedback}")
            
            if feedback == "good":
                print("[OK] Response accepted!")
                break
            
            # Adjust response based on feedback
            current_response = self.adjust_response(query, current_response, feedback)
            
            iteration_history[-1]["adjusted_response"] = current_response
            
            print(f"Adjusted Response: {current_response}")
        
        # Store in response history
        self.response_history.append({
            "query": query,
            "final_response": current_response,
            "iterations": iteration_history,
            "timestamp": datetime.now().isoformat()
        })
        
        logging.info(f"Query processing completed after {len(iteration_history)} iterations")
        
        return {
            "query": query,
            "final_response": current_response,
            "iteration_history": iteration_history
        }
    
    def run_batch_queries(self, queries: List[str]) -> List[Dict]:
        """
        Process multiple queries with feedback loops.
        
        Args:
            queries: List of user queries
            
        Returns:
            List of results for each query
        """
        logging.info(f"Processing batch of {len(queries)} queries")
        
        results = []
        for i, query in enumerate(queries, 1):
            print(f"\n{'#'*80}")
            print(f"PROCESSING QUERY {i}/{len(queries)}")
            print(f"{'#'*80}")
            
            result = self.run_query_with_feedback(query)
            results.append(result)
        
        return results
    
    def get_statistics(self) -> Dict:
        """Get statistics about the bot's performance."""
        total_queries = len(self.response_history)
        total_feedback = len(self.feedback_history)
        
        if total_feedback == 0:
            return {"message": "No queries processed yet"}
        
        feedback_counts = {}
        for feedback_entry in self.feedback_history:
            feedback = feedback_entry["feedback"]
            feedback_counts[feedback] = feedback_counts.get(feedback, 0) + 1
        
        avg_iterations = sum(len(result.get("iterations", [])) for result in self.response_history) / total_queries if total_queries > 0 else 0
        
        return {
            "total_queries": total_queries,
            "total_feedback_instances": total_feedback,
            "feedback_distribution": feedback_counts,
            "average_iterations_per_query": avg_iterations,
            "document_sections": len(self.sections),
            "log_file": self.log_file
        }
    
    def save_session_data(self, filename: str = "session_data.json"):
        """Save session data for analysis."""
        session_data = {
            "timestamp": datetime.now().isoformat(),
            "document_path": self.document_path,
            "statistics": self.get_statistics(),
            "response_history": self.response_history,
            "feedback_history": self.feedback_history
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Session data saved to {filename}")


def main():
    """Main execution function."""
    print("Customer Support Bot with Document Training")
    print("=" * 50)
    
    # Initialize the bot
    try:
        bot = SupportBotAgent("faq.txt")
        print("[OK] Bot initialized successfully!")
        
        # Sample queries for testing
        sample_queries = [
            "How do I reset my password?",
            "What's the refund policy?",
            "How can I contact support?",
            "What are the system requirements?",
            "How do I enable two-factor authentication?",
            "What payment methods do you accept?",
            "How do I upgrade my subscription?",
            "What's your privacy policy?",
            "How do I troubleshoot connection issues?",
            "How do I fly to the moon?"  # Out-of-scope query
        ]
        
        # Process queries
        results = bot.run_batch_queries(sample_queries)
        
        # Display statistics
        print(f"\n{'='*60}")
        print("SESSION STATISTICS")
        print(f"{'='*60}")
        
        stats = bot.get_statistics()
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for sub_key, sub_value in value.items():
                    print(f"  {sub_key}: {sub_value}")
            else:
                print(f"{key}: {value}")
        
        # Save session data
        bot.save_session_data()
        
        print(f"\n[SUCCESS] Session completed! Check the log file for detailed information.")
        print(f"Log file: {bot.log_file}")
        print(f"Session data: session_data.json")
        
    except Exception as e:
        print(f"[ERROR] Error: {str(e)}")
        logging.error(f"Main execution error: {str(e)}")


if __name__ == "__main__":
    main()
