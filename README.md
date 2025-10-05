# Customer Support Bot with Document Training

A sophisticated AI-powered customer support bot that trains on documents and provides intelligent responses with feedback-based learning capabilities.

## 🚀 Features

- **Document Processing**: Supports both PDF and text files
- **Semantic Search**: Uses sentence transformers for intelligent document retrieval
- **Question Answering**: Powered by Hugging Face transformer models
- **Feedback Learning**: Simulates user feedback and adjusts responses iteratively
- **Comprehensive Logging**: Tracks all decisions and actions for transparency
- **Fallback Handling**: Gracefully handles out-of-scope queries
- **Multiple Search Methods**: Semantic embeddings, TF-IDF, and keyword matching
- **Session Analytics**: Provides detailed statistics and performance metrics

## 📋 Requirements

- Python 3.8+
- CUDA-compatible GPU (optional, for faster processing)
- 4GB+ RAM recommended

## 🛠️ Installation

### 1. Clone or Download the Project

```bash
# If using git
git clone <repository-url>
cd customer-support-bot

# Or download and extract the files to a directory
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv support_bot_env

# Activate virtual environment
# On Windows:
support_bot_env\Scripts\activate
# On macOS/Linux:
source support_bot_env/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: The first run will download pre-trained models (~500MB), which may take a few minutes.

## 🎯 Quick Start

### Basic Usage

1. **Prepare your document**: Place your FAQ or training document in the project directory
   - Supported formats: `.txt`, `.pdf`
   - Example: `faq.txt` (included in the project)

2. **Run the bot**:
   ```bash
   python support_bot_agent.py
   ```

3. **View results**: The bot will process sample queries and display results in the console

### Custom Document Usage

```python
from support_bot_agent import SupportBotAgent

# Initialize with your document
bot = SupportBotAgent("your_document.pdf")

# Process a single query
result = bot.run_query_with_feedback("How do I reset my password?")

# Process multiple queries
queries = ["Question 1", "Question 2", "Question 3"]
results = bot.run_batch_queries(queries)
```

## 📁 Project Structure

```
customer-support-bot/
├── support_bot_agent.py      # Main bot implementation
├── faq.txt                   # Sample FAQ document
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── logs/                     # Log files (created automatically)
│   └── support_bot_log_*.txt
└── session_data.json         # Session analytics (created after run)
```

## 🔧 Configuration

### Model Selection

You can customize the question-answering model by modifying the initialization:

```python
# Use different models
bot = SupportBotAgent("faq.txt", model_name="bert-base-uncased")
bot = SupportBotAgent("faq.txt", model_name="roberta-base")
```

### Search Methods

The bot supports multiple search methods:

- **Semantic** (default): Uses sentence transformers for meaning-based search
- **Keyword**: Uses TF-IDF for keyword-based matching
- **Fallback**: Simple keyword intersection for basic matching

## 📊 Output Files

### Log Files
- **Location**: `logs/support_bot_log_YYYYMMDD_HHMMSS.txt`
- **Content**: Detailed logs of all bot decisions, queries, and responses
- **Format**: Timestamped entries with different log levels

### Session Data
- **File**: `session_data.json`
- **Content**: Complete session analytics, query history, and feedback data
- **Usage**: Analysis and debugging

## 🧪 Testing

The bot comes with built-in test queries covering:

- **In-scope queries**: Password reset, refund policy, contact info, etc.
- **Out-of-scope queries**: Questions not covered in the training document
- **Edge cases**: Various response lengths and complexities

### Sample Queries Included

1. "How do I reset my password?"
2. "What's the refund policy?"
3. "How can I contact support?"
4. "What are the system requirements?"
5. "How do I enable two-factor authentication?"
6. "What payment methods do you accept?"
7. "How do I upgrade my subscription?"
8. "What's your privacy policy?"
9. "How do I troubleshoot connection issues?"
10. "How do I fly to the moon?" (out-of-scope)

## 🔍 How It Works

### 1. Document Processing
- Loads and cleans the training document
- Splits content into meaningful sections
- Creates embeddings for semantic search
- Builds TF-IDF matrix for keyword matching

### 2. Query Processing
- Analyzes user queries
- Finds most relevant document sections
- Generates answers using transformer models
- Handles out-of-scope queries gracefully

### 3. Feedback Loop
- Simulates user feedback ("good", "not helpful", "too vague", "unclear")
- Adjusts responses based on feedback type
- Limits iterations to prevent infinite loops
- Tracks improvement strategies

### 4. Learning and Adaptation
- Maintains response history
- Analyzes feedback patterns
- Improves response quality over time
- Provides performance statistics

## 🎛️ Advanced Usage

### Custom Feedback Simulation

```python
# Override feedback simulation
def custom_feedback_simulation(query, response):
    # Your custom logic here
    return "good"  # or "not helpful", "too vague", "unclear"

bot.simulate_feedback = custom_feedback_simulation
```

### Batch Processing

```python
# Process multiple documents
documents = ["faq1.txt", "faq2.txt", "manual.pdf"]
for doc in documents:
    bot = SupportBotAgent(doc)
    results = bot.run_batch_queries(your_queries)
```

### Performance Monitoring

```python
# Get detailed statistics
stats = bot.get_statistics()
print(f"Total queries processed: {stats['total_queries']}")
print(f"Average iterations: {stats['average_iterations_per_query']}")
print(f"Feedback distribution: {stats['feedback_distribution']}")
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Solution: The bot automatically falls back to CPU processing
   - Alternative: Reduce batch size or use smaller models

2. **Model Download Issues**
   - Solution: Ensure stable internet connection
   - Alternative: Pre-download models manually

3. **PDF Processing Errors**
   - Solution: Ensure PDF is not password-protected
   - Alternative: Convert PDF to text manually

4. **Low Response Quality**
   - Solution: Improve training document quality
   - Alternative: Adjust confidence thresholds

### Debug Mode

Enable detailed logging by modifying the logging level:

```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## 📈 Performance Metrics

The bot tracks several performance indicators:

- **Query Processing Time**: Time to generate responses
- **Confidence Scores**: Model confidence in answers
- **Feedback Distribution**: Types of feedback received
- **Iteration Count**: Average iterations per query
- **Success Rate**: Percentage of "good" feedback

## 🔮 Future Improvements

### Planned Features

1. **Multi-language Support**: Support for non-English documents
2. **Real-time Learning**: Learn from actual user feedback
3. **Integration APIs**: REST API for web integration
4. **Advanced Analytics**: More detailed performance metrics
5. **Custom Models**: Fine-tune models on specific domains

### Known Limitations

1. **Context Window**: Limited by transformer model context length
2. **Language Support**: Primarily optimized for English
3. **Real-time Feedback**: Currently uses simulated feedback
4. **Model Size**: Large models require significant memory

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request


For questions or issues:

1. Check the troubleshooting section above
2. Review the log files for error details
3. Open an issue on the project repository
4. Contact the development team

## 🙏 Acknowledgments

- **Hugging Face**: For transformer models and pipelines
- **Sentence Transformers**: For semantic search capabilities
- **PyPDF2**: For PDF processing
- **scikit-learn**: For TF-IDF implementation

---

**Note**: This bot is designed for educational and demonstration purposes. For production use, consider additional security measures, real user feedback integration, and performance optimization.

