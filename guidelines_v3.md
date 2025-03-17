# Sentiment Analysis Project - V3 Investigation Guidelines

## 1. Data Analysis Investigation
### 1.1 Review Current Data Sources
- Currently only using Amazon reviews dataset
- Need to analyze:
  * Text length distribution
  * Word frequency analysis
  * Missing or truncated reviews
  * Special characters handling
  * Common phrases analysis

### 1.2 Data Quality Check
- Examine cleaned data output:
  * Compare raw vs cleaned text samples
  * Check for lost sentiment-critical words
  * Analyze impact of current preprocessing steps
  * Verify stopwords removal impact
  * Check for patterns in misclassified cases

## 2. Error Analysis Focus Areas
### 2.1 False Positive Cases
- Investigate cases where negative words result in positive sentiment:
  * Words like "terrible", "awful", "bad" being misclassified
  * Context understanding failures
  * Negation handling ("not good" vs "good")
  * Complex sentence structure impact

### 2.2 Preprocessing Enhancement
- Review current text cleaning process:
  * Impact of removing punctuation
  * Effect of lowercase conversion
  * Stopwords strategy revision
  * Handling of negation words
  * Special character preservation

## 3. Technical Improvements
### 3.1 Feature Engineering
- Implement additional features:
  * N-gram analysis (unigrams, bigrams, trigrams)
  * Part-of-speech tagging
  * Dependency parsing
  * Sentiment-specific word embeddings
  * Contextual features

### 3.2 Model Enhancements
- Try additional modeling approaches:
  * Ensemble methods
  * Attention mechanisms
  * Rule-based sentiment modifiers
  * Custom word embeddings
  * Hybrid approaches

## 4. Specific Investigation Tasks
1. Create data analysis pipeline:
   - Compare raw vs cleaned text
   - Analyze lost information
   - Check sentiment word preservation

2. Implement error analysis tools:
   - Track misclassified cases
   - Analyze patterns in errors
   - Create confusion matrix per word type

3. Test enhanced preprocessing:
   - Keep sentiment-critical punctuation
   - Preserve negation context
   - Maintain phrase integrity

4. Expand feature engineering:
   - Add POS tagging
   - Implement dependency parsing
   - Create custom embeddings

## 5. Success Metrics
- Track improvements in:
  * Overall accuracy
  * Per-class performance
  * Specific word handling
  * Negation accuracy
  * Complex sentence understanding

## 6. Next Steps Priority
1. Data analysis pipeline implementation
2. Error case collection and analysis
3. Preprocessing enhancement testing
4. Feature engineering expansion
5. Model architecture improvements

## 7. Documentation Requirements
- Document all findings
- Track performance changes
- Record error patterns
- Maintain preprocessing impact logs
- Create comparison reports 