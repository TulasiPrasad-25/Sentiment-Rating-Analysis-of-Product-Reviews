
### **Sentiment & Rating Analysis of Product Reviews**

This project involved developing and comparing machine learning models for sentiment and rating prediction based on a comprehensive dataset of product reviews. Leveraging natural language processing (NLP) techniques, I focused on understanding customer feedback and identifying critical insights.

**Key Highlights:**

*   **Data Preparation & EDA**: Loaded, cleaned, and explored a large product review dataset. Performed extensive exploratory data analysis, visualizing distributions of review lengths, sentiment categories, product ratings, and identifying the most common words.
*   **Feature Engineering**: Transformed raw text data into meaningful numerical features using TF-IDF (Term Frequency-Inverse Document Frequency) for traditional machine learning models and sequence padding for deep learning models.
*   **Model Development & Evaluation (Sentiment)**: Implemented and evaluated both a **Logistic Regression** model and a **Long Short-Term Memory (LSTM)** neural network for sentiment prediction (positive, neutral, negative). A comparative analysis revealed that Logistic Regression achieved an impressive overall accuracy of 89.52%, outperforming the LSTM model (81.24%) in this context, particularly due to the LSTM's struggle with minority classes.
*   **Model Development & Evaluation (Rating)**: Developed a **Logistic Regression** model for multi-class product rating prediction, achieving a strong overall accuracy of 88% by effectively classifying ratings from 1 to 5.
*   **Urgency/Complaint Detection**: Engineered a novel feature, `urgency_complaint_score`, through keyword extraction to quantify the intensity of complaints or urgency in reviews. This feature helps in prioritizing critical customer feedback.
*   **Insights**: Identified significant challenges posed by class imbalance, especially for the 'neutral' sentiment, highlighting the need for advanced techniques (e.g., oversampling, focal loss) in future iterations. The project underscores the importance of choosing the right model and addressing data characteristics for optimal performance.

**Technologies Used**: Python, Pandas, NLTK, Scikit-learn, TensorFlow/Keras, Matplotlib, Seaborn.
