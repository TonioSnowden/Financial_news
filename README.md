# ML4Fin
## Abstract
This study explores the application of machine learning techniques to analyze the sentiment of earnings calls and develop trading strategies based on those sentiments. With access to a dataset comprising 18,755 earnings call transcripts, we implement various NLP models to extract sentiment signals. Our approach includes state-of-the-art machine learning algorithms, including transformers like FinBERT, optimized for financial contexts. The sentiments derived from earnings calls are then used to predict short-term stock returns, aiming to capture market reactions post-earnings announcements. Furthermore, we experiment with multiple trading strategies leveraging these sentiment signals to guide investment decisions. The performance of these strategies is assessed by back-testing on historical data, demonstrating the potential of sentiment analysis as a predictive tool.

## Installation
Project for the Machine Learning For Finance course (FIN-407)

To install the project, run the following commands:
```
pip install -r requirements.txt
pip install -e .
```

## Project Organization
```
    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    ├── notebooks          <- Jupyter notebooks.    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    └── report.pdf         <- The final report for the project.
```

