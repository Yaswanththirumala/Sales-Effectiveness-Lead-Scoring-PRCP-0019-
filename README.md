# Sales Effectiveness: Lead Scoring (PRCP-0019)

Machine learning project that predicts whether a sales lead is **High Potential** or **Low Potential**, so a sales team can prioritize the leads most likely to convert.

## Business Problem
A technology solutions company generates most of its leads through digital channels and its website. Lead quality is currently categorized manually by the sales staff, and this categorization is mostly used for post-analysis rather than to guide the sales team in time. The goal is to pre-categorize lead quality using machine learning to improve sales effectiveness.

## Project Goals
1. Explore the data to find insights about sales effectiveness.
2. Build a model that predicts the lead category (High Potential or Low Potential).
3. Suggest ways to improve sales effectiveness.

## Dataset
- 7,422 leads and 9 columns: Created, Product_ID, Source, Mobile, EMAIL, Sales_Agent, Location, Delivery_Mode, Status
- Missing values: Mobile (1,810), Product_ID (58), Location (58), Sales_Agent (23), Source (17)
- The dataset is not included in this repository because it comes from a private training database.

## Approach
1. **Data quality checks:** missing values, duplicates, cardinality
2. **Target creation:** grouped the original lead `Status` values into High Potential (1) and Low Potential (0)
3. **Feature engineering:** extracted hour and month from the creation date; frequency encoding for Source and Location (high-cardinality); one-hot encoding for Delivery_Mode
4. **Data leakage and privacy:** excluded `Status` (used to build the target) and the personal fields (Mobile, EMAIL, Sales_Agent) from the features
5. **Class imbalance:** stratified train/test split (80/20) and class weights
6. **Modeling:** compared 8 classification models using a scikit-learn pipeline

## Models Compared
Logistic Regression, Support Vector Machine, Naive Bayes, Decision Tree, Random Forest, K-Nearest Neighbours, Gradient Boosting, XGBoost

## Results
| Metric | Best Model (`<model name>`) |
|---|---|
| Accuracy | `<value>` |
| Precision | `<value>` |
| Recall | `<value>` |
| F1-score | `<value>` |
| ROC-AUC | `<value>` |
| PR-AUC | `<value>` |

Because the classes are not perfectly balanced, models were compared on F1-score, recall, precision and PR-AUC, not on accuracy alone.

## Key Insights
- `<insight 1: for example, which lead source has the highest High Potential rate>`
- `<insight 2: for example, a location or delivery-mode pattern>`
- `<insight 3: for example, a time-of-day or month pattern>`

## Recommendations for the Sales Team
1. Prioritize leads with a higher predicted High Potential probability.
2. Compare lead quality across sources and improve the weaker channels.
3. Use location-level insights to allocate sales resources.
4. Use creation-time patterns to plan staffing and follow-up.
5. Retrain the model as new lead outcomes become available.
6. Treat the model's probability as a prioritization score, not a guarantee of conversion.

## Challenges and Techniques
| Challenge | Technique | Reason |
|---|---|---|
| Missing values | Imputation and explicit "Unknown" handling | Keep useful records |
| High-cardinality Source and Location | Frequency encoding | Avoid too many sparse columns |
| Categorical Delivery_Mode | One-hot encoding | Convert categories to numbers |
| Date/time text | Feature extraction | Make time information usable |
| Personal data | Excluded from modeling | Privacy and identifier concerns |
| Class imbalance | Stratified split and class weights | Preserve class proportions |
| Status used to build the target | Removed from features | Prevent data leakage |

## Limitations
- The target is derived from historical lead statuses.
- Random train/test splitting is a baseline; time-based validation would better simulate real use.
- A high predicted probability does not guarantee conversion.

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn, Jupyter Notebook

## How to Run
1. Clone the repository.
2. Install the requirements: `pip install -r requirements.txt`
3. Provide the dataset as a CSV file in the same folder as the notebook, or connect to your own database.
4. Run `PRCP_0019_Sales_Effectiveness.ipynb` from top to bottom.
