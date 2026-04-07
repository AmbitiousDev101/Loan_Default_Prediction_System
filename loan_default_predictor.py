import pandas as pd
import matplotlib.pyplot as plt
import os
import boto3
from dotenv import load_dotenv
from io import StringIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from carousel import Carousel
from database_manager import LoanDatabase



load_dotenv()


AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
BUCKET_NAME = os.getenv('AWS_BUCKET_NAME')
REGION = os.getenv('AWS_REGION')

if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
    print("Warning: AWS Credentials not found. Model will attempt to use local data files.")

def read_from_s3(filename):
    """
    Connects to AWS S3, downloads the CSV, and loads it into Pandas.
    """
    try:
        print(f"Connecting to AWS S3 to fetch {filename}...")
        s3 = boto3.client(
            's3', 
            aws_access_key_id=AWS_ACCESS_KEY, 
            aws_secret_access_key=AWS_SECRET_KEY, 
            region_name=REGION
        )
        
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=filename)
        csv_string = obj['Body'].read().decode('utf-8')
        return pd.read_csv(StringIO(csv_string))
        
    except Exception as e:
        print(f"Error fetching from S3: {e}")
        print("Falling back to local file...")
        return pd.read_csv(filename) 


print("Initializing Data Ingestion Layer...")
train_df = read_from_s3("credit_risk_train.csv")
test_df = read_from_s3("credit_risk_test.csv")
request_df = read_from_s3("loan_requests.csv")

# SQL Persistence Layer Initialization
db = LoanDatabase()
db.sync_csv_to_db(request_df) # Sync CSV source to Database

#Feature SetuP
target = 'loan_status'
cat_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
num_cols = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt',
            'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']

X_train, y_train = train_df.drop(columns=target), train_df[target]
X_test, y_test = test_df.drop(columns=target), test_df[target]

#Preprocessing & Model Pipeline
preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ]), num_cols),
    
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', RandomForestClassifier(class_weight='balanced', random_state=42))
])

#training the model and evaluating it
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Default', 'Default']))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#visualisation using matplotlib
def plot_age_distribution(df):
    plt.figure(figsize=(10, 5))
    for label, color in zip([0, 1], ['green', 'red']):
        df[df[target] == label]['person_age'].hist(alpha=0.6, label=f"{'Not ' if label == 0 else ''}Default", color=color)
    plt.title("Age Distribution by Loan Status")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_homeowner_pie(df):
    homeowners = df[df['person_home_ownership'] == 'OWN']
    sizes = homeowners[target].value_counts().sort_index()
    plt.pie(sizes, labels=['Not Default', 'Default'], autopct='%1.1f%%',
            colors=['#66b3ff', '#ff9999'], startangle=90, explode=(0, 0.1), shadow=True)
    plt.title("Homeowners Default Rate")
    plt.axis("equal")
    plt.show()

plot_age_distribution(train_df)
plot_homeowner_pie(train_df)

#Carousel Interface (For The Doubly Linked List & SQL Querying)
def run_carousel_interface():
    # Initial load from Database
    current_df = db.get_applications()
    
    def build_carousel(df):
        if df.empty:
            print("\n[!] No records found for this criteria.")
            return None
        
        # Prepare data for prediction (drop metadata like 'borrower' and 'id')
        predict_ready_df = df.drop(columns=['borrower', 'id', 'created_at'])
        preds = model.predict(predict_ready_df)
        
        carousel = Carousel()
        for i, row in df.iterrows():
            data = row.to_dict()
            data['prediction'] = "Reject" if preds[i] == 1 else "Accept"
            carousel.add(data)
        return carousel

    carousel = build_carousel(current_df)

    while True:
        if carousel:
            current = carousel.getCurrentData()
            print("\n" + "=" * 45)
            print(f" BORROWER: {current['borrower']} (ID: {current['id']})")
            print("-" * 45)
            for k, v in current.items():
                if k not in ['borrower', 'id', 'prediction']:
                    print(f"{k:.<25} {v}")
            print("-" * 45)
            print(f" MODEL DECISION: {current['prediction']}")
            print("=" * 45)
            
            # Persist decision to SQL
            db.record_decision(current['borrower'], current['prediction'])
        
        print("\n[M]enu: 1.Next | 2.Prev | 3.Filter (SQL) | 4.Analytics | 0.Quit")
        choice = input("Select an option: ")
        
        if choice == '1' and carousel:
            carousel.movePrevious()
        elif choice == '2' and carousel:
            carousel.moveNext()
        elif choice == '3':
            while True:
                print("\n--- Safe SQL Filter Mode ---")
                print("Columns: person_age, person_income, loan_amnt, loan_intent, loan_grade, person_home_ownership")
                col = input("Enter Column (or ENTER to cancel): ").strip()
                if not col:
                    break
                    
                op = input("Enter Operator (>, <, =, >=, <=): ").strip()
                val = input("Enter Value: ").strip()
                
                try:
                    # Type casting for safety (numeric columns)
                    if col in ['person_income', 'person_age', 'loan_amnt']:
                        val = float(val)
                    
                    current_df = db.get_applications(col, op, val)
                    carousel = build_carousel(current_df)
                    break # Success, exit filter loop
                except ValueError as ve:
                    print(f"\n[!] Input Error: {ve}")
                    print("Please try again.")
                except Exception as e:
                    print(f"\n[!] System Error: {e}")
                    break
        elif choice == '4':
            print("\n--- Model Analytics (SQL Aggregations) ---")
            stats = db.get_analytics()
            print("\n[ Loans by Intent ]")
            for intent, count in stats['by_intent']:
                print(f" - {intent}: {count}")
            print("\n[ Avg Loan Amount by Grade ]")
            for grade, avg_amt in stats['avg_by_grade']:
                print(f" - Grade {grade}: ${avg_amt:,.2f}")
            input("\nPress ENTER to return to Carousel...")
        elif choice == '0':
            break
        else:
            print("Invalid input or no data loaded.")

if __name__ == "__main__":
    run_carousel_interface()
