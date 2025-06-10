import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from carousel import Carousel


train_df = pd.read_csv("credit_risk_train.csv")
test_df = pd.read_csv("credit_risk_test.csv")
request_df = pd.read_csv("loan_requests.csv")

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

#Carousel Interface (For The Doubly Linked List)
def run_carousel_interface(request_df):
    df = request_df.copy()
    ids = df.pop('borrower') if 'borrower' in df.columns else [f"Borrower {i+1}" for i in range(len(df))]

    preds = model.predict(df)
    carousel = Carousel()

    for i, row in request_df.iterrows():
        data = row.to_dict()
        data['prediction'] = "Reject" if preds[i] == 1 else "Accept"
        data['borrower'] = ids[i]
        carousel.add(data)

    while True:
        current = carousel.getCurrentData()
        print("\n" + "-" * 40)
        for k, v in current.items():
            print(f"{k}: {v}")
        print("-" * 40)
        
        choice = input("1 = Next | 2 = Previous | 0 = Quit: ")
        if choice == '1':
            carousel.movePrevious()
        elif choice == '2':
            carousel.moveNext()
        elif choice == '0':
            break
        else:
            print("Invalid input.")

run_carousel_interface(request_df)
