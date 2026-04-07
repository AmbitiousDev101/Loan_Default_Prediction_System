import sqlite3
import pandas as pd
import os
from datetime import datetime

class LoanDatabase:
    """
    Manages the SQLite database for loan applications and predictions.
    This replaces the static CSV interaction with a persistent relational layer.
    """
    def __init__(self, db_path="loan_system.db"):
        self.db_path = db_path
        self._initialize_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _initialize_db(self):
        """
        Creates the tables if they don't exist.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Table for raw application data
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS applications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    borrower TEXT,
                    person_age INTEGER,
                    person_income INTEGER,
                    person_home_ownership TEXT,
                    person_emp_length REAL,
                    loan_intent TEXT,
                    loan_grade TEXT,
                    loan_amnt INTEGER,
                    loan_int_rate REAL,
                    loan_percent_income REAL,
                    cb_person_default_on_file TEXT,
                    cb_person_cred_hist_length INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Table for model predictions (Audit Log)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    borrower TEXT,
                    prediction_result TEXT,
                    probability REAL,
                    model_version TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()

    def sync_csv_to_db(self, csv_df):
        """
        Syncs data from a Pandas DataFrame into the SQL database.
        This demonstrates an ETL (Extract, Transform, Load) process.
        """
        with self._get_connection() as conn:
            # Check if we already have data to avoid duplicates
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM applications")
            if cursor.fetchone()[0] == 0:
                print(f"Database is empty. Migrating {len(csv_df)} records from source...")
                csv_df.to_sql('applications', conn, if_exists='append', index=False)
                print("ETL Sync Complete: Data migrated to SQL.")
            else:
                print("Database already contains records. Skipping migration.")

    def get_applications(self, column=None, operator=None, value=None):
        """
        Retrieves applications using safely parameterized queries to prevent SQL injection.
        """
        query = "SELECT * FROM applications"
        params = []
        
        # Whitelist of allowed columns for extra security
        allowed_columns = [
            'person_age', 'person_income', 'loan_amnt', 'loan_intent', 
            'loan_grade', 'person_home_ownership'
        ]

        allowed_operators = ['>', '<', '=', '>=', '<=']

        if column:
            if column not in allowed_columns:
                raise ValueError(f"Unsupported column: '{column}'. Allowed: {', '.join(allowed_columns)}")
            
            if operator not in allowed_operators:
                raise ValueError(f"Invalid operator: '{operator}'. Supported: {', '.join(allowed_operators)}")
                
            query += f" WHERE {column} {operator} ?"
            params.append(value)
        
        with self._get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def record_decision(self, borrower, result, probability=None):
        """
        Persists a model decision to the relational database.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO predictions (borrower, prediction_result, probability, model_version)
                VALUES (?, ?, ?, ?)
            ''', (borrower, result, probability, "RandomForest_v1.0"))
            conn.commit()

    def get_analytics(self):
        """
        Runs advanced SQL aggregations to provide system insights.
        """
        with self._get_connection() as conn:
            stats = {}
            cursor = conn.cursor()
            
            # Total apps by intent
            cursor.execute("SELECT loan_intent, COUNT(*) FROM applications GROUP BY loan_intent")
            stats['by_intent'] = cursor.fetchall()
            
            # Average loan amount by grade
            cursor.execute("SELECT loan_grade, AVG(loan_amnt) FROM applications GROUP BY loan_grade")
            stats['avg_by_grade'] = cursor.fetchall()
            
            return stats
