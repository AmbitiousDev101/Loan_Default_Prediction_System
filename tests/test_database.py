import pytest
import pandas as pd
import os
from database_manager import LoanDatabase

@pytest.fixture
def temp_db(tmp_path):
    """
    Fixture to create a temporary database for each test.
    Using a physical file in a temp directory ensures multiple connections work.
    """
    db_file = tmp_path / "test_loan_system.db"
    db = LoanDatabase(str(db_file))
    return db

@pytest.fixture
def sample_data():
    """
    Fixture providing sample loan application data.
    """
    return pd.DataFrame({
        'borrower': ['Alice', 'Bob'],
        'person_age': [25, 45],
        'person_income': [50000, 120000],
        'person_home_ownership': ['RENT', 'MORTGAGE'],
        'person_emp_length': [2.0, 10.0],
        'loan_intent': ['EDUCATION', 'MEDICAL'],
        'loan_grade': ['A', 'B'],
        'loan_amnt': [5000, 20000],
        'loan_int_rate': [10.5, 12.0],
        'loan_percent_income': [0.1, 0.16],
        'cb_person_default_on_file': ['N', 'N'],
        'cb_person_cred_hist_length': [3, 15]
    })

def test_db_initialization(temp_db):
    """Test if tables are created correctly."""
    assert os.path.exists(temp_db.db_path)
    # Check if we can query the tables
    apps = temp_db.get_applications()
    assert isinstance(apps, pd.DataFrame)
    assert apps.empty

def test_sync_csv_to_db(temp_db, sample_data):
    """Test the ETL sync process."""
    temp_db.sync_csv_to_db(sample_data)
    apps = temp_db.get_applications()
    assert len(apps) == 2
    assert 'Alice' in apps['borrower'].values

def test_record_decision(temp_db):
    """Test recording a model decision."""
    temp_db.record_decision("Alice", "Accept", 0.98)
    with temp_db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT borrower, prediction_result FROM predictions")
        row = cursor.fetchone()
        assert row[0] == "Alice"
        assert row[1] == "Accept"

def test_filtering(temp_db, sample_data):
    """Test SQL filtering logic."""
    temp_db.sync_csv_to_db(sample_data)
    # Filter for high income
    high_income = temp_db.get_applications("person_income > 100000")
    assert len(high_income) == 1
    assert high_income.iloc[0]['borrower'] == 'Bob'

def test_analytics(temp_db, sample_data):
    """Test SQL aggregation analytics."""
    temp_db.sync_csv_to_db(sample_data)
    stats = temp_db.get_analytics()
    assert 'by_intent' in stats
    assert len(stats['by_intent']) == 2 # EDUCATION and MEDICAL
    assert 'avg_by_grade' in stats
