# ============================================================
# data_manager.py — Data generation, validation, and constants
# ============================================================

import pandas as pd
import numpy as np
import uuid
from datetime import datetime, timedelta

# --------------- CONSTANTS ---------------

# Global Country-to-Continent Knowledge Base
COUNTRIES_CONTINENTS = {
    # ---- Africa ----
    'Algeria': 'Africa', 'Angola': 'Africa', 'Benin': 'Africa', 'Botswana': 'Africa', 
    'Burkina Faso': 'Africa', 'Burundi': 'Africa', 'Cabo Verde': 'Africa', 'Cameroon': 'Africa', 
    'Central African Republic': 'Africa', 'Chad': 'Africa', 'Comoros': 'Africa', 'Congo': 'Africa', 
    "Cote d'Ivoire": 'Africa', 'Djibouti': 'Africa', 'Egypt': 'Africa', 'Equatorial Guinea': 'Africa', 
    'Eritrea': 'Africa', 'Eswatini': 'Africa', 'Ethiopia': 'Africa', 'Gabon': 'Africa', 'Gambia': 'Africa', 
    'Ghana': 'Africa', 'Guinea': 'Africa', 'Guinea-Bissau': 'Africa', 'Kenya': 'Africa', 'Lesotho': 'Africa', 
    'Liberia': 'Africa', 'Libya': 'Africa', 'Madagascar': 'Africa', 'Malawi': 'Africa', 'Mali': 'Africa', 
    'Mauritania': 'Africa', 'Mauritius': 'Africa', 'Morocco': 'Africa', 'Mozambique': 'Africa', 'Namibia': 'Africa', 
    'Niger': 'Africa', 'Nigeria': 'Africa', 'Rwanda': 'Africa', 'Sao Tome and Principe': 'Africa', 'Senegal': 'Africa', 
    'Seychelles': 'Africa', 'Sierra Leone': 'Africa', 'Somalia': 'Africa', 'South Africa': 'Africa', 
    'South Sudan': 'Africa', 'Sudan': 'Africa', 'Tanzania': 'Africa', 'Togo': 'Africa', 'Tunisia': 'Africa', 
    'Uganda': 'Africa', 'Zambia': 'Africa', 'Zimbabwe': 'Africa',
    
    # ---- Asia ----
    'China': 'Asia', 'India': 'Asia', 'Japan': 'Asia', 'South Korea': 'Asia', 'Indonesia': 'Asia', 
    'Pakistan': 'Asia', 'Bangladesh': 'Asia', 'Philippines': 'Asia', 'Vietnam': 'Asia', 'Thailand': 'Asia', 
    'Malaysia': 'Asia', 'Singapore': 'Asia', 'Sri Lanka': 'Asia', 'Saudi Arabia': 'Asia', 'UAE': 'Asia', 
    'Israel': 'Asia', 'Qatar': 'Asia', 'Kuwait': 'Asia', 'Iraq': 'Asia', 'Iran': 'Asia', 'Turkey': 'Asia',
    
    # ---- Europe ----
    'United Kingdom': 'Europe', 'Germany': 'Europe', 'France': 'Europe', 'Italy': 'Europe', 'Spain': 'Europe', 
    'Netherlands': 'Europe', 'Belgium': 'Europe', 'Switzerland': 'Europe', 'Austria': 'Europe', 'Sweden': 'Europe', 
    'Norway': 'Europe', 'Denmark': 'Europe', 'Finland': 'Europe', 'Poland': 'Europe', 'Ireland': 'Europe', 
    'Portugal': 'Europe', 'Greece': 'Europe', 'Romania': 'Europe', 'Czech Republic': 'Europe', 'Ukraine': 'Europe',
    
    # ---- North America ----
    'United States': 'North America', 'Canada': 'North America', 'Mexico': 'North America', 
    'Guatemala': 'North America', 'Cuba': 'North America', 'Haiti': 'North America', 
    'Dominican Republic': 'North America', 'Honduras': 'North America', 'Jamaica': 'North America', 
    'Panama': 'North America', 'Costa Rica': 'North America',
    
    # ---- South America ----
    'Brazil': 'South America', 'Argentina': 'South America', 'Colombia': 'South America', 
    'Chile': 'South America', 'Peru': 'South America', 'Venezuela': 'South America', 
    'Ecuador': 'South America', 'Bolivia': 'South America', 'Paraguay': 'South America', 
    'Uruguay': 'South America',
    
    # ---- Oceania ----
    'Australia': 'Oceania', 'New Zealand': 'Oceania', 'Papua New Guinea': 'Oceania', 
    'Fiji': 'Oceania'
}

SERVICE_PAGES = [
    'AI Solutions', 'Cloud Services', 'Data Analytics',
    'Cybersecurity', 'IoT Platform', 'Digital Marketing',
]

AGE_GROUPS = ['18-24', '25-34', '35-44', '45-54', '55+']
GENDERS = ['Male', 'Female', 'Other']

REQUIRED_COLUMNS = [
    'timestamp', 'country', 'continent', 'gender', 'age_group',
    'service_page', 'request_demo', 'ai_assistant',
    'promotion_event', 'ip_address', 'visitor_id',
]

# --------------- GENERATION ---------------

def generate_sample_data(n_rows: int = 5000) -> pd.DataFrame:
    """Generate realistic, UNIQUE sample log data for the dashboard."""
    countries = list(COUNTRIES_CONTINENTS.keys())

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # FIX: Create truly random, unsorted timestamps like real server logs
    random_seconds = np.random.randint(0, 365 * 24 * 60 * 60, n_rows)
    timestamps = start_date + pd.to_timedelta(random_seconds, unit='s')

    data = {
        'timestamp': timestamps,
        'country': np.random.choice(countries, n_rows),
        'gender': np.random.choice(GENDERS, n_rows, p=[0.48, 0.48, 0.04]),
        'age_group': np.random.choice(
            AGE_GROUPS, n_rows, p=[0.15, 0.30, 0.25, 0.18, 0.12]
        ),
        'service_page': np.random.choice(SERVICE_PAGES, n_rows),
        'request_demo': np.random.choice(
            [True, False], n_rows, p=[0.25, 0.75]
        ),
        'ai_assistant': np.random.choice(
            [True, False], n_rows, p=[0.30, 0.70]
        ),
        'promotion_event': np.random.choice(
            [True, False], n_rows, p=[0.15, 0.85]
        ),
        'ip_address': [
            f'{np.random.randint(1,223)}.'
            f'{np.random.randint(0,255)}.'
            f'{np.random.randint(0,255)}.'
            f'{np.random.randint(1,254)}'
            for _ in range(n_rows)
        ],
        # FIX: Use UUID to guarantee every visitor ID is 100% unique every time
        'visitor_id': [f'V-{uuid.uuid4().hex[:8].upper()}' for _ in range(n_rows)],
    }

    df = pd.DataFrame(data)
    
    # Intelligent Region Assignment: Automatically maps the country to its continent
    df['continent'] = df['country'].map(COUNTRIES_CONTINENTS)
    
    # Sort by timestamp to mimic chronological log extraction
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df

# --------------- VALIDATION ---------------

def validate_data(df: pd.DataFrame):
    """
    Validate uploaded CSV data.
    Returns (is_valid: bool, errors: list[str], warnings: list[str])
    """
    errors = []
    warnings = []

    # ---- required columns ----
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {', '.join(missing)}")
        return False, errors, warnings          # can't continue

    # ---- completeness ----
    for col in REQUIRED_COLUMNS:
        nulls = df[col].isnull().sum()
        if nulls:
            pct = nulls / len(df) * 100
            if pct > 20:
                errors.append(
                    f"Column '{col}' has {pct:.1f}% missing values "
                    f"(threshold 20%)"
                )
            else:
                warnings.append(
                    f"Column '{col}' has {nulls} missing values ({pct:.1f}%)"
                )

    # ---- timestamp format ----
    try:
        pd.to_datetime(df['timestamp'])
    except Exception:
        errors.append("Column 'timestamp' contains invalid date/time values")

    # ---- boolean columns ----
    for col in ('request_demo', 'ai_assistant', 'promotion_event'):
        uniq = set(df[col].dropna().unique())
        valid = {True, False, 0, 1, 'True', 'False', 'true', 'false', '0', '1'}
        if not uniq.issubset(valid):
            errors.append(f"Column '{col}' contains invalid boolean values")

    # ---- gender values ----
    unexpected = set(df['gender'].dropna().unique()) - set(GENDERS)
    if unexpected:
        warnings.append(f"Unexpected gender values: {', '.join(map(str, unexpected))}")

    return len(errors) == 0, errors, warnings