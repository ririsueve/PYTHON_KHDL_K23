# 1. Tham số cho bước cleaning
my_golden_specs = {
    'GENDER': ['male', 'female'],
    'EDUCATION': ['none', 'high school', 'university'],
    'INCOME': ['poverty', 'working class', 'middle class', 'upper class'],
    'DRIVING_EXPERIENCE': ['0-9y', '10-19y', '20-29y', '30y+'],
    'VEHICLE_YEAR': ['before 2015', 'after 2015'],
    'TYPE_OF_VEHICLE': ['sedan', 'sports car', 'suv', 'hatchback'],
    'AGE': ['16-25', '26-39', '40-64', '65+']
}

# 2. Chuẩn giới hạn cho cột số
my_range_rules = {
    'CREDIT_SCORE': (0.0, 1.0),
    'VEHICLE_OWNERSHIP': (0, 1),
    'MARRIED': (0, 1),
    'CHILDREN': (0, 1),
    'ANNUAL_MILEAGE': (0, None),
    'PAST_ACCIDENTS': (0, None)
}
cols_to_drop = ['ID','POSTAL_CODE','DUIS','SPEEDING_VIOLATIONS','PAST_ACCIDENTS']
max_drop_ratio = 0.05
imputation_strategy = 'auto'  # 'ffill', 'bfill', 'auto'
fuzzy_threshold = 90

# 3. Tham số cho bước transformation
my_outlier_strategies = {
    # Cột gốc
    'ANNUAL_MILEAGE': 'log',
    'PAST_ACCIDENTS': 'log',
    'CREDIT_SCORE': 'capping',
    'CHILDREN': None, 'MARRIED': None, 'VEHICLE_OWNERSHIP': None,
    'FAMILY_STABILITY': 'log',
}

# 4. Map thứ tự để biến chữ thành số
my_ordinal_mappings = {
    'AGE': {'16-25': 0, '26-39': 1, '40-64': 2, '65+': 3},
    'DRIVING_EXPERIENCE': {'0-9y': 0, '10-19y': 1, '20-29y': 2, '30y+': 3},
    'EDUCATION': {'none': 0, 'high school': 1, 'university': 2},
    'INCOME': {'poverty': 0, 'working class': 1, 'middle class': 2, 'upper class': 3},
    'VEHICLE_YEAR': {'before 2015': 0, 'after 2015': 1},
    'GENDER': {'female': 0, 'male': 1}
}
nominal_columns = ['TYPE_OF_VEHICLE']
ignore_cols= ['OUTCOME']
scaling_strategy='minmax' # 'standard' or 'minmax' or 'auto' or 'robust'
