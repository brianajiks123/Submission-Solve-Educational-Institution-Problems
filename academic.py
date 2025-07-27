import pandas as pd
from sqlalchemy import create_engine

df = pd.read_csv("cleaned_students.csv", encoding='windows-1252')

URL = "postgresql://postgres:academic@localhost:5432/academic"

engine = create_engine(URL)
df.to_sql('academic_data', engine, if_exists='replace', index=False)
