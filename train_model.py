
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier

# Загрузка данных
df = pd.read_excel("таблица_бариатрические_пациенты_ (1).xlsx", sheet_name="Worksheet", header=1)

# Выбор признаков
features = [
    'Наличие рвоты после операции',
    'Абдоминальная боль, ВАШ',
    'ЧСС, уд/мин',
    'АД, мм.рт.ст',
    'Наличие осложнения, связанного с операцией'
]
df = df[features].dropna()
df['Наличие осложнения, связанного с операцией'] = df['Наличие осложнения, связанного с операцией'].astype(int)

X = df.drop(columns='Наличие осложнения, связанного с операцией')
y = df['Наличие осложнения, связанного с операцией']

# Кодирование категориальных признаков
for col in X.select_dtypes(include='object').columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

X = X.fillna(X.mean(numeric_only=True))
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Обучение модели
model = LGBMClassifier(random_state=42)
model.fit(X_train, y_train)

# Сохранение модели
joblib.dump(model, "model_lgb.pkl")
print("Модель сохранена как model_lgb.pkl")
