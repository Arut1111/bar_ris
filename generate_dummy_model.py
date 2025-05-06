
from lightgbm import LGBMClassifier
import joblib

model = LGBMClassifier()
X = [[0, 0, 60, 90], [1, 5, 90, 120]]  # фиктивные данные: [рвота, боль, ЧСС, АД]
y = [0, 1]  # 0 — нет осложнений, 1 — есть

model.fit(X, y)
joblib.dump(model, "model_lgb.pkl")
print("Фиктивная модель сохранена как model_lgb.pkl")
