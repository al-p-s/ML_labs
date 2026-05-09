import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ------------------------------
# 1. Загрузка данных
# ------------------------------
train = pd.read_csv('z_datasets/titanic_train.csv')
test = pd.read_csv('z_datasets/titanic_test.csv')

# Удаляем строки с пропущенным Embarked в трейне (их 2: индексы 61 и 829)
train = train.dropna(subset=['Embarked']).copy()

# ------------------------------
# 2. Инженерия признаков (аналогично предыдущему)
# ------------------------------
def preprocess(df):
    df = df.copy()
    df['HasCabin'] = df['Cabin'].notna().astype(int)
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    common_titles = ['Mr', 'Miss', 'Mrs', 'Master']
    df['Title'] = df['Title'].apply(lambda x: x if x in common_titles else 'Rare')
    return df

train = preprocess(train)
test = preprocess(test)

# Убираем столбцы, которые не используем как признаки
drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
X_train = train.drop(columns=drop_cols + ['Embarked'])  # Embarked - цель
y_train = train['Embarked']

X_test = test.drop(columns=drop_cols + ['Embarked'])    # для теста Embarked известен
y_test = test['Embarked']

# Категориальные и числовые признаки
cat_features = ['Sex', 'Title']         # без Embarked, он целевой
num_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'HasCabin']

# ------------------------------
# 3. Пайплайн предобработки
# ------------------------------
num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

# ------------------------------
# 4. Стекинг-классификатор
# ------------------------------
base_estimators = [
    ('dt', DecisionTreeClassifier(max_depth=5, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=7)),
    ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
    ('nb', GaussianNB())
]

meta_clf = LogisticRegression(max_iter=1000, random_state=42)

stacking_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('stacking', StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_clf,
        cv=5
    ))
])

# ------------------------------
# 5. Обучение и оценка на тестовом наборе
# ------------------------------
stacking_pipeline.fit(X_train, y_train)

# Предсказание на тесте
y_pred = stacking_pipeline.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

print(f"Точность мета-классификатора (Embarked) на titanic_test.csv: {test_accuracy:.4f}")
print("\nОтчёт по классификации:")
print(classification_report(y_test, y_pred, target_names=['C', 'Q', 'S']))
