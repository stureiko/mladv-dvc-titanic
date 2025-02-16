import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import pickle

def main():
    df = sns.load_dataset('titanic')
    # Первичная предобработка
    # Удалим колонку 'deck' (много пропусков) и строки, где пропущен 'embarked' 
    df.drop(columns=['deck'], inplace=True)
    df.dropna(subset=['embarked'], inplace=True)

    # Целевая переменная и признаки
    y = df['survived']
    X = df[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']]

    # Определим числовые и категориальные столбцы
    numeric_features = ['age', 'sibsp', 'parch', 'fare']
    categorical_features = ['pclass', 'sex', 'embarked']

    # Настраиваем трансформацию для числовых признаков
    #    - Заполним пропуски медианой (SimpleImputer)
    #    - Применим StandardScaler (для нормализации)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Настраиваем трансформацию для категориальных признаков
    #    - Заполним пропуски самой частотной категорией (хотя в этом наборе их уже удалили, но для примера)
    #    - Затем закодируем OneHotEncoder
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Комбинируем обработку в ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
        ]
    )

    # Создаём финальный Pipeline
    #    Шаги в конвейере:
    #      - Предобработка (preprocessor)
    #      - Обучение классификатора (RandomForestClassifier)
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Разделение данных на обучающую/тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42
    )

    # Обучение модели (Pipeline автоматически применит все трансформации к X_train)
    model_pipeline.fit(X_train, y_train)
    with open("owesome_pipeline.pkl", "wb") as f:
        pickle.dump(model_pipeline, f)

    # Предсказание на тестовой выборке
    y_pred = model_pipeline.predict(X_test)

    # Оценка качества
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()