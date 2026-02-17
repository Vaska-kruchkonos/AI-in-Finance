# Credit Default Prediction – Logistic Regression

**Автор:** Васька Крючконос  
**Курс:** AI in Finance  
**Задача:** Бинарная классификация кредитного дефолта

---

## 1. Описание задачи

Цель проекта — построить модель прогнозирования кредитного дефолта (Credit Default) с использованием **логистической регрессии**. Модель предсказывает вероятность дефолта на основе финансовых и кредитных характеристик заёмщика.

Задача является **бинарной классификацией** с несбалансированными классами (~28% дефолтов в обучающей выборке).

---

## 2. Данные

### 2.1 Описание датасета

- **Training set:** 7,500 наблюдений, 16 исходных признаков + 1 целевая переменная (`Credit Default`)
- **Test set:** 2,500 наблюдений (без целевой переменной)
- **Целевая переменная:** `Credit Default` (0 = no default, 1 = default)
- **Train/Val split:** 6,000 / 1,500 (80/20, stratified)

### 2.2 Признаки

**Категориальные:**
- Home Ownership (Own Home, Rent, Home Mortgage, HaveMortgage)
- Years in current job (< 1 year, 1 year, 2 years, ..., 10+ years)
- Purpose (Debt Consolidation, Home Improvements, Business Loan, etc.)
- Term (Short Term, Long Term)

**Числовые:**
- Annual Income
- Tax Liens
- Number of Open Accounts
- Years of Credit History
- Maximum Open Credit
- Number of Credit Problems
- Months since last delinquent
- Bankruptcies
- Current Loan Amount
- Current Credit Balance
- Monthly Debt
- Credit Score

### 2.3 Выявленные проблемы данных

- **Пропуски:** Значительные пропуски в `Annual Income`, `Years in current job`, `Months since last delinquent`
- **Outliers:** Placeholder-значение `99999999` в `Current Loan Amount` заменено на NaN
- **Дисбаланс классов:** ~71.8% класса 0, ~28.2% класса 1 (после split)

---

## 3. Pipeline шаги

### 3.1 EDA (Exploratory Data Analysis)

- Анализ размера выборки, количества признаков, баланса классов
- Исследование пропущенных значений и их распределения
- Описательная статистика числовых признаков
- Корреляционная матрица для выявления связей признаков с таргетом и мультиколлинеарности
- Визуализация распределений и выбросов

**Основные находки:**
- Дисбаланс классов (28.2% дефолтов) требует внимания к Recall
- Высокая корреляция между `Current Credit Balance`, `Maximum Open Credit`, `Number of Open Accounts`
- Placeholder 99999999 в `Current Loan Amount` заменён на NaN для корректной обработки

### 3.2 Preprocessing

Для устранения **data leakage** использован **Pipeline с ColumnTransformer:**

- **Числовые признаки:**
  1. SimpleImputer (strategy='median') – заполнение пропусков медианой
  2. StandardScaler – стандартизация

- **Категориальные признаки:**
  1. SimpleImputer (strategy='most_frequent') – заполнение пропусков модой
  2. OneHotEncoder (handle_unknown='ignore') – кодирование

- **Финальный Pipeline:** ColumnTransformer → LogisticRegression

Данная структура гарантирует, что импутация и масштабирование выполняются только на train-фолдах при кросс-валидации, исключая утечку информации из тестовых данных.

### 3.3 Feature Engineering

Созданы 7 экономически обоснованных производных признаков:

1. **DTI_Ratio** = Monthly Debt / (Annual Income / 12)  
   *Коэффициент долговой нагрузки*
2. **Credit_Utilization** = Current Credit Balance / Maximum Open Credit  
   *Степень использования кредитной линии*
3. **Loan_to_Income** = Current Loan Amount / Annual Income  
   *Отношение размера кредита к доходу*
4. **Delinquency_Flag** = 1, если Months since last delinquent > 0  
   *Наличие просрочек в истории*
5. **Problem_Rate** = Number of Credit Problems / Years of Credit History  
   *Плотность кредитных проблем*
6. **Open_Account_per_Year** = Number of Open Accounts / Years of Credit History  
   *Интенсивность открытия счетов*
7. **Bankruptcy_Flag** = 1, если Bankruptcies > 0  
   *Наличие банкротств*

**Обоснование:** эти признаки отражают ключевые факторы кредитного риска, используемые в банковском скоринге, и улучшают интерпретируемость модели.

---

## 4. Модели

### 4.1 Baseline Model

Логистическая регрессия с параметрами по умолчанию (без тюнинга), обученная на train (6,000), оценена на validation (1,500).

### 4.2 Hyperparameter Tuning

**GridSearchCV** по следующим гиперпараметрам (только валидные комбинации):
- `penalty`: ['l1', 'l2']
- `solver`: ['liblinear']
- `C`: [0.001, 0.01, 0.1, 1, 10, 100]
- `class_weight`: [None, 'balanced']
- `max_iter`: [1000, 2000]

**Всего:** 48 комбинаций  
**Scoring:** `roc_auc` (критично для несбалансированных данных)  
**Cross-validation:** 5 фолдов (stratified)

**Best parameters (CV ROC-AUC = 0.7299):**
```python
{
  'model__C': 0.1,
  'model__class_weight': None,
  'model__penalty': 'l1',
  'model__solver': 'liblinear',
  'model__max_iter': 1000
}
```

### 4.3 Final Model

Обученная на **всех** training данных (7,500), применена для предсказания вероятностей дефолта на test (2,500). Результат сохранён в `predictions.csv` с колонками `id` и `probability_default`.

**Статистика predictions.csv:**
| Metric | Value |
|--------|-------|
| Count  | 2,500 |
| Mean   | 0.2889 |
| Std    | 0.2117 |
| Min    | 0.0001 |
| 25%    | 0.1691 |
| 50%    | 0.2317 |
| 75%    | 0.3285 |
| Max    | 0.9992 |

---

## 5. Метрики

### 5.1 Baseline Model (параметры по умолчанию)

| Metric      | Value  |
|-------------|--------|
| **Accuracy**  | 0.7793 |
| **Precision** | 0.8333 |
| **Recall**    | 0.2719 |
| **F1-score**  | 0.4100 |
| **ROC-AUC**   | 0.7267 |
| **Gini**      | 0.4535 |

**Confusion Matrix (Validation):**
```
              Predicted
              0     1
Actual 0   1054    23
       1    308   115
```

**Анализ:**
- Высокая Precision (83%), но **низкий Recall (27%)** – модель пропускает 73% дефолтов
- Accuracy завышена из-за дисбаланса классов
- ROC-AUC 0.73 – приемлемое качество ранжирования

### 5.2 Tuned Model (GridSearchCV)

| Metric      | Value  | Δ vs Baseline |
|-------------|--------|---------------|
| **Accuracy**  | 0.7773 | -0.0020       |
| **Precision** | 0.8678 | +0.0345       |
| **Recall**    | 0.2482 | -0.0237       |
| **F1-score**  | 0.3860 | -0.0240       |
| **ROC-AUC**   | 0.7280 | +0.0013       |
| **Gini**      | 0.4560 | +0.0025       |

**Confusion Matrix (Validation):**
```
              Predicted
              0     1
Actual 0   1061    16
       1    318   105
```

**Анализ:**
- Tuned модель показала **незначительное улучшение ROC-AUC** (+0.0013)
- Recall **снизился** с 27.2% до 24.8% (более консервативная модель)
- Precision выросла до 86.8% – меньше false positives
- L1 regularization (penalty='l1') обеспечила feature selection

**Важный вывод:**  
GridSearchCV **не** выбрал `class_weight='balanced'` – это означает, что кросс-валидация показала, что для максимизации ROC-AUC лучше работает несбалансированная модель с высокой Precision. Если бизнес-приоритет – поймать больше дефолтов (высокий Recall), нужно:
- Явно зафиксировать `class_weight='balanced'`
- Или использовать другой scoring (например, `f1` или кастомную метрику с весами FN/FP)

---

## 6. Интерпретация коэффициентов

Топ-10 признаков по абсолютному значению коэффициента логистической регрессии (финальная модель):

| Feature                      | Coefficient | Интерпретация |
|------------------------------|-------------|---------------|
| **Credit Score**             | **+1.4414** | ⬆ Парадокс! Высокий credit score **увеличивает** вероятность класса 1. **Проверить кодировку таргета!** |
| **Term_Short Term**          | **−0.8435** | ⬇ Краткосрочные кредиты **снижают** риск (меньше времени для дефолта) |
| **Annual Income**            | **−0.3886** | ⬇ Высокий доход **снижает** риск |
| **Purpose_debt consolidation** | **−0.3315** | ⬇ Рефинансирование долга **снижает** риск |
| **Credit_Utilization**       | **+0.2204** | ⬆ Высокая утилизация кредита **увеличивает** риск |
| **Current Loan Amount**      | **+0.1807** | ⬆ Большой кредит **увеличивает** риск |
| **Home Ownership_Home Mortgage** | **−0.1736** | ⬇ Ипотека **снижает** риск (стабильность) |
| **Home Ownership_Rent**      | **+0.1560** | ⬆ Аренда **увеличивает** риск (нестабильность) |
| **DTI_Ratio**                | **+0.1404** | ⬆ Высокая долговая нагрузка **увеличивает** риск |
| **Purpose_business loan**    | **+0.1242** | ⬆ Бизнес-кредиты **увеличивают** риск (волатильность) |

### ⚠️ ВАЖНОЕ ЗАМЕЧАНИЕ

**Credit Score имеет положительный коэффициент** (+1.44) – это **контринтуитивно** для кредитного скоринга (обычно высокий credit score должен снижать риск дефолта).

**Возможные причины:**
1. **Кодировка таргета перепутана** – возможно, `Credit Default = 1` означает "no default", а `= 0` означает "default" (обратная логика)
2. **Нелинейная зависимость** – возможно, очень высокие credit scores коррелируют с рискованным поведением (overleveraging)
3. **Артефакт данных** – placeholder-значения или ошибки в датасете

**Рекомендация:** Проверить описание данных и при необходимости инвертировать таргет или переобучить модель.

Остальные коэффициенты соответствуют экономической логике кредитного скоринга.

---

## 7. Общее качество модели

**Сильные стороны:**
- Модель интерпретируема: коэффициенты имеют экономический смысл (кроме Credit Score)
- ROC-AUC 0.73 – приемлемое качество ранжирования для baseline подхода
- Pipeline исключает data leakage, обеспечивает воспроизводимость
- Feature engineering добавил релевантные признаки (DTI_Ratio, Credit_Utilization в топ-10)

**Слабости:**
- **Низкий Recall (~25%)** – модель пропускает 75% дефолтов (критично!)
- Высокая Precision (87%) достигается за счёт консервативных предсказаний
- Линейная модель не учитывает нелинейные взаимодействия признаков
- **Аномалия Credit Score** требует разбирательства

**Вывод:** Модель **не готова** к production без решения проблемы низкого Recall. Для реального кредитного скоринга пропуск 75% дефолтов недопустим.

---

## 8. Возможные улучшения

1. **Решить проблему Credit Score:** проверить кодировку таргета, визуализировать зависимость
2. **Повысить Recall:**
   - Зафиксировать `class_weight='balanced'` вместо `None`
   - Использовать SMOTE/ADASYN для балансировки
   - Настроить threshold (вместо 0.5 использовать 0.3 для более агрессивных предсказаний)
   - Изменить scoring на `f1` или кастомную метрику с высоким весом FN
3. **Другие алгоритмы:** Random Forest, Gradient Boosting (XGBoost, LightGBM) для учёта нелинейностей
4. **Feature selection:** Recursive Feature Elimination, Lasso для отбора признаков
5. **Polynomial features:** добавление взаимодействий признаков (например, DTI_Ratio × Credit Score)
6. **Ensemble methods:** Stacking, Blending нескольких моделей
7. **Calibration:** калибровка вероятностей (Platt scaling, isotonic regression)

---

## 9. Структура репозитория

```
.
├── HW_regression_fixed.ipynb     # Jupyter-ноутбук с полным кодом
├── README.md                      # Документация проекта
├── course_project_train.csv      # Обучающая выборка (7,500 строк)
├── course_project_test.csv       # Тестовая выборка (2,500 строк)
├── predictions.csv                # Предсказания (id, probability_default)
└── requirements.txt               # Зависимости Python
```

---

## 10. Environment & Requirements

### Python версия
```
Python 3.8+
```

### Зависимости
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
```

**Установка:**
```bash
pip install -r requirements.txt
```

---

## 11. Как воспроизвести результаты

1. Установить зависимости: `pip install -r requirements.txt`
2. Поместить `course_project_train.csv` и `course_project_test.csv` в рабочую директорию
3. Открыть `HW_regression_fixed.ipynb` в Jupyter/Colab
4. Запустить все ячейки (Run All)
5. Результаты сохранятся в `predictions.csv`

**Время выполнения:** ~5-10 минут (зависит от CPU)

---

## 12. Контакты

**Автор:** Васька Крючконос  
**Курс:** AI in Finance  
**Дата:** 2026-02-16
