import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
# from sklearn.linear_model import LinearRegression, Lasso
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.metrics import mean_squared_error

# Wczytanie danych z pliku CSV 
data = pd.read_csv('PRSA_data_2010.1.1-2014.12.csv', sep = "," , encoding = 'utf-8')

# Wykresy przedstawiające dane z pliku (przed wyczyszczeniem)

    # Wykres 1: Przykład zależności między PM2.5 a temperaturą (TEMP)
plt.figure(figsize=(10, 6))
plt.scatter(data['TEMP'], data['pm2.5'])
plt.xlabel('Temperatura (°C)')
plt.ylabel('Poziom PM2.5 (ug/m3)')
plt.title('Zależność między PM2.5 a temperaturą')
plt.show()

    # Wykres 2: Histogram rozkładu stężenia PM2.5
plt.figure(figsize=(10, 6))
plt.hist(data['pm2.5'], bins=30)
plt.xlabel('Poziom PM2.5 (ug/m3)')
plt.ylabel('Częstość')
plt.title('Rozkład stężenia PM2.5')
plt.show()

    # Wykres 3: Zmiany stężenia PM2.5 w czasie
        
        # Tworzenie kolumny Timestamp na podstawie kolumn Rok, Miesiąc, Dzień i Godzina
data['Timestamp'] = pd.to_datetime(data[['year','month','day','hour']])
        # Sformatowanie osi czasu
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))  # Formatowanie daty i godziny
# plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())  # Automatyczne dostosowanie podziałki

plt.figure(figsize=(12, 6))
plt.plot(data['Timestamp'], data['pm2.5'])
plt.xlabel('Czas')
plt.ylabel('Poziom PM2.5 (ug/m3)')
plt.title('Zmiany stężenia PM2.5 w czasie')
plt.xticks(rotation=45)
plt.show()

    # Wykres 4: pudełkowy (box plot) - rozkład poziomu PM2.5 w różnych miesiącach
plt.figure(figsize=(10, 6))
sns.boxplot(data['month'], data['pm2.5']) # type: ignore
plt.xlabel('Miesiąc')
plt.ylabel('Poziom PM2.5 (ug/m3)')
plt.title('Rozkład poziomu PM2.5 w różnych miesiącach')
plt.show()

    # Wykres 5: słupkowy (bar plot) - porównanie średnich poziomów PM2.5 w różnych kierunkach wiatru
mean_pm25_by_cbwd = data.groupby('cbwd')['pm2.5'].mean()
plt.figure(figsize=(10, 6))
mean_pm25_by_cbwd.plot(kind='bar')
plt.xlabel('Kierunek wiatru')
plt.ylabel('Średni poziom PM2.5 (ug/m3)')
plt.title('Porównanie średnich poziomów PM2.5 w różnych kierunkach wiatru')
plt.show()

    # Wykres 6: korelacji - macierz korelacji między atrybutami jakości powietrza
corr_matrix = data[['pm2.5', 'TEMP', 'DEWP', 'PRES', 'Iws', 'Is', 'Ir']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Macierz korelacji między atrybutami jakości powietrza')
plt.show()

# Wyczyszczenie danych

# Wykresy przedstawiające dane z pliku (po wyczyszczeniu)

# Zmiana zmiennych kategorycznych na numeryczne 
data['cbwd'] = LabelEncoder().fit_transform(data['cbwd'])

print(f"data:\n", 
      data)

# Standaryzacja danych numerycznych 

# Przygotowanie danych wejściowych i wyjściowych
    # X = data[['DEWP', 'TEMP', 'PRES', 'Iws']]  # Przykładowe atrybuty jakości powietrza
    # y = data['PM2.5']  # Poziom zanieczyszczenia PM2.5


# Podział danych na zbiór treningowy i testowy
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# regresja liniowa 

# Regresja liniowa LASSO

# Regresja wielomianowa stopnia 3/2/..

# Regresja z wykorzystaniem k-NN 

# Regresja z wykorzystaniem drzewa decyzyjnego

# Dane do wyniki
    # train_scores = [linReg_score_train,lasso_score_train, poly_score_train, knn_score_train, tree_score_train]
    # test_scores = [linReg_score_test, lasso_score_test, poly_score_test, knn_score_test, tree_score_test]
    # models = ['Linear Regression','Lasso', 'Polynomial Regression', 'k-NN', 'Decision Tree']

#Wykres z wynikami
    # plt.figure(figsize=(10, 5))
    # plt.plot(models, train_scores, label='Train Score')
    # plt.plot(models, test_scores, label='Test Score')
    # plt.xlabel('Model')
    # plt.ylabel('Score')
    # plt.title('Model Comparison')
    # plt.legend()
    # plt.show()

#Tabela wyników
    # results = pd.DataFrame({'Model': models,
    #                         'Train Score': train_scores,
    #                         'Test Score': test_scores})