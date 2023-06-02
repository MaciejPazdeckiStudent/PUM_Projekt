import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV


# Wczytanie danych z pliku CSV 
data = pd.read_csv('PRSA_data_2010.1.1-2014.12.csv', sep = "," , encoding = 'utf-8')

# Wykresy przedstawiające dane z pliku (przed wyczyszczeniem)

    # Wykres 1: Przykład zależności między PM2.5 a temperaturą (TEMP)
plt.figure(figsize=(10, 6))
plt.scatter(data['TEMP'], data['pm2.5'])
plt.xlabel('Temperatura (°C)')
plt.ylabel('Poziom PM2.5 (ug/m3)')
plt.title('Zależność między PM2.5 a temperaturą')
plt.savefig('nieoczyszcone/zaleznosc_PM25_temp.png')
#plt.show()

    # Wykres 2: Histogram rozkładu stężenia PM2.5
plt.figure(figsize=(10, 6))
plt.hist(data['pm2.5'], bins=30)
plt.xlabel('Poziom PM2.5 (ug/m3)')
plt.ylabel('Częstość')
plt.title('Rozkład stężenia PM2.5')
plt.savefig('nieoczyszcone/histogram_rozkladPM25.png')
#plt.show()

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
plt.savefig('nieoczyszcone/zmiany_stezeniaPM25_wczasie.png')
#plt.show()

    # Wykres 4: pudełkowy (box plot) - rozkład poziomu PM2.5 w różnych miesiącach
plt.figure(figsize=(10, 6))
sns.boxplot(x='month', y='pm2.5', data=data)
plt.xlabel('Miesiąc')
plt.ylabel('Poziom PM2.5 (ug/m3)')
plt.title('Rozkład poziomu PM2.5 w różnych miesiącach')
plt.savefig('nieoczyszcone/miesieczny_rozkladPM25.png')
#plt.show()

    # Wykres 5: słupkowy (bar plot) - porównanie średnich poziomów PM2.5 w różnych kierunkach wiatru
mean_pm25_by_cbwd = data.groupby('cbwd')['pm2.5'].mean()
plt.figure(figsize=(10, 6))
mean_pm25_by_cbwd.plot(kind='bar')
plt.xlabel('Kierunek wiatru')
plt.ylabel('Średni poziom PM2.5 (ug/m3)')
plt.title('Porównanie średnich poziomów PM2.5 w różnych kierunkach wiatru')
plt.savefig('nieoczyszcone/sredniePM25_kierunkiWiatru.png')
#plt.show()

    # Wykres 6: korelacji - macierz korelacji między atrybutami jakości powietrza
corr_matrix = data[['pm2.5', 'TEMP', 'DEWP', 'PRES', 'Iws', 'Is', 'Ir']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Macierz korelacji między atrybutami jakości powietrza')
plt.savefig('nieoczyszcone/korelacje_atrybutow.png')
#plt.show()

# Wyczyszczenie danych
    # Uzupełnienie brakujących wartości w kolumnie "year", "month", "day" (forward propagation)
data['year'].fillna(method='ffill', inplace=True)
data['month'].fillna(method='ffill', inplace=True)
data['day'].fillna(method='ffill', inplace=True)
data['hour'].interpolate(method='linear', inplace=True)

data['DEWP'].interpolate(method='linear', inplace=True)
data['TEMP'].interpolate(method='linear', inplace=True)
data['PRES'].interpolate(method='linear', inplace=True)
    # Zamiana wartości na całkowite
data['TEMP'] = data['TEMP'].astype(float)
data['PRES'] = data['PRES'].astype(float)
data['TEMP']=data['TEMP'].round().astype(int)
data['PRES']=data['PRES'].round().astype(int)
    # ujednolicenie wartości 
data['Iws']=data['Iws'].round(2).astype(float)

    # Usunięcie wierszy zawierających wartości NA z kolumny PM2.5
data = data.dropna(subset=['pm2.5'])
    
    # Usunięcie pierwszej kolumny indeksującej 
data = data.drop('No', axis=1)
   

# Wykresy przedstawiające dane z pliku (po wyczyszczeniu)



# Zmiana zmiennych kategorycznych na numeryczne 
le = LabelEncoder()
data['cbwd'] = le.fit_transform(data['cbwd'])

print(f"data:\n", 
      data)

# Standaryzacja danych numerycznych 
scaler = MinMaxScaler()
col_stand = ['TEMP', 'DEWP', 'PRES', 'Iws', 'Is', 'Ir']
for i in data[col_stand]:
    data[i] = scaler.fit_transform(data[[i]])
print(f"cleanded data:\n", 
      data.head())

  # Wykres 6: korelacje - macierz korelacji między atrybutami jakości powietrza
corr_matrix = data[['TEMP', 'DEWP', 'PRES', 'Iws', 'Is', 'Ir']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Macierz korelacji między atrybutami jakości powietrza')
plt.savefig('oczyszczone/korelacje_atrybutow.png')
#plt.show()

# Wykres 7: korelacje - macierz korelacji między PM2.5 a atrybutami jakości powietrza
    # wymnożenie *100 pozwala na odczytanie wartości w procentach
pd.DataFrame(abs(data.corr()['pm2.5'].drop('pm2.5')*100).sort_values( ascending=False)).plot.bar(figsize = (10,8))
plt.title('Macierz korelacji między PM2.5 a atrybutami jakości powietrza')
plt.xlabel('Atrybuty')
plt.ylabel('Korelacja (%)')
plt.savefig('oczyszczone/korelacje_PM25.png')
#plt.show()

#pytanie czy usuwamy  najmniej skorelowane cechy ? #todo

# Wykluczanie niektórych kolumn z analizy (np. 'Unnamed: 0', 'Credit_Score')
exclude_filter = ~data.columns.isin(['year','month','day','hour', 'Timestamp', 'Credit_Score'])

# Przygotowanie danych wejściowych i wyjściowych
X = data[['TEMP', 'DEWP', 'PRES', 'Iws', 'Is', 'Ir']]  # Przykładowe atrybuty jakości powietrza
y = data['pm2.5']  # Poziom zanieczyszczenia PM2.5
print(f"y:\n", 
      y.head())
print(f"X:\n", 
      X.head())

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# regresja liniowa 
linReg = LinearRegression().fit(X_train, y_train)
        # to zwraca R^2
linReg_score_train = linReg.score(X_train, y_train)
linReg_score_test = linReg.score(X_test, y_test)
  
y_predict = linReg.predict(X_test)
mean_squared_error(y_true=y_test, y_pred= y_predict)
 
print(f"linear:\n", 
      linReg_score_train)
# print(f"predict:\n", 
#       y_predict)

# Regresja liniowa LASSO
lasso = Lasso(alpha=1.0)
lasso.fit(X_train, y_train)

lasso_score_train = lasso.score(X_train, y_train)
lasso_score_test = lasso.score(X_test, y_test)

print(f"lasso:\n", 
      lasso_score_train)


# Regresja wielomianowa stopnia 3/2/..
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

lin_reg = LinearRegression()
lin_reg.fit(X_train_poly, y_train)

poly_score_train = lin_reg.score(X_train_poly, y_train)
poly_score_test = lin_reg.score(X_test_poly, y_test)

print(f"poly:\n", 
      poly_score_train)

# Regresja z wykorzystaniem k-NN 
knn = KNeighborsRegressor()
knn.fit(X_train, y_train)

knn_score_train = knn.score(X_train, y_train)
knn_score_test = knn.score(X_test, y_test)


print(f"knn:\n", 
      knn_score_train)

# Regresja z wykorzystaniem drzewa decyzyjnego
tree = DecisionTreeRegressor(max_depth=8)
tree.fit(X_train, y_train)

tree_score_train = tree.score(X_train, y_train)
tree_score_test = tree.score(X_test, y_test)

print(f"tree:\n", 
      tree_score_train)

# Dane do wyniki
train_scores = [linReg_score_train,lasso_score_train, poly_score_train, knn_score_train, tree_score_train]
test_scores = [linReg_score_test, lasso_score_test, poly_score_test, knn_score_test, tree_score_test]
models = ['Linear Regression','Lasso', 'Polynomial Regression', 'k-NN', 'Decision Tree']

#Wykres z wynikami
plt.figure(figsize=(10, 5))
plt.plot(models, train_scores, label='Train Score')
plt.plot(models, test_scores, label='Test Score')
plt.xlabel('Model')
plt.ylabel('Score')
plt.title('Model Comparison')
plt.legend()
plt.savefig('porownanie/porownanie modeli.png')
# plt.show()

#Tabela wyników
results = pd.DataFrame({'Model': models,
                        'Train Score': train_scores,
                        'Test Score': test_scores})