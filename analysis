import pandas as pd
import numpy as np
import datetime
import plotly.express as px
from sklearn.linear_model import LinearRegression
from matplotlib.pyplot import figure, grid
import matplotlib.pyplot as plt


def get_dataframe(dataset):
  url = f"https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_{dataset.lower()}_global.csv"
  df = pd.read_csv(url, error_bad_lines=False)
  return df
  
def get_all_data_to_plot(key_word): #funkcja przygotowująca dane pod wykresy
  dfD = get_dataframe(str(key_word))
  columns = list(dfD.columns.values)
  new_df = dfD[[columns[1]] + columns[4:]] # utworzenie nowej tabeli skladajacej sie z nazw krajow i 7 ostatnich dni
  this_day = datetime.date.today() - datetime.timedelta(days=1)
  print(this_day)
  new_df = new_df.groupby(["Country/Region"]).sum().sort_values([this_day.strftime('%m/%d/%y').lstrip("0").replace("/0", "/")], ascending = False)

  print(f"{key_word} cases")
  return new_df

def get_top_countries(key_word, number_of_countries = 50):
  dfD = get_dataframe(str(key_word))
  columns = list(dfD.columns.values)
  new_df = dfD[[columns[1]] + columns[-10:]] # utworzenie nowej bazy skladajacej sie z nazw krajow i kilku ostatnich dni
  this_day = datetime.date.today() - datetime.timedelta(days=1)
  new_df = new_df.groupby(["Country/Region"]).sum().sort_values([this_day.strftime('%m/%d/%y').lstrip("0").replace("/0", "/")], ascending = False).head(number_of_countries)
  new_df['Increase in 7 days'] = 0 # utworzenie dodatkowej kolumny na koncu i uzupelnienie jej zerami
  # pętla zapisująca w ostatniej kolumnie stosunek potwierdzonych przypadkow z dnia dzisiejszego i z dnia, ktory byl tydzien temu
  new_df.reset_index()
  for row in range(len(new_df)):
     first_data = new_df.iloc[row, -8]
     if first_data == 0:
       new_df.iloc[row, -1] = np.nan
     else:
       last_data = new_df.iloc[row, -2]
       new_df.iloc[row, -1] = round(last_data/first_data, 2)
     
  print(f"Top {number_of_countries} most total {key_word} countries and their increase in 7 days")
  return new_df.sort_values([this_day.strftime('%m/%d/%y').lstrip("0").replace("/0", "/")], ascending = False)
  
def show_country_plot(keyword, country_name):

  frame = get_all_data_to_plot(f"{keyword}")# funkcja użyta z pierwszego punktu
  print(frame)
  frame = frame.reset_index() 

  kraj = frame.loc[frame["Country/Region"] == f"{country_name}"] #pobranie danych na temat danego kraju

  kraj_ostatnie = kraj.iloc[0, -6:] # pobranie ostatnich 7 wartosci danego kraju
  kraj_wszystkie_dane = kraj.iloc[0, 2:] #pobranie wszystkich wartosci danego kraju
  wszystkie_lista = [] #inicjalizacja pustej listy
  for n in range(len(kraj_wszystkie_dane)): #pętla w celu pobrania wartosci z tabeli i umieszczeniu ich w tabeli
    value = kraj_wszystkie_dane[:][n] # iteracja po odpowiednich elementach tabeli 
    wszystkie_lista.append(value) # dodanie elementu do listy

  ostatnie_lista = []# jak wyzej z tym że zakres pętli jest mniejszy
  for n in range(len(kraj_ostatnie)):
    value = kraj_ostatnie[:][n]
    ostatnie_lista.append(value)

  # dates = pd.date_range('20200123', periods=(len(kraj_wszystkie_dane)))
  # print(dates[-1])


  numbers = [x for x in range(len(kraj_wszystkie_dane))] # utworzenie listy elementow potrzebnej do wykresu
  ostatnie_numbers = [i for i in range(len(kraj_wszystkie_dane)-6, len(kraj_wszystkie_dane))]
  to_predict = [n for n in range(len(kraj_wszystkie_dane), len(kraj_wszystkie_dane)+3)]
  to_predict.pop(0)

  y1 = np.array(wszystkie_lista).reshape(-1,1) # zamiana na typ listy numpy i wydzielenie każdego elementu jako osobną listę
  X1 = np.array(numbers).reshape(-1,1)
  y2 = np.array(ostatnie_lista).reshape(-1,1)
  X2 = np.array(ostatnie_numbers).reshape(-1, 1)
  
  regsr=LinearRegression() #utworzenie obiektu z biblioteki sklearn
  regsr.fit(X2,y2) #dopasowanie osi
  m= regsr.coef_  #pobranie współczynnikow do prostej
  c= regsr.intercept_
  new_y=[ m*i+c for i in np.append(X2,to_predict)] # ustalenie wartosci dla kazdego punktu na krzywej
  new_y=np.array(new_y).reshape(-1,1) #

  figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k') # ustalenie wielkosci okna i innych ustawien
  grid(color='b', linestyle='-', linewidth=0.2) # dodanie siatki w celu poprawienia widoczności wykresu
  plt.plot(X1, y1, marker='.') # wyrysowanie wartosci dla wszystkich danych
  plt.plot(np.append(X2,to_predict),new_y,color="red") # dodanie na wykres linii regresji opartej na danych z kilku ostatnich dni
  plt.xlabel("ilosc dni od poczatku pomiarow")
  plt.ylabel("ilosc przypadkow")
  plt.title(f"wykres dla {country_name}")
  plt.show() # wyswietlenie wykresu
  
def combined_deaths_confirmed():
  no_death_counter = 0
  index_counter = 0
  dfD = get_dataframe('deaths')
  columnsD = list(dfD.columns.values)
  dfC = get_dataframe('confirmed')
  columnsC = list(dfC.columns.values)
  dfR = get_dataframe('recovered')
  columnsR = list(dfR.columns.values)

  this_day = datetime.date.today() - datetime.timedelta(days=1)
  this_day = this_day.strftime('%m/%d/%y').lstrip("0").replace("/0", "/")
  df = dfD[[columnsD[1]] + [columnsD[-1]]] # utworzenie nowej tabeli z dwoma pierwszymi kolumnami jak z dfD
  
  df['confirmed'] = dfC[columnsC[-1]] # dodanie kolumny 'confirmed' na koniec nowej tabeli
  df['recovered'] = 0 # dodanie kolejnej kolumny 'recovered' 
  df = df.groupby(["Country/Region"]).sum() # pogrupowanie krajow
  
  df.rename(columns = {this_day: "deaths"}, inplace = True)
  for row in df.iterrows(): #pętla w celu pobrania właściwych wartości dla kolumny recovered
    value = dfR.loc[dfR["Country/Region"] == f"{row[0]}"][f"{this_day}"].sum()# przypisanie wartosci z danego kraju z ostatniego dnia
    df.iloc[index_counter, -1] = value # zapisanie wartosci do tabeli
    index_counter += 1
  df = df.loc[(df["confirmed"] > 50) & (df["deaths"] > 1) & (df["recovered"] > 1)]# w tabeli umieszczam przypadki gdzie jest co najmniej 50 potwierdzonych, 1 smiertelny i 1 wyleczony
  df['ratio'] = 0 # dodanie ostatniej kolumny 'ratio' (stosunek smiertelnych przypadkow do zakonczonych)
  for row in range(len(df)): # petla działajaca na elementach tabeli obliczająca 'ratio'
    confirmed = df.iloc[row, 1]
    death = df.iloc[row, 0]
    recovered = df.iloc[row, 2]
    df.iloc[row, -1] = round(death/(death+recovered), 3) # zapisywanie wartosci z dokladnoscia do 3 miejsc po przecinku
  return df
  
def show_global_ratio_map():
    geo_df = combined_deaths_confirmed()
    geo_df["size"] = geo_df["ratio"] 

    geo_df = geo_df.reset_index()
    fig = px.choropleth(geo_df, locations="Country/Region", locationmode='country names', 
                        color="ratio", hover_name="Country/Region", 
                        range_color= [0, 1], 
                        projection="natural earth",  
                        title='Smiertelnosc COVID19 na swiecie')
    fig.show()
