from bs4 import BeautifulSoup
import bs4
import pandas as pd

######################################################## 
# UTILS
########################################################

def getTrList(path):
    with open(path, 'r') as handle:
        raw = handle.read()

    soup = BeautifulSoup(raw, 'html.parser')

    return soup.find_all("tr")

def getUSstates():

    state_list = []

    for tr in getTrList('data/geographic/raw/us_states.txt'):
        gen = tr.children
        state_list.append(str.lower(next(gen).next.next))
        next(gen)
        state_list.append(str.lower(next(gen).next.next))

    return state_list

def getUSCities(max = 150):

    city_list = []

    for i, tr in enumerate(getTrList('data/geographic/raw/us_cities.txt')):
        if i == max :
            return city_list
        
        gen = tr.children
        next(gen)
        city = next(gen)
        while isinstance(city, bs4.element.Tag):
            city = city.next
        
        city_list.append(str.lower(city))
        
    return city_list

def getEuropeanCities():
    df = pd.read_csv('data/geographic/raw/europe-cities.csv')
    return df['city']

def getCentralAmericaCities():
    df = pd.read_csv('data/geographic/raw/central-america-cities.csv')
    return df['city']

def getCanadaCities():
    df = pd.read_csv('data/geographic/raw/canada-cities.csv')
    return df['city']

def getSwissCities():
    return ["zurich", "zürich", 'basel', 'geneva', 'bern', 'lausanne', 'lucerne', 'luzern', 'montreux', 'vevey', 'interlaken', 'lugano', 'zermatt', 'grindelwald', 'zoug']

def getSwissCantons():
    canton_list = []

    for tr in getTrList('data/geographic/raw/swiss_cantons.txt'):
        
        gen = tr.children
        next(gen)
        next(gen)
        next(gen)
        next(gen)
        next(gen)
        next(gen)
        next(gen)
        next(gen)
        next(gen)
        canton = next(gen)
        while isinstance(canton, bs4.element.Tag):
            canton = canton.next
        
        canton_list.append(str.lower(canton))

    canton_list.pop()
    canton_list.append('appenzell')
    return canton_list

def getFrenchCities(max = 100):
    
    city_list = []

    for i, tr in enumerate(getTrList('data/geographic/raw/french_cities.txt')):
        if i == max :
            return city_list
        
        gen = tr.children
        next(gen)
        next(gen)
        next(gen)
        next(gen)
        next(gen)
        city = next(gen)
        while isinstance(city, bs4.element.Tag):
            city = city.next
        
        city_list.append(str.strip(str.lower(city)))

    return city_list

def getFrenchDepartement():

    d_list = []

    for tr in getTrList('data/geographic/raw/french_departements.txt'):
        
        gen = tr.children
        next(gen)
        next(gen)
        next(gen)
        d = next(gen)
        while isinstance(d, bs4.element.Tag):
            d = d.next
        
        d_list.append(str.lower(d))

    d_list.remove('lot')
    d_list.remove('ain')
        
    return d_list

def getAsianCities():

    city_list = []

    for i, tr in enumerate(getTrList('data/geographic/raw/asian_cities.txt')):
        if i == max :
            return city_list
        
        gen = tr.children
        next(gen)
        city = next(gen)
        while isinstance(city, bs4.element.Tag):
            city = city.next
        
        city_list.append(str.strip(str.lower(city)))

    return city_list

def getAfricanCities():
    city_list = []

    for i, tr in enumerate(getTrList('data/geographic/raw/african_cities.txt')):
        if i == max :
            return city_list
        
        gen = tr.children
        next(gen)
        next(gen)
        next(gen)
        city = next(gen)

        while isinstance(city, bs4.element.Tag):
            city = city.next
        
        city_list.append(str.strip(str.lower(city)))

    return city_list

def getOceanianCities():
    city_list = []

    for i, tr in enumerate(getTrList('data/geographic/raw/oceanian_cities.txt')):
        if i == max :
            return city_list
        
        gen = tr.children
        next(gen)
        city = next(gen)

        while isinstance(city, bs4.element.Tag):
            city = city.next
        
        city_list.append(str.strip(str.lower(city)))

    return city_list

def getSouthAmericaCities():
    city_list = []

    for i, tr in enumerate(getTrList('data/geographic/raw/south_american_cities.txt')):
        if i == max :
            return city_list
        
        gen = tr.children
        next(gen)
        next(gen)
        next(gen)
        city = next(gen)

        while isinstance(city, bs4.element.Tag):
            city = city.next
        
        city_list.append(str.strip(str.lower(city)))

    return city_list

def getCountries():
    country_list = []

    for i, tr in enumerate(getTrList('data/geographic/raw/countries.txt')):
        
        gen = tr.children
        next(gen)
        gen = next(gen).children
        next(gen)
        country = next(gen)

        while isinstance(country, bs4.element.Tag):
            country = country.next
        
        country_list.append(str.strip(str.lower(country)))

    return country_list