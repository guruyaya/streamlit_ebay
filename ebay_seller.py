import streamlit as st
import pandas as pd
import numpy as np
from urllib.request import urlopen
import json
from datetime import datetime, timedelta, time
import joblib

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

from ebay_auction_sales_prediction import * # all our libraries needed for the pickle model

def get_dollar_value():
    check_days = (datetime.now(), datetime.now() - timedelta(days=1), 
                datetime.now() - timedelta(days=2))
    
    format_date = lambda date: date.strftime('%Y-%m-%d')
    try:
        with open(dir_path + '/usd_value.cache', 'r') as f:
            json_data = json.load(f)
            if json_data['date'] not in ([format_date(date) for date in check_days]):
                raise Exception('File is too old') # this will be caught by except statement
    except Exception:
        st.write('Loading usd rates from remote...')
        with open(dir_path + '/usd_value.cache', 'w') as f:
            r = urlopen("https://api.exchangeratesapi.io/latest?base=USD")
            json_data = json.load(r)
            f.write(json.dumps(json_data))
    
    
    return {
        'US': 1, 'C': json_data['rates']['CAD'], 'AU': json_data['rates']['AUD'], 
        'EUR': json_data['rates']['EUR'], 'GBP': json_data['rates']['GBP']
    }

def gen_countries_list():
    with open(dir_path + '/country-by-name.json') as f:
        countries_json = json.load(f)   
    countries = [c['country'] for c in countries_json]

    # popping the main countries to the top of the list
    main_countries = ['United States', 'Canada', 'China', 'United Kingdom']
    for c in main_countries: countries.remove(c)
    return main_countries + countries

def load_phone_models():
    return pd.read_csv(dir_path + '/iphone_data_plus_phrases.csv'
                  ,parse_dates= ['Release date']
                  ,usecols= ['Model', 'Release date','Price at launce', 'discontinued', 'support ended', 'phrase']
    )

def get_model_information(model, dim):
    dim['Price at launce'] = dim['Price at launce'].str.replace(r'[,$]', '').astype('float64')
    dim['support ended'] = pd.to_datetime((dim['support ended'].str.findall('[A-Z][a-z]+\s\d+,\s\d+').str[0]))
    dim['discontinued'] = pd.to_datetime(dim['discontinued'].str.findall('[A-Z][a-z]+\s\d+,\s\d+').str[0])

    return dim[ dim.Model == model ]
def main():
    # Preparing the data needed for this project
    with open(dir_path + '/saved_model.joblib', 'rb') as f:
        final_model = joblib.load(f)
    
    dollar_values = get_dollar_value()
    countries = gen_countries_list()

    currencies_to_our_format = {'USD': 'US', 'CAD': 'C', 'EUR': 'EUR', 'AUD': 'AU', 'GBP': 'GBP'}
    # Loading dim df

    # front end elements of the web page 
    html_temp = """ 
    <style>
        header { background-color:yellow;padding:13px; }
        h1 { color:black;text-align:center; }
        h2 { color: red }
    </style>
    <header> 
        <h1>Predict your iPhone auction success</h1> 
    </header>
    """
    

    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
    st.markdown("## Item Info Model")

    # following lines create boxes in which user can enter data required to make prediction 
    dim = load_phone_models()
    iphone_model = model_info = st.selectbox('iPhone', dim['Model'].values, index=5)
    model_info = get_model_information(iphone_model, dim)

    item_condition_options = ['For parts or not working', 'Used', 'Seller refurbished', 'Open box', 'New']
    condition = st.selectbox('Item Condition',item_condition_options, index=1) 

    item_location = st.selectbox('Item Country', countries, index=0)
    st.markdown("## Sale Details")
    now = datetime.now()

    currency = st.selectbox('Currency', ['USD', 'EUR', 'CAD', 'AUD', 'GBD'])
    starting_bid = st.number_input("Starting Bid", min_value=0.01)

    start_date = st.date_input("When will you start your sale",
                    value=now, 
                    min_value=now - timedelta(days=31),
                    max_value=now + timedelta(days=356),)
    
    start_time = st.slider("Start Time", value=time((now.hour + 1) % 24, now.minute))
    sale_duration = st.selectbox('Sale Duration',(1, 3, 5, 7, 10), index=3)
    start_datetime =  datetime.combine(start_date, start_time)
    
    end_datetime = start_datetime + timedelta(days=sale_duration)
    st.write(f"Your sale is scheduled to start as {start_datetime} and end at {end_datetime}")

    accept_returns = st.checkbox("Do you accept returns", value=True)
    is_free_shipping  = st.checkbox("Do you offer free shipping", value=True)

    st.markdown("## Seller information")
    member_from = st.selectbox('Seller Location', countries, index=0)
    seller_rating = st.number_input('Seller Rating', min_value=0)
    seller_seniority = st.date_input("When did the seller register on eBay",
                    value=now, 
                    min_value=now - timedelta(days=10950), # 365 * 30
                    max_value=now)

    seller_seniority = datetime(seller_seniority.year, seller_seniority.month, seller_seniority.day)

    positive_feedback = st.number_input('Seller Feedback', min_value=0)
    how_many_feedbacks = st.number_input('Seller Feedback count', min_value=0)

    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        our_currency = currencies_to_our_format[ currency ]
        currency_dollar_value = dollar_values[our_currency]

        test_case = pd.DataFrame([{

            'positive_feedback': positive_feedback,
            
            'seller_rating':seller_rating, 
            
            'all_votes': how_many_feedbacks,
            'member_since': seller_seniority,

            'item_condition': condition,
            'date_started': start_datetime,
            'date_ended': end_datetime,
            'duration': sale_duration,
            'starting_bid_price_currancy': our_currency,
            'starting_bid_price_value': starting_bid,
            'starting_bid_usd': currency_dollar_value,

            'IsReturnsAccepted': int(accept_returns), 
            'HasFreeShipping': is_free_shipping,

            'member_from': member_from,
            'item_location': item_location,

            'discontinued': model_info.iloc[0]['discontinued'],
            'support ended': model_info.iloc[0]['support ended'],
            'Release date': model_info.iloc[0]['Release date'],
            'Model': model_info.iloc[0]['Model'],
            'phrase': model_info.iloc[0]['phrase'],
            'Price at launce': model_info.iloc[0]['Price at launce'], 
        }])

        test_case['item_condition'] = pd.Categorical(test_case.item_condition, 
            categories=['For parts or not working', 'Used', 'Seller refurbished', 'Open box', 'New'], 
            ordered=True)

        success_chance = final_model.predict_proba(test_case)[0][1] * 100
        st.success('You have {:.2f}% chance of selling this iPhone'.format(success_chance))
     
if __name__=='__main__': 
    main()




