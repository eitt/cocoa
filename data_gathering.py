import os
import json
import requests
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
import numpy as np 
import math
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime as dt


#Dataset adquisiction
def rename_columns(df):
    renamed_columns = ["Date","All_Sky_Irradiance","Clear_Sky_Irradiance", "Max_Temp","Min_Temp","Average_Temp", "Precipitation", "Relative Humidity", "Wind Speed"]
    df.columns = renamed_columns
    return df
variables = ["ALLSKY_SFC_PAR_TOT","CLRSKY_SFC_PAR_TOT","T2M_MAX","T2M_MIN","T2MDEW","PRECTOTCORR","RH2M","WS2M"]
def get_api_data(latitude, longitude, start_date, end_date, variables):
    parameters = ','.join(variables)
    base_url = r"https://power.larc.nasa.gov/api/temporal/daily/point?parameters={parameters}&community=AG&longitude={longitude}&latitude={latitude}&start={start}&end={end}&format=JSON"
    api_request_url = base_url.format(parameters=parameters, longitude=longitude, latitude=latitude, start=start_date, end=end_date)
    response = requests.get(url=api_request_url, timeout=30.0)  
    content = json.loads(response.content.decode('utf-8'))  
    for keys in content['properties']['parameter'].keys():
        content['properties']['parameter'][keys] = transform_date_keys(content['properties']['parameter'][keys])
    return content['properties']['parameter'] 
    
def transform_date_keys(dictionary):
    transformed_dict = {}  
    for key, value in dictionary.items():
        transformed_key = key[:4] + '-' + key[4:6] + '-' + key[6:]  
        transformed_dict[transformed_key] = value  
    return transformed_dict  
def get_data_for_locations(locations, variables):
    today = datetime.today()
    date_180_days_ago = today - timedelta(days=365)
    num_days = (today - date_180_days_ago).days
    start_date = date_180_days_ago.strftime('%Y%m%d')
    end_date = today.strftime('%Y%m%d')

    dfs = []  

    for latitude, longitude in locations:
        parameters = get_api_data(latitude, longitude, start_date, end_date, variables)
        reshaped_data = {'date': []}
        for var in variables:
            reshaped_data[var] = []

        for date in pd.date_range(start=start_date, end=end_date, freq='D'):
            date_str = date.strftime('%Y-%m-%d')  
            if any(date_str in parameters[var] for var in variables):  
                reshaped_data['date'].append(date)
                for var in variables:
                    reshaped_data[var].append(parameters[var].get(date_str, None))

        # Create a DataFrame from the reshaped data and append it to the list
        dfs.append(pd.DataFrame(reshaped_data))

    # Concatenate all dataframes in the list
    df = pd.concat(dfs, ignore_index=True)
    return df

def replace_negative_values(df):
    for column in df.columns:
        if column != 'Date':
            for i in range(len(df)):
                if df[column][i] < 0:
                    if i-3 >= 0 and i+3 < len(df): 
                        # If there are 3 values both before and after
                        surrounding_values = df[column][i-3:i+4]
                        if (surrounding_values > 0).any(): # check if there are any positive values in the surrounding values
                            df[column][i] = np.mean(surrounding_values[surrounding_values > 0]) # calculate the mean of positive surrounding values
                        else: 
                            # If there are no positive values in the surrounding values, replace with average of entire column
                            df[column][i] = np.mean(df[df[column] > 0][column])
                    else: 
                        # If not, replace with average of entire column
                        df[column][i] = np.mean(df[df[column] > 0][column]) 
    return df
#Plotting information
def plot_data(df):
    df['Date'] = pd.to_datetime(df['Date'])
    start_date = st.date_input("Start date", df['Date'].min())
    end_date = st.date_input("End date", df['Date'].max())
    df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]
    options = df.columns[1:].tolist()  
    selected_trace = st.selectbox("Select a trace to display:", options)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df[selected_trace], name=selected_trace))

    fig.update_layout(
        title="Weather Parameters Over Time",
        xaxis_title="Date",
        yaxis_title="Values",
        legend_title="Parameters",
    )

    st.plotly_chart(fig)
def plot_df(df,df2):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df2[df2.columns[0]], mode='lines', name=df2.columns[0]))
    fig.update_layout(title=str(df2.columns[0])+' over Time', xaxis_title='Date', yaxis_title=df.columns[0])
    st.plotly_chart(fig) 

#Biomass estimation
def fTemp(Tbase, Topt, T):
    ftemp = np.zeros(len(T))  # Initialize an array of zeros with the same length as T

    # Calculate FTemp values based on temperature conditions

    # Set FTemp to 0 for temperatures below Tbase
    ftemp[T < Tbase] = 0

    # Set FTemp to the relative value between Tbase and Topt for temperatures within that range
    ftemp[(T >= Tbase) & (T <= Topt)] = (T[(T >= Tbase) & (T <= Topt)] - Tbase) / (Topt - Tbase)

    # Set FTemp to 1 for temperatures above Topt
    ftemp[T > Topt] = 1

    df_ftemp = pd.DataFrame(ftemp, columns=['df_ftemp'])  # Return ftemp as a DataFrame with a single column 'FTemp'


    return df_ftemp
def fHeat(Theat, Textreme, Tmax):
    fheat = np.ones(len(Tmax))  # Initialize an array of ones with the same length as Tmax

    # Calculate FHeat values based on temperature conditions

    # Set FHeat to 1 for temperatures less than or equal to Theat
    fheat[Tmax <= Theat] = 1

    # Set FHeat to the relative value between Theat and Textreme for temperatures within that range
    fheat[(Tmax > Theat) & (Tmax <= Textreme)] = 1 - np.round((Tmax[(Tmax > Theat) & (Tmax <= Textreme)] - Theat) / (Textreme - Theat), 3)

    # Set FHeat to 0 for temperatures greater than Textreme
    fheat[Tmax > Textreme] = 0

    return pd.DataFrame(fheat, columns=['FHeat'])  # Return fheat as a DataFrame with a single column 'FHeat'
def fSolar(fsolar, i50a, i50b, tsum, tbase, temp):
    temp = [(abs(i - tbase) + (i - tbase)) / 2 for i in temp]  # Calculate the temperature values based on tbase
    tt = []  # Initialize an empty list to store cumulative temperature values
    for i in range(len(temp)):
        if i == 0:
            tt.append(0.0)  # Append 0.0 for the first temperature value
        else:
            tt.append(temp[i] + tt[i-1])  # Calculate and append cumulative temperature values
    
    growth = [fsolar / (1 + math.exp(-0.01 * (i - i50a))) for i in tt]  # Calculate growth values
    senescence = [fsolar / (1 + math.exp(0.01 * (i - (tsum - i50b)))) for i in tt]  # Calculate senescence values
    
    result = [growth[i] if senescence[i] >= growth[i] else senescence[i] for i in range(len(senescence))]  # Determine final result based on growth and senescence values
    
    return pd.DataFrame(result, columns=['df_fsolar'])  # Return result as a DataFrame with a single column 'FSolar'
def RN(alfa,RS,RSO,SIGMA,TMIN,TMAX,RHMEAN,temp,Altitude,n,E,Cp,G,WS2M):    
    eTMAX = [0.6108*math.exp((17.27*TMAX[i])/(TMAX[i]+237.3)) for i in range(len(TMAX))] #O
    eTMIN = [0.6108*math.exp((17.27*TMIN[i])/(TMIN[i]+237.3)) for i in range(len(TMIN))] #N
    ea = [(RHMEAN[i]/100)*((eTMAX[i]+eTMIN[i])/2) for i in range(len(RHMEAN))] #M
    TMAXK = [TMAX[i]+273.16 for i in range(len(TMAX))] #L
    TMINK = [TMIN[i]+273.16 for i in range(len(TMIN))] #J
    RNS = [(1-alfa)*RS[i] for i in range(len(RS))] #D
    RNL =[SIGMA*((TMINK[i]**4+TMAXK[i]**4)/2)*(0.34-0.14*(ea[i]**0.5))*(1.35*(RS[i]/RSO[i])-0.35) for i in range(len(TMIN))] #C
    RN = [RNS[i]-RNL[i] for i in range(len(TMIN))] #A
    A = [4098*(0.6108*math.exp(17.27*temp[i]/(temp[i]+237.3)))/(temp[i]+237.3)**2for i in range(len(temp))] #ETO R
    es = [(eTMAX[i]+eTMIN[i])/2 for i in range(len(TMIN))] #Eto P
    es_ea = [es[i]-ea[i] for i in range(len(temp))] #Eto M
    P = 101.3*((293-0.0065*Altitude)/293)**5.26 #Eto H
    y = ((Cp*P)/(n*E)) # Eto F
    ETO = [(0.408*A[i]*(RN[i]- G)+y*(900/(temp[i]+273))*WS2M[i]*(es_ea[i]))/(A[i]+y*(1+0.34*WS2M[i]))for i in range(len(temp))] #eto B 
    KS = [1]*(int(len(ETO)*5/18))+[1.05]*(int(len(ETO)*13/18)) #La igualo a longitud de ETO pero en el Excel aparece con longitud de 130 debo arreglarlo para múltiples periodos
    ETC = [KS[i]*ETO[i] for i in range(len(ETO))]
    return ETO, ETC
def fWater(Swater, TAW,RAW,p, alfa, SIGMA, Precipitation,RS,RSO,TMIN,TMAX,RHMEAN,temp,Altitude,n,E,Cp,G,WS2M,la,S):
    ETO , ETC = RN(alfa,RS,RSO,SIGMA,TMIN,TMAX,RHMEAN,temp,Altitude,n,E,Cp,G,WS2M)
    #Water gain
    # Irrigation
    Irrigation = [0]*len(Precipitation)
    # Capillary Rise
    CapillaryRise = [0]*len(Precipitation)
    #Water gain
    WaterGain = [Precipitation[i] + Irrigation[i] +CapillaryRise[i] for i in range(len(Precipitation))]
    
    #Water Loss
    #Evaporated
    Evaporated = [ ETO[i]*0.2 if  Precipitation[i]>= ETO[i]*0.2 else Precipitation[i] for i in range(len(Precipitation))] #Simplyfied
    #Drain
    Drain = [((Precipitation[i]-la)**2)/(Precipitation[i]-la+S) if (Precipitation[i]>la) else 0 for i in range(len(Precipitation))]
    
    #Water Loss
    WaterLoss_DP = [ETC[i]+Drain[i]+Evaporated[i] for i in range(len(Precipitation))]
    #Depletion size
    Depletion = [0.0]*len(Precipitation)
    #Depletion lagged 1 step
    Depletion_lag = Depletion

    #Depletion
    for i in range(len(Depletion)):
        
        if i==0:
            Depletion[i] = 0.0
            Depletion_lag[i] = 0.0

        else:
            Depletion_lag[i] = Depletion[i-1]
            if WaterLoss_DP[i]-WaterGain[i]+Depletion_lag[i]>0:
                if WaterLoss_DP[i]-WaterGain[i]+Depletion_lag[i]>TAW:
                    Depletion[i] = TAW
                    # Depletion_lag[i] = Depletion[i-1]
                else:
                    Depletion[i] = WaterLoss_DP[i]-WaterGain[i]+Depletion_lag[i]  
                    # Depletion_lag[i] = Depletion[i-1]
            else:
                Depletion[i] = 0
                
                
    #Paw
    PAW=[TAW-Depletion[i] if TAW > Depletion[i] else 0 for i in range(len(Depletion))]
    #Arid
    ARID = [1-min(ETO[i],0.096*PAW[i])/ETO[i] for i in range(len(PAW))]
    

    # return pd.DataFrame(ETC)
    return pd.DataFrame([1-Swater*ARID[i] for i in range(len(ARID))],columns = ['FWater'])


# Crop modeling
def calculate_biomass(df_concat, CO2, RUE, harvest_index):
    
    # calculate biomass_rate
    biomass_rate = CO2 * RUE * df_concat.iloc[:,0] * df_concat.iloc[:,2] * df_concat.iloc[:,[3,1]].min(axis=1)

    # calculate cumulative_biomass
    cumulative_biomass = biomass_rate.cumsum()
    cocoa_yield = cumulative_biomass.max() * harvest_index

    # create a DataFrame to return
    df_result = pd.DataFrame({
        'biomass_rate': biomass_rate,
        'cumulative_biomass': cumulative_biomass
    })

    return df_result, cocoa_yield

    
#pages navigation
def data_acquisition():
    st.title("Data Acquisition")
    latitude = st.number_input("Insert Latitude:", min_value=-90.0, max_value=90.0, value=6.8773, step=0.0001)
    longitude = st.number_input("Insert Longitude:", min_value=-180.0, max_value=180.0, value=-73.4723, step=0.0001)
    if st.button('Retrieve data from the last year'):
        

        df_temporal = get_data_for_locations([(latitude, longitude)],variables)

        df_temporal = rename_columns(df_temporal)
        df_temporal = replace_negative_values(df_temporal)
        df = df_temporal.tail(180).reset_index().drop("index",axis=1).copy()
        # Save the DataFrame in session state so we can access it from other pages
        st.session_state.df = df
        st.markdown("This is the retrieved dataset")
        st.dataframe(df_temporal)
        

def data_visualization():
    st.title("Data Visualization")
    # Check if the DataFrame has been defined in session state
    if 'df' in st.session_state:
        # Access the DataFrame from session state
        df = st.session_state.df
        # Now you can use the DataFrame for data visualization
        # For example, let's display the DataFrame
        plot_data(df)
        st.markdown("You can donwload the dataset in the folowing buttom")
        csv = df.to_csv(index=False)
        st.download_button(
        "Download data as CSV",
        data=csv,
        file_name="data.csv",
        mime="text/csv")
        st.dataframe(df)
        
    else:
        # If the DataFrame has not been defined, display a message asking the user to visit the Data Acquisition page
        st.error('Please go to the Data Acquisition page to load the data first.')

def environmental_factors():
    #Loading the information
    df = st.session_state.df
    st.title("Environmental Factors")
    st.markdown("The biomass density estimation requieres the envronmental factor analysis and estimation. Next, there are five equations to estimate Biomass density.")
    st.title("Function of Temperature")
    st.markdown("1. `fTemp(Tbase, Topt, T)`: This function calculates the temperature factor (FTemp) based on given temperature values and thresholds. It determines the suitability of temperature conditions for a specific range and returns a DataFrame containing the FTemp values.")
    Tbase = st.number_input('Enter base temperature', value = 10.0)
    Topt = st.number_input('Enter optimal temperature',value = 24.0)
    df_ftemp = fTemp(Tbase, Topt, df['Average_Temp'])

    st.markdown("A visual representation")
    plot_df(df,df_ftemp)
    st.markdown("The dataframe output")
    st.dataframe(df_ftemp)
    st.title("Function of Heat")
    st.markdown("2. `fHeat(Theat, Textreme, Tmax)`: The fHeat function computes the heat factor (FHeat) by evaluating maximum temperature values against specified thresholds. It assesses the heat stress levels and generates a DataFrame with the FHeat values indicating the severity of heat conditions.")
    Theat = st.number_input('Enter heat temperature', value = 32.0)
    Textreme = st.number_input('Enter the extreme temperature',value = 38.0)
    df_fheat = fHeat(Theat, Textreme, df['Max_Temp'])
    st.markdown("A visual representation")
    plot_df(df,df_fheat)
    st.markdown("The dataframe output")
    st.dataframe(df_fheat)
    st.title("Function of Solar")
    st.markdown("3. `fSolar(fsolar, i50a, i50b, tsum, tbase, temp)`: This function calculates the solar radiation factor (FSolar) based on temperature and growth parameters. It considers the cumulative temperature values and sigmoid functions to estimate solar radiation effects on growth. The resulting FSolar values are returned in a DataFrame.")
    fsolarmax = st.number_input('Enter the fsolar max', value = 0.94)
    i50a = st.number_input('Enter i50a', value = 680.0)
    i50b = st.number_input('Enter i50b', value = 200.0)
    tsum = st.number_input('Enter tsum', value = 2764.0)
    tbasesolar = st.number_input('Enter tbase', value = 10.0)
    df_fsolar = fSolar(fsolarmax, i50a, i50b, tsum, tbasesolar, df['Average_Temp'])
    st.markdown("A visual representation")
    plot_df(df,df_fsolar)    
    st.markdown("The dataframe output")
    st.dataframe(df_fsolar)
    st.title("The function of Water")
    st.markdown("This is the modified version of water function in the original SIMPLE crop model. In This function we estimate the effect of water stress considering and adjustec ETc. These are the two functions to estimate water stress:")
    st.title("The evapotranspiration function")
    st.markdown("4. `RN(alfa, RS, RSO, SIGMA, TMIN, TMAX, RHMEAN, tempe, Altitude, n, E, Cp, G, WS2M)`: The RN function calculates reference evapotranspiration (ETO) and adjusted crop evapotranspiration (ETC) using various meteorological parameters. It employs formulas and constants to estimate evapotranspiration rates for agricultural and environmental analyses. The function returns two lists, ETO and ETC, representing the calculated values.")
    alfa = st.number_input('Enter the alfa', value = 0.23)
    
    SIGMA = st.number_input('Enter the SIGMA', value = 0.00000000490300000000403)

    tempe = st.number_input('Enter the tempe', value = 0.94)
    Altitude = st.number_input('Enter the Altitude', value = 658.0)
    n = st.number_input('Enter the n', value = 2.45)
    E = st.number_input('Enter the E', value = 0.622622622)
    Cp = st.number_input('Enter the Cp', value = 1.013)
    G = st.number_input('Enter the G', value = 0.0  )
    Swater = st.number_input('Enter the Swater', value = 0.6053)
    Taw = st.number_input('Enter the TAW', value = 140.0)
    Raw = st.number_input('Enter the RAW', value = 4242.0)
    pp = st.number_input('Enter the p', value = 0.3)
    S = st.number_input('Enter the S', value = 75.87)
    la = st.number_input('Enter the la', value = 115.1)
    #Parameters derived from dataframe
    RS = df['All_Sky_Irradiance']
    RSO = df['Clear_Sky_Irradiance']
    TMIN = df['Min_Temp']
    TMAX = df['Max_Temp']
    RHMEAN = df['Relative Humidity']
    tempe = df['Average_Temp']
    WS2M = df['Wind Speed']
    Precipitation = df['Precipitation']
    
    df_water = fWater(Swater, Taw,Raw,pp, alfa, SIGMA, Precipitation,RS,RSO,TMIN,TMAX,RHMEAN,tempe,Altitude,n,E,Cp,G,WS2M,la,S)
    
    st.markdown("5. `fWater(Swater, TAW, RAW, p, alfa, SIGMA, Precipitation, RS, RSO, TMIN, TMAX, RHMEAN, temp, Altitude, n, E, Cp, G, WS2M, la, S)`: This function computes the water factor (FWater) by considering various water-related parameters, such as soil moisture, available water, precipitation, and evapotranspiration. It evaluates water gain, water loss, depletion, and available water components to assess the suitability of water supply. The function returns a DataFrame containing the FWater values.")
    # Add your environmental factors code here
    st.markdown("A visual representation")
    plot_df(df,df_water)    
    st.markdown("The dataframe output")
    st.dataframe(df_water)
    # save the dataframes as global variables
    st.session_state['df_fheat'] = df_fheat
    st.session_state['df_fsolar'] = df_fsolar
    st.session_state['df_ftemp'] = df_ftemp
    st.session_state['df_water'] = df_water
    st.session_state['df'] = df

    
    
def simple_model():
    st.title("SIMPLE Model")
    #Loading dataframe
        # check if dataframes are in session state
    for var in ['df_fheat', 'df_fsolar', 'df_ftemp', 'df_water']:
        if var not in st.session_state:
            st.error(f'Please go to the Data Acquisition page to load the {var} data first.')
            return

    # load dataframes from session state
    df_fheat = st.session_state['df_fheat']
    df_fsolar = st.session_state['df_fsolar']
    df_ftemp = st.session_state['df_ftemp']
    df_water = st.session_state['df_water']
    df = st.session_state['df']
    
    # Add your SIMPLE model code here
    CO2 = st.number_input('Enter the $CO_2$', value = 6.7249)
    RUE = st.number_input('Enter the RUE', value = 0.296)
    harvest_index = st.number_input('Enter the harvest index', value = 0.07066)
    df_concat = pd.concat([df_ftemp, df_fheat, df_fsolar, df_water],axis=1)
    st.dataframe(df_concat)
    [df_yield,cocoa_yield] = calculate_biomass(df_concat, CO2, RUE, harvest_index)
    df_yield['Date'] = df["Date"]
    df_yield = df_yield[['Date','biomass_rate','cumulative_biomass']]
    st.markdown("A visual representation")
    plot_data(df_yield)
    st.markdown(f"The expected crop yield is:  {round(cocoa_yield,2)}")
    st.dataframe(df_yield)

def lp_model_analysis():
    st.title("LP Model Analysis")
    # Add your LP model analysis code here

def about():
    st.title("About")
    # Add your about info here

# Create a dictionary of pages
pages = {
    "Data Acquisition": data_acquisition,
    "Data Visualization": data_visualization,
    "Environmental Factors": environmental_factors,
    "SIMPLE Model": simple_model,
    "LP Model Analysis": lp_model_analysis,
    "About": about,
    }

# Create a sidebar for navigation
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(pages.keys()))

# Display the selected page
pages[selection]()
