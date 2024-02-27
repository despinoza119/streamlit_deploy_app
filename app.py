import os
import numpy as np
import streamlit as st
import openai
from PIL import Image
from langchain.chat_models import ChatOpenAI
from langchain.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
import plotly.express as px
import folium
from folium import GeoJson
import geopandas as gpd
from openai import OpenAI
from datetime import datetime
import sqlite3
import pandas as pd

# GeoJSON Information - SPAIN
f = r"esp_maps/ESP_adm2.shp"
shapes = gpd.read_file(f)
shapes = shapes.rename(columns={'NAME_2': 'Provincia'})
shapes['Precio'] = np.round(np.random.uniform(200, 250, len(shapes)), 2)
fecha_actual = datetime.now().date()


# CHAT-GPT API Key
credentials_file = "credentials.txt"

if os.path.exists(credentials_file):
    with open(credentials_file, "r") as file:
        api_key = file.read().strip()
        os.environ["OPENAI_API_KEY"] = api_key
else:
    raise FileNotFoundError(f"The file {credentials_file} was not found.")



st.sidebar.image("images/agia.png",caption="¡Bienvenido a Intelligent Assistant!")

def Home():
    st.markdown("<h1 style='text-align: center;'>Demo</h1>", unsafe_allow_html=True)
    st.markdown("Descubre la potencia de nuestro demo, donde puedes realizar consultas sobre precios de cultivo. Sumérgete en la experiencia de obtener insights con gráficos en tiempo real, ofreciéndote una visión dinámica y detallada de tus datos. ¡Explora las posibilidades y toma decisiones informadas de manera intuitiva con nuestra solución web!")
    st.markdown("Explore the power of our demo, where you can make queries about crop prices. Immerse yourself in the experience of gaining insights with real-time graphs, providing you with a dynamic and detailed view of your data. Discover the possibilities and make informed decisions intuitively with our web solution!")
    
    st.sidebar.markdown("<h1 style='text-align: center; font-size: small;'>Powered by AGIA®</h1>", unsafe_allow_html=True)

    image = Image.open("images/infra.png")
    st.image(image,caption='Estadisticas')



def Chat():
    st.title("Abastores Assistant")
    st.write("¡Bienvenido al ChatBot! Aquí puedes hacer preguntas sobre datos y análisis relacionados con Precios de Cultivos. "
         "Recuerda que el ChatBot está en fase de pruebas, por lo que puede que no tenga respuestas para todas las preguntas.")

    db = SQLDatabase.from_uri("sqlite:///test_prices.db")
    llm = ChatOpenAI(temperature=0,model_name='gpt-3.5-turbo')
    cadena = SQLDatabaseChain(llm = llm, database = db, verbose=False)
    formato = """
            Dada una pregunta del usuario:
            0. Convierte cualquier fecha al formato YYYY-MM-DD para mejor comprension del query
            1. crea una o varias consultas en las tablas de sqlite3 segun la pregunta (no le pongas limit a la consulta), si no se puede crear un query comunica que no tenemos data disponible de esos temas
            2. revisa los resultados, considera todas las columnas importantes al dar la respuesta
            3. devuelve el dato o los datos como un listado (incluye los valores de todas las columnas que consideres relevantes)
            4. añade un resumen con un analisis relevante de los datos mostrados y si hay noticias relacionadas en español
            5. dale una estructura a la respuesta para que sea comoda de leer para el lector
            6. Si no sabes que responder, pregunta al usuario para obtener mas informacion, si no puedes obtener mas informacion, responde con un mensaje de error
            7. Siempre responde en español, y con palabras no tecnicas , un lenguaje cordial
            #{question}
            """

    def consulta(input_usuario):
        consulta = formato.format(question = input_usuario)
        try:
            resultado = cadena.run(consulta)
            return resultado
        
        except Exception as e:
            # If there's an exception, prompt ChatGPT for more information
            error_message = f"No tenemos datos disponibles sobre ese tema. ¿Puede proporcionar más detalles o hacer otra pregunta relacionada?"

        ### Uncomment the following code to use ChatGPT to handle the error message
            # client=OpenAI()
            # response = client.chat.completions.create(
            #     model="gpt-3.5-turbo",
            #     messages =[
            #         {"role": "assistant", "content": "Eres un asistente que se activa cuando el usuario no ha consultado data que tenemos disponible dentro de la base de datos."}, 
            #         {"role": "system", "content": "Puedes comunicarle al usuario que los datos presentes en la base de datos son de precios de cultivo y noticias"}, 
            #         {"role": "system", "content": "Puedes guiar a los usuarios para preguntas generales, que no tengan que ver con la base de datos, hacelo saber al usuario."}, 
            #         {"role": "system", "content": "Responde la pregunta del usuario:"}, 
            #         {"role": "system", "content":input_usuario}
            #     ]
            # )

            # # Append ChatGPT's response to the error message
            # error_message += f"\nChatGPT: {response.choices[0].message.content}"
            
            return error_message

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input()

    if prompt:
        with st.chat_message("You"):
            st.markdown(prompt)

        st.session_state.messages.append({"role": "Abastores", "content": prompt})

        response = consulta(prompt)

        with st.chat_message("assistant"):
            st.markdown(response)
            st.session_state.messages.append({"role": "Abastores", "content": response})


# Choropleth map
def make_choropleth(input_df, input_id, input_column, input_color_theme):
    choropleth = px.choropleth(input_df, locations=input_id, color=input_column, locationmode="USA-states",
                               color_continuous_scale=input_color_theme,
                               range_color=(0, max(df_selected_year.population)),
                               scope="usa",
                               labels={'population':'Population'}
                              )
    choropleth.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        margin=dict(l=0, r=0, t=0, b=0),
        height=350
    )
    return choropleth




def Dashboards_1():
    st.markdown("<h1 style='text-align: center;'>Analítica en Vivo</h1>", unsafe_allow_html=True)
    dashboard_selector = st.selectbox("Selecciona un Dashboard", ["Mapa de precios", "Series Temporales"])
    

    if dashboard_selector == "Mapa de precios":
        st.write(f"Precios en tiempo real (Actualizado el {fecha_actual.strftime('%d/%m/%Y %H:%M:%S')})")
        m = folium.Map(location=[39.6, -4], zoom_start=6)

        GeoJson(
            data=shapes.to_json(),
            name='geojson',
            tooltip=folium.features.GeoJsonTooltip(fields=['Provincia', 'Precio'], labels=True, sticky=False),
        ).add_to(m)

        map_html_path = 'folium_choropleth_map.html'
        m.save(map_html_path)

        with open(map_html_path, 'r') as f:
            folium_html = f.read()

        st.components.v1.html(folium_html, width=700, height=500)

    elif dashboard_selector == "Series Temporales":
        st.write("Contenido del Dashboard 2")
        # Conectar a la base de datos SQLite
        conn = sqlite3.connect("test_prices.db")

        # Supongamos que tu tabla se llama 'precio_del_cultivo'
        query = "SELECT Fecha, Provincia, Precio FROM precio_del_cultivo WHERE Provincia = 'Madrid'"
        df_madrid = pd.read_sql(query, conn)

        # Convertir la columna 'Fecha' a tipo datetime
        df_madrid['Fecha'] = pd.to_datetime(df_madrid['Fecha'])

        # Filtrar por el año 2019
        df_madrid_2019 = df_madrid[df_madrid['Fecha'].dt.year == 2019]

        # Cerrar la conexión a la base de datos
        conn.close()

        # Configurar la aplicación de Streamlit
        st.title('Análisis de Serie Temporal para Madrid - 2019')
        st.markdown('Visualización de la serie temporal de precios en Madrid para el año 2019')

        # Crear un gráfico de serie temporal interactivo con Plotly Express
        fig_total = px.line(df_madrid_2019, x='Fecha', y='Precio', title='Precio a lo largo del tiempo en Madrid - 2019')

        # Mostrar el gráfico total
        st.plotly_chart(fig_total)

        # Crear un gráfico separado para cada mes en 2019
        for month in df_madrid_2019['Fecha'].dt.month.unique():
            df_month = df_madrid_2019[df_madrid_2019['Fecha'].dt.month == month]
            fig_month = px.line(df_month, x='Fecha', y='Precio', title=f'Precio en Madrid - Mes {month} - 2019')
            st.plotly_chart(fig_month)
            
page_name_to_funcs = {
    "Home": Home,
    "Chat": Chat,
    "Dashboards": Dashboards_1
}

selected_page = st.sidebar.selectbox("Go to", list(page_name_to_funcs.keys()))
page_name_to_funcs[selected_page]()