import sys
import types
from pathlib import Path

# 1. Definicja ścieżki (bezwzględna lokalizacja folderu zad71)
current_dir = Path(__file__).resolve().parent

# 2. MECHANIZM NAPRAWCZY: Tworzymy wirtualny moduł 'zad71' w pamięci
# Jeśli model szuka klas w 'zad71', przekierujemy go tutaj
if 'zad71' not in sys.modules:
    import __main__
    # Tworzymy alias: moduł 'zad71' staje się kopią głównego skryptu
    sys.modules['zad71'] = __main__
    
    # Dodajemy folder do ścieżek wyszukiwania, aby importy działały
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))

# 3. DOPIERO TERAZ DALSZE IMPORTY
import streamlit as st
import joblib

import json
import pandas as pd  # type: ignore
import plotly.express as px  # type: ignore
from dotenv import load_dotenv, set_key
from openai import OpenAI
from pycaret.clustering import load_model, predict_model
import os

# To wymusza, aby os był widoczny wszędzie tam, gdzie joblib go szuka
sys.modules['os'] = os 
  
#@st.cache_data
def handle_openai_key():
    env_path = Path(".env")
    
    # 1. Próbujemy załadować istniejący plik .env
    load_dotenv(env_path)
    api_key = os.getenv("OPENAI_API_KEY")

    # 2. Jeśli klucza nie ma w środowisku ani w pliku
    if not api_key:
        st.warning("Brak klucza OpenAI API!")
        
        # Formularz do podania klucza
        with st.form("api_key_form"):
            user_key = st.text_input("Wprowadź swój klucz OpenAI API:", type="password")
            submitted = st.form_submit_button("Zapisz klucz")
            
            if submitted and user_key:
                # Zapisujemy klucz do pliku .env, aby pamiętać go przy następnym uruchomieniu
                set_key(str(env_path), "OPENAI_API_KEY", user_key)
                st.success("Klucz został zapisany! Odśwież aplikację (R).")
                st.stop()  # Zatrzymujemy dalsze wykonywanie, dopóki klucz nie zostanie załadowany
            else:
                st.stop() # Blokujemy aplikację do czasu podania klucza
                
    return api_key

#@st.cache_data
def build_model(MODEL_NAME,DATA,num_clusters=8):
    #df.head()
    from pycaret.clustering import setup, create_model, save_model
    s = setup(DATA, session_id=123, html=False, verbose=False)
    #s.dataset.head()
    #s.dataset_transformed.head()
    
    kmeans = create_model('kmeans', num_clusters)
    df_with_clusters = assign_model(kmeans)
    #df_with_clusters
    #df_with_clusters["Cluster"].value_counts()
    plot_model(kmeans, plot='cluster', display_format='streamlit')
    save_model(kmeans, MODEL_NAME, verbose=False)

#@st.cache_data
def make_descriptions(_data_model,new_data,FILE_CLUSTER_NAMES_AND_DESCRIPTIONS,api_key):
    #api_key = handle_openai_key()
    openai_client = OpenAI(api_key=api_key)
    
    kmeans_pipeline = _data_model
    df_with_clusters = predict_model(model=kmeans_pipeline, data=new_data)
    df_with_clusters["Cluster"].value_counts()
    
    cluster_descriptions = {}
    for cluster_id in df_with_clusters['Cluster'].unique():
        cluster_df = df_with_clusters[df_with_clusters['Cluster'] == cluster_id]
        summary = ""
        for column in df_with_clusters:
            if column == 'Cluster':
                continue

            value_counts = cluster_df[column].value_counts()
            value_counts_str = ', '.join([f"{idx}: {cnt}" for idx, cnt in value_counts.items()])
            summary += f"{column} - {value_counts_str}\n"

        cluster_descriptions[cluster_id] = summary

    prompt = "Użyliśmy algorytmu klastrowania."
    for cluster_id, description in cluster_descriptions.items():
        prompt += f"\n\nKlaster {cluster_id}:\n{description}"

    prompt += """
    Wygeneruj najlepsze nazwy dla każdego z klasterów oraz ich opisy

    Użyj formatu JSON. Przykładowo:
    {
        "Cluster 0": {
            "name": "Klaster 0",
            "description": "W tym klastrze znajdują się osoby, które..."
        },
        "Cluster 1": {
            "name": "Klaster 1",
            "description": "W tym klastrze znajdują się osoby, które..."
        }
    }
    """
    #print(prompt)
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ],
    )   
    result = response.choices[0].message.content.replace("```json", "").replace("```", "").strip()
    cluster_names_and_descriptions = json.loads(result)
    with open(FILE_CLUSTER_NAMES_AND_DESCRIPTIONS, "w") as f:
        f.write(json.dumps(cluster_names_and_descriptions))
    

#@st.cache_data
def get_model(MODEL_PATH):
    # Wymuś pełną ścieżkę z rozszerzeniem .pkl
    full_path = str(Path(MODEL_PATH).with_suffix('.pkl'))
    return joblib.load(full_path)

@st.cache_data
def get_cluster_names_and_descriptions(CLUSTER_NAMES_AND_DESCRIPTIONS):
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r", encoding='utf-8') as f:
        return json.loads(f.read())

@st.cache_data
def get_all_participants(_data_model,new_data):
    df_with_clusters = predict_model(_data_model, data = new_data)
    return df_with_clusters

if 'data_df' not in st.session_state:
    st.session_state.data_df = None
if st.session_state.data_df is None:
    lista_csv = [f.name for f in current_dir.glob("*.csv")]
    if lista_csv:
        # 2. Wyświetlamy rozwijaną listę (selectbox)
        wybrany_plik = st.selectbox("Wybierz plik danych do analizy:", lista_csv)
    
        # 3. Akcja po wyborze (np. wczytanie ramki danych)
        if st.button("Wczytaj dane"):
            st.session_state.data_df = pd.read_csv(current_dir / wybrany_plik, sep=';')
            st.success(f"Pomyślnie wczytano: {wybrany_plik}")
            st.dataframe(st.session_state.data_df.head())
            if st.button("OK"):
                st.rerun()
    else:
        st.warning("⚠️ W folderze aplikacji nie znaleziono żadnych plików CSV.") 
        if st.button("OK"):
            st.rerun()
   
else:
    if 'd_model' not in st.session_state:
        st.session_state.d_model = None
    if st.session_state.d_model is None:
        lista_pkl = [f.name for f in current_dir.glob("*.pkl")]
        #if lista_pkl:
        # 2. Wyświetlamy rozwijaną listę (selectbox)
        wybrany_plik = st.selectbox("Wybierz plik modelu treningowego:", lista_pkl)
            
        if st.button("Wczytaj dane"):
            wybrany_plik = wybrany_plik.replace('.pkl', '')
            model_name = Path(wybrany_plik).stem
            target_path = str(current_dir / model_name)
            st.session_state.d_model = get_model(target_path)
            st.success(f"Pomyślnie wczytano: {wybrany_plik}")
            if st.button("OK"):
                st.rerun()
                
        #else:
        with st.form("Zbuduj nowy model treningowy"):
            MODEL_NAME = st.text_input("Nazwa modelu:", value="welcome_survey_clustering_pipeline_v1")
            num_clusters_input = st.number_input(
            "Wybierz liczbę klastrów:", 
            min_value=2, 
            max_value=20, 
            value=8, 
            step=1
            )
            submit_button = st.form_submit_button("Zatwierdź model")
            if not submit_button:
                st.stop()
            build_model(MODEL_NAME,st.session_state.data_df,num_clusters_input)
            st.session_state.d_model = get_model(MODEL_NAME)
        if st.button("OK"):
            st.rerun()
    else:   
        if 'json_cluster_names_and_descriptions' not in st.session_state:
            st.session_state.json_cluster_names_and_descriptions = None
        if st.session_state.json_cluster_names_and_descriptions is None:
            api_key = handle_openai_key()
            lista_json = [f.name for f in current_dir.glob("*.json")]
            #lista_pkl = [f.name for f in current_dir.iterdir() if f.is_file() and f.suffix == ".json"]
            #if lista_pkl:
            # 2. Wyświetlamy rozwijaną listę (selectbox)
            wybrany_plik = st.selectbox("Wybierz plik opisu grup modelu treningowego:", lista_json)
            
            if st.button("Wczytaj dane"):
                st.session_state.json_cluster_names_and_descriptions = get_cluster_names_and_descriptions(current_dir / wybrany_plik)
                st.success(f"Pomyślnie wczytano: {wybrany_plik}")
                if st.button("OK"):
                    st.rerun()
            #else:
            
            with st.form("Wygeneruj opisy dla modelu treningowego"):
                CLUSTER_NAMES_AND_DESCRIPTIONS = st.text_input("Nazwa modelu:", value='welcome_survey_cluster_names_and_descriptions_v1.json')
                submit_button = st.form_submit_button("Zatwierdź nazwę opisu modelu")
                if not submit_button:
                    st.stop()
                make_descriptions(st.session_state.d_model,st.session_state.data_df,CLUSTER_NAMES_AND_DESCRIPTIONS,api_key)
                st.write('Skończyłem generowanie opisów do modelu treningowego')
                st.session_state.json_cluster_names_and_descriptions = get_cluster_names_and_descriptions(CLUSTER_NAMES_AND_DESCRIPTIONS)
            if st.button("OK"):
                st.rerun()
        else:
            with st.sidebar:
                st.header("Powiedz nam coś o sobie")
                st.markdown("Pomożemy Ci znaleźć osoby, które mają podobne zainteresowania")
                age = st.selectbox("Wiek", ['<18', '25-34', '45-54', '35-44', '18-24', '>=65', '55-64', 'unknown'])
                edu_level = st.selectbox("Wykształcenie", ['Podstawowe', 'Średnie', 'Wyższe'])
                fav_animals = st.selectbox("Ulubione zwierzęta", ['Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy'])
                fav_place = st.selectbox("Ulubione miejsce", ['Nad wodą', 'W lesie', 'W górach', 'Inne'])
                gender = st.radio("Płeć", ['Mężczyzna', 'Kobieta'])

                person_df = pd.DataFrame([
                {
                    'age': age,
                    'edu_level': edu_level,
                    'fav_animals': fav_animals,
                    'fav_place': fav_place,
                    'gender': gender,
                }
            ])

            all_df = get_all_participants(st.session_state.d_model,st.session_state.data_df)
            predicted_cluster_id = predict_model(st.session_state.d_model, data=person_df)["Cluster"].values[0]
            predicted_cluster_data = st.session_state.json_cluster_names_and_descriptions[predicted_cluster_id]

            st.header(f"Najbliżej Ci do grupy {predicted_cluster_data['name']}")
            st.markdown(predicted_cluster_data['description'])
            same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id]
            st.metric("Liczba twoich znajomych", len(same_cluster_df))

            st.header("Osoby z grupy")
            fig = px.histogram(same_cluster_df.sort_values("age"), x="age")
            fig.update_layout(
                title="Rozkład wieku w grupie",
                xaxis_title="Wiek",
                yaxis_title="Liczba osób",
            )
            st.plotly_chart(fig)

            fig = px.histogram(same_cluster_df, x="edu_level")
            fig.update_layout(
                title="Rozkład wykształcenia w grupie",
                xaxis_title="Wykształcenie",
                yaxis_title="Liczba osób",
            )
            st.plotly_chart(fig)

            fig = px.histogram(same_cluster_df, x="fav_animals")
            fig.update_layout(
                title="Rozkład ulubionych zwierząt w grupie",
                xaxis_title="Ulubione zwierzęta",
                yaxis_title="Liczba osób",
            )
            st.plotly_chart(fig)

            fig = px.histogram(same_cluster_df, x="fav_place")
            fig.update_layout(
                title="Rozkład ulubionych miejsc w grupie",
                xaxis_title="Ulubione miejsce",
                yaxis_title="Liczba osób",
            )
            st.plotly_chart(fig)

            fig = px.histogram(same_cluster_df, x="gender")
            fig.update_layout(
                title="Rozkład płci w grupie",
                xaxis_title="Płeć",
                yaxis_title="Liczba osób",
            )
            st.plotly_chart(fig)
