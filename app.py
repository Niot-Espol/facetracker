import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2

 
# USARLO PARA CORRER Y ABIR EL LOCALHOST :b - python -m streamlit run app.py
#Usar ctrl + c papra apagar el servidor local


#Conf de la página
st.set_page_config(
    page_title= "Reconocimiento Facial",
    page_icon= "🎭",
    layout= "wide",
    initial_sidebar_state= "expanded"
)

#css: estética de la página - uso de paleta de colores
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;600&family=Barlow+Condensed:wght@800&display=swap');
 
    html, body, .stApp { background-color: #1a2035; font-family: 'Barlow', sans-serif; }
 
    [data-testid="stSidebar"] { background: #25344F !important; border-right: 2px solid #6F4D38; }
    [data-testid="stSidebar"] * { color: #D5B893 !important; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        font-family: 'Barlow Condensed', sans-serif !important;
        text-transform: uppercase; letter-spacing: 1px;
    }
 
    h1 {
        font-family: 'Barlow Condensed', sans-serif !important;
        font-weight: 800 !important; font-size: 2.6rem !important;
        color: #D5B893 !important; text-transform: uppercase;
        border-bottom: 3px solid #6F4D38; padding-bottom: 10px;
    }
 
    .video-label {
        color: #617891; font-size: 0.72rem;
        text-transform: uppercase; letter-spacing: 2px; margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

#Sidebar para módulo estadístico
st.sidebar.header("Métricas de la Sesión")
st.sidebar.write("----Grafico de pastel (luegooooo)----")

#Procesa el video cuadro por cuadro
def procesar_video(frame):
    img = frame.to_ndarray(format = "bgr24")

    img = cv2.flip(img, 1)

    #APARTADO PARA EL MODELO ENTRENADO

    #Simulador visual de detección de rostro
    cv2.rectangle(img, (150, 100), (450, 400), (0, 255, 0), 3)

    #Simulador de emoción detectada
    cv2.putText(img, "Emocion: Feliz 74%", (150, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return av.VideoFrame.from_ndarray(img, format = "bgr24")


st.title("Sistema de Reconocimiento Facial 🎭")
st.markdown('<div class = "video-label"> 📸 &nbsp; Cámara web en vivo</div>', unsafe_allow_html= True)

#Componente WebRTC para encendido de cámara
webrtc_streamer(
    key= "reconocimiento_facial",
    video_frame_callback= procesar_video,
    media_stream_constraints= {
        "video": {
            "width": {"ideal": 1280},
            "height": {"ideal": 720},
        },
        "audio": False
    }
)
