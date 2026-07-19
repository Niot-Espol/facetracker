import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import time
import threading
import plotly.graph_objects as go
from collections import Counter

#USARLO PARA CORRER Y ABRIR EL LOCALHOST :b - python -m streamlit run app.py
#Usar ctrl + c para apagar el servidor local


#CONFIGURACIÓN DE LA PAGINA
st.set_page_config(
    page_title="Reconocimiento Facial",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

#ESTILOS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;600&family=Barlow+Condensed:wght@800&display=swap');

    html, body, .stApp { background-color: #1e1728; font-family: 'Barlow', sans-serif; }

    [data-testid="stSidebar"] { background: #312A44 !important; border-right: 2px solid #6a5278 !important; }
    [data-testid="stSidebar"] * { color: #D7C5D6 !important; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        font-family: 'Barlow Condensed', sans-serif !important;
        text-transform: uppercase; letter-spacing: 1px;
        color: #DAD4DF !important;
    }

    h1 {
        font-family: 'Barlow Condensed', sans-serif !important;
        font-weight: 800 !important; font-size: 2.6rem !important;
        color: #D7C5D6 !important; text-transform: uppercase;
        border-bottom: 3px solid #6a5278;
        padding-bottom: 10px;
    }

    .video-label {
        color: #BAB0C8; font-size: 0.72rem;
        text-transform: uppercase; letter-spacing: 2px; margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)


#MEMORIA DE LA SESIÓN
#Aqui se va guardando cada emocion detectada mientras la camara esta prendida, para luego armar el grafico de pastel del sidebar.
if "historial_emociones" not in st.session_state:
    st.session_state.historial_emociones = []

#El modelo de IA no se ejecuta en cada frame para no saturar el CPU. Con este número se decide cada cuantos frames se le pide una prediccion nueva.
#Ejemplo: con 5, el modelo corre 1 de cada 5 frames que llegan de la camara.
FRAMES_ANTES_DE_PREDECIR = 5


#Esta clase recibe el video de la cámara, frame por frame, y decide que dibujar encima (el recuadro, la emocion y el contador de FPS).

class DetectorDeEmociones(VideoProcessorBase):
    def __init__(self):
        #Candado para evitar que dos partes del programa escriban esta información exactamente al mismo tiempo y se pisen entre si.
        self.candado = threading.Lock()

        self.contador_de_frames = 0

        #Guarda el ultimo recuadro + emocion detectados, para poder seguir
        #dibujandolos aunque en este frame no se haya vuelto a calcular.
        self.ultima_deteccion = None  #(x_inicio, y_inicio, x_fin, y_fin, emocion, confianza)

        #Emociones detectadas que todavia no se han pasado al historial general.
        self.emociones_sin_guardar = []

        #Datos para calcular los FPS (cuadros por segundo) en tiempo real.
        self.momento_ultimo_frame = time.time()
        self.fps_actual = 0.0

    def recv(self, frame):
        imagen = frame.to_ndarray(format="bgr24")
        imagen = cv2.flip(imagen, 1)  #efecto espejo, como una webcam normal
        alto, ancho = imagen.shape[:2]

        #Calculo de FPS
        #Se mide cuanto tiempo paso desde el frame anterior y se calcula cuantos frames por segundo representa eso.
        momento_actual = time.time()
        tiempo_transcurrido = momento_actual - self.momento_ultimo_frame
        self.momento_ultimo_frame = momento_actual
        if tiempo_transcurrido > 0:
            fps_de_este_frame = 1.0 / tiempo_transcurrido
            #Se suaviza el numero para que no este saltando feo en pantalla
            #(90% del valor viejo + 10% del valor nuevo).
            self.fps_actual = self.fps_actual * 0.9 + fps_de_este_frame * 0.1

        #Deteccion de emocion (cada N frames, no en todos)
        with self.candado:
            self.contador_de_frames += 1
            le_toca_predecir = (self.contador_de_frames % FRAMES_ANTES_DE_PREDECIR == 0)

            if le_toca_predecir:
        
                # AQUÍ VA EL MODELO ENTRENADO 
                # Debe devolver 6 valores:
                #x_inicio, y_inicio, x_fin, y_fin - esquinas del recuadro
                #emocion: texto, ej. "Feliz", "Triste", "Enojado"...
                #confianza: número del 0 al 100
                #Por ahora se simula un cuadrado fijo en el centro:

                lado_del_cuadro = int(min(ancho, alto) * 0.35)
                centro_x, centro_y = ancho // 2, alto // 2
                x_inicio = centro_x - lado_del_cuadro // 2
                y_inicio = centro_y - lado_del_cuadro // 2
                x_fin = centro_x + lado_del_cuadro // 2
                y_fin = centro_y + lado_del_cuadro // 2
                emocion, confianza = "Feliz", 74

                self.ultima_deteccion = (x_inicio, y_inicio, x_fin, y_fin, emocion, confianza)
                self.emociones_sin_guardar.append(emocion)

            deteccion_a_dibujar = self.ultima_deteccion

        #Dibujar el recuadro y la emocion sobre el video Se dibuja SIEMPRE la ultima deteccion conocida, aunque este frame
        #en particular no haya calculado una nueva. Asi el recuadro no parpadea ni el video se ve congelado.
        if deteccion_a_dibujar:
            x_inicio, y_inicio, x_fin, y_fin, emocion, confianza = deteccion_a_dibujar
            cv2.rectangle(imagen, (x_inicio, y_inicio), (x_fin, y_fin), (0, 255, 0), 3)
            cv2.putText(
                imagen, f"{emocion} {confianza}%", (x_inicio, y_inicio - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

        #Dibujar el contador de FPS arriba a la derecha
        texto_fps = f"{self.fps_actual:.1f} FPS"
        (ancho_texto, alto_texto), _ = cv2.getTextSize(texto_fps, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(
            imagen, texto_fps, (ancho - ancho_texto - 15, alto_texto + 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

        return av.VideoFrame.from_ndarray(imagen, format="bgr24")


#SIDEBAR: metricas de la sesion
st.sidebar.header("Métricas de la Sesión")

if st.session_state.historial_emociones:

    #Cuenta cuantas veces se repitio cada emocion durante la sesion
    conteo_por_emocion = Counter(st.session_state.historial_emociones)

    grafico_pastel = go.Figure(go.Pie(
        labels=list(conteo_por_emocion.keys()),
        values=list(conteo_por_emocion.values()),
        hole=0.4,
        marker=dict(
            colors=["#BAB0C8", "#887003", "#D7C5D6", "#312A44", "#6a5278"],
            line=dict(color="#1e1728", width=2)
        ),
        textinfo="label+percent",
        textfont=dict(color="#DAD4DF", size=12),
    ))
    grafico_pastel.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(t=4, b=4, l=0, r=0),
        height=220,
    )
    st.sidebar.plotly_chart(grafico_pastel, use_container_width=True)

else:
    st.sidebar.write("El gráfico aparece al iniciar la cámara...")

#Boton para borrar todo el historial y empezar de cero
if st.sidebar.button("Limpiar Sesión"):
    st.session_state.historial_emociones = []
    st.rerun()


#PANTALLA PRINCIPAL
st.title("Sistema de Reconocimiento Facial 🎭")

#Controla que tan ancho se ve el video en pantalla.
#Sube este numero para agrandarlo, bajalo para achicarlo (rango sugerido 1-6).
ANCHO_DEL_VIDEO = 5
margen_lateral = (8 - ANCHO_DEL_VIDEO) / 2
columna_izquierda, columna_video, columna_derecha = st.columns(
    [margen_lateral, ANCHO_DEL_VIDEO, margen_lateral]
)

with columna_video:
    st.markdown('<div class="video-label"> 📸 &nbsp; Cámara web en vivo</div>', unsafe_allow_html=True)

    #Enciende la camara del usuario y manda cada frame a DetectorDeEmociones
    camara = webrtc_streamer(
        key="reconocimiento_facial",
        video_processor_factory=DetectorDeEmociones,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
            },
            "audio": False
        }
    )

#Traer las emociones nuevas hacia el historial de la sesion
if camara.video_processor:
    with camara.video_processor.candado:
        emociones_nuevas = camara.video_processor.emociones_sin_guardar.copy()
        camara.video_processor.emociones_sin_guardar.clear()
    if emociones_nuevas:
        st.session_state.historial_emociones.extend(emociones_nuevas)
        st.rerun()