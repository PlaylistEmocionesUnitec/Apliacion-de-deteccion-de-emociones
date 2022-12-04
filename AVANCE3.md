<em>**Universidad Tecnológica de México Procesamiento Digital de imágenes**</em>

Nombre de los Alumnos:<br>
Petriciolet Cortes Josué Michel. Matricula: 20922674<br>
Santa Cruz Hernández Logan Diego. Matricula: 20580787<br>
Tovar Barrera Israel. Matricula: 21128157<br>
Vargas Anaya Luisa Fernanda. Matricula: 14899294<br>
Ingeniería en sistemas computacionales<br>
Grupo: EC08S<br>
Profesor: M. En C. Raymundo Soto Soto.<br>
Avance del proyecto: #4 Emotion (Playlist).<br>
Fecha de entrega: 20/ noviembre / 2022</p>

<h2>Objetivos</h2><br>
En esta parte del proyecto, en la que ya está definida la problemática y el por qué, de nuestro proyecto, nos<br>
enfocamos en mejorar el código, buscamos que la aplicación sea agradable para todas las personas, buscamos<br>
implementar voz cuando la aplicación detecte las emociones para eso utilizaremos la librería de Python PyttsX3.<br>
Nos concentramos en que los puntos que nos ayudan a detectar las emociones tengan los parámetros adecuados<br>
para poder pasar al siguiente paso, como lo es implementar el uso de la librería Pyttsx3.<br>

<h2>Reconocimiento facial de IA</h2><br>
El reconocimiento facial es lo que nosotros conocemos como un conjunto de algoritmos que trabajan en equipo<br>
para que se pueda identificar o detectar a las personas en tiempo real o una imagen, en los últimos años se ha ido<br>
innovando esta idea.<br>
Nos permite la identificación en tiempo real de una persona, nos brinda medidas contra la suplantación de<br>
identidad y se puede llegar a utilizar en varios dispositivos o cámaras.<br>
¿Cómo funciona el reconocimiento facial? La cara al dividirse en numerosos puntos de daros, como los es la<br>
distancia entre los ojos, la altura de los pómulos, la distancia entre la boca y la nariz, etc. De esta manera es posible<br>
que se detecten las diferentes emociones, ya que cada vez que presentamos una emoción diferente, los<br>
parámetros de nuestro rostro conforme a la emoción cambian lo cual permite que se detecte de manera fácil.<br>

<h2>Pyttsx3 Python</h2><br>
Es una biblioteca de conversión de texto a voz multiplataforma en Python. Funciona sin conexión y es compatible<br>
con Python 2 y 3. Es una herramienta fácil de usar que convierte el texto convertido en voz. Puede establecer<br>
metadatos de voz como edad, sexo, idioma, etc.<br>
La aplicación admite dos voces, femenina y masculina, proporcionados por “Sapi5”, una interfaz de<br>
reconocimiento del habla y de síntesis de voz para la programación de aplicaciones basadas en win 32.<br>
Es compatible con los siguientes motores:<br>
• SAPI5 para Windows<br>
• NSSpeechSynthesixer para Mac OS X<br>
• eSpeak para otras plataformas<br>
Previo al uso de Pyttsx3 se requiere instalar “PyWin32”, el cual es un paquete de extensiones para usar algunas<br>
características del sistema desde Python.</p>
<p class="has-line-data" data-line-start="42" data-line-end="134">Instalación<br>
• Instalar pywin32-extensions<br>
• Una vez descargado lo ejecutamos<br>
• Realizamos la instalación del módulo pyttsx a través del gestor de paquetes pip<br>
• Listo<br>
La ventaja de Pyttsx3 ante otras aplicaciones es que es totalmente offline, así que solo nos preocuparemos de<br>
instalar y después se usara sin problemas.<br>
Open cv<br>
En nuestro proyecto utilizamos OpenCV, es una biblioteca de código abierto de visión artificial y machine learning,<br>
La librería tiene más de 2500 algoritmos, que incluye algoritmos de machine learning y de visión artificial estos<br>
algoritmos permiten identificar objetos, caras, clasificar acciones humanas en vídeo, hacer tracking de<br>
movimientos de objetos, extraer modelos 3D y encontrar imágenes similares.<br>
OpenCV utiliza la visión artificial que es la visión artificial es un campo de la IA que permite que las computadoras<br>
y los sistemas obtengan información significativa de imágenes digitales, videos y otras entradas visuales, y tomen<br>
acciones o hagan recomendaciones basadas en esa información.<br>
  
<h2>Codigo fuente</h2>

```
import cv2
import mediapipe as mp
import math

#-----------TOMAMOS LA VIDEOCAPTURA----------
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap.set(3,1280) #ancho de ventana de la camara
cap.set(4,720) #alto de ventana de la camara
#--------------- FUNCION DE DIBUJO------------
mpDibujo = mp.solutions.drawing_utils
confdibu = mpDibujo.DrawingSpec(thickness=1, circle_radius=1)#--AJUSTE LA CONFIGURACION DEL DIBUJO
#------------CREACION DE OBJETO DONDE ALMACENE LA MALLA FACIAL------------
mpMallaFacial = mp.solutions.face_mesh #PRIMERO LLAMAMOS LA FUNCION
MallaFacial = mpMallaFacial.FaceMesh(max_num_faces=1)#---SE CREO EL OBJETO
#----------CREAMOS EL WHILE PRINCIPAL------
while True:
 ret,frame = cap.read()
 #-----CORRECCION DE COLOR--------
 frameRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
 #-----OBSERVAR LOS RESULTADOS-----
 resultados = MallaFacial.process(frameRGB)
 #SE CRERA UNA LISTA DONDE SE ALAMACENARAN LOS RESULTADOS
 px = []
 py = []
 lista = []
 r = 5
 t = 3
 if resultados.multi_face_landmarks: #SI DETECTA ALGUN ROSTRO
 for rostros in resultados.multi_face_landmarks: #MUESTRA EL ROSTRO DETECTADXO
 mpDibujo.draw_landmarks(frame, rostros,mpMallaFacial.FACEMESH_CONTOURS,confdibu, confdibu)
 #EXTRAER LOS PUNTOS DEL ROSTRO DETECTADO
 for id, puntos in enumerate(rostros.landmark):
 al, an,c = frame.shape
 x , y = int(puntos.x*an), int(puntos.y*al)
 px.append(x)
 py.append(y)
 lista.append([id,x,y])
 if len(lista) ==468:
 #CEJA DERECHA
 x1, y1 = lista[65][1:]
 x2, y2 = lista[158][1:]
 cx, cy = (x1 + x2 ) // 2, (y1 + y2) //2
 cv2.line(frame,(x1 , y1), (x2 , y2 ), (0,0,0),t)
 cv2.circle(frame,(x1 , y1), r, (0,0,0), cv2.FILLED)
 cv2.circle(frame,(x2 , y2), r, (0,0,0), cv2.FILLED)
 cv2.circle(frame, (cx, cy), r, (0, 0, 0), cv2.FILLED)
 logitud1 = math.hypot(x2 - x1, y2 - y1)
 #print(logitud1)
 #CEJA IZQUIERDA
 x3, y3 = lista[295][1:]
 x4, y4 = lista[385][1:]
 cx2, cy2 = (x3 + x4) // 2, (y3 + y4) // 2
 logitud2 = math.hypot(x4 - x3, y4 - y3)
 #print(logitud2)
 #BOCA EXTREMOS
 x5, y5 = lista[78][1:]
 x6, y6 = lista[308][1:]
 cx3, cy3 = (x5 + x6) // 2, (y5 + y6) // 2
 logitud3 = math.hypot(x6 - x5, y6 - y5)
 #print(logitud3)
 #BOCA APERTURA
 x7, y7 = lista[13][1:]
 x8, y8 = lista[14][1:]
 cx4, cy4 = (x7 + x8) // 2, (y7 + y8) // 2
 logitud4 = math.hypot(x8 - x7, y8 - y7)
 #print(logitud4)
 #CLASIFICACION
 if logitud1 < 19 and logitud2 < 19 and logitud3 > 80 and logitud3 < 95 and logitud4 <15:
 cv2.putText(frame,'PERSONA ENOJADA',(480,80),cv2.FONT_HERSHEY_SIMPLEX,1,
 (0,0,255),3)
 elif logitud1 > 20 and logitud1 < 30 and logitud2 > 20 and logitud2 < 30 and logitud3 > 109 and 
logitud4 > 10 and logitud4 <20:
 cv2.putText(frame, 'PERSONA FELIZ', (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
 (0, 255, 0), 3)
 elif logitud1 > 35 and logitud2 > 35 and logitud3 > 80 and logitud3 <90 and logitud4 >20:
 cv2.putText(frame, 'PERSONA ASOMBRADA', (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
 (255, 255, 0), 3)
 elif logitud1 >10 and logitud1 >35 and logitud2 >10 and logitud2 >35 and logitud3 > 90 and logitud3 < 
95 and logitud4 <5:
 cv2.putText(frame, 'PERSONA TRISTE', (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
 (255, 0, 255), 3)
 cv2.imshow("RECONOCIMIENTO DE EMOCIONES", frame)
 t= cv2.waitKey(1)
 if t == 27:
 break
cap.release()
cv2.destroyAllWindows()
```

<h2>Bibliografía</h2><br>
Sangpal, R., Gawand, T., Vaykar, S. y Madhavi, N. (julio de 2019). JARVIS: una interpretación de AIML con<br>
integración de gTTS y Python. En 2019 2ª Conferencia Internacional sobre Tecnologías Inteligentes de<br>
Computación, Instrumentación y Control (ICICICT) (Vol. 1, pp. 486-489). IEEE.<br>
Nazira, Farzana Arefin, et al. “Face Recognition Based Driver Detection System.” 2021 International Conference<br>
on Data Analytics for Business and Industry (ICDABI). IEEE, 2021.<br>
Gupta, S., Pandey, A., Naruka, S. y Gupta, K. (2022). Un novedoso sistema de reconocimiento de voz con<br>
inteligencia artificial. En Ingeniería en Microelectrónica y Telecomunicación (págs. 573-579). Springer, Singapur.<br>
¿Qué es la Visión Artificial? (s/f). <a href="http://Ibm.com">Ibm.com</a>. Recuperado el 19 de octubre de 2022, de <a href="https://www.ibm.com/mx">https://www.ibm.com/mx</a>es/topics/computer-vision<br>
Modulo Pyttsx: Reproduciendo texto desde Python. (2018, mayo 14). <a href="http://Wordpress.com">Wordpress.com</a>.<br>
<a href="https://carlosjuliopardoblog.wordpress.com/2018/05/14/motores-de-voz/">https://carlosjuliopardoblog.wordpress.com/2018/05/14/motores-de-voz/</a><br>
Mordvintsev, Alexander, and K. Abid. “Opencv-python tutorials documentation.” Obtenido de <a href="https://media">https://media</a>.<br>
readthedocs. org/pdf/opencv-python-tutroals/latest/opencv-python-tutroals. pdf (2014)</p>
