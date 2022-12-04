<em>**Universidad Tecnológica de México Procesamiento Digital de imágenes**</em><br>

Nombre de los Alumnos:<br>
Petriciolet Cortes Josue Michel. Matricula: 20922674<br>
Santa Cruz Hernández Logan Diego. Matricula: 20580787<br>
Tovar Barrera Israel. Matricula: 21128157<br>
Vargas Anaya Luisa Fernanda. Matricula: 14899294<br>
Ingeniería en sistemas computacionales<br>
Grupo: EC08S<br>
Profesor: M. En C. Raymundo Soto Soto.<br>
Avance del proyecto: #3 Emotion (Playlist).<br>
Fecha de entrega:<br>
27/ Noviembre / 2022<br>

<h2>Objetivos</h2><br>
Los objetivos en esta parte del proyecto es mejorar las métricas de los algoritmos para que las emociones se<br>
puedan identificar de una manera más sencilla, una vez mejorado e implementado esa parte queremos usar la<br>
biblioteca Pyttsx3 de Python para poder convertir texto en voz, cada que la aplicación sea capaz de detectar una<br>
emoción se escuche o nos diga la emoción detectada y que se acaba de recomendar una serie de canciones para<br>
el estado de animo que tiene el usuario.<br>

<h2>Reconocimiento facial de IA</h2><br>
El reconocimiento fácil es distinto a la detección de rostros dentro de una imagen ya que el reconocimiento facial<br>
buscara identificar a la persona a la que le pertenece aquel rostro detectado y la detección de rostros busca caras<br>
dentro de una imagen o dentro de un video.<br>
El reconocimiento fácil y métrica de los rostros constituyen una parte fundamental de diversas aplicaciones o<br>
“experiencias” que toman como referencia el rostro de los humanos.<br>
La librería OpenCV está dirigida fundamentalmente a la visión por computador en tiempo real. Entre sus muchas<br>
áreas de aplicación destacarían: interacción hombre-máquina (HCI4); segmentación y reconocimiento de objetos;<br>
reconocimiento de gestos; seguimiento del movimiento; estructura del movimiento (SFM5); y robots móviles.<br>
Nos ayuda a poder detectar objetos y rostros, es aspectos como lo son la seguridad, marketing o la fotografía<br>
incluso. Nos provee infraestructura para aplicaciones relacionadas con la visión artificial, al ser multiplataforma<br>
lo podemos ejecutar con los principales sistemas operativos, lo podemos utilizar en lenguaje c, c++ o Python.<br>
En nuestro proyecto utilizamos OpenCV, es una biblioteca de código abierto de visión artificial y machine learning,<br>
La librería tiene más de 2500 algoritmos, que incluye algoritmos de machine learning y de visión artificial estos<br>
algoritmos permiten identificar objetos, caras, clasificar acciones humanas en vídeo, hacer tracking de<br>
movimientos de objetos, extraer modelos 3D y encontrar imágenes similares.<br>
OpenCV utiliza la visión artificial que es la visión artificial es un campo de la IA que permite que las computadoras<br>
y los sistemas obtengan información significativa de imágenes digitales, videos y otras entradas visuales, y tomen<br>
acciones o hagan recomendaciones basadas en esa información.<br>
La librería OpenCV está dirigida fundamentalmente a la visión por computador en tiempo real. Entre sus muchas<br>
áreas de aplicación destacarían: interacción hombre-máquina (HCI4); segmentación y reconocimiento de objetos;<br>
reconocimiento de gestos; seguimiento del movimiento; estructura del movimiento (SFM5); y robots móviles.</p>

<h2>Codigo fuente</h2>

```
import cv2
import mediapipe as mp
import math
import pyttsx3
import time
engine = pyttsx3.init()
voice_id = 'spanish-latin-am'
engine.setProperty('voice', voice_id)
rate = engine.getProperty('rate')
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
 
 engine.setProperty('rate', rate-0)
 string='Veo que estas muy enojado'
 engine.say('Veo que estas muy enojado')
 engine.runAndWait()
 time.sleep(1)
 
 elif logitud1 > 15 and logitud1 < 30 and logitud2 > 15 and logitud2 < 30 and logitud3 >110 and 
logitud4 > 10 and logitud4 <20:
 cv2.putText(frame, 'PERSONA FELIZ', (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
 (0, 255, 0), 3)
 
 engine.setProperty('rate', rate-0)
 string='Vaya que contento estas'
 engine.say('Vaya que feliz estas')
 engine.runAndWait()
 time.sleep(1)
 
 elif logitud1 > 35 and logitud2 > 35 and logitud3 > 80 and logitud3 <90 and logitud4 >20:
 cv2.putText(frame, 'PERSONA ASOMBRADA', (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
 (255, 255, 0), 3)
 
 engine.setProperty('rate', rate-0)
 string='Que asombrado estas'
 engine.say('Que sorprendido estas')
 engine.runAndWait()
 time.sleep(1)
 
 elif logitud1 >30 and logitud1 >25 and logitud2 >30 and logitud2 >25 and logitud3 > 80 and logitud3 < 
95 and logitud4 <2:
 cv2.putText(frame, 'PERSONA TRISTE', (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
 (255, 0, 255), 3)
 engine.setProperty('rate', rate-0)
 string='Porque tan achicopalado animo'
 engine.say('Porque tan triste mucho animo')
 engine.runAndWait()
 time.sleep(1)
 cv2.imshow("RECONOCIMIENTO DE EMOCIONES", frame)
 t= cv2.waitKey(1)
 if t == 27:
 break
cap.release()
cv2.destroyAllWindows()
 
```

![image](https://user-images.githubusercontent.com/114814184/205480771-e63a9c8a-98b2-42da-b6de-9410ee53827b.png)

![image](https://user-images.githubusercontent.com/114814184/205480777-a24fca6d-fc91-4268-ba0a-da27e89e2e04.png)

![image](https://user-images.githubusercontent.com/114814184/205480783-cb3c8126-96d6-4e6e-b498-6431d387dcda.png)

![image](https://user-images.githubusercontent.com/114814184/205480790-c742ecc8-aa48-4de9-96bf-68474fb7a88c.png)


<h2>Conclusiones</h2><br>
Israel: En este proyecto vimos las aplicaciones de la programación de imágenes digitales y sus diferentes<br>
aplicaciones, en este caso nuestro proyecto viene enfocado hacia identificar las emociones de las personas y en<br>
base a eso identificar su estado de humor de una persona fue muy interesante ver el proceso de este proyecto<br>
porque nuestra aplicación utiliza inteligencia artificial y dependiendo los umbrales con los cuales lo programamos<br>
identifica una emoción de una persona con este tipo de aplicaciones podemos implementarlas en cuestiones de<br>
sugerencias de canciones dependiendo nuestro estado de humor o también recomendarnos películas u otros<br>
cosas donde queramos contenido en base a nuestras emociones, nuestro proyecto lo enfocamos más hacia la<br>
sugerencia de aplicaciones con las cuales dependiendo nuestro estado de humor nos recomiende una canción y<br>
de esa forma escuchar música dependiendo de cómo nos sentimos en ese momento pero si tuviéramos más<br>
recursos podríamos escalarlo a más que canciones.<br>

Logan: En esta fase del proyecto se repararon las fallas y errores que se tuvieron previamente atrás, también se<br>
implementó e investigo más aplicaciones para el proyecto y mejorarlo cómo inteligencia artificial y Pyttsx3.<br>

Michel: Este proyecto se conoció el funcionamiento de los sistemas inteligentes que se puede implementar en<br>
diferentes proyectos como también se puso en práctica el software anaconda donde nos deja crear un ambiente<br>
de trabajo y en este caso utilizamos spyder para que se pueda facilitar el uso de las librerías y con la facilidad de<br>
programar. Este proyecto lo podemos implementar con el fin de entrenamiento a las personas para que el sistema<br>
pueda detectar sus emociones fáciles a las personas y por medio de una librería implementar código que<br>
reproduzca un mensaje que diga la emoción que el sistema identifico , este proyecto a futuro puede tener muy<br>
buenas versiones , este proyecto me ayudó cómo funciona la visión artificial ya que me da muchas ideas de cómo<br>
hacer diferentes programas, pero ahora hacer programas que estén relacionados con redes neuronales para así<br>
poder entrenar el sistema para que pueda tomar decisiones con más exactitud.<br>

Fernanda: En conclusión, en esta fase del proyecto implementamos la librería PYttsx3 que nos va a ayudar a que<br>
el usuario tengo una experiencia más agradable utilizando la aplicación ya que a la hora de detectar la emoción la<br>
aplicación será capaz de decirla al usuario si se encuentra triste, feliz enojado y/o asombrado, lo cual desde mi<br>
perspectiva hace que la aplicación incluya a personas con distintas discapacida</p>
