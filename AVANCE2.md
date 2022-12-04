Universidad Tecnológica de México Procesamiento Digital de imágenes<br>

Nombre de los Alumnos:<br>
Petriciolet Cortes Josue Michel. Matricula: 20922674<br>
Santa Cruz Hernández Logan Diego. Matricula: 20580787<br>
Tovar Barrera Israel. Matricula: 21128157<br>
Vargas Anaya Luisa Fernanda. Matricula: 14899294<br>
Ingeniería en sistemas computacionales<br>
Grupo: EC08S<br>
Profesor: M. En C. Raymundo Soto Soto.<br>

<h1>Avance del proyecto: #3 Emotion (Playlist)</h1><br>
Fecha de entrega: 30 / octubre / 2022</p>
<h2>Objetivos</h2><br>

En esta parte del proyecto lo que buscamos es avanzar en el código, encontrando que al realizarlo utilizamos otra<br>
herramienta como lo fue Pycharm, opencv y deepface.<br>

<h2>Marco teórico</h2><br>
En la actualidad Python es uno de los lenguajes de programación más usados en la industria de desarrollo de<br>
software, esto se debe principalmente a sus potentes características:<br>

• Orientado a objetos y de alto nivel<br>
• Desarrollo rápido de aplicaciones<br>
• Mecanografiado dinámico<br>
• Reutilización y modularidad del código<br>
• Ciclos rápidos en la edición, inspección y depuración del código.<br>
• Muchos paquetes y componentes<br>
• Código abierto.<br>

Al ser un código más usado y que tiene la capacidad de tener varias formas de implementación decidimos utilizar<br>
el lenguaje de programación Python<br>
Para el desarrollo de nuestra aplicación de reconocimiento facial hicimos uso de la herramienta PyCharm que<br>
proporciona una finalización del código inteligente, inspecciones del código, indicación de errores sobre la marcha<br>
y arreglos rápidos, así como refactorización de código automática y completas funcionalidades de navegación.<br>
El editor de código inteligente de PyCharm ofrece compatibilidad de primer nivel con Python, JavaScript,<br>
CoffeeScript, TypeScript, CSS, lenguajes de plantilla populares y más. Como en este caso utilizamos el lenguaje<br>
Python para nuestro desarrollo de aplicación se llego a la decisión de utilizar este tipo de ambiente de desarrollo.<br>
Otra herramienta que utilizamos en este proyecto es la librería OpenCV que proporciona un marco de trabajo de<br>
alto nivel para el desarrollo de aplicaciones de visión por computador en tiempo real, estructuras de datos,<br>
procesamiento y análisis de imágenes, análisis estructural, etc.<br>
La librería OpenCV está dirigida fundamentalmente a la visión por computador en tiempo real. Entre sus muchas<br>
áreas de aplicación destacarían: interacción hombre-máquina (HCI4); segmentación y reconocimiento de objetos;<br>
reconocimiento de gestos; seguimiento del movimiento; estructura del movimiento (SFM5); y robots móviles.</p>

<h2>DeepFace</h2><br>

DeepFace es la biblioteca de análisis de atributos y reconocimientos faciales másligera para Python, incluye<br>
todos los modelos de IA de vanguardia para el reconocimiento facial<br>
El reconocimiento facial con DeepFace nos da las siguientes funciones:</p>

<ul>
<li class="has-line-data" data-line-start="44" data-line-end="45">Verificación de rostros</li>
<li class="has-line-data" data-line-start="45" data-line-end="46">Reconocimiento de rostros</li>
<li class="has-line-data" data-line-start="46" data-line-end="47">Análisis de atributos faciales</li>
<li class="has-line-data" data-line-start="47" data-line-end="69">Análisis facial en tiempo real<br>
 </ul>
 
Fue producido por una colección de científicos del equipo de investigación de inteligencia artificial de Facebook.<br>
Este define rostros humanos en imágenes digitales. Emplea una red neuronal de 9 capas con más de 120<br>
millones de conexiones, y ha sido entrenado en 4 millones de imágenes subidas por los usuarios de Facebook.<br>
Comenzó el despliegue de la tecnología a sus usuarios a principios de 2015<br>
DeepFace utiliza detectores de puntos fiduciales basados en bases de datos existentes para dirigir la alineación<br>
de caras. La alineación facial comienza con una alineación en 2D y luego continúa con una alineación y<br>
frontalización en 3D. Es decir, el proceso de DeepFace consta de dos pasos. Primero, corrige los ángulos de una<br>
imagen para que la cara en la foto mire hacia adelante. Para lograr esto, utiliza un modelo tridimensional de una<br>
cara. Luego, el aprendizaje profundo produce una descripción numérica del rostro. Si DeepFace presenta una<br>
descripción lo suficientemente similar para dos imágenes, asume que estas dos imágene s comparten una cara.<br>
La idea principal de DeepFace es integrar las mejores herramientas de reconocimiento de imágenes para el<br>
análisis facial profundo en una biblioteca liviana y flexible. Cualquiera puede adoptar DeepFace en tareas de<br>
nivel de producción con una puntuación de confianza alta para utilizar los algoritmos de código abierto más<br>
potentes.<br>
  
<h2>Opencv</h2><br>
Opencv es una biblioteca de código abierto en cual se implementa más de 2500 algoritmos, se especializa en el<br>
sistema de visión artificial y machine learning.<br>
Nos ayuda a poder detectar objetos y rostros, es aspectos como lo son la seguridad, marketing o la fotografía<br>
incluso.Nos provee infraestructura para aplicacionesrelacionadas con la visión artificial, al ser multiplataforma<br>
lo podemos ejecutar con los principales sistemas operatiovos, lo podemos utilizar en lenguaje c, c++ o Python.</li>

<h2>Código desarrollado</h2><br>

```
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
 mpDibujo.draw_landmarks(frame, rostros,mpMallaFacial.FACE_CONNECTIONS,confdibu, confdibu)
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
 #CEJA IZQUIERDA
 x3, y3 = lista[295][1:]
 x4, y4 = lista[385][1:]
 cx2, cy2 = (x3 + x4) // 2, (y3 + y4) // 2
 logitud2 = math.hypot(x4 - x3, y4 - y3)
 #BOCA EXTREMOS
 x5, y5 = lista[78][1:]
 x6, y6 = lista[308][1:]
 cx3, cy3 = (x5 + x6) // 2, (y5 + y6) // 2
 logitud3 = math.hypot(x6 - x5, y6 - y5)
 #BOCA APERTURA
 x7, y7 = lista[13][1:]
 x8, y8 = lista[14][1:]
 cx4, cy4 = (x7 + x8) // 2, (y7 + y8) // 2
 logitud4 = math.hypot(x8 - x7, y8 - y7)
 #CLASIFICACION
 if logitud1 < 19 and logitud2 < 19 and logitud3 > 80 and logitud3 < 95 and logitud4 <5:
 cv2.putText(frame,'PERSONA ENOJADA',(480,80),cv2.FONT_HERSHEY_SIMPLEX,1,
 (0,0,255),3)
 elif logitud1 > 20 and logitud1 < 30 and logitud2 > 20 and logitud2 < 30 and logitud3 > 109 and logitud4 
> 10 and logitud4 <20:
 cv2.putText(frame, 'PERSONA FELIZ', (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
 (0, 0, 255), 3)
 elif logitud1 > 35 and logitud2 > 35 and logitud3 > 80 and logitud3 <90 and logitud4 >20:
 cv2.putText(frame, 'PERSONA ASOMBRADA', (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
 (0, 0, 255), 3)
 elif logitud1 >20 and logitud1 >35 and logitud2 >20 and logitud2 >35 and logitud3 > 90 and logitud3 < 
95 and logitud4 <5:
 cv2.putText(frame, 'PERSONA TRISTE', (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
 (0, 0, 255), 3)
 cv2.imshow("RECONOCIMIENTO DE EMOCIONES", frame)
 t= cv2.waitkey(1)
 if t == 27:
 break
cap.release()
cv2.destroyAllWindows()
```

<h2>Conclusiones</h2><br>

En esta parte del proyecto al empezar a realizar nuestro código nos encontramos con vayas fallas, es por eso que<br>
tardamos para la entrega de este avance ya que quisimos saber el por que de estos errores y poder corregirlos.<br>
También analizando las distintas fuentes de información y por nuestros conocimientos decidimos mejor utilizar<br>
Pycharm .<br>

<h2>Bibliografía</h2><br>

Arévalo, V., González, J., &amp; Ambrosio, G. (2004). La librería de visión artificial opencv. aplicación a la docencia e<br>
investigación. Base Informática, 40, 61-66.<br>
School, T. (2022, 29 julio). Cómo programar en PyCharm. Tokio School.<br>
<a href="https://www.tokioschool.com/noticias/como-programar-en-pycharm/">https://www.tokioschool.com/noticias/como-programar-en-pycharm/</a><br>
Funcionalidades de PyCharm. (s. f.). Jet Brains. <a href="https://www.jetbrains.com/es-es/pycharm/features/">https://www.jetbrains.com/es-es/pycharm/features/</a><br>
Sefik Serengi, <a href="http://viso.ai">viso.ai</a>, DeepFace: la biblioteca de reconocimiento facial de código abierto más popular, Read<br>
more at: <a href="https://viso.ai/computer-vision/deepface/">https://viso.ai/computer-vision/deepface/</a><br>
DeepFace, <a href="http://hmong.es">hmong.es</a>, <a href="https://hmong.es/wiki/DeepFace">https://hmong.es/wiki/DeepFace</a></p>
