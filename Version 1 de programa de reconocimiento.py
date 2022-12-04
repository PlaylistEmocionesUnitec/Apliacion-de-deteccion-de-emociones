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

