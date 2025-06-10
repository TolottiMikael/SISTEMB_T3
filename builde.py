import cv2
import math
import mediapipe as mp

# Função para cálculo da distância euclidiana entre dois pontos.
def distancia_euclidiana(x1, y1, x2, y2):
    distancia = math.sqrt((x2-x1)**2+(y2-y1)**2)
    return distancia

# Cálculo de ângulo.
def calculo_angulo(x1, y1, x2, y2):
    theta = math.acos((y2-y1)*(-y1)/(math.sqrt((x2-x1)**2+(y2-y1)**2)*y1))
    degree = int(180/math.pi)*theta
    return degree

# Fonte a ser usada nos textos colocados na imagem.
fonte = cv2.FONT_HERSHEY_SIMPLEX
 
# Tabela de cores.
blue    = (255,0,0)
red     = (0,0,255)
green   = (0,255,0)
yellow  = (0,255,255)
pink    = (255,0,255)


# Inicialização do "mediapipe pose class".
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Captura da câmera
captura = cv2.VideoCapture(0)

while captura.isOpened():
    _,imagem = captura.read()
        
    # Pegar o tamanho da imagem (height - altura, width - largura)
    h, w = imagem.shape[0:2]
     
    # Converte a imagem de BGR para RGB.
    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
    
    # Processamento da imagem.
    resultados = pose.process(imagem)

    imagem = cv2.cvtColor(imagem, cv2.COLOR_RGB2BGR)
    
    # Usar o lm e o lmPose para representar os métodos.
    lm = resultados.pose_landmarks
    lmPose  = mp_pose.PoseLandmark

    #  Determinar as coordenadas x e y de cada ponto no esqueleto.
    ombro_esq_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
    ombro_esq_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)

    pulso_esq_x = int(lm.landmark[lmPose.LEFT_WRIST].x * w)
    pulso_esq_y = int(lm.landmark[lmPose.LEFT_WRIST].y * h)

    cotov_esq_x = int(lm.landmark[lmPose.LEFT_ELBOW].x * w)
    cotov_esq_y = int(lm.landmark[lmPose.LEFT_ELBOW].y * h)

    ombro_dir_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
    ombro_dir_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)

    pulso_dir_x = int(lm.landmark[lmPose.RIGHT_WRIST].x * w)
    pulso_dir_y = int(lm.landmark[lmPose.RIGHT_WRIST].y * h)

    cotov_dir_x = int(lm.landmark[lmPose.RIGHT_ELBOW].x * w)
    cotov_dir_y = int(lm.landmark[lmPose.RIGHT_ELBOW].y * h)


    # Desenhando os círculos nos pontos de coordenadas.
    cv2.circle(imagem, (ombro_esq_x, ombro_esq_y), 7, red, -1)
    cv2.circle(imagem, (ombro_dir_x, ombro_dir_y), 7, red, -1)

    # desenhando nos pulsos
    cv2.circle(imagem, (pulso_esq_x, pulso_esq_y), 7, blue, -1)
    cv2.circle(imagem, (pulso_dir_x, pulso_dir_y), 7, blue, -1)

    # Usando Distância Euclidiana, determinar a distância entre os pontos do ombros.
    valor = distancia_euclidiana(ombro_esq_x, ombro_esq_y, ombro_dir_x, ombro_dir_y)
    valorOMB_PUL_ESQ = distancia_euclidiana(ombro_esq_x, ombro_esq_y, pulso_esq_x, pulso_esq_y)
    valorOMB_PUL_DIR = distancia_euclidiana(ombro_dir_x, ombro_dir_y, pulso_dir_x, pulso_dir_y)
    
    # Unir as duas marcações feitas (cv2.circle) com uma linha.
    cv2.line(imagem, (ombro_esq_x, ombro_esq_y), (pulso_esq_x, pulso_esq_y), blue, 4)
    cv2.line(imagem, (ombro_dir_x, ombro_dir_y), (pulso_dir_x, pulso_dir_y), blue, 4)
    

    # Anotar na imagem o valor da distância entre os pontos dos ombros.
    cv2.putText(imagem, str(int(valor)), (10, 30), fonte, 0.9, green, 2)

    if valorOMB_PUL_ESQ < 200:
        if (cotov_esq_y - ombro_esq_y) < 30:
            cv2.putText(imagem, "Triceps Banco feito na esquerda", (10, 60), fonte, 0.9, green, 2)
    if valorOMB_PUL_DIR < 200:
        if (cotov_dir_y - ombro_dir_y) < 30:
            cv2.putText(imagem, "Triceps Banco feito na direita", (10, 90), fonte, 0.9, green, 2)

    
    # Unir as duas marcações feitas (cv2.circle) com uma linha.
    cv2.line(imagem, (ombro_esq_x, ombro_esq_y), (ombro_dir_x, ombro_dir_y), yellow, 4)

    # Mostrar a imagem com as marcações acima.
    cv2.imshow("Imagem com contornos", imagem)
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

captura.release()
cv2.destroyAllWindows()