import cv2
import math
import mediapipe as mp

inicioMovimento = False
fimMovimento = False
repeticoes = 0

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

    print("vendo os resultados")
    print(lm)

    if lm is not None:
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

        quadril_esq_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
        quadril_esq_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)

        quadril_dir_x = int(lm.landmark[lmPose.RIGHT_HIP].x * w)
        quadril_dir_y = int(lm.landmark[lmPose.RIGHT_HIP].y * h)

        joelho_esq_x = int(lm.landmark[lmPose.LEFT_KNEE].x * w)
        joelho_esq_y = int(lm.landmark[lmPose.LEFT_KNEE].y * h)

        joelho_dir_x = int(lm.landmark[lmPose.RIGHT_KNEE].x * w)
        joelho_dir_y = int(lm.landmark[lmPose.RIGHT_KNEE].y * h)

        tornoza_esq_x = int(lm.landmark[lmPose.LEFT_ANKLE].x * w)
        tornoza_esq_y = int(lm.landmark[lmPose.LEFT_ANKLE].y * h)

        tornoza_dir_x = int(lm.landmark[lmPose.RIGHT_ANKLE].x * w)
        tornoza_dir_y = int(lm.landmark[lmPose.RIGHT_ANKLE].y * h)


        # Desenhando os círculos nos pontos de coordenadas.
        cv2.circle(imagem, (ombro_esq_x, ombro_esq_y), 7, red, -1)
        cv2.circle(imagem, (ombro_dir_x, ombro_dir_y), 7, red, -1)

        # desenhando nos pulsos
        cv2.circle(imagem, (pulso_esq_x, pulso_esq_y), 7, blue, -1)
        cv2.circle(imagem, (pulso_dir_x, pulso_dir_y), 7, blue, -1)


        # desenhando nos COTOVELOS
        cv2.circle(imagem, (cotov_esq_x, cotov_esq_y), 7, blue, -1)
        cv2.circle(imagem, (cotov_dir_x, cotov_dir_y), 7, blue, -1)

        #DESENHANDO NOS QUADRIS
        cv2.circle(imagem, (quadril_esq_x, quadril_esq_y), 7, blue, -1)
        cv2.circle(imagem, (quadril_dir_x, quadril_dir_y), 7, blue, -1)

        #DESENHANDO NOS JOELHOS
        cv2.circle(imagem, (joelho_esq_x, joelho_esq_y), 7, blue, -1)
        cv2.circle(imagem, (joelho_dir_x, joelho_dir_y), 7, blue, -1)

        #DESENHANDO NOS TORNOZELOS
        cv2.circle(imagem, (tornoza_esq_x, tornoza_esq_y), 7, blue, -1)
        cv2.circle(imagem, (tornoza_dir_x, tornoza_dir_y), 7, blue, -1)



        # Usando Distância Euclidiana, determinar a distância entre os pontos do ombros.
        valor = distancia_euclidiana(ombro_esq_x, ombro_esq_y, ombro_dir_x, ombro_dir_y)
        valorOMB_PUL_ESQ = distancia_euclidiana(ombro_esq_x, ombro_esq_y, pulso_esq_x, pulso_esq_y)
        valorOMB_PUL_DIR = distancia_euclidiana(ombro_dir_x, ombro_dir_y, pulso_dir_x, pulso_dir_y)

        # Unir as duas marcações feitas (cv2.circle) com uma linha.
        cv2.line(imagem, (ombro_esq_x, ombro_esq_y), (pulso_esq_x, pulso_esq_y), blue, 4)
        cv2.line(imagem, (ombro_dir_x, ombro_dir_y), (pulso_dir_x, pulso_dir_y), blue, 4)


        # Anotar na imagem o valor da distância entre os pontos dos ombros.
        cv2.putText(imagem, str(int(valor)), (10, 30), fonte, 0.9, green, 2)


        # Unir as duas marcações feitas (cv2.circle) com uma linha.
        cv2.line(imagem, (ombro_esq_x, ombro_esq_y), (ombro_dir_x, ombro_dir_y), yellow, 4)

        cv2.line(imagem, (ombro_dir_x, ombro_dir_y), (quadril_dir_x, quadril_dir_y), yellow, 4)
        cv2.line(imagem, (ombro_esq_x, ombro_esq_y), (quadril_esq_x, quadril_esq_y), yellow, 4)

        cv2.line(imagem, (quadril_esq_x, quadril_esq_y), (quadril_dir_x, quadril_dir_y), yellow, 4)

        cv2.line(imagem, (joelho_dir_x, joelho_dir_y), (quadril_dir_x, quadril_dir_y), yellow, 4)
        cv2.line(imagem, (joelho_esq_x, joelho_esq_y), (quadril_esq_x, quadril_esq_y), yellow, 4)
        
        cv2.line(imagem, (joelho_dir_x, joelho_dir_y), (tornoza_dir_x, tornoza_dir_y), yellow, 4)
        cv2.line(imagem, (joelho_esq_x, joelho_esq_y), (tornoza_esq_x, tornoza_esq_y), yellow, 4)

        cv2.line(imagem, (ombro_dir_x, ombro_dir_y), (cotov_dir_x, cotov_dir_y), yellow, 4)
        cv2.line(imagem, (ombro_esq_x, ombro_esq_y), (cotov_esq_x, cotov_esq_y), yellow, 4)

        cv2.line(imagem, (pulso_dir_x, pulso_dir_y), (cotov_dir_x, cotov_dir_y), yellow, 4)
        cv2.line(imagem, (pulso_esq_x, pulso_esq_y), (cotov_esq_x, cotov_esq_y), yellow, 4)


        #anota na imagem o angulo dessa linha:
        anguloOmbroAoCotovelo = calculo_angulo(ombro_esq_x, ombro_esq_y, cotov_esq_x, cotov_esq_y)
        anguloCotoveloAoPulso = calculo_angulo(cotov_esq_x, cotov_esq_y, pulso_esq_x, pulso_esq_y)
        anguloJoelhoAoTornoza = calculo_angulo(joelho_esq_x, joelho_esq_y, tornoza_esq_x, tornoza_esq_y)
        if (inicioMovimento == False) and (anguloOmbroAoCotovelo < 50) and (anguloCotoveloAoPulso < 50) and (anguloJoelhoAoTornoza < 161):
            inicioMovimento = True
            fimMovimento = False
        if (inicioMovimento == True) and (anguloOmbroAoCotovelo > 165) and (anguloCotoveloAoPulso > 165) and (anguloJoelhoAoTornoza > 169):
            inicioMovimento = False
            fimMovimento = True
            repeticoes += 1

        cv2.putText(imagem, str(anguloOmbroAoCotovelo), (10, 90), fonte, 0.9, green, 2)
        cv2.putText(imagem, str(anguloCotoveloAoPulso), (10, 120), fonte, 0.9, green, 2)
        cv2.putText(imagem, str(anguloJoelhoAoTornoza), (10, 150), fonte, 0.9, green, 2)
        cv2.putText(imagem, str(repeticoes), (10, 180), fonte, 0.9, green, 2)


    else:
        # Pose não detectada nesta frame, apenas continue o loop
        pass
    # Mostrar a imagem com as marcações acima.
    cv2.imshow("Imagem com contornos", imagem)
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

captura.release()
cv2.destroyAllWindows()