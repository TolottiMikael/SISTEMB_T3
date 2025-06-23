import json
import sys
import math
import cv2
import numpy as np
import mediapipe as mp
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget,QInputDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer


def distancia_euclidiana(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def ConferePoses(RefPose, PoseAtual, distancia):

    for atributo in RefPose.__dict__:
        d = distancia_euclidiana(RefPose.__dict__[atributo].x, RefPose.__dict__[atributo].y, 
                                 PoseAtual.__dict__[atributo].x, PoseAtual.__dict__[atributo].y)
        
        if (d > distancia):
            #print(f"{atributo}: maior => {d}")
            return False
    return True

class posicaoCorpo:
    def __init__(self, poseAtual, lmPose):
        self.ombro_esq =    poseAtual[lmPose.LEFT_SHOULDER]
        self.ombro_dir =    poseAtual[lmPose.RIGHT_SHOULDER]
        self.quadril_esq =  poseAtual[lmPose.LEFT_HIP]
        self.quadril_dir =  poseAtual[lmPose.RIGHT_HIP]
        self.joelho_esq =   poseAtual[lmPose.LEFT_KNEE]
        self.joelho_dir =   poseAtual[lmPose.RIGHT_KNEE]
        self.tornoza_esq =  poseAtual[lmPose.LEFT_ANKLE]
        self.tornoza_dir =  poseAtual[lmPose.RIGHT_ANKLE]
        self.cotovelo_esq = poseAtual[lmPose.LEFT_ELBOW]
        self.cotovelo_dir = poseAtual[lmPose.RIGHT_ELBOW]
        self.pulso_esq =    poseAtual[lmPose.LEFT_WRIST]
        self.pulso_dir =    poseAtual[lmPose.RIGHT_WRIST]
        self.ponta_pe_esq = poseAtual[lmPose.LEFT_FOOT_INDEX]
        self.ponta_pe_dir = poseAtual[lmPose.RIGHT_FOOT_INDEX]
        self.cabeca =       poseAtual[lmPose.NOSE]
        
    def imprime_Pose(self, camputerVision, frame, height, width, color = (255,0,0)):
        # camputerVision.circle(frame, (x1, y1), 8, (0, 255, 0), -1)
        # camputerVision.line(frame, (x1, y1), (x2, y2), (0, 255, 0), -1)
        camputerVision.line(frame, (int(self.ombro_esq.x * width), int(self.ombro_esq.y * height)),
                            (int(self.ombro_dir.x * width), int(self.ombro_dir.y * height)), color, 3)
        
        camputerVision.line(frame, (int(self.ombro_esq.x * width), int(self.ombro_esq.y * height)),
                            (int(self.quadril_esq.x * width), int(self.quadril_esq.y * height)),
                            color, 3)
        camputerVision.line(frame, (int(self.ombro_dir.x * width), int(self.ombro_dir.y * height)),
                            (int(self.quadril_dir.x * width), int(self.quadril_dir.y * height)),
                            color, 3)
        camputerVision.line(frame, (int(self.quadril_esq.x * width), int(self.quadril_esq.y * height)),
                            (int(self.joelho_esq.x * width), int(self.joelho_esq.y * height)),
                            color, 3)
        camputerVision.line(frame, (int(self.quadril_dir.x * width), int(self.quadril_dir.y * height)),
                            (int(self.joelho_dir.x * width), int(self.joelho_dir.y * height)),
                            color, 3)
        camputerVision.line(frame, (int(self.joelho_esq.x * width), int(self.joelho_esq.y * height)),
                            (int(self.tornoza_esq.x * width), int(self.tornoza_esq.y * height)),
                            color, 3)
        camputerVision.line(frame, (int(self.joelho_dir.x * width), int(self.joelho_dir.y * height)),
                            (int(self.tornoza_dir.x * width), int(self.tornoza_dir.y * height)),
                            color, 3)
        camputerVision.line(frame, (int(self.ombro_esq.x * width), int(self.ombro_esq.y * height)),
                            (int(self.cotovelo_esq.x * width), int(self.cotovelo_esq.y * height)),
                            color, 3)
        camputerVision.line(frame, (int(self.ombro_dir.x * width), int(self.ombro_dir.y * height)),
                            (int(self.cotovelo_dir.x * width), int(self.cotovelo_dir.y * height)),
                            color, 3)
        camputerVision.line(frame, (int(self.cotovelo_esq.x * width), int(self.cotovelo_esq.y * height)),
                            (int(self.pulso_esq.x * width), int(self.pulso_esq.y * height)),
                            color, 3)
        camputerVision.line(frame, (int(self.cotovelo_dir.x * width), int(self.cotovelo_dir.y * height)),
                            (int(self.pulso_dir.x * width), int(self.pulso_dir.y * height)),
                            color, 3)
        camputerVision.line(frame, (int(self.tornoza_esq.x * width), int(self.tornoza_esq.y * height)),
                            (int(self.ponta_pe_esq.x * width), int(self.ponta_pe_esq.y * height)),
                            color, 3)
        camputerVision.line(frame, (int(self.tornoza_dir.x * width), int(self.tornoza_dir.y * height)),
                            (int(self.ponta_pe_dir.x * width), int(self.ponta_pe_dir.y * height)),
                            color, 3)
        
        camputerVision.circle(frame, (int(self.cabeca.x * width), int(self.cabeca.y * height)), 8, (0, 255, 0), -1)
    
    def carregar_de_arquivo(nome_arquivo):
        with open(f"poses/{nome_arquivo}", 'r') as arquivo:
            dados = json.load(arquivo)

        from mediapipe.framework.formats import landmark_pb2
        pose = posicaoCorpo.__new__(posicaoCorpo)  # Cria objeto sem chamar __init__

        for nome, ponto in dados.items():
            lm = landmark_pb2.NormalizedLandmark()
            lm.x = ponto["x"]
            lm.y = ponto["y"]
            lm.z = ponto.get("z", 0)
            lm.visibility = ponto.get("visibility", 0)
            setattr(pose, nome, lm)

        return pose

    
    def salva_arquivo(self, nome_arquivo="dados.json"):
        dados = {}

        for nome,ponto in self.__dict__.items():
            dados[nome] = {
                "x": ponto.x,
                "y": ponto.y,
                "z": getattr(ponto, "z", 0),
                "visibility": getattr(ponto, "visibility", 0)
            }
        with open(f"poses/{nome_arquivo}", 'w') as arquivo:
            json.dump(dados, arquivo, indent=4)

class CameraApp(QWidget):
    def __init__(self):
        super().__init__()
        
        self.PoseObjetivo = None
        self.PosesParaImprimir = []
        self.ContagemExercicios = 0
        self.contaUmaVez = False
        
        
        self.setWindowTitle("Detecção de Exercícios")
        self.setGeometry(100, 100, 900, 700)

        # Layout
        
        columns = QHBoxLayout()
        layout = QVBoxLayout()

        self.label_video = QLabel("Câmera não iniciada")
        columns.addWidget(self.label_video)

        self.botao_iniciar = QPushButton("Iniciar Câmera")
        self.botao_iniciar.clicked.connect(self.iniciar_camera)
        layout.addWidget(self.botao_iniciar)

        self.botao_parar = QPushButton("Parar Câmera")
        self.botao_parar.clicked.connect(self.parar_camera)
        layout.addWidget(self.botao_parar)
        
        self.botao_salvar_Pose = QPushButton("Salvar Pose")
        self.botao_salvar_Pose.clicked.connect(self.salvar_Pose)
        layout.addWidget(self.botao_salvar_Pose)
        
        
        self.botao_carregar_Pose = QPushButton("carregar Pose")
        self.botao_carregar_Pose.clicked.connect(self.carregar_Pose)
        layout.addWidget(self.botao_carregar_Pose)

        self.botao_limpaPoses = QPushButton("Clear nice!")
        self.botao_limpaPoses.clicked.connect(self.apagarPoses)
        layout.addWidget(self.botao_limpaPoses)
        
        self.botao_carregar_exerc = QPushButton("Carregar Exercícios")
        self.botao_carregar_exerc.clicked.connect(self.carregar_exercicios)
        layout.addWidget(self.botao_carregar_exerc)
        
        columns.addLayout(layout)
        self.setLayout(columns)

        # Configurações da câmera
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        self.timerExercicio = QTimer()
        self.timerExercicio.timeout.connect(self.checkExercicio)

        # MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()

    def iniciar_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.timer.start(30)
        #checa os exercícios
        self.timerExercicio.start(100)

    def parar_camera(self):
        self.timer.stop()
        self.timerExercicio.stop()
        
        self.PoseObjetivo = None
        if self.cap is not None:
            self.cap.release()
        self.label_video.setPixmap(QPixmap())

    def salvar_Pose(self):
        nome_arquivo, ok = QInputDialog.getText(self, "Salvar Pose", "Digite o nome do arquivo:")
        if ok and nome_arquivo:
            self.PoseFinal.salva_arquivo(f"{nome_arquivo}.json")
            self.nome_arquivo = self.PoseFinal
            self.PosesParaImprimir.append(self.nome_arquivo)
            print(f"Pose salva no arquivo {nome_arquivo}.json")
    
    def carregar_Pose(self):
        nome_arquivo, ok = QInputDialog.getText(self, "Carregar Pose", "Digite o nome do arquivo:")
        if ok and nome_arquivo:
            pose_carregada = posicaoCorpo.carregar_de_arquivo(f"{nome_arquivo}.json")
            self.PosesParaImprimir.append(pose_carregada)
            
            #apagar pois é para teste
            self.PoseObjetivo = pose_carregada
            
            print(f"Pose carregada do arquivo {nome_arquivo}.json")

    def apagarPoses(self):
        self.PosesParaImprimir = []
        self.PoseObjetivo = None
        self.ExercicioCarregado = None
        self.ContagemExercicios = 0
    
    def carregar_exercicios(self):
        self.PosesParaImprimir = []
        self.ContagemExercicios = 0
        arquivoExerc, ok = QInputDialog.getText(self, "Carregar Exercícios", "Digite o nome do arquivo:")
        
        if ok and arquivoExerc:
            with open(f"exercicios/{arquivoExerc}.json", 'r') as arquivo:
                dados = json.load(arquivo)
                self.ExercicioCarregado = dados['exercicios']
                self.etapa_exercicio = 0  # Começa no primeiro
                print("Exercício carregado com sucesso!")
    
    def checkExercicio(self):
        if not hasattr(self, 'ExercicioCarregado'):
            return
        
        if self.ExercicioCarregado is not None:
            for exerci in self.ExercicioCarregado:
                if exerci['indice'] == self.etapa_exercicio:
                    self.PoseObjetivo = posicaoCorpo.carregar_de_arquivo(f"{exerci['nome']}.json")
                    self.PosesParaImprimir = [self.PoseObjetivo]
                    #print(f"Pose objetivo: {exerci['nome']}")
                    if self.PoseObjetivo is not None and hasattr(self, 'PoseFinal'):
                        if ConferePoses(self.PoseObjetivo, self.PoseFinal, 0.1):
                            print(f"Pose {exerci['nome']} correta. Indo para próximo exercício...")
                            self.etapa_exercicio = exerci.get('proxIndice', self.etapa_exercicio)  # Avança
                            if self.etapa_exercicio == 0:
                                if self.contaUmaVez == False:
                                    self.ContagemExercicios += 1
                                    self.contaUmaVez = True
                                    print(f"Exercício concluido! Contagem de exercícios: {self.ContagemExercicios}")
                            else:
                                self.contaUmaVez = False
                                print(f"Proximos exercícios: {self.etapa_exercicio}")
                    else:
                        print("Pose não detectada ou sem pose final.")
                
    def update_frame(self):
        
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)  # Espelhar imagem

        # Processamento com MediaPipe
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado = self.pose.process(img_rgb)

        h, w = frame.shape[:2]

        if resultado.pose_landmarks:
            lm = resultado.pose_landmarks
            lmPose = self.mp_pose.PoseLandmark

            corpinho = posicaoCorpo(lm.landmark, lmPose)
            corpinho.imprime_Pose(cv2, frame, h, w)
            self.PoseFinal = corpinho
            
            for pose in self.PosesParaImprimir:
                pose.imprime_Pose(cv2, frame, h, w, color = (0, 0, 255))

            if self.ContagemExercicios is not None:
                cv2.putText(frame, f"Contagem de exercicios: {self.ContagemExercicios}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Converter a imagem para exibir no QLabel
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(
            img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(800, 600)
        self.label_video.setPixmap(QPixmap.fromImage(p))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec_())
