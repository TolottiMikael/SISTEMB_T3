import json
import sys
import math
import cv2
import numpy as np
import mediapipe as mp
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget,QInputDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer


def distancia_euclidiana(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

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
        
    def imprime_Pose(self, camputerVision, frame, height, width):
        # camputerVision.circle(frame, (x1, y1), 8, (0, 255, 0), -1)
        # camputerVision.line(frame, (x1, y1), (x2, y2), (0, 255, 0), -1)
        camputerVision.line(frame, (int(self.ombro_esq.x * width), int(self.ombro_esq.y * height)),
                            (int(self.ombro_dir.x * width), int(self.ombro_dir.y * height)), (255, 0, 0), 3)
        
        camputerVision.line(frame, (int(self.ombro_esq.x * width), int(self.ombro_esq.y * height)),
                            (int(self.quadril_esq.x * width), int(self.quadril_esq.y * height)),
                            (255, 0, 0), 3)
        camputerVision.line(frame, (int(self.ombro_dir.x * width), int(self.ombro_dir.y * height)),
                            (int(self.quadril_dir.x * width), int(self.quadril_dir.y * height)),
                            (255, 0, 0), 3)
        camputerVision.line(frame, (int(self.quadril_esq.x * width), int(self.quadril_esq.y * height)),
                            (int(self.joelho_esq.x * width), int(self.joelho_esq.y * height)),
                            (255, 0, 0), 3)
        camputerVision.line(frame, (int(self.quadril_dir.x * width), int(self.quadril_dir.y * height)),
                            (int(self.joelho_dir.x * width), int(self.joelho_dir.y * height)),
                            (255, 0, 0), 3)
        camputerVision.line(frame, (int(self.joelho_esq.x * width), int(self.joelho_esq.y * height)),
                            (int(self.tornoza_esq.x * width), int(self.tornoza_esq.y * height)),
                            (255, 0, 0), 3)
        camputerVision.line(frame, (int(self.joelho_dir.x * width), int(self.joelho_dir.y * height)),
                            (int(self.tornoza_dir.x * width), int(self.tornoza_dir.y * height)),
                            (255, 0, 0), 3)
        camputerVision.line(frame, (int(self.ombro_esq.x * width), int(self.ombro_esq.y * height)),
                            (int(self.cotovelo_esq.x * width), int(self.cotovelo_esq.y * height)),
                            (255, 0, 0), 3)
        camputerVision.line(frame, (int(self.ombro_dir.x * width), int(self.ombro_dir.y * height)),
                            (int(self.cotovelo_dir.x * width), int(self.cotovelo_dir.y * height)),
                            (255, 0, 0), 3)
        camputerVision.line(frame, (int(self.cotovelo_esq.x * width), int(self.cotovelo_esq.y * height)),
                            (int(self.pulso_esq.x * width), int(self.pulso_esq.y * height)),
                            (255, 0, 0), 3)
        camputerVision.line(frame, (int(self.cotovelo_dir.x * width), int(self.cotovelo_dir.y * height)),
                            (int(self.pulso_dir.x * width), int(self.pulso_dir.y * height)),
                            (255, 0, 0), 3)
        camputerVision.line(frame, (int(self.tornoza_esq.x * width), int(self.tornoza_esq.y * height)),
                            (int(self.ponta_pe_esq.x * width), int(self.ponta_pe_esq.y * height)),
                            (255, 0, 0), 3)
        camputerVision.line(frame, (int(self.tornoza_dir.x * width), int(self.tornoza_dir.y * height)),
                            (int(self.ponta_pe_dir.x * width), int(self.ponta_pe_dir.y * height)),
                            (255, 0, 0), 3)
        
        camputerVision.circle(frame, (int(self.cabeca.x * width), int(self.cabeca.y * height)), 8, (0, 255, 0), -1)
    
    def carregar_de_arquivo(nome_arquivo):
        with open(nome_arquivo, 'r') as arquivo:
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
                "x": int(ponto.x),
                "y": int(ponto.y),
            }

        with open(nome_arquivo, 'w') as arquivo:
            json.dump(dados, arquivo, indent=4)

class CameraApp(QWidget):
    def __init__(self):
        super().__init__()
        self.PosesParaImprimir = []
        self.setWindowTitle("Detectação de Exercícios")
        self.setGeometry(100, 100, 900, 700)

        # Layout
        layout = QVBoxLayout()

        self.label_video = QLabel("Câmera não iniciada")
        layout.addWidget(self.label_video)

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

        
        
        self.setLayout(layout)

        # Configurações da câmera
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()

    def iniciar_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.timer.start(30)

    def parar_camera(self):
        self.timer.stop()
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
            print(f"Pose carregada do arquivo {nome_arquivo}.json")


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
                pose.imprime_Pose(cv2, frame, h, w)
            
                
                      

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
