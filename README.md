
# 🚀 Reconhecimento Facial com MTCNN e FaceNet 🤖

Este projeto utiliza o **MTCNN** para detecção de faces e o **FaceNet** (via TensorFlow Hub) para extração de embeddings faciais. O objetivo é reconhecer faces em imagens e exibir os resultados de forma visual.

---

## 📋 Índice
- [🔧 Funcionalidades](#-funcionalidades)
- [💻 Requisitos](#-requisitos)
- [🔧 Instalação](#-instalação)
- [🚀 Uso](#-uso)
- [📂 Estrutura do Projeto](#-estrutura-do-projeto)
- [📝 Notas Técnicas](#-notas-técnicas)
- [📜 Licença](#-licença)
- [🙏 Reconhecimentos](#-reconhecimentos)
- [🌐 Links Úteis](#-links-úteis)

---

## 🔧 Funcionalidades
- **👤 Detecção de Faces**: Utiliza o MTCNN para identificar rostos em imagens.  
- **📊 Extração de Embeddings**: Gera embeddings faciais usando o FaceNet.  
- **🖼️ Visualização de Resultados**: Exibe as faces detectadas com bounding boxes.  
- **🤖 Integração com TensorFlow Hub**: Utiliza modelos pré-treinados para extração de características.  

---

## 💻 Requisitos
- **Google Colab** (ou ambiente Python com GPU).  
- Bibliotecas:  
  tensorflow, opencv-python, numpy, matplotlib, scikit-learn, keras, mtcnn
## 🔧 Instalação
1. Instalar Dependências
```bash
!pip install tensorflow opencv-python numpy matplotlib scikit-learn keras mtcnn
```
2. Importar Bibliotecas
```bash

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mtcnn import MTCNN
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
```
## 🚀 Uso
1. Inicializar o Detector MTCNN
```bash detector = MTCNN()
```
3. Função para Detectar Faces
```bash
def detect_faces(image):
    faces = detector.detect_faces(image)
    return faces
```
3. Função para Desenhar Bounding Boxes
```bash
def draw_faces(image, faces):
    for face in faces:
        x, y, w, h = face['box']
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image
```
4. Carregar o Modelo FaceNet

```bash
facenet_model = hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5", input_shape=(224, 224, 3))
```
5. Pré-processamento e Extração de Embeddings
```bash
def preprocess_face(face_pixels):
    face_pixels = cv2.resize(face_pixels, (224, 224))
    face_pixels = face_pixels.astype('float32') / 255.0
    return face_pixels

def get_face_embedding(face_pixels):
    face_pixels = preprocess_face(face_pixels)
    samples = np.expand_dims(face_pixels, axis=0)
    embedding = facenet_model(samples)
    reduced_embedding = dense_layer(embedding)
    return reduced_embedding.numpy()[0]
```
6. Reconhecer Faces em uma Imagem
```bash
from google.colab import files
from IPython.display import Image, display
```
# 📤 Upload de uma imagem
```bash
uploaded = files.upload()
image_path = list(uploaded.keys())[0]
```
# 🖼️ Carrega a imagem
```bash
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
```
# 🔍 Detecta e desenha as faces
```bash
faces = detect_faces(image)
result = draw_faces(image, faces)
```
# 📊 Exibe o resultado
```bash
plt.figure(figsize=(10, 10))
plt.imshow(result)
plt.axis('off')
plt.show()
```
## 📂 Estrutura do Projeto
```bash
/content/
├── facial_recognition_project/       # Pasta principal do projeto
│   ├── main.ipynb                    # Notebook principal
│   ├── requirements.txt              # Lista de dependências
│   ├── models/                       # Pasta para armazenar modelos
│   │   └── facenet_model/            # Modelo FaceNet (se salvo localmente)
│   ├── images/                       # Pasta para armazenar imagens de teste
│   │   ├── input/                    # Imagens de entrada para processamento
│   │   └── output/                   # Imagens processadas com faces detectadas
│   ├── embeddings/                   # Pasta para armazenar embeddings faciais
│   └── utils/                        # Funções utilitárias
│       ├── face_detection.py         # Funções para detecção de faces (MTCNN)
│       ├── face_embedding.py         # Funções para extração de embeddings (FaceNet)
│       └── visualization.py          # Funções para visualização de resultados
└── README.md                         # Documentação do projeto
```
### 📝 Notas Técnicas
👤 MTCNN: Detecta faces com alta precisão, mesmo em condições desafiadoras.

📊 FaceNet: Gera embeddings de 128 dimensões para reconhecimento facial.

🤖 TensorFlow Hub: Facilita o uso de modelos pré-treinados.

### 📜 Licença
Este projeto é distribuído sob a licença MIT. Para mais detalhes, consulte o arquivo LICENSE.

### 🙏 Reconhecimentos
MTCNN: Biblioteca para detecção de faces.

TensorFlow Hub: Plataforma para modelos pré-treinados.

Google Colab: Ambiente de execução com suporte a GPU.

### 🌐 Links Úteis
- **[📚 Documentação do TensorFlow](https://www.tensorflow.org/api_docs)**  
- **[🔗 Repositório MTCNN no GitHub](https://github.com/ipazc/mtcnn)**  
- **[📄 Paper FaceNet](https://arxiv.org/abs/1503.03832)**  
- **[🎓 Tutorial Google Colab](https://colab.research.google.com/)**  
