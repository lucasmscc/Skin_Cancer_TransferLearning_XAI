# Reprodução do Artigo: Detecção de Câncer de Pele com IA Explicável

**Referência completa:**
> Shah, S.A.H.; Shah, S.T.H.; Khaled, R.; Buccoliero, A.; Shah, S.B.H.; Di Terlizzi, A.; Di Benedetto, G.; Deriu, M.A. *Explainable AI-Based Skin Cancer Detection Using CNN, Particle Swarm Optimization and Machine Learning.* J. Imaging **2024**, 10, 332. https://doi.org/10.3390/jimaging10120332

**Notebook de reprodução:** `../notebooks/reproducao_artigo.ipynb`

---

## 1. Contexto e Motivação

O câncer de pele é um dos cânceres mais prevalentes no mundo. O diagnóstico precoce e preciso é fundamental para melhorar os desfechos clínicos. Métodos tradicionais dependem muito da avaliação visual por dermatologistas, sendo subjetivos e demorados.

O artigo propõe um pipeline baseado em IA para classificar lesões cutâneas em **benignas** e **malignas**, combinando:

- Transfer learning com a rede CNN Xception
- Extração de features profundas
- Seleção de features via Particle Swarm Optimization (PSO)
- Classificadores de aprendizado de máquina clássicos
- Técnicas de IA Explicável (XAI) para interpretabilidade

---

## 2. Dataset

### 2.1 ISIC 2018 — Dataset Principal

O dataset utilizado é o **ISIC 2018 Skin Cancer: Malignant vs. Benign**, disponível publicamente no Kaggle. É derivado do desafio *Skin Lesion Analysis Toward Melanoma Detection 2018* promovido pela International Skin Imaging Collaboration (ISIC).

| Conjunto | Benigno | Maligno | Total |
|----------|---------|---------|-------|
| Treino   | 1440    | 1197    | 2637  |
| Teste    | 360     | 300     | 660   |

- Formato: imagens RGB em `.jpg`
- Dimensão original: 224×224 pixels

### 2.2 HAM10000 — Dataset de Holdout

O artigo também valida o pipeline no **HAM10000** (Human Against Machine with 10,000 training images), usado como conjunto de generalização completamente não visto. Na reprodução deste notebook, o foco está no ISIC 2018 (conjunto principal).

---

## 3. Pré-processamento e Augmentação

### 3.1 Normalização

Todas as imagens são redimensionadas para **224×224 pixels** e normalizadas para o intervalo **[0, 1]**, dividindo os valores de pixel por 255.

### 3.2 Augmentação de Dados (apenas treino)

O artigo aplica três técnicas de augmentação **exclusivamente sobre os dados de treino**, preservando as imagens originais. Cada imagem gera 3 versões adicionais, totalizando **4× o volume original**:

| Técnica | Parâmetros | Resultado |
|---------|-----------|-----------|
| Rotação aleatória | −180° a +180° | Captura variações de orientação |
| Gaussian blur | σ = 2 | Suaviza ruído de alta frequência |
| Sharpening | amount=2, radius=1, threshold=0 | Realça bordas e detalhes finos |

**Volume após augmentação:**

| Classe    | Antes | Depois |
|-----------|-------|--------|
| Benigno   | 1440  | 5760   |
| Maligno   | 1197  | 4788   |
| **Total** | **2637** | **10548** |

### 3.3 Divisão Treino/Validação

O conjunto de treino aumentado é dividido em **70% treino e 30% validação** (split estratificado), conforme descrito no artigo.

---

## 4. Arquitetura e Experimentos

O artigo conduz **três experimentos** progressivos, todos baseados na rede Xception.

### 4.1 Xception como Backbone

**Xception** (*Extreme Inception*) é uma CNN proposta por Chollet (2017) baseada em *depthwise separable convolutions*. Com pesos pré-treinados no ImageNet, ela oferece excelente balanço entre eficiência computacional e capacidade de extração de features.

- Input shape: 224×224×3
- Output (include_top=False): feature map de 7×7×2048
- Após GlobalAveragePooling2D: vetor de **2048 features**

**Estratégia de congelamento:** O artigo realiza um estudo de ablação e conclui que congelar **100% das camadas** da Xception é a configuração ótima — preserva o conhecimento do ImageNet e reduz custo computacional.

---

### Experimento 1 — Transfer Learning Direto (Classificação Softmax)

**Objetivo:** Classificar diretamente benigno vs. maligno com a Xception + cabeça classificadora customizada.

**Arquitetura completa:**

```
Input (224×224×3)
    ↓
Xception (pré-treinado, 100% frozen)
    ↓
GlobalAveragePooling2D  →  (2048,)
    ↓
Dense(512, ReLU)
    ↓
Dense(256, ReLU)
    ↓
Dense(2, Softmax)  →  [P(benigno), P(maligno)]
```

**Configuração de treinamento:**

| Hiperparâmetro | Valor |
|---------------|-------|
| Otimizador | Adam (lr padrão) |
| Loss | Categorical Crossentropy |
| Épocas | 5 |
| Batch size | 32 |

**Resultados reportados no artigo:**

| Conjunto | Acurácia | Sensibilidade | Especificidade |
|----------|----------|---------------|----------------|
| Validação | 94.15% | 95.9% | 92.2% |
| **Teste** | **89.24%** | **93.9%** | **84.6%** |

---

### Experimento 2 — Extração de Features + Classificadores ML

**Objetivo:** Usar o Xception treinado como extrator de features e avaliar múltiplos classificadores clássicos de ML.

**Pipeline:**

```
Imagens originais de treino (sem augmentação)
    ↓
Xception → GlobalAveragePooling2D
    ↓
Vetor de features: (2048,) por imagem
    ↓
StandardScaler
    ↓
Classificadores ML (SVM, KNN, Ensemble)
```

**Classificadores testados:**

| Grupo | Modelo |
|-------|--------|
| SVM | Linear, Medium Gaussian, Coarse Gaussian, Fine Gaussian |
| KNN | Fine (k=1), Medium (k=10), Coarse (k=100), Cosine (k=5), Weighted (k=10) |
| Ensemble | Boosted Trees, Bagged Trees, Subspace Discriminant |

**Melhor resultado reportado no artigo — Medium Gaussian SVM:**

| Conjunto | Acurácia | Sensibilidade | Especificidade |
|----------|----------|---------------|----------------|
| Validação | 98.6% | 98.8% | 98.0% |
| **Teste** | **89.5%** | **93.4%** | **85.5%** |

---

### Experimento 3 — PSO + Subspace KNN

**Objetivo:** Usar o PSO para selecionar o subconjunto ótimo de features e depois classificar com Subspace KNN.

#### 3a. Seleção de Features com PSO

O **Particle Swarm Optimization** (Kennedy & Eberhart, 1995) é um algoritmo meta-heurístico inspirado no comportamento coletivo de pássaros e peixes. Cada partícula representa uma solução candidata (vetor binário indicando quais features usar).

**Configuração do PSO (Binary PSO):**

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| n_particles | 30 | Número de partículas no enxame |
| iterations | 100 | Número de iterações |
| c1 | 0.5 | Componente cognitivo (atração ao melhor pessoal) |
| c2 | 0.3 | Componente social (atração ao melhor global) |
| w | 0.9 | Inércia (peso da velocidade anterior) |
| Fitness function | KNN (k=5) | Avalia o subconjunto de features com KNN |
| Objetivo | Minimizar `1 − acurácia` | |

**Redução de dimensionalidade:** De 2048 → ~508 features (redução de ~75%).

#### 3b. Subspace KNN

O **Subspace KNN** é um ensemble de classificadores KNN treinados em subconjuntos aleatórios de features (implementado como `BaggingClassifier` com `KNeighborsClassifier` de base).

**Melhor resultado reportado no artigo:**

| Conjunto | Acurácia | Sensibilidade | Especificidade | Precisão | F1 |
|----------|----------|---------------|----------------|----------|----|
| Validação | 97.6% | 97.9% | 97.1% | 97.5% | 97.8% |
| **Teste** | **98.5%** | **98.1%** | **98.9%** | **99.1%** | **98.6%** |

---

## 5. Técnicas de IA Explicável (XAI)

A interpretabilidade é central no artigo. Três técnicas são aplicadas sobre amostras do conjunto de teste para visualizar as regiões que o modelo considera mais relevantes.

### 5.1 Grad-CAM (Gradient-weighted Class Activation Mapping)

Utiliza os gradientes da pontuação de classe em relação aos **feature maps da última camada convolucional** do Xception (7×7×2048). Os gradientes são ponderados por cada canal e combinados para produzir um mapa de calor 7×7, que é redimensionado para 224×224.

**Fórmula:**

```
heatmap[i,j] = ReLU( Σ_k  α_k · A^k[i,j] )

onde:  α_k = média_global(∂score_c / ∂A^k)
       A^k = feature map do canal k
```

> **Nota de implementação:** O pacote `tf-keras-vis` apresentou incompatibilidade com a API Funcional do Keras 3 ao utilizar sub-modelos encapsulados (como o Xception). O Grad-CAM foi implementado manualmente usando `tf.GradientTape`, resultando na mesma abordagem matemática.

### 5.2 LIME (Local Interpretable Model-Agnostic Explanations)

Perturba superpixels da imagem e mede o impacto de cada região na predição do modelo. As regiões com maior influência positiva na classe predita são destacadas.

- Biblioteca: `lime.lime_image.LimeImageExplainer`
- num_samples: 300 (padrão no notebook; artigo usa valores maiores)
- Visualização: bordas dos superpixels sobrepostas à imagem original

### 5.3 Occlusion Sensitivity

Mascara sistematicamente regiões da imagem com um quadrado cinza (valor=0.5) usando uma janela deslizante, e mede a **queda na confiança da predição** para a classe de interesse.

- Tamanho da janela: 32×32 pixels
- Passo (stride): 16 pixels
- Resultado: mapa de sensibilidade 2D normalizado

---

## 6. Resumo Comparativo dos Experimentos

| Experimento | Pipeline | Acurácia Teste (Artigo) |
|-------------|----------|------------------------|
| Exp. 1 | Xception TL + Softmax | 89.24% |
| Exp. 2 | Xception Features + MG-SVM | 89.50% |
| Exp. 3 | Xception + PSO + Subspace KNN | **98.50%** |

O Experimento 3 demonstra que a seleção inteligente de features via PSO, combinada com Subspace KNN, supera significativamente os experimentos anteriores no conjunto ISIC 2018.

---

## 7. Decisões e Adaptações da Reprodução

| Aspecto | Artigo Original | Reprodução |
|---------|----------------|------------|
| Linguagem/Framework | MATLAB 2023b | Python 3.10, TensorFlow 2.21, Keras 3.12 |
| Dimensão features | 1024 | 2048 (Xception com include_top=False) |
| Grad-CAM | tf-keras-vis (implícito) | Implementação manual via `tf.GradientTape` |
| PSO | Implementação MATLAB | `pyswarms.discrete.BinaryPSO` |
| Subspace KNN | MATLAB fitcknn/templateKNN | `sklearn.ensemble.BaggingClassifier` + `KNeighborsClassifier` |
| Augmentação | Estática (pré-gerada) | Estática em memória (mesmos parâmetros) |
| Hardware | Intel i7 10ª gen + RTX 2060 | CPU (sem GPU disponível no ambiente) |

**Nota sobre dimensão de features:** O artigo menciona 1024 features, enquanto o Xception com `include_top=False` produz 2048 após GlobalAveragePooling2D no input 224×224. A diferença pode decorrer de uma versão customizada do Xception usada no MATLAB ou de um pooling adicional. Na reprodução em Python, utiliza-se o valor padrão de 2048.

---

## 8. Estrutura do Notebook

O notebook `reproducao_artigo.ipynb` está organizado nas seguintes seções:

| Célula | Conteúdo |
|--------|----------|
| Instalação | `pip install pyswarms lime tf-keras-vis` |
| Importações | Todas as bibliotecas + configuração de seeds |
| Seção 1 | Carregamento de dados, EDA, visualização de amostras |
| Seção 2 | Funções de augmentação, criação do dataset aumentado |
| Seção 3 | Experimento 1: modelo Xception, treinamento, curvas, métricas |
| Seção 4 | Experimento 2: extrator de features, 12 classificadores ML, tabela comparativa |
| Seção 5 | Experimento 3: PSO binário, Subspace KNN, curva de convergência |
| Seção 6 | XAI: Grad-CAM, LIME, Occlusion Sensitivity com visualizações |
| Seção 7 | Tabela e gráfico final comparando resultados obtidos vs. artigo |

---

## 9. Dependências

```
tensorflow==2.21.0
keras==3.12.1
scikit-learn>=1.7
pyswarms==1.3.0
lime==0.2.0.1
tf-keras-vis==0.8.7   # instalado, mas Grad-CAM implementado manualmente
Pillow
numpy
pandas
matplotlib
seaborn
scikit-image
```

---

## 10. Próxima Etapa — Melhoria Proposta

A proposta do projeto vai além da reprodução: implementar um **Ensemble com Weighted Averaging** combinando CNNs (Xception, EfficientNet) e Vision Transformers (ViT), buscando superar os resultados individuais e o pipeline do artigo base.

As técnicas XAI (Grad-CAM, LIME, Mapas de Atenção) serão aplicadas ao ensemble para validar se as regiões de interesse médico coincidem com as ativações das redes.
