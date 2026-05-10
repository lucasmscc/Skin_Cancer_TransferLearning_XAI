# Comparação de Resultados: Ensemble CNN+ViT vs Artigo Base

**Referência base:** Shah et al. (2024) — *Explainable AI-Based Skin Cancer Detection Using CNN, Particle Swarm Optimization and Machine Learning*. J. Imaging, 10, 332.  
**Dataset:** ISIC 2018 — conjunto de teste fixo: 660 imagens (360 benignas, 300 malignas)  
**Data da análise:** 2026-05-10

---

## 1. Nossos Resultados (notebook `ensemble_xception_efficientnet_vit.ipynb`)

### Modelos individuais — 5 épocas, backbone congelado

| Modelo | Acc | Sen | Spe | Pre | F1 | AUC |
|--------|-----|-----|-----|-----|-----|-----|
| Xception TL | 82.27% | 84.33% | 80.56% | 78.33% | 81.22% | 0.9191 |
| EfficientNetV2B0 TL | 86.82% | 83.33% | 89.72% | 87.11% | 85.18% | 0.9512 |
| ViT-B16 TL | 86.82% | 86.67% | 86.94% | 84.69% | 85.67% | 0.9521 |

### Ensemble (média de probabilidades, pesos iguais 1/3)

| Modelo | Acc | Sen | Spe | Pre | F1 | AUC |
|--------|-----|-----|-----|-----|-----|-----|
| **Ensemble Xception + EfficientNetV2B0 + ViT-B16** | **87.88%** | **87.33%** | **88.33%** | **86.18%** | **86.75%** | **0.9586** |

> Pesos otimizados via grid search na validação resultaram em distribuição uniforme (33.3% cada), indicando que os modelos contribuem de forma equivalente.

---

## 2. Resultados do Artigo Base (ISIC 2018, conjunto de teste)

O artigo conduz três experimentos progressivos sobre o mesmo dataset:

| Exp | Método | Acc | Sen | Spe | Pre | F1 |
|-----|--------|-----|-----|-----|-----|-----|
| 1 | Xception + Softmax (end-to-end) | 89.7% | 93.9% | 84.6% | 85.8% | 89.7% |
| 2 | Xception features + MG-SVM | 89.6% | 93.4% | 85.5% | 86.9% | 90.0% |
| **3** | **Xception + PSO (1024→504 features) + Subspace KNN** | **98.5%** | **98.1%** | **98.9%** | **99.2%** | **98.6%** |

> O Exp3 usa 5-fold cross-validation com classificadores ML clássicos sobre features extraídas da CNN — **pipeline fundamentalmente diferente** do nosso (end-to-end neural).

---

## 3. Análise Comparativa

### 3.1 Xception isolado: nós vs artigo

| Métrica | Nosso Xception | Artigo Exp1 | Diferença |
|---------|---------------|-------------|-----------|
| Acurácia | 82.27% | 89.7% | **−7.4 p.p.** |
| Sensibilidade | 84.33% | 93.9% | −9.6 p.p. |
| Especificidade | 80.56% | 84.6% | −4.0 p.p. |

**Causas identificadas do gap:**
- **Épocas:** nosso treino usa 5 épocas; artigo usa até 30 épocas com parada antecipada
- **Cabeça de classificação:** artigo usa GAP → Dense(1024) → Dropout(0.5) → Dense(2); nós usamos GAP → Dense(512) → Dense(256) → Dense(2) (sem dropout nas densas)
- **Learning rate:** ambos usam Adam com LR inicial similar (`1×10⁻³` nós vs `1×10⁻⁴` artigo), mas o artigo parte de LR menor, favorecendo convergência mais suave em mais épocas

### 3.2 Ensemble vs Artigo Exp1 (comparação mais justa)

| Métrica | Nosso Ensemble | Artigo Exp1 | Diferença |
|---------|---------------|-------------|-----------|
| Acurácia | 87.88% | 89.7% | −1.8 p.p. |
| Sensibilidade | 87.33% | 93.9% | **−6.6 p.p.** |
| Especificidade | **88.33%** | 84.6% | **+3.7 p.p.** |
| AUC | 0.9586 | ~0.95* | +0.009 |

> *AUC do Exp1 do artigo não é reportado explicitamente; estimado com base nas curvas ROC apresentadas.

**Pontos positivos do nosso ensemble:**
- Especificidade superior (88.33% vs 84.6%): menos falsos positivos — diagnósticos benignos mais precisos
- AUC ligeiramente maior: melhor discriminação global entre classes
- Arquitetura mais rica: combina receptive fields locais (CNNs) com atenção global (ViT)

**Ponto crítico:**
- Sensibilidade de 87.33% vs 93.9% do artigo: em contexto clínico, sensibilidade baixa significa **mais malignos não detectados (falsos negativos)** — esse é o gap mais preocupante

### 3.3 Ensemble vs Artigo Exp3 (pipeline híbrido)

| Métrica | Nosso Ensemble | Artigo Exp3 | Diferença |
|---------|---------------|-------------|-----------|
| Acurácia | 87.88% | 98.5% | −10.6 p.p. |
| Sensibilidade | 87.33% | 98.1% | −10.8 p.p. |
| Especificidade | 88.33% | 98.9% | −10.6 p.p. |

**Importante:** esta comparação não é direta. O Exp3 do artigo é um pipeline de duas etapas (extração de features CNN → seleção via PSO → classificador ML com CV), enquanto nosso ensemble é completamente end-to-end. Os dois abordam o problema por caminhos metodologicamente distintos.

---

## 4. Diferencial da Nossa Abordagem

| Aspecto | Artigo Base | Nossa Abordagem |
|---------|-------------|-----------------|
| Arquitetura | Apenas Xception | Xception + EfficientNetV2B0 + ViT-B16 |
| Paradigma | CNN + ML clássico (Exp3) | Ensemble neural end-to-end |
| Diversidade de features | Apenas features CNN | CNN local + CNN moderna + Transformer global |
| Interpretabilidade XAI | Grad-CAM, LIME, Occlusion Sensitivity | Grad-CAM, LIME, Occlusion Sensitivity (aplicado ao ensemble) |
| Implementação | MATLAB 2023b | Python / TensorFlow / Keras |
| Épocas por modelo | 30 | 5 |

---

## 5. Melhorias Planejadas para Próxima Execução

Com base na análise do gap em relação ao artigo, as seguintes melhorias serão implementadas:

### 5.1 Treinamento mais longo
- Aumentar de **5 para 20–30 épocas** por modelo
- Manter `EarlyStopping` com `patience=5` e `restore_best_weights=True`
- Reduzir LR inicial para `1×10⁻⁴` (alinhado ao artigo) para convergência mais suave

### 5.2 Arquitetura da cabeça de classificação
- Adicionar **Dropout(0.5)** após a primeira Dense em todos os modelos (como o artigo faz no Xception)
- Avaliar Dense(1024) no Xception para replicar exatamente a configuração do artigo (Exp1)

### 5.3 Fine-tuning parcial (descongelamento seletivo)
- Após convergência das camadas de topo (5 épocas), **descongelar as últimas N camadas** do backbone e continuar treinamento com LR reduzido (`1×10⁻⁵`)
- Xception: últimos 2 blocos separáveis
- EfficientNetV2B0: últimos 3 blocos MBConv
- ViT: últimas 2 camadas de atenção

### 5.4 Learning rate scheduling
- Usar `ReduceLROnPlateau` com `factor=0.5`, `patience=3` além do LR constante atual
- Avaliar `CosineDecay` como alternativa

### 5.5 Otimização de pesos do ensemble
- Testar busca de pesos por **Optuna** (otimização bayesiana) na validação em vez de grid search uniforme
- Avaliar stacking com meta-learner simples (Logistic Regression sobre as probabilidades dos modelos)

### 5.6 Métricas de foco
- Priorizar **sensibilidade** como métrica principal de seleção de modelo (contexto médico: minimizar falsos negativos)
- Avaliar threshold customizado (ex.: 0.4 em vez de 0.5) para aumentar recall de malignos

---

## 6. Referências de Comparação (outros trabalhos citados no artigo)

| Trabalho | Dataset | Pipeline | Acc | Sen | Spe |
|----------|---------|----------|-----|-----|-----|
| Al-Rasheed et al. [39] | ISIC 2019 | Ensemble + CGAN | 92–93.5% | — | — |
| Akilandasowmya et al. [42] | ISIC 2019 | SCSO + ResNet50 + EHS | 92% | 93.9% | 85.5% |
| Saha et al. [49] | ISIC 2019 | ViT + MobileNet + segmentação | 91.2% | ~93% | ~90% |
| Ahmad et al. [48] | HAM10000 | ViT + EfficientNet | ~90% | ~92% | ~89% |
| **Nosso Ensemble** | **ISIC 2018** | **Xception + EffNet + ViT** | **87.88%** | **87.33%** | **88.33%** |
| **Artigo Base (Exp3)** | **ISIC 2018** | **Xception + PSO + KNN** | **98.5%** | **98.1%** | **98.9%** |
