# Experimentos e Resultados

**Dataset:** ISIC 2018 — conjunto de teste fixo: 660 imagens (360 benignas, 300 malignas)

---

## 1. Configuração Experimental

| Parâmetro | Valor |
|-----------|-------|
| Framework | TensorFlow 2.21.0 / Keras 3.12.1 / Python 3.11 |
| Hardware | CPU (sem GPU) |
| Otimizador | Adam, lr = 1×10⁻³ |
| Batch size | 32 |
| Épocas máximas | 10 |
| EarlyStopping | patience=5, monitor=val_loss, restore_best_weights |
| ReduceLROnPlateau | factor=0,5, patience=3, lr_min=1×10⁻⁶ |
| Divisão treino/val | 70%/30% estratificado, SEED=42 |
| Backbone | 100% congelado (pesos ImageNet) |

### Augmentação de dados (4×)

| Classe | Antes | Após |
|--------|-------|------|
| Benigno | 1.440 | 5.760 |
| Maligno | 1.197 | 4.788 |
| **Total** | **2.637** | **10.548** |

---

## 2. Modelos Individuais

| Modelo | Acc (%) | Sen (%) | Spe (%) | Pre (%) | F1 (%) | AUC |
|--------|---------|---------|---------|---------|--------|-----|
| Xception TL | 82,27 | 84,33 | 80,56 | 78,33 | 81,22 | 0,9191 |
| EfficientNetV2B0 TL | 87,42 | 84,33 | **90,00** | **87,54** | 85,91 | 0,9474 |
| ViT-B16 TL | **88,03** | **91,33** | 85,28 | 83,79 | **87,40** | **0,9594** |

### Arquiteturas das cabeças de classificação

- **Xception:** GAP → Dense(512, ReLU) → Dense(256, ReLU) → Dense(2, Softmax) — 22.042.410 params (1.180.930 treináveis)
- **EfficientNetV2B0:** Rescaling(255) → backbone → GAP → Dropout(0,2) → Dense(256, ReLU) → Dense(128, ReLU) → Dense(2, Softmax) — 6.280.402 params (361.090 treináveis)
- **ViT-B16:** backbone → Dropout(0,2) → Dense(128, ReLU) → Dense(2, Softmax)

---

## 3. Ensemble (Xception + EfficientNetV2B0 + ViT-B16)

**Método de combinação:** Weighted Averaging com otimização SLSQP no conjunto de validação.

**Pesos otimizados:** w₁ = w₂ = w₃ = 1/3 (convergência para pesos uniformes)

**Acurácia de validação do ensemble:** 91,06%

| Modelo | Acc (%) | Sen (%) | Spe (%) | Pre (%) | F1 (%) | AUC |
|--------|---------|---------|---------|---------|--------|-----|
| **Ensemble** | **88,33** | **89,67** | 87,22 | 85,40 | 87,48 | **0,9609** |

### Ganhos do ensemble sobre modelos individuais

| Métrica | vs Xception | vs EfficientNetV2B0 | vs ViT-B16 |
|---------|-------------|----------------------|------------|
| Acurácia | +6,06 p.p. | +0,91 p.p. | +0,30 p.p. |
| Sensibilidade | +5,34 p.p. | +5,34 p.p. | −1,66 p.p. |
| AUC | +0,0418 | +0,0135 | +0,0015 |

---

## 4. Comparação com o Artigo Base (Shah et al., 2024)

O artigo base conduz três experimentos progressivos sobre o mesmo dataset ISIC 2018:

| Exp | Método | Acc (%) | Sen (%) | Spe (%) | Pre (%) | F1 (%) |
|-----|--------|---------|---------|---------|---------|--------|
| Exp1 | Xception + Softmax (end-to-end) | 89,7 | 93,9 | 84,6 | 85,8 | 89,7 |
| Exp2 | Xception features + MG-SVM | 89,6 | 93,4 | 85,5 | 86,9 | 90,0 |
| Exp3 | Xception + PSO (1024→504) + Subspace KNN | **98,5** | **98,1** | **98,9** | **99,2** | **98,6** |
| **Ensemble (nosso)** | **Xception + EfficientNetV2B0 + ViT-B16** | **88,33** | **89,67** | **87,22** | **85,40** | **87,48** |

### Nosso ensemble vs Exp1 (comparação mais justa — mesmo paradigma end-to-end)

| Métrica | Nosso Ensemble | Exp1 | Diferença |
|---------|---------------|------|-----------|
| Acurácia | 88,33% | 89,7% | −1,4 p.p. |
| Sensibilidade | 89,67% | 93,9% | −4,2 p.p. |
| **Especificidade** | **87,22%** | **84,6%** | **+2,6 p.p. ✓** |
| AUC | 0,9609 | n/r | — |

### Nosso ensemble vs Exp2

| Métrica | Nosso Ensemble | Exp2 | Diferença |
|---------|---------------|------|-----------|
| Acurácia | 88,33% | 89,6% | −1,3 p.p. |
| Sensibilidade | 89,67% | 93,4% | −3,7 p.p. |
| **Especificidade** | **87,22%** | **85,5%** | **+1,7 p.p. ✓** |

### Nosso ensemble vs Exp3

> ⚠️ Comparação **não direta**: Exp3 usa validação cruzada 5-fold + pipeline de duas etapas (CNN → PSO → KNN), enquanto nosso ensemble é end-to-end com conjunto de teste fixo.

| Métrica | Nosso Ensemble | Exp3 | Diferença |
|---------|---------------|------|-----------|
| Acurácia | 88,33% | 98,5% | −10,2 p.p. |
| Sensibilidade | 89,67% | 98,1% | −8,4 p.p. |
| Especificidade | 87,22% | 98,9% | −11,7 p.p. |

---

## 5. Testes Estatísticos

### Intervalos de confiança (95%), n = 660

| Modelo | Acc (%) | IC 95% |
|--------|---------|--------|
| Xception TL | 82,27 | [79,4%; 85,1%] |
| EfficientNetV2B0 TL | 87,42 | [84,8%; 90,1%] |
| ViT-B16 TL | 88,03 | [85,5%; 90,5%] |
| **Ensemble** | **88,33** | **[85,9%; 90,8%]** |

- **Ensemble vs Xception:** ICs **não se sobrepõem** → diferença estatisticamente significativa (α=5%)
- **Ensemble vs ViT-B16:** ICs **se sobrepõem substancialmente** → ganho de 0,30 p.p. não é estatisticamente significativo com n=660

### Teste de McNemar

- Ensemble vs Xception: Δ ≈ 40 predições → χ² compatível com rejeição de H₀
- Ensemble vs ViT-B16: Δ ≈ 2 predições → χ² ≈ 0, sem conclusão estatística
- Para significância com potência 80% (α=5%) entre ensemble e ViT-B16: seriam necessárias ~3.000–5.000 amostras adicionais de teste

---

## 6. XAI — Inteligência Artificial Explicável

Três técnicas aplicadas sobre 4 amostras do conjunto de teste (2 benignas + 2 malignas):

| Técnica | Escopo | Resultado observado |
|---------|--------|---------------------|
| **Grad-CAM** | Backbone Xception (gradiente) | Foco em bordas irregulares e pigmentação heterogênea em malignos; distribuição uniforme em benignos |
| **LIME** | Ensemble (perturbação de superpixels) | Superpixels críticos em áreas de variação de coloração e textura (consistente com regra ABCD) |
| **Occlusion Sensitivity** | Ensemble (mascaramento 32×32, passo 16px) | Região central das lesões malignas é a mais impactante; mascaramento reduz acentuadamente a confiança do modelo |
