# Experimentos de Ensemble — Skin Cancer Detection

## Contexto

Notebooks de referência:
- **Exploração:** `skin_cancer_TL_ensemble.ipynb` (branch `feat/ensemble`)
- **Pipeline completo:** `skin_cancer_TL_final copy.ipynb`

O objetivo é encontrar a melhor estratégia de ensemble para ser utilizada nas etapas de **PSO** (Exp. 3) e **XAI** (Grad-CAM, LIME, Occlusion Sensitivity) do notebook final.

---

## Opções exploradas

### Opção A — Concatenação dos vetores GAP

Concatena diretamente as saídas do Global Average Pooling de cada backbone.

```
Xception → GAP (2048-d) ─┐
ResNet50 → GAP (2048-d) ──┼── Concatenate → Dense(512) → Dense(256) → Softmax
VGG16    → GAP  (512-d) ─┘
```

**Resultado (Xception + EfficientNetB0, 5 épocas):** val_acc ≈ 79.9%

**Limitação:** vetor de fusão sem nome fixo → índice de camada varia, frágil para extração de features.

---

### Opção B — Fusão ponderada (Weighted Add)

Projeta cada vetor GAP para uma dimensão comum via Dense(512) e soma element-wise.

```
Xception → GAP → Dense(512) ─┐
ResNet50 → GAP → Dense(512) ──┼── Add(512-d) → Dropout → Dense(512) → Dense(256) → Softmax
EfficientNet → GAP → Dense(512)─┘
```

**Resultado (Xception + ResNet50 + VGG, 5 épocas):** val_acc ≈ 83.5%

**Limitação:** os backbones colapsam num único vetor sem identidade — Grad-CAM perde o backbone de origem.

---

### Opção D — Decision-Level Stacking ✅ (escolhida)

**Combinação 2: Xception + EfficientNetB0 + DenseNet121**

Cada backbone tem sua própria cabeça de classificação. O meta-learner recebe os vetores GAP **e** as probabilidades individuais concatenados.

```
                    Imagem (224×224×3)
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
     Xception         EfficientNetB0    DenseNet121
  (frozen, ImageNet)  (frozen, ImageNet) (frozen, ImageNet)
          │                │                │
    GAP → 2048-d     GAP → 1280-d     GAP → 1024-d
          │                │                │
   Dense(2, softmax) Dense(2, softmax) Dense(2, softmax)
   pred_xception     pred_efficientnet  pred_densenet
          │                │                │
          └────── Concatenate (fusion_features) ──────┘
                     4358-d total
                  (2048 + 1280 + 1024 + 2 + 2 + 2)
                           │
                  BatchNormalization
                  Dense(256, relu)
                  Dropout(0.3)
                  Dense(2, softmax)   ← output_layer
```

---

## Por que Xception + EfficientNetB0 + DenseNet121?

| Backbone | Dimensão | Justificativa |
|----------|----------|---------------|
| **Xception** | 2048-d | Backbone do artigo original; depthwise separable convolutions capturam texturas finas (bordas irregulares, variação de cor de lesões) |
| **EfficientNetB0** | 1280-d | Compound scaling (profundidade + largura + resolução); boa generalização com poucos parâmetros; complementar ao Xception |
| **DenseNet121** | 1024-d | Dense connections reutilizam features de todas as camadas; amplamente validado em diagnóstico médico por imagem (base do CheXNet) |

---

## Por que Opção D é melhor para o pipeline PSO + XAI?

### PSO (Experimento 3)
- O vetor `fusion_features` (4358-d) é **nomeado** → extração via `model.get_layer('fusion_features').output`, sem hardcoding de índice de camada
- O PSO pode descobrir que as features de um backbone específico são mais discriminativas (ex.: apenas o DenseNet já é suficiente)
- Nas opções A/B, o índice `model.layers[2].output` quebra quando o número de backbones muda

### Grad-CAM (XAI)
- Cada backbone tem sua própria cabeça (`pred_xception`, etc.) → Grad-CAM independente por backbone
- Produz **3 heatmaps por imagem** para análise comparativa: "o que Xception vê" vs "o que DenseNet vê"
- Implementação: chama cada camada diretamente no `tf.GradientTape` (sem probe models — compatível com Keras 3)

```python
with tf.GradientTape() as tape:
    conv_outputs = backbone_layer(img_tensor, training=False)
    tape.watch(conv_outputs)
    gap_out     = gap_layer(conv_outputs)
    pred_out    = pred_layer(gap_out)          # cabeça individual
    class_score = pred_out[:, class_idx]
```

### LIME / Occlusion Sensitivity
- Ambas são model-agnostic → funcionam sem alteração com qualquer arquitetura

---

## Comparativo de resultados (ensemble notebook, 5 épocas)

| Configuração | Fusão | val_acc |
|---|---|---|
| Xception + ResNet50 + VGG16 | Opção B (add) | ~83.5% |
| Xception + ResNet50 + EfficientNetB0 | Opção B (add) | ~82.2% |
| Xception + EfficientNetB0 | Opção A (concat) | ~79.9% |
| Xception + EfficientNetB0 + DenseNet121 | Opção D (stacking) | a rodar |

---

## Implementação no notebook final

| Seção | Célula | O que muda |
|-------|--------|------------|
| Exp. 1 — Modelo | `1de5d072` | Substituiu Xception+ViT por Stacking Ensemble 3 backbones |
| Exp. 2 — Extrator | `cell-exp2-extract` | `model.layers[2].output` → `model.get_layer('fusion_features').output` |
| XAI — Grad-CAM | `cell-xai-gradcam` | Grad-CAM por backbone com `tf.GradientTape` direto; grade 4×4 (3 backbones + original) |
