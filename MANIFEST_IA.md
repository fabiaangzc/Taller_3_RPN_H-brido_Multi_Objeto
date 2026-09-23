# MANIFEST_IA — Taller 3: RPN híbrido multi-objeto sobre FundidoraPC

Dataset: Aquarium Combined (Roboflow), 7 clases + fondo, imágenes a 224×224. Modelo final: época 30. Recall de propuestas en validación = 0.742 (IoU 0.5, top-N 1200); mAP@0.5 = 0.1195.

## 1. ¿Por qué las propuestas necesitan `.detach()` antes de ROI Align, y qué pasaría si se omite?

Las propuestas salen de `decode_boxes(anchors, bbox_deltas)`, así que sus coordenadas dependen de los pesos de la cabeza de regresión del RPN. ROI Align usa esas coordenadas para decidir **dónde recortar** el mapa de features. Ese recorte no es una operación diferenciable útil respecto a la posición de la caja. Si no se desconectan, la `detection_loss` intentaría mover las cajas del RPN a través de ROI Align. Según la consigna, eso hace que el entrenamiento falle en silencio o dé un error de grafo, y además mezcla dos señales que deben estar separadas:

- **Dónde está el objeto** se aprende solo con `rpn_loss` (la SmoothL1 sobre las anclas positivas).
- **Qué hay en la región** se aprende con `detection_loss` (CrossEntropy y el refinamiento de la segunda etapa).

En nuestro código, `.detach()` se aplica a las propuestas en `detection_stage_loss` (entrenamiento) y en `predict_batch` (inferencia), y un `assert not p.requires_grad` lo verifica antes de cada llamada a `roi_align`. Lo que se corta es solo el gradiente hacia las coordenadas. El `feature_map` sí recibe gradiente de la `detection_loss` a través de ROI Align, así que el backbone compartido aprende de las dos pérdidas, como en el entrenamiento conjunto `loss_total = rpn_loss + detection_loss`. En el Módulo 6 aplicamos la misma regla: la caja analizada por Grad-CAM se mantiene fija y desconectada.

## 2. ¿Por qué el mapa de objectness y el Smooth Grad-CAM no coinciden exactamente? (sobre nuestros resultados)

En nuestras 5 imágenes de validación **no coinciden**:

- **El objectness es un mapa focalizado y geométrico.** Ocupa cerca del 20 % del mapa y se ubica sobre regiones con forma de objeto: frailecillos, rayas, tiburones y el grupo de medusas. El RPN aprendió bien *dónde hay algo* (recall de 0.742).
- **El Smooth Grad-CAM pedido casi no tiene contenido.** Sale vacío en 2 de 5 imágenes; en las otras solo quedan puntos en las esquinas (0.6 % del mapa), que atribuimos al *padding*. El diagnóstico lo explica: el logit de la clase analizada es negativo en las 5 imágenes (de −0.67 a −2.24) y el CAM antes del ReLU es negativo en casi todo el mapa, así que el ReLU lo anula. Además, en 2 de las 5 imágenes la detección analizada tiene la clase equivocada (`jellyfish` sobre un frailecillo y sobre un tiburón). Que las propias features del modelo voten en contra de esa etiqueta es coherente, no un error.
- **El Grad-CAM contrafactual (`ReLU(-CAM)`) es muy difuso** (73 % del mapa) y se parece poco al objectness (correlación promedio 0.16, entre −0.13 y 0.30). Hay que leerlo bien: la cabeza de detección solo recibe la región 7×7 de la caja, así que no "mira" fuera de ella. Grad-CAM calcula el peso de cada canal con el gradiente de dentro de la caja y luego proyecta esos pesos sobre todo el mapa. Un contrafactual difuso significa que los canales que votan en contra de la clase son canales activos en casi toda la escena (roca, arena, agua), es decir, canales de "fondo".

Nuestra interpretación: los dos mapas responden a preguntas distintas y con distinta calidad de aprendizaje. El objectness responde a bordes y formas, que se conservan aunque el objeto sea pequeño. El Grad-CAM depende de que la cabeza reconozca la especie a partir de una región 7×7 recortada de un mapa de 14×14. En nuestro dataset, el 41 % de los objetos mide menos de 16 px tras el resize (menos de una celda), así que la región contiene sobre todo features de fondo, y la cabeza termina con evidencia neta en contra de la clase (scores de 0.13 a 0.29). Esto es coherente con la brecha entre un recall alto (0.742) y una mAP baja (0.1195). Es un resultado de solo 5 imágenes y una detección por imagen, así que lo tomamos como indicio, no como conclusión general.

## 3. Una decisión de diseño del grupo: anchors de 3 escalas (16, 32, 64) × 3 proporciones (0.5, 1, 2), k = 9

Antes de entrenar medimos el tamaño de las cajas reales ya redimensionadas: p25 = 11.9 px, mediana = 18.8 px, p75 = 30 px. Las escalas típicas (32, 64, 128) quedaban grandes para este dataset. Comparamos tres configuraciones con la misma asignación por IoU sobre 40 imágenes de train. La mejor IoU promedio entre cada caja real y su mejor ancla fue:

| Escalas | IoU promedio |
|---|---:|
| (16, 32, 64) | **0.485** |
| (24, 48, 96) | 0.454 |
| (32, 64, 128) | 0.408 |

Elegimos (16, 32, 64) con 9 anclas por celda, la recomendación de la consigna, en lugar de reducir a 6. Probar escalas más pequeñas, como (8, 16, 32), no mejoró la cobertura, porque con un stride de 16 una ancla de 8 px casi nunca queda centrada sobre el objeto. Por eso, cuando el recall quedó bajo, no cambiamos los anchors. Primero corregimos el muestreo, como sugiere la consigna: la muestra supuestamente balanceada dejaba en la práctica 1 positiva por cada 84 negativas, y al pasarla a un 1:1 real el recall subió de 0.45 a 0.55. Con el entrenamiento conjunto completo terminó en 0.742.
