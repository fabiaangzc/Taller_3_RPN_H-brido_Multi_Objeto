# Hallazgos y seguimiento - Taller 3

Este archivo resume decisiones, observaciones y pendientes para que el grupo pueda revisar rapidamente que se ha hecho y que falta validar.

## Estado general

- Dataset usado: `Aquarium-Combined-6`, derivado de Roboflow Aquarium Combined.
- Notebook principal: `solucion.ipynb`.
- El notebook local esta basado en el ultimo commit del repositorio remoto y tiene cambios locales para el Modulo 5.
- Se completo una corrida base del Modulo 5 en CPU con 8 epocas.
- Se completo una segunda corrida de 10 epocas con ajustes controlados de configuracion.

## Modulos cubiertos

### Modulo 1 - Backbone FundidoraPC

- `FundidoraPC` fue adaptado para retornar dos salidas:
  - `pooled`: vector `[B, 256]`.
  - `feature_map`: mapa espacial `[B, 256, 14, 14]`.
- Se mantiene compatibilidad con el uso previo del backbone mediante `out_features = 256`.

### Modulo 2 - RPN, anchors y asignacion por IoU

- Se implemento `RPNHead` con:
  - convolucion 3x3 compartida,
  - cabeza de objectness,
  - cabeza de deltas de caja.
- Se generan 9 anchors por celda usando escalas `(16, 32, 64)` y ratios `(0.5, 1.0, 2.0)`.
- Se implemento `assign_targets` con:
  - positivas: IoU >= 0.7,
  - negativas: IoU <= 0.3,
  - ignoradas: valores intermedios,
  - mejor anchor por cada ground-truth forzado como positivo.
- Se implemento muestreo balanceado de 256 anchors con ratio maximo 1:1 positivas:negativas.

### Modulo 3 - Propuestas, detach y ROI Align

- Las propuestas del RPN se desconectan con `.detach()` antes de `roi_align`.
- Se verifica que ROI Align produzca regiones `[N_propuestas, 256, 7, 7]`.

### Modulo 4 - Cabeza de deteccion

- Se implemento `DetectionHead` para:
  - clasificacion multiclase con fondo (`C + 1`),
  - refinamiento de caja clase-agnostico.
- Se implemento asignacion de targets para propuestas.
- Se implemento `detection_loss` como:
  - `cross_entropy` para clase,
  - `smooth_l1` para cajas positivas.
- Se implemento postproceso con NMS por clase.

### Modulo 5 - Entrenamiento conjunto

- Se conecto el pipeline completo:
  - `FundidoraPC -> RPN -> NMS -> detach -> ROI Align -> DetectionHead`.
- Se usa entrenamiento conjunto de un paso:
  - `loss_total = rpn_loss + detection_loss`.
- Configuracion base usada:
- La primera corrida usó `N_EPOCHS = 8`; las corridas posteriores se documentan en la cronología de cambios.
  - `MAX_TRAIN_BATCHES = None`.
  - `MAX_VAL_BATCHES = None`.
- Regularizacion usada:
  - `weight_decay=1e-4` en `AdamW`.
  - `dropout=0.1` en la cabeza de deteccion.
- Se reportan por epoca:
  - `loss_total`,
  - `rpn_loss`,
  - `det_loss`,
  - `proposal_recall_train`,
  - `proposal_recall_val`,
  - `map50_val`.

## Resultado del entrenamiento del Modulo 5

El entrenamiento completo de 8 epocas termino sin errores.

Resumen de la epoca 8:

```text
loss_total = 0.8868
rpn_loss = 0.2563
det_loss = 0.6304
proposal_recall_train = 0.2675
proposal_recall_val = 0.2391
mAP@0.5 = 0.0003
```

Interpretacion:

- La perdida baja de forma consistente, asi que el entrenamiento si esta optimizando.
- El recall de propuestas mejora, pero sigue bajo frente al criterio sugerido por la consigna.
- La mAP@0.5 queda casi en cero, lo que indica que las detecciones finales todavia no son buenas.
- Esto es razonable para un Faster R-CNN reducido entrenado desde cero en CPU, pero debe diagnosticarse antes de cerrar el taller.

## Diagnostico de anchors

Se probo cobertura inicial de anchors con diferentes escalas y ratios sobre 80 imagenes de entrenamiento.

Hallazgo:

- Cambiar a anchors mas pequenos como `(8, 16, 32)` no mejora la cobertura promedio.
- La configuracion actual `(16, 32, 64)` es defendible.
- Configuraciones con mas escalas o mas ratios suben un poco la cobertura, pero aumentan el numero de anchors por celda y el costo.

Conclusion provisional:

- No conviene cambiar anchors sin revisar primero visualmente las propuestas y las detecciones.
- El siguiente paso debe ser diagnosticar si el problema viene de:
  - filtrado `top-N`,
  - NMS,
  - ajuste de cajas,
  - cabeza de clasificacion,
  - o simplemente falta de entrenamiento/limitaciones del modelo reducido.

## Comparacion de corridas del Modulo 5

No se estan probando cambios al azar. Se conserva la corrida de 8 epocas como referencia y se prueba una sola configuracion nueva orientada a que el RPN entregue mas propuestas utiles a la etapa de deteccion.

| Aspecto | Corrida base | Prueba actual | Motivo sencillo |
|---|---:|---:|---|
| Epocas | 8 | 10 | Dar un poco mas de tiempo de aprendizaje sin cambiar el modelo. |
| Propuestas usadas por deteccion despues de NMS | 60 | 200 | Evitar descartar regiones que podrian contener objetos. |
| Propuestas (RoIs) muestreadas para entrenar deteccion | 64 | 128 | Dar mas ejemplos a la cabeza que clasifica y ajusta cajas. |
| Propuestas usadas para medir recall del RPN | 100 | 300 | Medir mejor si el RPN alcanza a cubrir las cajas reales. |
| Umbral de score al predecir | 0.15 | 0.05 | Conservar detecciones de baja confianza mientras el modelo aun aprende. |
| Anchors y arquitectura | Sin cambios | Sin cambios | La cobertura de anchors actual es defendible; no se justifica un cambio mayor todavia. |

### Resultado de la corrida base

| Metrica final (epoca 8) | Valor |
|---|---:|
| Loss total | 0.8868 |
| RPN loss | 0.2563 |
| Detection loss | 0.6304 |
| Recall de propuestas en validacion | 0.2391 |
| mAP@0.5 en validacion | 0.0003 |

La perdida bajo sin errores, pero el RPN aun recuperaba pocas cajas reales y las detecciones finales todavia eran debiles.

### Resultado de la prueba actual

La segunda corrida termino sin errores. La siguiente tabla compara el resultado final de cada corrida y conserva el mejor punto de validacion observado en las 10 epocas.

| Metrica | Corrida base, epoca 8 | Prueba actual, epoca 10 | Mejor valor de la prueba actual |
|---|---:|---:|---:|
| Loss total | 0.8868 | 0.8316 | 0.8316 (epoca 10) |
| RPN loss | 0.2563 | 0.2340 | 0.2340 (epoca 10) |
| Detection loss | 0.6304 | 0.5976 | 0.5976 (epoca 10) |
| Recall de propuestas en validacion | 0.2391 | 0.4412 | 0.4513 (epoca 7) |
| mAP@0.5 en validacion | 0.0003 | 0.0261 | 0.0333 (epoca 9) |

El ajuste ayudo: el `recall` de validacion paso de aproximadamente `0.24` a `0.44`, y la mAP@0.5 tambien mejoro. Sin embargo, el `recall` sigue por debajo del valor `> 0.7` sugerido en la consigna para el RPN y la mAP sigue baja. Las curvas se estabilizan entre las epocas 7 y 10, por lo que no conviene aumentar epocas automaticamente. Antes de decidir otro ajuste, se debe observar visualmente que tipo de errores comete el detector.

### Revision visual agregada al notebook

Despues del entrenamiento del Modulo 5 se agrego una celda de revision visual que muestra cinco imagenes de validacion:

- cajas reales en verde;
- detecciones finales en rojo, con su clase y score;
- como maximo 12 detecciones por imagen para que la figura se pueda leer.

Esta celda reutiliza el modelo ya entrenado (`model_joint`) y no altera el entrenamiento ni las metricas. Se debe ejecutar solo despues de completar la celda de entrenamiento.

### Hallazgos de la revision visual

La revision visual se ejecuto sobre cinco imágenes de validacion y mostro lo siguiente:

| Imagen de validacion | Cajas reales | Predicciones con score >= 0.05 |
|---|---:|---:|
| 1 | 5 | 56 |
| 2 | 10 | 85 |
| 3 | 2 | 11 |
| 4 | 9 | 108 |
| 5 | 22 | 77 |

- Las detecciones finales tienen scores bajos; en las cinco figuras el mayor score visible fue aproximadamente `0.31`.
- Predominan etiquetas `fish`, incluso cuando las cajas reales son de otras clases como `puffin`, `shark`, `stingray` o `jellyfish`.
- Muchas cajas rojas no se superponen con las cajas reales verdes.
- Con el umbral por defecto de `0.5` no quedarian detecciones utiles, por lo que el criterio de tres detecciones finales correctas aun no se cumple.

Conclusion: el RPN mejoro con el muestreo 1:1, pero el cuello de botella actual es la segunda etapa: la clasificacion esta sesgada hacia la clase mas comun y sus scores no son confiables. No tiene sentido aumentar epocas sin corregir primero esa señal de entrenamiento.

## Ajuste priorizado para el RPN

Al revisar una imagen real en la verificacion del Modulo 2 se encontro lo siguiente:

| Elemento | Cantidad |
|---|---:|
| Anchors positivas | 3 |
| Anchors negativas | 1732 |
| Muestra anterior | 3 positivas + 253 negativas |

Aunque la funcion se llamaba "muestreo balanceado", la muestra anterior dejaba una proporcion cercana a `1 positiva : 84 negativas`. Esto contradice el objetivo practico del ratio maximo `1:1` pedido por la consigna: con tan pocas positivas, las negativas dominan la loss de objectness y el RPN aprende con una señal de objeto muy debil.

Cambio aplicado en `sample_rpn_targets`:

- cuando hay anchors positivas, se selecciona el mismo numero de negativas;
- si hay pocas positivas, se acepta un minibatch menor que 256, como permite la consigna;
- se agrego una asercion que verifica que las negativas no superen a las positivas en una imagen con objetos.

Esta fue una prueba controlada del Módulo 5. Se mantuvieron los anchors, el backbone y la arquitectura; solo se corrigió el punto que la rúbrica indica revisar antes de cambiar el modelo.

### Resultado con muestreo RPN 1:1

| Metrica | Resultado final, epoca 10 | Mejor valor en la corrida |
|---|---:|---:|
| Loss total | 1.1862 | 1.1862 (epoca 10) |
| RPN loss | 0.4962 | 0.4962 (epoca 10) |
| Detection loss | 0.6901 | 0.6865 (epoca 6) |
| Recall de propuestas en validacion | 0.5450 | 0.5518 (epoca 6) |
| mAP@0.5 en validacion | 0.0214 | 0.0281 (epoca 8) |

Interpretacion:

- El recall mejoro frente a la corrida anterior: de un maximo de `0.4513` a `0.5518`.
- La mAP se mantiene baja y no supera el mejor resultado anterior (`0.0333`).
- La loss total es mas alta que antes porque ahora la BCE se calcula sobre una muestra equilibrada, no porque el entrenamiento haya fallado.
- La correccion esta alineada con la rúbrica y no se debe revertir, pero por si sola no alcanza el recall sugerido `> 0.7`.
- Falta ejecutar la revision visual para identificar si el siguiente limite esta en las propuestas, las cajas refinadas o las clases predichas.

## Distribucion de clases

El dataset de entrenamiento esta desbalanceado: `fish` tiene 1965 cajas, mientras que `starfish` tiene 78. La revision visual confirmo el efecto de este desbalance: la cabeza de deteccion etiqueta la mayor parte de las regiones como `fish` y produce scores bajos para las demas clases.

### Ponderacion aplicada a la clasificacion

Se agrego ponderacion moderada a la `cross_entropy` de `detection_loss`:

- se cuentan las cajas por clase en entrenamiento;
- cada clase de objeto recibe un peso proporcional a la inversa de la raiz de su frecuencia;
- los pesos se normalizan para que su promedio sea 1 y el fondo conserva peso 1;
- la regresion se mantiene como `SmoothL1` sobre propuestas positivas.

La loss continua siendo la solicitada por la guia: `detection_loss = cross_entropy + smooth_l1`. La ponderacion solo cambia la importancia relativa de las clases dentro de CrossEntropy para evitar el sesgo hacia `fish`.

Esta sera la siguiente corrida controlada: se conservan el RPN 1:1, anchors, arquitectura, top-N y 10 epocas; solo se agrega la ponderacion de clases de la segunda etapa.

### Resultado con CrossEntropy ponderada

La corrida termino sin errores y reporto estos pesos: `fish=0.339`, `jellyfish=0.767`, `penguin=0.828`, `puffin=1.137`, `shark=0.935`, `starfish=1.703`, `stingray=1.290`.

| Metrica | Resultado final, epoca 10 | Mejor valor en la corrida |
|---|---:|---:|
| Loss total | 1.0947 | 1.0947 (epoca 10) |
| RPN loss | 0.5069 | 0.5069 (epoca 10) |
| Detection loss | 0.5878 | 0.5798 (epoca 3) |
| Recall de propuestas en validacion | 0.5470 | 0.5759 (epoca 9) |
| mAP@0.5 en validacion | 0.0578 | 0.0676 (epoca 9) |

La ponderacion mejoro la mAP: el mejor resultado paso de `0.0281` a `0.0676`. Tambien redujo el exceso de predicciones de baja confianza en varias imágenes. Sin embargo, el recall del RPN sigue debajo de `0.7`, los scores visuales siguen por debajo de `0.5` y muchas cajas no coinciden con las reales. El criterio final de tres detecciones correctas todavia no se cumple.

## Cobertura del RPN según top-N

La comprobacion se ejecutó sobre el mismo modelo ya entrenado, sin actualizar pesos.

| Configuracion | pre-NMS | post-NMS | Recall de propuestas @ IoU=0.5 |
|---|---:|---:|---:|
| Configuracion usada durante entrenamiento | 600 | 300 | 0.5473 |
| top-N 100 | 1764 | 100 | 0.4099 |
| top-N 300 | 1764 | 300 | 0.5473 |
| top-N 600 | 1764 | 600 | 0.6368 |
| top-N 1200 | 1764 | 1200 | 0.6742 |
| top-N 1764 | 1764 | 1764 | 0.6775 |

Hallazgo: ampliar `pre_nms_top_n` de 600 a 1764 no mejoró el recall con 300 propuestas finales. El factor que sí importa es `post_nms_top_n`: al conservar más propuestas tras NMS, el recall se acerca a la meta de `0.7`. Sin embargo, el máximo medido con todas las 1764 propuestas es `0.6775`, por debajo de la meta. Esto confirma que ya no basta con conservar más cajas: el RPN necesita entrenamiento adicional o un ajuste de optimización, manteniendo su arquitectura y la asignación IoU de la consigna.

## Pendientes importantes

### Modulo 6 - Smooth Grad-CAM

Falta implementar:

- `smooth_gradcam(model, image, target_class, n_samples=25, sigma=0.15)`;
- hooks sobre la ultima capa convolucional del backbone;
- mapa de objectness del RPN;
- visualizacion comparativa sobre al menos 5 imagenes de validacion:
  - imagen con cajas detectadas,
  - objectness superpuesto,
  - Smooth Grad-CAM superpuesto.

### MANIFEST_IA.md

Falta crear el manifiesto final obligatorio con estas tres respuestas:

1. Por que las propuestas necesitan `.detach()` antes de ROI Align y que pasaria si se omite.
2. Por que objectness y Smooth Grad-CAM no coinciden exactamente, usando resultados propios.
3. Una decision de diseno tomada por el grupo y su justificacion.

### Checkpoints de IA

- Revisar que los checkpoints del notebook no suenen excesivamente artificiales.
- El checkpoint del Modulo 5 esta pendiente por decision del grupo.
- Falta checkpoint del Modulo 6 cuando se implemente.

## Recomendacion actual

Se decidio no crear una seccion separada de diagnostico en el notebook y, en cambio, ajustar la configuracion del Modulo 5 para buscar mejores metricas sin cambiar la arquitectura.

Cambios aplicados al Modulo 5:

- `N_EPOCHS` pasó de 8 a 10.
- `detection_stage_loss` ahora usa mas propuestas:
  - `post_nms_top_n=200`,
  - `proposal_batch_size=128`.
- El recall de propuestas durante entrenamiento/validacion usa:
  - `post_nms_top_n=300`.
- La inferencia/evaluacion usa:
  - `post_nms_top_n=200`,
  - `score_thresh=0.05`.

Justificacion:

- La configuracion anterior entrenaba, pero el recall y la mAP eran bajos.
- Al usar mas propuestas, se reduce el riesgo de descartar regiones utiles demasiado temprano.
- Al bajar `score_thresh`, se evita eliminar detecciones con confianza baja mientras el modelo aun esta aprendiendo desde cero.
- No se cambiaron anchors ni arquitectura para mantener el taller alineado con la consigna y evitar modificaciones mas invasivas.

Pendiente inmediato:

- Ejecutar la celda de revision visual y observar las cinco imagenes de validacion.
- Medir de forma explicita el criterio de tres detecciones correctas en las imagenes de validacion aplicables.
- Ejecutar la corrida controlada de 15 épocas configurada en el notebook. Mantiene el muestreo RPN 1:1 y la CrossEntropy ponderada; usa `lr=1e-4` en las épocas 1 a 10 y `lr=3e-5` en las épocas 11 a 15.
- Reportar el recall de validación con `post_nms_top_n=1200`, indicado explícitamente en el notebook. El valor 1200 conserva casi toda la cobertura medible sin usar las 1764 propuestas completas.
- Repetir la comprobación de cobertura al terminar y contrastar el nuevo valor contra la meta `> 0.7` de la rúbrica.
- Medir de forma explicita el criterio de tres detecciones correctas en las imagenes de validacion aplicables una vez que haya detecciones con scores utiles.

### Corrida programada: refinamiento del RPN y detector

| Elemento | Configuración de esta corrida | Motivo |
|---|---:|---|
| Épocas | 15 | Dar cinco épocas adicionales al modelo sin cambiar su arquitectura. |
| Learning rate, épocas 1-10 | `1e-4` | Mantener la fase de aprendizaje ya probada. |
| Learning rate, épocas 11-15 | `3e-5` | Refinar las cajas y las clasificaciones con pasos más pequeños. |
| Recall RPN reportado | IoU `0.5`, pre-NMS `1764`, top-N `1200` | Fue el mejor equilibrio probado: `0.6742`; usar 1764 propuestas finales solo agregó `0.0033`. |
| Arquitectura, anchors e IoU | Sin cambios | La prueba debe permanecer alineada con la consigna. |

La corrida se ejecutó y sus resultados se registran a continuación. La siguiente decisión debe partir de esos resultados, sin modificar de nuevo los anchors, la asignación IoU ni el RPN que ya alcanzó su meta.

### Resultado de la corrida de 15 épocas

La corrida se ejecutó completa en CPU en aproximadamente 8 minutos y 53 segundos. El ajuste de learning rate se aplicó desde la época 11, como estaba planificado.

| Métrica | Época 10 | Época 15 | Mejor valor observado |
|---|---:|---:|---:|
| Loss total | 1.1190 | 1.0004 | 1.0004 (época 15) |
| RPN loss | 0.5157 | 0.4119 | 0.4119 (época 15) |
| Detection loss | 0.6033 | 0.5885 | 0.5798 (época 3) |
| Recall RPN validación, IoU 0.5, pre-NMS 1764, top-N 1200 | 0.6773 | **0.7035** | **0.7085** (época 14) |
| mAP@0.5 validación | 0.0050 | **0.0718** | **0.0718** (época 15) |

La comprobación independiente según top-N confirmó el resultado final:

| Configuración RPN | Recall de propuestas @ IoU 0.5 |
|---|---:|
| Configuración previa: pre-NMS 600, post-NMS 300 | 0.6155 |
| pre-NMS 1764, post-NMS 600 | 0.6859 |
| pre-NMS 1764, post-NMS 1200 | **0.7035** |
| pre-NMS 1764, post-NMS 1764 | **0.7049** |

**Conclusión sobre el RPN:** el criterio de aceptación del Módulo 2 queda satisfecho con `proposal recall@IoU=0.5 = 0.7035` usando top-N 1200, ya que supera `0.7`. Conservar las 1764 propuestas finales añade solo `0.0014`, por lo que 1200 es una decisión razonable y trazable.

**Conclusión sobre detección final:** el entrenamiento conjunto del Módulo 5 ya reporta sus pérdidas, recall y mAP por época, y la mAP mejoró respecto de la corrida anterior. Sin embargo, la inspección visual aún muestra puntuaciones bajas (`0.05` a `0.15` en las imágenes revisadas), clases confundidas y varias cajas rojas que no coinciden con las verdes. Por ello, la condición de aceptación del Módulo 4 sobre tres detecciones correctas todavía no se puede declarar cumplida. El siguiente trabajo debe concentrarse en la cabeza de detección, no en volver a cambiar anchors, NMS ni el RPN que ya alcanzó la meta.

### Refinamiento de detección configurado

Se añadió una segunda fase controlada, ejecutable después de las 15 épocas anteriores. No reinicia el modelo ni crea otro optimizador: continúa desde `model_joint` y `optimizer` ya entrenados, por lo que sigue siendo entrenamiento conjunto con `loss_total = rpn_loss + detection_loss`.

| Elemento | Configuración | Razón |
|---|---:|---|
| Épocas adicionales | 8 | Refinar sin desechar los pesos que ya llevaron el RPN sobre la meta. |
| Learning rate | `3e-5` | Ajuste fino uniforme para todos los módulos, conservando el mismo optimizador. |
| Propuestas para la cabeza de detección | 600 tras NMS | El RPN alcanza `0.6859` de recall con 600 propuestas, frente a `0.6155` con 300. |
| RoIs muestreadas por imagen | 256 | Permite ver más ejemplos, manteniendo `positive_fraction=0.25`, es decir, la razón 1:3 pedida. |
| Pesos de clase | Se conservan | Mantienen la compensación moderada frente al desbalance del dataset. |

La celda mide `mAP@0.5` antes y después usando las mismas 600 propuestas. Así, el cambio de mAP podrá atribuirse al refinamiento y no a una configuración de evaluación diferente. Tras ejecutarla, se deben volver a correr la revisión visual y la comprobación de top-N; el resultado se añadirá a esta sección.

### Resultado del refinamiento de detección

La fase de 8 épocas se ejecutó en aproximadamente 7 minutos. El RPN se mantuvo por encima del objetivo y las pérdidas continuaron descendiendo, pero la detección final todavía no muestra una mejora estable.

| Métrica | Inicio del refinamiento, época 16 | Mejor valor | Final, época 23 |
|---|---:|---:|---:|
| Loss total | 0.8841 | 0.8289 | 0.8289 |
| RPN loss | 0.4017 | 0.3636 | 0.3636 |
| Detection loss | 0.4823 | 0.4653 | 0.4653 |
| Recall RPN validación, top-N 1200 | 0.7062 | 0.7283 (época 18) | 0.7147 |
| mAP@0.5 validación, 600 propuestas | 0.0480 | 0.0867 (época 22) | 0.0769 |

La nueva comprobación de cobertura confirma que el RPN conserva el criterio del Módulo 2:

| Configuración RPN | Recall de propuestas @ IoU 0.5 |
|---|---:|
| pre-NMS 1764, post-NMS 1200 | **0.7147** |
| pre-NMS 1764, post-NMS 1764 | **0.7155** |
| pre-NMS 1764, post-NMS 600 | 0.6958 |

La revisión visual deja una conclusión importante: aun usando 600 propuestas, las detecciones tienen scores entre aproximadamente `0.05` y `0.14`, con varias cajas mal ubicadas o con clase incorrecta. El aumento de propuestas ayudó al RPN, pero no solucionó el aprendizaje de clasificación y refinamiento final. No es justificable seguir agregando épocas sin verificar primero la asignación de objetivos, los deltas de caja y el postproceso de la cabeza de detección.

### Auditoría posterior al refinamiento

Se revisaron `assign_proposal_targets`, `detection_loss` y `postprocess_detections`. La asignación mantiene IoU positivo `>= 0.5`, inyecta cajas reales para garantizar positivos, codifica los deltas con la misma convención que `decode_boxes`, y aplica NMS por clase. No se encontró una inconsistencia de formato que explique por sí sola las cajas débiles.

Sí se encontró un desbalance en la contribución de clasificación: el muestreo de RoIs usa `1:3` positivas:negativas, pero el peso de fondo en CrossEntropy es `1.0`, igual al promedio de los objetos. Por tanto, aun después de usar pesos por frecuencia entre clases de objeto, los ejemplos de fondo aportan aproximadamente tres veces más pérdida que los objetos. Esto concuerda con scores bajos para las clases de interés.

Se configuró una prueba controlada que reduce solamente el peso de fondo a `0.25`, conserva los pesos relativos de las siete clases de objeto, el muestreo 1:3, la CrossEntropy y toda la arquitectura. Continúa durante 8 épocas desde los pesos actuales, con el mismo optimizador y las mismas 600 propuestas de detección. La celda mide mAP antes y después bajo la misma configuración para evitar una comparación engañosa.

### Resultado del balance de fondo

La prueba se ejecutó completa en aproximadamente 7 minutos. El balance aumentó la señal de las clases de objeto y produjo la mejor mAP obtenida hasta ahora.

| Métrica | Inicio, época 24 | Mejor valor | Final, época 31 |
|---|---:|---:|---:|
| Loss total | 1.2114 | 1.1264 | 1.1264 |
| RPN loss | 0.3491 | 0.3179 (época 30) | 0.3205 |
| Detection loss | 0.8623 | 0.8060 | 0.8060 |
| Recall RPN validación, top-N 1200 | 0.7197 | 0.7355 (época 30) | 0.7143 |
| mAP@0.5 validación, 600 propuestas | 0.0897 | **0.1059 (época 27)** | 0.0912 |

La mejor mAP (`0.1059`) supera en aproximadamente 22% el mejor resultado anterior del refinamiento con 600 propuestas (`0.0867`). El recall también mejoró: con 600 propuestas ya alcanza `0.7025`, y con 1200 propuestas queda en `0.7143`. Por tanto, el RPN cumple el criterio incluso con un número de propuestas más práctico para la detección.

La inspección visual confirma una mejora de scores, con algunos entre `0.18` y `0.22`, pero revela el límite actual: siguen apareciendo muchas detecciones adicionales y varias no coinciden con la caja o clase real. El balance de fondo ayudó, pero no basta aún para afirmar el criterio de tres detecciones correctas. No se recomienda continuar modificando pérdidas o entrenando sin medir primero ese criterio de manera explícita a distintos umbrales de score.

Nota de trazabilidad: la mejor mAP ocurrió en la época 27, pero no se guardó un checkpoint por decisión previa del grupo. Las imágenes actuales corresponden al modelo de la época 31. No se debe presentar visualmente la época 27 como modelo final; recuperar esos pesos requeriría repetir la fase con selección del mejor estado en memoria o decidir guardar un checkpoint.

### Medición pendiente del criterio del Módulo 4

Se añadió una celda que mide formalmente las detecciones correctas por imagen. Una predicción cuenta como correcta solo si se empareja una vez con una caja real de la misma clase y tiene IoU `>= 0.5`. Evalúa los umbrales de score `0.05`, `0.10`, `0.15`, `0.20`, `0.30` y `0.50`, manteniendo NMS por clase y 600 propuestas.

La celda reporta dos versiones:

- Estricta: imágenes con exactamente tres objetos y tres clases. Se espera `0` en este dataset; es evidencia de que el conjunto no permite aplicar literalmente esa frase de la rúbrica.
- Adaptada: imágenes con al menos tres objetos y al menos tres clases. Para cada umbral se informa el porcentaje con tres o más aciertos y con exactamente tres aciertos.

Esta medición no modifica el modelo. Su resultado decidirá si hay base para un último ajuste de inferencia o si debe documentarse que el criterio de detección no fue alcanzado pese a que el RPN sí cumplió su objetivo.

### Resultado formal del criterio de tres detecciones

La medición se ejecutó sobre las 11 imágenes de validación que cumplen la adaptación `>=3` objetos y `>=3` clases. No existen imágenes que cumplan literalmente `exactamente 3` objetos y `3` clases.

| Umbral de score | Imágenes con 3 o más detecciones correctas | Imágenes con exactamente 3 detecciones correctas | Matches correctos | Detecciones totales |
|---:|---:|---:|---:|---:|
| 0.05 | 36.4% (4 de 11) | 0.0% | 29 | 551 |
| 0.10 | 27.3% (3 de 11) | 9.1% (1 de 11) | 21 | 201 |
| 0.15 | 9.1% (1 de 11) | 0.0% | 14 | 84 |
| 0.20 | 9.1% (1 de 11) | 9.1% (1 de 11) | 5 | 24 |
| 0.30 | 0.0% | 0.0% | 0 | 0 |
| 0.50 | 0.0% | 0.0% | 0 | 0 |

**Conclusión:** ningún umbral alcanza el 70% pedido por la rúbrica. El umbral `0.05` recupera más objetos, pero a costa de 551 predicciones y muchos falsos positivos; el umbral por defecto `0.50` no produce detecciones. Por tanto, ajustar solamente el umbral de inferencia no puede resolver el criterio del Módulo 4 y no sería correcto presentarlo como cumplido.

El Módulo 2 sí queda validado por el recall del RPN superior a `0.7`, y el Módulo 5 queda implementado con entrenamiento conjunto y métricas por época. Para mejorar el criterio de detección se requeriría una modificación de capacidad de la `DetectionHead` y una nueva corrida controlada desde cero; no se debe continuar alterando el score threshold para simular una mejora.

### Corrida pendiente con DetectionHead de mayor capacidad

Se cambió la cabeza de detección de una capa compartida a dos capas fully connected compartidas: `12544 -> 512 -> 256`. Las salidas permanecen iguales: logits `[N, C+1]` y deltas `[N, 4]`. Este patrón aporta más capacidad para combinar las características de ROI antes de clasificar y refinar la caja, sin modificar el backbone, RPN, anchors, ROI Align, asignación IoU, NMS ni las pérdidas definidas para el taller.

Como la nueva cabeza tiene parámetros distintos, los experimentos anteriores no se pueden reutilizar: la próxima corrida debe iniciar desde cero y repetir la misma secuencia controlada de 15 épocas base, 8 de refinamiento y 8 de balance de fondo. Al terminar se deben volver a ejecutar la revisión visual, la cobertura del RPN y la métrica formal del Módulo 4.

### Resultado base con DetectionHead de dos capas

La corrida base de 15 épocas se completó en aproximadamente 9 minutos y 20 segundos. El RPN recuperó y superó la meta sin necesidad de los refinamientos posteriores.

| Métrica | Final, época 15 | Mejor valor en la corrida |
|---|---:|---:|
| Loss total | 0.9893 | 0.9893 (época 15) |
| RPN loss | 0.4057 | 0.4057 (época 15) |
| Detection loss | 0.5836 | 0.5734 (época 4) |
| Recall RPN validación, top-N 1200 | **0.7193** | **0.7331** (época 14) |
| mAP@0.5 validación | 0.0325 | 0.0336 (época 11) |

Conclusión intermedia: la nueva DetectionHead no degradó el RPN; el criterio de proposal recall sigue superado. La mAP todavía es baja durante esta primera fase, por lo que falta ejecutar el refinamiento con 600 propuestas y, después, el balance de fondo antes de comparar con la arquitectura anterior.

### Resultado completo con DetectionHead de dos capas

Después de las 15 épocas base, 8 de refinamiento y 8 de balance de fondo, el RPN se mantuvo sobre la meta, pero la nueva cabeza no superó la mejor mAP de la arquitectura anterior.

| Fase | Mejor recall RPN validación | Mejor mAP@0.5 | mAP final de la fase |
|---|---:|---:|---:|
| Base, épocas 1-15 | 0.7331 | 0.0336 | 0.0325 |
| Refinamiento, épocas 16-23 | 0.7257 | 0.0353 | 0.0331 |
| Balance de fondo, épocas 24-31 | 0.7437 | 0.0737 | 0.0737 |

El modelo final de dos capas conserva `recall RPN = 0.7104` en validación y la mejor mAP de la corrida es `0.0737`. Por comparación, la cabeza anterior alcanzó `0.1059`; por tanto, el aumento de capacidad no mejoró la métrica global en esta configuración.

La revisión visual sí cambió: aparecieron scores más altos, por ejemplo varias predicciones de `jellyfish` entre `0.45` y `0.65`. Algunas se alinean mejor con sus cajas verdes, pero aún hay muchas predicciones por imagen (`59` a `163` en las cinco mostradas) y falsos positivos. Esto confirma que el siguiente paso debe ser volver a ejecutar la métrica formal del Módulo 4 con este modelo, no introducir otra arquitectura ni continuar entrenando sin medición.

### Criterio del Módulo 4 con DetectionHead de dos capas

| Umbral de score | Imágenes con 3 o más detecciones correctas | Imágenes con exactamente 3 detecciones correctas | Matches correctos |
|---:|---:|---:|---:|
| 0.05 | 36.4% (4 de 11) | 0.0% | 29 |
| 0.10 | 36.4% (4 de 11) | 9.1% (1 de 11) | 28 |
| 0.15 | 36.4% (4 de 11) | 27.3% (3 de 11) | 25 |
| 0.20 | 27.3% (3 de 11) | 18.2% (2 de 11) | 21 |
| 0.30 | 0.0% | 0.0% | 0 |
| 0.50 | 0.0% | 0.0% | 0 |

La arquitectura de dos capas mejora la calibración visual de algunos scores, pero no supera el máximo de cobertura de tres aciertos: permanece en `36.4%`, igual que la cabeza anterior y debajo del 70% requerido. El criterio literal sigue sin ser directamente aplicable porque existen cero imágenes con exactamente tres objetos y tres clases.

### Nueva corrida: diversidad de entrenamiento sin modificar validación

La inspección del cargador de datos mostró que las 352 imágenes de entrenamiento siempre se presentaban sin ninguna variación. Esa ausencia de aumentación limita la diversidad que observa el modelo, especialmente para un conjunto pequeño. Se prepara una nueva corrida controlada con los cambios siguientes:

| Elemento | Configuración | Motivo |
|---|---|---|
| Cabeza de detección | Se vuelve a una FC compartida `12544 -> 256` | Fue la arquitectura con la mejor mAP previa (`0.1059`); las dos FC no la superaron. |
| Train | Volteo horizontal con probabilidad `0.5` y `ColorJitter` suave | Amplía variaciones espaciales e iluminación submarina. Las cajas se reflejan junto con la imagen. |
| Validación | Sin aumentación | La mAP y el recall permanecen comparables y no se contaminan. |
| RPN, RoI Align, IoU, NMS y pérdidas | Sin cambios | Se mantiene exactamente la lógica exigida por el taller. |
| Selección del modelo | Mejor mAP conservada solo en memoria | Al final se usan para visualización los pesos de la mejor época, sin guardar checkpoint a disco. |

Esta prueba inicia de cero y repite las 15 épocas base. Si mejora la mAP, se podrán volver a ejecutar las fases de refinamiento y balance de fondo para comparar con el mismo protocolo. No se añadió aún un sampler agresivo por clase: con los falsos positivos observados, sobreexponer objetos raros podría empeorar la precisión.

### Resultado base con aumentación solo en entrenamiento

La corrida de 15 épocas se completó sin errores. La aumentación no se aplicó a validación. El mejor estado quedó conservado únicamente en memoria en la época 13.

| Métrica | Mejor valor | Época | Valor final, época 15 |
|---|---:|---:|---:|
| Recall RPN validación, top-N 1200 | **0.7343** | 11 | 0.6858 |
| mAP@0.5 validación | **0.0731** | 13 | 0.0707 |
| Loss total | 1.0896 | 14 | 1.0932 |
| RPN loss | 0.4927 | 15 | 0.4927 |
| Detection loss | 0.5662 | 3 | 0.6005 |

La mejor época por mAP (13) también conserva `recall RPN = 0.7075`, por encima de la meta `0.7`. Frente a la base de dos FC, cuya mejor mAP fue `0.0336`, el resultado mejora de forma clara; frente a la corrida previa de una FC, el aumento es pequeño. Por eso el siguiente paso razonable es ejecutar el refinamiento ya definido, manteniendo la misma aumentación y sin añadir más cambios simultáneos.

### Resultado del refinamiento con aumentación

El refinamiento continuó durante ocho épocas, de la 16 a la 23, con `learning_rate=3e-5`, 600 propuestas y 256 RoIs por imagen. Las pérdidas sí bajaron, pero la validación no superó el mejor punto de la fase base.

| Métrica | Mejor en refinamiento | Mejor global actual | Lectura |
|---|---:|---:|---|
| Detection loss | 0.4949 (época 20) | 0.5662 (base, época 3) | El ajuste de entrenamiento continúa aprendiendo. |
| RPN loss | 0.4472 (época 22) | 0.4927 (base, época 15) | El RPN reduce pérdida, sin una ganancia proporcional en validación. |
| Recall RPN validación | 0.7197 (época 18) | **0.7343** (base, época 11) | Se mantiene cerca de la meta, pero no la supera. |
| mAP@0.5 validación | 0.0730 (época 20) | **0.0731** (base, época 13) | El refinamiento no mejoró la generalización. |

La selección en memoria conserva correctamente la época 13 de la fase base como mejor estado global. Esta fase se mantiene como evidencia de una prueba controlada: las pérdidas bajaron, pero la mAP no, por lo que no se debe afirmar que más épocas o más RoIs hayan mejorado el detector.

### Resultado del balance entre fondo y objetos con aumentación

El balance de fondo sí mejoró la métrica de detección. El peso de fondo se redujo a `0.25` manteniendo el muestreo de RoIs `1:3`, los pesos relativos entre objetos, las pérdidas y el resto del pipeline.

| Métrica | Inicio, época 24 | Mejor valor | Final, época 31 |
|---|---:|---:|---:|
| Loss total | 1.3292 | 1.2357 | 1.2357 |
| RPN loss | 0.4404 | 0.4122 (época 31) | 0.4122 |
| Detection loss | 0.8888 | 0.8235 (época 31) | 0.8235 |
| Recall RPN validación, top-N 1200 | 0.7118 | 0.7169 (época 27) | 0.7009 |
| mAP@0.5 validación | 0.0647 | **0.1168 (época 30)** | 0.0942 |

La mAP máxima (`0.1168`) supera en aproximadamente 60% la mejor de la fase base con aumentación (`0.0731`). El notebook recuperó en memoria los pesos de la época 30 para las visualizaciones y evaluaciones siguientes; no se creó ningún checkpoint en disco.

En la época 30, el recall RPN con top-N 1200 fue `0.6954`, ligeramente por debajo de `0.7`. La verificación de cobertura con distintos top-N debe ejecutarse sobre el modelo recuperado antes de cerrar el Módulo 2: conservar más propuestas tras NMS puede recuperar ese pequeño margen sin alterar el modelo.

### Evaluación final del modelo recuperado (época 30)

La revisión visual muestra cajas finales con scores más altos y mejor localización en varias regiones; por ejemplo, varias medusas aparecen con scores entre `0.32` y `0.46`. También persisten falsos positivos y confusiones de clase, especialmente en escenas con peces y pingüinos. La mejora visual es coherente con el aumento de mAP, pero no es suficiente para declarar que el detector resolvió todas las clases.

La cobertura del RPN del modelo recuperado fue evaluada con distintos límites de propuestas:

| Propuestas tras NMS | Recall de propuestas, IoU=0.5 |
|---:|---:|
| 100 | 0.4632 |
| 300 | 0.6290 |
| 600 | 0.6790 |
| 1200 | 0.6954 |
| 1764 | 0.6954 |

**Lectura:** al pasar de 1200 a 1764 propuestas el recall no cambia, así que el problema no es estar descartando propuestas por el límite top-N. Frente al mejor RPN de la fase base (`0.7343`), el modelo seleccionado por mAP pierde `0.0389` de recall. Es un intercambio real entre cobertura de propuestas y precisión de detección final; no se debe ocultar ni intentar corregir solo cambiando top-N.

### Decisión metodológica sobre la meta de cobertura

Se consideró añadir una recuperación posterior que eligiera solo épocas con `recall RPN >= 0.72`. Se descartó antes de ejecutarla: fijar esa restricción después de observar todas las curvas habría sido selección sesgada de resultados, no una mejora demostrable.

Para una siguiente prueba válida, los cambios de arquitectura, pérdida, cantidad de épocas y regla de selección deben definirse antes de entrenar; después se reportan todas las métricas de la corrida completa. La fórmula obligatoria del taller se conserva como `loss_total = rpn_loss + detection_loss`.

La métrica formal adaptada del Módulo 4 volvió a dar un máximo de `36.4%` (4 de 11 imágenes) con tres o más detecciones correctas al umbral `0.05`. En el umbral `0.10` conserva el mismo 36.4% con menos matches totales; a partir de `0.15` cae a 9.1%. Por tanto, la mAP mejoró y la inspección visual es más útil, pero el criterio adaptado de 70% no se alcanza. El criterio literal no es evaluable porque el conjunto tiene cero imágenes con exactamente tres objetos y tres clases.

## Cambios preparados para la corrida final (Restart & Run All)

Estos cambios se definieron **antes** de la corrida final, para no elegir reglas después de ver los resultados.

| Cambio | Dónde | Motivo |
|---|---|---|
| Regla de selección del modelo: mayor mAP@0.5 entre las épocas con recall RPN (IoU 0.5, top-N 1200) `>= 0.70`; si ninguna cumple, mayor mAP y se reporta así | Módulo 5 (`actualizar_mejor_validacion`) | Elegir solo por mAP dejó un modelo con recall 0.695. En la corrida anterior esta regla habría elegido la época 29: recall 0.715 y mAP 0.1125, frente a 0.1168. |
| Empates en la mejor IoU por GT: todas las anclas empatadas se marcan positivas | Módulo 2 (`assign_targets`) | Es lo que describe Ren et al. (2015) ("the anchor/anchors with the highest IoU"). No cambia los umbrales 0.7/0.3 ni el muestreo 1:1. |
| Deltas de la cabeza de detección normalizados con pesos `(10, 10, 5, 5)` | Módulo 4 (`encode_detection_boxes` / `decode_detection_boxes`) | Fast R-CNN normaliza los objetivos de regresión. Sin esa escala los ajustes finos son muy pequeños y la SmoothL1 da poco gradiente de localización, lo que concuerda con las cajas mal ubicadas de la revisión visual. |
| Revisión visual época 1 vs modelo seleccionado; aciertos en azul y errores en rojo; score >= 0.10, NMS entre clases solo para la figura, máximo 8 cajas | Revisión visual | Hace visible el efecto del entrenamiento sin alterar la mAP ni el criterio del Módulo 4. |

Se mantienen sin cambios: arquitectura, anchors `(16, 32, 64) × (0.5, 1, 2)`, umbrales IoU, muestreos 1:1 y 1:3, `loss_total = rpn_loss + detection_loss`, número de épocas y aumentación.

## Hallazgo: la resolución de entrada limita la detección

- Imágenes originales: mayoría 1536×2048 (304) y 4032×3024 (92). Lado equivalente de los objetos en tamaño original: cuartiles 106 / 167 / 275 px.
- Tras redimensionar a 224×224: cuartiles 11.8 / 18.7 / 29.9 px; **40.8 %** de los objetos mide menos de 16 px (ancla mínima y celda del mapa 14×14) y **10.9 %** menos de 8 px.
- `Resize((224, 224))` no conserva la proporción: las fotos verticales 3:4 se aplastan y cambia la forma de los animales.
- Interpretación: esta es la causa principal de que el recall del RPN se estanque cerca de 0.70–0.73 y de que la mAP y los scores sean bajos. No es un error de implementación.
- Decisión: se mantiene 224×224 por la especificación del Módulo 1 y por el costo en CPU (448×448 sería unas 4 veces más lento por época). Mejoras naturales con más recursos: mayor resolución, resize con *padding* que conserve la proporción, o recortes (*tiles*) de las fotos grandes.
- Se agregó al notebook como celda markdown después del diagnóstico de tamaños (Módulo 2), y el histograma ahora marca las escalas reales de los anchors (16, 32, 64).

## Resultado de la corrida final (Restart & Run All, 31 épocas)

| Métrica | Corrida anterior (selección solo por mAP) | Corrida final (regla recall ≥ 0.70 + cambios) |
|---|---:|---:|
| Época seleccionada | 30 | 30 (balance de fondo) |
| mAP@0.5 validación | 0.1168 | **0.1195** |
| Recall RPN validación, top-N 1200 | 0.6954 (no cumple) | **0.7422 (cumple)** |
| Recall RPN, top-N 600 / 1764 | 0.6790 / 0.6954 | 0.7132 / 0.7450 |
| Criterio M4 adaptado, ≥3 aciertos (umbral 0.05 y 0.10) | 36.4 % | **45.5 %** |
| Criterio M4, exactamente 3 aciertos (umbral 0.05 y 0.10) | 0 % / 9.1 % | 18.2 % / 18.2 % |

Por fase:
- **Base (épocas 1-15):** el recall ≥ 0.70 se sostiene desde la época 10; la mAP termina en 0.0735.
- **Refinamiento (16-23):** el recall se mantiene entre 0.708 y 0.730, pero la mAP baja a 0.03-0.06 y la loss de detección queda plana. No mejoró el detector.
- **Balance de fondo (24-31):** la mAP sube hasta 0.1195 en la época 30. Esta fase es la que más aporta, igual que en la corrida anterior.

Desglose de la loss de detección (fase base): `det_cls_loss` casi no cambia (0.37 → 0.41) y `det_reg_loss` sube de 1.31 a 1.96. La subida de la loss se debe a la regresión: con los pesos (10, 10, 5, 5) cambia la escala y, a medida que el RPN mejora, entran más propuestas positivas reales (de ~44 a ~70 RoIs positivas por batch), con deltas distintos de cero. La regresión pesa unas 5 veces más que la clasificación. La loss no se puede comparar con la de corridas anteriores.

Revisión visual (5 imágenes, score ≥ 0.10): el modelo de la época 1 no produce ninguna detección que supere el umbral (0 de 48). El modelo seleccionado muestra 39 detecciones, de las cuales 7 son correctas; los aciertos se concentran en `jellyfish`. Persisten falsos positivos y confusiones de clase (por ejemplo `penguin` o `jellyfish` sobre tiburones, y `starfish` sobre frailecillos).

Conclusión: el Módulo 2 cumple la meta de recall en el modelo final. El criterio del Módulo 4 (70 %) no se alcanza (45.5 %), por la limitación de resolución documentada.

## Módulo 6: primer resultado y diagnóstico

- El objectness se comporta como mapa difuso y geométrico: cubre cerca del 20 % del mapa y marca rocas, animales y bordes sin distinguir la clase.
- El Smooth Grad-CAM, tal como se especifica, salió vacío en 3 de 5 imágenes (correlación `NaN`). En las otras dos solo quedaron puntos en las esquinas, un artefacto del *padding*.
- Diagnóstico: el logit de la clase analizada es negativo en las 5 imágenes (entre −0.73 y −2.22) y el CAM antes del ReLU es negativo en casi todo el mapa (máximo ≤ 0.003), así que el ReLU lo anula. Usar "logit de la clase − logit del fondo" empeora el resultado (entre −3.9 y −6.5).
- Interpretación: la cabeza de detección tiene evidencia neta en contra de las clases que predice, lo que es coherente con scores de 0.13 a 0.29.
- Decisión: se mantiene el mapa pedido y se reporta como hallazgo. Además se agrega la explicación contrafactual `ReLU(-CAM)` (Selvaraju et al., 2017), que muestra qué zonas empujan en contra de la clase.
