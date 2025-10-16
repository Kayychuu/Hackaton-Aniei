Necesitan: tensorflow, pickle, numpy

## Para cargar modelo keras
modelo = tf.keras.models.load_model('modelo_pinn_entrenado.keras')
## Para cargar escalador
with open('scaler.pkl', 'rb') as f:
    escalador = pickle.load(f)
## Hacer una prediccion
punto_nuevo = np.array([[-115.35, 32.4]])
punto_nuevo = escalador.transform(punto_nuevo)
prediccion_normalizada = modelo.predict(punto_nuevo_norm)

*OJO NECESITAN DESNORMALIZAR LOS DATOS PARA QUE LES DEN VALORES REALES*
T_MIN = 15.0
T_MAX = 350.0
temperatura_final = T_MIN + (T_MAX - T_MIN) * prediccion_normalizada[0][0]
print(f"\n--- Predicción ---")
print(f"Coordenadas de entrada: {punto_nuevo[0]}")
print(f"Temperatura predicha: {temperatura_final:.2f} °C")
