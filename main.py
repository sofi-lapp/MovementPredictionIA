
from steering_system import (
    SteeringWheelDataCollector,
    SteeringWheelModel,
    RealTimeSteeringPredictor,
    MODEL_DIR
)
import os

def collect_data():
    #Opción 1
    print("\n" + "="*70 + "\nRECOPILACIÓN DE DATOS\n" + "="*70)
    num_samples = int(input("\nNumero de muestras (recomendado 100-300): "))
    collector = SteeringWheelDataCollector()
    collector.collect_training_data(num_samples=num_samples)

def train_model():
    #Opción 2

    print("\n" + "="*70 + "\nENTRENAMIENTO DEL MODELO\n" + "="*70)
    #print("\nUsando modelo ADVANCED (máxima precisión)")
    print("Arquitectura: 5 capas con conexiones residuales + Huber loss")
    
    model = SteeringWheelModel()
    
    try:
        # Cargar datos
        X, y = model.load_all_training_data()
        
        # Construir modelo
        model.build_model()
        print("\nResumen del modelo:")
        model.model.summary()
        
        # Épocas recomendadas según cantidad de datos
        num_samples = len(X)
        if num_samples < 100:
            recommended = 100
        elif num_samples < 300:
            recommended = 150
        elif num_samples < 1000:
            recommended = 300
        else:
            recommended = 500
        
        print(f"\n Épocas recomendadas para {num_samples} muestras: {recommended}")
        print("Early Stopping activado: se detendrá automáticamente si no mejora")
        
        epochs_input = input(f"\nNumero de epocas (Enter = {recommended}): ").strip()
        epochs = int(epochs_input) if epochs_input else recommended
        
        # Entrenar
        model.train(X, y, epochs=epochs)
        
        # Asegurar que directorio de modelos existe
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Guardar
        model.save_model()
        
        print("\nEntrenamiento completado exitosamente")
        
    except FileNotFoundError as e:
        print(f"\nError: No se encontraron datos de entrenamiento")
        print("Asegurate de haber recopilado datos primero (opcion 1)")
    except ValueError as e:
        print(f"\nError: {e}")
        print("Verifica que los archivos de datos sean válidos")
    except Exception as e:
        print(f"\nError inesperado: {e}")
        import traceback
        traceback.print_exc()


def predict_realtime():
    #Opción 3
    print("\n" + "="*70)
    print("PREDICCION EN TIEMPO REAL")
    print("="*70)
    
    # Detectar tipo de modelo si existe
    model = SteeringWheelModel()
    
    try:
        model.load_model()
        
        show_console = input("\nMostrar valores en consola? (s/n): ").lower() == 's'
        
        predictor = RealTimeSteeringPredictor(model)
        predictor.run(show_console_output=show_console)
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Asegurate de haber entrenado el modelo primero (opcion 2)")


def full_workflow():
    #Opcion 4
    print("\n" + "="*70)
    print("FLUJO COMPLETO")
    print("="*70)
    print("\nEste proceso incluye:")
    print("  1. Recopilacion de datos (manual)")
    print("  2. Entrenamiento del modelo")
    print("  3. Prediccion en tiempo real")
    
    input("\nPresiona ENTER para continuar...")
    
    # 1. Recopilar datos
    print("\n[1/3] Recopilando datos...")
    print("Iniciando en 3...")
    import time
    time.sleep(1)
    print("2...")
    time.sleep(1)
    print("1...")
    time.sleep(1)
    print("¡GO!\n")
    
    collector = SteeringWheelDataCollector()
    collector.collect_training_data(num_samples=150)
    
    # 2. Entrenar
    print("\n[2/3] Entrenando modelo...")
    print("Usando modelo ADVANCED (máxima precisión)")
    
    try:
        model = SteeringWheelModel()
        X, y = model.load_all_training_data()
        
        print(f"\nDatos cargados: {len(X)} muestras")
        
        model.build_model()
        model.train(X, y, epochs=150)
        
        # Asegurar que directorio de modelos existe
        os.makedirs(MODEL_DIR, exist_ok=True)
        model.save_model()
        
    except Exception as e:
        print(f"\n❌ Error en entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. Predecir
    print("\n[3/3] Iniciando prediccion...")
    input("Presiona ENTER para comenzar...")
    predictor = RealTimeSteeringPredictor(model)
    predictor.run(show_console_output=True)


def show_info():
    #Opcion 5
    print("\n" + "="*70)
    print("INFORMACION DEL SISTEMA - VOLANTE VIRTUAL IA")
    print("="*70)
    
    print("\nDESCRIPCION:")
    print("  Sistema de Deep Learning para control de volante virtual")
    print("  Detecta manos con MediaPipe y predice el ángulo en tiempo real")
    
    print("\nTIPO DE RED NEURONAL:")
    print("  DNN (Deep Neural Network) - Red Neuronal Profunda")
    print("  Optimizada para regresión con conexiones residuales")
    
    print("\nARQUITECTURA DEL MODELO:")
    print("  ├─ Tipo: Fully Connected (Dense)")
    print("  ├─ Capas totales: 7 capas (5 ocultas + input + output)")
    print("  ├─ Neuronas por capa:")
    print("  │   • Capa 1: 256 neuronas")
    print("  │   • Capa 2: 256 neuronas (con conexión residual)")
    print("  │   • Capa 3: 128 neuronas")
    print("  │   • Capa 4: 64 neuronas")
    print("  │   • Capa 5: 32 neuronas")
    print("  │   • Salida: 1 neurona (ángulo del volante)")
    print("  └─ Parámetros totales: ~230,000")
    
    print("\nTECNICAS AVANZADAS:")
    print("  • BatchNormalization - Estabiliza el entrenamiento")
    print("  • Dropout progresivo - Previene overfitting (40% → 20%)")
    print("  • Regularización L2 - Penaliza pesos grandes")
    print("  • Conexiones residuales - Mejora flujo de información")
    print("  • Huber Loss - Robusto ante valores atípicos")
    print("  • Learning Rate Decay - Ajusta velocidad de aprendizaje")
    print("  • Early Stopping - Detiene si no mejora (15 épocas)")
    
    print("\nENTRADA DEL MODELO:")
    print("  126 valores numéricos:")
    print("  ├─ 2 manos × 21 landmarks × 3 coordenadas (x, y, z)")
    print("  └─ MediaPipe extrae puntos clave de las manos detectadas")
    
    print("\nSALIDA DEL MODELO:")
    print("  1 valor continuo en el rango [-1.0, +1.0]:")
    print("  ├─ -1.0 = Volante girado completamente a la izquierda")
    print("  ├─  0.0 = Volante en posición central (recto)")
    print("  └─ +1.0 = Volante girado completamente a la derecha")
    
    print("\RENDIMIENTO:")
    print("  • Velocidad de predicción: ~5-15ms (60-200 FPS)")
    print("  • Funciona en CPU sin necesidad de GPU")
    print("  • Detección automática de ambas manos")
    print("  • Suavizado de predicciones para estabilidad")
    
    print("\nMODO DE CAPTURA (Automático Inteligente):")
    print("  1. El volante se mueve automáticamente por posiciones clave")
    print("  2. Coloca tus manos siguiendo el volante virtual")
    print("  3. Captura solo cuando AMBAS manos están detectadas")
    print("  4. Mayor densidad de muestras en centro y extremos")
    print("  5. Semáforo estilo F1 para inicio")
    
    print("\nRECOMENDACIONES:")
    print("  • Captura mínimo: 150-200 muestras")
    print("  • Óptimo: 300-500 muestras para mejor precisión")
    print("  • Mantén buena iluminación constante")
    print("  • Posiciona la cámara a la altura del pecho")
    print("  • Entrena con 100-300 épocas (auto-stop incluido)")
    
    print("\nESTRUCTURA DEL PROYECTO:")
    print("  steering_system/")
    print("    ├─ config.py        - Configuración del sistema")
    print("    ├─ data_collector.py - Captura inteligente de datos")
    print("    ├─ model.py         - Red neuronal DNN avanzada")
    print("    ├─ predictor.py     - Predicción en tiempo real")
    print("    └─ ui_utils.py      - Interfaz visual")
    
    print("\nARCHIVOS GENERADOS:")
    print("  • steering_data/*.npy - Datasets de entrenamiento")
    print("  • models/steering_model.h5 - Modelo entrenado principal")
    print("  • models/best_steering_model.h5 - Mejor modelo (checkpoint)")
    print("  • models/steering_stats.pkl - Parámetros de normalización")
    
    print("\n¿POR QUE DNN Y NO CNN?")
    print("  • CNN es para imágenes con patrones espaciales")
    print("  • DNN es ideal para coordenadas y relaciones numéricas")
    print("  • Tu input son 126 números")
    print("  • DNN es 10x más rápida y precisa para este caso")
    print("  • Menos parámetros = menos overfitting")
    
    print("\n" + "="*70)
    print("Para más info técnica, consulta: MEJORAS.md y README.md")
    print("="*70)


def main():
    while True:
        print("\n" + "="*70)
        print("SISTEMA DE VOLANTE VIRTUAL CON DEEP LEARNING")
        print("="*70)
        #print("\nControla un volante virtual con tus manos")
        #print("Salida: valores entre -1 (izquierda) y +1 (derecha)")
        print("\nOpciones:")
        print("  1. Recopilar datos de entrenamiento.")
        print("  2. Entrenar modelo.")
        print("  3. Prediccion en tiempo real.")
        print("  4. Flujo completo (recopilación, entrenamiento y predicción)")
        print("  5. Informacion del sistema")
        print("  6. Salir")
        print("="*70)
        
        option = input("\nSelecciona una opcion (1-6): ")
        
        options = {
            '1': collect_data,
            '2': train_model,
            '3': predict_realtime,
            '4': full_workflow,
            '5': show_info
        }
        
        if option == '6':
            print("Apagando el sistema.")
            break

        if option in options:
            options[option]()
        else:
            print("Opcion invalida")


if __name__ == "__main__":
    main()
