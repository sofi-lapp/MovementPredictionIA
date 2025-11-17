"""
Punto de entrada principal del Sistema de Volante Virtual

Este módulo proporciona un menú interactivo para:
    1. Recopilar datos de entrenamiento
    2. Entrenar el modelo4
    3. Ejecutar predicción en tiempo real
    4. Flujo completo (recopilar + entrenar + predecir)
    5. Ver información del sistema
"""

from steering_system import (
    SteeringWheelDataCollector,
    SteeringWheelModel,
    RealTimeSteeringPredictor,
    MODEL_DIR
)
import os


def collect_data():
    """Opción 1: Recopilar datos de entrenamiento"""
    print("\n" + "="*70)
    print("RECOPILACION DE DATOS")
    print("="*70)
    
    num_samples = int(input("\nNumero de muestras (recomendado 100-300): "))
    collector = SteeringWheelDataCollector()
    collector.collect_training_data(num_samples=num_samples)


def train_model():
    """Opción 2: Entrenar modelo"""
    print("\n" + "="*70)
    print("ENTRENAMIENTO DEL MODELO")
    print("="*70)
    print("\nUsando modelo ADVANCED (máxima precisión)")
    print("Arquitectura: 5 capas con conexiones residuales + Huber loss")
    
    model = SteeringWheelModel()
    
    try:
        # Cargar datos
        X, y = model.load_all_training_data()
        
        print(f"\n📊 Datos cargados: {len(X)} muestras")
        
        # Construir modelo
        model.build_model()
        print("\n📊 Modelo ADVANCED creado exitosamente")
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
        
        print(f"\n💡 Épocas recomendadas para {num_samples} muestras: {recommended}")
        print("⚡ Early Stopping activado: se detendrá automáticamente si no mejora")
        
        epochs_input = input(f"\nNumero de epocas (Enter = {recommended}): ").strip()
        epochs = int(epochs_input) if epochs_input else recommended
        
        # Entrenar
        model.train(X, y, epochs=epochs)
        
        # Asegurar que directorio de modelos existe
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Guardar
        model.save_model()
        
        print("\n✅ Entrenamiento completado exitosamente")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: No se encontraron datos de entrenamiento")
        print("Asegurate de haber recopilado datos primero (opcion 1)")
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("Verifica que los archivos de datos sean válidos")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()


def predict_realtime():
    """Opción 3: Predicción en tiempo real"""
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
        print(f"\n❌ Error: {e}")
        print("Asegurate de haber entrenado el modelo primero (opcion 2)")


def full_workflow():
    """Opción 4: Flujo completo"""
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
        
        print(f"\n📊 Datos cargados: {len(X)} muestras")
        
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
    """Opción 5: Información del sistema"""
    print("\n" + "="*70)
    print("INFORMACION DEL SISTEMA")
    print("="*70)
    print("\n-- Descripcion:")
    print("  Sistema de Deep Learning para volante virtual")
    print("  Detecta manos con MediaPipe y predice angulo de volante")
    print("\n-- Modelo: ADVANCED (Máxima Precisión)")
    print("  - Arquitectura: 5 capas densas con conexiones residuales")
    print("  - Parámetros: ~150K")
    print("  - BatchNormalization + L2 Regularization + Dropout")
    print("  - Huber Loss (robusto ante outliers)")
    print("  - Learning Rate Decay exponencial")
    print("\n-- Arquitectura:")
    print("  - Input: 126 valores (2 manos x 21 landmarks x 3 coords)")
    print("  - 5 capas densas: 256 → 256 → 128 → 64 → 32")
    print("  - Conexión residual en capa 2")
    print("  - Output: Valor continuo [-1, +1]")
    print("    * -1.0 = Volante girado completamente a la izquierda")
    print("    *  0.0 = Volante en posicion central (recto)")
    print("    * +1.0 = Volante girado completamente a la derecha")
    print("\n-- Estructura modular:")
    print("  steering_system/")
    print("    |-- __init__.py      - Paquete principal")
    print("    |-- config.py        - Configuracion y constantes")
    print("    |-- data_collector.py - Captura de datos")
    print("    |-- model.py         - Modelo de Deep Learning")
    print("    |-- predictor.py     - Prediccion en tiempo real")
    print("    +-- ui_utils.py      - Utilidades de interfaz")
    print("\n-- Modo de captura (MANUAL):")
    print("  1. Ajusta el angulo con flechas <- -> (o W A S D)")
    print("  2. Posiciona tus manos como si condujeras")
    print("  3. Presiona ESPACIO para capturar esa combinacion")
    print("  4. Repite para diferentes angulos y posiciones")
    print("\n-- Consejos:")
    print("  * Captura 100-300 muestras variadas")
    print("  * Incluye todas las posiciones del volante")
    print("  * Multiples capturas por angulo (con variaciones)")
    print("  * Manten buena iluminacion")
    print("\n-- Archivos generados:")
    print("  * steering_data/*.npy - Datos de entrenamiento")
    print("  * models/steering_model.h5 - Modelo entrenado")
    print("  * models/steering_stats.pkl - Estadisticas de normalizacion")
    print("="*70)


def main():
    """Función principal con menú de opciones"""
    
    print("\n" + "="*70)
    print("SISTEMA DE VOLANTE VIRTUAL CON DEEP LEARNING")
    print("="*70)
    print("\nControla un volante virtual con tus manos")
    print("Salida: valores entre -1 (izquierda) y +1 (derecha)")
    print("\nOpciones:")
    print("  1. Recopilar datos de entrenamiento")
    print("  2. Entrenar modelo")
    print("  3. Prediccion en tiempo real")
    print("  4. Flujo completo (recopilar + entrenar + predecir)")
    print("  5. Informacion del sistema")
    
    option = input("\nSelecciona una opcion (1-5): ")
    
    options = {
        '1': collect_data,
        '2': train_model,
        '3': predict_realtime,
        '4': full_workflow,
        '5': show_info
    }
    
    if option in options:
        options[option]()
    else:
        print("Opcion invalida")


if __name__ == "__main__":
    main()
