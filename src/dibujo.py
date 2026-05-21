import matplotlib.pyplot as plt

def plot_learning_curves(historiales, titulo="Comparativa de Entrenamiento"):
    """
    Genera una gráfica profesional comparando múltiples configuraciones de entrenamiento.
    
    Args:
        historiales (dict): Diccionario donde cada clave es el nombre del experimento 
                            y el valor es otro diccionario con 'train' y 'test'.
    """
    plt.figure(figsize=(12, 6))
    
    for nombre, hist in historiales.items():
        # Asumimos que hist es un diccionario con las claves 'train' y 'test'
        epochs = range(1, len(hist['train']) + 1)
        
        # Línea de Train: sólida y más clara
        plt.plot(epochs, hist['train'], label=f"{nombre} (Train)", alpha=0.6, linestyle='--')
        
        # Línea de Test: sólida y más gruesa
        plt.plot(epochs, hist['test'], label=f"{nombre} (Test)", linewidth=2)
        
    plt.xlabel("Época")
    plt.ylabel("Loss (MSE)")
    plt.title(titulo)
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Esto es el toque profesional: guardar en alta resolución para tu memoria
    plt.savefig('comparativa_modelos.png', dpi=300, bbox_inches='tight')
    plt.show()