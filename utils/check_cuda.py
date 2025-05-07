import torch

# Comprueba si CUDA está disponible
print(f"¿CUDA disponible?: {torch.cuda.is_available()}")

# Si CUDA está disponible, muestra información adicional
if torch.cuda.is_available():
    print(f"Número de dispositivos CUDA: {torch.cuda.device_count()}")
    print(f"Dispositivo actual: {torch.cuda.current_device()}")
    print(f"Nombre del dispositivo: {torch.cuda.get_device_name(0)}")
    
    # Prueba simple para verificar que las operaciones funcionen
    x = torch.rand(5, 3)
    print("Tensor en CPU:")
    print(x)
    
    # Mover a GPU
    x = x.cuda()
    print("Tensor en GPU:")
    print(x)
    
    # Operación simple en GPU
    y = x + x
    print("Resultado de operación en GPU:")
    print(y)
    
    print("Todo parece estar funcionando correctamente con CUDA!")
else:
    print("CUDA no está disponible. Verifica tu instalación.")