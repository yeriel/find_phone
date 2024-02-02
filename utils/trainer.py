import torch
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

# Función para calcular el MAE
def calculate_mae(predictions, targets):
    return torch.mean(torch.abs(predictions - targets))

def train(epochs, model, train_dataloader, val_dataloader, optimizer, device, scheduler=None):

    best_mae = float('inf')  # Inicializar el mejor MAE con infinito positivo

    # Bucle de entrenamiento y validación
    for epoch in range(epochs):
        # Bucle de entrenamiento
        model.train()
        running_loss = 0.0
        running_mae_train = 0.0

        for inputs, targets in tqdm(train_dataloader, desc=f"train epoch {epoch+1}"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = torch.nn.MSELoss()(outputs, targets)
            loss.backward()
            optimizer.step()

            if scheduler:
                scheduler.step()

            running_loss += loss.item()
            running_mae_train += calculate_mae(outputs, targets).item()

        average_loss_train = running_loss / len(train_dataloader)
        average_mae_train = running_mae_train / len(train_dataloader)

        # Bucle de validación
        model.eval()
        running_mae_val = 0.0

        with torch.no_grad():
            for inputs, targets in tqdm(val_dataloader, desc=f"val epoch {epoch+1}"):
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                running_mae_val += calculate_mae(outputs, targets).item()

        average_mae_val = running_mae_val / len(val_dataloader)

        # Guardar el modelo si el MAE en validación es mejor que el mejor MAE hasta ahora
        if average_mae_val < best_mae:
            best_mae = average_mae_val
            torch.save(model.state_dict(), 'weights/best.pth')  # Guardar el modelo

        # Imprimir métricas
        print(f"Epoch {epoch + 1}/{epochs} -> "
              f"Train Loss: {average_loss_train:.4f}, Train MAE: {average_mae_train:.4f}, "
              f"Val MAE: {average_mae_val:.4f}, Best Val MAE: {best_mae:.4f}")
