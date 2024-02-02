import torch
from tqdm import tqdm

# Función para calcular el MAE
def calculate_mae(predictions, targets):
    return torch.mean(torch.abs(predictions - targets))


def train(epochs, model, train_dataloader, val_dataloader, optimizer, device, scheduler=None, save_every_n_epochs=5):
    history_train_loss = []
    history_train_mae = []
    history_val_loss = []
    history_val_mae = []

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
        history_train_loss.append(average_loss_train)
        history_train_mae.append(average_mae_train)

        # Bucle de validación
        model.eval()
        running_mae_val = 0.0
        running_loss_val = 0.0

        with torch.no_grad():
            for inputs, targets in tqdm(val_dataloader, desc=f"val epoch {epoch+1}"):
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss_val = torch.nn.MSELoss()(outputs, targets)
                running_loss_val += loss_val.item()
                running_mae_val += calculate_mae(outputs, targets).item()

        average_loss_val = running_loss_val / len(val_dataloader)
        average_mae_val = running_mae_val / len(val_dataloader)
        history_val_loss.append(average_loss_val)
        history_val_mae.append(average_mae_val)

        # Guardar el modelo cada n epochs y el último modelo entrenado
        if epoch % save_every_n_epochs == 0 or epoch == epochs - 1:
            # Guardar el modelo
            torch.save(model.state_dict(), f'weights/model_epoch_{epoch+1}.pth')

        # Imprimir métricas
        print(f"Epoch {epoch + 1}/{epochs} -> "
              f"Train Loss: {average_loss_train:.4f}, Train MAE: {average_mae_train:.4f}, "
              f"Val Loss: {average_loss_val:.4f}, Val MAE: {average_mae_val:.4f}")

    return history_train_loss, history_train_mae, history_val_loss, history_val_mae
