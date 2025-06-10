import os
os.add_dll_directory('C:/ffmpeg/bin/')  # если требуется ffmpeg
import torch
from torch.utils.data import DataLoader
from claude_upsampling import AudioSuperResolutionNet, AudioDataset, train_model
import matplotlib.pyplot as plt
from tqdm import tqdm

num_workers = min(2, os.cpu_count() or 1)

# Параметры
UPSCALE_FACTOR = 4
BATCH_SIZE = 24
NUM_EPOCHS = 100
SR_RATE = 44100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Пути к данным
TRAIN_AUDIO_DIR = 'audio/train'
VAL_AUDIO_DIR = 'audio/val'

# Папка для сохранения моделей
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

def save_model(model, epoch, train_losses, val_losses, val_snrs, best_snr=None):
    """Сохранение модели и метрик"""
    
    # Основное сохранение модели
    model_path = os.path.join(MODELS_DIR, f'audio_sr_model_epoch_{epoch}.pth')
    
    # Сохраняем состояние модели и дополнительную информацию
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'upscale_factor': UPSCALE_FACTOR,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_snrs': val_snrs,
        'best_snr': best_snr
    }
    
    torch.save(checkpoint, model_path)
    print(f"Модель сохранена: {model_path}")
    
    # Также сохраняем только веса модели (более легкий файл)
    weights_path = os.path.join(MODELS_DIR, f'audio_sr_weights_epoch_{epoch}.pth')
    torch.save(model.state_dict(), weights_path)
    
    return model_path

def load_model(model_path, device='cpu'):
    """Загрузка сохраненной модели"""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Создаем модель с теми же параметрами
    model = AudioSuperResolutionNet(
        upscale_factor=checkpoint['upscale_factor'], 
        num_blocks=4
    )
    
    # Загружаем веса
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"Модель загружена с эпохи {checkpoint['epoch']}")
    print(f"Лучший SNR: {checkpoint.get('best_snr', 'N/A')} dB")
    
    return model, checkpoint

def train_model_with_progress(model, train_loader, val_loader, num_epochs, device):
    """Обучение модели с прогресс-баром"""
    import torch.nn as nn
    import torch.optim as optim
    
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    val_losses = []
    val_snrs = []
    
    # Основной прогресс-бар для эпох
    epoch_pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")
    
    for epoch in epoch_pbar:
        # Обучение
        model.train()
        train_loss = 0.0
        
        # Прогресс-бар для батчей обучения
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Train", 
                         leave=False, unit="batch")
        
        for lr_audio, hr_audio in train_pbar:
            lr_audio, hr_audio = lr_audio.to(device), hr_audio.to(device)
            
            optimizer.zero_grad()
            outputs = model(lr_audio)
            loss = criterion(outputs, hr_audio)
            loss.backward()
            optimizer.step()
            
            batch_loss = loss.item()
            train_loss += batch_loss
            
            # Обновляем описание батч-прогресса
            train_pbar.set_postfix({'Loss': f'{batch_loss:.6f}'})
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Валидация
        model.eval()
        val_loss = 0.0
        total_snr = 0.0
        
        with torch.no_grad():
            # Прогресс-бар для валидации
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Val", 
                           leave=False, unit="batch")
            
            for lr_audio, hr_audio in val_pbar:
                lr_audio, hr_audio = lr_audio.to(device), hr_audio.to(device)
                
                outputs = model(lr_audio)
                loss = criterion(outputs, hr_audio)
                val_loss += loss.item()
                
                # Расчет SNR
                mse = torch.mean((outputs - hr_audio) ** 2)
                snr = 10 * torch.log10(torch.mean(hr_audio ** 2) / (mse + 1e-8))
                total_snr += snr.item()
                
                val_pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
        
        val_loss /= len(val_loader)
        avg_snr = total_snr / len(val_loader)
        
        val_losses.append(val_loss)
        val_snrs.append(avg_snr)
        
        # Обновляем основной прогресс-бар
        epoch_pbar.set_postfix({
            'Train Loss': f'{train_loss:.6f}',
            'Val Loss': f'{val_loss:.6f}',
            'SNR': f'{avg_snr:.2f}dB'
        })
        
        # Сохранение промежуточных результатов каждые 10 эпох
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(MODELS_DIR, f'checkpoint_epoch_{epoch+1}.pth')
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'upscale_factor': UPSCALE_FACTOR,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_snrs': val_snrs,
            }
            torch.save(checkpoint, checkpoint_path)
            tqdm.write(f"Checkpoint saved: {checkpoint_path}")
    
    epoch_pbar.close()
    return train_losses, val_losses, val_snrs

def main():
    print(f"Using device: {DEVICE}")
    print(f"Training for {NUM_EPOCHS} epochs with batch size {BATCH_SIZE}")
    
    # Создание модели
    model = AudioSuperResolutionNet(upscale_factor=UPSCALE_FACTOR, num_blocks=4)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Датасеты
    print("Loading datasets...")
    train_dataset = AudioDataset(TRAIN_AUDIO_DIR, upscale_factor=UPSCALE_FACTOR, sample_rate=SR_RATE)
    val_dataset = AudioDataset(VAL_AUDIO_DIR, upscale_factor=UPSCALE_FACTOR, sample_rate=SR_RATE)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Тренировка с прогресс-баром
    print("\nStarting training...")
    train_losses, val_losses, val_snrs = train_model_with_progress(
        model, train_loader, val_loader, NUM_EPOCHS, DEVICE
    )
    
    # Находим лучший SNR
    best_snr = max(val_snrs) if val_snrs else None
    
    # СОХРАНЕНИЕ МОДЕЛИ после обучения
    final_model_path = save_model(
        model, NUM_EPOCHS, train_losses, val_losses, val_snrs, best_snr
    )
    
    # Дополнительно сохраняем лучшую модель
    if best_snr:
        best_epoch = val_snrs.index(best_snr) + 1
        best_model_path = os.path.join(MODELS_DIR, 'best_audio_sr_model.pth')
        
        checkpoint = {
            'epoch': best_epoch,
            'model_state_dict': model.state_dict(),
            'upscale_factor': UPSCALE_FACTOR,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_snrs': val_snrs,
            'best_snr': best_snr
        }
        
        torch.save(checkpoint, best_model_path)
        print(f"Лучшая модель сохранена: {best_model_path}")
    
    # Графики
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')
    
    plt.subplot(1, 3, 2)
    plt.plot(val_snrs)
    plt.xlabel('Epoch')
    plt.ylabel('SNR (dB)')
    plt.title('Validation SNR')
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()
    
    print(f"\nОбучение завершено!")
    print(f"Финальная модель: {final_model_path}")
    print(f"Графики сохранены: training_curves.png")

# Пример использования сохраненной модели
def test_saved_model():
    """Пример загрузки и использования сохраненной модели"""
    model_path = os.path.join(MODELS_DIR, 'best_audio_sr_model.pth')
    
    if os.path.exists(model_path):
        model, checkpoint = load_model(model_path, DEVICE)
        model.eval()
        print("Модель готова к использованию!")
        return model
    else:
        print("Сохраненная модель не найдена")
        return None

if __name__ == '__main__':
    main()
    
    # Опционально: тест загрузки модели
    # test_saved_model()