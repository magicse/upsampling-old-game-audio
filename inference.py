import os
os.add_dll_directory('C:/ffmpeg/bin/')  
import torch

from claude_upsampling import AudioSuperResolutionNet, process_long_audio, process_long_audio_test

UPSCALE_FACTOR = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


INPUT_AUDIO_PATH = 'test_inputs/input_audio.wav'     
OUTPUT_AUDIO_PATH = 'results/output_sr_audio.wav'    
MODEL_PATH = 'models/checkpoint_epoch_30_fin.pth'              

def main():
    print(f"Using device: {DEVICE}")


    model = AudioSuperResolutionNet(upscale_factor=UPSCALE_FACTOR, num_blocks=4)
    #model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])    


    process_long_audio(model, INPUT_AUDIO_PATH, OUTPUT_AUDIO_PATH, upscale_factor=UPSCALE_FACTOR, device=DEVICE)

if __name__ == '__main__':
    main()
