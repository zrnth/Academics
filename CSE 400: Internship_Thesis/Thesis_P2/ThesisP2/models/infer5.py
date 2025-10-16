import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import argparse
from train5 import DenseNet, DenseLayer, DenseBlock, TransitionLayer


def process_image(image_path, model, device, transform):
    try:
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            # Get intermediate feature maps
            features = []
            x = input_tensor

            # Extract features from dense blocks
            x = model.features[0:4](x)  # Initial conv + bn + relu + maxpool
            features.append(x)

            current_idx = 4
            for i in range(len(model.features) - 4):  # Skip final bn and relu
                x = model.features[current_idx + i](x)
                if isinstance(model.features[current_idx + i], DenseBlock):
                    features.append(x)

            # Get final prediction
            output = model(input_tensor)

            # Create attention map
            attention_map = torch.zeros(48, 48)
            for feat in features:
                # Upsample and aggregate feature maps
                feat_mean = feat.mean(dim=1, keepdim=True)
                upsampled = torch.nn.functional.interpolate(
                    feat_mean, size=(48, 48), mode='bilinear', align_corners=True
                )
                attention_map += upsampled.squeeze().cpu()

            # Normalize attention map
            attention_map = (attention_map - attention_map.min()) / \
                (attention_map.max() - attention_map.min())
            prob_map = attention_map.numpy()

            # Calculate overall probability
            idc_probability = float(np.mean(prob_map > 0.5) * 100)

            return {
                'probability': idc_probability,
                'prediction_map': prob_map,
                'has_idc': idc_probability > 5
            }
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None


def visualize_prediction(image_path, prediction_map, save_path=None):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((48, 48))
    image_array = np.array(image)

    heatmap = plt.cm.jet(prediction_map)[:, :, :3]
    alpha = 0.4
    blended = (1 - alpha) * image_array / 255.0 + alpha * heatmap
    blended = np.clip(blended, 0, 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(131)
    plt.imshow(image_array)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(prediction_map, cmap='jet')
    plt.title('IDC Prediction Map')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(blended)
    plt.title('Overlay')
    plt.axis('off')

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def visualize_predictions(results, save_dir):
    num_images = len(results)
    images_per_plot = 10
    num_plots = (num_images + images_per_plot - 1) // images_per_plot

    for plot_idx in range(num_plots):
        start_idx = plot_idx * images_per_plot
        end_idx = min(start_idx + images_per_plot, num_images)
        current_batch = results[start_idx:end_idx]
        num_current_images = len(current_batch)

        fig = plt.figure(figsize=(15, 3 * num_current_images))

        for idx, result in enumerate(current_batch):
            image = Image.open(result['image']).convert('RGB')
            image = image.resize((48, 48))
            image_array = np.array(image)

            prediction_map = result['prediction_map']
            heatmap = plt.cm.jet(prediction_map)[:, :, :3]

            alpha = 0.4
            blended = (1 - alpha) * image_array / 255.0 + alpha * heatmap
            blended = np.clip(blended, 0, 1)

            plt.subplot(num_current_images, 3, idx * 3 + 1)
            plt.imshow(image_array)
            plt.title('Original Image')
            plt.axis('off')

            plt.subplot(num_current_images, 3, idx * 3 + 2)
            plt.imshow(prediction_map, cmap='jet')
            plt.colorbar(label='IDC Probability')
            plt.title('IDC Prediction Map')
            plt.axis('off')

            plt.subplot(num_current_images, 3, idx * 3 + 3)
            plt.imshow(blended)
            prediction_text = f"IDC: {'Yes' if result['has_idc'] else 'No'}\n"
            prediction_text += f"Confidence: {result['probability']:.1f}%"
            plt.title(prediction_text,
                      color='red' if result['has_idc'] else 'green')
            plt.axis('off')

            plt.figtext(0.01, 1 - (idx + 0.5) / num_current_images,
                        os.path.basename(result['image']),
                        fontsize=8)

        plt.tight_layout()
        save_path = os.path.join(
            save_dir, f'predictions_batch_{plot_idx + 1}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()


def process_and_visualize(image_paths, model, device, transform, save_dir, threshold=5.0):
    results = []

    for image_path in tqdm(image_paths, desc="Processing images"):
        try:
            result = process_image(image_path, model, device, transform)
            if result is not None:
                result['file'] = os.path.basename(image_path)
                result['image'] = image_path
                results.append(result)
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue

    if results:
        visualize_predictions(results, save_dir)

        results_file = os.path.join(save_dir, 'results.txt')
        with open(results_file, 'w') as f:
            f.write("IDC Detection Results\n")
            f.write("===================\n\n")
            for r in results:
                f.write(f"File: {r['file']}\n")
                f.write(f"IDC Prediction: {
                        'Positive' if r['has_idc'] else 'Negative'}\n")
                f.write(f"Probability: {r['probability']:.2f}%\n")
                f.write("-------------------\n")

        print("\nProcessing Complete!")
        print(f"Total images processed: {len(results)}")
        positive_cases = sum(1 for r in results if r['has_idc'])
        print(f"IDC Positive cases: {positive_cases} ({
              positive_cases/len(results)*100:.1f}%)")
        print(f"Results saved to: {results_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='IDC Detection Inference using DenseNet')
    parser.add_argument('--model_path', type=str, default='best_model.pth',
                        help='Path to trained model')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to input image or directory')
    parser.add_argument('--output_dir', type=str, default="./prediction5",
                        help='Directory to save prediction5')
    parser.add_argument('--threshold', type=float, default=5.0,
                        help='IDC probability threshold (%)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading model...")
    model = DenseNet(growth_rate=32, block_config=(6, 12, 24, 16)).to(device)
    model.load_state_dict(torch.load(
        args.model_path, map_location=device, weights_only=True))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    if os.path.isfile(args.input_path):
        image_paths = [args.input_path]
    else:
        image_paths = [os.path.join(args.input_path, f) for f in os.listdir(args.input_path)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_paths:
        print("No valid image files found.")
        return

    process_and_visualize(image_paths, model, device,
                          transform, args.output_dir, args.threshold)


if __name__ == '__main__':
    main()
