import pygame
import torch 
import torch.nn as nn
import numpy as np
from NN.CNN import CNN

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)

# Define brush size and canvas size
resolution = 28
scale = 20
dim_brush = 10

# Draw a grid with spacing equal to scale representing the 28x28 pixel grid.
def draw_grid():
    for x in range(0, scaled_resolution, scale):
        pygame.draw.line(screen, (GRAY), (x, 0), (x, scaled_resolution))
    for y in range(0, scaled_resolution, scale):
        pygame.draw.line(screen, (GRAY), (0, y), (scaled_resolution, y))

def main():
    global screen, scaled_resolution, scale
    pygame.init()
    scaled_resolution = resolution * scale
    screen = pygame.display.set_mode((scaled_resolution, scaled_resolution))
    pygame.display.set_caption("Draw a digit (0-9) and press Enter to predict")
    screen.fill((BLACK))
    draw_grid()
    pygame.display.flip()

    # Load the trained model
    model = CNN()
    try:
        model.load_state_dict(torch.load('mnist_cnn.pth', map_location=torch.device('cpu')))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
    model.eval()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEMOTION:
                if pygame.mouse.get_pressed()[0]:  # Left mouse button is held down
                    x, y = event.pos
                    pygame.draw.circle(screen, (WHITE), (x, y), dim_brush)
                    pygame.display.flip()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:  # Enter key to predict
                    # Capture the drawing area
                    data = pygame.surfarray.array3d(screen)
                    data = np.dot(data[..., :3], [0.2989, 0.5870, 0.1140])  # Convert to grayscale
                    data = np.flipud(data)  # Flip vertically to match the drawing orientation
                    data = data[::scale, ::scale]  # Downsample to 28x28

                    # Normalize and reshape for the model
                    data = data / 255.0
                    data = data.reshape(1, 1, resolution, resolution)
                    data_tensor = torch.tensor(data, dtype=torch.float32)

                    # Predict using the model
                    with torch.no_grad():
                        output = model(data_tensor)
                        _, predicted = torch.max(output.data, 1)
                        print(f'Predicted Digit: {predicted.item()}')

                elif event.key == pygame.K_c:  # 'c' key to clear the screen
                    screen.fill((BLACK))
                    draw_grid()
                    pygame.display.flip()

    pygame.quit()

main()