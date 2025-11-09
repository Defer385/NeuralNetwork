import pygame
import numpy as np

pygame.init()
screen = pygame.display.set_mode((280, 280))
pygame.display.set_caption("MNIST Draw")

canvas = np.zeros((28, 28), dtype=np.float32)
brush_size = 10

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s:  # Save on pressing 'S'
                # Flatten and save
                with open("drawing.txt", "w") as f:
                    for row in canvas:
                        f.write(" ".join(f"{pixel:.4f}" for pixel in row) + "\n")
                print("Saved to drawing.txt")
                running = False
    
    # Drawing logic
    if pygame.mouse.get_pressed()[0]:
        pos = pygame.mouse.get_pos()
        x, y = pos[0] // 10, pos[1] // 10  # Scale to 28x28
        if 0 <= x < 28 and 0 <= y < 28:
            canvas[y, x] = 1.0  # Set pixel
            # Draw visual feedback
            pygame.draw.rect(screen, (255, 255, 255), (x*10, y*10, 10, 10))
    
    pygame.display.flip()

pygame.quit()