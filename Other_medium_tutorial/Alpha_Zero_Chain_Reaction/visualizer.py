import pygame

class Visualizer:
    def __init__(self, height=8, width=8, cell_edge=50) -> None:
        self.height = height
        self.width = width
        self.cell_edge = cell_edge

        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Chain Reaction")

        self.R = 15
    
    def draw_grid(self, color):
        for row in self.height:
            for col in self.width:
                pygame.draw.rect(
                    self.display, 
                    color,
                    ((col * self.cell_edge, row*self.cell_edge), (self.cell_edge, self.cell_edge)),
                    1
                )
    
    def gameloop(self, state_view):
        self.display.fill(0)

        if state_view['player_turn'] == 'red':
            self.draw_grid((255, 0, 0))
        else:
            self.draw_grid((0, 255, 0))
        
        for row in range(self.height):
            for col in range(self.width):
                matrix = state_view['array_view']

                if matrix[row][col] > 0:
                    color = (255, 0, 0)
                elif matrix[row][col] < 0:
                    color = (0, 255, 0)
                else:
                    continue
                
                # Get center of the cell
                cX = col * self.cell_edge + self.cell_edge//2
                cY = row * self.cell_edge + self.cell_edge//2

                if abs(matrix[row][col]) == 1:
                    pygame.draw.circle(self.display, color, (cX, cY), self.R)
                elif abs(matrix[row][col]) == 2:
                    pygame.draw.circle(self.display, color, (cX - self.R//2, cY), self.R)
                    pygame.draw.circle(self.display, color, (cX + self.R//2, cY), self.R)
                elif abs(matrix[row][col]) == 3:
                    pygame.draw.circle(self.display, color, (cX, cY-self.R//2), self.R)
                    pygame.draw.circle(self.display, color, (cX - self.R//2, cY + self.R//2), self.R)
                    pygame.draw.circle(self.display, color, (cX + self.R//2, cY + self.R//2), self.R)
        pygame.display.update()
        
