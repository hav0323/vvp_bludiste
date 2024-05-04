import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.sparse import lil_matrix
import random


class Maze:
    def __init__(self, size: int = None, template_type: str = "empty"):
        if size is not None:
            self.size = size
            self.maze = self.create_maze(size, template_type)

    def load_maze(self, file):
        data = np.genfromtxt(file, delimiter=',')
        self.size = data.shape[0]
        self.maze = data == 1

    def maze_picture(self, path):
        maze_vis = np.zeros_like(self.maze, dtype=int)
        maze_vis[self.maze] = 1  # Set the walls to 1
        for cell in path:
            maze_vis[cell] = 2  # Set the cells in the path to 2
        cmap = colors.ListedColormap(['white', 'black', 'red'])
        plt.imshow(maze_vis, cmap=cmap)
        plt.show()

    def IncidentMatrix(self):
        n = self.maze.shape[0]
        matrix = lil_matrix((n*n, n*n))
        for i in range(n):
            for j in range(n):
                if self.maze[i][j] == False:
                    if i > 0 and self.maze[i-1:i, j:j+1] == False:
                        matrix[n*i+j, n*(i-1)+j] = 1
                        matrix[n*(i-1)+j, n*i+j] = 1
                    if j > 0 and self.maze[i:i+1, j-1:j] == False:
                        matrix[n*i+j, n*i+(j-1)] = 1
                        matrix[n*i+(j-1), n*i+j] = 1
        return matrix

    @staticmethod
    def neighbors(maze: np.ndarray, node: tuple):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        result = []
        for dx, dy in directions:
            x, y = node[0] + dx, node[1] + dy
            if 0 <= x < maze.shape[0] and 0 <= y < maze.shape[1] and not maze[x, y]:
                result.append((x, y))
        return result

    def shortest_path(self):
        start = (0, 0)
        end = (self.size - 1, self.size - 1)
        queue = [(0, start)]
        distances = {start: 0}
        paths = {start: []}
        while queue:
            current_distance, current_node = min(queue, key=lambda x: x[0])
            queue.remove((current_distance, current_node))
            if current_node == end:
                return paths[current_node] + [end]
            for neighbor in self.neighbors(self.maze, current_node):
                distance = current_distance + 1
                if neighbor not in distances or distance < distances[neighbor]:
                    distances[neighbor] = distance
                    paths[neighbor] = paths[current_node] + [current_node]
                    queue.append((distance, neighbor))
        return None

    def path_exists(self, maze: np.ndarray):
        start = (0, 0)
        end = (self.size - 1, self.size - 1)
        queue = [start]
        visited = set([start])
        while queue:
            current_node = queue.pop(0)
            if current_node == end:
                return True
            for neighbor in self.neighbors(maze, current_node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return False

    def create_maze(self, size: int, template_type: str = "empty"):
        start = (0, 0)
        end = (size-1, size - 1)
        if template_type == 'empty':
            maze = np.zeros((size, size), dtype=bool)
            control = 0
            while control < 5:
                if self.path_exists(maze):
                    x, y = random.randint(
                        0, size - 1), random.randint(0, size - 1)
                    if (x, y) != start and (x, y) != end:
                        maze[x, y] = True
                else:
                    maze[x, y] = False
                    control += 1
            maze[start] = False
            maze[end] = False
            self.maze = maze
            return self.maze
        if template_type == 'slalom':
            path_size = size // 10
            wall_size = 2
            maze = np.zeros((size, size), dtype=bool)
            velikost_bloku = (wall_size+path_size)*2
            pocet_cyklu = (size-1) // velikost_bloku
            for i in range(pocet_cyklu):
                pozice = i*velikost_bloku
                maze[pozice+path_size:pozice+path_size +
                     wall_size, :size-path_size] = 1
                maze[pozice+path_size*2+wall_size:pozice +
                     path_size*2+wall_size*2, path_size:] = 1
            control = 0
            while control < 5:
                if self.path_exists(maze):
                    x, y = random.randint(
                        0, size - 1), random.randint(0, size - 1)
                    maze[x, y] = True
                else:
                    maze[x, y] = False
                    control += 1
            maze[start] = False
            maze[end] = False
            self.maze = maze
            return self.maze
        if template_type == "smiley":
            maze = np.zeros((size, size), dtype=bool)
            eye_size = size // 10
            eye_y = size // 3
            eye_x_left = size // 3
            eye_x_right = 2 * size // 3
            mouth_y = 2 * size // 3
            mouth_x_left = size // 4
            mouth_x_right = 3 * size // 4
            mouth_thickness = size // 20
            maze[eye_y-eye_size:eye_y+eye_size,
                 eye_x_left-eye_size:eye_x_left+eye_size] = 1
            maze[eye_y-eye_size:eye_y+eye_size,
                 eye_x_right-eye_size:eye_x_right+eye_size] = 1
            for i in range(mouth_thickness):
                mouth_x = np.arange(mouth_x_left, mouth_x_right)
                mouth_y_temp = mouth_y + i + \
                    (size // 10) * np.sin((mouth_x - mouth_x_left)
                                          * np.pi / (mouth_x_right - mouth_x_left))
                mouth_y_temp = mouth_y_temp.astype(int)
                for x, y in zip(mouth_x, mouth_y_temp):
                    maze[y: y + mouth_thickness, x] = 1
            control = 0
            while control < 5:
                if self.path_exists(maze):
                    x, y = random.randint(
                        0, size - 1), random.randint(0, size - 1)
                    maze[x, y] = True
                else:
                    maze[x, y] = False
                    control += 1
            maze[start] = False
            maze[end] = False
            self.maze = maze
            return self.maze
        if template_type == 'cross':
            maze = np.zeros((size, size), dtype=bool)
            maze[size//2, 1:size-1] = 1
            maze[1:size-1, size//2] = 1
            control = 0
            while control < 5:
                if self.path_exists(maze):
                    x, y = random.randint(
                        0, size - 1), random.randint(0, size - 1)
                    maze[x, y] = True
                else:
                    maze[x, y] = False
                    control += 1
            maze[start] = False
            maze[end] = False
            self.maze = maze
            return self.maze
        if template_type == 'x':
            maze = np.zeros((size, size), dtype=bool)
            for i in range(1, size-1):
                maze[i, i] = 1
                maze[size-i-1, i] = 1
            control = 0
            while control < 5:
                if self.path_exists(maze):
                    x, y = random.randint(
                        0, size - 1), random.randint(0, size - 1)
                    maze[x, y] = True
                else:
                    maze[x, y] = False
                    control += 1
            maze[start] = False
            maze[end] = False
            self.maze = maze
            return self.maze
