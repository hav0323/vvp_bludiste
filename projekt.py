import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import random


class Maze:
    '''
    Stucture for creating and loading mazes.
    Parameters:
    type - "create" or "load" type of initialization,
    size - size of the maze,
    template_type - type of the maze either "empty", "slalom", "smiley", "cross" or "x",
    file - file to load the maze from
    '''

    def __init__(self, type: str = "create", size: int = None, template_type: str = "empty", file: str = None):
        if type == "create" and size is not None:
            self.size = size
            self.start = (0, 0)
            self.end = (size - 1, size - 1)
            self.template_type = template_type
            self.maze = self.create_maze(size)
        elif type == "load" and file is not None:
            self.load_maze(file)
        else:
            raise ValueError("Invalid parameters for Maze initialization")

    def load_maze(self, file):
        '''
        Load maze from file.
        Parameters:
        file - file to load the maze from
        '''
        data = np.genfromtxt(file, delimiter=',')
        self.size = data.shape[0]
        self.maze = data == 1

    def maze_picture(self, path):
        '''
        Display maze with path.
        Parameters:
        path - list of cells in the path
        '''
        maze_vis = np.zeros_like(self.maze, dtype=int)
        maze_vis[self.maze] = 1
        for cell in path:
            maze_vis[cell] = 2
        cmap = colors.ListedColormap(['white', 'black', 'red'])
        plt.imshow(maze_vis, cmap=cmap)
        plt.show()

    def neighbors(self, maze: np.ndarray, node: tuple):
        '''
        Support function for shortest_path and path_exists. Which returns accesible neighbors of the node.
        Parameters:
        maze - maze to search in,
        node - node to search neighbors for,
        return - list of accessible neighbors
        '''
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        result = []
        for dx, dy in directions:
            x, y = node[0] + dx, node[1] + dy
            if 0 <= x < maze.shape[0] and 0 <= y < maze.shape[1] and not maze[x, y]:
                result.append((x, y))
        return result

    def shortest_path(self):
        '''
        Function to find the shortest path in the maze.
        return - list of cells in the path  
        '''
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
        '''
        Function to check if path exists in the maze.
        Parameters:
        maze - maze to search in,
        return - True if path exists, False otherwise
        '''
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

    def filling(self, maze: np.ndarray):
        '''
        Function to fill the maze with walls until path exists.
        Parameters:
        maze - maze to fill,
        return - filled maze
        '''
        control = 0
        while control < 5:
            if self.path_exists(maze):
                x, y = random.randint(
                    0, self.size - 1), random.randint(0, self.size - 1)
                if (x, y) != self.start and (x, y) != self.end:
                    maze[x, y] = True
            else:
                maze[x, y] = False
                control += 1
        return maze

    def create_slalom(self):
        '''
        Function to create slalom in a matrix as a template for a maze
        return - maze
        '''
        path_size = self.size // 10
        wall_size = 2
        maze = np.zeros((self.size, self.size), dtype=bool)
        velikost_bloku = (wall_size+path_size)*2
        pocet_cyklu = (self.size-1) // velikost_bloku
        for i in range(pocet_cyklu):
            pozice = i*velikost_bloku
            maze[pozice+path_size:pozice+path_size +
                 wall_size, :self.size-path_size] = 1
            maze[pozice+path_size*2+wall_size:pozice +
                 path_size*2+wall_size*2, path_size:] = 1
        return maze

    def create_smiley(self):
        '''
        Function to create smiley in a matrix as a template for a maze
        return - maze
        '''
        maze = np.zeros((self.size, self.size), dtype=bool)
        eye_size = self.size // 10
        eye_y = self.size // 3
        eye_x_left = self.size // 3
        eye_x_right = 2 * self.size // 3
        mouth_y = 2 * self.size // 3
        mouth_x_left = self.size // 4
        mouth_x_right = 3 * self.size // 4
        mouth_thickness = self.size // 20
        maze[eye_y-eye_size:eye_y+eye_size,
             eye_x_left-eye_size:eye_x_left+eye_size] = 1
        maze[eye_y-eye_size:eye_y+eye_size,
             eye_x_right-eye_size:eye_x_right+eye_size] = 1
        for i in range(mouth_thickness):
            mouth_x = np.arange(mouth_x_left, mouth_x_right)
            mouth_y_temp = mouth_y + i + \
                (self.size // 10) * np.sin((mouth_x - mouth_x_left)
                                           * np.pi / (mouth_x_right - mouth_x_left))
            mouth_y_temp = mouth_y_temp.astype(int)
            for x, y in zip(mouth_x, mouth_y_temp):
                maze[y: y + mouth_thickness, x] = 1
        return maze

    def create_cross(self):
        '''
        Function to create cross in a matrix as a template for a maze
        return - maze
        '''
        maze = np.zeros((self.size, self.size), dtype=bool)
        maze[self.size//2, 1:self.size-1] = 1
        maze[1:self.size-1, self.size//2] = 1
        return maze

    def create_x(self):
        maze = np.zeros((self.size, self.size), dtype=bool)
        for i in range(1, self.size-1):
            maze[i, i] = 1
            maze[self.size-i-1, i] = 1
        return maze

    def create_maze(self, size: int, template_type: str = "empty"):
        '''
        Function to create a maze.
        return - maze
        '''
        if self.template_type == 'empty':
            maze = np.zeros((self.size, self.size), dtype=bool)
        if self.template_type == 'slalom':
            maze = self.create_slalom()
        if self.template_type == "smiley":
            maze = self.create_smiley()
        if self.template_type == 'cross':
            maze = self.create_cross()
        if self.template_type == 'x':
            maze = self.create_x()
        self.maze = self.filling(maze)
        return self.maze
