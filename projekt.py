import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import random
from scipy.sparse import lil_matrix


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
            self.template_type = template_type
            self.start = (0, 0)
            self.end = (size - 1, size - 1)
            self.maze = self.create_maze()
        elif type == "load" and file is not None:
            self.load_maze(file)
        else:
            raise ValueError("Invalid parameters for Maze initialization")

    def create_incident_matrix(self, maze):
        '''
        Function to create incident matrix from maze.
        Parameters:
        maze - maze to create incident matrix from,
        return - incident matrix
        '''
        size = len(maze)
        matrix = lil_matrix((size*size, size*size), dtype=int)
        for i in range(size):
            for j in range(size):
                if maze[i][j] == 0:
                    cell_index = i * size + j
                    for dx, dy in [(1, 0), (0, 1)]:
                        nx, ny = i + dx, j + dy
                        if 0 <= nx < size and 0 <= ny < size and maze[nx][ny] == 0:
                            neighbor_index = nx * size + ny
                            matrix[cell_index, neighbor_index] = 1
                            matrix[neighbor_index, cell_index] = 1
        return matrix

    def get_accessible_neighbors(self, cell):
        '''
        Function to get accessible neighbors of a cell using incident matrix.
        Parameters:
        cell - cell to get neighbors of,
        return - list of accessible neighbors
        '''
        indexes = self.incident_matrix.rows[cell[0] * self.size + cell[1]]
        return [(index // self.size, index % self.size) for index in indexes]

    def load_maze(self, file):
        '''
        Load maze from file.
        Parameters:
        file - file to load the maze from
        '''
        data = np.genfromtxt(file, delimiter=',')
        self.size = data.shape[0]
        self.maze = data == 1
        self.start = (0, 0)
        self.end = (self.size - 1, self.size - 1)
        self.incident_matrix = self.create_incident_matrix(self.maze)

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

    def shortest_path(self):
        '''
        Function to find the shortest path in the maze.
        return - list of cells in the path  
        '''
        queue = [(0, self.start)]
        distances = {self.start: 0}
        paths = {self.start: []}
        while queue:
            current_distance, current_node = min(queue, key=lambda x: x[0])
            queue.remove((current_distance, current_node))
            if current_node == self.end:
                return paths[current_node] + [self.end]
            for neighbor in self.get_accessible_neighbors(current_node):
                distance = current_distance + 1
                if neighbor not in distances or distance < distances[neighbor]:
                    distances[neighbor] = distance
                    paths[neighbor] = paths[current_node] + [current_node]
                    queue.append((distance, neighbor))
        return None

    def path_exists(self):
        '''
        Function to check if path exists in the maze.
        Parameters:
        maze - maze to search in,
        return - True if path exists, False otherwise
        '''
        queue = [self.start]
        visited = set([self.start])
        while queue:
            current_node = queue.pop(0)
            if current_node == self.end:
                return True
            for neighbor in self.get_accessible_neighbors(current_node):
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
            if self.path_exists():
                while True:
                    x, y = random.randint(
                        0, self.size - 1), random.randint(0, self.size - 1)
                    if (x, y) != self.start and (x, y) != self.end and maze[x, y] == False:
                        break
                maze[x, y] = True
                indexes = tuple(self.incident_matrix.rows[x*self.size + y])
                self.incident_matrix[x*self.size + y, indexes] = 0
                self.incident_matrix[indexes, x*self.size + y] = 0

            else:
                maze[x, y] = False
                self.incident_matrix[x*self.size + y, indexes] = 1
                self.incident_matrix[indexes, x*self.size + y] = 1
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

    def create_maze(self):
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
        self.incident_matrix = self.create_incident_matrix(maze)
        self.maze = self.filling(maze)
        return self.maze
