#Lab1 A* algorithm

from math import sqrt

class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent # set parent node
        self.position = position # (row, column)

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        # comparison of nodes
        return self.position == other.position


def load_maze(filename):
    """
    open maze file
    argument:
        filename: file name of maze
    return:
        n(the number of rows) dimensional array
    """
    with open(filename) as f:
        maze = [list(line.strip()) for line in f]
    return maze

def print_maze(maze):
    """
    print the maze on the console
    arguments:
        maze: n(the number of rows) dimensional array
    """
    for row in maze:
        print(''.join(row))

def find_start_end(maze):
    """
    fine start and end position
    arguments:
        maze: n(the number of rows) dimensional array
    return:
        start and end position as tuple
    """
    for i in range(len(maze)):
        for j in range(len(maze[0])):
            if maze[i][j] == 'S':
                start = (i, j)
            elif maze[i][j] == 'E':
                end = (i, j)
    return start, end

def manhattan_distance(a, b):
    """
    calculate the Manhattan distance heuristic
    arguments:
        a: current position
        b: end position
    return:
        heuristic distance
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def euclidean_distance(a, b):
    """
    calculate the Euclidean distance heuristic
    arguments:
        a: current position
        b: end position
    return:
        heuristic distance
    """
    return sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def astar(maze, start, end, heuristic):
    """
    find the shortest path between start and end position by using A* algorithm and visualize the step of searching
    arguments:
        maze: n(the number of rows) dimensional array
        start: start position
        end: end position
        heuristic: heuristic type
    return:
        a list of tuples as a path from the given start to the given end
    """

    # initialize the start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = [] # collection of all gathered nodes
    closed_list = [] # collection of all expanded nodes
    # Add the start node to open list
    open_list.append(start_node)

    # Loop until you find the end
    while len(open_list) > 0:

        # Get the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            # select the node which has minimum f value
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # when it found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1] # Return reversed path

        # when it hasn't found the goal, Generate children
        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]: # searching in 4 directions

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] == '#':
                continue
            if maze[node_position[0]][node_position[1]] == 'S':
                continue


            # Create new node
            new_node = Node(current_node, node_position)

            # if maze[new_node.position[0]][new_node.position[1]] != 'E':
            #     maze[new_node.position[0]][new_node.position[1]] = '.'
            #     print_maze(maze)
            #     #print()
            
            # Append to the children list
            children.append(new_node)

        # Loop through children and calculate G, H, F value for each nodes
        for child in children:

            # Child is on the closed list
            if len([closed_child for closed_child in closed_list if closed_child == child]) > 0:
                continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = heuristic(child.position, end_node.position)
            # F = G + H
            child.f = child.g + child.h

            # check if child is already in the open list
            if len([open_node for open_node in open_list if child.position == open_node.position and child.g > open_node.g]) > 0:
                continue

            # Add the child to the open list
            open_list.append(child)
       
        maze1=maze
        path_1=[]
        current = current_node
        while current is not None:
            path_1.append(current.position)
            current = current.parent
            print(current)
        for step in path_1:
            if step != start and step != end:
                maze1[step[0]][step[1]] = '.'
        print_maze(maze1)

if __name__ == '__main__':
    maze = load_maze('maze2.txt')
    print('Maze:')
    print('')
    print_maze(maze)
    start, end = find_start_end(maze)
    print(start, end)

    print('\nSolving with Euclidean distance heuristic:')
    path1 = astar(maze, start, end, euclidean_distance)
    # visualization of the final path
    for step in path1:
        if step != start and step != end:
            maze[step[0]][step[1]] = '*'

    print_maze(maze)

    #Reset maze
    maze = load_maze('maze2.txt')

    print('\nSolving with Manhattan distance heuristic:')
    path2 = astar(maze, start, end, manhattan_distance)
    # visualization of the final path
    for step in path2:
        if step != start and step != end:
            maze[step[0]][step[1]] = '*'

    print_maze(maze)