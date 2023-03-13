#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 16:49:18 2023

@author: sotarosuzuki
"""
import heapq
from math import sqrt

def load_maze(filename):
    with open(filename) as f:
        maze = [list(line.strip()) for line in f]
    return maze

def print_maze(maze):
    for row in maze:
        print(''.join(row))

def find_start_end(maze):
    for i in range(len(maze)):
        for j in range(len(maze[0])):
            if maze[i][j] == 'S':
                start = (i, j)
            elif maze[i][j] == 'E':
                end = (i, j)
    return start, end

def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def euclidean_distance(a, b):
    return sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def astar(maze, start, end, heuristic):
    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0
    
    while len(frontier) > 0:
        current = heapq.heappop(frontier)[1]
        
        if current == end:
            break
        
        for next in get_neighbors(maze,current):
            new_cost = cost_so_far[current] + 1
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(next,end)
                heapq.heappush(frontier,(priority,next))
                came_from[next]=current
                
                # Visualization
                if next != end:
                    maze[next[0]][next[1]]='.'
                    print_maze(maze)
                    print()
    
    return came_from,cost_so_far


def get_neighbors(maze,current):
    
     # Get all possible neighbors
     neighbors=[(current[0]-1,current[1]),(current[0]+1,current[1]),(current[0],current[1]-1),(current[0],current[1]+1)]
     
     # Filter out invalid neighbors
     valid_neighbors=[]
     for neighbor in neighbors:
         if neighbor[0]>=0 and neighbor[0]<len(maze) and neighbor[1]>=0 and neighbor[1]<len(maze[0]):
             if maze[neighbor[0]][neighbor[1]]!='#':
                 valid_neighbors.append(neighbor)
     
     return valid_neighbors

def reconstruct_path(came_from,start,end):
    current=end
    path=[]
    while current!=start:
        path.append(current)
        current=came_from[current]
    path.append(start)
    path.reverse()
    return path


def main():
    maze = load_maze('maze.txt')
    print('Maze:')
    print_maze(maze)
    
    start, end = find_start_end(maze)
    
    print('\nSolving with Manhattan distance heuristic:')
    came_from, cost_so_far = astar(maze, start, end, manhattan_distance)
    
    path = reconstruct_path(came_from,start,end)
    
    for step in path:
        if step != start and step != end:
            maze[step[0]][step[1]]='*'
            
    print_maze(maze)
    
    # Reset maze
    maze = load_maze('maze.txt')
    
    print('\nSolving with Euclidean distance heuristic:')
    came_from,cost_so_far=astar(maze,start,end,euclidean_distance)
    
    path=reconstruct_path(came_from,start,end)
    
    for step in path:
        if step != start and step != end:
            maze[step[0]][step[1]]='*'
            
    print_maze(maze)

if __name__ == '__main__':
     main()                      

                          

                          