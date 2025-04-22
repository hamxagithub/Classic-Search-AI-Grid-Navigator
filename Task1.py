from typing import List, Tuple, Set
import heapq as pq
import matplotlib.pyplot as plt
import time
from matplotlib import animation
import numpy as np
import seaborn as sns

class SearchAlgorithm:
    dir = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    @staticmethod
    def get_neighbors(x: int, y: int, grid: List[List[str]], walls: set) -> List[Tuple[int, int]]:
        neigbor = []
        row= len(grid)
        col  = len(grid[0])

        for dx, dy in SearchAlgorithm.dir:
            nx = x + dx
            ny = y + dy

            if 0 <= nx < row and 0 <= ny < col  and (nx, ny) not in walls:
                neigbor.append((nx, ny))

        return neigbor

    @staticmethod
    def get_start_target(grid: List[List[str]]) -> Tuple[Tuple[int, int], Tuple[int, int], set]:

        s  = (-1, -1)
        t  = (-1, -1)

        walls = set()

        row  = len(grid)
        col  = len(grid[0])


        for i in range(row ):
            for j in range(col ):
                cell = grid[i][j]

                if cell == 's':
                    s  = (i, j)
                elif cell == 't':
                    t  = (i, j)
                elif cell == '-1':
                    walls.add((i, j))

        return s , t , walls

    @staticmethod
    def manhattan_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])


    def dfs(grid: List[List[str]]) -> Tuple[int, List[Tuple[int, int]]]:
            s, t, walls = SearchAlgorithm.get_start_target(grid)
            stack = [s]
            visited: Set[Tuple[int, int]] = set()
            root: dict[Tuple[int, int], Tuple[int, int] | None] = {s: None}

            while stack:
                current = stack.pop()
                if current == t:
                    path = []
                    while current:
                        path.append(current)
                        current = root[current]
                    return 1, path[::-1]

                visited.add(current)

                for neighbor in SearchAlgorithm.get_neighbors(current[0], current[1], grid, walls):
                    if neighbor not in visited and neighbor not in root:
                        stack.append(neighbor)
                        root[neighbor] = current

            return -1, []

    @staticmethod
    def bfs(grid: List[List[str]]) -> Tuple[int, List[Tuple[int, int]]]:
        s, t, walls = SearchAlgorithm.get_start_target(grid)
        queue: List[Tuple[Tuple[int, int], List[Tuple[int, int]]]] = [(s, [s])]
        visited: Set[Tuple[int, int]] = {s}

        while queue:
            current, path = queue.pop(0)
            if current == t:
                return 1, path

            for neighbor in SearchAlgorithm.get_neighbors(current[0], current[1], grid, walls):
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))
                    visited.add(neighbor)

        return -1, []

    @staticmethod
    def ucs(grid: List[List[str]]) -> Tuple[int, List[Tuple[int, int]]]:
        s, t, walls = SearchAlgorithm.get_start_target(grid)

        if grid[s[0]][s[1]] == '-1':
            print("Start is an obstacle!")
            return -1, []

        open_set = [(0, s)]
        root = {s: None}
        cost = {s: 0}
        visited = set()

        while open_set:
            c_cost, current = pq.heappop(open_set)

            if current in visited:
                continue

            visited.add(current)

            print("Current Node: %s, Cost: %s" % (current, c_cost))

            x, y = current
            if grid[x][y] == 't':
                path = []
                while current:
                    path.append(current)
                    current = root[current]
                print("Target `t` found! Path:", path[::-1])
                return 1, path[::-1]

            for neighbor in SearchAlgorithm.get_neighbors(*current, grid, walls):
                x, y = neighbor
                if grid[x][y] != '-1':
                    if grid[x][y] == 't':
                        n_cost = c_cost + 1
                    elif grid[x][y].isdigit():
                        n_cost = c_cost + int(grid[x][y])
                    else:
                        print("Invalid value '%s' at position (%d, %d). Skipping this cell." % (grid[x][y], x, y))
                        continue

                    print("Neighbor: %s, Cost from %s -> %s: %s" % (neighbor, current, neighbor, n_cost))

                    if neighbor not in cost or n_cost < cost[neighbor]:
                        cost[neighbor] = n_cost
                        root[neighbor] = current
                        pq.heappush(open_set, (n_cost, neighbor))

        print("Target `t` not found.")
        return -1, []

    @staticmethod
    def best_first_search(grid: List[List[str]]) -> Tuple[int, List[Tuple[int, int]]]:
        s, t, walls = SearchAlgorithm.get_start_target(grid)

        open_set = [(SearchAlgorithm.manhattan_distance(s, t), s)]
        root = {s: None}
        visited = set()

        while open_set:
            h_cost, current = pq.heappop(open_set)

            print("Processing Node: %s, Heuristic Value: %s" % (current, h_cost))

            if current == t:
                path = []
                while current:
                    path.append(current)
                    current = root[current]
                return 1, path[::-1]

            visited.add(current)
            for neighbor in SearchAlgorithm.get_neighbors(*current, grid, walls):
                if neighbor not in visited:
                    n_h_cost = SearchAlgorithm.manhattan_distance(neighbor, t)
                    pq.heappush(open_set, (n_h_cost, neighbor))
                    root[neighbor] = current

                    print("Neighbour Node: %s, Heuristic Value: %s" % (neighbor, n_h_cost))

        return -1, []

    @staticmethod
    def a_star(grid: List[List[str]]) -> Tuple[int, List[Tuple[int, int]]]:
        s , t , walls = SearchAlgorithm.get_start_target(grid)
        if s  == (-1, -1) or t  == (-1, -1):
            print("Start or target  not defined.")
            return -1, []

        open_set = [(0 + SearchAlgorithm.manhattan_distance(s , t ), 0, s )]
        root  = {s : None}
        cost_n  = {s : 0}

        print("Current Node -> g(x): Actual Cost, h(x): Heuristic Cost")

        while open_set:
            _, c_cost , current = pq.heappop(open_set)

            if current == t :
                path = []
                while current:
                    path.append(current)
                    current = root [current]
                return 1, path[::-1]

            for neighbor in SearchAlgorithm.get_neighbors(*current, grid, walls):
                x, y = neighbor
                if grid[x][y] != '-1':
                    if grid[x][y] in ["s", "t"]:
                        actual_cost = 1
                    elif grid[x][y].isdigit():
                        actual_cost = int(grid[x][y])
                    else:
                        print("wrong grid value '%s' at position (%d, %d). Skipping." % (grid[x][y], x, y))
                        continue

                    new_cost = c_cost  + actual_cost
                    if neighbor not in cost_n  or new_cost < cost_n [neighbor]:
                        cost_n [neighbor] = new_cost
                        h_cost  = SearchAlgorithm.manhattan_distance(neighbor, t )
                        total_cost = new_cost + h_cost
                        pq.heappush(open_set, (total_cost, new_cost, neighbor))
                        root [neighbor] = current

                        print("%s -> g(x): %d, h(x): %d" % (neighbor, new_cost, h_cost ))

        print("A* is not found")
        return -1, []



def visualize_grid(grid, path):
    visual_grid = np.zeros((len(grid), len(grid[0])))

    # Map grid values to custom colors
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 's':
                visual_grid[i][j] = 3
            elif grid[i][j] == 't':
                visual_grid[i][j] = 4
            elif grid[i][j] == '-1':
                visual_grid[i][j] = 5
            elif grid[i][j].isdigit():
                visual_grid[i][j] = 1  #

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['lightblue', 'purple', 'yellow', 'blue', 'green', 'red'])

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(visual_grid, annot=False, cmap=cmap, cbar=False, linewidths=0.5, ax=ax)

    for i in range(len(grid)):
        for j in range(len(grid[0])):
            text = grid[i][j]
            if text.isdigit() or text in {'s', 't', '-1'}:
                color = 'black' if text.isdigit() else 'black'
                ax.text(j + 0.5, i + 0.5, text, ha='center', va='center', color=color, fontsize=10, fontweight='bold')
    visited_x = []
    visited_y = []
    def update(i):
        if i < len(path) - 1:
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            visited_x.append(y1 + 0.5)
            visited_y.append(x1 + 0.5)

            ax.scatter(visited_x, visited_y, color='yellow', s=50, marker='o', edgecolor='black')

            ax.plot([y1 + 0.5, y2 + 0.5], [x1 + 0.5, x2 + 0.5], color='yellow', linewidth=2)

            time.sleep(0.2)
    ani = animation.FuncAnimation(fig, update, frames=len(path), repeat=False)
    plt.title("Grid Visualization from Start to Target")
    plt.show()

if __name__ == "__main__":
    filename = "example_3.txt"
    with open(filename, 'r') as file:
        grid = [line.strip().split() for line in file]

    print("\n---------------***********Grid********-------------------:")
    for row in grid:
        print(" ".join(row))

    algorithms = {
        "1": ("DFS", SearchAlgorithm.dfs),
        "2": ("BFS", SearchAlgorithm.bfs),
        "3": ("UCS", SearchAlgorithm.ucs),
        "4": ("Best-First Search", SearchAlgorithm.best_first_search),
        "5": ("A* Search", SearchAlgorithm.a_star)
    }

    while True:
        print("\nChoose a Search Algorithm:")
        for key, (name, _) in algorithms.items():
            print("%s. %s" % (key, name))
        print("0. Exit")

        choice = input("Enter your choice: ")
        if choice == "0":
            print("Exiting the program.")
            break
        elif choice in algorithms:
            name, func = algorithms[choice]
            print("\nRunning %s..." % name)
            found, path = func(grid)
            if found == 1:
                print("Path found using %s!" % name)

                print("Path:", path)
                visualize_grid(grid, path)
            else:
                print("No path found using %s"%name)
        else:
            print("Wrong Input")
