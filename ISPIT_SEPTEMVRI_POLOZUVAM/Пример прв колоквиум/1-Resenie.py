
from searching_framework import *

class Boxes(Problem):
    def __init__(self, initial, boxes,n):
        super().__init__(initial)
        self.boxes = boxes
        self.n = n

    def actions(self, state):
        return self.successor(state).keys()

    def result(self, state, action):
        return self.successor(state)[action]

    def goal_test(self, state):
       polni = state[1]
       return len(self.boxes)==len(polni)
    def check_valid(self,state):
        man_pos = state[0]
        x,y = man_pos
        if man_pos in self.boxes:
            return False
        if not (0<=x<self.n and 0<=y<self.n):
            return False
        return True
    def successor(self, state):
        succ = dict()
        moves = {'Gore':(0,1),'Desno':(1,0)}
        nearby = {(-1,1),(0,1),(1,1),
                  (-1,0),(0,0),(1,0),
                  (-1,-1),(0,-1),(1,-1)}
        x1,y1 = state[0]
       

        for action,(dx,dy) in moves.items():
            visited_boxes = list(state[1])
            nx1,ny1 = x1+dx,y1+dy #coveceto e ova
            new_state = ((nx1,ny1),tuple(visited_boxes))
            if self.check_valid(new_state):
                for (a,b) in nearby:
                    nx2,ny2 = nx1+a,ny1+b
                    if (nx2,ny2) not in visited_boxes and (nx2,ny2) in self.boxes:
                        visited_boxes.append((nx2,ny2))


                succ[f'{action}'] = ((nx1, ny1), tuple(visited_boxes))



        return succ


if __name__ == '__main__':
    n = int(input())
    man_pos = (0, 0)

    num_boxes = int(input())
    boxes = list()
    for _ in range(num_boxes):
        boxes.append(tuple(map(int, input().split(','))))

    visited_boxes = tuple()
    initial_state=(man_pos, visited_boxes)
    problem = Boxes(initial_state,boxes,n)
    solution = breadth_first_graph_search(problem)
    if solution is not None:
     print(solution.solution())
    else:
        print('No Solution!')
