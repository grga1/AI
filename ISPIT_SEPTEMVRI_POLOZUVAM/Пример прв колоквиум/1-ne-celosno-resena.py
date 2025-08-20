from searching_framework import *


# from utils import *
# from uninformed_search import *
# from informed_search import *

class Boxes(Problem):
    def __init__(self, initial,N,boxes):
        super().__init__(initial)
        self.N = N
        self.boxes = boxes

    def actions(self, state):
        return sorted(self.successor(state).keys())

    def result(self, state, action):
        return self.successor(state)[action]

    def goal_test(self, state):
       filled_boxes = state[0]
       return  len(filled_boxes)==len(self.boxes)
    def check_valid(self,x,y):
        if (x,y) in self.boxes:
            return False
        if not (0 <= x < self.N and 0 <= y < self.N):
            return False
        return True
    def successor(self, state):
        successors = dict()
        moves = {'Gore':(0,1),'Desno':(1,0)}
        nearby = {(1,0),(-1,0),(0,1),(0,-1),(-1,-1),(1,1),(-1,1),(1,-1)}
        filled_boxes, (px,py) = state

        for action, (dx, dy) in moves.items():
              npx,npy = px + dx,py + dy
              if self.check_valid(npx,npy):
                new_filled = list(filled_boxes)
                for fx,fy in nearby:
                    fxx, fyy = npx + fx, npy + fy
                    if (fxx,fyy) in self.boxes and (fxx,fyy) not in filled_boxes:
                     new_filled.append((fxx,fyy))

                successors[f'{action}'] = (tuple(new_filled),(npx,npy))
        return successors


if __name__ == '__main__':
    n = int(input()) #golemina na tablata
    man_pos = (0, 0)  #pozicija na coveceto

    num_boxes = int(input())   #broj na topki/kutii
    boxes = list()
    for _ in range(num_boxes):
        boxes.append(tuple(map(int, input().split(','))))
    filled_boxes = frozenset()
    initial_state=(filled_boxes,man_pos)
    problem = Boxes(initial_state,n,boxes)
    solution = breadth_first_graph_search(problem)
    if solution is not None:
        print(solution.solution())
    else:
        print('No Solution!')
