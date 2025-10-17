import math
import random

def minimax (position, point_list, depth, alpha, beta, flag):
    if depth == 0:
        return point_list[position]
    if flag:
        max_eval = -math.inf
        for i in range(2):
            n_eval = minimax(position*2+i, point_list, depth-1, alpha, beta, False)
            max_eval = max(max_eval, n_eval)
            alpha = max(alpha, n_eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = math.inf
        for i in range(2):
            n_eval = minimax(position*2+i, point_list, depth-1, alpha, beta, True)
            min_eval = min(min_eval, n_eval)
            beta = min(beta, n_eval)
            if beta <= alpha:
                break
        return min_eval
    
def winner_finding(student_id):
    minimum_points = int(student_id[4])
    points_to_win = int(student_id[-1:-3:-1])
    maximum_points = int(points_to_win * 1.5)
    number_of_shuffles = int(student_id[3])
    random_point_list = random.sample(range(minimum_points, maximum_points), 8)
    shuffle_maximum_points_list, count = [], 0
    print(f'Generated 8 random points between the minimum and maximum point limits: {random_point_list}\nTotal points to win: {points_to_win}')
    result = minimax(0, random_point_list, 3, -math.inf, math.inf, True)
    print(f'Achieved point by applying alpha-beta pruning = {result}')
    if result >= points_to_win: print('The winner is Optimus Prime')
    else: print('The winner is Megatron')
    print('\nAfter the shuffle:')
    for i in range(number_of_shuffles):
        random_point_list = random.sample(range(minimum_points, maximum_points), 8)
        max_value = max(random_point_list)
        shuffle_maximum_points_list.append(max_value)
        if max_value >= points_to_win: count += 1
    shuffle_max_point = max(shuffle_maximum_points_list)    
    print(f'List of all points values from each shuffle: {shuffle_maximum_points_list}\nThe maximum value of all shuffles: {shuffle_max_point}\nWon {count} times out of {number_of_shuffles} number of shuffles')

student_id = '20301065'.replace('0', '8')
winner_finding(student_id)
